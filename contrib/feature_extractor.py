import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from enum import Enum

from torch_geometric.nn import GraphSAGE
from torch_geometric.utils import from_networkx
from hamap.distance_matrix import get_distance_matrix_swap_number
from hamap.hardware import IBMQHardwareArchitecture
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch_geometric as pyg


class DenseGNNType(Enum):
    GCN_CONV = pyg.nn.DenseGCNConv
    GRAPH_CONV = pyg.nn.DenseGraphConv
    GIN_CONV = pyg.nn.DenseGINConv
    GAT_CONV = pyg.nn.DenseGATConv
    SAGE_CONV = pyg.nn.DenseSAGEConv

    @property
    def name(self):
        return self.value.__name__


class QubitEmbedding(nn.Module):
    """
    Encoe one qubit.
    """

    def __init__(self, num_qubits: int, embedding_dim: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.embedding_dim = embedding_dim


class OnehotQubitEmbedding(QubitEmbedding):

    def __init__(self, num_qubits: int, embedding_dim: int = None):
        super().__init__(num_qubits, num_qubits)
        self.onehot_table = torch.eye(num_qubits)

    def forward(self, qubit_indices: torch.LongTensor):
        return self.onehot_table[qubit_indices, :]


class LearnableQubitEmbedding(QubitEmbedding):
    def __init__(self, num_qubits: int, embedding_dim: int = 32):
        super().__init__(num_qubits, embedding_dim)
        self.learnable_parameters = nn.Embedding(num_qubits, embedding_dim)

    def forward(self, qubit_indices: torch.LongTensor):
        return self.learnable_parameters(qubit_indices)


QUBIT_EMBED = {
    'onehot': OnehotQubitEmbedding,
    'param': LearnableQubitEmbedding,
}


def inverse_permutation_batched(p: torch.Tensor) -> torch.Tensor:
    """
    p: (B, N) 的排列张量，每行是 0..N-1 的排列
    return: (B, N) 的逆排列张量
    """
    B, N = p.shape
    inv = torch.empty_like(p)
    # 对每一行执行 scatter
    inv.scatter_(dim=1, index=p, src=torch.arange(N, device=p.device).repeat(B, 1))
    return inv


class DenseGNNModule(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels,
                 conv_mode: DenseGNNType = DenseGNNType.GCN_CONV):
        super().__init__()
        self.conv1 = conv_mode.value(in_channels, hidden_channels)
        self.conv2 = conv_mode.value(hidden_channels, out_channels)

    def forward(self, node_feat, adjacent_matrix):
        x = self.conv1(node_feat, adjacent_matrix)
        x = nn.functional.relu(x)
        x = self.conv2(x, adjacent_matrix)
        x = nn.functional.relu(x)
        return x


class QubitEmbeddingMode(Enum):
    DISTANCE_MATRIX_MLP = 0
    GNN_EDGE_INDEX = 1


def normalize_distance_matrix(distance_matrix: torch.Tensor):
    min_val, max_val = distance_matrix.min(), distance_matrix.max()
    return (distance_matrix - min_val) / (max_val - min_val)


class HardwareAwareQubitEmbedding(nn.Module):
    # TODO: Use GNN or distance-matrix + MLP? NEED ABLATION!

    def __init__(self, hardware: IBMQHardwareArchitecture,
                 qubit_embed: str = "param",
                 qubit_embedding_dim: int = 32,
                 hidden_channels: int = 32,
                 qubit_embed_mode: QubitEmbeddingMode = QubitEmbeddingMode.DISTANCE_MATRIX_MLP):
        super().__init__()
        self.hardware = hardware
        self.edge_index = from_networkx(hardware).edge_index
        self.num_qubits: int = hardware.qubit_number
        self.distance_matrix = torch.tensor(get_distance_matrix_swap_number(hardware), dtype=torch.float32)
        self.distance_matrix = normalize_distance_matrix(self.distance_matrix)
        self.qubit_embed_class = QUBIT_EMBED[qubit_embed]
        self.output_channels = qubit_embedding_dim
        self.qubit_embed_mode = qubit_embed_mode

        if qubit_embed_mode == QubitEmbeddingMode.GNN_EDGE_INDEX:
            # 1. 逻辑比特嵌入（随映射变化）
            self.qubit_embedding = self.qubit_embed_class(self.num_qubits, qubit_embedding_dim)
            self.gnn = GraphSAGE(
                in_channels=qubit_embedding_dim,
                out_channels=qubit_embedding_dim,
                hidden_channels=hidden_channels,
                num_layers=3,
            )
        elif qubit_embed_mode == QubitEmbeddingMode.DISTANCE_MATRIX_MLP:
            self.mlp = nn.Sequential(
                nn.Linear(self.num_qubits, qubit_embedding_dim),
                nn.ReLU(),
                nn.Linear(qubit_embedding_dim, qubit_embedding_dim),
                # nn.ReLU(),
                # nn.Linear(qubit_embedding_dim, qubit_embedding_dim),
            )

    def forward(self, physical2log: torch.LongTensor):
        # physical2log: [B, N]  每行是一个排列，表示物理->逻辑的映射
        B, N = physical2log.shape
        if self.qubit_embed_mode == QubitEmbeddingMode.DISTANCE_MATRIX_MLP:
            node_feat = self.distance_matrix.expand(B, -1, -1)
            ha_embed = self.mlp(node_feat)
            return ha_embed

        if self.qubit_embed_mode == QubitEmbeddingMode.GNN_EDGE_INDEX:
            node_feat = self.qubit_embedding(physical2log)  # [B, N, D]
            ha_embed = self.gnn(node_feat, self.edge_index)  # [B, N, D]
            return ha_embed

        raise ValueError(self.qubit_embed_mode)


class GateSeqEncoder(nn.Module):
    """
    Encode a sequence of gates into a sequence of embeddings.
    1. Lookup the qubit embeddings given each pair of logical qubits of a gate.
    2. Concat the qubit embeddings and send to a shared MLP to obtain the embedding of a gate.
    """

    def __init__(self,
                 input_dim: int,
                 embed_dim: int,
                 ):
        super().__init__()
        self.output_channels = embed_dim
        self.mlp = nn.Sequential(
            nn.Identity()
            # nn.Linear(input_dim, embed_dim),
            # nn.ReLU(),
            # nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, gate_seq: torch.LongTensor, qubit_embed: torch.FloatTensor):
        """
        gate_seq: [B, S, 2]   每行是 (q0, q1)
        return:        [B, S, 2*D]
        """
        # qubit_embed : (B, N, D)   —— 每行是 N 个逻辑比特的嵌入
        # gate_seq    : (B, S, 2)   —— 每个门是 (q0, q1) 逻辑比特编号
        B, S = gate_seq.shape[:2]
        _, N, _ = qubit_embed.shape

        # 把 q0, q1 展平到 (B*S,) 方便 gather
        q0_flat = gate_seq[..., 0].reshape(-1)  # (B*S,)
        q1_flat = gate_seq[..., 1].reshape(-1)  # (B*S,)

        # 构造 batch 偏移索引
        batch_offset = torch.arange(B, device=qubit_embed.device).unsqueeze(1) * N
        batch_offset = batch_offset.expand(-1, S).reshape(-1)  # (B*S,)

        idx0 = q0_flat + batch_offset.reshape(-1)  # (B*S,)
        idx1 = q1_flat + batch_offset.reshape(-1)  # (B*S,)

        # 展平 qubit_embed 到 (B*N, D) 后 gather
        embed_flat = qubit_embed.reshape(-1, qubit_embed.size(-1))  # (B*N, D)
        e0 = embed_flat[idx0]  # (B*S, D)
        e1 = embed_flat[idx1]  # (B*S, D)

        # reshape 回 (B, S, D) 并 concat
        e0 = e0.reshape(B, S, -1)
        e1 = e1.reshape(B, S, -1)
        gate_vec = torch.cat([e0, e1], dim=-1)  # (B, S, 2*D)
        gate_emb = self.mlp(gate_vec.reshape(B*S, -1)).reshape(B, S, -1)
        return gate_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, S, d_model)
        return x + self.pe[:, :x.size(1)]


class SequenceEncoder(nn.Module):
    def __init__(self, in_dim, mode='gru', num_layers: int = 4, nhead: int = 2):
        super().__init__()
        self.output_channels = in_dim
        assert mode in ['gru', 'lstm', 'transformer']
        self.mode = mode
        if mode == 'gru':
            self.rnn = nn.GRU(in_dim, in_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        elif mode == 'lstm':
            self.rnn = nn.LSTM(in_dim, in_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        else:  # Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_dim, nhead=nhead, dim_feedforward=in_dim * 2, batch_first=True
            )
            self.pos_enc = PositionalEncoding(in_dim, max_len=1024)  # 或用可学习版本
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.out_dim = in_dim

    def forward(self, x, lengths):
        """
        x:        (B, S, in_dim)   已 pad 到 batch 最大长度
        lengths:  (B,)            每一样本的真实门数
        return:   (B, hidden)     定长状态向量
        """
        if self.mode in ['gru', 'lstm']:
            lengths = lengths.view(-1).long()
            packed = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, h_last = self.rnn(packed)
            # h_last: (1, B, hidden) for GRU, (h_n, c_n) for LSTM
            state = h_last[-1] if self.mode == 'gru' else h_last[0][-1]
            return state  # (B, hidden)
        else:  # Transformer
            x = self.pos_enc(x)  # [B, L, F]
            B = x.shape[0]
            # 1. 构造 key_padding_mask
            max_len = x.size(1)
            # print('lengths', lengths.shape, 'arange',
            #       torch.arange(max_len, device=x.device).expand(B, -1).shape, 'x', x.shape)

            mask = torch.arange(max_len, device=x.device).expand(B, -1) >= lengths  # [B, L]
            # 2. Transformer 前向
            x_enc = self.transformer(x, src_key_padding_mask=mask)  # [B, L, F]

            # 3. mean-pool 忽略 pad
            mask_float = (~mask).float().unsqueeze(-1)  # [B, L, 1]
            denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, 1, 1]
            state = (x_enc * mask_float).sum(dim=1, keepdim=False) / denom.squeeze(1)  # [B, F]

            return state  # (B, in_dim)


class HierarchicalCircuitFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, hardware: IBMQHardwareArchitecture,
                 feature_dim: int, mode: str = 'gru', nhead: int = 2, num_layers: int = 4,
                 qubit_embed_mode: QubitEmbeddingMode = QubitEmbeddingMode.DISTANCE_MATRIX_MLP):
        super().__init__(observation_space, features_dim=feature_dim)
        dim = feature_dim // 2
        self.qubit_embed = HardwareAwareQubitEmbedding(hardware, qubit_embedding_dim=dim,
                                                       hidden_channels=dim,
                                                       qubit_embed_mode=qubit_embed_mode)
        self.gate_seq_encoder = GateSeqEncoder(self.qubit_embed.output_channels * 2, feature_dim)
        self.circuit_encoder = SequenceEncoder(self.gate_seq_encoder.output_channels, mode=mode,
                                               nhead=nhead, num_layers=num_layers)

    def forward(self, obs: dict[str, torch.Tensor]):
        # SB3会把Box无脑转成float32.
        mapping = obs['mapping'].long()  # [B, N] Logical to physical mapping
        qubit_embed = self.qubit_embed(mapping)  # Logical + Physical qubit embed [B, N, D]

        gate_seq = obs['gate_seq'].long()  # [B, S, 2] Gate seq of qubit pairs. (padded)
        gate_len = obs['gate_len'].long()  # [B, 1] Gate seq len of each seq.

        gate_embed = self.gate_seq_encoder(gate_seq, qubit_embed)  # [B, S, D], D is embed_dim
        circuit_embed = self.circuit_encoder(gate_embed, gate_len)  # [B, D]

        return circuit_embed


def get_policy_kwargs(hardware: IBMQHardwareArchitecture, embed_dim: int = 128, mode: str = 'gru'):
    return dict(
        activation_fn=torch.nn.ReLU,
        features_extractor_class=HierarchicalCircuitFeaturesExtractor,
        features_extractor_kwargs=dict(
            hardware=hardware,
            feature_dim=embed_dim,
            mode=mode,
        ),
        net_arch=dict(
            pi=[embed_dim, embed_dim, embed_dim],
            vf=[embed_dim, embed_dim, embed_dim],
        ),
    )


def get_policy(env, hardware: IBMQHardwareArchitecture, embed_dim: int = 128, ):
    return ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
        activation_fn=torch.nn.ReLU,
        features_extractor_class=HierarchicalCircuitFeaturesExtractor,
        features_extractor_kwargs=dict(
            hardware=hardware,
            feature_dim=embed_dim,
        ),
        net_arch=dict(
            pi=[embed_dim, embed_dim],
            vf=[embed_dim, embed_dim],
        ),
    )


if __name__ == '__main__':
    import torch
    from hamap.hardware import IBMQHardwareArchitecture  # 假设已安装
    from torch_geometric.utils import from_networkx  # 同上

    # 1. 创建硬件图
    hw = IBMQHardwareArchitecture("tokyo")  # 或任何支持的芯片
    num_qubits = hw.qubit_number

    # 2. 组装一个单样本 batch
    B, S, D = 2, 7, 32  # 样本数=1，门序列最大长度=7，嵌入32维
    K = 5
    gate_seq = torch.randint(0, num_qubits, (B, S, 2))
    gate_len = torch.tensor([5, 4])  # 样本真实长度=5（后面 2 个是 pad）

    # 3. 随机映射（逻辑→物理）
    mapping = torch.randperm(num_qubits).unsqueeze(0)  # (1, N)
    mapping = torch.cat([mapping, mapping], dim=0)

    cands = torch.cat([torch.randint(0, num_qubits, (B, K, 2)), torch.randint(0, 1, (B, K, 1))], dim=-1)
    cand_len = torch.tensor([3, 4], dtype=torch.long)  # 注意是 1-D LongTensor

    # 4. 构造模型
    extractor = HierarchicalCircuitFeaturesExtractor(None, hw, D)

    # 5. 前向
    obs = {
        "mapping": mapping,
        "gate_seq": gate_seq,
        "gate_len": gate_len,
        "cands": cands,
        "cand_len": cand_len,
    }
    state_vec = extractor(obs)
    print(state_vec.shape)  # 应为 (1, 64)  = 2*D
    print(state_vec)
