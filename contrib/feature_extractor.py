import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from hamap.hardware import IBMQHardwareArchitecture
from torch_geometric.nn import GraphSAGE
from torch_geometric.utils import from_networkx
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class HardwareAwareQubitEmbedding(nn.Module):
    """
    Create hardware-aware qubit embedings.
    1. reflects the hardware graph affinity and
    2. preserve the qubit identity.
    """
    def __init__(self, hardware: IBMQHardwareArchitecture,
                 qubit_embed: str = "onehot",
                 qubit_embedding_dim: int = 32,
                 num_layers: int = 3, hidden_channels: int = 32,
                 ):
        super().__init__()
        self.hardware = hardware
        self.edge_index = from_networkx(hardware).edge_index
        self.num_qubits: int = hardware.qubit_number
        self.qubit_embed_class = QUBIT_EMBED[qubit_embed]
        self.qubit_embedding = self.qubit_embed_class(self.num_qubits, qubit_embedding_dim)

        self.gnn = GraphSAGE(
            in_channels=self.qubit_embedding.embedding_dim,
            out_channels=qubit_embedding_dim,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
        )

    def forward(self, physical2log: torch.LongTensor):
        """
        logical2phy: [B, N]
        """
        # physical2log = inverse_permutation_batched(logical2phy)
        # 1. Convert indices to initial embeddings
        init_phy_embed = self.qubit_embedding(physical2log)
        # 2. Go through gnn.
        ha_phy_embed = self.gnn(init_phy_embed, self.edge_index)
        # 3. Look up the embeddings of each logical bit.
        # gnn_embeddings: (B, N, d)   —— 物理比特顺序
        # logical2phy   : (B, N)     —— 每行是 0..N-1 的排列
        # B, N, d = ha_phy_embed.shape
        # # 构造索引 (B, N, d) 的最后一个维度广播
        # idx = logical2phy.unsqueeze(-1).expand(-1, -1, d)
        # # 按逻辑比特顺序重排
        # logic_embed = torch.gather(ha_phy_embed, dim=1, index=idx)
        # # 结果 shape 仍为 (B, N, d)
        return ha_phy_embed


class GateSeqEncoder(nn.Module):
    """
    Encode a sequence of gates into a sequence of embeddings.
    1. Lookup the qubit embeddings given each pair of logical qubits of a gate.
    2. Concat the qubit embeddings and send to a shared MLP to obtain the embedding of a gate.
    """
    def __init__(self,
                 embed_dim: int,  # embed dim of a qubit.
                 mlp_hidden: int = None):
        super().__init__()
        self.embed_dim = embed_dim

        if mlp_hidden is None:
            mlp_hidden = embed_dim * 4          # 可调

        # self.mlp = nn.Sequential(
        #     nn.Linear(embed_dim * 2, mlp_hidden),
        #     nn.ReLU(),
        #     nn.Linear(mlp_hidden, mlp_hidden),
        #     nn.ReLU(),
        #     nn.Linear(mlp_hidden, embed_dim * 2)  # 输出 2*D
        # )

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
        embed_flat = qubit_embed.view(-1, qubit_embed.size(-1))  # (B*N, D)
        e0 = embed_flat[idx0]  # (B*S, D)
        e1 = embed_flat[idx1]  # (B*S, D)

        # reshape 回 (B, S, D) 并 concat
        e0 = e0.view(B, S, -1)
        e1 = e1.view(B, S, -1)
        gate_vec = torch.cat([e0, e1], dim=-1)  # (B, S, 2*D)

        # 共享 MLP：对每个 (B, S, 2*D) 的向量独立过 MLP
        # gate_emb = self.mlp(gate_vec.reshape(B*S, -1)).reshape(B, S, -1)  # (B, S, 2*D)
        return gate_vec


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


class CircuitEncoder(nn.Module):
    def __init__(self, in_dim, mode='gru', num_layers: int = 2):
        super().__init__()
        assert mode in ['gru', 'lstm', 'transformer']
        self.mode = mode
        if mode == 'gru':
            self.rnn = nn.GRU(in_dim, in_dim, num_layers=num_layers, batch_first=True)
        elif mode == 'lstm':
            self.rnn = nn.LSTM(in_dim, in_dim, num_layers=num_layers, batch_first=True)
        else:   # Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_dim, nhead=8, dim_feedforward=in_dim * 2, batch_first=True
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
            return state                    # (B, hidden)
        else:   # Transformer
            x = self.pos_enc(x)
            # 构造 key_padding_mask: True 表示 pad 位置要被忽略
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            x_enc = self.transformer(x, src_key_padding_mask=mask)
            # mean-pool 忽略 pad
            mask_float = (~mask).float().unsqueeze(-1)
            state = (x_enc * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            return state                    # (B, in_dim)


class HierarchicalCircuitFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, hardware: IBMQHardwareArchitecture, embed_dim: int):
        super().__init__(observation_space, features_dim=2 * embed_dim)
        self.qubit_embed = HardwareAwareQubitEmbedding(hardware, qubit_embedding_dim=embed_dim)
        self.gate_seq_encoder = GateSeqEncoder(embed_dim)
        self.circuit_encoder = CircuitEncoder(2 * embed_dim)

    def forward(self, obs: dict[str, torch.Tensor]):
        # SB3会把Box无脑转成float32.
        mapping = obs['mapping'].long()  # [B, N] Logical to physical mapping
        qubit_embed = self.qubit_embed(mapping)  # Logical qubit embed

        gate_seq = obs['gate_seq'].long() # [B, S, 2] Gate seq of qubit pairs. (padded)
        gate_len = obs['gate_len'].long() # [B, 1] Gate seq len of each seq.
        gate_embed = self.gate_seq_encoder(gate_seq, qubit_embed)  # [B, S, 2*D], D is embed_dim
        circuit_embed = self.circuit_encoder(gate_embed, gate_len)  # [B, 2*D]
        return circuit_embed


if __name__ == '__main__':
    import torch
    from hamap.hardware import IBMQHardwareArchitecture  # 假设已安装
    from torch_geometric.utils import from_networkx  # 同上

    # 1. 创建硬件图
    hw = IBMQHardwareArchitecture("tokyo")  # 或任何支持的芯片
    num_qubits = hw.qubit_number

    # 2. 组装一个单样本 batch
    B, S, D = 1, 7, 32  # 样本数=1，门序列最大长度=7，单比特嵌入32维
    gate_seq = torch.randint(0, num_qubits, (B, S, 2))
    gate_len = torch.tensor([5])  # 样本真实长度=5（后面 2 个是 pad）

    # 3. 随机映射（逻辑→物理）
    mapping = torch.randperm(num_qubits).unsqueeze(0)  # (1, N)

    # 4. 构造模型
    obs_space = None  # SB3 里可填 gym.spaces.Dict，这里不用
    extractor = HierarchicalCircuitFeaturesExtractor(hw, None, embed_dim=D)

    # 5. 前向
    obs = {
        "mapping": mapping,
        "gate_seq": gate_seq,
        "gate_len": gate_len
    }
    state_vec = extractor(obs)
    print(state_vec.shape)  # 应为 (1, 64)  = 2*D
    print(state_vec)
