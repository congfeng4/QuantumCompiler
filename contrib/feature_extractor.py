from typing import Dict, Union

import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from torch import nn
from enum import Enum
import math

from contrib.state_space import GateType, OpRepPosition, StateSpace
from hamap.hardware import IBMQHardwareArchitecture
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def normalize_distance_matrix(distance_matrix: torch.Tensor):
    min_val, max_val = distance_matrix.min(), distance_matrix.max()
    return (distance_matrix - min_val) / (max_val - min_val)


def positional_encoding_matrix(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe



class OpsEmbedding(nn.Module):
    # Op rep => Op embedding
    def __init__(self, qubit_number: int, params=None):
        super().__init__()
        if params is None:
            params = {}
        # Allow model distinguish different gate-types.
        self.gate_type_embed = nn.Embedding(GateType.GATE_TYPE_MAX, params.get('gate_type_embed_dim', 5))
        self.qubit_embed = nn.Embedding(qubit_number, params.get('qubit_embed_dim', 5))
        with torch.no_grad():
            out = self.forward(StateSpace().to_gym_space().sample()[0])
        self.embed_dim = out.shape[0]

    def forward(self, ops):  # [S, F] => [S, F']
        # Encode one Op sequence (ops).
        gate_type = self.gate_type_embed(ops[:, OpRepPosition.POS_GATE_TYPE])
        qubit_one = self.qubit_embed(ops[:, OpRepPosition.POS_QUBIT_ONE])
        qubit_two = self.qubit_embed(ops[:, OpRepPosition.POS_QUBIT_TWO])
        level = ops[:, OpRepPosition.POS_LEVEL]
        rest = torch.tensor([ops[:, :OpRepPosition.POS_EMB_OFFSET]], device=gate_type.device)
        feature = torch.cat([gate_type, qubit_one, qubit_two, rest], dim=1)
        pe = positional_encoding_matrix(ops.shape[0], feature.shape[0])
        return feature + pe


class SequenceEncoder(nn.Module):
    def __init__(self, in_dim, mode, num_layers: int, nhead: int, pe_mode: PositionalEncodingMode):
        super().__init__()
        self.output_channels = in_dim
        assert mode in ['gru', 'lstm', 'transformer', 'mean']
        self.mode = mode
        if mode == 'gru':
            self.rnn = nn.GRU(in_dim, in_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        elif mode == 'lstm':
            self.rnn = nn.LSTM(in_dim, in_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        elif mode == 'transformer':  # Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_dim, nhead=nhead, dim_feedforward=in_dim * 2, batch_first=True
            )
            if pe_mode == PositionalEncodingMode.LEVEL_PE:
                self.pos_enc = PositionalEncoding(in_dim)
            else:
                self.pos_enc = PositionalEncodingLevel(in_dim)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.out_dim = in_dim
        else:
            raise ValueError(mode)

    def forward(self, x, lengths, levels):
        """
        x:        (B, S, in_dim)   已 pad 到 batch 最大长度
        lengths:  (B,)            每一样本的真实门数
        return:   (B, hidden)     定长状态向量
        """
        if self.mode in ['gru', 'lstm']:
            lengths = lengths.view(-1).long().cpu()
            packed = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, h_last = self.rnn(packed)
            # h_last: (1, B, hidden) for GRU, (h_n, c_n) for LSTM
            state = h_last[-1] if self.mode == 'gru' else h_last[0][-1]
            return state  # (B, hidden)
        elif self.mode == 'transformer':  # Transformer
            if isinstance(self.pos_enc, PositionalEncodingLevel):
                x = self.pos_enc(x, levels)
            else:
                x = self.pos_enc(x)  # [B, L, F]
            B = x.shape[0]
            # 1. 构造 key_padding_mask
            max_len = x.size(1)
            # print('lengths', lengths.shape, 'arange',
            #       torch.arange(max_len, device=x.device).expand(B, -1).shape, 'x', x.shape)
            tmp = torch.arange(max_len, device=x.device).expand(B, -1)
            if len(lengths.shape) != 2:
                lengths = lengths.unsqueeze(1)
            mask = tmp >= lengths  # [B, L]
            # 2. Transformer 前向
            if len(mask.shape) != 2:
                print(f'!!Mask shape is not 2: {mask.shape=} {lengths.shape=} {tmp.shape=}', flush=True)
                raise RuntimeError

            x_enc = self.transformer(x, src_key_padding_mask=mask)  # [B, L, F]

            # 3. mean-pool 忽略 pad
            mask_float = (~mask).float().unsqueeze(-1)  # [B, L, 1]
            denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, 1, 1]
            state = (x_enc * mask_float).sum(dim=1, keepdim=False) / denom.squeeze(1)  # [B, F]

            return state  # (B, in_dim)
        elif self.mode == 'mean':
            return torch.mean(x, dim=1)
        else:
            raise ValueError(self.mode)


class CircuitFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, qubit_number: int,
                 feature_dim: int, params: Dict[str, Union[int, float]]):
        super().__init__(observation_space, features_dim=feature_dim)
        self.ops_embed = OpsEmbedding(qubit_number=qubit_number, params=params)

    def forward(self, observations):
        # observations: List[Tensor[F]]，长度不一
        lengths = [len(seq) for seq in observations]
        seq_list = []
        for seq in observations:
            seq_list.append(self.ops_embed(seq))

        padded = pad_sequence(seq_list, batch_first=True)  # [B, T, F]
        packed = pack_padded_sequence(padded, lengths,
                                      batch_first=True,
                                      enforce_sorted=False)
        return packed, lengths


def get_policy_kwargs(hardware: IBMQHardwareArchitecture, feature_dim: int):
    return dict(
        activation_fn=torch.nn.ReLU,
        features_extractor_class=CircuitFeaturesExtractor,
        features_extractor_kwargs=dict(
            hardware=hardware,
            feature_dim=feature_dim,
        ),
        net_arch=dict(
            pi=[feature_dim],
            vf=[feature_dim],
        ),
    )


def get_policy(env, hardware: IBMQHardwareArchitecture, embed_dim: int = 128, ):
    return ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
        activation_fn=torch.nn.ReLU,
        features_extractor_class=CircuitFeaturesExtractor,
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
    extractor = CircuitFeaturesExtractor(None, hw, D)

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
