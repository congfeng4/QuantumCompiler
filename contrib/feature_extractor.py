from typing import Dict, Union
import torch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import math
from contrib.state_space import GATE_NAME_TO_ID, OpRepPosition


def positional_encoding_matrix(position: torch.Tensor, d_model: int):
    """
    position: [B, S]  通常是 level 或绝对位置
    return:   [B, S, d_model]
    """
    B, S = position.shape
    # 1. 生成「列方向」角度因子
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=position.device)
                         * (-math.log(10000.0) / d_model))  # [d_model//2]

    # 2. 利用广播：position [B, S, 1] * div_term [1, 1, d_model//2]
    angle = position.unsqueeze(-1) * div_term.unsqueeze(0).unsqueeze(0)  # [B, S, d_model//2]

    # 3. 计算 sin/cos
    pe_sin = torch.sin(angle)  # [B, S, d_model//2]
    pe_cos = torch.cos(angle)  # [B, S, d_model//2]

    # 4. 交错放入最终张量
    pe = torch.zeros(B, S, d_model, dtype=torch.float32, device=position.device)
    pe[:, :, 0::2] = pe_sin
    pe[:, :, 1::2] = pe_cos
    return pe


class OpsEmbedding(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.hidden_size})'

    # Ops rep => Ops embedding
    def __init__(self, qubit_number: int, feature_dim: int, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.feature_dim = feature_dim
        # Allow model distinguish different gate-types.
        self.gate_type_embed = nn.Embedding(len(GATE_NAME_TO_ID), params.get('gate_type_embed_dim', 16))
        self.qubit_embed = nn.Embedding(qubit_number, params.get('qubit_embed_dim', 16))
        self.hidden_size = sum([self.gate_type_embed.embedding_dim, 2 * self.qubit_embed.embedding_dim,
                                OpRepPosition.POS_EMB_OFFSET])
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))  # 可学习的 [CLS]
        self.feedforward = nn.Linear(self.hidden_size, self.feature_dim)

    def encode(self, x):
        B, S, _ = x.shape
        # Encode one Op sequence (ops).
        gate_type = self.gate_type_embed(x[:, :, OpRepPosition.POS_GATE_TYPE])
        qubit_one = self.qubit_embed(x[:, :, OpRepPosition.POS_QUBIT_ONE])
        qubit_two = self.qubit_embed(x[:, :, OpRepPosition.POS_QUBIT_TWO])
        rest = x[:, :, 0:OpRepPosition.POS_EMB_OFFSET]
        feature = torch.cat([gate_type, qubit_one, qubit_two, rest], dim=2)
        feature = self.feedforward(feature.reshape(B * S, -1)).reshape(B, S, -1)
        # Adapt each to feature_dim.
        return feature

    def forward(self, x, mask):  # [B, S, F] => [B, S, F']
        # Add positional embeddings. Add cls token.
        B, S, F = x.shape
        x = x.long()
        feature = self.encode(x)
        feature = torch.cat([self.cls_token.expand(B, 1, self.feature_dim), feature], dim=1)
        level = x[:, :, OpRepPosition.POS_LEVEL]  # [B, S]
        position = torch.cat([torch.zeros(B, 1, device=level.device), level + 1], dim=1)  # [B, S+1]
        pe = positional_encoding_matrix(position=position, d_model=self.feature_dim)
        # mask: [B, S] => [B, S+1]
        mask = torch.cat([torch.zeros(B, 1, device=mask.device), mask], dim=1)
        return feature + pe, mask


class TransformerCircuitEncoder(nn.Module):

    def __init__(self, feature_dim, params: dict):
        # Inputs: [B, S, feature_dim], Outputs: [B, feature_dim]
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=params.get('nhead', 4),
            dim_feedforward=params.get('dim_ff', 64),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=params.get('num_layers', 4),
            enable_nested_tensor=False,  # Prevent warning.
        )

    def forward(self, x, mask):
        x = self.transformer(x, src_key_padding_mask=mask)  # shape: [seq_len, batch_size, model_dim]
        cls_token = x[:, 0, :]  # 或者做 pooling
        return cls_token


class CircuitFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space,
                 feature_dim: int,  # Output dim of extractor, used in value/policy networks.
                 qubit_number: int,
                 params: Dict[str, Union[int, float]]):
        super().__init__(observation_space, feature_dim)

        self.ops_embed = OpsEmbedding(qubit_number=qubit_number, feature_dim=feature_dim, params=params)
        self.transformer = TransformerCircuitEncoder(feature_dim=feature_dim, params=params)

    def forward(self, observations):
        x, mask = observations['x'], observations['mask']
        x, mask = self.ops_embed(x, mask)
        return self.transformer(x, mask)


def get_policy_kwargs(qubit_number: int, feature_dim: int, params: dict):
    backbone_layers = params.get('backbone_layers', 2)
    return dict(
        activation_fn=torch.nn.ReLU,
        features_extractor_class=CircuitFeaturesExtractor,
        features_extractor_kwargs=dict(
            qubit_number=qubit_number,
            feature_dim=feature_dim,
            params=params,
        ),
        net_arch=dict(
            pi=[feature_dim] * backbone_layers,
            vf=[feature_dim] * backbone_layers,
        ),
    )


if __name__ == '__main__':
    pass
