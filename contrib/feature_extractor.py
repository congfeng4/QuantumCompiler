from typing import Dict, Union
import torch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import math

from contrib.state_space import GateType, OpRepPosition, StateSpace
from torch.nn.utils.rnn import pad_sequence


def positional_encoding_matrix(position, max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = position.unsqueeze(1) # Add F dim.
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class OpsEmbedding(nn.Module):
    # ✅

    def __repr__(self):
        return f'{self.__class__.__name__}({self.hidden_size})'

    # Ops rep => Ops embedding
    def __init__(self, qubit_number: int, params=None):
        super().__init__()
        if params is None:
            params = {}
        # Allow model distinguish different gate-types.
        self.gate_type_embed = nn.Embedding(GateType.GATE_TYPE_MAX, params.get('gate_type_embed_dim', 5))
        self.qubit_embed = nn.Embedding(qubit_number, params.get('qubit_embed_dim', 5))
        self.hidden_size = sum([2 * self.gate_type_embed.embedding_dim, self.gate_type_embed.embedding_dim,
                                OpRepPosition.POS_EMB_OFFSET])
        self.cls_token = nn.Parameter(torch.randn(1, self.hidden_size))  # 可学习的 [CLS]

    def encode(self, ops):
        # Encode one Op sequence (ops).
        gate_type = self.gate_type_embed(ops[:, OpRepPosition.POS_GATE_TYPE])
        qubit_one = self.qubit_embed(ops[:, OpRepPosition.POS_QUBIT_ONE])
        qubit_two = self.qubit_embed(ops[:, OpRepPosition.POS_QUBIT_TWO])
        rest = ops[:, :OpRepPosition.POS_EMB_OFFSET]
        feature = torch.cat([gate_type, qubit_one, qubit_two, rest], dim=1)
        return feature

    def forward(self, ops):  # [S, F] => [S, F']
        ops = ops.long()
        # Add positional embeddings. Add cls token.
        feature = self.encode(ops)
        feature = torch.cat([self.cls_token, feature], dim=0)
        level = ops[:, OpRepPosition.POS_LEVEL]
        position = torch.cat([torch.tensor([0], device=ops.device), level + 1], dim=0)
        pe = positional_encoding_matrix(position=position, max_len=feature.shape[0], d_model=self.hidden_size)
        return feature + pe


def _generate_mask(x, lengths):
    # 返回 True 表示该位置是 padding
    batch_size, seq_len, _ = x.shape
    mask = torch.arange(seq_len).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
    return mask


class TransformerCircuitEncoder(nn.Module):
    # ✅
    def __init__(self, hidden_size, num_classes, params: dict):
        # Inputs: [B, S, hidden_size], Outputs: [B, num_classes]
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=params.get('nhead', 4),
            dim_feedforward=params.get('dim_ff', 64),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=params.get('num_layers', 4),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        mask = _generate_mask(x, lengths)  # shape: [batch_size, seq_len]
        # x = x.transpose(0, 1)  # Transformer expects [seq_len, batch_size, model_dim]
        x = self.transformer(x, src_key_padding_mask=mask)  # shape: [seq_len, batch_size, model_dim]
        # x = x.transpose(0, 1)  # back to [batch_size, seq_len, model_dim]
        cls_token = x[:, 0, :]  # 或者做 pooling
        return self.classifier(cls_token)


class CircuitFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space,
                 feature_dim: int,  # Output dim of extractor, used in value/policy networks.
                 qubit_number: int,
                 params: Dict[str, Union[int, float]]):
        super().__init__(observation_space, feature_dim)

        self.ops_embed = OpsEmbedding(qubit_number=qubit_number, params=params)

        self.transformer = TransformerCircuitEncoder(
            hidden_size=self.ops_embed.hidden_size,
            num_classes=feature_dim,
            params=params
        )

    def forward(self, observations):
        # observations: List[Tensor[F]]，长度不一
        lengths = [len(seq) for seq in observations]
        seq_batch = []
        for seq in observations:
            seq_batch.append(self.ops_embed(seq))
        x = pad_sequence(seq_batch, batch_first=True)  # [B, T, F]
        return self.transformer(x, lengths)


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
    import torch, numpy as np
    from gymnasium.spaces import Sequence, Box
    from contrib.state_space import GateType, OpRepPosition, RoutedStatus

    obs_space = Sequence(Box(low=-np.inf, high=np.inf, shape=(6,)))
    extractor = CircuitFeaturesExtractor(
        observation_space=obs_space,
        feature_dim=64,
        qubit_number=3,
        params={"nhead": 2, "num_layers": 2, "gate_type_embed_dim": 4, "qubit_embed_dim": 4},
    )

    # 模拟 gym 传来的 List[Tensor]
    obs_list = [
        torch.tensor([
            [GateType.H, RoutedStatus.UNROUNTED, 1, 0, 0, 0],
            [GateType.CX, RoutedStatus.UNROUNTED, 1, 1, 0, 1],
        ], dtype=torch.float),
        torch.tensor([
            [GateType.SWAP, RoutedStatus.ROUTED, -1, 2, 1, 2],
        ], dtype=torch.float),
    ]

    features = extractor(obs_list)  # [B, feature_dim]
    print("obs_list length:", len(obs_list))
    print("features out:", features.shape)
