from typing import Dict, Union
import transformers
import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
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
    # Ops rep => Ops embedding
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


class CircuitFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, qubit_number: int,
                 params: Dict[str, Union[int, float]]):
        super().__init__(observation_space)
        self.ops_embed = OpsEmbedding(qubit_number=qubit_number, params=params)
        self.build(params)

    def build(self, params):
        self.transformer = transformers.models.AutoModelForSequenceClassification

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
    pass
