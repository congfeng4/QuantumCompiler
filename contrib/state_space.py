"""
State space encoding.
"""
from enum import IntEnum
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from contrib.common import build_op_node_level, GATE_NAME_MAPPING

GATE_NAME_TO_ID = {name: i for i, name in enumerate(GATE_NAME_MAPPING.keys())}


class RoutedStatus(IntEnum):
    ROUTED = -1
    UNROUNTED = 1


class OpRepPosition(IntEnum):
    POS_ROUNTED = 0
    POS_DISTANCE = 1
    POS_LEVEL = 2
    POS_GATE_TYPE = 3
    POS_QUBIT_ONE = 4
    POS_QUBIT_TWO = 5
    POS_SIZE = 6
    POS_EMB_OFFSET = 2  # Starting from 2, features need embedding.


def pad_truncate_2d(arr, maxlen: int):
    """
    仅对第 0 维（batch）截断或补零，第 1 维保持不变
    """
    arr = np.asarray(arr)
    B = arr.shape[0]
    if B >= maxlen:  # 截断
        return arr[:maxlen]
    else:  # 补零
        pad_width = ((0, maxlen - B), (0, 0))
        return np.pad(arr, pad_width, 'constant')


class StateSpace:

    def __init__(self, max_len: int = 500):
        self.max_len = max_len

    @property
    def feature_dim(self):
        return int(OpRepPosition.POS_SIZE)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.feature_dim}, {self.max_len})'

    def encode(self,
               resulting_dag: DAGCircuit,
               remaining_dag: DAGCircuit,
               current_mapping: dict[Qubit, int] = None,
               distance_matrix: np.ndarray = None
               ):
        routed_seq, max_level = self.encode_dag(resulting_dag, RoutedStatus.ROUTED)
        unrouted_seq, _ = self.encode_dag(remaining_dag, RoutedStatus.UNROUNTED,
                                       current_mapping=current_mapping,
                                    #    level_offset=max_level,
                                       distance_matrix=distance_matrix)
        ops = np.concatenate((routed_seq, unrouted_seq), axis=0)
        # assert len(ops) <= self.max_len, f'Max len too small for total len: {len(ops)} vs {self.max_len}'
        x = pad_truncate_2d(ops, self.max_len)
        mask = np.ones((self.max_len,), np.int32)
        mask[:len(x)] = 0
        return {'x': x, 'mask': mask}  # x [S, F], mask [S, 1]

    def encode_dag(self, dag: DAGCircuit,
                   routed_status: RoutedStatus,
                   current_mapping: dict[Qubit, int] = None,
                   level_offset: int = 0,  # Should be able to differentiate routed and unrouted.
                   distance_matrix: np.ndarray = None):

        topological_nodes: list[DAGOpNode] = list(dag.topological_op_nodes())
        # Resort according to node levels.
        gate_levels = build_op_node_level(dag, topological_nodes, sort_by_level=True)
        seqlen = len(topological_nodes)
        gate_seq = np.zeros((seqlen, self.feature_dim), np.int64)

        for i, op in enumerate(topological_nodes):
            gate_type = GATE_NAME_MAPPING[op.name]
            gate_seq[i, OpRepPosition.POS_GATE_TYPE] = GATE_NAME_TO_ID[op.name]
            gate_seq[i, OpRepPosition.POS_ROUNTED] = routed_status
            gate_seq[i, OpRepPosition.POS_LEVEL] = level_offset + gate_levels[op._node_id]

            num_qubits = gate_type.num_qubits
            assert num_qubits <= 2, f'Three qubits gate should not be used: {gate_type}'
            qargs = op.qargs if num_qubits == 2 else op.qargs * 2

            if routed_status == RoutedStatus.ROUTED:
                qargs = [q._index for q in qargs]  # No need to translate to physical
            else:
                qargs = [current_mapping[q] for q in qargs]

            gate_seq[i, OpRepPosition.POS_QUBIT_ONE] = qargs[0]
            gate_seq[i, OpRepPosition.POS_QUBIT_TWO] = qargs[1]

            if num_qubits == 1 or routed_status == RoutedStatus.ROUTED:
                gate_seq[i, OpRepPosition.POS_DISTANCE] = -1  # 0 is bad for NN.
            else:
                # Two qubit ops needed to route.
                gate_seq[i, OpRepPosition.POS_DISTANCE] = distance_matrix[qargs[0], qargs[1]]

        return gate_seq, max(gate_levels.values()) if gate_levels else 0

    def to_gym_space(self):
        # Use sequence instead of box since our length is hard to tell in advance.
        # Fixme: stable-baseline3 don't support Sequence currently.
        # return gym.spaces.Sequence(space=gym.spaces.Box(low=float('-inf'), high=float('inf'),
        #                                                 shape=(self.num_op_features,)))
        return gym.spaces.Dict({
            "x": Box(low=-1, high=1, shape=(self.max_len, self.feature_dim), dtype=np.int32),
            "mask": Box(low=0, high=1, shape=(self.max_len,), dtype=np.int32)
        })
