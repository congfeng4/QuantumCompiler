"""
State space encoding.
"""
from enum import IntEnum
import gymnasium as gym
import numpy as np

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from contrib.common import build_op_node_level


class GateType(IntEnum):
    H = 0 # Will be embedded so just use 0
    CX = 1
    SWAP = 2

    @classmethod
    def from_name(cls, name):
        if name == 'h':
            return cls.H
        if name == 'cx':
            return cls.CX
        if name == 'swap':
            return cls.SWAP
        raise ValueError(name)

    @property
    def num_qubits(self):
        if self == self.H:
            return 1
        return 2


class RoutedStatus(IntEnum):
    ROUTED = -1
    UNROUNTED = 1


class OpRepPosition(IntEnum):
    POS_GATE_TYPE = 0
    POS_ROUNTED = 1
    POS_DISTANCE = 2
    POS_LEVEL = 3
    POS_QUBIT_ONE = 4
    POS_QUBIT_TWO = 5
    POS_SIZE = 6


class StateSpace:

    def encode(self, dag: DAGCircuit,
               routed_status: RoutedStatus,
               current_mapping: dict[Qubit, int] = None,
               level_offset: int = 0, # Should be able to differentiate routed and unrouted.
               distance_matrix: np.ndarray = None):

        topological_nodes: list[DAGOpNode] = list(dag.topological_op_nodes())
        # Resort according to node levels.
        gate_levels = build_op_node_level(dag, topological_nodes, sort_by_level=True)
        seqlen = len(topological_nodes)
        gate_seq = np.zeros((seqlen, OpRepPosition.POS_SIZE), np.int64)

        for i, op in enumerate(topological_nodes):
            gate_type = GateType.from_name(op.name)
            gate_seq[i, OpRepPosition.POS_GATE_TYPE] = gate_type
            gate_seq[i, OpRepPosition.POS_ROUNTED] = routed_status
            gate_seq[i, OpRepPosition.POS_LEVEL] = level_offset + gate_levels[op._node_id]

            num_qubits = gate_type.num_qubits
            qargs = op.qargs if num_qubits == 2 else op.qargs * 2

            if routed_status == RoutedStatus.ROUTED:
                qargs = [q._index for q in qargs]  # No need to translate to physical
            else:
                qargs = [current_mapping[q] for q in qargs]

            gate_seq[i, OpRepPosition.POS_QUBIT_ONE] = qargs[0]
            gate_seq[i, OpRepPosition.POS_QUBIT_TWO] = qargs[1]

            if num_qubits == 1 or routed_status == RoutedStatus.ROUTED:
                gate_seq[i, OpRepPosition.POS_DISTANCE] = -1 # 0 is bad for NN.
            else:
                # Two qubit ops needed to route.
                gate_seq[i, OpRepPosition.POS_DISTANCE] = distance_matrix[qargs[0], qargs[1]]

        return gate_seq

    def get_space(self):
        # Use sequence instead of box since our length is hard to tell in advance.
        return gym.spaces.Sequence(space=gym.spaces.Box(low=0, high=float('inf'),
                                                        shape=(OpRepPosition.POS_SIZE,)))
