"""
Helper function to convert states
"""
import random

import gymnasium as gym
import numpy as np
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGNode

from contrib.common import SWAP_INDEX, BRIDGE_INDEX
from hamap import IBMQHardwareArchitecture
from hamap.gates import TwoQubitGate, BridgeTwoQubitGate
from hamap.heuristics import _gate_op_cost
from hamap.layer import QuantumLayer


class StateSpace:

    def __init__(self, N: int, L: int):
        self.N = N
        self.L = L

    def get_space(self):
        N, L = self.N, self.L

        return gym.spaces.Dict({
            # 逻辑 → 物理映射 (permutation)
            "mapping": gym.spaces.Box(
                low=0, high=N - 1, shape=(N,), dtype=np.int64
            ),

            # 门序列：每行是 (q0, q1) 两个逻辑比特编号
            "gate_seq": gym.spaces.Box(
                low=0, high=N - 1, shape=(L, 2), dtype=np.int64
            ),

            # 真实门序列长度
            "gate_len": gym.spaces.Box(
                low=0, high=L, shape=(), dtype=np.int64
            ),
        })

    def encode(self, front_layer: QuantumLayer, gates: list[DAGNode], current_mapping: dict[Qubit, int]):
        mapping = np.zeros((self.N,), np.int64)
        for qb, j in current_mapping.items():
            mapping[j] = qb._index  # Phy to logic

        gate_seq = np.zeros((self.L, 2), np.int64)
        gate_len = 0
        # assert self.L >= len(front_layer), f"L too small to cover front layer!! {self.L=} {len(front_layer)=}"
        for op in front_layer.ops + gates:
            if gate_len >= self.L:
                break
            if op.name != 'cx':
                continue
            gate_seq[gate_len] = current_mapping[op.qargs[0]], current_mapping[op.qargs[1]]
            gate_len += 1

        return dict(mapping=mapping, gate_seq=gate_seq, gate_len=gate_len)

    def get_cost(self, front_layer: QuantumLayer, gates: list[DAGNode], current_mapping: dict[Qubit, int],
                 distance_matrix: np.ndarray, hardware: IBMQHardwareArchitecture):
        gate_len = 0
        cost = 0
        for op in front_layer.ops + gates:
            if gate_len >= self.L:
                break
            gate_len += 1
            cost += _gate_op_cost(op, distance_matrix, current_mapping, hardware)

        return cost / gate_len if gate_len else 0
