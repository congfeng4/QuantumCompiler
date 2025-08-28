"""
State space encoding.
"""

import gymnasium as gym
import numpy as np
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGNode, DAGOpNode

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

            # 门的层级（整数）
            "gate_level":  gym.spaces.Box(
                low=0, high=L, shape=(L, 1), dtype=np.int64
            ),
        })

    def encode(self, front_layer: QuantumLayer, gates: list[DAGNode], current_mapping: dict[Qubit, int],
               levels: dict[int, int]):
        mapping = np.zeros((self.N,), np.int64)
        for qb, j in current_mapping.items():
            mapping[j] = qb._index  # Phy to logic

        gate_seq = np.zeros((self.L, 2), np.int64)
        gate_level = np.zeros((self.L, 1), np.int64)
        ops: list[DAGOpNode] = front_layer.ops + gates
        ops = list(filter(lambda op: op.name == 'cx', ops))
        level_base = levels[ops[0].sort_key] if len(ops) else -1
        gate_len = min(len(ops), self.L)

        for i in range(gate_len):
            op = ops[i]
            gate_seq[i] = current_mapping[op.qargs[0]], current_mapping[op.qargs[1]]
            gate_level[i] = levels[op.sort_key] - level_base

        return dict(mapping=mapping, gate_seq=gate_seq, gate_len=gate_len, gate_level=gate_level)
