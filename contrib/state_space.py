"""
Helper function to convert states
"""
import random

import gymnasium as gym
import numpy as np
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGNode

from contrib.common import SWAP_INDEX, BRIDGE_INDEX
from hamap.gates import TwoQubitGate, BridgeTwoQubitGate
from hamap.layer import QuantumLayer


class StateSpace:

    def __init__(self, N: int, L: int, K: int):
        self.N = N
        self.L = L
        self.K = K

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

            # 候选动作
            "cands": gym.spaces.Box(
                low=0, high=float('inf'), shape=(self.K, 3), dtype=np.int64,
            ),

            # 候选动作
            "cand_len": gym.spaces.Box(
                low=0, high=self.K, shape=(), dtype=np.int64,
            )
        })

    def encode_swap(self, swap: TwoQubitGate, current_mapping: dict[Qubit, int], initial_mapping: dict[Qubit, int]):
        if isinstance(swap, BridgeTwoQubitGate):  # Already physical
            q0, q1 = initial_mapping[swap.left], initial_mapping[swap.right]
            return SWAP_INDEX, q0, q1
        else:
            q0, q1 = current_mapping[swap.left], current_mapping[swap.right]
            return BRIDGE_INDEX, q0, q1

    def patch_candidates(self, raw_cands: list[TwoQubitGate]):
        assert len(raw_cands) <= self.K
        # random.shuffle(raw_cands)
        patched_cands = [None] * self.K
        patched_cands[:len(raw_cands)] = raw_cands
        return patched_cands

    def encode_obs(self, front_layer: QuantumLayer, gates: list[DAGNode], current_mapping: dict[Qubit, int],
                   initial_mapping: dict[Qubit, int], swap_candidates: list[TwoQubitGate]):
        mapping = np.zeros((self.N,), np.int64)
        for qb, j in current_mapping.items():
            mapping[j] = qb._index  # Phy to logic

        gate_seq = np.zeros((self.L, 2), np.int64)
        gate_len = 0
        for op in front_layer.ops + gates:
            if gate_len >= self.L:
                break
            if op.name != 'cx':
                continue
            gate_seq[gate_len] = current_mapping[op.qargs[0]], current_mapping[op.qargs[1]]
            gate_len += 1

        cands = np.zeros((self.K, 3), np.int64)
        cand_len = 0
        for i, swap in enumerate(swap_candidates):
            if swap is None:
                continue
            cands[i] = self.encode_swap(swap, current_mapping, initial_mapping)
            cand_len += 1

        return dict(mapping=mapping, gate_seq=gate_seq, gate_len=gate_len, cands=cands, cand_len=cand_len)
