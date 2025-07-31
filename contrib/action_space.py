from functools import cached_property

import gymnasium as gym
import numpy as np

from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit.dagcircuit import DAGNode

from hamap.gates import TwoQubitGate, SwapTwoQubitGate, BridgeTwoQubitGate


NUM_ACTIONS = 2
EXE_INDEX = None
SWAP_INDEX = 0
BRIDGE_INDEX = 1


class ActionSpace:

    def __init__(self, N: int, A: int):
        self.A = A
        self.N = N

    def get_space(self):
        A, N = self.A, self.N
        return gym.spaces.Discrete(A*N*N)

    def get_size(self):
        return self.A * self.N * self.N

    def encode_execute_list(self, execute_gate_list: list[DAGNode], current_mapping: dict[Qubit, int]):
        action = self.empty_action()
        num_exe_cx = 0
        for op in execute_gate_list:
            if op.name != 'cx':
                continue
            q0, q1 = current_mapping[op.qargs[0]], current_mapping[op.qargs[1]]
            action[EXE_INDEX, q0, q1] = 1
            num_exe_cx += 1
        return action

    def empty_action(self):
        action = np.zeros((self.A, self.N, self.N), np.float32)
        return action

    def encode_swap_cands(self, swap_cands: list[TwoQubitGate], current_mapping: dict[Qubit, int]):
        action = self.empty_action()
        for swap in swap_cands:
            if isinstance(swap, BridgeTwoQubitGate):
                action[BRIDGE_INDEX, swap.left._index, swap.right._index] = 1
            else:
                q0, q1 = current_mapping[swap.left], current_mapping[swap.right]
                action[SWAP_INDEX, q0, q1] = 1
        return action

    @cached_property
    def _N2(self):
        return self.N * self.N

    def encode(self, index: int, q0: int, q1: int):
        return index * self._N2 + q0 * self.N + q1

    def encode_best_swap(self, swap: TwoQubitGate, current_mapping: dict[Qubit, int], initial_mapping: dict[Qubit, int]):
        if isinstance(swap, BridgeTwoQubitGate):  # Already physical
            q0, q1 = initial_mapping[swap.left], initial_mapping[swap.right]
            return self.encode(BRIDGE_INDEX, q0, q1)
            # action[BRIDGE_INDEX, swap.left._index, swap.right._index] = 1
        else:
            q0, q1 = current_mapping[swap.left], current_mapping[swap.right]
            return self.encode(SWAP_INDEX, q0, q1)
            # action[SWAP_INDEX, q0, q1] = 1
        # return action

    def decode(self, policy: int):
        num_params = self._N2
        index, params = policy // num_params, policy % num_params
        left = params // self.N
        right = params % self.N
        return index, left, right

    def decode_best_swap(self, policy: int, inverse_current_mapping: dict[int, Qubit], inverse_mapping: dict[int, Qubit]):
        index, left, right = self.decode(policy)
        swap_class = SwapTwoQubitGate if index == SWAP_INDEX else BridgeTwoQubitGate
        if index == SWAP_INDEX:
            return SwapTwoQubitGate(inverse_current_mapping[left], inverse_current_mapping[right])
        else:
            return BridgeTwoQubitGate(inverse_mapping[left], None, inverse_mapping[right])

class ActionSpaceSwapOnly:

    def __init__(self, N: int, A: int):
        self.A = A
        self.N = N

    def get_space(self):
        A, N = self.A, self.N
        return gym.spaces.Discrete(N*N)

    def get_size(self):
        return self.N * self.N

    def encode(self, q0: int, q1: int):
        return q0 * self.N + q1

    def encode_best_swap(self, swap: TwoQubitGate, current_mapping: dict[Qubit, int], initial_mapping: dict[Qubit, int]):
        q0, q1 = current_mapping[swap.left], current_mapping[swap.right]
        return self.encode(q0, q1)

    def decode(self, policy: int):
        left = policy // self.N
        right = policy % self.N
        return left, right

    def decode_best_swap(self, policy: int, inverse_current_mapping: dict[int, Qubit], inverse_mapping: dict[int, Qubit]):
        left, right = self.decode(policy)
        return SwapTwoQubitGate(inverse_current_mapping[left], inverse_current_mapping[right])
