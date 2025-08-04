import logging
from functools import cached_property
from typing import Optional

import gymnasium as gym
import numpy as np

from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit.dagcircuit import DAGNode

from hamap import IBMQHardwareArchitecture
from hamap.gates import TwoQubitGate, SwapTwoQubitGate, BridgeTwoQubitGate

from contrib.common import EXE_INDEX, SWAP_INDEX, BRIDGE_INDEX

logger = logging.getLogger("action_space")


def find_middle(best_swap_qubits: BridgeTwoQubitGate, hardware, initial_mapping, inverse_mapping) -> Optional[Qubit]:
    control, target = best_swap_qubits.left, best_swap_qubits.right
    control_index = initial_mapping[control]
    target_index = initial_mapping[target]
    # For each qubit q linked with control, check if target is linked with q.
    for _, potential_middle_index in hardware.out_edges(control_index):
        for _, potential_target_index in hardware.out_edges(potential_middle_index):
            if potential_target_index == target_index:
                return inverse_mapping[potential_middle_index]

    raise ValueError(f"Cannot find middle qubit for BRIDGE {best_swap_qubits}. Your circuit is probably wrong")


def two_qubit_gate_to_tuple(swap: TwoQubitGate, current_mapping: dict[Qubit, int], initial_mapping: dict[Qubit, int]):
    if isinstance(swap, BridgeTwoQubitGate):  # Already physical
        q0, q1 = initial_mapping[swap.left], initial_mapping[swap.right]
        return BRIDGE_INDEX, q0, q1
    else:
        q0, q1 = current_mapping[swap.left], current_mapping[swap.right]
        return SWAP_INDEX, q0, q1


class ActionSpace:
    # IMPORTANT: Swap and Bridge can share one NxN matrix since their qubits CANNOT conflict!

    def __init__(self, N: int):
        self.N = N

    def get_space(self):
        N = self.N
        return gym.spaces.Discrete(N*N)

    def get_size(self):
        return self.N * self.N

    def _encode(self, q0: int, q1: int):
        return q0 * self.N + q1

    def encode(self, swap: TwoQubitGate, current_mapping: dict[Qubit, int], initial_mapping: dict[Qubit, int],
               hardware: IBMQHardwareArchitecture):
        _, q0, q1 = two_qubit_gate_to_tuple(swap, current_mapping, initial_mapping)
        if isinstance(swap, BridgeTwoQubitGate):
            inverse_mapping = {val: key for key, val in initial_mapping.items()}
            find_middle(swap, hardware, initial_mapping, inverse_mapping)
        assert ((q0, q1) in hardware.edges) == isinstance(swap, SwapTwoQubitGate)
        return self._encode(q0, q1)

    def _decode(self, policy: int):
        left = policy // self.N
        right = policy % self.N
        return left, right

    def decode(self, policy: int, initial_mapping,
                         inverse_current_mapping: dict[int, Qubit], inverse_mapping: dict[int, Qubit],
                         hardware: IBMQHardwareArchitecture):
        left, right = self._decode(policy)
        swap_class = SWAP_INDEX if (left, right) in hardware.edges else BRIDGE_INDEX
        if swap_class == SWAP_INDEX:
            return SwapTwoQubitGate(inverse_current_mapping[left], inverse_current_mapping[right])

        swap = BridgeTwoQubitGate(inverse_mapping[left], None, inverse_mapping[right])
        swap._middle = find_middle(swap, hardware, initial_mapping, inverse_mapping)
        return swap


class ActionSpaceCandsAndCost:

    def __init__(self, N: int):
        self.N = N

    def get_size(self):
        return self.N * self.N

    def get_space(self):
        return gym.spaces.MultiBinary(self.get_size())

    def encode(self, candidates_with_cost, current_mapping, initial_mapping):
        action = np.zeros((self.N, self.N), np.float32)
        # Swap and bridge
        for swap, cost in candidates_with_cost:
            _, q0, q1 = two_qubit_gate_to_tuple(swap, current_mapping, initial_mapping)
            assert action[q0, q1] == 0, f"Conflict in action: {swap=} {q0=} {q1=}"
            prob = np.exp(-cost)  # Map [0->inf] to [0, 1]
        return action.reshape(-1)

    def decode(self, action, determistic=True):
        if determistic:
            index = np.argmin(action)

