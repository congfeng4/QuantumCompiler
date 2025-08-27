"""
Action space encoding
"""
import logging
from typing import Optional

import gymnasium as gym
import numpy as np

from qiskit.circuit.quantumregister import Qubit

from hamap import IBMQHardwareArchitecture
from hamap.gates import TwoQubitGate, SwapTwoQubitGate, BridgeTwoQubitGate

from contrib.common import SWAP_INDEX, BRIDGE_INDEX, non_adj_common_pairs
from hamap.heuristics import sabre_heuristic

logger = logging.getLogger("action_space")


def find_middle(best_swap_qubits: BridgeTwoQubitGate, hardware, initial_mapping, inverse_mapping) -> Optional[Qubit]:
    control, target = best_swap_qubits.left, best_swap_qubits.right
    control_index = initial_mapping[control]
    target_index = initial_mapping[target]
    # For each qubit q linked with control, check if the target is linked with q.
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
        return gym.spaces.Discrete(N * N)

    def get_size(self):
        return self.N * self.N

    def _encode(self, q0: int, q1: int):
        return q0 * self.N + q1

    def encode(self, swap: TwoQubitGate, current_mapping: dict[Qubit, int], initial_mapping: dict[Qubit, int],
               hardware: IBMQHardwareArchitecture):
        _, q0, q1 = two_qubit_gate_to_tuple(swap, current_mapping, initial_mapping)
        if isinstance(swap, BridgeTwoQubitGate):
            inverse_mapping = {val: key for key, val in initial_mapping.items()}
            find_middle(swap, hardware, initial_mapping, inverse_mapping)  # Check this bridge is valid.
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


def check_symmetric(pairs: list[tuple[int, int]]):
    for a, b in pairs:
        assert (b, a) in pairs, f'{(a, b)}'


def make_symmetric(pairs: set[tuple[int, int]]):
    for a, b in list(pairs):
        pairs.add((b, a))


class ActionSpaceEdge:

    def __init__(self, hardware: IBMQHardwareArchitecture):
        self.N = hardware.qubit_number
        swap_set = set(hardware.edges)
        bridge_set = set(non_adj_common_pairs(hardware.to_undirected()))
        make_symmetric(bridge_set)
        self.action_list = sorted(swap_set) + sorted(bridge_set)
        check_symmetric(self.action_list)
        assert len(self.action_list) == len(swap_set) + len(bridge_set), \
            f'{len(swap_set)=} {len(bridge_set)=} {len(self.action_list)=}'
        self.action_to_index = {act: i for i, act in enumerate(self.action_list)}

    def __repr__(self):
        ratio = round(self.get_size() / self.N ** 2, 2)
        return f'<{self.__class__.__name__}(Size={self.get_size()}, N^2={self.N ** 2}, Ratio={ratio})>'

    def get_size(self):
        return len(self.action_list)

    def get_space(self):
        return gym.spaces.Discrete(self.get_size())

    def encode(self, swap: TwoQubitGate, current_mapping: dict[Qubit, int], initial_mapping: dict[Qubit, int]):
        _, q0, q1 = two_qubit_gate_to_tuple(swap, current_mapping, initial_mapping)
        return self.action_to_index[(q0, q1)]

    def decode(self, policy: int, initial_mapping,
               inverse_current_mapping: dict[int, Qubit], inverse_mapping: dict[int, Qubit],
               hardware: IBMQHardwareArchitecture):
        left, right = self.action_list[policy]
        swap_class = SWAP_INDEX if (left, right) in hardware.edges else BRIDGE_INDEX
        if swap_class == SWAP_INDEX:
            return SwapTwoQubitGate(inverse_current_mapping[left], inverse_current_mapping[right])

        swap = BridgeTwoQubitGate(inverse_mapping[left], None, inverse_mapping[right])
        swap._middle = find_middle(swap, hardware, initial_mapping, inverse_mapping)
        return swap


class ActionSpaceSelectQubitToSwap:
    """
    Select a physical qubit to swap with.
    """

    def __init__(self, hardware: IBMQHardwareArchitecture):
        self.N = hardware.qubit_number

    def get_space(self):
        return gym.spaces.Discrete(self.N)


class ActionSpaceEdgeWithMap(ActionSpaceEdge):
    """
    Select a physical qubit to swap with.
    """

    def __init__(self, hardware: IBMQHardwareArchitecture):
        super().__init__(hardware)

    def get_size(self):
        return super().get_size() + self.N

    def get_size_nomap(self):
        return super().get_size()

    def decode(self, policy: int, initial_mapping,
               inverse_current_mapping: dict[int, Qubit], inverse_mapping: dict[int, Qubit],
               hardware: IBMQHardwareArchitecture):
        if 0 <= policy < self.N:
            return int(policy)
        return super().decode(policy - self.N, initial_mapping, inverse_mapping, inverse_mapping, hardware)


if __name__ == '__main__':
    for name in ['tokyo', 'sycamore', 'rochester']:
        hardware = IBMQHardwareArchitecture(name)
        space = ActionSpaceEdge(hardware)
        print(space)
