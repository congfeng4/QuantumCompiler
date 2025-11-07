"""
Action space encoding
"""
import logging
from typing import Optional, List, Union, Tuple

import gymnasium as gym
import numpy as np

from qiskit.circuit import Qubit
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.passes import *
from qiskit.transpiler.passmanager import PassManager
from hamap import IBMQHardwareArchitecture
from hamap.gates import TwoQubitGate, SwapTwoQubitGate, BridgeTwoQubitGate

from contrib.common import SWAP_INDEX, BRIDGE_INDEX, non_adj_common_pairs
from qiskit.transpiler.passes import OptimizeCliffords

logger = logging.getLogger("action_space")


class MyOptimizeCliffords(TransformationPass):

    def __init__(self):
        super().__init__()
        self.collect = CollectCliffords()
        self.optimize = OptimizeCliffords()

    def run(self, dag):
        self.collect.run()

    def name(self):
        return 'OptimizeCliffords'


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


def check_symmetric(pairs: list[tuple[int, int]]):
    for a, b in pairs:
        assert (b, a) in pairs, f'{(a, b)}'


def make_symmetric(pairs: set[tuple[int, int]]):
    for a, b in list(pairs):
        pairs.add((b, a))


class ActionSpace:
    ACTION_FINISH = '<finish>'
    ACTION_START = '<start>'

    TRANS_ROUTED = 0
    TRANS_UNROUTED = 1

    index_to_action: List[Union[Tuple[int, int], Tuple[int, TransformationPass], str]]

    def __init__(self, hardware: IBMQHardwareArchitecture, basic_gates):
        self.trans = [
            # Cancellation
            CommutativeCancellation(basis_gates=basic_gates),
            CommutativeInverseCancellation(),
            InverseCancellation(),
            # Optimize
            Optimize1qGates(basis=basic_gates),
            Optimize1qGatesDecomposition(basis=basic_gates),
            # Optimize1qGatesSimpleCommutation(basis=basic_gates),
            # TemplateOptimization(), # 这个非常慢
            # MyOptimizeCliffords(),
            # Remove
            RemoveIdentityEquivalent(),
        ]

        self.num_qubits = hardware.qubit_number
        swap_set = set(hardware.edges)
        bridge_set = set(non_adj_common_pairs(hardware.to_undirected()))
        make_symmetric(bridge_set)
        routing_actions = sorted(swap_set) + sorted(bridge_set)
        check_symmetric(routing_actions)
        assert len(routing_actions) == len(swap_set) + len(bridge_set)
        self.num_bridge = len(bridge_set)
        self.num_swap = len(swap_set)

        def make_trans_action():
            return [(num, opt) for opt in self.trans for num in [self.TRANS_ROUTED, self.TRANS_UNROUTED]]

        # [SpecialActions, RoutingActions, TransActions]
        self.index_to_action = [self.ACTION_START, self.ACTION_FINISH] + routing_actions + make_trans_action()

        self.action_to_index = {act: i for i, act in enumerate(self.index_to_action)}

    @property
    def num_trans(self):
        return len(self.trans)

    @property
    def num_route_actions(self):
        return self.num_swap + self.num_bridge

    @property
    def num_trans_actions(self):
        return 2 * self.num_trans

    def __repr__(self):
        return f'{self.__class__.__name__}({self.size})'

    @property
    def size(self):
        return len(self.action_to_index)

    def to_gym_space(self):
        return gym.spaces.Discrete(self.size)

    def encode(self, swap: TwoQubitGate, current_mapping: dict[Qubit, int], initial_mapping: dict[Qubit, int]):
        _, q0, q1 = two_qubit_gate_to_tuple(swap, current_mapping, initial_mapping)
        return self.action_to_index[(q0, q1)]

    def decode(self, policy: int, initial_mapping,
               inverse_current_mapping: dict[int, Qubit], inverse_mapping: dict[int, Qubit],
               hardware: IBMQHardwareArchitecture):
        action = self.index_to_action[policy]
        if isinstance(action, str):  # special
            return action
        if isinstance(action, tuple) and isinstance(action[1], (TransformationPass, PassManager)):  # transform
            return action
        left, right = action
        swap_class = SWAP_INDEX if (left, right) in hardware.edges else BRIDGE_INDEX
        if swap_class == SWAP_INDEX:
            return SwapTwoQubitGate(inverse_current_mapping[left], inverse_current_mapping[right])

        swap = BridgeTwoQubitGate(inverse_mapping[left], None, inverse_mapping[right])
        swap._middle = find_middle(swap, hardware, initial_mapping, inverse_mapping)
        return swap


if __name__ == '__main__':
    for name in ['tokyo', 'sycamore', 'rochester']:
        hardware = IBMQHardwareArchitecture(name)
        space = ActionSpace(hardware)
        print(space.action_to_index)
