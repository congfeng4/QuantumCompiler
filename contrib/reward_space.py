from enum import Enum
import enum

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGNode

from hamap.gates import SwapTwoQubitGate, BridgeTwoQubitGate, TwoQubitGate
from hamap.heuristics import sabre_heuristic, _gate_op_cost
from hamap.layer import QuantumLayer, update_layer
from hamap.mapping import _adapt_quantum_circuit_and_mapping_arity, _create_empty_dagcircuit_from_existing
from hamap import IBMQHardwareArchitecture, mapping_to_str

from contrib.common import get_distance_matrix
import logging

from hamap.swap import get_all_swap_bridge_candidates



@enum.unique
class RewardMode(Enum):
    HEURISTIC_COST = 0
    GATE_NUM_COST = 1
    GATE_NUM_AND_HEURISTIC_COST = 2


@enum.unique
class BaselineMode(Enum):
    NONE = 0
    SUBTRACT_MIN = 1
    SUBTRACT_AVG =2
    DIVIDE_MIN = 3
    DIVIDE_AVG =4
    MINMAX = 5


class RewardSpace:

    def __init__(self,
                 reward_mode: RewardMode = RewardMode.HEURISTIC_COST,
                 look_ahead_depth: int = 16,
                 look_ahead_weight: float = 0.5,
                 baseline_mode: BaselineMode = BaselineMode.SUBTRACT_MIN,
                 ):
        self.reward_mode = reward_mode
        self.look_ahead_depth = look_ahead_depth
        self.look_ahead_weight = look_ahead_weight
        self.baseline_mode = baseline_mode

    def __call__(self, hardware, front_layer, topological_nodes,
               current_node_index, current_mapping, initial_mapping, trans_mapping, distance_matrix, swap):
        raw_h_cost = sabre_heuristic(
            hardware=hardware, front_layer=front_layer, topological_nodes=topological_nodes,
            current_node_index=current_node_index, current_mapping=current_mapping,
            initial_mapping=initial_mapping, trans_mapping=trans_mapping,
            distance_matrix=distance_matrix, tentative_gate=swap, look_ahead_depth=self.look_ahead_depth,
            look_ahead_weight=self.look_ahead_weight,
        )
        normalized_h_cost = self.normalize_h_cost(raw_h_cost)
        h_cost = raw_h_cost if self.baseline_mode == BaselineMode.NONE else normalized_h_cost

        gate_num_cost = 0.01

        # IMPORTANT: two terms have different effects:
        # -cost can converge model quickly on startup but rebound later.
        # num_exe - 3 converge slowly but will not rebound.
        # TODO: an annealing scheme needed
        if self.reward_mode == RewardMode.GATE_NUM_AND_HEURISTIC_COST:
            reward = -h_cost - gate_num_cost
        elif self.reward_mode == RewardMode.GATE_NUM_COST:
            reward = -gate_num_cost
        elif self.reward_mode == RewardMode.HEURISTIC_COST:
            reward = -h_cost
        else:
            raise ValueError(self.reward_mode)
        return reward

    def normalize_h_cost(self, h_cost: float, min_cost: float, max_cost: float, avg_cost: float):
        # IMPORTANT: subtract baseline from heuristic cost can stablize late-term training. No explosion!
        if self.baseline_mode == BaselineMode.SUBTRACT_MIN:
            return h_cost - min_cost
        if self.baseline_mode == BaselineMode.SUBTRACT_AVG:
            return h_cost - avg_cost
        if self.baseline_mode == BaselineMode.DIVIDE_AVG:
            return h_cost / avg_cost
        if self.baseline_mode == BaselineMode.DIVIDE_MIN:
            return h_cost / min_cost
        if self.baseline_mode == BaselineMode.MINMAX:
            base = max_cost - min_cost
            return 0 if base == 0 else (h_cost - min_cost) / base
        return h_cost


def get_circuit_cost(front_layer: QuantumLayer, gates: list[DAGNode], current_mapping: dict[Qubit, int],
                    distance_matrix: np.ndarray, hardware: IBMQHardwareArchitecture):
    cost = 0
    for op in front_layer.ops + gates:
        cost += _gate_op_cost(op, distance_matrix, current_mapping, hardware)
    return cost
