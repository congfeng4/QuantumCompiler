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


def get_circuit_cost(front_layer: QuantumLayer, gates: list[DAGNode], current_mapping: dict[Qubit, int],
                    distance_matrix: np.ndarray, hardware: IBMQHardwareArchitecture, maxlen: int = -1):
    cost = 0
    for op in (front_layer.ops + gates)[:maxlen]:
        cost += _gate_op_cost(op, distance_matrix, current_mapping, hardware)
    return cost
