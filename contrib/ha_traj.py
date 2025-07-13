"""
Run HA algorithm and obtain trajectories
"""
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from qiskit import QuantumCircuit
from typing import Any

from contrib.action import ActionType
from hamap._cli.compare_initial_mappings import get_mapping_cost, get_initial_mapping_from_annealing, \
    get_random_mapping, \
    initial_mapping_from_sabre, wrap_iterative_mapping_algorithm

from hamap import (
    ha_mapping,  # Bridge selection performed with a
    # different algorithm than the one described in the paper.
    ha_mapping_paper_compliant,  # Bridge selection using the exact same algorithm
    # described in the paper.
    IBMQHardwareArchitecture,
)
from contrib.common import show_mapping, qknob_metrics, RESULT_DIR
from hamap.gates import SwapTwoQubitGate, BridgeTwoQubitGate


HA_RESULT_DIR = RESULT_DIR / 'ha'


class InitialMappingStrategy(Enum):
    SIMULATE_ANNEALING = 'sa'
    SABRE = 'sabre'
    RANDOM = 'random'
    IDENTITY = 'identity'


def convert_action_to_gate(action: ActionType, quantum_circuit: QuantumCircuit):
    left, right = action['left'], action['right']
    if action['action'] == 'SWAP':
        best_swap_qubits = SwapTwoQubitGate(left=quantum_circuit.qubits[left], right=quantum_circuit.qubits[right])
    elif action['action'] == 'BRIDGE':
        middle = action.get('middle', 0)
        best_swap_qubits = BridgeTwoQubitGate(left=quantum_circuit.qubits[left], right=quantum_circuit.qubits[right],
                                              middle=quantum_circuit.qubits[middle])
    else:
        raise ValueError(action)
    return best_swap_qubits


def get_initial_mapping(circuit: QuantumCircuit, hardware: IBMQHardwareArchitecture,
                        initial_mapping_strategy: InitialMappingStrategy):
    if initial_mapping_strategy == InitialMappingStrategy.RANDOM:
        return get_random_mapping(circuit)
    if initial_mapping_strategy == InitialMappingStrategy.IDENTITY:
        return {qubit: i for i, qubit in enumerate(circuit.qubits)}
    if initial_mapping_strategy == InitialMappingStrategy.SIMULATE_ANNEALING:
        return get_initial_mapping_from_annealing(get_mapping_cost, circuit, hardware)[0]
    if initial_mapping_strategy == InitialMappingStrategy.SABRE:
        return initial_mapping_from_sabre(circuit, hardware, wrap_iterative_mapping_algorithm)
    raise ValueError(initial_mapping_strategy)


def run_ha(circ_path: str, hardware_name: str,
           initial_mapping_strategy: InitialMappingStrategy = InitialMappingStrategy.RANDOM):
    """
    Returns dict(metrics=metrics, init=readable_initial_mapping, input_circuit=circ_path,
                initial_mapping_strategy=initial_mapping_strategy.value, trajectory=trajectory)
    """
    trajectory = []
    circuit = QuantumCircuit.from_qasm_file(circ_path)
    hardware = IBMQHardwareArchitecture(hardware_name)
    initial_mapping = get_initial_mapping(circuit, hardware, initial_mapping_strategy)
    readable_initial_mapping = show_mapping(initial_mapping)
    mapped_circuit, final_mapping = ha_mapping(circuit, initial_mapping, hardware, trajectory=trajectory)
    metrics = qknob_metrics(circuit, mapped_circuit)

    return dict(metrics=metrics, hardware_name=hardware_name,
                initial_mapping=readable_initial_mapping, input_circuit=circ_path,
                init=initial_mapping_strategy.value, trajectory=trajectory)


def load_and_group_trajectories(traj_root: Path = None) -> dict[str, list[Path]]:
    """
    Utility function to load trajectories and group them by hardware_name.
    """
    if traj_root is None:
        traj_root = HA_RESULT_DIR
    result = defaultdict(list)
    for traj_file in traj_root.glob('*.json'):
        traj_data = json.loads(traj_file.read_text())
        try:
            hardware_name = traj_data['hardware_name']
        except KeyError:
            print(f'Discard file {traj_file} due to no hardware_name')
            continue
        result[hardware_name].append(traj_file)
    return result


if __name__ == '__main__':
    print(load_and_group_trajectories())
    exit()

    from pprint import pp
    hardware = IBMQHardwareArchitecture("tokyo")
    result = run_ha("../data/20Q_depth_Tokyo/circuits/20Q_depth_Tokyo_large_None_5_2.55_no.2.qasm",
                    "tokyo")
    pp(result)
    # ok = verify_trajectory()
