"""
Run HA algorithm and obtain trajectories
"""
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGNode

from contrib.ha_traj import run_ha, InitialMappingStrategy
from hamap import (
    IBMQHardwareArchitecture, mapping_to_str,
)
from hamap.gates import SwapTwoQubitGate, BridgeTwoQubitGate
from hamap.layer import QuantumLayer, update_layer
from hamap.mapping import _create_empty_dagcircuit_from_existing, _adapt_quantum_circuit_and_mapping_arity
from contrib.common import get_all_qknob_circuit_paths, get_hardware_name
from pathlib import Path

import pytest

def verify_trajectory(
    quantum_circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    trajectory: list[dict[str, int | str]]
) -> bool:
    """Verify the correctness of the trajectory

    :param quantum_circuit: the quantum circuit to map.
    :param trajectory: collected from ha_mapping().
    :param hardware: hardware data such as connectivity, gate time, gate errors, ...
    :return: The final circuit along with the mapping obtained at the end of the
        iterative procedure.
    """
    initial_mapping = {}
    traj_index = 0
    for action in trajectory:
        if action['action'] == 'MAP':
            logical, physical = action['logical'], action['physical']
            initial_mapping[quantum_circuit.qubits[logical]] = physical
            traj_index += 1
        else:
            break
    if len(initial_mapping) != quantum_circuit.num_qubits:
        return False

    _adapt_quantum_circuit_and_mapping_arity(quantum_circuit, initial_mapping, hardware)
    # Creating the internal data structures that will be used in this function.
    dag_circuit = circuit_to_dag(quantum_circuit)
    resulting_dag_quantum_circuit = _create_empty_dagcircuit_from_existing(dag_circuit)

    current_mapping = initial_mapping
    explored_mappings: set[str] = set()

    # Sorting all the quantum operations in topological order once for all.
    # May require significant memory on large circuits...
    topological_nodes: list[DAGNode] = list(dag_circuit.topological_op_nodes())
    current_node_index = 0
    # Creating the initial front layer.
    front_layer = QuantumLayer()
    current_node_index = update_layer(
        front_layer, topological_nodes, current_node_index
    )
    trans_mapping = initial_mapping.copy()

    # Start of the iterative algorithm
    while not front_layer.is_empty():
        execute_gate_list = QuantumLayer()
        for op in front_layer.ops:
            if hardware.can_natively_execute_operation(op, current_mapping):
                execute_gate_list.add_operation(op)
                # Delaying the remove operation because we do not want to remove from
                # a container we are iterating on.
                # front_layer.remove_operation(op)
        if not execute_gate_list.is_empty():
            front_layer.remove_operations_from_layer(execute_gate_list)
            execute_gate_list.apply_back_to_dag_circuit(
                resulting_dag_quantum_circuit, initial_mapping, trans_mapping
            )
            # Empty the explored mappings because at least one gate has been executed.
            explored_mappings.clear()
        else:
            inverse_mapping = {val: key for key, val in initial_mapping.items()}

            # Convert the action into a TwoQubitGate.
            best_swap_qubits = None
            try:
                action = trajectory[traj_index]
                traj_index += 1
            except IndexError:
                return False
            left, right = action['left'], action['right']
            if action['action'] == 'SWAP':
                best_swap_qubits = SwapTwoQubitGate(left=quantum_circuit.qubits[left], right=quantum_circuit.qubits[right])
            elif action['action'] == 'BRIDGE':
                middle = action['middle']
                best_swap_qubits = BridgeTwoQubitGate(left=quantum_circuit.qubits[left], right=quantum_circuit.qubits[right],
                                                      middle=quantum_circuit.qubits[middle])

            # We now have our best SWAP/Bridge, let's perform it!
            current_mapping = best_swap_qubits.update_mapping(current_mapping)
            if isinstance(best_swap_qubits, SwapTwoQubitGate):
                control, target = current_mapping[best_swap_qubits.left], current_mapping[best_swap_qubits.right]
                swap_control, swap_target = inverse_mapping[control], inverse_mapping[target]
                best_swap_qubits = SwapTwoQubitGate(
                    swap_control, swap_target
                )
                #print("swap gates is :", best_swap_qubits.left, best_swap_qubits.right)
                trans_mapping[best_swap_qubits.left], trans_mapping[best_swap_qubits.right] = (
                    trans_mapping[best_swap_qubits.right],
                    trans_mapping[best_swap_qubits.left],
                )
            else:
                #print("brige gate is :", best_swap_qubits.left, best_swap_qubits.middle, best_swap_qubits.right)
                pass
            explored_mappings.add(mapping_to_str(current_mapping))
            best_swap_qubits.apply(resulting_dag_quantum_circuit, front_layer, initial_mapping, trans_mapping)
        # Anyway, update the current front_layer
        current_node_index = update_layer(
            front_layer, topological_nodes, current_node_index
        )

    # We are done here, we just need to return the results
    # resulting_dag_quantum_circuit.draw(scale=1, filename="qcirc.dot")
    resulting_circuit = dag_to_circuit(resulting_dag_quantum_circuit)
    return True


DATA_ROOT = Path(__file__).parent.parent / 'data'
assert DATA_ROOT.exists(), DATA_ROOT

ALL_QKNOB_CIRCUIT_PATHS = get_all_qknob_circuit_paths(DATA_ROOT)

@pytest.mark.parametrize('data', ALL_QKNOB_CIRCUIT_PATHS.keys())
def test_traj(data: str):
    for circuit_path in ALL_QKNOB_CIRCUIT_PATHS[data]:
        hardware_name = get_hardware_name(data)
        result = run_ha(str(circuit_path), hardware_name, InitialMappingStrategy.RANDOM)
        trajectory = result['trajectory']
        circuit = QuantumCircuit.from_qasm_file(str(circuit_path))
        hardware = IBMQHardwareArchitecture(hardware_name)
        assert verify_trajectory(circuit, hardware, trajectory), f'Trajectory failed to map this circuit: {circuit_path}'
