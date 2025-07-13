"""
Run HA algorithm on QKNOB dataset.
"""
from qiskit import QuantumCircuit

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
from contrib.common import show_mapping, qknob_metrics


if __name__ == '__main__':
    max_steps = 1000

    circuit = QuantumCircuit.from_qasm_file("./data/")
    hardware = IBMQHardwareArchitecture("tokyo")
    # initial_mapping, cost, iter_num = get_initial_mapping_from_annealing(get_mapping_cost, circuit, hardware, max_steps=max_steps)
    initial_mapping = initial_mapping_from_sabre(circuit, hardware, wrap_iterative_mapping_algorithm)
    print(f'Initial mapping {show_mapping(initial_mapping)}')

    # Map the circuit with our hardware-aware heuristic and using SWAP & Bridge gates.
    # Replace "ha_mapping" with "ha_mapping_paper_compliant" to use the version 100%
    # compliant with the paper.
    mapped_circuit, final_mapping = ha_mapping(
        circuit, initial_mapping, hardware
    )

    # print(mapped_circuit.draw())

    print(qknob_metrics(circuit, mapped_circuit))
