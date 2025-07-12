"""
Run HA algorithm on QKNOB dataset.
"""
from qiskit import QuantumCircuit

from hamap._cli.compare_initial_mappings import get_mapping_cost, get_initial_mapping_from_annealing

from hamap import (
    ha_mapping,  # Bridge selection performed with a
                 # different algorithm than the one described in the paper.
    ha_mapping_paper_compliant,  # Bridge selection using the exact same algorithm
                                 # described in the paper.
    IBMQHardwareArchitecture,
)
from common import show_mapping, qknob_metrics

max_steps = 1000

circuit = QuantumCircuit.from_qasm_file("./data/20Q_depth_Tokyo/circuits/20Q_depth_Tokyo_large_None_5_2.55_no.2.qasm")
hardware = IBMQHardwareArchitecture("tokyo")
initial_mapping, cost, iter_num = get_initial_mapping_from_annealing(get_mapping_cost, circuit, hardware, max_steps=max_steps)
print(f'Initial mapping {show_mapping(initial_mapping)}, cost {cost}, iter_num {iter_num}')

# Map the circuit with our hardware-aware heuristic and using SWAP & Bridge gates.
# Replace "ha_mapping" with "ha_mapping_paper_compliant" to use the version 100%
# compliant with the paper.
mapped_circuit, final_mapping = ha_mapping(
    circuit, initial_mapping, hardware
)

# print(mapped_circuit.draw())

print(qknob_metrics(circuit, mapped_circuit))
