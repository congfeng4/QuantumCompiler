from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from hamap._cli.compare_initial_mappings import get_mapping_cost, get_initial_mapping_from_annealing

from hamap import (
    ha_mapping,  # Bridge selection performed with a
                 # different algorithm than the one described in the paper.
    ha_mapping_paper_compliant,  # Bridge selection using the exact same algorithm
                                 # described in the paper.
    IBMQHardwareArchitecture,
)

def show_mapping(mapping: dict[Qubit, int]):
    return {bit._index:int(idx) for bit, idx in mapping.items()}


circuit = QuantumCircuit.from_qasm_file("./data/53Q_gate_Rochester/circuits/53Q_gate_Rochester_large_1_0_1.5_no.0.qasm")
hardware = IBMQHardwareArchitecture("rochester")
initial_mapping, cost, iter_num = get_initial_mapping_from_annealing(get_mapping_cost, circuit, hardware, max_steps=10)
print(f'Initial mapping {show_mapping(initial_mapping)}, cost {cost}, iter_num {iter_num}')

# Map the circuit with our hardware-aware heuristic and using SWAP & Bridge gates.
# Replace "ha_mapping" with "ha_mapping_paper_compliant" to use the version 100%
# compliant with the paper.
mapped_circuit, final_mapping = ha_mapping(
    circuit, initial_mapping, hardware
)

print(mapped_circuit.draw())
