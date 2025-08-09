from qiskit import QuantumCircuit

from hamap import (
    ha_mapping,  # Bridge selection performed with a
    # different algorithm than the one described in the paper.
    ha_mapping_paper_compliant,  # Bridge selection using the exact same algorithm
    # described in the paper.
    IBMQHardwareArchitecture,
)

# Create a Quantum Circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

hardware = IBMQHardwareArchitecture("tokyo")
initial_mapping = {qubit: i for i, qubit in enumerate(circuit.qubits)}

# Map the circuit with our hardware-aware heuristic and using SWAP & Bridge gates.
# Replace "ha_mapping" with "ha_mapping_paper_compliant" to use the version 100%
# compliant with the paper.
mapped_circuit, final_mapping = ha_mapping(
    circuit, initial_mapping, hardware
)

print(mapped_circuit.draw())
