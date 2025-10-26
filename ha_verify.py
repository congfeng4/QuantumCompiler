import numpy as np
from qiskit import QuantumCircuit, transpile

from contrib.common import convert_to_int_mapping, write_circuit, write_json
from contrib.random_graphs import create_coupling_graph
from hamap import IBMQHardwareArchitecture
from hamap.mapping import ha_mapping
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy
from contrib.verify_circuit import verify_circuit_equivalent, verify_circuit_routed


if __name__ == '__main__':
    path = './data/nam_circs/barenco_tof_3.qasm'
    qc = QuantumCircuit.from_qasm_file(path)
    hardware = IBMQHardwareArchitecture('Star', num_nodes=qc.num_qubits)

    qc_out = transpile(qc, coupling_map=create_coupling_graph(hardware),
                       optimization_level=0, #initial_layout=init
                       )
    # init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)
    # qc_out, _ = ha_mapping(qc, init, hardware)
    # init = show_mapping(init)

    assert verify_circuit_equivalent(qc, qc_out)
    assert verify_circuit_routed(qc_out, hardware)
