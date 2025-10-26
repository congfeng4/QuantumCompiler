import numpy as np
from qiskit import QuantumCircuit, transpile

from contrib.common import convert_to_int_mapping, write_circuit, write_json
from contrib.random_graphs import create_coupling_graph
from hamap import IBMQHardwareArchitecture
from hamap.mapping import ha_mapping
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy
from contrib.verify_circuit import verify_circuit_equivalent, Operator


if __name__ == '__main__':
    path = './data/nam_circs/barenco_tof_3.qasm'
    qc = QuantumCircuit.from_qasm_file(path)
    hardware = IBMQHardwareArchitecture('Star', num_nodes=qc.num_qubits)

    # init = {q: i for i, q in enumerate(qc.qubits)}
    # print('init', init)

    qc_out = transpile(qc, coupling_map=create_coupling_graph(hardware),
                       optimization_level=0, #initial_layout=init
                       )
    # init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)
    # qc_out, _ = ha_mapping(qc, init, hardware)
    # init = show_mapping(init)
    # 获取实际 layout
    # layout = qc_out.layout.final_layout  # 或 qc_out.layout if final_layout 不存在
    print(qc_out.layout.initial_layout)
    print(qc_out.layout.final_layout)

    # 构造映射 P: virtual → physical
    # P = {q: layout[q] for q in qc.qubits}

    # write_circuit('./qc_out.qasm', qc_out)
    # write_circuit('./qc_in.qasm', qc)
    # write_json('./init.json', show_mapping(P))
    # U1 = Operator(qc).data
    # U2 = Operator(qc_out).dat
    # np.savetxt('./U_in.txt', U1)
    # np.savetxt('./U_out.txt', U2)

    assert verify_circuit_equivalent(qc, qc_out)
