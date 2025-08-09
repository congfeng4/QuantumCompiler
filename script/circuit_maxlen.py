"""
Data 53Q_depth_Sycamore, maxlen: 895
Data 53Q_gate_Sycamore, maxlen: 807
Data 20Q_depth_Tokyo, maxlen: 502
Data 53Q_gate_Rochester, maxlen: 794
Data 20Q_gate_Tokyo, maxlen: 556
Data 53Q_depth_Rochester, maxlen: 559
"""
from qiskit import QuantumCircuit
from pathlib import Path
from contrib.common import get_all_qknob_circuit_paths


def get_circuit_cnot_number(qc: QuantumCircuit):
    return qc.count_ops().get('cx', 0)


def get_circuit_maxlen(qc_list: list[Path]):
    return max(map(lambda path: get_circuit_cnot_number(QuantumCircuit.from_qasm_file(str(path))),
                   qc_list))


if __name__ == '__main__':
    ALL_QKNOB_CIRCUIT_PATHS = get_all_qknob_circuit_paths()

    for dataname, circuit_list in ALL_QKNOB_CIRCUIT_PATHS.items():
        maxlen = get_circuit_maxlen(circuit_list)
        print(f'Data {dataname}, maxlen: {maxlen}')
