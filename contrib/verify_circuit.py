from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit.quantum_info import Statevector, Operator
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.circuit.library import Permutation
from qiskit.transpiler.passes import ApplyLayout, SetLayout
from qiskit import transpile

from contrib.common import GATE_NAME_MAPPING, get_inverse_mapping, convert_to_int_mapping, read_mapping, read_json
from contrib.random_graphs import closest_factors
from qiskit.quantum_info.random import random_pauli
from qiskit.primitives.backend_estimator_v2 import BackendEstimatorV2 as Estimator



THRESHOLD = 1e-10


def check_equivalence_under_mapping(qc_in, qc_out, initial_mapping):
    # 对于qiskit transpile出来的电路，不需要Apply Layout。
    if isinstance(initial_mapping, Layout):
        initial_layout = initial_mapping
    else:
        vreg = QuantumRegister(len(initial_mapping), name='q')
        initial_layout = Layout({vreg[v]: p for v, p in initial_mapping.items()})

    pm = PassManager([SetLayout(initial_layout), ApplyLayout()])
    qc_mapped = pm.run(qc_in)
    return Statevector(qc_mapped).equiv(Statevector(qc_out))


def verify_circuit_routed(qc: QuantumCircuit, graph: Union[CouplingMap, nx.Graph], initial_mapping):
    inverse_mapping = get_inverse_mapping(initial_mapping)

    def has_edge(u, v):
        if isinstance(graph, CouplingMap):
            return graph.distance(u, v) == 1
        if isinstance(graph, nx.Graph):
            return graph.has_edge(u, v)
        raise TypeError(graph)

    for inst in qc.data:  # type: CircuitInstruction
        qubits = inst.qubits
        if len(qubits) == 1:
            continue
        if len(qubits) == 2:
            u, v = qubits
            if not has_edge(inverse_mapping[u], inverse_mapping[v]):
                print('Error:', inst.operation.name, (u, v), 'not in edges')
                return False
            if inst.operation.name == 'swap':
                inverse_mapping[u], inverse_mapping[v] = inverse_mapping[v], inverse_mapping[u]
        else:
            print('Error: only 1 and 2 qubit ops are allowed', inst)
            return False

    return True


def check_output_ours(ours_dir: Path, qubits_threshold=10):
    for subdir in Path(ours_dir).iterdir():
        qc_in = QuantumCircuit.from_qasm_file(subdir / 'input_circuit.qasm')
        qc_out = QuantumCircuit.from_qasm_file(subdir / 'output_circuit.qasm')
        initial_mapping = read_mapping(subdir / 'initial_mapping.json')
        # final_mapping = read_mapping(subdir / 'final_mapping.json')
        if qc_in.num_qubits > qubits_threshold:
            print('skip', subdir.name, qc_in.num_qubits)
            continue
        ok = check_equivalence_under_mapping(qc_in, qc_out, initial_mapping)
        # print(subdir.name, ok)
        assert ok


def check_qiskit_transpile(data_dir: Path, qubits_threshold=10):
    ok_basic = set()
    notok_basic = set()

    for qc_path in Path(data_dir).glob('*.qasm'):
        qc_in = QuantumCircuit.from_qasm_file(qc_path)
        num_qubits = qc_in.num_qubits
        if num_qubits > qubits_threshold:
            # print('skip', qc_path.name, num_qubits)
            continue

        rows, cols = closest_factors(num_qubits)

        for name, cm in [('line', CouplingMap.from_line(num_qubits)), ('grid', CouplingMap.from_grid(rows, cols)),
                   ('ring', CouplingMap.from_ring(num_qubits)) ]:
            for routing_method in ['sabre', 'basic']:
                for layout_method in ['trivial', 'dense', 'sabre']:
                    for opt_level in range(4):
                        qc_out = transpile(qc_in, coupling_map=cm,
                                           layout_method=layout_method, routing_method=routing_method,
                                           optimization_level=opt_level)
                        assert qc_out.num_qubits == qc_in.num_qubits

                        ok = check_equivalence_under_mapping(qc_in, qc_out, qc_out.layout.initial_layout)
                        if not ok:
                            print(qc_in.global_phase, qc_out.global_phase)
                            notok_basic.update(qc_in.count_ops().keys())
                            print(qc_path.name, routing_method, layout_method, opt_level, name)
                        else:
                            ok_basic.update(qc_in.count_ops().keys())

    print(ok_basic, notok_basic)


if __name__ == '__main__':
    check_qiskit_transpile(Path('../data/nam_circs'))
    # check_output_ours('../output/ours')