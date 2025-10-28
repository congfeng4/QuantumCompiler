from typing import Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit.quantum_info import Statevector, Operator
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.circuit.library import Permutation
from qiskit.transpiler.passes import ApplyLayout, SetLayout

from contrib.common import GATE_NAME_MAPPING, get_inverse_mapping, convert_to_int_mapping

THRESHOLD = 1e-10


def check_equivalence_under_mapping(qc_in, qc_out, initial_mapping):
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

    for inst in qc.data: # type: CircuitInstruction
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


if __name__ == '__main__':
    pass
