from typing import Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from contrib.common import GATE_NAME_MAPPING, get_inverse_mapping

THRESHOLD = 1e-10


def verify_circuit_equivalent(qc_origin: QuantumCircuit,
                              qc_mapped: QuantumCircuit,
                              threshold: float = THRESHOLD):

    S_origin = Statevector(qc_origin).data
    S_mapped = Statevector(qc_mapped).data
    err = np.linalg.norm(S_origin - S_mapped)
    if err < threshold:
        print('OK')
        return True
    print("Error (Frobenius):", err)
    return False


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
