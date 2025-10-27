from typing import Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit.quantum_info import Statevector, Operator
from qiskit.transpiler import CouplingMap, Layout
from qiskit.circuit.library import Permutation

from contrib.common import GATE_NAME_MAPPING, get_inverse_mapping

THRESHOLD = 1e-10


def verify_circuit_equivalent(qc_origin: QuantumCircuit,
                              qc_mapped: QuantumCircuit,
                              final_mapping=None,
                              threshold: float = THRESHOLD):

    S_origin = Statevector.from_instruction(qc_origin)
    S_mapped = Statevector.from_instruction(qc_mapped)

    if final_mapping is None:
        # Fast checking.
        S_origin = np.sort(np.abs(S_origin.data))
        S_mapped = np.sort(np.abs(S_mapped.data))
        err = np.linalg.norm(S_origin - S_mapped)
        if err < threshold:
            print('OK')
            return True
        print("Error (Frobenius):", err)
        return False
    else:
        # Slow but accurate checking.
        qc = qc_origin
        final_layout = Layout(final_mapping) if not isinstance(final_mapping, Layout) else final_mapping
        layout = final_layout
        perm = [layout[qc.qubits[i]] for i in range(qc.num_qubits)]
        perm_gate = Permutation(qc.num_qubits, perm)
        S_origin = S_origin.evolve(Operator(perm_gate))
        if S_origin.equiv(S_mapped):
            print('OK')
            return True
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
