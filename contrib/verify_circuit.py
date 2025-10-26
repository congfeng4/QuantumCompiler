import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

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


if __name__ == '__main__':
    pass
