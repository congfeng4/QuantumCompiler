# verify_mapping.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

THRESHOLD = 1e-10  # 误差容忍


def remap_circuit(qc: QuantumCircuit, P: dict) -> QuantumCircuit:
    """
    把 qc 的虚拟比特 i 映射到 P[i] 上，返回新电路。
    P[i] 必须是 0~n-1 的整数。
    """
    n = qc.num_qubits
    new_qc = QuantumCircuit(n)
    # 按原始电路 gate 顺序重放，只改 qubit 索引
    for inst in qc.data:
        gate = inst.operation
        old_qargs = inst.qubits  # 虚拟 qubit 对象
        new_qargs = [new_qc.qubits[P[q._index]] for q in old_qargs]
        new_qc.append(gate, new_qargs)
    return new_qc


def verify_under_mapping(qc_origin: QuantumCircuit,
                         qc_mapped: QuantumCircuit,
                         P: dict,
                         threshold: float = THRESHOLD):
    """
    比较 qc_mapped 的酉矩阵 与 按 P 重映射后的 qc_origin 的酉矩阵
    """
    # 按相同顺序重放 origin，但虚拟比特用 P 映射
    qc_origin_remapped = remap_circuit(qc_origin, P)

    U_origin = Operator(qc_origin_remapped).data
    U_mapped = Operator(qc_mapped).data

    err = np.linalg.norm(U_origin - U_mapped)
    print("Operator error (Frobenius):", err)
    return err < threshold


if __name__ == '__main__':
    pass
