from qiskit.circuit import Qubit, QuantumCircuit


def get_cnot_num(cirt: QuantumCircuit):
    count = cirt.count_ops()
    return count.get('cx', 0) + 3 * count.get('swap', 0)


def qknob_metrics(in_cirt: QuantumCircuit, out_cirt: QuantumCircuit):
    depth_ratio = out_cirt.depth() / in_cirt.depth()
    in_cx_num = get_cnot_num(in_cirt)
    out_cx_num = get_cnot_num(out_cirt)
    cx_ratio = out_cx_num / in_cx_num
    return dict(depth_ratio=depth_ratio, cx_ratio=cx_ratio)



def show_mapping(mapping: dict[Qubit, int]):
    return {bit._index:int(idx) for bit, idx in mapping.items()}
