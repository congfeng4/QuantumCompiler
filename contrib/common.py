from qiskit.circuit import Qubit, QuantumCircuit
from pathlib import Path
import networkx as nx

from qiskit.dagcircuit import DAGNode, DAGCircuit

from hamap.gates import TwoQubitGate

RESULT_DIR = Path(__file__).parent.parent / 'result'

NUM_ACTIONS = 2
EXE_INDEX = None
SWAP_INDEX = 0
BRIDGE_INDEX = 1


def get_cnot_num(cirt: QuantumCircuit | DAGCircuit):
    count = cirt.count_ops()
    return count.get('cx', 0) + 3 * count.get('swap', 0)


def qknob_metrics(in_cirt: QuantumCircuit, out_cirt: QuantumCircuit | DAGCircuit):
    depth_ratio = out_cirt.depth() / in_cirt.depth()
    in_cx_num = get_cnot_num(in_cirt)
    out_cx_num = get_cnot_num(out_cirt)
    cx_ratio = out_cx_num / in_cx_num
    num_swap = out_cirt.count_ops().get('swap', 0)
    return dict(depth_ratio=depth_ratio, cx_ratio=cx_ratio, num_swap=num_swap)


def show_mapping(mapping: dict[Qubit, int]):
    return {bit._index: int(idx) for bit, idx in mapping.items()}


def get_hardware_name(data_name: str):
    return data_name.split('_')[-1].lower()


def get_all_qknob_circuit_paths(data_root: Path = None) -> dict[str, list[Path]]:
    if data_root is None:
        data_root = Path(__file__).parent.parent / 'data'

    result = {}
    for subdir in data_root.iterdir():
        if 'Q' not in subdir.name or not subdir.is_dir():
            continue
        result[subdir.name] = list((subdir / 'circuits').glob('*.qasm'))

    return result


def qubit_index_from_op(op: DAGNode):
    return op.qargs[0]._index, op.qargs[1]._index


def qubit_index_from_swap(swap: TwoQubitGate):
    return swap.left._index, swap.right._index


def non_adj_common_pairs(G):
    """
    返回所有不相邻且有公共邻居的点对（u, v），u < v
    """
    pairs = set()
    # 遍历所有不相邻的点对
    for u, v in nx.non_edges(G):
        # 只要存在公共邻居就加入
        if nx.common_neighbors(G, u, v):
            pairs.add((u, v))
    return pairs
