import json
from enum import IntEnum

import numpy as np
from qiskit.circuit import Qubit, QuantumCircuit
from pathlib import Path
import networkx as nx

from qiskit.dagcircuit import DAGNode, DAGCircuit

from hamap.distance_matrix import get_distance_matrix_swap_number, get_distance_matrix_swap_number_and_error
from hamap.gates import TwoQubitGate
from hamap.hardware import IBMQHardwareArchitecture
from hamap.heuristics import _gate_op_cost
from hamap.layer import QuantumLayer

RESULT_DIR = Path(__file__).parent.parent / 'result'

NUM_ACTIONS = 2
EXE_INDEX = None
SWAP_INDEX = 0
BRIDGE_INDEX = 1


class Unit(IntEnum):
    K = 1024
    M = K * K
    G = M * M
    P = G * G


get_distance_matrix = get_distance_matrix_swap_number_and_error


def get_cnot_num(cirt: QuantumCircuit | DAGCircuit):
    count = cirt.count_ops()
    return count.get('cx', 0) + 3 * count.get('swap', 0)


def get_circuit_depth(cirt: QuantumCircuit | DAGCircuit):
    return cirt.depth()


def get_circuit_cost(front_layer: QuantumLayer, gates: list[DAGNode], current_mapping: dict[Qubit, int],
                     distance_matrix: np.ndarray, hardware: IBMQHardwareArchitecture, maxlen: int = -1):
    cost = 0
    for op in (front_layer.ops + gates)[:maxlen]:
        cost += _gate_op_cost(op, distance_matrix, current_mapping, hardware)
    return cost


def qknob_metrics(in_cirt: QuantumCircuit, out_cirt: QuantumCircuit | DAGCircuit):
    depth_ratio = out_cirt.depth() / in_cirt.depth()
    in_cx_num = get_cnot_num(in_cirt)
    out_cx_num = get_cnot_num(out_cirt)
    cx_ratio = out_cx_num / in_cx_num
    swap_num = out_cirt.count_ops().get('swap', 0)
    return dict(depth_ratio=depth_ratio, cx_ratio=cx_ratio, swap_num=swap_num)


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


def readable_float_dict(data: dict[str, float], places: int = 2):
    # Use float() to get rid of np.float32
    return {k: round(float(v), places) for k, v in data.items()}


def write_json(out_file: Path | str, data):
    if not isinstance(out_file, Path):
        out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding='utf8')


def read_json(in_file: Path | str):
    if not isinstance(in_file, Path):
        in_file = Path(in_file)
    return json.loads(in_file.read_text(encoding='utf8'))


def read_circuit(in_file: Path | str):
    if not isinstance(in_file, str):
        in_file = str(in_file)
    return QuantumCircuit.from_qasm_file(in_file)


def write_circuit(out_file: Path | str, qc: QuantumCircuit):
    from qiskit.qasm3 import dumps

    if not isinstance(out_file, Path):
        out_file = Path(out_file)
    out_file.write_text(dumps(qc), encoding='utf8')
