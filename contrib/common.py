import itertools
import json
from collections import defaultdict
from enum import IntEnum, Enum
from typing import Union
import jsons
import numpy as np
from qiskit.circuit import Qubit, QuantumCircuit
from qiskit.quantum_info import Operator
from pathlib import Path
import networkx as nx

from qiskit.dagcircuit import DAGNode, DAGCircuit, DAGOpNode
from qiskit.circuit.library import get_standard_gate_name_mapping
from hamap.distance_matrix import get_distance_matrix_swap_number, get_distance_matrix_swap_number_and_error
from hamap.gates import TwoQubitGate
from hamap.hardware import IBMQHardwareArchitecture
from hamap.heuristics import _gate_op_cost
from hamap.layer import QuantumLayer, update_layer

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

GATE_NAME_MAPPING = get_standard_gate_name_mapping()


def get_cnot_num(cirt: Union[QuantumCircuit, DAGCircuit]):
    count = cirt.count_ops()
    return count.get('cx', 0) + 3 * count.get('swap', 0)


def get_circuit_depth(cirt: Union[QuantumCircuit, DAGCircuit]):
    return cirt.depth()


def get_front_layer(dag: DAGCircuit):
    topological_nodes: list[DAGOpNode] = list(dag.topological_op_nodes())
    front_layer = QuantumLayer()
    update_layer(
        front_layer, topological_nodes, 0,
    )
    return front_layer


def dag_to_sequence(dag: DAGCircuit):
    topological_nodes: list[DAGOpNode] = list(dag.topological_op_nodes())
    # Resort according to node levels.
    build_op_node_level(dag, topological_nodes, sort_by_level=True)

    front_layer = QuantumLayer()  # Obtain the front layer, ensuring they are in the sequence front.
    current_node_index = update_layer(
        front_layer, topological_nodes, 0,
    )
    return front_layer.ops + topological_nodes[current_node_index:]


def get_circuit_cost(dag: DAGCircuit, current_mapping: dict[Qubit, int],
                     distance_matrix: np.ndarray, hardware: IBMQHardwareArchitecture, maxlen: int = -1):
    cost = 0
    sequence = dag_to_sequence(dag)
    for op in sequence[:maxlen]:
        cost += _gate_op_cost(op, distance_matrix, current_mapping, hardware)
    return cost


def qknob_metrics(in_cirt: QuantumCircuit, out_cirt: Union[QuantumCircuit, DAGCircuit]):
    depth_ratio = out_cirt.depth() / in_cirt.depth()
    ops_ratio = get_total_ops(out_cirt) / get_total_ops(in_cirt)
    in_cx_num = get_cnot_num(in_cirt)
    out_cx_num = get_cnot_num(out_cirt)
    cx_ratio = out_cx_num / in_cx_num
    return dict(depth_ratio=depth_ratio, ops_ratio=ops_ratio, cx_ratio=cx_ratio)


def get_total_ops(qc):
    return sum(qc.count_ops().values())


def get_gate_set(qc: QuantumCircuit):
    return list(qc.count_ops().keys())


def get_weighted_ops(ops_count: dict, one_qubit_gate_weight: float = None):
    if one_qubit_gate_weight is None:
        return sum(ops_count.values())
    if not ops_count:
        return 0
    qubits_to_count = defaultdict(int)
    for name, count in ops_count.items():
        num_qubits = GATE_NAME_MAPPING[name].num_qubits
        qubits_to_count[num_qubits] += 1
    assert max(qubits_to_count.keys()) <= 2, qubits_to_count
    return qubits_to_count[1] * one_qubit_gate_weight + qubits_to_count[2] * (1 - one_qubit_gate_weight)


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


def write_json(out_file: Union[Path, str], data):
    if not isinstance(out_file, Path):
        out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(jsons.dump(data), indent=4, ensure_ascii=False), encoding='utf8')


def read_json(in_file: Union[Path, str]):
    if not isinstance(in_file, Path):
        in_file = Path(in_file)
    return json.loads(in_file.read_text(encoding='utf8'))


def read_circuit(in_file: Union[Path, str]):
    if not isinstance(in_file, str):
        in_file = str(in_file)
    return QuantumCircuit.from_qasm_file(in_file)


def write_circuit(out_file: Union[Path, str], qc: QuantumCircuit):
    from qiskit.qasm3 import dumps

    if not isinstance(out_file, Path):
        out_file = Path(out_file)
    out_file.write_text(dumps(qc), encoding='utf8')


def build_op_node_level(dag: DAGCircuit, topological_nodes: list[DAGOpNode], sort_by_level: bool = False):
    """
    Compute the level of all op nodes using a lookup table.
    """
    memo = {}
    for node in topological_nodes:
        predecessors = list(filter(lambda node: isinstance(node, DAGOpNode), dag.predecessors(node)))
        if not predecessors:
            level = 0
        else:
            level = 1 + max(map(lambda p: memo[p._node_id], predecessors))
        memo[node._node_id] = level

    if sort_by_level:
        # Sort by level. Note that H gates also have their levels.
        topological_nodes.sort(key=lambda nd: memo[nd._node_id])
    return memo


def dict_product(input_dict):
    keys = input_dict.keys()
    value_lists = input_dict.values()

    # 使用itertools.product生成所有值的组合
    value_combinations = itertools.product(*value_lists)

    # 将每个值的组合与键配对，生成字典列表
    result = [dict(zip(keys, combo)) for combo in value_combinations]

    return result
