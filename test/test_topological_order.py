"""
拓扑序列关系到电路翻译的正确性。
"""

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.converters import circuit_to_dag

from contrib.common import get_all_qknob_circuit_paths, read_circuit
from contrib.environs import build_op_node_level

import pytest


def check_topological_order(dag: DAGCircuit, sequence: list[DAGOpNode]):
    """
    Return True iff `sequence` is a topological order of `graph`.
    """
    # Map each node to its position in the proposed order
    pos = {node: idx for idx, node in enumerate(sequence)}

    for node in sequence:
        for prev in filter(lambda nd: isinstance(nd, DAGOpNode) and nd.name == 'cx', dag.predecessors(node)):
            assert pos[prev] < pos[node], f'{prev._node_id=} {node._node_id=} {pos[prev]=} {pos[node]=}'


ALL_QKNOB_CIRCUIT_PATHS = get_all_qknob_circuit_paths()


@pytest.mark.parametrize('data', ALL_QKNOB_CIRCUIT_PATHS.keys())
def test_topological_order(data: str):
    for circuit_path in ALL_QKNOB_CIRCUIT_PATHS[data]:
        qc = read_circuit(circuit_path)
        dag = circuit_to_dag(qc)
        topological_op_nodes = list(dag.topological_op_nodes())
        level_memo = build_op_node_level(dag, topological_op_nodes, sort_by_level=True)
        # Make sure that after sorted by levels, topological_op_nodes is still topological.
        check_topological_order(dag, topological_op_nodes)
        node_levels = [level_memo[n._node_id] for n in topological_op_nodes]
        # Make sure this level sequence is sorted.
        assert sorted(node_levels) == node_levels, f'{node_levels=} is not sorted!!'
        assert len(level_memo) == len(topological_op_nodes), f'{len(level_memo)=} vs {len(topological_op_nodes)=}'
        assert all(
            op._node_id in level_memo for op in topological_op_nodes), f'{level_memo=} does not contain all op nodes!'
