import rustworkx as rx
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()

dag = circuit_to_dag(qc)

# 缓存，避免重复计算
depth_cache = {}

def node_depth(dag, node_id):
    """返回 node_id 的深度（到输入的最长路径边数）"""
    if node_id in depth_cache:
        return depth_cache[node_id]

    preds = list(rx.dag_predecessors(dag._multi_graph, node_id))
    if not preds:                       # 输入节点
        depth_cache[node_id] = 0
        return 0

    depth_cache[node_id] = 1 + max(node_depth(dag, p) for p in preds)
    return depth_cache[node_id]

# 遍历算子节点并打印深度
for node in dag.topological_op_nodes():
    depth = node_depth(dag, node._node_id)
    print(f"{node.name} on qubits {node.qargs} -> depth = {depth}")
