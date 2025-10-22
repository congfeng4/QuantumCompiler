"""
State space encoding.
"""

import gymnasium as gym
import numpy as np
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGOpNode, DAGCircuit

from contrib.common import qknob_metrics, readable_float_dict, get_circuit_cost, TopologicalOrderMode, \
    build_op_node_level

from hamap.layer import QuantumLayer, update_layer

from hamap.layer import QuantumLayer


class StateSpace:

    def __init__(self, N: int, L: int):
        self.N = N
        self.L = L

    def get_space(self):
        N, L = self.N, self.L

        return gym.spaces.Dict({
            # 逻辑 → 物理映射 (permutation)
            "mapping": gym.spaces.Box(
                low=0, high=N - 1, shape=(N,), dtype=np.int64
            ),

            # 门序列：每行是 (q0, q1) 两个逻辑比特编号
            "gate_seq": gym.spaces.Box(
                low=0, high=N - 1, shape=(L, 2), dtype=np.int64
            ),

            # 真实门序列长度
            "gate_len": gym.spaces.Box(
                low=0, high=L, shape=(), dtype=np.int64
            ),

            # 门的层级（整数）
            "gate_level": gym.spaces.Box(
                low=0, high=L, shape=(L, 1), dtype=np.int64
            ),
        })

    def encode(self, dag: DAGCircuit, current_mapping: dict[Qubit, int]):
        mapping = np.zeros((self.N,), np.int64)
        for qb, j in current_mapping.items():
            mapping[j] = qb._index  # Phy to logic

        topological_nodes: list[DAGOpNode] = list(dag.topological_op_nodes())
        # Resort according to node levels.
        gate_levels = build_op_node_level(dag, topological_nodes, sort_by_level=True)

        front_layer = QuantumLayer()  # Obtain the front layer, ensuring they are in the sequence front.
        current_node_index = update_layer(
            front_layer, topological_nodes, 0,
        )

        gate_seq = np.zeros((self.L, 2), np.int64)
        gate_level = np.zeros((self.L, 1), np.int64)
        ops: list[DAGOpNode] = front_layer.ops + topological_nodes[current_node_index:]
        gate_len = min(len(ops), self.L)

        for i, op in zip(range(gate_len), ops):
            if op.name == 'cx':
                gate_seq[i] = current_mapping[op.qargs[0]], current_mapping[op.qargs[1]]
            elif op.name == 'h':
                gate_seq[i] = current_mapping[op.qargs[0]], current_mapping[op.qargs[0]]

            gate_level[i] = gate_levels[op._node_id]

        if gate_len > 0:
            gate_level[:gate_len] -= gate_level[0]
            assert np.all(gate_level >= 0), gate_level

        return dict(mapping=mapping, gate_seq=gate_seq, gate_len=gate_len, gate_level=gate_level)
