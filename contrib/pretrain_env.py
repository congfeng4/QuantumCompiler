import pickle
import random
from collections import defaultdict

import gymnasium as gym
import numpy as np

import typing as ty

import numpy
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import Trajectory
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.quantumregister import Qubit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit, DAGNode

from contrib.common import show_mapping
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy
from hamap.distance_matrix import (
    get_distance_matrix_mixed,
    get_distance_matrix_swap_number,
    get_distance_matrix_swap_number_and_error,
)
from hamap.gates import TwoQubitGate, SwapTwoQubitGate, BridgeTwoQubitGate
from hamap.hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from hamap.heuristics import sabre_heuristic, sabre_heuristic_with_effect
from hamap.layer import QuantumLayer, update_layer
from hamap.mapping import _adapt_quantum_circuit_and_mapping_arity, _create_empty_dagcircuit_from_existing
from hamap.mapping_to_str import mapping_to_str
from hamap.swap import get_all_swap_bridge_candidates, get_all_swap_candidates
import logging
from imitation.data import serialize, rollout
from imitation.data.types import DictObs

from pathlib import Path


logger = logging.getLogger("hamap.swap")


def qubit_index_from_op(op: DAGNode):
    return op.qargs[0]._index, op.qargs[1]._index


def qubit_index_from_swap(swap: TwoQubitGate):
    return swap.left._index, swap.right._index


class TrajectoryCollector:

    def __init__(self, N: int, L: int, outdir: Path, prefix: str):
        """
        N (int): number of qubits
        L (int): max len of gate seq.
        """
        self.N = N
        self.L = L
        self.trajectories = []
        self.current_traj = None
        self.outdir = outdir
        self.prefix = prefix

    def begin_trajectory(self):
        self.current_traj = defaultdict(list)

    def end_trajectory(self):
        current_traj = self.current_traj
        self.current_traj = None
        if not current_traj:
            return

        obs, acts = current_traj['obs'][:-1], current_traj['acts'][:-1]
        assert len(obs) == len(acts) + 1, (len(obs), len(acts))
        idx = 0
        for ob, act in zip(obs, acts):
            if ob['gate_len'] == 0:
                print(idx, len(acts))
            idx += 1

        traj = Trajectory(obs=DictObs.from_obs_list(obs), acts=np.asarray(acts), terminal=True, infos=None)
        self.trajectories.append(traj)

    def add_state(self, gates: list[DAGNode], current_mapping: dict[Qubit, int]):
        mapping = np.zeros((self.N,), np.int64)
        for qb, j in current_mapping.items():
            mapping[qb._index] = j

        gate_seq = np.zeros((self.L, 2), np.int64)
        index = 0
        for op in gates:
            if index >= self.L:
                break
            if op.name != 'cx':
                continue
            gate_seq[index] = qubit_index_from_op(op)
            index += 1

        self.current_traj['obs'].append({
            'mapping': mapping,
            'gate_seq': gate_seq,
            'gate_len': index,
        })

    def add_execute(self, execute_gate_list: list[DAGNode]):
        action = np.zeros((1, self.N, self.N), np.float32)
        num_exe_cx = 0
        for op in execute_gate_list:
            if op.name != 'cx':
                continue
            q0, q1 = qubit_index_from_op(op)
            action[0, q0, q1] = True
            num_exe_cx += 1

        self.current_traj['acts'].append(action.reshape(-1))

    def add_swap_cands(self, swap_cands: list[TwoQubitGate]):
        action = np.zeros((3, self.N, self.N), np.float32)
        for swap in swap_cands:
            pass

    def save(self):
        transitions = flatten_trajectories(self.trajectories)
        save_file = self.outdir / f'{self.prefix}.trans'
        with save_file.open('wb') as f:
            pickle.dump(transitions, f)

    def __repr__(self):
        total = len(self.trajectories)
        minlen = min(len(traj) for traj in self.trajectories)
        maxlen = max(len(traj) for traj in self.trajectories)
        avglen = round(sum(len(traj) for traj in self.trajectories) / total, 2)
        return f'Collected {total}, prefix {self.prefix}, {minlen=}, {maxlen=}, {avglen=}'


class PretrainEnv(gym.Env):
    """
    An env that lets the model determine the gate state (Executable or not).
    """

    def __init__(self, N: int, L: int = 100):
        super().__init__()

        self.action_space = gym.spaces.MultiBinary(N*N)

        self.observation_space = gym.spaces.Dict({
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
            )
        })


def ha_mapping(
    collector: TrajectoryCollector,
    quantum_circuit: QuantumCircuit,
    initial_mapping: ty.Dict[Qubit, int],
    hardware: IBMQHardwareArchitecture,
    swap_cost_heuristic: ty.Callable[
        [
            IBMQHardwareArchitecture,  # Hardware information
            QuantumLayer,  # Current front layer
            ty.List[DAGNode],  # Topologically sorted list of nodes
            int,  # Index of the first non-processed gate.
            ty.Dict[Qubit, int],  # The mapping before applying the tested SWAP/Bridge
            ty.Dict[Qubit, int],  # The initial mapping
            ty.Dict[Qubit, int],  # The trans mapping
            numpy.ndarray,  # The distance matrix between each qubits
            TwoQubitGate,  # The SWAP/Bridge we want to rank
        ],
        float,
    ] = sabre_heuristic,
    get_candidates: ty.Callable[
        [QuantumLayer, IBMQHardwareArchitecture, ty.Dict[Qubit, int], ty.Dict[Qubit, int], ty.Dict[Qubit, int], ty.Set[str],],
        ty.List[TwoQubitGate],
    ] = get_all_swap_bridge_candidates,
    get_distance_matrix: ty.Callable[
        [IBMQHardwareArchitecture], numpy.ndarray
    ] = get_distance_matrix_swap_number_and_error,
    strategy: str = 'random', # 'random' or 'mincost'
) -> ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]]:
    """Map the given quantum circuit to the hardware topology provided.

    :param quantum_circuit: the quantum circuit to map.
    :param initial_mapping: the initial mapping used to start the iterative mapping
        algorithm.
    :param hardware: hardware data such as connectivity, gate time, gate errors, ...
    :param swap_cost_heuristic: the heuristic cost function that will estimate the cost
        of a given SWAP/Bridge according to the current state of the circuit.
    :param get_candidates: a function that takes as input the current front
        layer and the hardware description and that returns a list of tuples
        representing the SWAP/Bridge that should be considered by the heuristic.
    :param get_distance_matrix: a function that takes as first (and only) parameter the
        hardware representation and that outputs a numpy array containing the cost of
        performing a SWAP between each pair of qubits.
    :return: The final circuit along with the mapping obtained at the end of the
        iterative procedure.
    """
    collector.begin_trajectory()

    _adapt_quantum_circuit_and_mapping_arity(quantum_circuit, initial_mapping, hardware)
    # Creating the internal data structures that will be used in this function.
    dag_circuit = circuit_to_dag(quantum_circuit)
    distance_matrix = get_distance_matrix(hardware)
    resulting_dag_quantum_circuit = _create_empty_dagcircuit_from_existing(dag_circuit)
    current_mapping = initial_mapping
    explored_mappings: ty.Set[str] = set()
    # Sorting all the quantum operations in topological order once for all.
    # May require significant memory on large circuits...
    topological_nodes: ty.List[DAGNode] = list(dag_circuit.topological_op_nodes())
    current_node_index = 0
    # Add state
    collector.add_state(topological_nodes[current_node_index:], current_mapping)
    # Creating the initial front layer.
    front_layer = QuantumLayer()
    current_node_index = update_layer(
        front_layer, topological_nodes, current_node_index
    )
    trans_mapping = initial_mapping.copy()
    front_layer_len = []

    # Start of the iterative algorithm
    while not front_layer.is_empty():
        front_layer_len.append(sum(1 for op in front_layer.ops if op.name == 'cx'))
        execute_gate_list = QuantumLayer()
        for op in front_layer.ops:
            if hardware.can_natively_execute_operation(op, current_mapping):
                execute_gate_list.add_operation(op)
                # Delaying the remove operation because we do not want to remove from
                # a container we are iterating on.
                # front_layer.remove_operation(op)
        # Add action
        collector.add_execute(execute_gate_list.ops)
        if not execute_gate_list.is_empty():
            front_layer.remove_operations_from_layer(execute_gate_list)
            execute_gate_list.apply_back_to_dag_circuit(
                resulting_dag_quantum_circuit, initial_mapping, trans_mapping
            )
            # Empty the explored mappings because at least one gate has been executed.
            explored_mappings.clear()
        else:
            inverse_mapping = {val: key for key, val in initial_mapping.items()}
            # We cannot execute any gate, that means that we should insert at least
            # one SWAP/Bridge to make some gates executable.
            # First list all the SWAPs/Bridges that may help us make some gates
            # executable.
            swap_candidates = get_candidates(
                front_layer, hardware, initial_mapping, current_mapping, trans_mapping, explored_mappings
            )
            # collector.add_swap_cands(swap_candidates)
            # Then rank the SWAPs/Bridge and take the best one.
            if strategy == 'random':
                best_swap_qubits = random.choice(swap_candidates)
            else:
                best_cost = float("inf")
                best_swap_qubits = None
                for potential_swap in swap_candidates:
                    cost = swap_cost_heuristic(
                        hardware,
                        front_layer,
                        topological_nodes,
                        current_node_index,
                        current_mapping,
                        initial_mapping,
                        trans_mapping,
                        distance_matrix,
                        potential_swap,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_swap_qubits = potential_swap

            # We now have our best SWAP/Bridge, let's perform it!
            current_mapping = best_swap_qubits.update_mapping(current_mapping)
            if isinstance(best_swap_qubits, SwapTwoQubitGate):
                control, target = current_mapping[best_swap_qubits.left], current_mapping[best_swap_qubits.right]
                swap_control, swap_target = inverse_mapping[control], inverse_mapping[target]
                best_swap_qubits = SwapTwoQubitGate(
                    swap_control, swap_target
                )
                #print("swap gates is :", best_swap_qubits.left, best_swap_qubits.right)
                trans_mapping[best_swap_qubits.left], trans_mapping[best_swap_qubits.right] = (
                    trans_mapping[best_swap_qubits.right],
                    trans_mapping[best_swap_qubits.left],
                )
            else:
                #print("brige gate is :", best_swap_qubits.left, best_swap_qubits.middle, best_swap_qubits.right)
                pass
            explored_mappings.add(mapping_to_str(current_mapping))
            best_swap_qubits.apply(resulting_dag_quantum_circuit, front_layer, initial_mapping, trans_mapping)
        # Add state
        collector.add_state(topological_nodes[current_node_index:], current_mapping)
        # Anyway, update the current front_layer
        current_node_index = update_layer(
            front_layer, topological_nodes, current_node_index
        )

    # We are done here, we just need to return the results
    # resulting_dag_quantum_circuit.draw(scale=1, filename="qcirc.dot")
    resulting_circuit = dag_to_circuit(resulting_dag_quantum_circuit)

    collector.end_trajectory()  # Finish one trajectory.
    print(f'maxlen of frontlayer {max(front_layer_len)}')

    return resulting_circuit, current_mapping


if __name__ == '__main__':
    hardware = IBMQHardwareArchitecture('tokyo')
    collector = TrajectoryCollector(N=hardware.qubit_number, L=10, outdir=Path('../result/pretrain/exe'),
                                    prefix='20Q_gate_Tokyo')
    qc = QuantumCircuit.from_qasm_file('../data/20Q_gate_Tokyo/circuits/20Q_gate_Tokyo_large_1_10_1.5_no.1.qasm')
    print(f'#Gates {qc.count_ops()["cx"]}, Depth {qc.depth()}')

    for i in range(10):
        init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)
        for j in range(10):
            ha_mapping(
                collector=collector,
                quantum_circuit=qc,
                initial_mapping=init,
                hardware=hardware,
                strategy='random',
            )

    collector.save()
