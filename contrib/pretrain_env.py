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

from contrib import state
from contrib.common import qubit_index_from_op, qubit_index_from_swap
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy
from contrib.state import ObservationSpace
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

NUM_ACTIONS = 3

class ActionSpace:

    def __init__(self, N: int, A: int):
        self.A = A
        self.N = N

    def get_space(self):
        A, N = self.A, self.N
        return gym.spaces.MultiBinary(A*N*N)

    def encode_execute_list(self, execute_gate_list: list[DAGNode], current_mapping: dict[Qubit, int]):
        action = self.empty_action()
        num_exe_cx = 0
        for op in execute_gate_list:
            if op.name != 'cx':
                continue
            q0, q1 = current_mapping[op.qargs[0]], current_mapping[op.qargs[1]]
            action[0, q0, q1] = 1
            num_exe_cx += 1
        return action

    def empty_action(self):
        action = np.zeros((self.A, self.N, self.N), np.float32)
        return action

    def encode_swap_cands(self, swap_cands: list[TwoQubitGate], current_mapping: dict[Qubit, int]):
        action = self.empty_action()
        for swap in swap_cands:
            if isinstance(swap, BridgeTwoQubitGate):
                action[2, swap.left._index, swap.right._index] = 1
            else:
                q0, q1 = current_mapping[swap.left], current_mapping[swap.right]
                action[1, q0, q1] = 1
        return action

    def encode_best_swap(self, swap: TwoQubitGate, current_mapping: dict[Qubit, int]):
        action = self.empty_action()
        if isinstance(swap, BridgeTwoQubitGate):  # Already physical
            action[2, swap.left._index, swap.right._index] = 1
        else:
            q0, q1 = current_mapping[swap.left], current_mapping[swap.right]
            action[1, q0, q1] = 1
        return action


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
        self.obs_space = ObservationSpace(N, L)
        self.act_space = ActionSpace(N, NUM_ACTIONS)

    def begin_trajectory(self):
        self.current_traj = defaultdict(list)

    def clean_trajectory(self, current_traj):
        """
        Remove empty gate sequences.
        """
        obs, acts = current_traj['obs'], current_traj['acts']
        assert len(obs) == len(acts) + 1, (len(obs), len(acts))

        new_obs, new_acts = [], []
        for ob, act in zip(obs, acts):
            if ob['gate_len'] > 0:
                new_obs.append(ob)
                new_acts.append(act)
        new_obs.append(obs[-1])

        assert len(new_obs) == len(new_acts) + 1, (len(new_obs), len(new_acts))
        return dict(obs=new_obs, acts=new_acts)

    def end_trajectory(self):
        current_traj = self.current_traj
        self.current_traj = None
        if not current_traj:
            return

        current_traj = self.clean_trajectory(current_traj)
        obs, acts = current_traj['obs'], current_traj['acts']
        traj = Trajectory(obs=DictObs.from_obs_list(obs), acts=np.asarray(acts), terminal=True, infos=None)
        self.trajectories.append(traj)

    def add_state(self, front_layer: QuantumLayer, gates: list[DAGNode], current_mapping: dict[Qubit, int]):
        observation = self.obs_space.encode_obs(front_layer, gates, current_mapping)
        self.current_traj['obs'].append(observation)

    def add_execute(self, execute_gate_list: list[DAGNode], current_mapping: dict[Qubit, int]):
        action = self.act_space.encode_execute_list(execute_gate_list, current_mapping)
        self.current_traj['acts'].append(action.reshape(-1))

    def add_swap_cands(self, swap_cands: list[TwoQubitGate], current_mapping: dict[Qubit, int]):
        action = self.act_space.encode_swap_cands(swap_cands, current_mapping)
        self.current_traj['acts'].append(action.reshape(-1))

    def add_best_swap(self, swap: TwoQubitGate, current_mapping: dict[Qubit, int]):
        action = self.act_space.encode_best_swap(swap, current_mapping)
        self.current_traj['acts'].append(action.reshape(-1))

    def save(self):
        transitions = flatten_trajectories(self.trajectories)
        save_file = self.outdir / f'{self.prefix}.trans'
        with save_file.open('wb') as f:
            pickle.dump(transitions, f)
        print(f'Save {len(self.trajectories)} Trajs ({len(transitions)} Trans) to {save_file}')

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
        self.action_space = ActionSpace(N, NUM_ACTIONS).get_space()
        self.observation_space = state.ObservationSpace(N, L).get_space()


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
    strategy: str = 'random', # 'random' or 'best'
) -> ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]]:
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
        # Add state
        collector.add_state(front_layer, topological_nodes[current_node_index:], current_mapping)
        execute_gate_list = QuantumLayer()
        for op in front_layer.ops:
            if hardware.can_natively_execute_operation(op, current_mapping):
                execute_gate_list.add_operation(op)
                # Delaying the remove operation because we do not want to remove from
                # a container we are iterating on.
                # front_layer.remove_operation(op)
        if not execute_gate_list.is_empty():
            # Add action
            collector.add_execute(execute_gate_list.ops, current_mapping)
            # collector.add_swap_cands([])
            front_layer.remove_operations_from_layer(execute_gate_list)
            execute_gate_list.apply_back_to_dag_circuit(
                resulting_dag_quantum_circuit, initial_mapping, trans_mapping
            )
            # Empty the explored mappings because at least one gate has been executed.
            explored_mappings.clear()
        else:
            # collector.add_state(front_layer, topological_nodes[current_node_index:], current_mapping)
            inverse_mapping = {val: key for key, val in initial_mapping.items()}
            # We cannot execute any gate, that means that we should insert at least
            # one SWAP/Bridge to make some gates executable.
            # First list all the SWAPs/Bridges that may help us make some gates
            # executable.
            swap_candidates = get_candidates(
                front_layer, hardware, initial_mapping, current_mapping, trans_mapping, explored_mappings
            )
            # Add action
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

            # collector.add_swap_cands(swap_candidates, current_mapping)
            collector.add_best_swap(best_swap_qubits, current_mapping)
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
        # Anyway, update the current front_layer
        current_node_index = update_layer(
            front_layer, topological_nodes, current_node_index
        )
    # Add state
    collector.add_state(front_layer, topological_nodes[current_node_index:], current_mapping)
    # We are done here, we just need to return the results
    # resulting_dag_quantum_circuit.draw(scale=1, filename="qcirc.dot")
    resulting_circuit = dag_to_circuit(resulting_dag_quantum_circuit)

    collector.end_trajectory()  # Finish one trajectory.
    print(f'maxlen of frontlayer {max(front_layer_len)}')

    return resulting_circuit, current_mapping


if __name__ == '__main__':
    hardware = IBMQHardwareArchitecture('tokyo')
    collector = TrajectoryCollector(N=hardware.qubit_number, L=10, outdir=Path('../result/pretrain/exe_swap'),
                                    prefix='20Q_gate_Tokyo')
    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits').glob('*.qasm'))

    for i in range(1):
        qc = QuantumCircuit.from_qasm_file(str(circuit_list[i]))
        init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)
        ha_mapping(
            collector=collector,
            quantum_circuit=qc,
            initial_mapping=init,
            hardware=hardware,
            strategy='best',
        )

    collector.save()
