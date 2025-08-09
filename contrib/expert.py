"""
Collect expert trajectories for validation of our environment.
"""
import pickle
import random
from collections import defaultdict, Counter

import gymnasium as gym
import numpy as np

import typing as ty

import numpy
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import Trajectory, Transitions
from qiskit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGNode

from contrib.common import qknob_metrics, get_distance_matrix
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy
from contrib.state_space import StateSpace
from contrib.action_space import ActionSpace, ActionSpaceEdge
from contrib.reward_space import get_circuit_cost

from hamap.distance_matrix import (
    get_distance_matrix_swap_number_and_error,
)
from hamap.gates import TwoQubitGate, SwapTwoQubitGate, BridgeTwoQubitGate
from hamap.hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from hamap.heuristics import sabre_heuristic
from hamap.layer import QuantumLayer, update_layer
from hamap.mapping import _adapt_quantum_circuit_and_mapping_arity, _create_empty_dagcircuit_from_existing
from hamap.mapping_to_str import mapping_to_str
from hamap.swap import get_all_swap_bridge_candidates
import logging
from imitation.data.types import DictObs

from pathlib import Path

logger = logging.getLogger("hamap.swap")



class TrajectoryCollector:

    def __init__(self, hardware: IBMQHardwareArchitecture, L: int, outdir: Path = None, prefix: str = None):
        """
        N (int): number of qubits
        L (int): max len of gate seq.
        K (int): max number of candidate swap/bridge.
        """
        self.N = N = hardware.qubit_number
        self.L = L
        self.trajectories = []
        self.metrics_list = []
        self.current_traj = None
        self.outdir = outdir
        self.prefix = prefix
        self.obs_space = StateSpace(N, L)
        self.act_space = ActionSpaceEdge(hardware)
        self.action_count = Counter()

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

    def end_trajectory(self, metrics: dict[dict, float] = None):
        current_traj = self.current_traj
        self.current_traj = None
        if not current_traj:
            return

        current_traj = self.clean_trajectory(current_traj)
        obs, acts = current_traj['obs'], current_traj['acts']
        traj = Trajectory(obs=DictObs.from_obs_list(obs), acts=np.asarray(acts), terminal=True, infos=None)
        self.trajectories.append(traj)
        self.metrics_list.append(metrics)
        print(f'End trajectory. len {len(traj)}, metrics {metrics}')

    def add_state(self, front_layer: QuantumLayer, gates: list[DAGNode], current_mapping: dict[Qubit, int]):
        observation = self.obs_space.encode(front_layer, gates, current_mapping)
        self.current_traj['obs'].append(observation)

    def add_action(self, swap: TwoQubitGate, current_mapping: dict[Qubit, int], initial_mapping: dict[Qubit, int]):
        action = self.act_space.encode(swap, current_mapping, initial_mapping)
        self.current_traj['acts'].append(action)
        act_key = 'swap' if isinstance(swap, SwapTwoQubitGate) else 'bridge'
        self.action_count[act_key] += 1

    def add_reward(self, rew: float):
        print(f'Reward {rew}')
        self.current_traj['rews'].append(rew)

    def save(self):
        self.outdir.mkdir(parents=True, exist_ok=True)

        transitions = flatten_trajectories(self.trajectories)
        save_file = self.outdir / f'{self.prefix}.trans'
        with save_file.open('wb') as f:
            pickle.dump(transitions, f)

        total = self.action_count.total()
        for key, count in self.action_count.items():
            ratio = round(count * 100 / total, 2)
            print(f'Action {key}: {ratio} %')

        print(f'Save {len(self.trajectories)} Trajs ({len(transitions)} Trans) to {save_file}')

    def load(self) -> Transitions:
        save_file = self.outdir / f'{self.prefix}.trans'
        with save_file.open('rb') as f:
            trans = pickle.load(f)

        print(f'Load {len(trans)} Trans from {save_file}')
        return trans

    def __repr__(self):
        total = len(self.trajectories)
        minlen = min(len(traj) for traj in self.trajectories)
        maxlen = max(len(traj) for traj in self.trajectories)
        avglen = round(sum(len(traj) for traj in self.trajectories) / total, 2)
        return f'Collected {total}, prefix {self.prefix}, {minlen=}, {maxlen=}, {avglen=}'


def heuristic_algorithm(
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
            [QuantumLayer, IBMQHardwareArchitecture, ty.Dict[Qubit, int], ty.Dict[Qubit, int], ty.Dict[Qubit, int],
             ty.Set[str], ],
            ty.List[TwoQubitGate],
        ] = get_all_swap_bridge_candidates,
        get_distance_matrix: ty.Callable[
            [IBMQHardwareArchitecture], numpy.ndarray
        ] = get_distance_matrix_swap_number_and_error,
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
        execute_gate_list = QuantumLayer()
        for op in front_layer.ops:
            if hardware.can_natively_execute_operation(op, current_mapping):
                execute_gate_list.add_operation(op)
                # Delaying the remove operation because we do not want to remove from
                # a container we are iterating on.
                # front_layer.remove_operation(op)
        if not execute_gate_list.is_empty():
            # Add action
            # collector.add_execute(execute_gate_list.ops, current_mapping)
            # collector.add_swap_cands([])
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
            collector.add_state(front_layer, topological_nodes[current_node_index:], current_mapping)
            # Then rank the SWAPs/Bridge and take the best one.
            best_cost = float("inf")
            best_swap_qubits = None
            candidates_with_cost = []
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
                candidates_with_cost.append((potential_swap, cost))
            # Add action
            collector.add_action(best_swap_qubits, current_mapping, initial_mapping)
            collector.add_reward(get_circuit_cost(front_layer, topological_nodes[current_node_index:], current_mapping,
                                                  distance_matrix, hardware))

            # We now have our best SWAP/Bridge, let's perform it!
            current_mapping = best_swap_qubits.update_mapping(current_mapping)
            if isinstance(best_swap_qubits, SwapTwoQubitGate):
                control, target = current_mapping[best_swap_qubits.left], current_mapping[best_swap_qubits.right]
                swap_control, swap_target = inverse_mapping[control], inverse_mapping[target]
                best_swap_qubits = SwapTwoQubitGate(
                    swap_control, swap_target
                )
                # print("swap gates is :", best_swap_qubits.left, best_swap_qubits.right)
                trans_mapping[best_swap_qubits.left], trans_mapping[best_swap_qubits.right] = (
                    trans_mapping[best_swap_qubits.right],
                    trans_mapping[best_swap_qubits.left],
                )
            else:
                # print("brige gate is :", best_swap_qubits.left, best_swap_qubits.middle, best_swap_qubits.right)
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

    metrics = qknob_metrics(quantum_circuit, resulting_circuit)
    collector.end_trajectory(metrics)  # Finish one trajectory.
    return resulting_circuit, current_mapping


def rollout_expert_trajectory(env: gym.Env, trajectory: Trajectory):
    """
    Rollout the expert's trajectory on an enviroment.
    """
    env.reset()
    done = False
    traj_index = 0
    info = {}
    for action in trajectory.acts:
        traj_index += 1
        state, reward, done, _, info = env.step(action)
        if done:
            break
    assert traj_index == len(trajectory) and done, f'{done=}, {traj_index=}, {len(trajectory)=}'
    return info['metrics']


if __name__ == '__main__':
    hardware = IBMQHardwareArchitecture('tokyo')

    collector = TrajectoryCollector(hardware, L=10,
                                          outdir=Path('../result/pretrain/ha'),
                                          prefix='20Q_gate_Tokyo')

    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits').glob('*.qasm'))
    random.shuffle(circuit_list)

    qc = QuantumCircuit.from_qasm_file(str(circuit_list[0]))
    init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)
    heuristic_algorithm(
        collector=collector,
        quantum_circuit=qc,
        initial_mapping=init,
        hardware=hardware,
        # get_distance_matrix=get_distance_matrix,
    )
