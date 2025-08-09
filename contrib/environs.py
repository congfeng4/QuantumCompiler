from typing import Any, SupportsFloat
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from gymnasium.utils.env_checker import check_env

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGNode

from contrib.common import qknob_metrics, readable_float_dict
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy, ha_baseline
from contrib.expert import TrajectoryCollector, rollout_expert_trajectory, heuristic_algorithm
from contrib.seed import set_all_seeds
from contrib.common import get_cnot_num
from contrib.action_space import ActionSpaceEdge
from contrib.state_space import StateSpace

from hamap.distance_matrix import get_distance_matrix_swap_number_and_error, get_distance_matrix_swap_number
from hamap.gates import SwapTwoQubitGate, BridgeTwoQubitGate, TwoQubitGate
from hamap.heuristics import sabre_heuristic
from hamap.layer import QuantumLayer, update_layer
from hamap.mapping import _adapt_quantum_circuit_and_mapping_arity, _create_empty_dagcircuit_from_existing
from hamap import IBMQHardwareArchitecture, mapping_to_str

import logging

from hamap.swap import get_all_swap_bridge_candidates

logger = logging.getLogger("contrib.env")


class BaseCircuitEnv(gym.Env):
    """
    An env that lets the model determine the gate state (Executable or not).
    """

    def __init__(self, hardware: IBMQHardwareArchitecture, L: int = 10):
        super().__init__()
        self.N = N = hardware.qubit_number
        self.L = L

        self.action = ActionSpaceEdge(hardware)
        self.state = StateSpace(N, L)

        self.action_space = self.action.get_space()
        self.observation_space = self.state.get_space()


class RewardMode(Enum):
    HEURISTIC_COST = 0
    GATE_NUM_COST = 1
    GATE_NUM_AND_HEURISTIC_COST = 2


class BaselineMode(Enum):
    NONE = 0
    SUBTRACT_MIN = 1
    SUBTRACT_AVG =2
    DIVIDE_MIN = 3
    DIVIDE_AVG =4
    MINMAX = 5


class CircuitEnvWithInitialMapping(BaseCircuitEnv):

    def __init__(self,
                 input_circuit: QuantumCircuit,
                 circuit_path: str,
                 hardware: IBMQHardwareArchitecture,
                 initial_mapping: dict[Qubit, int],
                 L: int,
                 reward_mode: RewardMode = RewardMode.HEURISTIC_COST,
                 look_ahead_depth: int = 16,
                 look_ahead_weight: float = 0.5,
                 baseline_mode: BaselineMode = BaselineMode.SUBTRACT_MIN):
        super().__init__(hardware, L=L)
        self.input_circuit = input_circuit
        self.circuit_path = circuit_path
        self.hardware = hardware
        self.initial_mapping = initial_mapping
        self.distance_matrix = get_distance_matrix_swap_number(self.hardware)
        self.reward_mode = reward_mode
        self.look_ahead_depth = look_ahead_depth
        self.look_ahead_weight = look_ahead_weight
        self.baseline_mode = baseline_mode

        _adapt_quantum_circuit_and_mapping_arity(self.input_circuit, initial_mapping, hardware)
        self.dag_circuit = circuit_to_dag(input_circuit)
        self.topological_nodes: list[DAGNode] = list(self.dag_circuit.topological_op_nodes())

        self.resulting_circuit = None
        self.metrics_baseline = ha_baseline(input_circuit, hardware, initial_mapping)

        # check_env(self)

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.invalid_actions = 0
        self.front_layer = QuantumLayer()
        self.current_node_index = 0
        self.resulting_dag_quantum_circuit = _create_empty_dagcircuit_from_existing(self.dag_circuit)
        self.current_mapping = self.initial_mapping.copy()
        self.trans_mapping = self.initial_mapping.copy()
        self.explored_mappings = set()
        self.inverse_mapping = {val: key for key, val in self.initial_mapping.items()}
        self.total_cost = 0
        self.normalized_total_cost = 0
        self.swap_candidates = None
        self.min_cost = self.max_cost = self.avg_cost = None

        self.update_front_layer()
        self.update()
        self.current_depth = self.resulting_dag_quantum_circuit.depth()
        self.current_cnot = get_cnot_num(self.resulting_dag_quantum_circuit)
        return self._get_obs(), {}

    def _get_obs(self):
        return self.state.encode(self.front_layer, self.topological_nodes[self.current_node_index:],
                                 self.current_mapping)

    def finalize_result(self):
        self.resulting_circuit = dag_to_circuit(self.resulting_dag_quantum_circuit)
        self.metrics = qknob_metrics(self.input_circuit, self.resulting_circuit)
        self.metrics.update(total_cost=self.total_cost, normalized_total_cost=self.normalized_total_cost)
        for key, value in self.metrics_baseline.items():
            self.metrics[key + '_diff'] = self.metrics[key] - value

    def apply_swap_action(self, best_swap_qubits: TwoQubitGate):
        trans_mapping = self.trans_mapping
        inverse_mapping = self.inverse_mapping
        # We now have our best SWAP/Bridge, let's perform it!
        self.current_mapping = best_swap_qubits.update_mapping(self.current_mapping)
        if isinstance(best_swap_qubits, SwapTwoQubitGate):
            control, target = self.current_mapping[best_swap_qubits.left], self.current_mapping[best_swap_qubits.right]
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
        self.explored_mappings.add(mapping_to_str(self.current_mapping))
        if not best_swap_qubits.apply(self.resulting_dag_quantum_circuit, self.front_layer, self.initial_mapping,
                                      trans_mapping):
            return False
        self.update_front_layer()
        return True

    def update_front_layer(self):
        self.current_node_index = update_layer(
            self.front_layer, self.topological_nodes, self.current_node_index
        )

    def update(self):
        num_executed_cnot = 0
        # Start of the iterative algorithm
        while not self.front_layer.is_empty():
            execute_gate_list = QuantumLayer()
            for op in self.front_layer.ops:
                if self.hardware.can_natively_execute_operation(op, self.current_mapping):
                    execute_gate_list.add_operation(op)
                    if len(op.qargs) == 2:
                        num_executed_cnot += 1
                        q1, q2 = op.qargs[0]._index, op.qargs[1]._index
                    # Delaying the remove operation because we do not want to remove from
                    # a container we are iterating on.
                    # front_layer.remove_operation(op)
            if not execute_gate_list.is_empty():
                self.front_layer.remove_operations_from_layer(execute_gate_list)
                execute_gate_list.apply_back_to_dag_circuit(
                    self.resulting_dag_quantum_circuit, self.initial_mapping, self.trans_mapping
                )
                self.explored_mappings.clear()
                self.update_front_layer()
            else:
                break
        # self.metrics = qknob_metrics(self.input_circuit, self.resulting_dag_quantum_circuit)
        return num_executed_cnot

    @property
    def num_qubits(self):
        return self.input_circuit.num_qubits

    def step_invalid(self):
        # Invalid Actions
        print('invalid')
        self.invalid_actions += 1
        if self.invalid_actions >= 100:
            return self._get_obs(), -0.1, False, True, {}
        return self._get_obs(), -0.1, False, False, {}

    def heuristic_cost(self, swap: TwoQubitGate):
        return sabre_heuristic(
            hardware=self.hardware, front_layer=self.front_layer, topological_nodes=self.topological_nodes,
            current_node_index=self.current_node_index, current_mapping=self.current_mapping,
            initial_mapping=self.initial_mapping, trans_mapping=self.trans_mapping,
            distance_matrix=self.distance_matrix, tentative_gate=swap, look_ahead_depth=self.look_ahead_depth,
            look_ahead_weight=self.look_ahead_weight,
        )

    def normalize_h_cost(self, h_cost: float):
        # IMPORTANT: subtract baseline from heuristic cost can stablize late-term training. No explosion!
        if self.baseline_mode == BaselineMode.SUBTRACT_MIN:
            return h_cost - self.min_cost
        if self.baseline_mode == BaselineMode.SUBTRACT_AVG:
            return h_cost - self.avg_cost
        if self.baseline_mode == BaselineMode.DIVIDE_AVG:
            return h_cost / self.avg_cost
        if self.baseline_mode == BaselineMode.DIVIDE_MIN:
            return h_cost / self.min_cost
        if self.baseline_mode == BaselineMode.MINMAX:
            base = self.max_cost - self.min_cost
            return 0 if base == 0 else (h_cost - self.min_cost) / base
        return h_cost

    def step(
            self, policy: int
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        inverse_current_mapping = {val: key for key, val in self.current_mapping.items()}
        best_swap_qubits = self.action.decode(policy, self.initial_mapping,
                                              inverse_current_mapping,
                                              self.inverse_mapping, self.hardware)
        raw_h_cost = self.heuristic_cost(best_swap_qubits)
        normalized_h_cost = self.normalize_h_cost(raw_h_cost)
        self.total_cost += raw_h_cost
        self.normalized_total_cost += normalized_h_cost
        h_cost = raw_h_cost if self.baseline_mode == BaselineMode.NONE else normalized_h_cost

        if not self.apply_swap_action(best_swap_qubits):
            return self.step_invalid()

        self.invalid_actions = 0
        num_exe = self.update()
        gate_num_cost = 0.01

        # IMPORTANT: two terms have different effects:
        # -cost can converge model quickly on startup but rebound later.
        # num_exe - 3 converge slowly but will not rebound.
        # TODO: an annealing scheme needed
        if self.reward_mode == RewardMode.GATE_NUM_AND_HEURISTIC_COST:
            reward = -h_cost - gate_num_cost
        elif self.reward_mode == RewardMode.GATE_NUM_COST:
            reward = -gate_num_cost
        elif self.reward_mode == RewardMode.HEURISTIC_COST:
            reward = -h_cost
        else:
            raise ValueError(self.reward_mode)

        done = not self.front_layer
        info = {}
        if done:
            self.finalize_result()
            metrics = self.metrics
            reward += -metrics['cx_ratio'] - metrics['depth_ratio']
            info['metrics'] = self.metrics
            readable_metrics = readable_float_dict(self.metrics)
            print(f'Game ends {readable_metrics}')
        return self._get_obs(), reward, done, False, info

    def get_candidates(self):
        return get_all_swap_bridge_candidates(self.front_layer, self.hardware, self.initial_mapping,
                                              self.current_mapping,
                                              self.trans_mapping, self.explored_mappings)

    def action_masks(self):
        masks = np.zeros(self.action.get_size(), dtype=bool)
        costs = set()
        self.swap_masks(masks, costs)
        self.bridge_masks(masks, costs)
        self.min_cost = min(costs)
        self.max_cost = max(costs)
        self.avg_cost = sum(costs) / len(costs)
        return masks.tolist()

    def bridge_masks(self, masks, costs: set[float]):
        trans_mapping = self.trans_mapping
        initial_mapping = self.initial_mapping

        inverse_trans_mapping = {val: key for key, val in trans_mapping.items()}
        inverse_mapping = {val: key for key, val in initial_mapping.items()}
        for op in self.front_layer.ops:
            if len(op.qargs) < 2:
                # We just pass 1 qubit gates because they do not participate in the
                # Bridge operation
                continue
            if len(op.qargs) != 2:
                logger.warning("A 3-qubit or more gate has been found in the circuit.")
                continue

            control, target = op.qargs
            control_index = initial_mapping[inverse_trans_mapping[initial_mapping[control]]]
            target_index = initial_mapping[inverse_trans_mapping[initial_mapping[target]]]
            # For each qubit q linked with control, check if target is linked with q.
            for _, potential_middle_index in self.hardware.out_edges(control_index):
                for _, potential_target_index in self.hardware.out_edges(potential_middle_index):
                    if potential_target_index == target_index:
                        two_qubit_gate = BridgeTwoQubitGate(
                            inverse_trans_mapping[initial_mapping[control]],
                            inverse_mapping[potential_middle_index],
                            inverse_trans_mapping[initial_mapping[target]],
                        )
                        costs.add(self.heuristic_cost(two_qubit_gate))
                        # Not using assert! bridge has deplicates.
                        # assert not masks[control_index, target_index], (control_index, target_index, masks[control_index, target_index])
                        masks[self.action.action_to_index[control_index, target_index]] = True

    def swap_masks(self, masks, costs: set[float]):
        # First compute all the qubits involved in the given layer
        qubits_involved_in_front_layer = set()
        for op in self.front_layer.ops:
            qubits_involved_in_front_layer.update(op.qargs)
        inverse_mapping = {val: key for key, val in self.current_mapping.items()}
        # Then for all the possible links that involve at least one of the qubits used by
        # the gates in the given layer, add this link as a possible SWAP.
        # all_swaps = list()
        for involved_qubit in qubits_involved_in_front_layer:
            qubit_index = self.current_mapping[involved_qubit]
            # For all the links that involve the current qubit.
            for source, sink in self.hardware.out_edges(qubit_index):
                masks[self.action.action_to_index[source, sink]] = True
                two_qubit_gate = SwapTwoQubitGate(
                    inverse_mapping[source], inverse_mapping[sink]
                )
                costs.add(self.heuristic_cost(two_qubit_gate))



gym.register("CircuitEnv", "contrib.environs:CircuitEnvWithInitialMapping")

if __name__ == '__main__':
    set_all_seeds()
    circuit_path = '../data/53Q_depth_Sycamore/circuits/53Q_depth_Sycamore_small_None_1_1.5_no.0.qasm'
    hardware_name = 'sycamore'
    circuit = QuantumCircuit.from_qasm_file(str(circuit_path))
    hardware = IBMQHardwareArchitecture(hardware_name)
    collector = TrajectoryCollector(hardware=hardware, L=10)

    init = get_initial_mapping(circuit, hardware, InitialMappingStrategy.IDENTITY)
    heuristic_algorithm(collector, circuit, init, hardware)
    traj = collector.trajectories[0]
    metrics = collector.metrics_list[0]
    env = CircuitEnvWithInitialMapping(circuit, str(circuit_path), hardware, init, L=10)
    metrics_env = rollout_expert_trajectory(env, traj)
