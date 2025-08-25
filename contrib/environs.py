from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.utils.env_checker import check_env

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGNode

from contrib.common import qknob_metrics, readable_float_dict, get_circuit_cost
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy, ha_baseline
from contrib.expert import TrajectoryCollector, rollout_expert_trajectory, heuristic_algorithm, DummyTrajectoryCollector
from contrib.seed import set_all_seeds
from contrib.common import get_cnot_num, get_distance_matrix
from contrib.action_space import ActionSpaceEdge, ActionSpaceSelectQubitToSwap
from contrib.state_space import StateSpace

from hamap.gates import SwapTwoQubitGate, BridgeTwoQubitGate, TwoQubitGate
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

    def __init__(self, hardware: IBMQHardwareArchitecture , action, L: int = 10, **kwargs):
        super().__init__()
        self.hardware = hardware
        self.N = N = hardware.qubit_number
        self.L = L

        self.action = action # ActionSpaceEdge(hardware)
        self.state = StateSpace(N, L)

        self.action_space = self.action.get_space()
        self.observation_space = self.state.get_space()


class CircuitEnvWithInitialMapping(BaseCircuitEnv):

    def __init__(self,
                 input_circuit: QuantumCircuit,
                 hardware: IBMQHardwareArchitecture,
                 initial_mapping: dict[Qubit, int],
                 L: int,
                 reward_shaping_weight: float = 1,
                 gamma: float = 0.99):
        super().__init__(hardware, L=L, action=ActionSpaceEdge(hardware))
        self.input_circuit = input_circuit
        self.initial_mapping = initial_mapping
        self.distance_matrix = get_distance_matrix(self.hardware)
        self.gamma = gamma
        self.reward_shaping_weight = reward_shaping_weight

        _adapt_quantum_circuit_and_mapping_arity(self.input_circuit, initial_mapping, hardware)
        self.dag_circuit = circuit_to_dag(input_circuit)
        self.topological_nodes: list[DAGNode] = list(self.dag_circuit.topological_op_nodes())

        self.resulting_circuit = None
        self.metrics_baseline = ha_baseline(input_circuit, hardware, initial_mapping)

        # check_env(self)

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.bridge_num = 0
        self.invalid_actions = 0
        self.front_layer = QuantumLayer()
        self.current_node_index = 0
        self.resulting_dag_quantum_circuit = _create_empty_dagcircuit_from_existing(self.dag_circuit)
        self.current_mapping = self.initial_mapping.copy()
        self.trans_mapping = self.initial_mapping.copy()
        self.explored_mappings = set()
        self.inverse_mapping = {val: key for key, val in self.initial_mapping.items()}
        self.state_potential = None
        self.update_front_layer()
        self.update()
        self.update_state_potential()
        return self._get_obs(), {}

    def update_state_potential(self):
        old_potential = self.state_potential
        # Phi(s) = - cost(s)
        self.state_potential = -get_circuit_cost(self.front_layer, self.topological_nodes[self.current_node_index:],
                                                self.current_mapping, self.distance_matrix, self.hardware)
        return old_potential

    def _get_obs(self):
        return self.state.encode(self.front_layer, self.topological_nodes[self.current_node_index:],
                                 self.current_mapping)

    def finalize_result(self):
        self.resulting_circuit = dag_to_circuit(self.resulting_dag_quantum_circuit)
        self.metrics = qknob_metrics(self.input_circuit, self.resulting_circuit)
        # self.metrics['bridge_num'] = self.bridge_num
        total_actions = self.bridge_num + self.metrics['swap_num']
        self.metrics['bridge_ratio'] = self.bridge_num / total_actions
        # self.metrics['swap_ratio'] = self.metrics['swap_num'] / total_actions
        self.metrics['in_cx_num'] = get_cnot_num(self.input_circuit)
        self.metrics['out_cx_num'] = get_cnot_num(self.resulting_circuit)
        self.metrics['in_depth'] = self.input_circuit.depth()
        self.metrics['out_depth'] = self.resulting_circuit.depth()
        for key, value in self.metrics_baseline.items():
            self.metrics[key + '_diff'] = self.metrics[key] - value
            # self.metrics[key + '_HA'] = value

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
            self.bridge_num += 1
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
        return num_executed_cnot

    @property
    def num_qubits(self):
        return self.input_circuit.num_qubits

    def step_invalid(self):
        # Invalid Actions
        print('invalid')
        self.invalid_actions += 1
        if self.invalid_actions >= 100:
            return self._get_obs(), 0, False, True, {}
        return self._get_obs(), -len(self.topological_nodes), False, False, {}

    def step(
            self, policy: int
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        inverse_current_mapping = {val: key for key, val in self.current_mapping.items()}
        best_swap_qubits = self.action.decode(policy, self.initial_mapping,
                                              inverse_current_mapping,
                                              self.inverse_mapping, self.hardware)
        if not self.apply_swap_action(best_swap_qubits):
            return self.step_invalid()

        self.invalid_actions = 0
        self.update()
        prev_potential = self.update_state_potential()
        # The cost of a circuit is a potential function of the state.
        # F(s', s) = gamma * phi(s') - phi(s)
        rs = self.state_potential * self.gamma - prev_potential
        reward = self.reward_shaping_weight * rs - 1
        done = not self.front_layer
        info = {}
        if not done:
            return self._get_obs(), reward, done, False, info

        self.finalize_result()
        reward += self.metrics['in_cx_num']

        readable_metrics = readable_float_dict(self.metrics)
        print(f'Game ends {readable_metrics}')
        return self._get_obs(), reward, done, False, info

    def action_masks(self):
        masks = np.zeros(self.action.get_size(), dtype=bool)
        costs = set()
        self.swap_masks(masks, costs)
        self.bridge_masks(masks, costs)
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
            # For each qubit q linked with control, check if the target is linked with q.
            for _, potential_middle_index in self.hardware.out_edges(control_index):
                for _, potential_target_index in self.hardware.out_edges(potential_middle_index):
                    if potential_target_index == target_index:
                        two_qubit_gate = BridgeTwoQubitGate(
                            inverse_trans_mapping[initial_mapping[control]],
                            inverse_mapping[potential_middle_index],
                            inverse_trans_mapping[initial_mapping[target]],
                        )
                        # Not using assert! The bridge has duplicates.
                        # assert not masks[control_index, target_index], (control_index, target_index, masks[control_index, target_index])
                        masks[self.action.action_to_index[control_index, target_index]] = True

    def swap_masks(self, masks, costs: set[float]):
        # First, compute all the qubits involved in the given layer
        qubits_involved_in_front_layer = set()
        for op in self.front_layer.ops:
            qubits_involved_in_front_layer.update(op.qargs)
        inverse_mapping = {val: key for key, val in self.current_mapping.items()}
        # Then, for all the possible links that involve at least one of the qubits used by
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


class InitialMappingCircuitEnv(BaseCircuitEnv):

    def __init__(self,
                 input_circuit: QuantumCircuit,
                 hardware: IBMQHardwareArchitecture,
                 initial_mapping: dict[Qubit, int],
                 reward_shaping_weight: float = 1,
                 L: int = None,
                 gamma: float = 0.99):
        super().__init__(hardware, L=get_cnot_num(input_circuit), action=ActionSpaceSelectQubitToSwap(hardware))
        self.initial_mapping = initial_mapping
        self.input_circuit = input_circuit
        self.metrics_baseline = ha_baseline(input_circuit, hardware, initial_mapping)
        self.distance_matrix = get_distance_matrix(self.hardware)
        self.gamma = gamma
        self.reward_shaping_weight = reward_shaping_weight
        self.front_layer = QuantumLayer()
        _adapt_quantum_circuit_and_mapping_arity(self.input_circuit, initial_mapping, hardware)
        self.dag_circuit = circuit_to_dag(input_circuit)
        self.topological_nodes: list[DAGNode] = list(self.dag_circuit.topological_op_nodes())

    def _get_obs(self):
        return self.state.encode(self.front_layer, self.topological_nodes, self.current_mapping)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.step_index = 0
        self.current_mapping = self.initial_mapping.copy()
        self.resulting_circuit = None
        self.state_potential = None
        self.update_state_potential()
        return self._get_obs(), {}

    def finalize_result(self):
        collector = DummyTrajectoryCollector()
        self.resulting_circuit = heuristic_algorithm(collector,
                                                     quantum_circuit=self.input_circuit,
                                                     initial_mapping=self.current_mapping,
                                                     hardware=self.hardware)
        self.metrics = collector.metrics
        self.metrics['in_cx_num'] = get_cnot_num(self.input_circuit)
        self.metrics['out_cx_num'] = get_cnot_num(self.resulting_circuit)
        self.metrics['in_depth'] = self.input_circuit.depth()
        self.metrics['out_depth'] = self.resulting_circuit.depth()
        for key, value in self.metrics_baseline.items():
            self.metrics[key + '_diff'] = self.metrics[key] - value

    def update_state_potential(self):
        old_potential = self.state_potential
        # Phi(s) = - cost(s)
        self.state_potential = -get_circuit_cost(self.front_layer, self.topological_nodes,
                                                self.current_mapping, self.distance_matrix, self.hardware)
        return old_potential

    def apply_action(self, action: int):
        a, b = self.step_index, action
        # Must use swapping to preserve the properties of a mapping.
        inverse_mapping = {val: key for key, val in self.current_mapping.items()}
        a, b = inverse_mapping[a], inverse_mapping[b]
        self.current_mapping[a], self.current_mapping[b] = self.current_mapping[b], self.current_mapping[a]

    def step(
        self, action: int
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Swap step_index with action
        self.apply_action(action)
        self.step_index += 1
        prev_potential = self.update_state_potential()
        # The cost of a circuit is a potential function of the state.
        # F(s', s) = gamma * phi(s') - phi(s)
        rs = self.state_potential * self.gamma - prev_potential
        reward = self.reward_shaping_weight * rs

        if self.step_index == self.N:
            self.finalize_result()
            reward -= self.metrics['out_cx_num']
            return self._get_obs(), reward, True, False, {}

        return self._get_obs(), reward, False, False, {}



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
    env = CircuitEnvWithInitialMapping(circuit, hardware, init, L=10)
    metrics_env = rollout_expert_trajectory(env, traj)
