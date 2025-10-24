from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from gymnasium.utils.env_checker import check_env

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit.transpiler import TransformationPass

from contrib.common import qknob_metrics, readable_float_dict, get_circuit_cost, TopologicalOrderMode, get_front_layer, \
    get_total_ops
from contrib.expert import ha_baseline
from contrib.common import get_cnot_num, get_distance_matrix
from contrib.action_space import ActionSpace
from contrib.state_space import StateSpace, RoutedStatus

from hamap.gates import SwapTwoQubitGate, BridgeTwoQubitGate, TwoQubitGate
from hamap.layer import QuantumLayer, update_layer
from hamap.mapping import _adapt_quantum_circuit_and_mapping_arity, _create_empty_dagcircuit_from_existing
from hamap import IBMQHardwareArchitecture

import logging

logger = logging.getLogger("contrib.env")


class BaseCircuitEnv(gym.Env):
    """
    An env that lets the model determine the gate state (Executable or not).
    """

    def __init__(self, hardware: IBMQHardwareArchitecture, L: int = 10, **kwargs):
        super().__init__()
        self.hardware = hardware
        self.N = N = hardware.qubit_number
        self.L = L

        self.action = ActionSpace(hardware)
        self.state = StateSpace(N, L)

        self.action_space = self.action.get_space()
        self.observation_space = self.state.get_space()


class CircuitEnvWithInitialMapping(BaseCircuitEnv):

    def __init__(self,
                 input_circuit: QuantumCircuit,
                 hardware: IBMQHardwareArchitecture,
                 initial_mapping: dict[Qubit, int],
                 L: int,
                 topological_order_mode: TopologicalOrderMode,
                 reward_shaping_weight: float = 10,
                 gamma: float = 0.99):
        super().__init__(hardware, L=L)
        self.input_circuit = input_circuit
        self.initial_mapping = initial_mapping
        self.distance_matrix = get_distance_matrix(self.hardware)
        self.gamma = gamma
        self.reward_shaping_weight = reward_shaping_weight
        self.trans_after_routing_limit = 10  # After the whole circuit is routed, perform some extra transformations.
        self.step_penalty = 1 # Encourage the agent uses fewer steps.
        self.bonus_weight = 5 # Weight for the final bonus reward.
        self.invalid_action_penalty = 10 # Penalty for each invalid action.
        self.invalid_action_limit = 100 # Max number of consecutive invalid actions.

        _adapt_quantum_circuit_and_mapping_arity(self.input_circuit, initial_mapping, hardware)
        self.input_circuit = input_circuit
        self.remaining_dag: Optional[DAGCircuit] = None
        self.resulting_circuit = None
        self.metrics_baseline = ha_baseline(input_circuit, hardware, initial_mapping, topological_order_mode)
        # check_env(self)

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.bridge_num = 0
        self.invalid_actions = 0
        self.current_mapping = self.initial_mapping.copy()
        self.trans_mapping = self.initial_mapping.copy()
        self.inverse_mapping = {val: key for key, val in self.initial_mapping.items()}
        self.state_potential = None
        # self.transform_state_potential = 0 # Initially resulting dag has no ops.
        self.remaining_dag = circuit_to_dag(self.input_circuit)
        self.resulting_dag = _create_empty_dagcircuit_from_existing(self.remaining_dag)
        self.action_stats = defaultdict(int)
        self.update()
        self.update_state_potential()
        self.trans_after_routing = 0
        return self._get_obs(), {}

    def is_routing_finished(self):
        return self.remaining_dag.size() == 0

    def update_state_potential(self):
        old_potential = self.state_potential
        # Phi(s) = - cost(s)
        self.state_potential = -get_circuit_cost(self.remaining_dag, self.current_mapping,
                                                 self.distance_matrix, self.hardware)
        return old_potential

    # def update_transform_state_potential(self):
    #     old_potential = self.transform_state_potential
    #     self.transform_state_potential = -sum(self.resulting_dag.count_ops().values())
    #     return old_potential

    def _get_obs(self):
        routed_rep = self.state.encode(self.resulting_dag, RoutedStatus.ROUTED)
        unrouted_rep = self.state.encode(self.remaining_dag, RoutedStatus.UNROUNTED,
                                         current_mapping=self.current_mapping,
                                         distance_matrix=self.distance_matrix)
        return np.concatenate((routed_rep, unrouted_rep), axis=0)

    def apply_transform_action(self, action: Tuple[int, TransformationPass]) -> Tuple[DAGCircuit, DAGCircuit]:
        phase, opt_pass = action
        if self.is_routing_finished() and phase == ActionSpace.TRANS_ROUTED:
            if self.trans_after_routing == self.trans_after_routing_limit:
                raise ValueError(f'Attemp more extra transformation than allowed: {self.trans_after_routing_limit}')

        old_dag = self.resulting_dag if phase == 0 else self.remaining_dag
        try:
            new_dag: DAGCircuit = opt_pass.run(deepcopy(old_dag))
        except Exception as e:
            raise ValueError(f'Failed to run transformation {opt_pass}, error: {e}')

        action_name = f't{str(phase)}:{opt_pass.name()}'
        self.action_stats[action_name] += 1
        if self.is_routing_finished():
            self.trans_after_routing += 1

        old_total, new_total = get_total_ops(old_dag), get_total_ops(new_dag)
        old_cx, new_cx = get_cnot_num(old_dag), get_cnot_num(new_dag)
        old_other, new_other = old_total - old_cx, new_total - new_cx
        reward = 0.8 * (old_cx - new_cx) + 0.2 * (old_other - new_other)
        if phase == ActionSpace.TRANS_ROUTED:
            self.resulting_dag = new_dag
            return reward
        # Phase is UNROUTED, Some gates may become executable. But the mapping don't change.
        # So, don't evaluate the swap cost potential.
        # Our hope is that a transformation can reduce ops and make some gate executable.
        self.remaining_dag = new_dag
        num_execute_cx = self.update()
        reward += num_execute_cx
        return reward

    def apply_route_action(self, action: TwoQubitGate):
        trans_mapping = self.trans_mapping
        inverse_mapping = self.inverse_mapping
        # We now have our best SWAP/Bridge, let's perform it!
        self.current_mapping = action.update_mapping(self.current_mapping)
        if isinstance(action, SwapTwoQubitGate):
            action_name = 'r:swap'
            control, target = self.current_mapping[action.left], self.current_mapping[action.right]
            swap_control, swap_target = inverse_mapping[control], inverse_mapping[target]
            action = SwapTwoQubitGate(
                swap_control, swap_target
            )
            # print("swap gates is :", best_swap_qubits.left, best_swap_qubits.right)
            trans_mapping[action.left], trans_mapping[action.right] = (
                trans_mapping[action.right],
                trans_mapping[action.left],
            )
        else:
            action_name = 'r:bridge'
        front_layer = get_front_layer(self.remaining_dag)
        if not action.apply(self.resulting_dag, front_layer, self.initial_mapping,
                            trans_mapping):
            raise ValueError(f'Cannot apply swap/bridge: {action}')

        self.action_stats[action_name] += 1
        # if isinstance(action, SwapTwoQubitGate): # Only swap actions change the current mapping.
        #     # Before update(), calculate the potential delta. The potential is to help us estimate how the
        #     # state will be affected by a mapping changing.
        #     prev_potential = self.update_state_potential()
        #     # The cost of a circuit is a potential function of the state.
        #     # F(s', s) = gamma * phi(s') - phi(s)
        #     shaping = self.state_potential * self.gamma - prev_potential
        # else:
        #     shaping = 0
        shaping = 0
        num_exe_cnot = self.update()
        reward = self.reward_shaping_weight * shaping + num_exe_cnot
        return reward

    def step(
            self, policy: int
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        info = {}
        inverse_current_mapping = {val: key for key, val in self.current_mapping.items()}
        action = self.action.decode(policy, self.initial_mapping,
                                              inverse_current_mapping,
                                              self.inverse_mapping, self.hardware)
        try:
            reward = self.apply_action(action)
        except ValueError as e:
            return self.invalid_action(why=str(e))
        self.invalid_actions = 0 # Clear the counter since we get a valid action.

        if self.is_routing_finished() and (self.trans_after_routing == self.trans_after_routing_limit or
                                           action == ActionSpace.ACTION_FINISH):
            # Routing is finished and agent just reaches extra transformation limit or outputs 'finish' action.
            self.finalize_result()
            reward += self.num_qubits * self.bonus_weight  # Final bonus to motivate agent to finish faster.
            # readable_metrics = readable_float_dict(self.metrics)
            # print(f'Game ends {readable_metrics}')
            return self._get_obs(), reward, True, False, info

        reward -= self.step_penalty # Except the last step, all preceding steps get a step penalty.
        return self._get_obs(), reward, False, False, info

    @property
    def num_qubits(self):
        return self.input_circuit.num_qubits

    def invalid_action(self, why: str):
        print(f'Invalid action, reason: {why}')
        self.invalid_actions += 1
        if self.invalid_actions >= self.invalid_action_limit: # Too many invalid action. Truncated.
            print('Truncated due to too many invalid actions. Limit is', self.invalid_action_limit)
            return self._get_obs(), 0, False, True, {}
        return self._get_obs(), -self.invalid_action_penalty, False, False, {}

    def apply_action(self, action: Union[TwoQubitGate, Tuple[int, TransformationPass]]):
        if isinstance(action[1], TransformationPass):
            return self.apply_transform_action(action)
        if isinstance(action, TwoQubitGate):
            return self.apply_route_action(action)
        # This is the finish action
        assert action == ActionSpace.ACTION_FINISH, f'Action should be <finished> but {action}!'
        if not self.is_routing_finished():
            raise ValueError('Routing is not finished but agent outputs finish action')
        # Routing is finished and agent outputs finish action. Bonus will be given later in step().
        return 0

    def finalize_result(self):
        assert self.is_routing_finished(), 'Routing is not finished!'
        self.resulting_circuit = dag_to_circuit(self.resulting_dag)

        record = {}
        metrics = qknob_metrics(self.input_circuit, self.resulting_circuit)
        for key, value in metrics.items():
            record['metric/' + key] = value

        total = sum(self.action_stats.values())  # py39 has no total() in Counter
        for key, value in self.action_stats.items():
            record['action/' + key] = value / total * 100

        for key, value in self.metrics_baseline.items():
            record['diff/' + key] = metrics[key] - value

        self.metrics = record

    def update(self):
        num_executed_cnot = 0
        front_layer = QuantumLayer()
        topological_nodes = list(self.remaining_dag.topological_op_nodes())
        # No need to build node level since we just scan through the whole list of nodes.
        current_node_index = update_layer(front_layer, topological_nodes, 0)

        # Start of the iterative algorithm
        while not front_layer.is_empty():
            execute_gate_list = QuantumLayer()
            for op in front_layer.ops:
                if self.hardware.can_natively_execute_operation(op, self.current_mapping):
                    execute_gate_list.add_operation(op)
                    self.remaining_dag.remove_op_node(op)
                    if len(op.qargs) == 2:
                        num_executed_cnot += 1
                        q1, q2 = op.qargs[0]._index, op.qargs[1]._index
                    # Delaying the remove operation because we do not want to remove from
                    # a container we are iterating on.
                    # front_layer.remove_operation(op)
            if not execute_gate_list.is_empty():
                front_layer.remove_operations_from_layer(execute_gate_list)
                execute_gate_list.apply_back_to_dag_circuit(
                    self.resulting_dag, self.initial_mapping, self.trans_mapping
                )
                current_node_index = update_layer(front_layer, topological_nodes, current_node_index)
            else:
                break
        return num_executed_cnot

    def action_masks(self):
        masks = np.zeros(self.action.get_size(), dtype=bool)
        # Since we apply transformations in the routed subscircuit, it needs to non-empty.
        self.swap_masks(masks)
        self.bridge_masks(masks)
        self.transformation_masks(masks)
        return masks.tolist()

    def transformation_masks(self, masks):
        # To transform something, you need to have something :)
        allow_transform_routed = self.resulting_dag.size() > 0
        allow_transform_unrouted = self.remaining_dag.size() > 0
        if self.is_routing_finished():
            allow_transform_routed &= self.trans_after_routing < self.trans_after_routing_limit

        for opt in self.action.OPT_PASSES:
            masks[self.action.action_to_index[ActionSpace.TRANS_ROUTED, opt]] = allow_transform_routed
            masks[self.action.action_to_index[ActionSpace.TRANS_UNROUTED, opt]] = allow_transform_unrouted

    def bridge_masks(self, masks):
        trans_mapping = self.trans_mapping
        initial_mapping = self.initial_mapping

        inverse_trans_mapping = {val: key for key, val in trans_mapping.items()}
        inverse_mapping = {val: key for key, val in initial_mapping.items()}
        front_layer = get_front_layer(self.remaining_dag)
        for op in front_layer.ops:
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

    def swap_masks(self, masks):
        # First, compute all the qubits involved in the given layer
        qubits_involved_in_front_layer = set()
        front_layer = get_front_layer(self.remaining_dag)
        for op in front_layer.ops:
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
                # two_qubit_gate = SwapTwoQubitGate(
                #     inverse_mapping[source], inverse_mapping[sink]
                # )
