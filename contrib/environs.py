from collections import defaultdict
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

from contrib.baselines import OptMethod, transpile_circuit
from contrib.common import qknob_metrics, readable_float_dict, get_circuit_cost, get_front_layer, get_weighted_ops, \
    get_total_ops, get_inverse_mapping
from contrib.expert import ha_baseline
from contrib.common import get_cnot_num, get_distance_matrix
from contrib.action_space import ActionSpace
from contrib.state_space import StateSpace

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

    def __init__(self, hardware: IBMQHardwareArchitecture, **kwargs):
        super().__init__()
        self.hardware = hardware
        self.N = hardware.qubit_number

        self.action = ActionSpace(hardware)
        self.state = StateSpace()

        self.action_space = self.action.to_gym_space()
        self.observation_space = self.state.to_gym_space()


class CircuitEnvWithInitialMapping(BaseCircuitEnv):

    def __init__(self,
                 input_circuit: QuantumCircuit,
                 hardware: IBMQHardwareArchitecture,
                 initial_mapping: dict[Qubit, int],
                 verbose=False,
                 max_len=None,
                 params=None):


        if params is None:
            params = {}

        if max_len is None:
            max_len = 2 * get_total_ops(input_circuit)
        self.max_len = max_len

        self.input_circuit = input_circuit
        self.hardware = hardware
        self.verbose = verbose
        self.initial_mapping_orig = initial_mapping.copy()
        self.distance_matrix = get_distance_matrix(self.hardware)

        # Hyperparameters

        # Discount factor for future rewards.
        self.gamma = params.get('gamma', 0.99)

        # After the whole circuit is routed, perform some extra transformations.
        self.trans_after_routing_limit = params.get('trans_after_routing_limit', 5)

        # Encourage the agent to use fewer steps.
        self.step_penalty = params.get('step_penalty', 1)

        # Weight for the final bonus reward.
        self.bonus_weight1 = params.get('bonus_weight1', 5)
        self.bonus_weight2 = params.get('bonus_weight2', 10)

        # Penalty for each invalid action.
        self.invalid_action_penalty = params.get('invalid_action_penalty', 10)

        # Max number of consecutive invalid actions.
        self.invalid_action_limit = params.get('invalid_action_limit', 100)

        # Relative weight to two-qubit gates' reduction.
        self.one_qubit_gate_weight = params.get('one_qubit_gate_weight', 0.2)

        # Before the whole circuit is routed, perform some swaps to adjust the mapping.
        self.swaps_before_routing_limit = params.get('swaps_before_routing_limit', 5)

        # When the agent uses the special actions correctly, give it a reward:
        self.reward_for_special_action = params.get('reward_for_special_action', 1)

        _adapt_quantum_circuit_and_mapping_arity(self.input_circuit, initial_mapping, hardware)
        self.input_circuit = input_circuit
        self.remaining_dag: Optional[DAGCircuit] = None
        self.resulting_circuit = None
        self.metrics_baseline = transpile_circuit(input_circuit, hardware, opt_method=OptMethod.QISKIT_LV2)

        self.action = ActionSpace(hardware)
        self.state = StateSpace(self.max_len)
        self.action_space = self.action.to_gym_space()
        self.observation_space = self.state.to_gym_space()

        # check_env(self)

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        if self.verbose: print('reset')
        super().reset(seed=seed)
        # Boolean flags.
        self.is_routing_started = False
        self.is_done = False

        # Counts
        self.invalid_actions = 0
        self.trans_after_routing = 0
        self.swaps_before_routing = 0

        # Mappings.
        self.current_mapping = self.initial_mapping_orig.copy()
        self.trans_mapping = self.current_mapping.copy()
        self.initial_mapping = self.current_mapping.copy()
        self.inverse_mapping = get_inverse_mapping(self.initial_mapping)  # Inverse of initial-mapping.

        # Circuits/DAGs.
        self.remaining_dag = circuit_to_dag(self.input_circuit)
        self.resulting_dag = _create_empty_dagcircuit_from_existing(self.remaining_dag)

        # Debug statistics.
        self.action_stats = defaultdict(int)
        self.reward_stats = defaultdict(float)

        return self._get_obs(), {}

    def is_routing_finished(self):
        return self.remaining_dag.size() == 0

    def _get_obs(self):
        return self.state.encode(resulting_dag=self.resulting_dag,
                                 remaining_dag=self.remaining_dag,
                                 current_mapping=self.current_mapping,
                                 distance_matrix=self.distance_matrix)

    def apply_transform_action(self, action: Tuple[int, TransformationPass]):
        assert isinstance(action[0], int), action[0]
        assert isinstance(action[1], TransformationPass), action[1]
        phase, opt_pass = action

        if self.is_routing_finished():
            if phase == ActionSpace.TRANS_UNROUTED:
                raise ValueError(f'Cannot transform the unrounted circuit after routing')
            action_name = f'after:{opt_pass.name()}'
            if self.trans_after_routing >= self.trans_after_routing_limit:
                # When too many trans after routing, although it is done, still give a penalty.
                # Although, this may not be possible since once limit is reached, we set is_done=True.
                raise ValueError(f'Attempt more trans after routing. Limit is {self.trans_after_routing_limit}')
            action_name = f'after/{opt_pass.name()}'
        elif phase == ActionSpace.TRANS_ROUTED:
            action_name = f'routed/{opt_pass.name()}'
        else:
            action_name = f'unrouted/{opt_pass.name()}'

        old_dag = self.resulting_dag if phase == ActionSpace.TRANS_ROUTED else self.remaining_dag
        old_dag_count = get_weighted_ops(old_dag.count_ops()) + old_dag.depth()
        try:
            new_dag: DAGCircuit = opt_pass.run(old_dag)
        except Exception as e:
            raise ValueError(f'Failed to run transformation {opt_pass}, error: {e}')

        if self.is_routing_finished():
            self.trans_after_routing += 1
            if self.trans_after_routing == self.trans_after_routing_limit:
                self.is_done = True

        new_dag_count = get_weighted_ops(new_dag.count_ops()) + new_dag.depth()
        reward = old_dag_count * self.gamma - new_dag_count
        # print('old_dag_count', old_dag_count, 'new_dag_count', new_dag_count)
        if phase == ActionSpace.TRANS_ROUTED:
            self.resulting_dag = new_dag
            return reward, action_name

        # Phase is UNROUTED, Some gates may become executable. But the mapping don't change.
        # So, don't evaluate the swap cost potential.
        # Our hope is that a transformation can reduce ops and make some gate executable.
        self.remaining_dag = new_dag
        if self.is_routing_started:
            self.update()
        return reward, action_name

    def get_circuit_routing_cost(self):
        return get_circuit_cost(self.remaining_dag, self.current_mapping, self.distance_matrix, self.hardware)

    def start_routing(self):
        assert not self.is_routing_started, 'Make sure self.is_routing_started is False!'
        if self.verbose: print('Routing is started')
        self.is_routing_started = True
        self.initial_mapping = self.current_mapping.copy()
        self.trans_mapping = self.initial_mapping.copy()
        self.inverse_mapping = get_inverse_mapping(self.initial_mapping)
        self.update()

    def apply_route_action(self, action: TwoQubitGate):
        trans_mapping = self.trans_mapping
        inverse_mapping = self.inverse_mapping

        if not self.is_routing_started:
            if not isinstance(action, SwapTwoQubitGate):
                raise ValueError('Only swap is allowed before routing is started')
            if self.swaps_before_routing >= self.swaps_before_routing_limit:
                raise ValueError(f'Attempt more swaps before routing, limit is {self.swaps_before_routing_limit}')
            self.swaps_before_routing += 1  # Must succeed so add one here.

        old_cost = self.get_circuit_routing_cost()  # For reward computation.
        old_ops = get_total_ops(self.remaining_dag)

        # We now have our best SWAP/Bridge, let's perform it!
        self.current_mapping = action.update_mapping(self.current_mapping)
        if isinstance(action, SwapTwoQubitGate):  # Recover the swap gate.
            action_name = 'route/swap' if self.is_routing_started else 'before/swap'
            control, target = self.current_mapping[action.left], self.current_mapping[action.right]
            swap_control, swap_target = inverse_mapping[control], inverse_mapping[target]
            action = SwapTwoQubitGate(swap_control, swap_target)
            # print("swap gates is :", best_swap_qubits.left, best_swap_qubits.right)
            trans_mapping[action.left], trans_mapping[action.right] = (
                trans_mapping[action.right],
                trans_mapping[action.left],
            )
            new_cost = self.get_circuit_routing_cost()
        else:
            assert self.is_routing_started, 'Bridge is only allowed after routing is started!'
            action_name = 'route/bridge'
            new_cost = old_cost  # Bridge doesn't change the mapping. The cost should be the same.

        reward = self.gamma * old_cost - new_cost
        if self.is_routing_started:  # Insert the routing gates.
            front_layer = get_front_layer(self.remaining_dag)
            if not action.apply(self.resulting_dag, front_layer, self.initial_mapping,
                                trans_mapping):
                raise ValueError(f'Cannot apply swap/bridge: {action}')
            self.update()
            new_ops = get_total_ops(self.remaining_dag)
            reward += self.gamma * old_ops - new_ops
        else:
            assert isinstance(action, SwapTwoQubitGate), 'Bridge is not allowed before routing'
            if self.swaps_before_routing == self.swaps_before_routing_limit:
                # Limit is reached. Automatically start routing.
                self.start_routing()

        return reward, action_name

    def step(self, policy: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        info = {}
        inverse_current_mapping = {val: key for key, val in self.current_mapping.items()}
        action = self.action.decode(policy,
                                    initial_mapping=self.initial_mapping,
                                    inverse_current_mapping=inverse_current_mapping,
                                    inverse_mapping=self.inverse_mapping,
                                    hardware=self.hardware)
        try:
            reward, action_name = self.apply_action(action)
        except ValueError as e:
            return self.invalid_action(why=str(e))
        self.invalid_actions = 0  # Clear the counter since we get a valid action.
        self.action_stats[action_name] += 1

        if self.is_done:
            # Routing is finished and agent just reaches transformation limit or outputs 'finish' action.
            self.finalize_result()
            ops_ratio, depth_ratio = self.metrics['metric/ops_ratio'], self.metrics['metric/depth_ratio']
            reward += np.exp((2 - depth_ratio - ops_ratio) / self.bonus_weight1) * self.bonus_weight2
            # Final bonus to motivate agent to finish faster.
            if self.verbose:
                readable_metrics = readable_float_dict(self.metrics)
                print(f'Game ends {readable_metrics}')
        else:
            reward -= self.step_penalty  # Except the last step, all preceding steps get a step penalty.

        self.reward_stats[action_name] += reward
        if self.verbose and any(x in action_name for x in 'routed unrouted'.split()):
            print(action_name, 'reward', round(reward, 2),
                  'remain', get_total_ops(self.remaining_dag), 'result', get_total_ops(self.resulting_dag))

        return self._get_obs(), reward, self.is_done, False, info

    @property
    def num_qubits(self):
        return self.input_circuit.num_qubits

    def invalid_action(self, why: str):
        print(f'Invalid action, reason: {why}')
        self.invalid_actions += 1
        if self.invalid_actions >= self.invalid_action_limit:  # Too many invalid action. Truncated.
            print('Truncated due to too many invalid actions. Limit is', self.invalid_action_limit)
            return self._get_obs(), 0, False, True, {}
        return self._get_obs(), -self.invalid_action_penalty, False, False, {}

    def apply_special_action(self, action: str):
        if action == ActionSpace.ACTION_FINISH:
            if not self.is_routing_finished():
                raise ValueError('Routing is not finished but agent outputs finish action')
            assert not self.is_done, 'This is not possible'
            self.is_done = True
            return self.reward_for_special_action, action
        if action == ActionSpace.ACTION_START:
            if self.is_routing_started:
                raise ValueError('Routing has already started but agent outputs start action')
            self.start_routing()
            return self.reward_for_special_action, action
        raise ValueError(f'Invalid special action: {action}')

    def apply_action(self, action: Union[str, TwoQubitGate, Tuple[int, TransformationPass]]):
        if isinstance(action, tuple):
            return self.apply_transform_action(action)
        if isinstance(action, TwoQubitGate):
            return self.apply_route_action(action)
        if isinstance(action, str):
            return self.apply_special_action(action)
        raise ValueError(f'Unknown action {action}')

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

        for key, value in metrics.items():
            record['diff/' + key] = value - self.metrics_baseline[key]

        for key, value in self.reward_stats.items():
            record['reward/' + key] = value

        self.metrics = record

    def update(self):
        assert self.is_routing_started, 'We cannot update the boundary unless routing is started.'

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
                    # q1, q2 = op.qargs[0]._index, op.qargs[1]._index
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

    def action_masks(self):
        masks = np.zeros(self.action.size, dtype=bool)
        # Since we apply transformations in the routed subscircuit, it needs to non-empty.
        self.swap_masks(masks)
        # self.bridge_masks(masks)
        self.transformation_masks(masks)
        # self.special_action_masks(masks)
        return masks.tolist()

    def special_action_masks(self, masks):
        allow_start = not self.is_routing_started
        allow_finish = self.is_routing_finished()
        masks[self.action.action_to_index[ActionSpace.ACTION_START]] = allow_start
        masks[self.action.action_to_index[ActionSpace.ACTION_FINISH]] = allow_finish

    def transformation_masks(self, masks):
        # To transform something, you need to have something :)
        allow_transform_routed = self.resulting_dag.size() > 0
        # allow_transform_unrouted = self.remaining_dag.size() > 0
        # if self.is_routing_finished():
        #     allow_transform_routed &= self.trans_after_routing < self.trans_after_routing_limit

        for opt in self.action.trans:
            masks[self.action.action_to_index[ActionSpace.TRANS_ROUTED, opt]] = allow_transform_routed
            # masks[self.action.action_to_index[ActionSpace.TRANS_UNROUTED, opt]] = allow_transform_unrouted

    def bridge_masks(self, masks):
        if not self.is_routing_started or self.is_routing_finished():
            # Before started or after finished, no bridge allowed.
            return  # by default, all zeros.

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
        if not self.is_routing_started and self.swaps_before_routing >= self.swaps_before_routing_limit:
            return  # Before started, but all swaps are used up.
        if self.is_routing_finished():
            return  # After finished, only transformation can be applied.

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
