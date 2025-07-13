import json
from collections import defaultdict
from typing import Any, SupportsFloat
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGNode
from stable_baselines3.common.monitor import Monitor

from contrib.common import show_mapping, qknob_metrics
from contrib.ha_traj import convert_action_to_gate, get_initial_mapping, InitialMappingStrategy
from contrib.action import ActionAsTuplePolicy, ActionType

from hamap.gates import SwapTwoQubitGate, BridgeTwoQubitGate
from hamap.layer import QuantumLayer, update_layer
from hamap.mapping import _adapt_quantum_circuit_and_mapping_arity, _create_empty_dagcircuit_from_existing
from hamap import IBMQHardwareArchitecture

import logging
logger = logging.getLogger("contrib.env")


def build_interact_graph(topological_nodes: list[DAGNode], num_qubits: int):
    graph = np.zeros((num_qubits, num_qubits), dtype=np.float32)
    for op in topological_nodes:
        if len(op.qargs) == 2:
            q1, q2 = op.qargs[0]._index, op.qargs[1]._index
            graph[q1][q2] += 1
            graph[q2][q1] += 1
    return graph


class CircuitEnvWithInitialMapping(gym.Env):

    NUM_ACTIONS = 2  # 2 for bridge and swap.

    def __init__(self, *,
                 input_circuit: QuantumCircuit = None,
                 hardware: IBMQHardwareArchitecture = None,
                 initial_mapping: dict[Qubit, int] = None):
        self.input_circuit= input_circuit
        self.hardware = hardware
        self.initial_mapping = initial_mapping

        _adapt_quantum_circuit_and_mapping_arity(self.input_circuit, initial_mapping, hardware)
        self.dag_circuit = circuit_to_dag(input_circuit)
        self.topological_nodes: list[DAGNode] = list(self.dag_circuit.topological_op_nodes())

        self.observation_space = self._get_obs_space()
        self.action_space = self._get_action_space()
        self.resulting_circuit = None

    def _get_obs_space(self):
        return gym.spaces.Box(low=0, high=float('inf'), shape=(self.num_qubits, self.num_qubits), dtype=np.float32)

    def _get_action_space(self):
        return ActionAsTuplePolicy.action_space(self.num_qubits)

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        self.front_layer = QuantumLayer()
        self.current_node_index = 0
        self.resulting_dag_quantum_circuit = _create_empty_dagcircuit_from_existing(self.dag_circuit)
        self.interact_graph = build_interact_graph(self.topological_nodes, self.num_qubits)
        self.current_mapping = self.initial_mapping.copy()
        self.trans_mapping = self.initial_mapping.copy()
        self.update_front_layer()
        self.update()
        return self._get_obs(), {}

    def _get_obs(self):
        return self.interact_graph

    def apply_map_action(self, action: ActionType):
        logical, physical = action['logical'], action['physical']
        qubit = self.input_circuit.qubits[logical]
        self.current_mapping[qubit] = physical

    def finalize_result(self):
        self.resulting_circuit = dag_to_circuit(self.resulting_dag_quantum_circuit)
        return qknob_metrics(self.input_circuit, self.resulting_circuit)

    def find_middle(self, best_swap_qubits: BridgeTwoQubitGate, trans_mapping, inverse_mapping):
        inverse_trans_mapping = {val: key for key, val in trans_mapping.items()}
        control, target = best_swap_qubits.left, best_swap_qubits.right
        control_index = self.trans_mapping[control]
        target_index = self.trans_mapping[target]
        # For each qubit q linked with control, check if target is linked with q.
        for _, potential_middle_index in self.hardware.out_edges(control_index):
            for _, potential_target_index in self.hardware.out_edges(potential_middle_index):
                if potential_target_index == target_index:
                    return inverse_trans_mapping[potential_middle_index]

        logger.warning("Cannot find middle qubit for BRIDGE %s. Your circuit is probably wrong",
                       best_swap_qubits)
        return self.input_circuit.qubits[0]

    def apply_swap_action(self, action: ActionType):
        inverse_mapping = {val: key for key, val in self.initial_mapping.items()}
        trans_mapping = self.trans_mapping
        best_swap_qubits = convert_action_to_gate(action, self.input_circuit)
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
            # best_swap_qubits._middle = self.find_middle(best_swap_qubits, trans_mapping, self.initial_mapping)
            pass
        best_swap_qubits.apply(self.resulting_dag_quantum_circuit, self.front_layer, self.initial_mapping, trans_mapping)
        self.update_front_layer()

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
                        self.interact_graph[q1][q2] -= 1
                        self.interact_graph[q2][q1] -= 1
                    # Delaying the remove operation because we do not want to remove from
                    # a container we are iterating on.
                    # front_layer.remove_operation(op)
            if not execute_gate_list.is_empty():
                self.front_layer.remove_operations_from_layer(execute_gate_list)
                execute_gate_list.apply_back_to_dag_circuit(
                    self.resulting_dag_quantum_circuit, self.initial_mapping, self.trans_mapping
                )
                self.update_front_layer()
            else:
                break

        return num_executed_cnot

    @property
    def num_qubits(self):
        return self.input_circuit.num_qubits

    def step_swap(self, action: ActionType):
        assert action['action'] != 'MAP', action
        self.apply_swap_action(action)
        num_executed_cnot = self.update()
        reward = num_executed_cnot - 3 + 0.2 * len(self.front_layer)
        done = not self.front_layer
        info = {}
        if done:
            metrics = self.finalize_result()
            info['metrics'] = metrics
        return self._get_obs(), reward, done, False, info

    def step(
        self, policy: ActionAsTuplePolicy.PolicyType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = ActionAsTuplePolicy.from_policy(policy, self.num_qubits)
        return self.step_swap(action)

    @classmethod
    def apply_trajectory(cls, traj_data: dict):
        traj_full = defaultdict(list)
        trajectory: list = traj_data['trajectory']
        metrics = traj_data['metrics']
        input_circuit = traj_data['input_circuit']
        input_circuit = QuantumCircuit.from_qasm_file(input_circuit)
        hardware_name = traj_data['hardware_name']
        initial_mapping = traj_data['initial_mapping']
        initial_mapping = { input_circuit.qubits[int(k)] : v for k, v in initial_mapping.items() }
        env = cls(input_circuit=input_circuit,
                  hardware=IBMQHardwareArchitecture(hardware_name),
                  initial_mapping=initial_mapping)
        state, _ = env.reset()
        traj_full['obs'].append(state)
        done = False
        traj_index = 0
        info = {}
        for action in trajectory:
            traj_index += 1
            if action['action'] == 'MAP':
                continue
            policy = ActionAsTuplePolicy.to_policy(action, env.num_qubits)
            state, reward, done, _, info = env.step(policy)
            traj_full['acts'].append(policy)
            traj_full['rews'].append(reward)
            traj_full['obs'].append(state)
            if done:
                break
        assert traj_index == len(trajectory) and done, f'{done=}, {traj_index=}, {len(trajectory)=}'
        return dict(traj_full=traj_full, metrics=metrics, metrics_env=info['metrics'],
                    traj_len=traj_index, terminate=done, hardware_name=hardware_name)

    @classmethod
    def make(cls, input_circuit_path: str, hardware_name: str, init: InitialMappingStrategy):
        """
        Utility to create an env properly.
        """
        input_circuit = QuantumCircuit.from_qasm_file(input_circuit_path)
        hardware = IBMQHardwareArchitecture(hardware_name)
        initial_mapping = get_initial_mapping(input_circuit, hardware, init)
        env = gym.make(
            "CircuitEnv",
            input_circuit=input_circuit,
            hardware=hardware,
            initial_mapping=initial_mapping,
        )
        env = Monitor(env)
        return env


def to_imitation_trajectory(traj_env: dict):
    from imitation.data.types import TrajectoryWithRew

    traj_full = traj_env['traj_full']
    return TrajectoryWithRew(
        obs=np.asarray(traj_full['obs']),
        acts=np.asarray(traj_full['acts']),
        rews=np.asarray(traj_full['rews']),
        infos=None,
        terminal=True,
    )


gym.register("CircuitEnv", "contrib.environs:CircuitEnvWithInitialMapping")

# def make_env():

if __name__ == '__main__':
    traj_data = json.load(Path('/Users/fengcong/HA/result/ha/53Q_gate_Sycamore_small_2_10_1.5_no.4-init=identity-data=53Q_gate_Sycamore.json').open())
    traj_full = CircuitEnvWithInitialMapping.apply_trajectory(traj_data)
    expert_traj = to_imitation_trajectory(traj_full)
    print(expert_traj)
