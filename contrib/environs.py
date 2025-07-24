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
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from contrib.common import qknob_metrics
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy
from contrib.pretrain_env import PretrainEnv

from hamap.distance_matrix import get_distance_matrix_swap_number_and_error
from hamap.gates import SwapTwoQubitGate, BridgeTwoQubitGate, TwoQubitGate
from hamap.heuristics import sabre_heuristic
from hamap.layer import QuantumLayer, update_layer
from hamap.mapping import _adapt_quantum_circuit_and_mapping_arity, _create_empty_dagcircuit_from_existing
from hamap import IBMQHardwareArchitecture, mapping_to_str

import logging

logger = logging.getLogger("contrib.env")


class CircuitEnvWithInitialMapping(PretrainEnv):

    def __init__(self,
                 input_circuit: QuantumCircuit,
                 hardware: IBMQHardwareArchitecture,
                 initial_mapping: dict[Qubit, int],
                 L: int,
                 cost_ceof: float = 0.1):
        super().__init__(N=hardware.qubit_number, L=L)
        self.input_circuit= input_circuit
        self.hardware = hardware
        self.cost_ceof = cost_ceof
        self.initial_mapping = initial_mapping
        self.distance_matrix = get_distance_matrix_swap_number_and_error(self.hardware)

        _adapt_quantum_circuit_and_mapping_arity(self.input_circuit, initial_mapping, hardware)
        self.dag_circuit = circuit_to_dag(input_circuit)
        self.topological_nodes: list[DAGNode] = list(self.dag_circuit.topological_op_nodes())

        self.resulting_circuit = None

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        self.invalid_actions = 0
        self.front_layer = QuantumLayer()
        self.current_node_index = 0
        self.resulting_dag_quantum_circuit = _create_empty_dagcircuit_from_existing(self.dag_circuit)
        self.current_mapping = self.initial_mapping.copy()
        self.trans_mapping = self.initial_mapping.copy()
        self.explored_mappings = set()
        self.inverse_mapping = {val: key for key, val in self.initial_mapping.items()}

        self.update_front_layer()
        self.update()
        return self._get_obs(), {}

    def _get_obs(self):
        return self.state.encode_obs(self.front_layer, self.topological_nodes[self.current_node_index:],
                                     self.current_mapping)

    def finalize_result(self):
        self.resulting_circuit = dag_to_circuit(self.resulting_dag_quantum_circuit)
        self.metrics = qknob_metrics(self.input_circuit, self.resulting_circuit)

    def find_middle(self, best_swap_qubits: BridgeTwoQubitGate, trans_mapping, inverse_mapping) -> Qubit:
        # inverse_trans_mapping = {val: key for key, val in trans_mapping.items()}
        control, target = best_swap_qubits.left, best_swap_qubits.right
        control_index = self.initial_mapping[control]
        target_index = self.initial_mapping[target]
        # For each qubit q linked with control, check if target is linked with q.
        for _, potential_middle_index in self.hardware.out_edges(control_index):
            for _, potential_target_index in self.hardware.out_edges(potential_middle_index):
                if potential_target_index == target_index:
                    return inverse_mapping[potential_middle_index]

        logger.warning("Cannot find middle qubit for BRIDGE %s. Your circuit is probably wrong",
                       best_swap_qubits)
        return self.input_circuit.qubits[0]

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
        if not best_swap_qubits.apply(self.resulting_dag_quantum_circuit, self.front_layer, self.initial_mapping, trans_mapping):
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
        self.invalid_actions += 1
        if self.invalid_actions >= 100:
            return self._get_obs(), -0.1, False, True, {}
        return self._get_obs(), -0.1, False, False, {}

    def heuristic_cost(self, swap: TwoQubitGate):
        return sabre_heuristic(
            hardware=self.hardware, front_layer=self.front_layer, topological_nodes=self.topological_nodes,
            current_node_index=self.current_node_index, current_mapping=self.current_mapping,
            initial_mapping=self.initial_mapping, trans_mapping=self.trans_mapping,
            distance_matrix=self.distance_matrix, tentative_gate=swap,
        )

    def step(
        self, policy
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        inverse_current_mapping = {val: key for key, val in self.current_mapping.items()}
        best_swap_qubits = self.action.decode_best_swap(policy, inverse_current_mapping, self.inverse_mapping)
        if isinstance(best_swap_qubits, BridgeTwoQubitGate):
            # Patch the middle qubit before changing the mapping.
            best_swap_qubits._middle = self.find_middle(best_swap_qubits, self.trans_mapping, self.inverse_mapping)

        cost = self.heuristic_cost(best_swap_qubits)
        if not self.apply_swap_action(best_swap_qubits):
            return self.step_invalid()

        self.invalid_actions = 0
        num_executed_cnot = self.update()
        reward = -cost * self.cost_ceof + num_executed_cnot - 3
        # reward = num_executed_cnot - 3
        done = not self.front_layer
        info = {}
        if done:
            self.finalize_result()
            info['metrics'] = self.metrics
            print(f'Game ends {self.metrics}')
        return self._get_obs(), reward, done, False, info

    def action_masks(self):
        masks = self.swap_masks() + self.bridge_masks()
        assert any(masks)
        return masks

    def bridge_masks(self):
        masks = np.zeros((self.num_qubits, self.num_qubits), dtype=bool)
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
                        masks[control_index, target_index] = True

        return masks.reshape(-1).tolist()

    def swap_masks(self):
        # First compute all the qubits involved in the given layer
        masks = np.zeros((self.num_qubits, self.num_qubits), dtype=bool)
        qubits_involved_in_front_layer = set()
        for op in self.front_layer.ops:
            qubits_involved_in_front_layer.update(op.qargs)
        # inverse_mapping = {val: key for key, val in self.current_mapping.items()}
        # Then for all the possible links that involve at least one of the qubits used by
        # the gates in the given layer, add this link as a possible SWAP.
        # all_swaps = list()
        for involved_qubit in qubits_involved_in_front_layer:
            qubit_index = self.current_mapping[involved_qubit]
            # For all the links that involve the current qubit.
            for source, sink in self.hardware.out_edges(qubit_index):
                masks[source, sink] = True

        return masks.reshape(-1).tolist()


gym.register("CircuitEnv", "contrib.environs:CircuitEnvWithInitialMapping")


def make_circuit_env(input_circuit_paths: list[Path], hardware_name: str, init: InitialMappingStrategy, L: int):
    """
    Utility to create an env properly.
    """
    hardware = IBMQHardwareArchitecture(hardware_name)
    env = DummyVecEnv([lambda : CircuitEnvWithInitialMapping(
        input_circuit=(input_circuit := QuantumCircuit.from_qasm_file(str(qc))),
        hardware=hardware,
        initial_mapping=get_initial_mapping(input_circuit, hardware, init),
        L=L,
    ) for qc in input_circuit_paths])
    env = VecMonitor(env)
    return env



if __name__ == '__main__':
    from contrib.pretrain_env import TrajectoryCollector, ha_mapping, rollout_expert_trajectory

    hardware = IBMQHardwareArchitecture('tokyo')

    collector_train = TrajectoryCollector(N=hardware.qubit_number, L=10,
                                          outdir=Path('../result/pretrain/ha'),
                                          prefix='20Q_gate_Tokyo_train')

    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits').glob('*.qasm'))

    qc = QuantumCircuit.from_qasm_file(str(circuit_list[0]))
    init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)
    ha_mapping(
        collector=collector_train,
        quantum_circuit=qc,
        initial_mapping=init,
        hardware=hardware,
        strategy='best',
    )

    env = CircuitEnvWithInitialMapping(qc, hardware, init, 10)

    metrics_rollout = rollout_expert_trajectory(env, collector_train.trajectories[0])
    metrics_expert = collector_train.metrics_list[0]
