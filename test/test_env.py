"""
Test the correctness of implementation of env
"""
import pytest
from qiskit import QuantumCircuit

from contrib.common import get_hardware_name, get_all_qknob_circuit_paths
from contrib.environs import CircuitEnvWithInitialMapping
from contrib.ha_traj import run_ha, InitialMappingStrategy, get_initial_mapping
from contrib.pretrain_env import TrajectoryCollector, ha_mapping, rollout_expert_trajectory

from hamap import IBMQHardwareArchitecture


ALL_QKNOB_CIRCUIT_PATHS = get_all_qknob_circuit_paths()


@pytest.mark.parametrize('data', ALL_QKNOB_CIRCUIT_PATHS.keys())
def test_env(data: str):
    for circuit_path in ALL_QKNOB_CIRCUIT_PATHS[data]:
        hardware_name = get_hardware_name(data)
        circuit = QuantumCircuit.from_qasm_file(str(circuit_path))
        hardware = IBMQHardwareArchitecture(hardware_name)
        collector = TrajectoryCollector(N=hardware.qubit_number, L=10)

        init = get_initial_mapping(circuit, hardware, InitialMappingStrategy.IDENTITY)
        ha_mapping(collector, circuit, init, hardware, strategy='best')
        traj = collector.trajectories[0]
        metrics = collector.metrics_list[0]
        env = CircuitEnvWithInitialMapping(circuit, hardware, init, L=10)
        metrics_env = rollout_expert_trajectory(env, traj)

        for key in metrics:
            val_1 = metrics[key]
            val_2 = metrics_env[key]
            assert abs(val_1 - val_2) < 1e-5, f"{key=}, {val_1=}, {val_2=}"
