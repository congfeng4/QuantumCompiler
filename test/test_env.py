"""
Test the correctness of implementation of env
"""
import json

import pytest
from qiskit import QuantumCircuit

from contrib.common import get_hardware_name, get_all_qknob_circuit_paths
from contrib.environs import CircuitEnvWithInitialMapping
from contrib.ha_traj import run_ha, InitialMappingStrategy
from hamap import IBMQHardwareArchitecture
from script.benchmark_ha_qknob import HA_RESULT_DIR
from pathlib import Path

ALL_QKNOB_CIRCUIT_PATHS = get_all_qknob_circuit_paths()

@pytest.mark.parametrize('data', ALL_QKNOB_CIRCUIT_PATHS.keys())
def test_env(data: str):
    for circuit_path in ALL_QKNOB_CIRCUIT_PATHS[data]:
        hardware_name = get_hardware_name(data)
        result = run_ha(str(circuit_path), hardware_name, InitialMappingStrategy.IDENTITY)
        traj = result['trajectory']
        circuit = QuantumCircuit.from_qasm_file(str(circuit_path))
        hardware = IBMQHardwareArchitecture(hardware_name)
        result = CircuitEnvWithInitialMapping.apply_trajectory(result)
        metrics, metrics_env = result['metrics'], result['metrics_env']
        terminate, traj_len = result['terminate'], result['traj_len']

        assert traj_len == len(traj) and terminate, f"Actual {traj_len=}, Expected {len(traj)=}, {terminate=}"

        for key in metrics:
            val_1 = metrics[key]
            val_2 = metrics_env[key]
            assert abs(val_1 - val_2) < 1e-5, f"{key=}, {val_1=}, {val_2=}"
