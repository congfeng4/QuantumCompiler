import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumCircuit

from contrib.initial_mapping import InitialMappingStrategy
from contrib.maskable_ppo import run_maskable_ppo
from hamap import IBMQHardwareArchitecture

if __name__ == '__main__':
    path = './data/nam_circs/barenco_tof_3.qasm'
    qc = QuantumCircuit.from_qasm_file(path)

    hardware = IBMQHardwareArchitecture('Star', num_nodes=qc.num_qubits)

    run_maskable_ppo(
        hardware=hardware,
        circuit_path=path,
        init_strategy=InitialMappingStrategy.SABRE
    )
