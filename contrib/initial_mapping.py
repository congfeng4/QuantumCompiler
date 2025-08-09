from enum import Enum
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from contrib.common import qknob_metrics

from hamap._cli.compare_initial_mappings import get_mapping_cost, get_initial_mapping_from_annealing, \
    get_random_mapping, \
    initial_mapping_from_sabre, wrap_iterative_mapping_algorithm

from hamap import IBMQHardwareArchitecture
from hamap.mapping import ha_mapping


class InitialMappingStrategy(Enum):
    SIMULATE_ANNEALING = 'sa'
    SABRE = 'sabre'
    RANDOM = 'random'
    IDENTITY = 'identity'


def get_initial_mapping(circuit: QuantumCircuit, hardware: IBMQHardwareArchitecture,
                        initial_mapping_strategy: InitialMappingStrategy):
    if initial_mapping_strategy == InitialMappingStrategy.RANDOM:
        return get_random_mapping(circuit)
    if initial_mapping_strategy == InitialMappingStrategy.IDENTITY:
        return {qubit: i for i, qubit in enumerate(circuit.qubits)}
    if initial_mapping_strategy == InitialMappingStrategy.SIMULATE_ANNEALING:
        return get_initial_mapping_from_annealing(get_mapping_cost, circuit, hardware)[0]
    if initial_mapping_strategy == InitialMappingStrategy.SABRE:
        return initial_mapping_from_sabre(circuit, hardware, wrap_iterative_mapping_algorithm)
    raise ValueError(initial_mapping_strategy)


def ha_baseline(qc: QuantumCircuit, hardware: IBMQHardwareArchitecture, initial_mapping: dict[Qubit, int]):
    """
    Run HA baseline and return QKNOB metrics.
    """
    mapped_circuit, final_mapping = ha_mapping(qc, initial_mapping, hardware)
    metrics = qknob_metrics(qc, mapped_circuit)

    return metrics
