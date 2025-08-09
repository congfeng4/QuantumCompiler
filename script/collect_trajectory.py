from contrib.expert import *


if __name__ == '__main__':
    from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy

    hardware = IBMQHardwareArchitecture('tokyo')

    collector_train = TrajectoryCollector(hardware, L=10,
                                          outdir=Path('../result/pretrain/ha'),
                                          prefix='20Q_gate_Tokyo_train')

    collector_val = TrajectoryCollector(hardware, L=10,
                                        outdir=Path('../result/pretrain/ha'),
                                        prefix='20Q_gate_Tokyo_val')

    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits').glob('*.qasm'))
    random.seed(22)
    random.shuffle(circuit_list)
    T_train = 8
    T_val = 2

    for i in range(T_train):
        qc = QuantumCircuit.from_qasm_file(str(circuit_list[i]))
        init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)
        heuristic_algorithm(
            collector=collector_train,
            quantum_circuit=qc,
            initial_mapping=init,
            hardware=hardware,
        )
    collector_train.save()

    for i in range(T_val):
        qc = QuantumCircuit.from_qasm_file(str(circuit_list[T_train + i]))
        init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)
        heuristic_algorithm(
            collector=collector_val,
            quantum_circuit=qc,
            initial_mapping=init,
            hardware=hardware,
        )

    collector_val.save()
