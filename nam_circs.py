import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumCircuit
from pathlib import Path
import shelve
from contrib.initial_mapping import InitialMappingStrategy
from contrib.maskable_ppo import run_maskable_ppo
from hamap import IBMQHardwareArchitecture
from joblib import delayed, Parallel


if __name__ == '__main__':

    db = shelve.open(f'./output/db/nam_circs', writeback=True)

    for path in Path('./data/nam_circs').glob("*.qasm"):
        qc = QuantumCircuit.from_qasm_file(path)
        hardware = IBMQHardwareArchitecture('Star', num_nodes=qc.num_qubits)
        db_key = '-'.join([path.name, 'Star'])
        if db_key in db:
            print('Skip', db_key)
            continue

        output_dir = Path('./output/ours/nam_circs') / db_key
        metrics = Parallel(1)([delayed(run_maskable_ppo)(
            hardware=hardware,
            circuit_path=path,
            init_strategy=InitialMappingStrategy.SABRE,
            output_dir=output_dir,
        )])[0]

        db[db_key] = metrics

    db.close()
