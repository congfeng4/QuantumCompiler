from qiskit import QuantumCircuit
from pathlib import Path
import shelve
from contrib.baselines import SUPPORTED_GRAPH_MODEL, LayoutMethod
from contrib.maskable_ppo import run_as_subprocess


if __name__ == '__main__':
    dataname = '20Q_gate_Tokyo'

    db = shelve.open(f'./output/db/{dataname}', writeback=True)
    hardware = 'tokyo'
    
    for path in Path(f'./data/{dataname}/circuits/').glob("*.qasm"):
        qc = QuantumCircuit.from_qasm_file(path)
        db_key = '-'.join([path.name, hardware.lower()])
        if db_key in db:
            print('Skip', db_key)
            continue

        output_dir = Path(f'./output/ours/{dataname}') / db_key
        print(db_key)
        try:
            metrics = run_as_subprocess(
                circuit_path=path,
                layout_method=LayoutMethod.SABRE,
                hardware_name=hardware,
                output_dir=output_dir,
            )
        except KeyboardInterrupt:
            raise
        except:
            continue
        else:
            if isinstance(metrics, dict):
                db[db_key] = metrics
    db.close()
