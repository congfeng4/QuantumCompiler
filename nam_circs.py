from qiskit import QuantumCircuit
from pathlib import Path
import shelve
from contrib.baselines import SUPPORTED_GRAPH_MODEL, LayoutMethod
from contrib.maskable_ppo import run_as_subprocess


if __name__ == '__main__':

    db = shelve.open(f'./output/db/nam_circs', writeback=True)

    for path in Path('./data/nam_circs').glob("*.qasm"):
        for hardware in [ 'star', 'line', 'grid', 'ring' ]:
            for i in range(5):
                qc = QuantumCircuit.from_qasm_file(path)
                db_key = '-'.join([path.name, hardware.lower(), str(i)])
                if db_key in db:
                    print('Skip', db_key)
                    continue

                output_dir = Path('./output/ours/nam_circs') / db_key
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
