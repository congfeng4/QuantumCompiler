from qiskit import QuantumCircuit
from pathlib import Path
import shelve
from contrib.baselines import SUPPORTED_GRAPH_MODEL, LayoutMethod
from contrib.maskable_ppo import load_circuits, run_as_subprocess


if __name__ == '__main__':

    with shelve.open(f'./output/db/nam_circs', writeback=True) as db:
        for path in load_circuits('./data/nam_circs'):
            for hardware in [ 'grid' ]:

                qc = QuantumCircuit.from_qasm_file(path)
                db_key = '-'.join([path.name, hardware.lower()])
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
                        n_envs=6,
                        total_timesteps=100,
                    )
                except KeyboardInterrupt:
                    raise
                except:
                    continue
                else:
                    if isinstance(metrics, dict):
                        db[db_key] = metrics
