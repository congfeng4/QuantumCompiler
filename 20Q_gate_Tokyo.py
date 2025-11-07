from qiskit import QuantumCircuit
from pathlib import Path
import shelve
from contrib.baselines import SUPPORTED_GRAPH_MODEL, LayoutMethod
from contrib.maskable_ppo import run_as_subprocess, load_circuits


if __name__ == '__main__':
    dataname = '20Q_gate_Tokyo'

    with shelve.open(f'./output/db/nam_circs', writeback=True) as db:
        for path in load_circuits(f'./data/{dataname}/circuits'):
            for hardware in [ 'tokyo' ]:

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
                        n_envs=6,
                        basic_gates=['cx', 'u', 'h'],
                        total_timesteps=100,  # 100K.
                    )
                except KeyboardInterrupt:
                    raise
                except:
                    continue
                else:
                    if isinstance(metrics, dict):
                        db[db_key] = metrics
