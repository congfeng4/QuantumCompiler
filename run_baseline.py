import time
from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path

from contrib.maskable_ppo import CircuitDataset
from contrib.common import dict_product
from contrib.baselines import *


def run_baseline(
        circuit_paths: list,
        opt_method: list,
        routing_method: list,
        layout_method: list,
        graph_model: list,
        save_file: Path = None,
        ecc_file=None,
        n_jobs=-1,
        verbose=True
):
    rounds = dict_product(dict(
        circuit_path=circuit_paths,
        opt_method=opt_method,
        routing_method=routing_method,
        graph_model=graph_model,
        layout_method=layout_method,
    ))

    results = Parallel(n_jobs=n_jobs, verbose=999)(delayed(transpile_circuit)(
        opt_params=dict(
            ecc_file=ecc_file,
        ),
        **rnd,
    ) for rnd in rounds)

    df = pd.DataFrame.from_records(filter(None, results))

    if save_file is None:
        save_file = f'./tranpile-result-{time.time()}.csv'

    save_file = Path(save_file)
    save_dir = save_file.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_file, index=False)
    if verbose:
        print(f"Done")


if __name__ == '__main__':
    data_list = '20Q_gate_Tokyo 20Q_depth_Tokyo 53Q_depth_Rochester 53Q_depth_Sycamore 53Q_gate_Rochester 53Q_gate_Sycamore'.split()
    # for data in data_list:
    data = '20Q_gate_Tokyo'
    # dataset = CircuitDataset(data, sort=True)
    circuit_paths = list(Path('./data/nam_circs').glob("*.qasm"))

    run_baseline(
        circuit_paths,
        opt_method=SUPPORTED_OPT_METHOD,
        routing_method=SUPPORTED_ROUTING_METHOD,
        layout_method=SUPPORTED_LAYOUT_METHOD,
        graph_model=SUPPORTED_GRAPH_MODEL,
        save_file=Path(f'./output/baseline/nam_circs.csv'),
        n_jobs=1,
    )
