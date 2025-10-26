import time
from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path

from contrib.maskable_ppo import CircuitDataset
from contrib.common import dict_product
from contrib.baselines import transpile_circuit


def make_rounds(circuit_dir: Path, opt_methods: list, repeats=3, graph=None, verbose=False):
    assert circuit_dir.is_dir(), circuit_dir
    circuit_paths = list(circuit_dir.glob('*.qasm'))

    assert len(circuit_paths), f'No qasm file found in {circuit_dir}'

    rounds = dict_product({
        'circuit_path': circuit_paths,
        'graph_model': SUPPORTED_GRAPH_MODEL if graph is None else [graph],
        'opt_method': opt_methods,
    }) * repeats

    if verbose:
        print('Circuit dir', circuit_dir)
        print('No. circuits', len(circuit_paths))
        print('Opt methods:', opt_methods)
        print('Repeats', repeats)
        print('Make rounds:', len(rounds))

    return rounds


def run_transpile_and_save_results(circuit_dir: Path,
                                   opt_methods: list,
                                   gate_set,
                                   save_file: Path = None,
                                   ecc_file=None,
                                   n_jobs=-1,
                                   graph=None,
                                   repeats: int = 3,
                                   verbose=True):
    rounds = make_rounds(circuit_dir, opt_methods, repeats=repeats, graph=graph, verbose=verbose)

    results = Parallel(n_jobs=n_jobs, verbose=1)(delayed(transpile_circuit)(
        gate_set=gate_set,
        ecc_file=ecc_file,
        **rnd,
    ) for rnd in rounds)

    df = pd.DataFrame.from_records(rec.__dict__ for rec in results)
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
    for data in data_list:
        dataset = CircuitDataset(data, sort=True)
        run_transpile_and_save_results(dataset.circuit_dir,
                                       opt_methods=['qiskit:0', 'qiskit:1', 'qiskit:2'],
                                       gate_set='h cx u'.split(),
                                       save_file=f'./output/baseline/{data}.csv',
                                       graph=dataset.hardware)
