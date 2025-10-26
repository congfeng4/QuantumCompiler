import subprocess
import sys
import os
from contrib.baselines import SUPPORTED_LAYOUT_METHOD
from contrib.common import dict_product, read_json
from contrib.maskable_ppo import CircuitDataset
from argparse import ArgumentParser
from pathlib import Path
import shelve

data_choices = list(Path('./data').iterdir())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='20Q_gate_Tokyo', choices=data_choices)
    args = parser.parse_args()

    db = shelve.open(f'./output/db/{args.data}', writeback=True)

    dataset = CircuitDataset(args.data, dataroot=Path('./data'))

    all_keys = dict_product({
        'layout_method': SUPPORTED_LAYOUT_METHOD,
        'circuit_path': dataset.circuit_paths,
    })

    for key in all_keys:
        layout_method = key['layout_method']
        circuit_path = key['circuit_path']
        db_key = '-'.join([layout_method, circuit_path.name])
        output_dir = Path('./output') / db_key

        if db_key in db:
            continue

        cmd = (f'{sys.executable} train.py '
               f'--path {circuit_path} '
               f'--layout {layout_method} '
               f'--output {output_dir} '
               f'--hardware {dataset.hardware_name}').split()

        try:
            subprocess.run(cmd, cwd=Path.cwd().absolute())
        except Exception as e:
            print(key, 'failed', 'error', e)
            continue

        metrics = read_json(output_dir / 'metrics.json')
        db[db_key] = metrics
