import subprocess
import sys
from contrib.baselines import SUPPORTED_LAYOUT_METHOD
from contrib.common import dict_product, read_json
from contrib.maskable_ppo import CircuitDataset
from argparse import ArgumentParser
from pathlib import Path
import shelve
from pprint import pp

data_choices = list(Path('./data').iterdir())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='20Q_gate_Tokyo', choices=data_choices)
    args = parser.parse_args()

    db = shelve.open(f'./output/db/{args.data}', writeback=True)
    pp(list(db.items()))

    dataset = CircuitDataset(args.data, dataroot=Path('./data'))

    all_keys = dict_product({
        'layout_method': SUPPORTED_LAYOUT_METHOD,
        'circuit_path': dataset.circuit_paths,
    })

    try:
        for key in all_keys:
            layout_method = key['layout_method']
            circuit_path = key['circuit_path']
            db_key = '-'.join([layout_method.value, circuit_path.name])
            output_dir = Path('./output') / db_key

            if db_key in db:
                continue

            cmd = (f'{sys.executable} train.py '
                   f'--path {circuit_path} '
                   f'--layout {layout_method.value} '
                   f'--output {output_dir} '
                   f'--hardware {dataset.hardware_name}').split()

            try:
                subprocess.check_call(cmd, cwd=Path.cwd().absolute())
            except Exception as e:
                print(key, 'failed', 'error', e)
                continue

            metrics = read_json(output_dir / 'metrics.json')
            db[db_key] = metrics
            print('Run', db_key, metrics)
    except:
        import traceback

        traceback.print_exc()
    finally:
        db.close()
