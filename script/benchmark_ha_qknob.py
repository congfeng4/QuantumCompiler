"""
在QKNOB数据集上对HA算法进行测评，主要参数是几种不同的初始映射策略。
"""
import jsons
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from contrib.common import RESULT_DIR, get_all_qknob_circuit_paths, get_hardware_name
from contrib.ha_traj import InitialMappingStrategy, run_ha
from pathlib import Path
import pandas as pd
from tqdm import tqdm


HA_RESULT_DIR = RESULT_DIR / 'ha'


def func(path: Path, init: InitialMappingStrategy, data_name: str):
    file = f'{path.stem}-init={init.value}-data={data_name}.json'
    out_path = HA_RESULT_DIR / file
    if out_path.exists():
        return {}
    res = run_ha(
        circ_path=str(path),
        hardware_name=get_hardware_name(data_name),
        initial_mapping_strategy=init,
    )
    out_path.write_text(jsons.dumps(res, jdkwargs=dict(indent=4, ensure_ascii=False)))
    record = dict(**res['metrics'], data_name=data_name, init=init.value)
    return record


def _func(args): return func(*args)


def get_tasks():
    initial_mapping_strategies = [
        InitialMappingStrategy.RANDOM,
        InitialMappingStrategy.IDENTITY,
        InitialMappingStrategy.SABRE,
        InitialMappingStrategy.SIMULATE_ANNEALING,
    ]

    for data_name, circuit_paths in sorted(get_all_qknob_circuit_paths().items(),
                                           key=lambda x: x[0], reverse=True):
        for path in circuit_paths:
            for init in initial_mapping_strategies:
                yield path, init, data_name


if __name__ == '__main__':

    with ProcessPoolExecutor(4) as executor:
        records = list(executor.map(_func, tqdm(get_tasks())))
        records = list(filter(None, records))
        df = pd.DataFrame.from_records(records)
        df.to_excel(HA_RESULT_DIR / 'result.xlsx')
