"""
在QKNOB数据集上对HA算法进行测评，主要参数是几种不同的初始映射策略。
"""
import jsons
from concurrent.futures import ProcessPoolExecutor
from contrib.common import get_all_qknob_circuit_paths, get_hardware_name
from contrib.ha_traj import InitialMappingStrategy, run_ha, HA_RESULT_DIR
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def func(path: Path, init: InitialMappingStrategy, data_name: str):
    file = f'{path.stem}-init={init.value}-data={data_name}.json'
    out_path = HA_RESULT_DIR / file
    if out_path.exists():
        return
    print(path, data_name, out_path)
    res = run_ha(
        circ_path=str(path),
        hardware_name=get_hardware_name(data_name),
        initial_mapping_strategy=init,
    )
    out_path.write_text(jsons.dumps(res, jdkwargs=dict(indent=4, ensure_ascii=False)))


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
        if 'rochester' not in data_name.lower():
            continue
        if 'sycamore' in data_name.lower():
            continue
        print(data_name)
        for path in circuit_paths:
            print(path)
            for init in initial_mapping_strategies:
                yield path, init, data_name


if __name__ == '__main__':

    with ProcessPoolExecutor(1) as executor:
        list(executor.map(_func, tqdm(get_tasks())))
