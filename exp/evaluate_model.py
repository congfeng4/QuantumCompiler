from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from contrib.environs import *
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy, run_ha
from contrib.seed import set_all_seeds
import pandas as pd

from script.maskable_ppo import create_vec_env_from_circuits


def evaluate_model_and_baseline(
        dataname: str,
        model_path: Path,
        hardware_name: str,
        num_repeat: int = 10,
        L: int = 15,
        seed: int = 42,
):
    set_all_seeds(seed)
    init_strategy = InitialMappingStrategy.SABRE
    circuit_list = list(Path(f'../data/{dataname}/circuits') .glob('*.qasm'))
    model = MaskablePPO.load(model_path)
    print(f'Model loaded: {model}')
    hardware = IBMQHardwareArchitecture(hardware_name)
    venv = create_vec_env_from_circuits(list(map(str, circuit_list)), hardware, 0, L=L)
    evaluate_policy(model, venv, 1, use_masking=True, deterministic=True)
    return

    result_list = []
    res = dict(dataname=dataname, hardware_name=hardware_name)

    for path in circuit_list:
        res['circuit'] = path.stem
        qc = QuantumCircuit.from_qasm_file(str(path))
        init_mapping = get_initial_mapping(qc, hardware, init_strategy)
        env = CircuitEnvWithInitialMapping(qc, hardware, init_mapping, L)
        print(f'Run {path.stem}')

        for i in range(num_repeat):
            res['idx'] = i
            res['method'] = 'PPO'
            evaluate_policy(model, env, 1, use_masking=True, deterministic=True)
            res['metrics'] = env.metrics
            result_list.append(res.copy())

            res['method'] = 'HA'
            ha_res = run_ha(str(path), hardware_name, init_strategy)
            res['metrics'] = ha_res['metrics']
            result_list.append(res.copy())

    return pd.DataFrame.from_records(result_list)


if __name__ == '__main__':

    evaluate_model_and_baseline(
        dataname='20Q_gate_Tokyo',
        model_path=Path('../result/maskable_ppo_v3_pretrain/models/20Q_gate_Tokyo-B=128-NS=1000-E=0.01-DS=199-M=gru-D=128/best_model.zip'),
        hardware_name='tokyo',
        num_repeat=1,
        L=15,
    ).to_excel(Path('../result/20Q_gate_Tokyo.xlsx'))

