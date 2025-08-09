import json
import random

from pathlib import Path
from contrib.environs import *
from script.maskable_ppo import create_vec_env_from_circuits, run_maskable_ppo
import optuna


def objective(trail: optuna.Trial):
    hardware = IBMQHardwareArchitecture('Tokyo')

    bs = 128
    ns = 1024
    embed_dim = trail.suggest_int("embed_dim", 64, 128, log=True)
    L = trail.suggest_int("L", 4, 16, log=True)
    ent_coef = trail.suggest_float('ent_coef', 1e-2, 1e-1, log=True)
    reward_mode = trail.suggest_categorical('reward_mode', [
        RewardMode.GATE_NUM_AND_HEURISTIC_COST,
        RewardMode.HEURISTIC_COST,
    ])

    # circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits/').glob('*.qasm'))
    # random.shuffle(circuit_list)
    result_dir = Path('../result/20Q_gate_Tokyo_0808')
    result_dir.mkdir(parents=True, exist_ok=True)
    circuit_path = Path('../data/20Q_gate_Tokyo/circuits/20Q_gate_Tokyo_large_1_3_1.5_no.2.qasm')
    env = create_vec_env_from_circuits([str(circuit_path)], hardware, num=1,
                                       add_sabre=True,
                                       add_simulated_anealing=False,
                                       add_random=False, L=L, reward_mode=reward_mode)
    circuit_name = Path(circuit_path).stem
    log_name = f'Q={circuit_name}-B={bs}-NS={ns}-E={ent_coef}-D={embed_dim}-L={L}-R={reward_mode.name}'

    run_maskable_ppo(
        log_name=log_name,
        hardware=hardware,
        env=env,
        embed_dim=embed_dim,
        batch_size=bs,
        n_steps=ns,
        seqlen=L,
        mode='gru',
        ent_coef=ent_coef,
        total_timesteps=200_000,
        output_dirname='20Q_gate_Tokyo_0808',
        early_stop=False,
    )
    metrics = env.get_attr('metrics', 0)
    return metrics['cx_ratio']


if __name__ == '__main__':
    study = optuna.create_study(study_name='20Q_gate_Tokyo_0808', direction='minimize')
    study.optimize(objective, n_trials=1000)
