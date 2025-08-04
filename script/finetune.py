import json

from sb3_contrib import MaskablePPO

from contrib.ha_traj import ha_baseline
from contrib.seed import set_all_seeds
import random

from pathlib import Path
from contrib.environs import *
from script.maskable_ppo import create_vec_env_from_circuits, run_maskable_ppo


def run_env():
    bs = 64
    ns = 1000
    embed_dim = 128
    L = 8
    ent_coef = 0.01

    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits/').glob('*.qasm'))
    # random.shuffle(circuit_list)
    result_dir = Path('../result/use_distance_matrix')
    result_dir.mkdir(parents=True, exist_ok=True)

    hardware = IBMQHardwareArchitecture('Tokyo')
    # circuit_path = '../data/20Q_gate_Tokyo/circuits/20Q_gate_Tokyo_large_1_25_1.5_no.1.qasm'

    for circuit_path in circuit_list:
        env = create_vec_env_from_circuits([str(circuit_path)], hardware, num=1, add_sabre=True,
                                           add_random=False, L=L)
        circuit_name = Path(circuit_path).stem
        log_name = f'Q={circuit_name}-B={bs}-NS={ns}-E={ent_coef}-D={embed_dim}-L={L}'

        metrics = env.get_attr('metrics_baseline', 0)
        print(circuit_name, 'HA', metrics)
        metrics_file = result_dir / (log_name + '.json')

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
            output_dirname='use_distance_matrix',
            early_stop=False,
            # pretrain=Path('../result/maskable_ppo_v3_pretrain/models/20Q_gate_Tokyo-B=128-NS=1000-E=0.01-DS=199-M=gru-D=128/best_model.zip')
        )
        metrics = env.get_attr('metrics', 0)
        print(circuit_name, 'HA & PPO', metrics)
        metrics_file.write_text(json.dumps(metrics))


if __name__ == '__main__':
    run_env()
