from contrib.seed import set_all_seeds
import json
import random

import jsons
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

from contrib.environs import *
from script.maskable_ppo import create_vec_env_from_circuits, run_maskable_ppo


def run_env():
    bs = 128
    ns = 1000
    embed_dim = 256
    L = 15
    ent_coef = 0.01

    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits').glob('*.qasm'))
    # random.shuffle(circuit_list)

    hardware = IBMQHardwareArchitecture('tokyo')

    for circuit_path in circuit_list:
        env = create_vec_env_from_circuits([str(circuit_path)], hardware, num_random=0, L=L)
        circuit_name = Path(circuit_path).stem
        log_name = f'Q={circuit_name}-B={bs}-NS={ns}-E={ent_coef}-D={embed_dim}'

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
            total_timesteps=50_0000,
            output_dirname='maskable_ppo_v2',
            early_stop=False,
            # pretrain=Path('../result/maskable_ppo/models/20Q_gate_Tokyo-B=128-NS=1000-E=0.01-DS=160/best_model.zip')
        )

    # 20Q_gate_Tokyo_large_2_3_1.5_no.7


if __name__ == '__main__':
    set_all_seeds()
    run_env()
