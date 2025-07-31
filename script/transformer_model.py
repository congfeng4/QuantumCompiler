import random
import json
import random

import jsons
from gymnasium.utils.env_checker import check_env
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

from contrib.environs import *
from contrib.feature_extractor import HierarchicalCircuitFeaturesExtractor
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy
from contrib.metrics_callback import CustomMetricsCallback
from contrib.seed import set_all_seeds
from script.maskable_ppo import create_vec_env_from_circuits, run_maskable_ppo


def run_env():
    bs = 200
    ns = 1000
    embed_dim = 32
    L = 15
    NR = 0
    ent_coef = 1e-2

    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits').glob('*.qasm'))
    random.shuffle(circuit_list)

    hardware = IBMQHardwareArchitecture('tokyo')

    for circuit_path in circuit_list:
        env = create_vec_env_from_circuits([str(circuit_path)], hardware, NR, L=L)
        eval_env = create_vec_env_from_circuits([str(circuit_path)], hardware, num=0, L=L)
        circuit_name = Path(circuit_path).stem
        log_name = f'Q={circuit_name}-B={bs}-NS={ns}-E={ent_coef}-NR={NR}'

        run_maskable_ppo(
            log_name=log_name,
            hardware=hardware,
            env=env,
            eval_env=eval_env,
            embed_dim=embed_dim,
            batch_size=bs,
            n_steps=ns,
            seqlen=L,
            mode='transformer',
            ent_coef=ent_coef,
            total_timesteps=10_0000,
            early_stop=False,
            pretrain=None,
        )

    # 20Q_gate_Tokyo_large_2_3_1.5_no.7


if __name__ == '__main__':
    run_env()
