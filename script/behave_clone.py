import pickle

import torch.nn
import torch as th
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from contrib.action import ActionAsPolicyInt
from contrib.environs import *
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy
from script.expert_trajectory import EXPERT_WITH_INIT_DIR
from imitation.util import logger as imit_logger

if __name__ == '__main__':
    bs = 128
    # 1. 配置 logger：同时写 stdout / csv / tensorboard
    log_dir = f"../log-bc/tokyo_bigger_net_int_bs={bs}"  # TensorBoard 日志根目录
    logger = imit_logger.configure(log_dir,  # 会自动创建子文件夹
                                   format_strs=["stdout", "csv", "tensorboard"])

    rng = np.random.default_rng(0)
    env = CircuitEnvWithInitialMapping.make(
        input_circuit_path='../data/20Q_gate_Tokyo/circuits/20Q_gate_Tokyo_large_1_10_1.5_no.1.qasm',
        hardware_name='tokyo',
        init=InitialMappingStrategy.IDENTITY,
        a2p=ActionAsPolicyInt(),
    )

    with (EXPERT_WITH_INIT_DIR / 'tokyo.trans').open('rb') as f:
        transitions = pickle.load(f)
    print(f'load transitions {len(transitions)}')

    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        # lr_schedule=lambda x: 1e-5,
        lr_schedule=lambda _: th.finfo(th.float32).max,
        activation_fn=torch.nn.ReLU,
        net_arch=dict(
            vf=[512, 256, 128],
            pi=[512, 256, 128],
        )
    )

    print('Train BC...')
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        custom_logger=logger,
        policy=policy,
        batch_size=bs,
    )
    bc_trainer.train(
        n_epochs=10000, reset_tensorboard=True
    )

    print('Eval policy')
    reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print("Reward:", reward)
