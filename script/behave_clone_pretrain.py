import pickle

import torch.nn
import torch as th

from imitation.algorithms import bc
from stable_baselines3.common.policies import ActorCriticPolicy

from contrib.environs import *
from contrib.feature_extractor import HierarchicalCircuitFeaturesExtractor
from contrib.pretrain_env import PretrainEnv

from imitation.util import logger as imit_logger


if __name__ == '__main__':
    bs = 128
    log_dir = f"../log/pretrain/exe"
    logger = imit_logger.configure(log_dir,  # 会自动创建子文件夹
                                   format_strs=["stdout", "csv", "tensorboard"])

    hardware = IBMQHardwareArchitecture('tokyo')
    rng = np.random.default_rng(0)
    env = PretrainEnv(N=hardware.qubit_number, L=10)

    with open('../result/pretrain/exe/20Q_gate_Tokyo.trans', 'rb') as f:
        transitions = pickle.load(f)

    print(f'load transitions {len(transitions)}')

    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: th.finfo(th.float32).max,
        activation_fn=torch.nn.LeakyReLU,
        features_extractor_class=HierarchicalCircuitFeaturesExtractor,
        features_extractor_kwargs=dict(
            hardware=hardware,
            embed_dim=32,
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
        n_epochs=10000,
        reset_tensorboard=True,
        progress_bar=False,
    )
