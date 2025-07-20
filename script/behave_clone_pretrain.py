import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch.nn
import torch as th

from imitation.algorithms import bc
from stable_baselines3.common.policies import ActorCriticPolicy

from contrib.feature_extractor import HierarchicalCircuitFeaturesExtractor
from contrib.pretrain_env import PretrainEnv, TrajectoryCollector
import shutil
from imitation.util import logger as imit_logger

from hamap import IBMQHardwareArchitecture

from imitation.algorithms.bc import BehaviorCloningLossCalculator


def main():
    bs = 256
    log_dir = f"../log/pretrain/exe-swap"
    shutil.rmtree(log_dir, ignore_errors=True)

    logger = imit_logger.configure(log_dir,  # 会自动创建子文件夹
                                   format_strs=["stdout", "csv", "tensorboard"])

    hardware = IBMQHardwareArchitecture('tokyo')
    rng = np.random.default_rng(0)
    env = PretrainEnv(N=hardware.qubit_number, L=10)

    train_trans = TrajectoryCollector(N=hardware.qubit_number, L=10,
                                          outdir=Path('../result/pretrain/exe_swap'),
                                          prefix='20Q_gate_Tokyo_train').load()

    val_trans = TrajectoryCollector(N=hardware.qubit_number, L=10,
                                        outdir=Path('../result/pretrain/exe_swap'),
                                        prefix='20Q_gate_Tokyo_val').load()

    loss_calc = BehaviorCloningLossCalculator(0, 0)
    embed_dim = 64

    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: th.finfo(th.float32).max,
        activation_fn=torch.nn.LeakyReLU,
        features_extractor_class=HierarchicalCircuitFeaturesExtractor,
        features_extractor_kwargs=dict(
            hardware=hardware,
            embed_dim=embed_dim,
        ),
        net_arch=dict(
            pi=[embed_dim * 2],
            vf=[embed_dim * 2],
        ),
    )

    print('Train BC...')
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=train_trans,
        rng=rng,
        custom_logger=logger,
        policy=policy,
        batch_size=bs,
    )

    # 2) epoch-end 回调
    current_epoch = 0
    log_interval = 100

    def on_epoch_end():
        nonlocal current_epoch
        if current_epoch % log_interval == 0:
            with torch.no_grad():
                metrics = loss_calc(bc_trainer.policy, val_trans.obs, val_trans.acts)
            # 写入 bc_trainer 的 logger，前缀 eval/
            bc_trainer.logger.record("eval/bce", float(metrics.loss))
            bc_trainer.logger.record("eval/accuracy", float(metrics.prob_true_act))
            bc_trainer.logger.dump(current_epoch)
        current_epoch += 1

    bc_trainer.train(
        n_epochs=2_0000,
        log_interval=log_interval,
        reset_tensorboard=True,
        on_epoch_end=on_epoch_end,
        progress_bar=False,
    )

    policy.save("../log/pretrain/model.zip")


if __name__ == '__main__':
    main()
