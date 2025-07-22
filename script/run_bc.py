"""
Test behaviour clone on our feature extractor.
"""
import json
import random

import gymnasium
import jsons
import more_itertools

from imitation.data.rollout import flatten_trajectories
from qiskit import QuantumCircuit

from contrib.ha_traj import InitialMappingStrategy, get_initial_mapping
from pathlib import Path

import numpy as np
import torch.nn
import torch as th
from dataclasses import dataclass
from imitation.algorithms import bc
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel

from contrib.environs import CircuitEnvWithInitialMapping
from contrib.feature_extractor import HierarchicalCircuitFeaturesExtractor
from contrib.ha_traj import InitialMappingStrategy
from contrib.pretrain_env import PretrainEnv, TrajectoryCollector, ha_mapping
import shutil
from imitation.util import logger as imit_logger
from hamap import IBMQHardwareArchitecture

from imitation.algorithms.bc import BehaviorCloningLossCalculator


def early_stop_callback(bc_trainer, val_trans):
    loss_calc = BehaviorCloningLossCalculator(0, 0)

    with torch.no_grad():
        metrics = loss_calc(bc_trainer.policy, val_trans.obs, val_trans.acts)
    val_bce = float(metrics.prob_true_act)
    if abs(val_bce - 1) < 1e-5:  # 必须拟合到1
        print('Reach 100% prob_true_act. Early STOPPED')
        raise KeyboardInterrupt


def rollout_policy(policy: ActorCriticPolicy, env: gymnasium.Env):
    state, _ = env.reset()
    done, truncated = False, False
    steps = 0
    total_reward = 0
    info = {}
    while not done and not truncated:
        action, _ = policy.predict(state, deterministic=True)
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    print(f'Steps {steps}, total reward {total_reward}')
    if truncated:
        print('Trucated')
    return info['metrics']


def fit_policy_with_bc(log_dir: str, hardware: IBMQHardwareArchitecture,
                       circuit_paths: list[Path], L: int = 10,
                       embed_dim: int = 64, batch_size: int = 128, n_epochs: int = 1_0000,
    init: InitialMappingStrategy = InitialMappingStrategy.SABRE):
    """
    Fit our policy on a few trajectories using BC and then evaluate the policy
    on these trajectories.
    """
    N = hardware.qubit_number
    collector = TrajectoryCollector(N=N, L=L)
    circuits = [QuantumCircuit.from_qasm_file(str(p)) for p in circuit_paths]
    init_mappings = [get_initial_mapping(circuit, hardware, init) for circuit in circuits]

    # Collect training data -- Expert trajectories.
    for circuith, initial_mapping in zip(circuits, init_mappings):
        ha_mapping(
            collector=collector,
            quantum_circuit=circuith,
            initial_mapping=initial_mapping,
            hardware=hardware,
        )

    metrics_ha_list = collector.metrics_list
    transitions = flatten_trajectories(collector.trajectories)
    print(f'Collect transitions {len(transitions)}')

    env = PretrainEnv(N=N, L=L)
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
    rng = np.random.default_rng(0)
    logger = imit_logger.configure(log_dir, format_strs=["stdout", "csv", "tensorboard"])

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        custom_logger=logger,
        policy=policy,
        batch_size=batch_size,
    )
    # Training.
    try:
        bc_trainer.train(
            n_epochs=n_epochs,
            reset_tensorboard=True,
            progress_bar=False,
            on_epoch_end=lambda : early_stop_callback(bc_trainer, transitions),
        )
    except KeyboardInterrupt:
        pass
    policy.save(log_dir + "/model.zip")

    print('Inference')
    # Inference.
    policy.eval()
    metrics_bc_list = []
    for circuit, initial_mapping, metrics_ha in zip(circuits, init_mappings, metrics_ha_list):
        env = CircuitEnvWithInitialMapping(
            input_circuit=circuit, hardware=hardware, initial_mapping=initial_mapping, L=L
        )
        metrics_bc = rollout_policy(policy, env)
        metrics_bc_list.append(metrics_bc)
        print(f'Metrics HA: {metrics_ha}')
        print(f'Metrics BC: {metrics_bc}')

    data = jsons.dump(dict(
        hardware=hardware.name,
        circuits=list(map(str, circuit_paths)),
        D=embed_dim,
        N=N, L=L, n_epochs=n_epochs, init=init.value,
        metrics_ha=metrics_ha_list,
        metrics_bc=metrics_bc_list,
    ))
    with open(log_dir + "/data.json", 'w') as f:
        json.dump(data, f, indent=4)

    return metrics_ha_list, metrics_bc_list


def run_20Q_gate_Tokyo():

    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits').glob('*.qasm'))
    random.shuffle(circuit_list)
    print(f'Load {len(circuit_list)} circuits')

    hardware = IBMQHardwareArchitecture('tokyo')

    for test_id, circuit_paths in enumerate(more_itertools.chunked(circuit_list, n=1)):

        metrics_ha_list, metrics_bc_list = fit_policy_with_bc(
            log_dir=f'../log/test/bc/20Q_gate_Tokyo/{test_id:04}',
            hardware=hardware,
            circuit_paths=circuit_paths,
            n_epochs=2000,
        )
        for metrics_bc, metrics_ha in zip(metrics_bc_list, metrics_ha_list):
            for key in metrics_bc:
                val_1 = metrics_ha[key]
                val_2 = metrics_bc[key]
                assert abs(val_1 - val_2) < 1e-5, f"{key=}, {val_1=}, {val_2=}"


if __name__ == '__main__':
    run_20Q_gate_Tokyo()
