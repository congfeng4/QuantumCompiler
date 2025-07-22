"""
直接用PPO是很难收敛的，因为非法动作空间十分巨大。
至少需要用MaskablePPO，并且把Action Mask定义好。
"""
import random

import torch
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from contrib.environs import *
from contrib.feature_extractor import HierarchicalCircuitFeaturesExtractor
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy


if __name__ == '__main__':
    bs = 256
    ns = 4000
    embed_dim = 20
    L = 15

    log_name = f"B={bs}-E={ns}-D={embed_dim}-L={L}"
    hardware = IBMQHardwareArchitecture('tokyo')
    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits').glob('*.qasm'))
    random.shuffle(circuit_list)

    qc = QuantumCircuit.from_qasm_file(str(circuit_list[0]))
    print(circuit_list[0], qc.depth())
    init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)

    rng = np.random.default_rng(0)
    env = CircuitEnvWithInitialMapping(qc, hardware, init, L)

    ppo = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=ns,
        batch_size=bs,
        tensorboard_log="../log/maskable_ppo/",
        verbose=1,

        policy_kwargs=dict(
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
    ).learn(
        total_timesteps=1_000_000,
        tb_log_name=log_name,
        progress_bar=True,
    )

    print('Eval policy')
    reward, _ = evaluate_policy(ppo, env, 10)
    print("Reward:", reward)
