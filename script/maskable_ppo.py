"""
直接用PPO是很难收敛的，因为非法动作空间十分巨大。
至少需要用MaskablePPO，并且把Action Mask定义好。
"""
import json
import random

import jsons
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from contrib.environs import *
from contrib.feature_extractor import HierarchicalCircuitFeaturesExtractor
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy
from contrib.metrics_callback import CustomMetricsCallback


def run_maskable_ppo(
        circuit_path: str,
        hardware_name: str,
        init_strategy: InitialMappingStrategy = InitialMappingStrategy.SABRE,
        batch_size: int = 256,
        n_steps: int = 4000,
        seqlen: int = 15,
        embed_dim: int = None,
        total_timesteps: int = 4_00_000,
        output_dir: str = None,
):
    """
    Run MaskablePPO on a circuit and record the metrics.
    """
    if output_dir is None:
        output_dir = '../result/maskable_ppo/'
    hardware = IBMQHardwareArchitecture(hardware_name)
    if embed_dim is None:
        embed_dim = hardware.qubit_number
    qc = QuantumCircuit.from_qasm_file(circuit_path)
    init = get_initial_mapping(qc, hardware, init_strategy)
    env = CircuitEnvWithInitialMapping(qc, hardware, init, seqlen)
    circuit_name = Path(circuit_path).stem
    log_name = f'qc={circuit_name}-init={init_strategy.value}-D={embed_dim}-L={seqlen}'

    ppo = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
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
                pi=[embed_dim * 4],
                vf=[embed_dim * 4],
            ),
        )
    ).learn(
        total_timesteps=total_timesteps,
        tb_log_name=log_name,
        progress_bar=True,
        callback=CustomMetricsCallback(),
    )

    print('Eval policy')
    reward, _ = evaluate_policy(ppo, env, 10)
    print(circuit_path)
    print("Reward:", reward)
    metrics = env.metrics
    data = jsons.dump(dict(
        circuit_path=circuit_path,
        hardware_name=hardware_name,
        metrics=metrics,
    ))
    json_file = output_dir + '/' + log_name + '.json'
    with open(json_file, 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    bs = 256
    ns = 4000
    embed_dim = 32
    L = 15

    run_maskable_ppo(
        circuit_path='../data/20Q_gate_Tokyo/circuits/20Q_gate_Tokyo_large_1_10_1.5_no.1.qasm',
        hardware_name='tokyo',
        batch_size=bs,
        n_steps=ns,
        seqlen=L,
        embed_dim=embed_dim,
    )
