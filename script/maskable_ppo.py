"""
直接用PPO是很难收敛的，因为非法动作空间十分巨大。
至少需要用MaskablePPO，并且把Action Mask定义好。
"""
import json
import random

import jsons
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

from contrib.environs import *
from contrib.feature_extractor import HierarchicalCircuitFeaturesExtractor
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy
from contrib.metrics_callback import CustomMetricsCallback
from script.seed import set_all_seeds


def run_maskable_ppo(
        circuit_path: str,
        hardware_name: str,
        init_strategy: InitialMappingStrategy = InitialMappingStrategy.SABRE,
        batch_size: int = 256,
        n_steps: int = 4000,
        seqlen: int = 15,
        embed_dim: int = None,
        total_timesteps: int = 40_0000,
        output_dir: str = None,
        mode: str = 'gru',
        ent_coef: float = 0.01,
        idx: int = 0,
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
    log_name = f'qc={circuit_name}-B={batch_size}-NS={n_steps}-E={ent_coef}-I={idx}'

    # 回调：连续 10 次评估无提升就停止
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,  # 前 5 次评估不计数
        verbose=1
    )
    eval_callback = MaskableEvalCallback(
        Monitor(env),
        eval_freq=5_0000,  # 每 10w 步评估一次
        callback_on_new_best=None,  # 可选
        callback_after_eval=stop_callback,
        verbose=1,
        deterministic=False,
        use_masking=True,
    )

    ppo = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
        tensorboard_log="../log/maskable_ppo/",
        verbose=1,
        ent_coef=ent_coef,
        policy_kwargs=dict(
            activation_fn=torch.nn.LeakyReLU,
            features_extractor_class=HierarchicalCircuitFeaturesExtractor,
            features_extractor_kwargs=dict(
                hardware=hardware,
                embed_dim=embed_dim,
                mode=mode,
            ),
            net_arch=dict(
                pi=[embed_dim * 2],
                vf=[embed_dim * 2],
            ),
        )
    )
    # ppo.policy = ppo.policy.double()
    ppo.learn(
        total_timesteps=total_timesteps,
        tb_log_name=log_name,
        progress_bar=True,
        callback=[CustomMetricsCallback(), eval_callback],
    )

    print('Eval policy')
    reward, _ = evaluate_policy(ppo, Monitor(env), 10,
                                deterministic=False, use_masking=True)
    print(circuit_path)
    print("Reward:", reward)
    metrics = env.metrics
    data = jsons.dump(dict(
        circuit_path=circuit_path,
        hardware_name=hardware_name,
        metrics=metrics,
        mode=mode,
        batch_size=batch_size,
        total_timesteps=total_timesteps,
        init_strategy=init_strategy.value,
        seqlen=seqlen,
        embed_dim=embed_dim,
        n_steps=n_steps,
    ))
    json_file = output_dir + '/' + log_name + '.json'
    with open(json_file, 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    # set_all_seeds()

    bs = 128
    ns = 4000
    embed_dim = 32
    L = 15
    times_per_circuit = 10
    ent_coef = 0.01
    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits').glob('*.qasm'))
    random.shuffle(circuit_list)

    # 20Q_gate_Tokyo_large_2_3_1.5_no.7

    for path in circuit_list:
        for i in range(times_per_circuit):
    # i = 0
    # path = '../data/20Q_gate_Tokyo/circuits/20Q_gate_Tokyo_large_1_25_1.5_no.9.qasm'
            run_maskable_ppo(
                circuit_path=str(path),
                hardware_name='tokyo',
                batch_size=bs,
                n_steps=ns,
                seqlen=L,
                mode='gru',
                ent_coef=ent_coef,
                total_timesteps=40_0000,
                idx=i,
            )
