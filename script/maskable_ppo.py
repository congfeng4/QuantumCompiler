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


def create_vec_env_from_circuits(circuit_paths: list[str], hardware: IBMQHardwareArchitecture,
                                 num_random: int = 10, add_sabre: bool = True, L: int = 10):
    vec_funcs = []

    def make_func(circ: QuantumCircuit, init):
        return lambda : CircuitEnvWithInitialMapping(circ, hardware, init, L)

    for path in circuit_paths:
        print(f'Path {path}')
        qc = QuantumCircuit.from_qasm_file(path)
        if add_sabre:
            init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)
            vec_funcs.append(make_func(qc, init))
        for i in range(num_random):
            init = get_initial_mapping(qc, hardware, InitialMappingStrategy.RANDOM)
            vec_funcs.append(make_func(qc, init))

    print(f'Create env with {len(circuit_paths)} circuits')
    return VecMonitor(DummyVecEnv(vec_funcs))


def run_maskable_ppo(
        env,
        hardware: IBMQHardwareArchitecture,
        log_name: str,
        batch_size: int = 256,
        n_steps: int = 4000,
        seqlen: int = 15,
        embed_dim: int = None,
        total_timesteps: int = 40_0000,
        output_dir: str = None,
        mode: str = 'gru',
        ent_coef: float = 0.01,
        eval_env = None,
        **kwargs,
):
    """
    Run MaskablePPO on a circuit and record the metrics.
    """
    if output_dir is None:
        output_dir = '../result/maskable_ppo/'
    # hardware = IBMQHardwareArchitecture(hardware_name)
    if embed_dim is None:
        embed_dim = hardware.qubit_number
    eval_env = eval_env or Monitor(env)

    # 回调：连续 10 次评估无提升就停止
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,  # 前 5 次评估不计数
        verbose=1
    )
    eval_callback = MaskableEvalCallback(
        eval_env,
        eval_freq=1_0000,  # 每 10w 步评估一次
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
    reward, _ = evaluate_policy(ppo, eval_env, 10,
                                deterministic=False, use_masking=True)
    print("Reward:", reward)
    metrics = env.metrics
    data = jsons.dump(dict(
        metrics=metrics,
        mode=mode,
        batch_size=batch_size,
        total_timesteps=total_timesteps,
        seqlen=seqlen,
        embed_dim=embed_dim,
        n_steps=n_steps,
        ent_coef=ent_coef,
        **kwargs,
    ))
    json_file = output_dir + '/' + log_name + '.json'
    with open(json_file, 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


def run_vec_env():
    bs = 128
    ns = 1000
    embed_dim = 32
    L = 15
    ent_coef = 0.01
    mode = 'gru'
    num_train = 5
    num_eval = 2
    hardware_name = 'tokyo'
    data_name = '20Q_gate_Tokyo'

    hardware = IBMQHardwareArchitecture(hardware_name)
    circuit_list = list(map(str, Path(f'../data/{data_name}/circuits').glob('*.qasm')))
    random.shuffle(circuit_list)
    env = create_vec_env_from_circuits(circuit_list[:num_train], hardware, L=L)
    eval_env = create_vec_env_from_circuits(circuit_list[num_train:num_train+num_eval], hardware, L=L)

    log_name = f'{data_name}-B={bs}-NS={ns}-E={ent_coef}-DS={num_train}'

    run_maskable_ppo(
        env,
        hardware=hardware,
        batch_size=bs,
        n_steps=ns,
        seqlen=L,
        embed_dim=embed_dim,
        ent_coef=ent_coef,
        total_timesteps=40_0000,
        mode=mode,
        log_name=log_name,
        eval_env=eval_env,
        num_train=num_train,
        data_name=data_name,
    )


def run_env():
    bs = 128
    ns = 4000
    embed_dim = 32
    L = 15
    times_per_circuit = 2
    ent_coef = 0.01
    init_strategy = InitialMappingStrategy.RANDOM
    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits').glob('*.qasm'))
    random.shuffle(circuit_list)

    for circuit_path in circuit_list:
        for i in range(times_per_circuit):
            qc = QuantumCircuit.from_qasm_file(str(circuit_path))
            init = get_initial_mapping(qc, hardware, init_strategy)
            env = CircuitEnvWithInitialMapping(qc, hardware, init, L)
            circuit_name = Path(circuit_path).stem
            log_name = f'qc={circuit_name}-B={bs}-NS={ns}-E={ent_coef}-I={i}'

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
                total_timesteps=40_0000,
            )

    # 20Q_gate_Tokyo_large_2_3_1.5_no.7


if __name__ == '__main__':
    # set_all_seeds()

    run_vec_env()
