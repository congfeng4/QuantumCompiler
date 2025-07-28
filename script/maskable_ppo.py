"""
直接用PPO是很难收敛的，因为非法动作空间十分巨大。
至少需要用MaskablePPO，并且把Action Mask定义好。
☀️🌛🎉🖼🏊🏻🏓✈️🚗
"""
import json
import os
import random

import jsons
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecEnv

from contrib.environs import *
from contrib.feature_extractor import HierarchicalCircuitFeaturesExtractor, get_policy_kwargs
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy
from contrib.metrics_callback import CustomMetricsCallback
from contrib.seed import set_all_seeds


M = int(1e6)


def create_vec_env_from_circuits(circuit_paths: list[str], hardware: IBMQHardwareArchitecture,
                                 num_random: int = 10, add_sabre: bool = True, L: int = 10,
                                 sparse_reward: bool = False):
    vec_funcs = []

    def make_func(circ: QuantumCircuit, init):
        return lambda : CircuitEnvWithInitialMapping(circ, hardware, init, L, sparse_reward)

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
    # SubProcVecEnv一开始就内存爆炸了💥
    return VecMonitor(DummyVecEnv(vec_funcs))


def run_maskable_ppo(
        env: VecEnv,
        hardware: IBMQHardwareArchitecture,
        log_name: str,
        batch_size: int = 256,
        n_steps: int = 4000,
        seqlen: int = 15,
        embed_dim: int = None,
        total_timesteps: int = 40_0000,
        output_dirname: str = None,
        mode: str = 'gru',
        ent_coef: float = 0.01,
        eval_env: VecEnv = None,
        eval_freq: int = 1_000,
        pretrain: Path = None,
        early_stop: bool = True,
        **kwargs,
):
    """
    ✅ Run MaskablePPO on a circuit and record the metrics.
    """
    if output_dirname is None:
        output_dirname = 'maskable_ppo'
    output_dir = f'../result/{output_dirname}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    log_dir = f'../log/{output_dirname}'
    if embed_dim is None:
        embed_dim = hardware.qubit_number
    eval_env = eval_env or VecMonitor(env)

    # 回调：连续 10 次评估无提升就停止
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=20,
        min_evals=5,  # 前 5 次评估不计数
        verbose=1
    )
    eval_callback = MaskableEvalCallback(
        eval_env,
        eval_freq=eval_freq,  # 每 10w 步评估一次
        callback_on_new_best=None,  # 可选
        callback_after_eval=stop_callback if early_stop else None,
        verbose=1,
        deterministic=False,
        use_masking=True,
        best_model_save_path=output_dir + "/models/" + log_name,
    )

    ppo = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
        tensorboard_log=log_dir,
        verbose=1,
        ent_coef=ent_coef,
        policy_kwargs=get_policy_kwargs(
            hardware, embed_dim, mode
        )
    ) if pretrain is None else MaskablePPO.load(pretrain, env)
    print(f'Model loaded: {ppo}')

    ppo.learn(
        total_timesteps=total_timesteps,
        tb_log_name=log_name,
        progress_bar=True,
        callback=[CustomMetricsCallback(), eval_callback],
    )

    print('Eval policy')
    reward, _ = evaluate_policy(ppo, eval_env, 1,
                                deterministic=False, use_masking=True)
    print("Reward:", reward)
    metrics = eval_env.get_attr('metrics', [0])[0]

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
    return ppo, data


def run_vec_env():
    bs = 128
    ns = 1000
    embed_dim = 32
    L = 15
    ent_coef = 0.01
    num_train = 1
    num_eval = 2
    hardware_name = 'tokyo'
    data_name = '20Q_gate_Tokyo'
    sparse = False
    # mode = 'transformer'
    mode = 'gru'

    hardware = IBMQHardwareArchitecture(hardware_name)
    circuit_list = list(map(str, Path(f'../data/{data_name}/circuits').glob('*.qasm')))
    random.shuffle(circuit_list)
    env = create_vec_env_from_circuits(circuit_list[:num_train], hardware, L=L, num_random=0, sparse_reward=sparse)
    eval_env = create_vec_env_from_circuits(circuit_list[num_train:num_train+num_eval], hardware, L=L,
                                            num_random=0, sparse_reward=sparse)  # Use sabre only.

    log_name = f'{data_name}-B={bs}-NS={ns}-E={ent_coef}-DS={num_train}-M={mode}-SR={sparse}'

    model, details = run_maskable_ppo(
        env,
        hardware=hardware,
        batch_size=bs,
        n_steps=ns,
        seqlen=L,
        embed_dim=embed_dim,
        ent_coef=ent_coef,
        total_timesteps=int(1e30),
        mode=mode,
        log_name=log_name,
        eval_env=eval_env,
        num_train=num_train,
        data_name=data_name,
    )
    evaluate_all(model, circuit_list, log_name, L, details=details)


def evaluate_all(model, circuit_list, log_name: str, L: int, **kwargs):
    result_file = f'../result/maskable_ppo/{log_name}.json'
    results = []
    init_strategy = InitialMappingStrategy.SABRE
    kwargs.update(init=init_strategy.value)

    for circuit_path in circuit_list:
        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        init = get_initial_mapping(qc, hardware, init_strategy)
        env = CircuitEnvWithInitialMapping(qc, hardware, init, L)
        evaluate_policy(model, env, n_eval_episodes=1, deterministic=False, use_masking=True)
        result = dict(
            circuit_path=circuit_path,
            metrics=env.metrics,
        )
        results.append(result)

    results = dict(results=results, config=kwargs)
    with open(result_file, 'w') as f:
        f.write(json.dumps(jsons.dump(results), indent=4, ensure_ascii=False))



if __name__ == '__main__':
    set_all_seeds()
    run_vec_env()
