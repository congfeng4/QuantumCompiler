"""
直接用PPO是很难收敛的，因为非法动作空间十分巨大。
至少需要用MaskablePPO，并且把Action Mask定义好。
☀️🌛🎉🖼🏊🏻🏓✈️🚗
"""
import json
import os
import random
from pathlib import Path
from typing import Callable

import jsons
import pandas as pd
from contrib.common import QuantumCircuit, IBMQHardwareArchitecture, write_json
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, DummyVecEnv

from contrib.environs import CircuitEnvWithInitialMapping
from contrib.feature_extractor import get_policy_kwargs
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy
from contrib.metrics_callback import MetricEvalCallback, evaluate_policy_for_metrics
from contrib.seed import set_all_seeds

M = int(1e6)


def create_vec_env_from_circuits(circuit_paths: list[QuantumCircuit], hardware: IBMQHardwareArchitecture,
                                 num: int = 1, L: int = 10,
                                 add_sabre: bool = True,
                                 add_random: bool = False,
                                 add_simulated_anealing=False,
                                 **kwargs,
                                 ):
    vec_funcs = []

    def make_func(circ: QuantumCircuit, init):
        return lambda: CircuitEnvWithInitialMapping(circ, hardware, init, L, **kwargs)

    for qc in circuit_paths:
        for i in range(num):
            if add_random:
                init = get_initial_mapping(qc, hardware, InitialMappingStrategy.RANDOM)
                vec_funcs.append(make_func(qc, init))

            if add_sabre:
                init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)
                vec_funcs.append(make_func(qc, init))

            if add_simulated_anealing:
                init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SIMULATE_ANNEALING)
                vec_funcs.append(make_func(qc, init))

    print(f'Create env with {len(circuit_paths)} circuits')
    # SubProcVecEnv一开始就内存爆炸了💥
    return VecMonitor(DummyVecEnv(vec_funcs))


def get_max_ep_len(env, model):
    episode_rewards, episode_lengths = evaluate_policy(model, env, deterministic=False, use_masking=True,
                                            return_episode_rewards=True)
    return max(episode_lengths)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def multistep_schedule(initial: float, milestones=None, gamma=0.3):
    """milestones 用 progress_remaining 的阈值"""
    if milestones is None:
        milestones = [0.5, 0.75]

    def func(p: float) -> float:
        factor = 1.0
        for m in milestones:
            if p <= m:
                factor *= gamma
        return initial * factor
    return func


def piecewise_linear(initial: float, plateau: float = 0.5, final: float = 1e-5):
    """
    plateau 以内保持 initial，之后线性降到 final。
    progress_remaining 从 1→0。
    """
    def func(p: float) -> float:
        if p >= plateau:          # plateau 阶段（前期）
            return initial
        else:                     # 下降阶段（后期）
            slope = (initial - final) / plateau
            return final + slope * p
    return func


def run_maskable_ppo(
        env: VecEnv,
        hardware: IBMQHardwareArchitecture,
        log_name: str,
        batch_size: int = 256,
        n_steps: int = 4000,
        embed_dim: int = None,
        total_timesteps: int = 40_0000,
        output_dirname: str = None,
        mode: str = 'gru',
        ent_coef: float = 0.01,
        eval_env: VecEnv = None,
        eval_freq: int = 1_000,
        gamma: float = 0.99,
        pretrain: Path = None,
        early_stop: bool = True,
        features_extractor_kwargs: dict = None,
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
    if features_extractor_kwargs is None:
        features_extractor_kwargs = {}

    eval_env = eval_env or VecMonitor(env)

    # 回调：连续 10 次评估无提升就停止
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=20,
        min_evals=5,  # 前 5 次评估不计数
        verbose=1
    )

    metrics_callback = MetricEvalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=10,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        eval_freq=eval_freq,  # 每 10w 步评估一次
        # callback_after_eval=metrics_callback,
        verbose=1,
        deterministic=False,
        use_masking=True,
        best_model_save_path=output_dir + "/models/" + log_name,
    )

    ppo = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=n_steps,
        # plateau: Remaining steps.
        learning_rate=piecewise_linear(3e-4, plateau=0.4, final=1e-5),  # 防止后期不稳定
        clip_range=piecewise_linear(0.2, plateau=0.4, final=0.01),  # 防止后期不稳定
        batch_size=batch_size,
        tensorboard_log=log_dir,
        verbose=1,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs=get_policy_kwargs(
            hardware, embed_dim, mode, **features_extractor_kwargs
        ),
    ) if pretrain is None else MaskablePPO.load(pretrain, env)
    ppo.tensorboard_log = log_dir
    print(f'Model loaded: {ppo}')

    # max_ep_len = get_max_ep_len(env, ppo)
    # env.set_attr("max_ep_len", max_ep_len)
    # print(f'Max ep len: {max_ep_len}')

    ppo.learn(
        total_timesteps=total_timesteps,
        tb_log_name=log_name,
        progress_bar=True,
        callback=[eval_callback, metrics_callback],
    )

    print('Eval policy')
    metrics = evaluate_policy_for_metrics(ppo, eval_env)
    metrics_file = output_dir + f'/{log_name}/metrics.json'
    write_json(metrics_file, metrics)
    return metrics


def run_vec_env(
        bs=128,
        ns=1000,
        embed_dim=128,
        L=15,
        ent_coef=0.01,
        num_train=199,
        hardware_name='tokyo',
        data_name='20Q_gate_Tokyo',
        mode='gru',
        output_dirname='maskable_ppo_v3_pretrain',
        pretrain: bool = False,
        swap_only: bool = False,
        total_timesteps: int = int(1e10),
):
    hardware = IBMQHardwareArchitecture(hardware_name)
    circuit_list = list(Path(f'../data/{data_name}/circuits').glob('*.qasm'))
    if num_train == -1:
        num_train = len(circuit_list)
    random.shuffle(circuit_list)
    circuit_list = circuit_list[:num_train]
    env = create_vec_env_from_circuits(list(map(str, circuit_list)), hardware, L=L, num=0, swap_only=swap_only)

    log_name = f'{data_name}-B={bs}-NS={ns}-E={ent_coef}-M={mode}-D={embed_dim}-SO={swap_only}'
    if pretrain:
        pretrain_path = Path(f'../result/{output_dirname}/models/{log_name}/best_model.zip')
    else:
        pretrain_path = None

    metrics, circuits = run_maskable_ppo(
        env,
        hardware=hardware,
        batch_size=bs,
        n_steps=ns,
        embed_dim=embed_dim,
        ent_coef=ent_coef,
        output_dirname=output_dirname,
        total_timesteps=total_timesteps,
        mode=mode,
        log_name=log_name,
        eval_env=None,
        pretrain=pretrain_path,
    )
    assert len(metrics) == num_train, (len(metrics), num_train)
    metrics_file = f'../result/{output_dirname}/{log_name}.xlsx'
    df = pd.DataFrame(data=metrics)
    df['circuit'] = circuits
    df.to_excel(metrics_file, index=False)
    print(f'Results {metrics}')
    print(f'Result saved to {metrics_file}')


def evaluate_all(model, hardware, circuit_list, log_name: str, L: int, **kwargs):
    result_file = f'../result/maskable_ppo/{log_name}.json'
    results = []
    init_strategy = InitialMappingStrategy.SABRE
    kwargs.update(init=init_strategy.value)

    for circuit_path in circuit_list:
        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        init = get_initial_mapping(qc, hardware, init_strategy)
        env = CircuitEnvWithInitialMapping(qc, circuit_path, hardware, init, L)
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
