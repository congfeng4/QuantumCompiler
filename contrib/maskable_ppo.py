"""
直接用PPO是很难收敛的，因为非法动作空间十分巨大。
至少需要用MaskablePPO，并且把Action Mask定义好。
☀️🌛🎉🖼🏊🏻🏓✈️🚗
"""
import os
import random
import time
from pathlib import Path
from typing import Callable

import jsons
from contrib.common import QuantumCircuit, IBMQHardwareArchitecture, write_json, get_cnot_num, readable_float_dict, \
    read_json
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, DummyVecEnv, SubprocVecEnv

from contrib.environs import CircuitEnvWithInitialMapping
from contrib.feature_extractor import get_policy_kwargs
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy
from contrib.metrics_callback import MetricEvalCallback, evaluate_policy_for_metrics


def create_vec_env_from_circuits(
        circuit: QuantumCircuit, seqlen: int,
        hardware: IBMQHardwareArchitecture,
        init_strategy: InitialMappingStrategy,
        num_envs: int = 1,
        rs_weight: float = 1,
        final_reward: float = 10,
        gamma: float = 0.99,
        **kwargs,
):
    assert num_envs >= 1
    vec_funcs = []
    init = get_initial_mapping(circuit, hardware, init_strategy)

    def make_func():
        return lambda: CircuitEnvWithInitialMapping(
            input_circuit=circuit,
            hardware=hardware,
            initial_mapping=init,
            L=seqlen,
            reward_shaping_weight=rs_weight,
            final_reward=final_reward,
            gamma=gamma,
        )

    for i in range(num_envs):
        vec_funcs.append(make_func())

    print(f'Create env with {num_envs} circuits')
    env_class = DummyVecEnv if num_envs == 1 else SubprocVecEnv
    return VecMonitor(env_class(vec_funcs))


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
        if p >= plateau:  # plateau 阶段（前期）
            return initial
        else:  # 下降阶段（后期）
            slope = (initial - final) / plateau
            return final + slope * p

    return func


def run_maskable_ppo(
        hardware: IBMQHardwareArchitecture | str,
        circuit_path: Path | str,
        batch_size: int = 128,
        n_steps: int = 1024,
        num_envs: int = 1,
        embed_dim: int = 128,
        reward_shaping_weight: float = 10,
        final_reward: float = 10,
        init_strategy: InitialMappingStrategy = InitialMappingStrategy.SABRE,
        seqlen: int | float = 16,
        total_timesteps: int = 100_000,
        output_dirname: str = None,
        mode: str = 'gru',
        ent_coef: float = 0.01,
        eval_freq: int = 1_000,
        gamma: float = 0.99,
        pretrain: Path = None,
        features_extractor_kwargs: dict = None,
        save_result: bool = True,
        skip_existing: bool = True,
):
    """
    ✅ Run MaskablePPO on a circuit and return the metrics.
    """
    if output_dirname is None:
        output_dirname = 'maskable_ppo'

    output_dir = f'../result/{output_dirname}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if features_extractor_kwargs is None:
        features_extractor_kwargs = {}

    if isinstance(hardware, str):
        hardware = IBMQHardwareArchitecture(hardware)

    qc = QuantumCircuit.from_qasm_file(str(circuit_path))
    gate_len = get_cnot_num(qc)

    if isinstance(seqlen, int):
        seqlen = min(gate_len, seqlen)
    elif isinstance(seqlen, float):
        assert 0 < seqlen < 1
        seqlen = int(seqlen * gate_len)

    print(f'{circuit_path} {gate_len=} {seqlen=}')

    env = create_vec_env_from_circuits(
        circuit=qc,
        hardware=hardware,
        seqlen=seqlen,
        init_strategy=init_strategy,
        num_envs=num_envs,
        rs_weight=reward_shaping_weight,
        final_reward=final_reward,
        gamma=gamma,
    )

    config = dict(
        circuit_path=str(circuit_path),
        batch_size=batch_size,
        n_steps=n_steps,
        num_envs=num_envs,
        embed_dim=embed_dim,
        reward_shaping_weight=reward_shaping_weight,
        final_reward=final_reward,
        init_strategy=init_strategy.name,
        seqlen=seqlen,
        total_timesteps=total_timesteps,
        mode=mode,
        ent_coef=ent_coef,
        gamma=gamma,
        qubit_number=hardware.qubit_number,
    )

    circuit_name = Path(circuit_path).stem
    depth = qc.depth()
    log_name = f'Q={circuit_name}-CX={gate_len}-D={depth}-L={seqlen}-RS={reward_shaping_weight}-FR={final_reward}'
    
    log_dir = f'../log/{output_dirname}'
    result_dir = f"../result/{output_dirname}"
    best_model_path = output_dir + "/models/" + log_name
    result_file = result_dir + f"/Q={circuit_name}-result.json"
    if os.path.exists(result_file) and skip_existing:
        print(f'Result exists: {result_file}')
        return read_json(result_file)

    eval_env = VecMonitor(env)

    metrics_callback = MetricEvalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=10,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        eval_freq=eval_freq,  # 每 10w 步评估一次
        callback_after_eval=None,
        verbose=1,
        deterministic=False,
        use_masking=True,
        best_model_save_path=best_model_path,
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

    learn_start = time.time()
    ppo.learn(
        total_timesteps=total_timesteps,
        tb_log_name=log_name,
        progress_bar=True,
        callback=[eval_callback, metrics_callback],
    )
    learn_end = time.time()

    print('Eval policy')
    metrics = evaluate_policy_for_metrics(ppo, eval_env)
    metrics.update(train_time=learn_end - learn_start)
    metrics = readable_float_dict(metrics)
    result = jsons.dump(dict(config=config, metrics=metrics))
    if save_result:
        write_json(result_file, result)
    return result


def get_short_20Q_circuits(min_gatelen=10, max_gatelen: int = 100):
    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits/').glob('*.qasm'))
    random.shuffle(circuit_list)

    for circuit_path in circuit_list:
        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        gate_len = get_cnot_num(qc)
        if not (min_gatelen <= gate_len <= max_gatelen):
            print(f'Skip {circuit_path} {gate_len=}')
            continue
        print(f'Get {circuit_path} {gate_len=}')
        yield circuit_path


if __name__ == '__main__':
    result = run_maskable_ppo(
        hardware='Tokyo',
        circuit_path=Path('../data/20Q_gate_Tokyo/circuits/20Q_gate_Tokyo_large_1_1_1.5_no.2.qasm'),
        total_timesteps=100,
        save_result=True,
    )
