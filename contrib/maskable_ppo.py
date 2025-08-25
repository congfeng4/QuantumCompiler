"""
直接用PPO是很难收敛的，因为非法动作空间十分巨大。
至少需要用MaskablePPO，并且把Action Mask定义好。
☀️🌛🎉🖼🏊🏻🏓✈️🚗
"""
import math
import os
import random
import time
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional, Literal
from pprint import pprint

import jsons
import torch.cuda
from stable_baselines3.common.env_util import make_vec_env

from contrib.common import QuantumCircuit, IBMQHardwareArchitecture, write_json, get_cnot_num, readable_float_dict, \
    read_json, show_mapping, Qubit, Unit, get_circuit_depth, read_circuit

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, DummyVecEnv, SubprocVecEnv

from contrib.environs import CircuitEnvWithInitialMapping
from contrib.feature_extractor import get_policy_kwargs
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


def average_metrics(metrics_list):
    """
    Calculate the average of a list of metrics.

    :param metrics_list: List of dictionaries containing metrics
    :return: Dictionary with averaged metrics
    """
    if not metrics_list:
        return {}

    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = sum(metric[key] for metric in metrics_list) / len(metrics_list)

    return avg_metrics


def evaluate_policy_for_metrics(model, eval_env):
    metrics_list = []
    
    evaluate_policy(model, eval_env, n_eval_episodes=1, use_masking=True, deterministic=False)
    metrics = eval_env.get_attr('metrics')
    metrics_list.extend(metrics)

    return average_metrics(metrics_list)


class MetricEvalCallback(BaseCallback):
    model: MaskablePPO

    def __init__(self, eval_env, eval_freq=10000):
        # deterministic=True，早期评估非常慢，几乎卡死。
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            metrics = evaluate_policy_for_metrics(self.model, self.eval_env)

            for key, value in metrics.items():
                self.logger.record(f"metric/{key}", round(value, 2))

        return True


def create_vec_env_from_circuits(
        circuit: QuantumCircuit, seqlen: int,
        hardware: IBMQHardwareArchitecture,
        init: dict[Qubit, int],
        num_envs: int = 1,
        use_subproc: bool = False,
        rs_weight: float = 1,
        gamma: float = 0.99,
):
    assert num_envs >= 1

    def make_func():
        return lambda: CircuitEnvWithInitialMapping(
            input_circuit=circuit,
            hardware=hardware,
            initial_mapping=init,
            L=seqlen,
            reward_shaping_weight=rs_weight,
            gamma=gamma,
        )

    print(f'Create env with {num_envs} circuits {use_subproc=}')

    return make_vec_env(
        env_id=make_func(),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv if use_subproc else DummyVecEnv,
    )


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
        n_steps: int = 32 * Unit.K,
        eval_freq: int = Unit.K,
        embed_dim: int = 128,
        reward_shaping_weight: float = 10,
        init_strategy: InitialMappingStrategy = InitialMappingStrategy.SABRE,
        seqlen: int | float = 16,
        num_epochs: int = 100,
        output_dirname: str = None,
        mode: str = 'gru',
        ent_coef: float = 0.01,
        gamma: float = 0.99,
        pretrain: Path = None,
        features_extractor_kwargs: dict = None,
        save_result: bool = True,
        save_model: bool = False,
        skip_existing: bool = True,
        n_eval_episodes: int = 10,
        max_no_improvement_evals=100,
        num_envs: int = None,
        learning_rate: float = 3e-4,
):
    """
    ✅ Run MaskablePPO on a circuit and return the metrics.
    """
    num_envs = num_envs or max(os.cpu_count() // 4, 8)
    while n_steps % num_envs != 0:
        num_envs += 1

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
    init = get_initial_mapping(qc, hardware, init_strategy)
    use_subproc = torch.cuda.is_available()  # On GPU server, use subproc to make full use of GPUs.

    env = create_vec_env_from_circuits(
        circuit=qc,
        hardware=hardware,
        seqlen=seqlen,
        init=init,
        num_envs=num_envs,
        rs_weight=reward_shaping_weight,
        use_subproc=use_subproc,
        gamma=gamma,
    )

    eval_env = create_vec_env_from_circuits(
        circuit=qc,
        hardware=hardware,
        seqlen=seqlen,
        init=init,
        num_envs=n_eval_episodes,
        rs_weight=reward_shaping_weight,
        use_subproc=use_subproc,
        gamma=gamma,
    )

    total_timesteps = num_epochs * n_steps

    config = dict(
        circuit_path=str(circuit_path),
        batch_size=batch_size,
        num_envs=num_envs,
        embed_dim=embed_dim,
        reward_shaping_weight=reward_shaping_weight,
        init_strategy=init_strategy.name,
        seqlen=seqlen,
        total_timesteps=total_timesteps,
        num_epochs=num_epochs,
        n_steps=n_steps,
        mode=mode,
        ent_coef=ent_coef,
        gamma=gamma,
        qubit_number=hardware.qubit_number,
        learning_rate=learning_rate,
    )
    pprint(config)
    
    circuit_name = Path(circuit_path).stem
    depth = qc.depth()
    log_name = f'Q={circuit_name}-CX={gate_len}-D={depth}-L={seqlen}-S={n_steps // Unit.K}-M={mode}-B={batch_size}'
    
    log_dir = f'../log/{output_dirname}'
    result_dir = f"../result/{output_dirname}"
    best_model_path = output_dir + "/models/" + log_name
    result_file = result_dir + f"/Q={circuit_name}-result.json"
    if os.path.exists(result_file) and skip_existing:
        print(f'Result exists: {result_file}')
        return read_json(result_file)

    metrics_callback = MetricEvalCallback(eval_env=eval_env, eval_freq=eval_freq // num_envs)

    eval_callback = MaskableEvalCallback(
        eval_env,
        eval_freq=eval_freq // num_envs,
        callback_after_eval=StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=max_no_improvement_evals) if max_no_improvement_evals > 0 else None,
        verbose=1,
        deterministic=False,
        use_masking=True,
        best_model_save_path=best_model_path if save_model else None,
        n_eval_episodes=n_eval_episodes,
    )

    ppo = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=n_steps // num_envs,
        batch_size=batch_size,
        tensorboard_log=log_dir,
        verbose=1,
        gamma=gamma,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
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

    result = dict(config=config, metrics=metrics, init=show_mapping(init))
    if save_result:
        write_json(result_file, jsons.dump(result))
    return result


class CircuitDataset:

    def __init__(self, dataname: str, dataroot: Path = None):
        if dataroot is None:
            dataroot = Path('../data')
        circuit_dir = dataroot / dataname / 'circuits/'
        if not circuit_dir.exists():
            raise FileNotFoundError(circuit_dir)
        circuit_paths = list(circuit_dir.glob('*.qasm'))
        random.shuffle(circuit_paths)
        circuit_paths.sort(key=lambda path: get_cnot_num(read_circuit(path)))
        self.circuit_paths = circuit_paths

    @cached_property
    def _circuits(self):
        return [QuantumCircuit.from_qasm_file(str(p)) for p in self.circuit_paths]

    def __len__(self):
        return len(self.circuit_paths)

    @cached_property
    def depth_range(self):
        return min(qc.depth() for qc in self._circuits), max(qc.depth() for qc in self._circuits)

    @cached_property
    def cx_num_range(self):
        return min(get_cnot_num(qc) for qc in self._circuits), max(get_cnot_num(qc) for qc in self._circuits)

    def sample(self, num_circuits=None, min_gatelen=10, max_gatelen=100):
        """
        按需迭代（yield）满足 CX 门数区间 [min_gatelen, max_gatelen] 的电路。
        若 num_circuits 为 None，则持续返回直到遍历完所有电路。
        """
        count = 0
        for qc, path in zip(self._circuits, self.circuit_paths):
            cx = get_cnot_num(qc)
            if min_gatelen <= cx <= max_gatelen:
                yield path
                if num_circuits is not None:
                    count += 1
                    if count >= num_circuits:
                        break

    def plot_distribution(self, stats: Literal['cx', 'depth'],
                             save_path: Optional[Path] = None,
                             figsize: tuple = (8, 5),
                             **sns_kwargs) -> None:
        """
        使用 seaborn 绘制数据集中所有电路的 CX 数量分布图。

        Parameters
        ----------
        save_path : Path, optional
            保存图片的路径；若为 None 则仅显示。
        figsize : tuple, optional
            画布大小。
        **sns_kwargs
            传给 seaborn.histplot 的额外关键字参数，如 bins, kde, color 等。
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        stats_map = dict(cx=dict(func=get_cnot_num, xlabel="Number of CNOT Gates", title="Distribution of CNOT Counts"),
                         depth=dict(func=get_circuit_depth, xlabel='Circuit Depth', title='Distribution of Circuit Depth'))
        entry = stats_map[stats]

        # 收集 CX 数量
        cx_counts = [entry['func'](qc) for qc in self._circuits]

        # 默认 seaborn 样式
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=figsize)
        sns.histplot(cx_counts,
                     binwidth=1,  # 每 1 个 CX 为一根柱
                     color="steelblue",
                     kde=True,
                     edgecolor="black",
                     **sns_kwargs)

        plt.title(entry['title'])
        plt.xlabel(entry['xlabel'])
        plt.ylabel("Frequency")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

