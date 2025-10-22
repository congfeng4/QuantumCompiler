"""
直接用PPO是很难收敛的，因为非法动作空间十分巨大。
至少需要用MaskablePPO，并且把Action Mask定义好。
☀️🌛🎉🖼🏊🏻🏓✈️🚗
"""
from collections import defaultdict
import os
import random
import time
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional, Literal
from pprint import pprint
from typing import Union

import gymnasium
import jsons
import torch.cuda
from stable_baselines3.common.env_util import make_vec_env

from contrib.common import QuantumCircuit, IBMQHardwareArchitecture, write_json, get_cnot_num, readable_float_dict, \
    read_json, show_mapping, Qubit, Unit, get_circuit_depth, read_circuit, qknob_metrics

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from contrib.environs import CircuitEnvWithInitialMapping, TopologicalOrderMode
from contrib.feature_extractor import get_policy_kwargs, QubitEmbeddingMode, PositionalEncodingMode
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from hamap.initial_mapping import initial_mapping_from_sabre


def average_metrics(metrics_list):
    """
    Calculate the average of a list of metrics.

    :param metrics_list: List of dictionaries containing metrics
    :return: Dictionary with averaged metrics
    """
    if not metrics_list:
        return {}

    avg_metrics = defaultdict(list)
    for item in metrics_list:
        for key, value in item.items():
            avg_metrics[key].append(value)

    avg_metrics = {key: sum(val) / len(val) for key, val in avg_metrics.items()}
    return avg_metrics


def evaluate_policy_for_metrics(model, eval_env, use_masking):
    evaluate_policy(model, eval_env, n_eval_episodes=1, use_masking=use_masking, deterministic=False)
    metrics_list = eval_env.get_attr('metrics')
    k = random.choice(range(eval_env.num_envs))
    final_circuit = eval_env.get_attr('resulting_circuit')[k]
    final_mapping = eval_env.get_attr('current_mapping')[k]
    metrics = average_metrics(metrics_list)
    return metrics, final_circuit, final_mapping


class MetricEvalCallback(BaseCallback):
    model: MaskablePPO

    def __init__(self, eval_env, eval_freq, use_masking):
        # deterministic=True，早期评估非常慢，几乎卡死。
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.use_masking = use_masking

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            metrics, *_ = evaluate_policy_for_metrics(self.model, self.eval_env, self.use_masking)

            for key, value in metrics.items():
                self.logger.record(key, round(value, 2))

        return True


def create_vec_env_from_circuits(
        env_cls,
        circuit: QuantumCircuit,
        seqlen: int,
        hardware: IBMQHardwareArchitecture,
        init: dict[Qubit, int],
        num_envs: int = 1,
        use_subproc: bool = False,
        rs_weight: float = 1,
        gamma: float = 0.99,
        topological_order_mode: TopologicalOrderMode = TopologicalOrderMode.DEFAULT_ORDER,
):
    assert num_envs >= 1

    def make_func():
        return lambda: env_cls(
            input_circuit=circuit,
            hardware=hardware,
            initial_mapping=init,
            L=seqlen,
            reward_shaping_weight=rs_weight,
            gamma=gamma,
            topological_order_mode=topological_order_mode,
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
        hardware: Union[IBMQHardwareArchitecture, str],
        circuit_path: Union[Path, str, QuantumCircuit],
        batch_size: int = Unit.K,
        n_steps: int = 16 * Unit.K,
        eval_freq: int = 16 * Unit.K,
        embed_dim: int = 128,
        reward_shaping_weight: float = 10,
        init_strategy: Union[InitialMappingStrategy, dict[Qubit, int]] = InitialMappingStrategy.SABRE,
        seqlen: Union[int, float] = 16,
        num_epochs: int = 100,
        total_timesteps: int = 800 * Unit.K,
        output_dirname: str = None,
        mode: str = 'transformer',
        ent_coef: float = 0,
        gamma: float = 0.99,
        pretrain: Path = None,
        save_result: bool = True,
        save_model: bool = False,
        skip_existing: bool = True,
        n_eval_episodes: int = 10,
        max_no_improvement_evals=100,
        num_envs: int = None,
        learning_rate: float = 3e-4,
        env_cls: gymnasium.Env = CircuitEnvWithInitialMapping,
        use_masking: bool = True,
        clip_range: float = 0.2,
        nhead: int = 4,
        num_layers: int = 8,
        min_evals: int = 5,
        qubit_embed_mode: QubitEmbeddingMode = QubitEmbeddingMode.DISTANCE_MATRIX_MLP,
        pe_mode: PositionalEncodingMode = PositionalEncodingMode.LEVEL_PE,
        topological_order_mode: TopologicalOrderMode = TopologicalOrderMode.LEVEL_ORDER,
):
    """
    ✅ Run MaskablePPO on a circuit and return the metrics.
    """
    if pe_mode == PositionalEncodingMode.LEVEL_PE and topological_order_mode != TopologicalOrderMode.LEVEL_ORDER:
        raise ValueError(f'{pe_mode=} must be used with {TopologicalOrderMode.LEVEL_ORDER}')

    num_envs = num_envs or max(os.cpu_count() // 4, 8)
    while n_steps % num_envs != 0:
        num_envs += 1

    if output_dirname is None:
        output_dirname = 'test'

    output_dir = f'./result/{output_dirname}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not isinstance(hardware, IBMQHardwareArchitecture):
        hardware = IBMQHardwareArchitecture(hardware)

    if not isinstance(circuit_path, QuantumCircuit):
        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
    else:
        qc = circuit_path

    gate_len = get_cnot_num(qc)

    if isinstance(seqlen, int):
        seqlen = min(gate_len, seqlen)
    elif isinstance(seqlen, float):
        assert 0 < seqlen < 1
        seqlen = int(seqlen * gate_len)

    print(f'{gate_len=} {seqlen=}')

    if not isinstance(init_strategy, dict):
        init = get_initial_mapping(qc, hardware, init_strategy)
    else:
        init = init_strategy

    use_subproc = torch.cuda.is_available()  # On GPU server, use subproc to make full use of GPUs.

    env = create_vec_env_from_circuits(
        env_cls=env_cls,
        circuit=qc,
        hardware=hardware,
        seqlen=seqlen,
        init=init,
        num_envs=num_envs,
        rs_weight=reward_shaping_weight,
        use_subproc=use_subproc,
        gamma=gamma,
        topological_order_mode=topological_order_mode,
    )

    eval_env = create_vec_env_from_circuits(
        env_cls=env_cls,
        circuit=qc,
        hardware=hardware,
        seqlen=seqlen,
        init=init,
        num_envs=n_eval_episodes,
        rs_weight=reward_shaping_weight,
        use_subproc=use_subproc,
        gamma=gamma,
        topological_order_mode=topological_order_mode,
    )

    if total_timesteps is None:
        total_timesteps = num_epochs * n_steps

    config = dict(
        env_cls=env_cls.__name__,
        circuit_path=str(circuit_path) if not isinstance(circuit_path, QuantumCircuit) else None,
        batch_size=batch_size,
        num_envs=num_envs,
        embed_dim=embed_dim,
        reward_shaping_weight=reward_shaping_weight,
        init_strategy=init_strategy if isinstance(init_strategy, InitialMappingStrategy) else None,
        seqlen=seqlen,
        total_timesteps=total_timesteps,
        num_epochs=num_epochs,
        n_steps=n_steps,
        mode=mode,
        ent_coef=ent_coef,
        gamma=gamma,
        qubit_number=hardware.qubit_number,
        learning_rate=learning_rate,
        clip_range=clip_range,
        use_masking=use_masking,
        nhead=nhead,
        num_layers=num_layers,
        qubit_embed_mode=qubit_embed_mode,
        pe_mode=pe_mode,
        topological_order_mode=topological_order_mode,
    )
    pprint(config)

    circuit_name = Path(circuit_path).stem if not isinstance(circuit_path, QuantumCircuit) else None
    log_name = f'Q={circuit_name}-CX={gate_len}-D={embed_dim}-L={seqlen}-S={n_steps // Unit.K}-TM={topological_order_mode.name}-PM={pe_mode.name}'
    log_dir = f'./log/{output_dirname}'
    result_dir = f"./result/{output_dirname}"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    best_model_path = output_dir + "/models/" + log_name
    result_file = result_dir + f"/Q={circuit_name}-result.json"
    if os.path.exists(result_file) and skip_existing:
        print(f'Result exists: {result_file}')
        return read_json(result_file)

    metrics_callback = MetricEvalCallback(eval_env=eval_env, eval_freq=eval_freq // num_envs, use_masking=use_masking)

    eval_callback = MaskableEvalCallback(
        eval_env,
        eval_freq=eval_freq // num_envs,
        callback_on_new_best=StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=max_no_improvement_evals,
            min_evals=min_evals,
        ) if max_no_improvement_evals > 0 else None,
        verbose=1,
        deterministic=False,
        use_masking=use_masking,
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
        clip_range=clip_range,
        policy_kwargs=get_policy_kwargs(
            hardware=hardware,
            embed_dim=embed_dim,
            mode=mode,
            nhead=nhead,
            num_layers=num_layers,
            qubit_embed_mode=qubit_embed_mode,
            pe_mode=pe_mode,
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
        use_masking=use_masking,
    )
    learn_end = time.time()

    print('Eval policy')
    metrics, final_circuit, final_mapping = evaluate_policy_for_metrics(ppo, eval_env, use_masking)
    metrics.update(train_time=learn_end - learn_start)
    metrics = readable_float_dict(metrics)

    result = dict(config=config, metrics=metrics, init=show_mapping(init))
    if save_result:
        write_json(result_file, jsons.dump(result))

    env.close()
    eval_env.close()

    return final_circuit, final_mapping, metrics


class CircuitDataset:

    def __init__(self, dataname: str, dataroot: Path = None, shuffle=True, sort=False):
        self.dataname = dataname
        if dataroot is None:
            dataroot = Path('./data')
        circuit_dir = dataroot / dataname / 'circuits/'
        if not circuit_dir.exists():
            raise FileNotFoundError(circuit_dir)
        circuit_paths = list(circuit_dir.glob('*.qasm'))
        if shuffle:
            random.shuffle(circuit_paths)
        if sort:
            circuit_paths.sort(key=lambda path: get_cnot_num(read_circuit(path)))
        self.circuit_paths = circuit_paths
        self.circuit_dir = circuit_dir

    @cached_property
    def hardware(self):
        return IBMQHardwareArchitecture(self.dataname.split('_')[-1])

    @cached_property
    def circuits(self):
        return [QuantumCircuit.from_qasm_file(str(p)) for p in self.circuit_paths]

    def __len__(self):
        return len(self.circuit_paths)

    @cached_property
    def depth_range(self):
        return min(qc.depth() for qc in self.circuits), max(qc.depth() for qc in self.circuits)

    @cached_property
    def cx_num_range(self):
        return min(get_cnot_num(qc) for qc in self.circuits), max(get_cnot_num(qc) for qc in self.circuits)

    def sample(self, num_circuits=None, min_gatelen=10, max_gatelen=100):
        """
        按需迭代（yield）满足 CX 门数区间 [min_gatelen, max_gatelen] 的电路。
        若 num_circuits 为 None，则持续返回直到遍历完所有电路。
        """
        count = 0
        for qc, path in zip(self.circuits, self.circuit_paths):
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
                         depth=dict(func=get_circuit_depth, xlabel='Circuit Depth',
                                    title='Distribution of Circuit Depth'))
        entry = stats_map[stats]

        # 收集 CX 数量
        cx_counts = [entry['func'](qc) for qc in self.circuits]

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


def maskable_ppo_mapping(
        qc: QuantumCircuit,
        hardware: IBMQHardwareArchitecture,
        initial_mapping: dict[Qubit, int],
        **kwargs,
):
    return run_maskable_ppo(
        hardware=hardware,
        circuit_path=qc,
        init_strategy=initial_mapping,
        save_result=False,
        skip_existing=False,
        **kwargs,
    )


def forward_backward_initial_mapping(
        qc: QuantumCircuit,
        hardware: IBMQHardwareArchitecture,
        initial_mapping: dict[Qubit, int] = None,
        **kwargs,
):
    initial_mapping = initial_mapping_from_sabre(
        quantum_circuit=qc,
        hardware=hardware,
        mapping_algorithm=lambda qc, hw, init: maskable_ppo_mapping(qc, hw, init, **kwargs),
        initial_mapping=initial_mapping
    )
    final_circuit, _ = maskable_ppo_mapping(qc=qc, hardware=hardware, initial_mapping=initial_mapping)
    metrics = qknob_metrics(qc, final_circuit)
    return metrics, final_circuit, initial_mapping
