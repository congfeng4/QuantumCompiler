"""
直接用PPO是很难收敛的，因为非法动作空间十分巨大。
至少需要用MaskablePPO，并且把Action Mask定义好。
☀️🌛🎉🖼🏊🏻🏓✈️🚗
"""
import numpy as np
import datetime
from collections import defaultdict
import os
import random
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional, Literal
from pprint import pprint
from typing import Union
from pathlib import Path

import gymnasium
import torch.cuda
from stable_baselines3.common.env_util import make_vec_env

from contrib.common import QuantumCircuit, IBMQHardwareArchitecture, read_json, write_circuit, write_json, get_cnot_num, \
    Qubit, Unit, get_circuit_depth, read_circuit, qknob_metrics, convert_to_int_mapping

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from contrib.environs import CircuitEnvWithInitialMapping
from contrib.feature_extractor import get_policy_kwargs
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from contrib.random_graphs import generate_graph_for_num_qubits
from contrib.verify_circuit import verify_circuit_equivalent
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

    avg_metrics = {key: np.mean(val) for key, val in avg_metrics.items()}
    return avg_metrics


def evaluate_policy_for_metrics(model, eval_env, key_metric='metric/depth_ratio'):
    evaluate_policy(model, eval_env, n_eval_episodes=1, use_masking=True, deterministic=False)
    metrics_list = eval_env.get_attr('metrics')
    k = np.argmin([m[key_metric] for m in metrics_list])
    output_circuit = eval_env.get_attr('resulting_circuit')[k]
    input_circuit = eval_env.get_attr('input_circuit')[k]
    initial_mapping = eval_env.get_attr('initial_mapping')[k]
    final_mapping = eval_env.get_attr('final_mapping')[k]
    metrics = average_metrics(metrics_list)
    return metrics, input_circuit, output_circuit, initial_mapping, final_mapping


class MetricEvalCallback(BaseCallback):
    model: MaskablePPO

    def __init__(self, eval_env, eval_freq,
                 verify_circuit=False,
                 best_save_dir=None):
        # deterministic=True，早期评估非常慢，几乎卡死。
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        # Monitor depth_ratio, more important than ops_ratio.
        self.best_depth_ratio = None
        self.best_save_dir = None
        self.verify_circuit = verify_circuit

        if best_save_dir is not None:
            self.best_save_dir = Path(best_save_dir)
            self.best_save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            metrics, input_circuit, output_circuit, initial_mapping, final_mapping = \
                evaluate_policy_for_metrics(self.model, self.eval_env)
            depth_ratio = metrics['metric/depth_ratio']

            if self.best_depth_ratio is None or depth_ratio < self.best_depth_ratio:
                self.best_depth_ratio = depth_ratio
                print('New best depth_ratio', depth_ratio)

                if self.best_save_dir is not None:
                    write_circuit(self.best_save_dir / 'output_circuit.qasm', output_circuit)
                    write_circuit(self.best_save_dir / 'input_circuit.qasm', input_circuit)
                    clean_metrics = {key: value for key, value in metrics.items() if key.startswith('metric/')}
                    write_json(self.best_save_dir / 'metrics.json', clean_metrics)
                    write_json(self.best_save_dir / 'initial_mapping.json', convert_to_int_mapping(initial_mapping))
                    write_json(self.best_save_dir / 'final_mapping.json', convert_to_int_mapping(final_mapping))

            if self.verify_circuit and initial_mapping is not None:
                print('Begin to verify circuit...')
                assert verify_circuit_equivalent(input_circuit, output_circuit)

            for key, value in metrics.items():
                self.logger.record(key, round(value, 2))

        return True


def create_vec_env_from_circuits(
        env_cls,
        circuit: QuantumCircuit,
        hardware: IBMQHardwareArchitecture,
        init: dict[Qubit, int],
        num_envs: int = 1,
        use_subproc: bool = False,
        verbose=False,
        params=None,
):
    assert num_envs >= 1

    def make_func():
        return lambda: env_cls(
            input_circuit=circuit,
            hardware=hardware,
            initial_mapping=init,
            verbose=verbose,
            params=params,
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
        init_strategy: Union[InitialMappingStrategy, dict[Qubit, int]] = InitialMappingStrategy.SABRE,
        env_cls: gymnasium.Env = CircuitEnvWithInitialMapping,
        output_dir: Path = None,

        # Numeric options.
        n_steps: int = 2048,
        total_timesteps: int = 50 * Unit.K,
        eval_freq: int = 2 * Unit.K,
        feature_dim: int = 64,
        n_eval_episodes: int = 5,
        max_no_improvement_evals=100,
        num_envs: int = None,
        min_evals: int = 5,

        # Boolean flags.
        save_model: bool = False,
        verbose=False,
        verify_circuit=False,

        # Other flags.
        ppo_params=None,
        model_params=None,
        env_params=None,
):
    """
    ✅ Run MaskablePPO on a circuit and return the metrics.
    """
    if model_params is None:
        model_params = {}
    if ppo_params is None:
        ppo_params = {}
    ppo_params.setdefault('batch_size', 256) # default 64 is too small.
    if env_params is None:
        env_params = {}

    num_envs = num_envs or max(os.cpu_count() // 4, 8)
    while n_steps % num_envs != 0:
        num_envs += 1

    if isinstance(circuit_path, QuantumCircuit):
        qc = circuit_path
    elif isinstance(circuit_path, (str, Path)):
        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
    else:
        raise TypeError(circuit_path)

    if qc.num_qubits > 10:
        if verify_circuit:
            print('Turn off verify_circuit because num_qubits is', qc.num_qubits)
        verify_circuit = False

    if isinstance(hardware, IBMQHardwareArchitecture):
        hardware = hardware
        hardware_name = hardware.name
    elif isinstance(hardware, str):
        hardware, hardware_name = generate_graph_for_num_qubits(hardware, num_qubits=qc.num_qubits)
    else:
        raise TypeError(hardware)

    if not isinstance(init_strategy, dict):
        init = get_initial_mapping(qc, hardware, init_strategy)
    else:
        init = init_strategy

    use_subproc = torch.cuda.is_available()  # On GPU server, use subproc to make full use of GPUs.

    env = create_vec_env_from_circuits(
        env_cls=env_cls,
        circuit=qc,
        hardware=hardware,
        init=init,
        num_envs=num_envs,
        use_subproc=use_subproc,
        verbose=verbose,
        params=env_params,
    )

    eval_env = create_vec_env_from_circuits(
        env_cls=env_cls,
        circuit=qc,
        hardware=hardware,
        init=init,
        num_envs=n_eval_episodes,
        use_subproc=use_subproc,
        params=env_params,
    )

    circuit_name = Path(circuit_path).stem if not isinstance(circuit_path, QuantumCircuit) else 'unknown'

    config = dict(
        env=env_cls.__name__,
        hardware=hardware_name,
        circuit=circuit_name,
        num_envs=num_envs,
        feature_dim=feature_dim,
        init_strategy=init_strategy if isinstance(init_strategy, InitialMappingStrategy) else None,
        total_timesteps=total_timesteps,
        n_steps=n_steps,
        qubit_number=hardware.qubit_number,
        model_params=model_params,
        ppo_params=ppo_params,
        env_params=env_params,
        verify_circuit=verify_circuit,
    )
    pprint(config)

    if output_dir is None:
        output_dir = Path(f'./output/ours/{hardware.name}/{circuit_name}')

    write_json(output_dir / 'config.json', config)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_dir = output_dir / "tb_log"

    metrics_callback = MetricEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq // num_envs,
        verify_circuit=verify_circuit,
        best_save_dir=output_dir,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        eval_freq=eval_freq // num_envs,
        callback_on_new_best=StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=max_no_improvement_evals,
            min_evals=min_evals,
        ) if max_no_improvement_evals > 0 else None,
        verbose=1,
        deterministic=False,
        use_masking=True,
        best_model_save_path=output_dir if save_model else None,
        n_eval_episodes=n_eval_episodes,
    )

    ppo = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=n_steps // num_envs,
        tensorboard_log=str(log_dir),
        verbose=1,
        policy_kwargs=get_policy_kwargs(
            qubit_number=hardware.qubit_number,
            feature_dim=feature_dim,
            params=model_params,
        ),
        **ppo_params,
    )

    ppo.tensorboard_log = log_dir
    print(f'Model loaded: {ppo}')
    ppo.learn(
        total_timesteps=total_timesteps,
        tb_log_name=str(datetime.datetime.now()),
        progress_bar=True,
        callback=[eval_callback, metrics_callback],
        use_masking=True,
    )
    env.close()
    eval_env.close()
    config.update(read_json(output_dir / "metrics.json"))
    return config


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

    @property
    def hardware_name(self):
        return self.dataname.split('_')[-1]

    @cached_property
    def hardware(self):
        return IBMQHardwareArchitecture(self.hardware_name)

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


def run_as_subprocess(circuit_path: Path, layout_method, hardware_name: str, output_dir: Path):
    assert isinstance(circuit_path, Path)
    assert isinstance(hardware_name, str)

    import subprocess, sys

    cmd = (f'{sys.executable} train.py '
           f'--path {circuit_path} '
           f'--layout {layout_method.value} '
           f'--output {output_dir} '
           f'--hardware {hardware_name}').split()

    try:
        subprocess.check_call(cmd, cwd=Path.cwd().absolute())
    except Exception as e:
        print(circuit_path, hardware_name, 'failed', 'error', e)
        raise

    metrics = read_json(output_dir / 'metrics.json')
    return metrics
