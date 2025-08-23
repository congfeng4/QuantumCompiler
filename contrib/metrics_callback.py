from sb3_contrib import MaskablePPO
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


def evaluate_policy_for_metrics(model, eval_env, n_eval_episodes=10, deterministic=False):
    metrics_list = []
    for _ in range(n_eval_episodes):  # 必须重复多次，早期单次eval的方差很大。
        evaluate_policy(model, eval_env, n_eval_episodes=1, use_masking=True, deterministic=deterministic)
        metrics = eval_env.get_attr('metrics')[0]
        metrics_list.append(metrics)

    return average_metrics(metrics_list)


class MetricEvalCallback(BaseCallback):
    model: MaskablePPO

    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=10, deterministic=False):
        # deterministic=True，早期评估非常慢，几乎卡死。
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            metrics = evaluate_policy_for_metrics(self.model, self.eval_env, self.n_eval_episodes, self.deterministic)

            for key, value in metrics.items():
                self.logger.record(f"metric/{key}", round(value, 2))

        return True
