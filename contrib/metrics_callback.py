from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
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


class CustomMetricsCallback(MaskableEvalCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        """
        This method is called after each call to `env.step()`.

        For a vectorized env, this is called after the `step` of all environments.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        metrics_env = self.eval_env.get_attr("metrics", range(self.eval_env.num_envs))
        metrics = average_metrics(metrics_env)
        if metrics:
            for key, value in metrics.items():
                self.logger.record(f"custom/{key}", round(value, 2))

        return True
