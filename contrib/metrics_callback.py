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


class CustomMetricsCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method is called after each call to `env.step()`.

        For a vectorized env, this is called after the `step` of all environments.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        metrics_env = []
        # 注意：self.locals['infos'] 可能不是 list，而是 dict（非向量化 env）
        infos = self.locals['infos']
        if not isinstance(infos, list):
            infos = [infos]

        for info in infos:
            if 'metrics' in info:
                metrics_env.append(info['metrics'])

        metrics = average_metrics(metrics_env)
        if metrics:
            for key, value in metrics.items():
                self.logger.record(f"custom/{key}", value)

        return True
