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
        super(CustomMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method is called after each call to `env.step()`.

        For a vectorized env, this is called after the `step` of all environments.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        metrics_env = []
        # self.locals['dones'] is a list of boolean flags, one for each env
        for i, done in enumerate(self.locals['dones']):
            if done:
                # self.locals['infos'] is a list of info dicts, one for each env
                info = self.locals['infos'][i]

                # Check if our custom info is present
                if 'metrics' in info:
                    metrics = info['metrics']
                    metrics_env.append(metrics)

        # Calculate average metrics across all environments
        metrics = average_metrics(metrics_env)
        if self.verbose >= 1:
            print(f"Average metrics: {metrics}")
        # Record the metrics in the logger
        for key, value in metrics.items():
            self.logger.record(f"custom/{key}", value)

        return True  # Continue training
