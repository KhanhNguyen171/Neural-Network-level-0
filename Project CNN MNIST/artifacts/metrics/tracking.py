from datetime import datetime


class ExperimentTracker:
    """
    Track experiment metadata and metrics.
    """

    def __init__(
        self,
        experiment_name: str,
    ):
        self.experiment_name = experiment_name

        self.start_time = (
            datetime.now()
        )

        self.metrics = {}

    def log_metric(
        self,
        name: str,
        value,
    ):
        self.metrics[name] = value

    def log_metrics(
        self,
        metrics: dict,
    ):
        self.metrics.update(metrics)

    def summary(self):
        return {
            "experiment": self.experiment_name,
            "start_time": str(
                self.start_time
            ),
            "metrics": self.metrics,
        }