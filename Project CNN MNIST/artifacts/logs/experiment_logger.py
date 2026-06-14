from pathlib import Path

from .manager import LogManager


class ExperimentLogger:
    """
    Track experiment execution.
    """

    def __init__(
        self,
        log_dir: str | Path,
    ):
        log_dir = Path(log_dir)

        self.logger = LogManager(
            log_dir / "experiment.log"
        )

    def start(
        self,
        experiment_name: str,
    ):
        self.logger.info(
            f"START_EXPERIMENT: "
            f"{experiment_name}"
        )

    def finish(
        self,
        experiment_name: str,
    ):
        self.logger.info(
            f"FINISH_EXPERIMENT: "
            f"{experiment_name}"
        )

    def metric(
        self,
        name: str,
        value,
    ):
        self.logger.info(
            f"METRIC {name}={value}"
        )

    def close(self):
        self.logger.close()