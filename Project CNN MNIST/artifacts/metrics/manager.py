from pathlib import Path
from datetime import datetime

from .export import (
    save_metrics_json,
    save_metrics_csv,
)


class MetricsManager:
    """
    Manage experiment metrics.
    """

    def __init__(
        self,
        root_dir: str | Path,
    ):
        self.root_dir = Path(root_dir)

        self.root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_run(
        self,
        experiment_name: str,
    ):
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        run_dir = (
            self.root_dir
            / f"{experiment_name}_{timestamp}"
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return run_dir

    def save(
        self,
        metrics: dict,
        run_dir,
    ):
        save_metrics_json(
            metrics,
            run_dir / "metrics.json",
        )

        save_metrics_csv(
            metrics,
            run_dir / "metrics.csv",
        )

    def list_runs(self):
        return sorted(
            [
                p
                for p in self.root_dir.iterdir()
                if p.is_dir()
            ]
        )