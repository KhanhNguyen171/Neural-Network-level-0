# src/training/history.py

from __future__ import annotations

from typing import Dict, List, Any

import json


class TrainingHistory:
    """
    Stores training and validation metrics across epochs.

    Example
    -------
    >>> history = TrainingHistory()
    >>> history.update(
    ...     train_loss=0.5,
    ...     train_accuracy=0.90,
    ...     val_loss=0.4,
    ...     val_accuracy=0.92,
    ... )
    >>> history.best("val_accuracy")
    0.92
    """

    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {}

    def update(self, **metrics: float) -> None:
        """
        Add metrics for a new epoch.

        Parameters
        ----------
        metrics : dict
            Metric name -> value.
        """
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(float(value))

    def get(self, metric_name: str) -> List[float]:
        """
        Return metric history.
        """
        return self.history.get(metric_name, [])

    def latest(self, metric_name: str) -> float:
        """
        Return latest metric value.
        """
        values = self.get(metric_name)

        if not values:
            raise ValueError(
                f"No values recorded for metric '{metric_name}'."
            )

        return values[-1]

    def best(
        self,
        metric_name: str,
        mode: str = "max",
    ) -> float:
        """
        Return best metric value.

        Parameters
        ----------
        metric_name : str
        mode : {"max", "min"}
        """
        values = self.get(metric_name)

        if not values:
            raise ValueError(
                f"No values recorded for metric '{metric_name}'."
            )

        if mode == "max":
            return max(values)

        if mode == "min":
            return min(values)

        raise ValueError(
            "mode must be either 'max' or 'min'"
        )

    def best_epoch(
        self,
        metric_name: str,
        mode: str = "max",
    ) -> int:
        """
        Return epoch index (1-based) of best metric.
        """
        values = self.get(metric_name)

        if not values:
            raise ValueError(
                f"No values recorded for metric '{metric_name}'."
            )

        if mode == "max":
            idx = values.index(max(values))

        elif mode == "min":
            idx = values.index(min(values))

        else:
            raise ValueError(
                "mode must be either 'max' or 'min'"
            )

        return idx + 1

    def num_epochs(self) -> int:
        """
        Number of completed epochs.
        """
        if not self.history:
            return 0

        first_metric = next(iter(self.history))
        return len(self.history[first_metric])

    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize history.
        """
        return {
            "history": self.history
        }

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
    ) -> None:
        """
        Restore history.
        """
        self.history = state_dict["history"]

    def save(self, filepath: str) -> None:
        """
        Save history to JSON file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                self.history,
                f,
                indent=4,
            )

    def load(self, filepath: str) -> None:
        """
        Load history from JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.history = json.load(f)

    def clear(self) -> None:
        """
        Remove all recorded metrics.
        """
        self.history.clear()

    def __len__(self) -> int:
        return self.num_epochs()

    def __contains__(self, metric_name: str) -> bool:
        return metric_name in self.history

    def __repr__(self) -> str:
        metrics = list(self.history.keys())

        return (
            f"TrainingHistory("
            f"epochs={self.num_epochs()}, "
            f"metrics={metrics}"
            f")"
        )