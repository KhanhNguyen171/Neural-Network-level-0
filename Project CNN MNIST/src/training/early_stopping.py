# src/training/early_stopping.py

from __future__ import annotations

from typing import Optional


class EarlyStopping:
    """
    Early stopping utility.

    Stops training when monitored metric
    has not improved for a specified number
    of epochs.

    Parameters
    ----------
    monitor : str
        Metric name to monitor.

    patience : int
        Number of epochs without improvement
        before stopping.

    mode : {"min", "max"}
        Optimization direction.

    min_delta : float
        Minimum improvement required to be
        considered an actual improvement.

    Example
    -------
    >>> early_stopping = EarlyStopping(
    ...     monitor="val_loss",
    ...     patience=5,
    ...     mode="min"
    ... )

    >>> stop = early_stopping.step(
    ...     current=0.42,
    ...     epoch=10
    ... )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:

        if mode not in ("min", "max"):
            raise ValueError(
                "mode must be either "
                "'min' or 'max'"
            )

        if patience < 0:
            raise ValueError(
                "patience must be >= 0"
            )

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score: Optional[float] = None

        self.best_epoch: Optional[int] = None

        self.counter = 0

        self.should_stop = False

    def _is_improvement(
        self,
        current: float,
    ) -> bool:
        """
        Check whether metric improved.
        """

        if self.best_score is None:
            return True

        if self.mode == "min":
            return (
                current
                < self.best_score
                - self.min_delta
            )

        return (
            current
            > self.best_score
            + self.min_delta
        )

    def step(
        self,
        current: float,
        epoch: int,
    ) -> bool:
        """
        Update state with current metric.

        Returns
        -------
        bool
            True if training should stop.
        """

        if self._is_improvement(current):

            self.best_score = current

            self.best_epoch = epoch

            self.counter = 0

            self.should_stop = False

        else:

            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """
        Reset internal state.
        """

        self.best_score = None

        self.best_epoch = None

        self.counter = 0

        self.should_stop = False

    @property
    def stopped(self) -> bool:
        """
        Alias for should_stop.
        """

        return self.should_stop

    @property
    def num_bad_epochs(self) -> int:
        """
        Number of consecutive
        non-improving epochs.
        """

        return self.counter

    def state_dict(self) -> dict:
        """
        Serialize state.
        """

        return {
            "monitor": self.monitor,
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        """
        Restore state.
        """

        self.monitor = state_dict["monitor"]

        self.patience = state_dict["patience"]

        self.mode = state_dict["mode"]

        self.min_delta = state_dict["min_delta"]

        self.best_score = state_dict["best_score"]

        self.best_epoch = state_dict["best_epoch"]

        self.counter = state_dict["counter"]

        self.should_stop = state_dict["should_stop"]

    def __repr__(self) -> str:
        return (
            f"EarlyStopping("
            f"monitor='{self.monitor}', "
            f"patience={self.patience}, "
            f"mode='{self.mode}', "
            f"best_score={self.best_score}, "
            f"counter={self.counter}, "
            f"should_stop={self.should_stop}"
            f")"
        )