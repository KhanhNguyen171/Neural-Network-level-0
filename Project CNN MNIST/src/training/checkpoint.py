# src/training/checkpoint.py

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import torch


class ModelCheckpoint:
    """
    Save and load training checkpoints.

    Supports:
    - model state
    - optimizer state
    - scheduler state
    - training history
    - current epoch

    Example
    -------
    checkpoint = ModelCheckpoint("artifacts/checkpoints")

    checkpoint.save(
        model=model,
        optimizer=optimizer,
        epoch=5,
        metric=0.92,
        filename="best.pt",
    )
    """

    def __init__(
        self,
        directory: str,
    ) -> None:
        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def path(self) -> Path:
        return self.directory

    def save(
        self,
        model,
        optimizer=None,
        scheduler=None,
        history=None,
        epoch: int = 0,
        metric: Optional[float] = None,
        filename: str = "checkpoint.pt",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save checkpoint to disk.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = (
                optimizer.state_dict()
            )

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = (
                scheduler.state_dict()
            )

        if history is not None:
            checkpoint["history"] = (
                history.state_dict()
                if hasattr(history, "state_dict")
                else history
            )

        if metric is not None:
            checkpoint["metric"] = metric

        if extra is not None:
            checkpoint["extra"] = extra

        filepath = self.directory / filename

        torch.save(
            checkpoint,
            filepath,
        )

        return filepath

    def load(
        self,
        filepath: str | Path,
        model,
        optimizer=None,
        scheduler=None,
        history=None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        """

        checkpoint = torch.load(
            filepath,
            map_location=map_location,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if (
            history is not None
            and "history" in checkpoint
        ):
            if hasattr(
                history,
                "load_state_dict",
            ):
                history.load_state_dict(
                    checkpoint["history"]
                )

        return checkpoint

    def exists(
        self,
        filename: str,
    ) -> bool:
        """
        Check if checkpoint exists.
        """
        return (
            self.directory / filename
        ).exists()

    def delete(
        self,
        filename: str,
    ) -> None:
        """
        Delete checkpoint file.
        """
        filepath = self.directory / filename

        if filepath.exists():
            filepath.unlink()

    def latest(self) -> Optional[Path]:
        """
        Return newest checkpoint file.
        """

        files = sorted(
            self.directory.glob("*.pt"),
            key=lambda x: x.stat().st_mtime,
        )

        if not files:
            return None

        return files[-1]

    def list_checkpoints(
        self,
    ) -> list[Path]:
        """
        Return all checkpoint files.
        """

        return sorted(
            self.directory.glob("*.pt")
        )

    def save_best(
        self,
        model,
        optimizer=None,
        scheduler=None,
        history=None,
        epoch: int = 0,
        metric: float = 0.0,
        filename: str = "best.pt",
    ) -> Path:
        """
        Convenience wrapper for best model.
        """

        return self.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            epoch=epoch,
            metric=metric,
            filename=filename,
        )

    def save_last(
        self,
        model,
        optimizer=None,
        scheduler=None,
        history=None,
        epoch: int = 0,
    ) -> Path:
        """
        Convenience wrapper for latest model.
        """

        return self.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            epoch=epoch,
            filename="last.pt",
        )

    def __repr__(self) -> str:
        return (
            f"ModelCheckpoint("
            f"directory='{self.directory}'"
            f")"
        )