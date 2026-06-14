from pathlib import Path

from .manager import LogManager


class TrainingLogger:
    """
    Specialized logger for training.
    """

    def __init__(
        self,
        log_dir: str | Path,
    ):
        log_dir = Path(log_dir)

        self.logger = LogManager(
            log_dir / "training.log"
        )

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        accuracy: float,
    ):
        self.logger.info(
            (
                f"Epoch={epoch} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"accuracy={accuracy:.4f}"
            )
        )

    def log_message(
        self,
        message: str,
    ):
        self.logger.info(message)

    def close(self):
        self.logger.close()