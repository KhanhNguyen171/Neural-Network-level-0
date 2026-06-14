from pathlib import Path
import logging


class LogManager:
    """
    Generic file logger manager.
    """

    def __init__(
        self,
        log_file: str | Path,
        level=logging.INFO,
    ):
        self.log_file = Path(log_file)

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.logger = logging.getLogger(
            str(self.log_file)
        )

        self.logger.setLevel(level)

        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        handler = logging.FileHandler(
            self.log_file,
            encoding="utf-8",
        )

        handler.setFormatter(
            formatter
        )

        self.logger.addHandler(
            handler
        )

    def info(
        self,
        message: str,
    ):
        self.logger.info(message)

    def warning(
        self,
        message: str,
    ):
        self.logger.warning(message)

    def error(
        self,
        message: str,
    ):
        self.logger.error(message)

    def close(self):
        handlers = self.logger.handlers[:]

        for handler in handlers:
            handler.close()
            self.logger.removeHandler(
                handler
            )