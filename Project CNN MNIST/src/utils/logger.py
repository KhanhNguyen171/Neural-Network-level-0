"""
Logging utilities.
"""

from pathlib import Path
import logging


def close_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def setup_logging(
    log_dir: str | Path,
    filename: str = "app.log",
    level: str = "INFO",
) -> logging.Logger:
    """
    Configure file and console logging.
    """

    log_dir = Path(log_dir)

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    import uuid

    logger = logging.getLogger(
        f"cnn_mnist_{uuid.uuid4().hex}"
    )

    logger.setLevel(
        getattr(
            logging,
            level.upper(),
        )
    )

    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        log_dir / filename,
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter,
    )

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter,
    )

    logger.addHandler(
        file_handler,
    )

    logger.addHandler(
        console_handler,
    )

    logger.propagate = False

    return logger


def get_logger(
    name: str,
) -> logging.Logger:
    """
    Get child logger.
    """

    return logging.getLogger(name)


def set_log_level(
    logger: logging.Logger,
    level: str,
) -> None:
    """
    Update logger level.
    """

    logger.setLevel(
        getattr(
            logging,
            level.upper(),
        )
    )


def log_exception(
    logger: logging.Logger,
    exception: Exception,
) -> None:
    """
    Log exception message.
    """

    logger.exception(
        str(exception),
    )
    
def close_logger(
    logger: logging.Logger,
) -> None:
    """
    Close all handlers.
    """

    handlers = logger.handlers[:]

    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)