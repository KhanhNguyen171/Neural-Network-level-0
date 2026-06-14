import logging
import tempfile
from pathlib import Path

from src.utils.logger import (
    setup_logging,
    get_logger,
    set_log_level,
    log_exception,
    close_logger,
)

# pytest tests/utils/test_logger.py -v

def test_setup_logging_creates_log_file():
    with tempfile.TemporaryDirectory() as tmpdir:

        logger = setup_logging(
            log_dir=tmpdir,
            filename="test.log",
        )

        logger.info("hello world")

        for handler in logger.handlers:
            handler.flush()

        log_file = Path(tmpdir) / "test.log"

        assert log_file.exists()

        close_logger(logger)


def test_setup_logging_returns_logger():
    with tempfile.TemporaryDirectory() as tmpdir:

        logger = setup_logging(
            log_dir=tmpdir,
        )

        assert isinstance(
            logger,
            logging.Logger,
        )
        
        close_logger(logger)


def test_get_logger():
    logger = get_logger(
        "unit_test"
    )

    assert isinstance(
        logger,
        logging.Logger,
    )

    assert logger.name == "unit_test"
    
    close_logger(logger)


def test_set_log_level():
    logger = logging.getLogger(
        "test_level"
    )

    set_log_level(
        logger,
        "DEBUG",
    )

    assert (
        logger.level
        == logging.DEBUG
    )
    
    close_logger(logger)


def test_log_exception():
    with tempfile.TemporaryDirectory() as tmpdir:

        logger = setup_logging(
            log_dir=tmpdir,
            filename="error.log",
        )

        try:
            raise ValueError(
                "test error"
            )

        except ValueError as exc:
            log_exception(
                logger,
                exc,
            )

        for handler in logger.handlers:
            handler.flush()

        log_file = (
            Path(tmpdir)
            / "error.log"
        )

        content = log_file.read_text(
            encoding="utf-8"
        )

        assert (
            "test error"
            in content
        )
        
        close_logger(logger)


def test_multiple_log_messages():
    with tempfile.TemporaryDirectory() as tmpdir:

        logger = setup_logging(
            log_dir=tmpdir,
            filename="messages.log",
        )

        logger.info("info")
        logger.warning("warning")
        logger.error("error")

        for handler in logger.handlers:
            handler.flush()

        log_file = (
            Path(tmpdir)
            / "messages.log"
        )

        content = log_file.read_text(
            encoding="utf-8"
        )

        assert "info" in content
        assert "warning" in content
        assert "error" in content
        
        close_logger(logger)