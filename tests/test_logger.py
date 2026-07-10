"""
Tests for the Logger module.
"""

import io
import logging
from pathlib import Path

import pytest

from src.utils.logger import add_file_handler, get_logger, set_log_level


@pytest.fixture(autouse=True)
def reset_configured_flag():
    """Reset the configured flag so each test gets fresh logger setup."""
    import src.utils.logger as logger_mod
    logger_mod._loggers_configured = False
    yield
    logger_mod._loggers_configured = False


class TestLogger:
    """Test suite for the logger module."""

    def test_get_logger(self):
        """get_logger should return a configured logger."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_with_level_str(self):
        """get_logger should accept string level."""
        logger = get_logger("test.debug", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_get_logger_with_level_int(self):
        """get_logger should accept int level."""
        logger = get_logger("test.info", level=logging.INFO)
        assert logger.level == logging.INFO

    def test_get_logger_logs_message(self):
        """get_logger should produce loggable messages."""
        logger = get_logger("test.cap")
        # Add a handler that captures output
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("hello from test")
        output = stream.getvalue()
        assert "hello from test" in output, f"Expected message in output, got: {output!r}"

    def test_set_log_level_str(self):
        """set_log_level should accept string level."""
        logger1 = get_logger("audithub.test1")
        logger2 = get_logger("audithub.test2")
        set_log_level("DEBUG")
        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.DEBUG

    def test_set_log_level_int(self):
        """set_log_level should accept int level."""
        logger = get_logger("audithub.test_int")
        set_log_level(logging.ERROR)
        assert logger.level == logging.ERROR

    def test_add_file_handler(self, tmp_path: Path):
        """add_file_handler should add a handler to the logger."""
        log_file = tmp_path / "handler_test.log"

        logger_name = "test.handler_test"
        handler = add_file_handler(logger_name, log_file, level="INFO")
        assert isinstance(handler, logging.Handler)

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.info("write to handler")
        handler.close()

        assert log_file.exists()
        content = log_file.read_text()
        assert "write to handler" in content, f"Expected message in file, got: {content!r}"

    def test_add_file_handler_with_custom_format(self, tmp_path: Path):
        """add_file_handler should accept custom format."""
        log_file = tmp_path / "custom_fmt.log"

        logger_name = "test.custom_fmt_test"
        handler = add_file_handler(
            logger_name, log_file,
            level=logging.DEBUG,
            fmt="CUSTOM %(message)s",
        )

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.debug("custom format msg")
        handler.close()

        content = log_file.read_text()
        assert "CUSTOM" in content, f"Expected CUSTOM in file, got: {content!r}"
        assert "custom format msg" in content

    def test_add_file_handler_with_int_level(self, tmp_path: Path):
        """add_file_handler should accept int level."""
        log_file = tmp_path / "int_level.log"
        handler = add_file_handler("test.int_level", log_file, level=logging.WARNING)
        assert handler.level == logging.WARNING
        handler.close()

    def test_get_logger_multiple_calls(self):
        """get_logger should return same logger for same name."""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is logger2

    def test_get_logger_audithub_namespace(self):
        """get_logger should work with audithub namespace."""
        logger = get_logger("audithub.ingestion.test")
        assert logger.name == "audithub.ingestion.test"
