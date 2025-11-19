# tests/test_utils_logger.py
"""Unit tests for the logger utility."""

import logging
import pytest
from pathlib import Path
import io

from tree_models.utils.logger import (
    get_logger,
    configure_logging,
    set_log_level,
    temporary_log_level,
    PerformanceLoggerAdapter,
    TreeModelsLogger,
)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before and after each test."""
    logging.getLogger("tree_models").handlers.clear()
    logging.getLogger("tree_models").filters.clear()
    logging.getLogger("tree_models").setLevel(logging.NOTSET)
    TreeModelsLogger._configured = False
    TreeModelsLogger._loggers = {}
    yield
    logging.getLogger("tree_models").handlers.clear()
    logging.getLogger("tree_models").filters.clear()
    logging.getLogger("tree_models").setLevel(logging.NOTSET)
    TreeModelsLogger._configured = False
    TreeModelsLogger._loggers = {}


class TestLogger:
    """Test cases for the logger utility."""

    def test_get_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)
        assert logger.name == f"tree_models.{__name__}"

    def test_get_logger_with_performance(self):
        """Test that get_logger with with_performance=True returns a PerformanceLoggerAdapter."""
        perf_logger = get_logger(__name__, with_performance=True)
        assert isinstance(perf_logger, PerformanceLoggerAdapter)

    def test_configure_logging_level(self):
        """Test that configure_logging sets the logging level."""
        configure_logging(level="DEBUG")
        assert logging.getLogger("tree_models").level == logging.DEBUG

    def test_configure_logging_file(self, tmp_path: Path):
        """Test that configure_logging sets up a file handler."""
        log_file = tmp_path / "test.log"
        configure_logging(log_file=log_file)

        logger = get_logger(__name__)
        logger.warning("This is a test.")

        assert log_file.exists()
        with open(log_file, "r") as f:
            content = f.read()
            assert "This is a test." in content

    def test_temporary_log_level(self):
        """Test the temporary_log_level context manager."""
        configure_logging(level="INFO")
        assert logging.getLogger("tree_models").level == logging.INFO

        with temporary_log_level("DEBUG"):
            assert logging.getLogger("tree_models").level == logging.DEBUG

        assert logging.getLogger("tree_models").level == logging.INFO

    def test_set_log_level(self):
        """Test that set_log_level changes the logging level."""
        configure_logging(level="INFO")
        assert logging.getLogger("tree_models").level == logging.INFO

        set_log_level("WARNING")
        assert logging.getLogger("tree_models").level == logging.WARNING

    def test_log_format(self, caplog):
        """Test that the log format is correct."""
        configure_logging(level="INFO")
        logger = get_logger(__name__)
        logger.info("Test message")

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == "INFO"
        assert record.name == f"tree_models.{__name__}"
        assert record.getMessage() == "Test message"

    def test_performance_logger_adapter(self, caplog):
        """Test the PerformanceLoggerAdapter."""
        configure_logging(level="DEBUG")
        perf_logger = get_logger(__name__, with_performance=True)

        with caplog.at_level(logging.DEBUG):
            perf_logger.start_timer("my_timer")
            perf_logger.stop_timer("my_timer")

        assert len(caplog.records) == 2
        start_record, stop_record = caplog.records
        assert "Timer 'my_timer' started" in start_record.getMessage()
        assert "Timer 'my_timer' completed" in stop_record.getMessage()
        assert "Duration:" in stop_record.getMessage()


if __name__ == "__main__":
    pytest.main([__file__])
