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
    TreeModelsLogger
)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before and after each test."""
    # Reset internal state of TreeModelsLogger
    TreeModelsLogger._configured = False
    TreeModelsLogger._loggers = {}
    
    # Get the root logger for the package
    root_logger = logging.getLogger('tree_models')
    
    # Remove all handlers
    root_logger.handlers.clear()
    
    # Remove all filters
    root_logger.filters.clear()
    
    # Set level to a default
    root_logger.setLevel(logging.NOTSET)
    
    yield
    
    # Repeat cleanup after test
    TreeModelsLogger._configured = False
    TreeModelsLogger._loggers = {}
    root_logger.handlers.clear()
    root_logger.filters.clear()
    root_logger.setLevel(logging.NOTSET)


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
        logger = get_logger(__name__)
        assert logger.level == logging.DEBUG

    def test_configure_logging_file(self, temporary_directory: Path):
        """Test that configure_logging sets up a file handler."""
        log_file = temporary_directory / "test.log"
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
        logger = get_logger(__name__)
        
        assert logging.getLogger('tree_models').level == logging.INFO

        with temporary_log_level("DEBUG"):
            assert logging.getLogger('tree_models').level == logging.DEBUG
        
        assert logging.getLogger('tree_models').level == logging.INFO

    def test_set_log_level(self):
        """Test that set_log_level changes the logging level."""
        configure_logging(level="INFO")
        logger = get_logger(__name__)
        
        assert logging.getLogger('tree_models').level == logging.INFO
        
        set_log_level("WARNING")
        assert logging.getLogger('tree_models').level == logging.WARNING

    def test_log_format(self):
        """Test that the log format is correct."""
        log_stream = io.StringIO()
        
        # Configure logger to use a stream handler
        TreeModelsLogger.configure(level="INFO")
        root_logger = logging.getLogger('tree_models')
        
        # Remove existing handlers and add a stream handler
        root_logger.handlers.clear()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(root_logger.handlers[0].formatter if root_logger.handlers else logging.Formatter())
        root_logger.addHandler(handler)

        logger = get_logger(__name__)
        logger.info("Test message")
        
        log_output = log_stream.getvalue()
        
        # Example format: [2023-10-27 10:00:00.123] INFO     | tree_models.tests.test_utils_logger | Test message
        assert "INFO" in log_output
        assert "tree_models.tests.test_utils_logger" in log_output
        assert "Test message" in log_output

    def test_performance_logger_adapter(self):
        """Test the PerformanceLoggerAdapter."""
        log_stream = io.StringIO()
        
        # Configure logger to use a stream handler
        TreeModelsLogger.configure(level="DEBUG")
        root_logger = logging.getLogger('tree_models')
        root_logger.handlers.clear()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(root_logger.handlers[0].formatter if root_logger.handlers else logging.Formatter())
        root_logger.addHandler(handler)

        perf_logger = get_logger(__name__, with_performance=True)
        
        perf_logger.start_timer("my_timer")
        perf_logger.stop_timer("my_timer")
        
        log_output = log_stream.getvalue()
        
        assert "Timer 'my_timer' started" in log_output
        assert "Timer 'my_timer' completed" in log_output
        assert "Duration" in log_output

if __name__ == "__main__":
    pytest.main([__file__])
