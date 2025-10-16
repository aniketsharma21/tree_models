# tree_models/utils/logger.py
"""Enhanced logging utilities for tree_models package.

This module provides centralized logging configuration with support for
multiple output formats, log levels, and performance tracking.
"""

import logging
import logging.handlers
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import threading

from .exceptions import FileOperationError


class TreeModelsFormatter(logging.Formatter):
    """Custom formatter for tree_models package logs.
    
    Provides structured logging with consistent format across
    all package modules, including timestamp, level, module,
    and optional context information.
    """
    
    def __init__(self, include_context: bool = True) -> None:
        """Initialize formatter.
        
        Args:
            include_context: Whether to include context fields in output
        """
        self.include_context = include_context
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with tree_models structure.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        # Base format
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname
        module = record.name
        message = record.getMessage()
        
        # Add context if available
        context_str = ""
        if self.include_context and hasattr(record, 'context'):
            context_str = f" | Context: {json.dumps(record.context, default=str)}"
        
        # Add performance info if available
        perf_str = ""
        if hasattr(record, 'duration'):
            perf_str = f" | Duration: {record.duration:.3f}s"
        
        return f"[{timestamp}] {level:8s} | {module:20s} | {message}{context_str}{perf_str}"


class PerformanceLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds performance tracking capabilities.
    
    Extends standard logger with timing functionality and
    structured context management for performance monitoring.
    """
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None) -> None:
        """Initialize performance adapter.
        
        Args:
            logger: Base logger instance
            extra: Additional context to include in all log messages
        """
        super().__init__(logger, extra or {})
        self._timers: Dict[str, float] = {}
    
    def start_timer(self, name: str) -> None:
        """Start a named timer.
        
        Args:
            name: Timer name for later reference
        """
        self._timers[name] = time.perf_counter()
        self.debug(f"Timer '{name}' started", extra={'context': {'timer_action': 'start', 'timer_name': name}})
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return duration.
        
        Args:
            name: Timer name to stop
            
        Returns:
            Duration in seconds
            
        Raises:
            ValueError: If timer was not started
        """
        if name not in self._timers:
            raise ValueError(f"Timer '{name}' was not started")
        
        duration = time.perf_counter() - self._timers[name]
        del self._timers[name]
        
        self.info(f"Timer '{name}' completed", extra={
            'context': {'timer_action': 'stop', 'timer_name': name},
            'duration': duration
        })
        
        return duration
    
    def log_with_context(self, level: int, message: str, **context: Any) -> None:
        """Log message with additional context.
        
        Args:
            level: Log level
            message: Log message
            **context: Additional context fields
        """
        self.log(level, message, extra={'context': context})


class TreeModelsLogger:
    """Centralized logger management for tree_models package.
    
    Provides package-wide logging configuration with support for
    multiple handlers, log levels, and performance tracking.
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured: bool = False
    _lock = threading.Lock()
    
    @classmethod
    def configure(
        cls,
        level: Union[str, int] = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        format_style: str = "detailed",
        include_console: bool = True
    ) -> None:
        """Configure package-wide logging settings.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            format_style: Formatting style ('simple' or 'detailed')
            include_console: Whether to include console output
        """
        with cls._lock:
            if cls._configured:
                return
            
            # Convert string level to int if needed
            if isinstance(level, str):
                level = getattr(logging, level.upper())
            
            # Create root logger for package
            root_logger = logging.getLogger('tree_models')
            root_logger.setLevel(level)
            
            # Remove existing handlers
            root_logger.handlers.clear()
            
            # Create formatter
            include_context = format_style == "detailed"
            formatter = TreeModelsFormatter(include_context=include_context)
            
            # Console handler
            if include_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)
            
            # File handler with rotation
            if log_file:
                try:
                    log_path = Path(log_file)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    file_handler = logging.handlers.RotatingFileHandler(
                        log_path,
                        maxBytes=max_file_size,
                        backupCount=backup_count,
                        encoding='utf-8'
                    )
                    file_handler.setLevel(level)
                    file_handler.setFormatter(formatter)
                    root_logger.addHandler(file_handler)
                    
                except Exception as e:
                    raise FileOperationError(
                        f"Failed to create log file handler: {log_file}",
                        error_code="LOG_FILE_SETUP_FAILED",
                        context={'log_file': str(log_file), 'error': str(e)}
                    ) from e
            
            cls._configured = True
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        with_performance: bool = False
    ) -> Union[logging.Logger, PerformanceLoggerAdapter]:
        """Get a logger instance for the specified module.
        
        Args:
            name: Logger name (typically __name__)
            with_performance: Whether to return performance-enhanced logger
            
        Returns:
            Logger instance, optionally with performance tracking
        """
        # Auto-configure if not done
        if not cls._configured:
            cls.configure()
        
        # Ensure name starts with package name
        if not name.startswith('tree_models'):
            if name == '__main__':
                name = 'tree_models.main'
            else:
                name = f'tree_models.{name.split(".")[-1]}'
        
        # Get or create logger
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        else:
            logger = cls._loggers[name]
        
        # Return performance adapter if requested
        if with_performance:
            return PerformanceLoggerAdapter(logger)
        
        return logger
    
    @classmethod
    def set_level(cls, level: Union[str, int]) -> None:
        """Change logging level for all package loggers.
        
        Args:
            level: New logging level
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        root_logger = logging.getLogger('tree_models')
        root_logger.setLevel(level)
        
        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(level)
    
    @classmethod
    def add_context_filter(cls, **context: Any) -> None:
        """Add global context to all loggers.
        
        Args:
            **context: Context fields to add to all log messages
        """
        class ContextFilter(logging.Filter):
            def filter(self, record):
                if not hasattr(record, 'context'):
                    record.context = {}
                record.context.update(context)
                return True
        
        root_logger = logging.getLogger('tree_models')
        root_logger.addFilter(ContextFilter())


# Convenience functions
def get_logger(name: str, with_performance: bool = False) -> Union[logging.Logger, PerformanceLoggerAdapter]:
    """Get a logger instance for the specified module.
    
    This is a convenience function that wraps TreeModelsLogger.get_logger()
    for easier imports and usage throughout the package.
    
    Args:
        name: Logger name (typically __name__)
        with_performance: Whether to return performance-enhanced logger
        
    Returns:
        Logger instance, optionally with performance tracking
        
    Example:
        >>> from tree_models.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Model training started")
        
        >>> # With performance tracking
        >>> perf_logger = get_logger(__name__, with_performance=True)
        >>> perf_logger.start_timer("training")
        >>> # ... do work ...
        >>> duration = perf_logger.stop_timer("training")
    """
    return TreeModelsLogger.get_logger(name, with_performance)


def configure_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> None:
    """Configure package-wide logging settings.
    
    This is a convenience function that wraps TreeModelsLogger.configure()
    for easier setup during package initialization.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        **kwargs: Additional configuration options
        
    Example:
        >>> from tree_models.utils.logger import configure_logging
        >>> configure_logging(level="DEBUG", log_file="logs/tree_models.log")
    """
    TreeModelsLogger.configure(level=level, log_file=log_file, **kwargs)


def set_log_level(level: Union[str, int]) -> None:
    """Change logging level for all package loggers.
    
    Args:
        level: New logging level
        
    Example:
        >>> from tree_models.utils.logger import set_log_level
        >>> set_log_level("DEBUG")
    """
    TreeModelsLogger.set_level(level)


# Context manager for temporary log level changes
class temporary_log_level:
    """Context manager for temporary log level changes.
    
    Allows temporarily changing the log level for a block of code,
    then restoring the original level when exiting the context.
    
    Example:
        >>> with temporary_log_level("DEBUG"):
        ...     # Debug logging enabled here
        ...     logger.debug("Detailed debugging info")
        >>> # Original log level restored here
    """
    
    def __init__(self, level: Union[str, int]) -> None:
        """Initialize context manager.
        
        Args:
            level: Temporary log level
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.temp_level = level
        self.original_level: Optional[int] = None
    
    def __enter__(self) -> None:
        """Enter context and set temporary log level."""
        root_logger = logging.getLogger('tree_models')
        self.original_level = root_logger.level
        TreeModelsLogger.set_level(self.temp_level)
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore original log level."""
        if self.original_level is not None:
            TreeModelsLogger.set_level(self.original_level)