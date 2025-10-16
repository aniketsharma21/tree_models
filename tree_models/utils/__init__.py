"""Tree Models - Utility Components.

This module provides shared utilities including logging, error handling,
timing, and common helper functions used throughout the framework.

Key Components:
- Logger: Structured logging with configurable formats
- Timer: Performance timing and benchmarking utilities  
- Exceptions: Custom exception hierarchy with context
- Error Handling: Standardized error context and recovery strategies

Example:
    >>> from tree_models.utils import get_logger, timer
    >>> from tree_models.utils.error_handling import model_operation_context
    >>> logger = get_logger(__name__)
    >>> with timer('operation_name'):
    ...     # timed operation
    >>> with model_operation_context('train_model', model_type='xgboost'):
    ...     # model operation with error handling
"""

# Core utilities
from .logger import (
    get_logger,
    configure_logging,
    set_log_level
)
from .timer import (
    timer,
    timed_operation,
    benchmark
)
from .exceptions import (
    TreeModelsError,
    ModelTrainingError,
    ModelEvaluationError,
    ConfigurationError,
    DataValidationError,
    handle_and_reraise
)

# Enhanced error handling
from .error_handling import (
    ErrorContext,
    ErrorHandler,
    model_operation_context,
    data_operation_context,
    config_operation_context,
    handle_errors
)

__all__ = [
    # Logging utilities
    'get_logger',
    'configure_logging',
    'set_log_level',
    
    # Timing utilities
    'timer',
    'timed_operation', 
    'benchmark',
    
    # Exception handling
    'TreeModelsError',
    'ModelTrainingError',
    'ModelEvaluationError',
    'ConfigurationError',
    'DataValidationError',
    'handle_and_reraise',
    
    # Enhanced error handling
    'ErrorContext',
    'ErrorHandler',
    'model_operation_context',
    'data_operation_context',
    'config_operation_context',
    'handle_errors'
]