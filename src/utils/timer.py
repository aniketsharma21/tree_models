"""Function timing decorator and utilities.

This module provides timing decorators and utilities to measure execution time.
"""

import time
import functools
from typing import Callable, Any, Dict
from .logger import get_logger

logger = get_logger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator to time function execution.

    Args:
        func: Function to be timed

    Returns:
        Wrapped function that logs execution time

    Example:
        >>> @timer
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()  # Logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"{func.__name__} executed in {duration:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"{func.__name__} failed after {duration:.4f} seconds: {e}")
            raise
    return wrapper


class Timer:
    """Context manager for timing code blocks.

    Example:
        >>> with Timer("data loading"):
        ...     data = load_large_dataset()
        >>> # Logs: data loading completed in X.XXXX seconds
    """

    def __init__(self, name: str = "operation"):
        """Initialize timer with operation name.

        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> 'Timer':
        """Start timing."""
        self.start_time = time.time()
        logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End timing and log duration."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is None:
            logger.info(f"{self.name} completed in {duration:.4f} seconds")
        else:
            logger.error(f"{self.name} failed after {duration:.4f} seconds")

    @property
    def duration(self) -> float:
        """Get the duration of the last timing operation.

        Returns:
            Duration in seconds
        """
        if self.end_time > self.start_time:
            return self.end_time - self.start_time
        return 0.0


def time_function(func: Callable, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Time a function call and return results with timing info.

    Args:
        func: Function to time
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Dictionary containing 'result', 'duration', and 'success' keys

    Example:
        >>> result_info = time_function(expensive_computation, data=my_data)
        >>> print(f"Result: {result_info['result']}")
        >>> print(f"Duration: {result_info['duration']:.4f}s")
    """
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        end_time = time.time()
        return {
            'result': result,
            'duration': end_time - start_time,
            'success': True,
            'error': None
        }
    except Exception as e:
        end_time = time.time()
        return {
            'result': None,
            'duration': end_time - start_time,
            'success': False,
            'error': str(e)
        }
