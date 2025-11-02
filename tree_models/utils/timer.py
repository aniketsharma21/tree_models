# tree_models/utils/timer.py
"""Performance timing utilities for tree_models package.

This module provides decorators and context managers for measuring
execution time and performance monitoring throughout the ML pipeline.
"""

import time
import functools
from typing import Callable, TypeVar, Any, Optional, Dict
from contextlib import contextmanager
from collections import defaultdict
import threading

from .logger import get_logger
from .exceptions import PerformanceError

logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class PerformanceTracker:
    """Thread-safe performance tracking utility.
    
    Tracks execution times, call counts, and performance statistics
    across the entire application with thread-safe operations.
    """
    
    def __init__(self) -> None:
        """Initialize performance tracker."""
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_time': 0.0,
            'call_count': 0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0
        })
        self._lock = threading.Lock()
    
    def record_execution(self, name: str, duration: float) -> None:
        """Record an execution time for a named operation.
        
        Args:
            name: Operation name
            duration: Execution duration in seconds
        """
        with self._lock:
            stats = self._stats[name]
            stats['total_time'] += duration
            stats['call_count'] += 1
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)
            stats['avg_time'] = stats['total_time'] / stats['call_count']
    
    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics.
        
        Args:
            name: Optional specific operation name
            
        Returns:
            Performance statistics dictionary
        """
        with self._lock:
            if name:
                return dict(self._stats[name]) if name in self._stats else {}
            return {k: dict(v) for k, v in self._stats.items()}
    
    def reset_stats(self, name: Optional[str] = None) -> None:
        """Reset performance statistics.
        
        Args:
            name: Optional specific operation name to reset
        """
        with self._lock:
            if name:
                if name in self._stats:
                    del self._stats[name]
            else:
                self._stats.clear()
    
    def get_summary(self) -> str:
        """Get a formatted summary of performance statistics.
        
        Returns:
            Formatted performance summary string
        """
        stats = self.get_stats()
        if not stats:
            return "No performance data recorded"
        
        lines = ["Performance Summary:", "=" * 50]
        
        for name, data in sorted(stats.items()):
            lines.append(f"\n{name}:")
            lines.append(f"  Calls: {data['call_count']:,}")
            lines.append(f"  Total: {data['total_time']:.3f}s")
            lines.append(f"  Average: {data['avg_time']:.3f}s")
            lines.append(f"  Min: {data['min_time']:.3f}s")
            lines.append(f"  Max: {data['max_time']:.3f}s")
        
        return "\n".join(lines)


# Global performance tracker
_performance_tracker = PerformanceTracker()


def timer(
    name: Optional[str] = None,
    log_result: bool = True,
    track_performance: bool = True,
    timeout: Optional[float] = None
) -> Callable[[F], F]:
    """Decorator to time function execution.
    
    Args:
        name: Optional custom name for the operation
        log_result: Whether to log the execution time
        track_performance: Whether to record in global performance tracker
        timeout: Optional timeout in seconds (raises PerformanceError if exceeded)
        
    Returns:
        Decorated function
        
    Example:
        >>> @timer(name="model_training", timeout=3600)
        ... def train_model(X, y):
        ...     # Training logic here
        ...     pass
    """
    def decorator(func: F) -> F:
        operation_name = name or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                
                # Check timeout
                if timeout and duration > timeout:
                    raise PerformanceError(
                        f"Operation '{operation_name}' exceeded timeout of {timeout}s",
                        error_code="OPERATION_TIMEOUT",
                        context={'operation': operation_name, 'duration': duration, 'timeout': timeout}
                    )
                
                # Log result
                if log_result:
                    logger.info(f"Operation '{operation_name}' completed in {duration:.3f}s")
                
                # Track performance
                if track_performance:
                    _performance_tracker.record_execution(operation_name, duration)
                
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start_time
                
                if log_result:
                    logger.error(f"Operation '{operation_name}' failed after {duration:.3f}s: {e}")
                
                raise
        
        return wrapper
    
    return decorator


@contextmanager
def timed_operation(
    name: str,
    log_result: bool = True,
    track_performance: bool = True,
    timeout: Optional[float] = None
):
    """Context manager for timing code blocks.
    
    Args:
        name: Operation name
        log_result: Whether to log the execution time
        track_performance: Whether to record in global performance tracker  
        timeout: Optional timeout in seconds
        
    Yields:
        Dictionary with timing information (updated during execution)
        
    Example:
        >>> with timed_operation("data_processing") as timing:
        ...     # Processing logic here
        ...     pass
        >>> print(f"Processing took {timing['duration']:.3f}s")
    """
    timing_info = {'duration': 0.0, 'start_time': 0.0}
    start_time = time.perf_counter()
    timing_info['start_time'] = start_time
    
    try:
        yield timing_info
        duration = time.perf_counter() - start_time
        timing_info['duration'] = duration
        
        # Check timeout
        if timeout and duration > timeout:
            raise PerformanceError(
                f"Operation '{name}' exceeded timeout of {timeout}s",
                error_code="OPERATION_TIMEOUT",
                context={'operation': name, 'duration': duration, 'timeout': timeout}
            )
        
        # Log result
        if log_result:
            logger.info(f"Operation '{name}' completed in {duration:.3f}s")
        
        # Track performance
        if track_performance:
            _performance_tracker.record_execution(name, duration)
            
    except Exception as e:
        duration = time.perf_counter() - start_time
        timing_info['duration'] = duration
        
        if log_result:
            logger.error(f"Operation '{name}' failed after {duration:.3f}s: {e}")
        
        raise


class TimerContext:
    """Context manager class for more advanced timing scenarios.
    
    Provides additional functionality like nested timing, checkpoints,
    and conditional logging based on duration thresholds.
    """
    
    def __init__(
        self,
        name: str,
        min_log_duration: float = 0.0,
        track_performance: bool = True
    ) -> None:
        """Initialize timer context.
        
        Args:
            name: Operation name
            min_log_duration: Minimum duration to trigger logging (seconds)
            track_performance: Whether to record in global performance tracker
        """
        self.name = name
        self.min_log_duration = min_log_duration
        self.track_performance = track_performance
        self.start_time = 0.0
        self.checkpoints: Dict[str, float] = {}
        self.duration = 0.0
    
    def __enter__(self) -> 'TimerContext':
        """Enter the timing context."""
        self.start_time = time.perf_counter()
        logger.debug(f"Timer '{self.name}' started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the timing context."""
        self.duration = time.perf_counter() - self.start_time
        
        # Log if duration exceeds threshold or if there was an exception
        should_log = self.duration >= self.min_log_duration or exc_type is not None
        
        if should_log:
            if exc_type is None:
                logger.info(f"Timer '{self.name}' completed in {self.duration:.3f}s")
            else:
                logger.error(f"Timer '{self.name}' failed after {self.duration:.3f}s")
        
        # Track performance
        if self.track_performance and exc_type is None:
            _performance_tracker.record_execution(self.name, self.duration)
    
    def checkpoint(self, name: str) -> float:
        """Record a checkpoint within the timed operation.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Time elapsed since start
        """
        elapsed = time.perf_counter() - self.start_time
        self.checkpoints[name] = elapsed
        logger.debug(f"Timer '{self.name}' checkpoint '{name}': {elapsed:.3f}s")
        return elapsed
    
    def get_checkpoint_duration(self, name: str) -> Optional[float]:
        """Get duration for a specific checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Checkpoint duration or None if not found
        """
        return self.checkpoints.get(name)


def benchmark(
    func: Callable,
    *args: Any,
    iterations: int = 1,
    warmup: int = 0,
    **kwargs: Any
) -> Dict[str, Any]:
    """Benchmark a function with multiple iterations.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for function
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations (not measured)
        **kwargs: Keyword arguments for function
        
    Returns:
        Dictionary with benchmark results
        
    Example:
        >>> results = benchmark(train_model, X, y, iterations=5, warmup=1)
        >>> print(f"Average time: {results['avg_time']:.3f}s")
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    
    func_name = f"{func.__module__}.{func.__qualname__}"
    logger.info(f"Benchmarking '{func_name}' with {iterations} iterations (warmup: {warmup})")
    
    # Warmup runs
    for i in range(warmup):
        logger.debug(f"Warmup iteration {i+1}/{warmup}")
        func(*args, **kwargs)
    
    # Benchmark runs
    times = []
    for i in range(iterations):
        logger.debug(f"Benchmark iteration {i+1}/{iterations}")
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            times.append(duration)
            
        except Exception as e:
            logger.error(f"Benchmark iteration {i+1} failed: {e}")
            raise
    
    # Calculate statistics
    total_time = sum(times)
    avg_time = total_time / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate standard deviation
    if len(times) > 1:
        variance = sum((t - avg_time) ** 2 for t in times) / (len(times) - 1)
    else:
        variance = 0.0
    std_dev = variance ** 0.5
    
    results = {
        'function': func_name,
        'iterations': iterations,
        'warmup': warmup,
        'times': times,
        'total_time': total_time,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_dev': std_dev,
        'throughput': iterations / total_time if total_time > 0 else 0.0
    }
    
    logger.info(f"Benchmark completed: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
    
    return results


# Convenience functions for global performance tracker
def get_performance_stats(name: Optional[str] = None) -> Dict[str, Any]:
    """Get performance statistics from global tracker.
    
    Args:
        name: Optional specific operation name
        
    Returns:
        Performance statistics dictionary
    """
    return _performance_tracker.get_stats(name)


def reset_performance_stats(name: Optional[str] = None) -> None:
    """Reset performance statistics in global tracker.
    
    Args:
        name: Optional specific operation name to reset
    """
    _performance_tracker.reset_stats(name)


def print_performance_summary() -> None:
    """Print formatted performance summary to console."""
    summary = _performance_tracker.get_summary()
    print(summary)
    logger.info("Performance summary printed")


# Memory monitoring utilities
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")


@contextmanager
def monitor_memory(name: str, log_result: bool = True):
    """Context manager for monitoring memory usage.
    
    Args:
        name: Operation name
        log_result: Whether to log memory usage
        
    Yields:
        Dictionary with memory information (updated during execution)
        
    Note:
        Requires psutil package for memory monitoring
    """
    if not PSUTIL_AVAILABLE:
        logger.warning("Memory monitoring requires psutil package")
        yield {
            'memory_available': False,
            'initial_rss': 0,
            'initial_vms': 0,
            'peak_rss': 0,
            'peak_vms': 0,
            'final_rss': 0,
            'final_vms': 0,
            'rss_delta': 0,
            'vms_delta': 0
        }
        return
    
    process = psutil.Process()
    initial_memory = process.memory_info()
    memory_info = {
        'memory_available': True,
        'initial_rss': initial_memory.rss,
        'initial_vms': initial_memory.vms,
        'peak_rss': initial_memory.rss,
        'peak_vms': initial_memory.vms
    }
    
    try:
        yield memory_info
        
        final_memory = process.memory_info()
        memory_info.update({
            'final_rss': final_memory.rss,
            'final_vms': final_memory.vms,
            'rss_delta': final_memory.rss - initial_memory.rss,
            'vms_delta': final_memory.vms - initial_memory.vms
        })
        
        if log_result:
            rss_mb = memory_info['rss_delta'] / (1024 * 1024)
            logger.info(f"Memory usage for '{name}': {rss_mb:+.1f} MB RSS")
            
    except Exception as e:
        if log_result:
            logger.error(f"Memory monitoring failed for '{name}': {e}")
        raise
