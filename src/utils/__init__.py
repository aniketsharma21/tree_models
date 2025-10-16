"""Utility functions for the fraud detection modeling package.

This module provides essential utilities for:
- Logging and monitoring
- Timing and performance tracking
- File I/O operations
- Configuration management

All utilities support the broader modeling pipeline with:
- Consistent logging across modules
- Performance monitoring decorators
- Safe file operations
- JSON/pickle serialization
"""

import os
from pathlib import Path
from typing import Any, Optional, Union
import logging

# Import all utility modules
from .logger import (
    get_logger,
    setup_logger,
    set_log_level,
    LoggerConfig
)

from .timer import (
    timer,
    TimerContext,
    time_function,
    benchmark
)

# Try to import io_utils if it exists
try:
    from .io_utils import (
        save_json,
        load_json,
        save_pickle,
        load_pickle,
        ensure_dir,
        safe_file_operation,
        get_file_size,
        list_files
    )
    IO_UTILS_AVAILABLE = True
except ImportError:
    # Define minimal versions if io_utils doesn't exist
    IO_UTILS_AVAILABLE = False

    def ensure_dir(path: Union[str, Path]) -> Path:
        """Ensure directory exists."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(data: Any, filepath: Union[str, Path], **kwargs):
        """Save data to JSON file."""
        import json
        filepath = Path(filepath)
        ensure_dir(filepath.parent)
        with open(filepath, 'w') as f:
            json.dump(data, f, **kwargs)

    def load_json(filepath: Union[str, Path]):
        """Load data from JSON file."""
        import json
        with open(filepath, 'r') as f:
            return json.load(f)

# Version information
__version__ = '1.0.0'
__author__ = 'Fraud Detection ML Team'

# Public API
__all__ = [
    # Logger functions
    'get_logger',
    'setup_logger',
    'set_log_level',
    'LoggerConfig',

    # Timer functions
    'timer',
    'TimerContext',
    'time_function',
    'benchmark',

    # I/O functions (if available)
    'save_json',
    'load_json',
    'ensure_dir',
]

# Add optional I/O functions to __all__ if available
if IO_UTILS_AVAILABLE:
    __all__.extend([
        'save_pickle',
        'load_pickle',
        'safe_file_operation',
        'get_file_size',
        'list_files',
    ])


# Convenience function for quick logger setup
def quick_setup(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """Quick logger setup for common use cases.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional log file path

    Returns:
        Configured logger instance

    Example:
        >>> from src.utils import quick_setup
        >>> logger = quick_setup(log_level='DEBUG', log_file='fraud_model.log')
        >>> logger.info('Starting model training...')
    """
    return setup_logger('src', level=log_level, log_file=log_file)


# Add quick_setup to public API
__all__.append('quick_setup')


# Display available utilities on import (optional, can be removed)
def show_available_utils():
    """Show available utility functions."""
    print("📦 SRC Utils Module")
    print("=" * 50)
    print("✅ Available utilities:")
    print("   📝 Logging: get_logger, setup_logger, set_log_level")
    print("   ⏱️  Timing: timer, TimerContext, benchmark")
    print(f"   💾 I/O: {'Enabled' if IO_UTILS_AVAILABLE else 'Basic only'}")
    if IO_UTILS_AVAILABLE:
        print("      • save_json, load_json, save_pickle, load_pickle")
        print("      • ensure_dir, safe_file_operation, list_files")
    else:
        print("      • save_json, load_json, ensure_dir (basic)")
    print()


# Optional: Show available utils when imported
# Uncomment to enable:
# show_available_utils()
