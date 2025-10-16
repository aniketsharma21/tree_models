"""Tree Models - High-Level API Components.

This module provides high-level API functions and workflows for common
ML tasks, making the framework easy to use with sensible defaults.

Key Components:
- Workflows: Complete end-to-end analysis pipelines
- Quick Functions: Fast utility functions for common tasks
- Info: Package information and help utilities

Example:
    >>> from tree_models.api import complete_model_analysis
    >>> results = complete_model_analysis(
    ...     model_type='xgboost',
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test
    ... )
"""

# High-level workflow functions
from .workflows import (
    complete_model_analysis,
    fraud_detection_pipeline
)

# Quick utility functions
from .quick_functions import (
    tune_hyperparameters,
    quick_shap_analysis,
    quick_robustness_test,
    convert_to_scorecard
)

# Package information
from .info import (
    show_package_info,
    get_version,
    get_supported_models,
    get_available_scorers
)

__all__ = [
    # Workflow functions
    'complete_model_analysis',
    'fraud_detection_pipeline',
    
    # Quick utility functions
    'tune_hyperparameters',
    'quick_shap_analysis',
    'quick_robustness_test',
    'convert_to_scorecard',
    
    # Package info functions
    'show_package_info',
    'get_version',
    'get_supported_models',
    'get_available_scorers'
]