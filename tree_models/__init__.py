"""Tree Models - Production-Ready ML Framework for Tree-Based Models.

A comprehensive machine learning framework for tree-based models with:
- Advanced hyperparameter tuning with Optuna
- Comprehensive explainability (SHAP, scorecards, reason codes)
- Robustness testing and stability analysis
- Type-safe configuration system
- MLOps integration with MLflow
- Production-ready features

Quick Start:
    >>> import tree_models as tm
    >>> 
    >>> # Complete analysis in one line
    >>> results = tm.complete_model_analysis(
    ...     model_type='xgboost',
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     sample_weight_train=weights,
    ...     scoring_function='recall',
    ...     n_trials=100
    ... )
    >>> 
    >>> # Fraud detection pipeline
    >>> fraud_results = tm.fraud_detection_pipeline(
    ...     model_type='xgboost',
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     focus='recall'
    ... )

Documentation: https://tree-models.readthedocs.io
Repository: https://github.com/aniketsharma21/tree_models
"""

# Package metadata
__version__ = "2.0.0"
__author__ = "Tree Models Development Team"
__email__ = "tree-models@company.com"
__license__ = "MIT"
__description__ = "Production-ready ML framework for tree-based models"
__url__ = "https://github.com/aniketsharma21/tree_models"

# Configure package-level logging
from .utils.logger import configure_logging, get_logger

# Configure with sensible defaults
configure_logging(
    level="INFO",
    format_style="detailed",
    include_console=True
)

logger = get_logger(__name__)
logger.info(f"Tree Models v{__version__} initialized")

# High-level API imports - core workflows
from .api.workflows import (
    complete_model_analysis,
    fraud_detection_pipeline
)

# Quick utility functions
from .api.quick_functions import (
    tune_hyperparameters,
    quick_shap_analysis,
    quick_robustness_test,
    convert_to_scorecard,
    quick_feature_importance,
    quick_model_comparison
)

# Package information functions
from .api.info import (
    show_package_info,
    get_version,
    get_supported_models,
    get_available_scorers,
    get_environment_info,
    show_installation_guide,
    check_installation
)

# Essential classes for direct instantiation
from .models.trainer import StandardModelTrainer
from .models.evaluator import StandardModelEvaluator
from .models.tuner import OptunaHyperparameterTuner, ScoringConfig
from .explainability.shap_explainer import SHAPExplainer
from .explainability.scorecard import ScorecardConverter
from .config.model_config import ModelConfig, XGBoostConfig, LightGBMConfig, CatBoostConfig
from .data.validator import DataValidator

# Utility imports
from .utils.logger import get_logger as get_logger_util
from .utils.timer import timer, timed_operation, benchmark
from .utils.exceptions import (
    TreeModelsError,
    ModelTrainingError,
    ConfigurationError,
    DataValidationError
)

# MLOps integration
from .tracking.mlflow_tracker import MLflowTracker

# Public API definition - what users should import
__all__ = [
    # High-level workflows
    'complete_model_analysis',
    'fraud_detection_pipeline',
    
    # Quick utility functions
    'tune_hyperparameters',
    'quick_shap_analysis',
    'quick_robustness_test',
    'convert_to_scorecard',
    'quick_feature_importance',
    'quick_model_comparison',
    
    # Package information
    'show_package_info',
    'get_version',
    'get_supported_models',
    'get_available_scorers',
    'get_environment_info',
    'show_installation_guide',
    'check_installation',
    
    # Core classes
    'StandardModelTrainer',
    'StandardModelEvaluator',
    'OptunaHyperparameterTuner',
    'ScoringConfig',
    'SHAPExplainer',
    'ScorecardConverter',
    'DataValidator',
    'MLflowTracker',
    
    # Configuration
    'ModelConfig',
    'XGBoostConfig',
    'LightGBMConfig',
    'CatBoostConfig',
    
    # Utilities
    'get_logger_util',
    'timer',
    'timed_operation',
    'benchmark',
    
    # Exceptions
    'TreeModelsError',
    'ModelTrainingError',
    'ConfigurationError',
    'DataValidationError',
    
    # Metadata
    '__version__',
    '__author__',
    '__license__',
    '__description__',
    '__url__'
]


# Backward compatibility - deprecated functions with warnings
def _deprecated_warning(old_func: str, new_func: str) -> None:
    """Show deprecation warning for old function usage."""
    import warnings
    warnings.warn(
        f"'{old_func}' is deprecated. Use '{new_func}' instead. "
        f"The old function will be removed in v3.0.0.",
        DeprecationWarning,
        stacklevel=3
    )


# Environment validation on import
def _validate_environment() -> None:
    """Validate that required packages are available."""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    
    try:
        import pandas
    except ImportError:
        missing.append('pandas')
    
    try:
        import sklearn
    except ImportError:
        missing.append('scikit-learn')
    
    if missing:
        import warnings
        warnings.warn(
            f"Missing required packages: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}",
            ImportWarning
        )
        logger.warning(f"Missing packages: {missing}")


# Check optional ML libraries
def _check_ml_libraries() -> None:
    """Check availability of optional ML libraries."""
    ml_libs = []
    
    try:
        import xgboost
        ml_libs.append('xgboost')
    except ImportError:
        pass
    
    try:
        import lightgbm
        ml_libs.append('lightgbm')
    except ImportError:
        pass
    
    try:
        import catboost
        ml_libs.append('catboost')
    except ImportError:
        pass
    
    if not ml_libs:
        import warnings
        warnings.warn(
            "No ML libraries (XGBoost, LightGBM, CatBoost) found. "
            "Install at least one with: pip install xgboost lightgbm catboost",
            ImportWarning
        )
        logger.warning("No ML libraries available - limited functionality")
    else:
        logger.info(f"Available ML libraries: {', '.join(ml_libs)}")


# Perform startup checks
_validate_environment()
_check_ml_libraries()

# Startup message
logger.info(
    f"Tree Models v{__version__} ready - "
    f"{len(get_supported_models())} models, {len(get_available_scorers())} scorers"
)
logger.info("Quick start: tm.show_package_info() or help(tm.complete_model_analysis)")

# Type annotations for dynamic imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np