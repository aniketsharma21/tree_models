"""Src - Professional Fraud Detection & Machine Learning Package.

A comprehensive toolkit for fraud detection model development including:
- Advanced hyperparameter tuning with custom scoring functions
- Model explainability (SHAP, scorecards, reason codes)
- Robustness testing (seed stability, sensitivity analysis, drift detection)
- Data preprocessing and feature engineering
- Complete end-to-end model development pipelines

Key Features:
- ⚖️ Full sample weights integration for imbalanced datasets
- 🎯 Optimized for fraud detection use cases
- 🔍 Comprehensive explainability for regulatory compliance
- 🛡️ Production-ready robustness testing
- 📊 Business-friendly scorecards and reason codes
- 🚀 One-line complete analysis pipelines

Quick Start:
    >>> from src.models import fraud_detection_pipeline
    >>> import xgboost as xgb
    >>> 
    >>> # Complete fraud detection pipeline in one call
    >>> results = fraud_detection_pipeline(
    ...     model_class=xgb.XGBClassifier,
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     sample_weight_train=weights_train,
    ...     focus='recall',  # Maximize fraud detection
    ...     output_dir='fraud_model_analysis'
    ... )

Advanced Usage:
    >>> from src.models import (
    ...     complete_model_analysis,
    ...     EnhancedOptunaHyperparameterTuner,
    ...     ModelExplainerSuite,
    ...     SeedRobustnessTester
    ... )
    >>> 
    >>> # Step-by-step with full control
    >>> tuner = EnhancedOptunaHyperparameterTuner(
    ...     model_type='xgboost',
    ...     scoring_config=scoring_config
    ... )
    >>> best_params, best_score = tuner.optimize(X, y, sample_weight=weights)
"""

# Package metadata
__version__ = '1.0.0'
__author__ = 'Fraud Detection ML Team'
__license__ = 'MIT'
__description__ = 'Professional fraud detection and ML modeling toolkit'

# Import subpackages - this makes them accessible
from . import models
from . import utils
from . import data

# Import most commonly used items to top level for convenience
from .models import (
    # Hyperparameter tuning
    EnhancedOptunaHyperparameterTuner,
    tune_hyperparameters,

    # Explainability
    ModelExplainerSuite,
    quick_shap_analysis,
    convert_to_scorecard,

    # Robustness
    quick_robustness_test,
    calculate_psi_simple,

    # Complete workflows (MOST IMPORTANT FOR USERS)
    complete_model_analysis,
    fraud_detection_pipeline,

    # Pre-configured settings
    CommonScoringConfigs,
)

# Import essential utilities
from .utils import (
    get_logger,
    timer,
    quick_setup,
)

# Public API - what users should use
__all__ = [
    # Subpackages
    'models',
    'utils',
    'data',

    # Complete workflows (TOP PRIORITY)
    'complete_model_analysis',
    'fraud_detection_pipeline',

    # Hyperparameter tuning
    'EnhancedOptunaHyperparameterTuner',
    'tune_hyperparameters',
    'CommonScoringConfigs',

    # Explainability
    'ModelExplainerSuite',
    'quick_shap_analysis',
    'convert_to_scorecard',

    # Robustness
    'quick_robustness_test',
    'calculate_psi_simple',

    # Utilities
    'get_logger',
    'timer',
    'quick_setup',

    # Metadata
    '__version__',
]


def show_package_info():
    """Display package information and quick start guide."""
    info = f"""
    ╔════════════════════════════════════════════════════════════════════╗
    ║              SRC - FRAUD DETECTION ML TOOLKIT v{__version__}              ║
    ╚════════════════════════════════════════════════════════════════════╝

    📦 PACKAGE STRUCTURE:
    ────────────────────
    • models/    : Hyperparameter tuning, explainability, robustness
    • utils/     : Logging, timing, I/O utilities
    • data/      : Data preprocessing and feature engineering

    🚀 QUICK START - ONE LINE COMPLETE ANALYSIS:
    ─────────────────────────────────────────────
    >>> from src import fraud_detection_pipeline
    >>> results = fraud_detection_pipeline(
    ...     model_class=xgb.XGBClassifier,
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     sample_weight_train=weights,
    ...     focus='recall'  # or 'precision', 'balanced', 'pr_auc'
    ... )

    🎯 KEY FEATURES:
    ────────────────
    ✅ Custom scoring functions (recall, precision, F1, PR-AUC)
    ✅ SHAP explainability with business scorecards
    ✅ Reason code generation for compliance
    ✅ Seed robustness testing (multi-seed training)
    ✅ Sensitivity analysis (feature perturbation)
    ✅ Drift detection (Population Stability Index)
    ✅ Sample weights support throughout

    📚 MAIN MODULES:
    ────────────────
    1️⃣ HYPERPARAMETER TUNING:
       from src.models import tune_hyperparameters
       best_params, best_score = tune_hyperparameters(
           'xgboost', X, y, scoring_function='recall',
           additional_metrics=['precision', 'f1'], 
           sample_weight=weights
       )

    2️⃣ MODEL EXPLAINABILITY:
       from src.models import quick_shap_analysis, convert_to_scorecard
       shap_results = quick_shap_analysis(model, X_test, sample_weight=weights)
       scores, converter = convert_to_scorecard(probabilities)

    3️⃣ ROBUSTNESS TESTING:
       from src.models import quick_robustness_test, calculate_psi_simple
       robustness = quick_robustness_test(xgb.XGBClassifier, params, X, y)
       psi_results = calculate_psi_simple(X_train, X_test)

    4️⃣ COMPLETE PIPELINE:
       from src import complete_model_analysis
       results = complete_model_analysis(
           xgb.XGBClassifier, params, 
           X_train, y_train, X_test, y_test
       )

    💡 PRE-CONFIGURED FRAUD DETECTION:
    ──────────────────────────────────
    from src.models import CommonScoringConfigs

    # Optimize for maximum fraud detection (recall)
    config = CommonScoringConfigs.fraud_detection_recall()

    # Optimize for minimum false alarms (precision)
    config = CommonScoringConfigs.fraud_detection_precision()

    # Balanced optimization (F1)
    config = CommonScoringConfigs.fraud_detection_balanced()

    # Best for imbalanced datasets (PR-AUC)
    config = CommonScoringConfigs.fraud_detection_pr_auc()

    📖 For detailed documentation:
       • help(src.models)
       • help(src.models.complete_model_analysis)
       • help(src.models.fraud_detection_pipeline)

    🌐 Support: See package documentation or contact ML team
    """
    print(info)


def get_version():
    """Get package version."""
    return __version__


def get_available_models():
    """Get list of available model types for tuning."""
    return ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'extra_trees']


def get_available_scorers():
    """Get list of available scoring functions."""
    return [
        'roc_auc', 'average_precision', 'accuracy', 'precision', 
        'recall', 'f1', 'balanced_accuracy', 'matthews_corrcoef',
        'neg_log_loss', 'cohen_kappa'
    ]


# Add utility functions to public API
__all__.extend(['show_package_info', 'get_version', 'get_available_models', 'get_available_scorers'])


# Optional: Validate environment on import
def _check_environment():
    """Check if required packages are available."""
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
            f"Missing recommended packages: {', '.join(missing)}. "
            "Some features may not work correctly.",
            ImportWarning
        )


# Perform environment check
_check_environment()


# Optional: Show info on import (can be disabled by user)
# Uncomment to show package info on every import:
# show_package_info()
