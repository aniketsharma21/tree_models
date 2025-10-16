"""Package information and help utilities.

This module provides functions to display package information,
versioning, and help content for users.
"""

from typing import List
import sys

# Import version from the main package
try:
    from .. import __version__, __author__, __email__, __url__
except ImportError:
    # Fallback values if not available
    __version__ = "2.0.0"
    __author__ = "Tree Models Development Team"
    __email__ = "tree-models@company.com"
    __url__ = "https://github.com/company/tree_models"


def show_package_info() -> None:
    """Display comprehensive package information and usage guide."""
    info = f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           TREE MODELS - ML FRAMEWORK v{__version__}                    ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    🚀 QUICK START:
    ──────────────
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
    ...     focus='recall'  # Maximize fraud detection
    ... )
    
    🎯 KEY FEATURES:
    ────────────────
    ✅ Advanced hyperparameter tuning with Optuna
    ✅ Comprehensive explainability (SHAP, scorecards, reason codes)
    ✅ Robustness testing and stability analysis  
    ✅ Sample weights support throughout
    ✅ Type-safe configuration system
    ✅ MLOps integration with MLflow
    ✅ Production-ready error handling
    ✅ Performance monitoring and optimization
    
    📚 MAIN MODULES:
    ────────────────
    • tree_models.api          - High-level workflows and quick functions
    • tree_models.models       - Training, tuning, evaluation
    • tree_models.explainability - SHAP, scorecards, reason codes
    • tree_models.data         - Data validation and preprocessing
    • tree_models.config       - Type-safe configuration system
    • tree_models.tracking     - MLflow experiment tracking
    • tree_models.utils        - Logging, timing, error handling
    
    🔧 CONFIGURATION:
    ─────────────────
    >>> from tree_models.config import XGBoostConfig
    >>> config = XGBoostConfig.for_fraud_detection()
    >>> config.n_estimators = 500
    >>> config.max_depth = 8
    
    📊 HYPERPARAMETER TUNING:
    ─────────────────────────
    >>> from tree_models.api import tune_hyperparameters
    >>> best_params, score = tune_hyperparameters(
    ...     'xgboost', X_train, y_train,
    ...     scoring_function='recall',
    ...     n_trials=100,
    ...     sample_weight=weights
    ... )
    
    🔍 EXPLAINABILITY:
    ──────────────────
    >>> from tree_models.api import quick_shap_analysis, convert_to_scorecard
    >>> shap_results = quick_shap_analysis(model, X_test)
    >>> scores, converter = convert_to_scorecard(probabilities)
    
    🛡️ ROBUSTNESS TESTING:
    ───────────────────
    >>> from tree_models.api import quick_robustness_test
    >>> robustness = quick_robustness_test(
    ...     model_type='xgboost',
    ...     best_params=params,
    ...     X=X_train, y=y_train,
    ...     n_seeds=10
    ... )
    
    📈 SUPPORTED MODELS:
    ────────────────────
    • XGBoost        - Gradient boosting with advanced features
    • LightGBM       - Fast gradient boosting with GPU support
    • CatBoost       - Gradient boosting with categorical features
    
    💡 TIPS:
    ────────
    • Use sample weights for imbalanced datasets
    • Set random_state for reproducible results
    • Enable logging for production monitoring
    • Use fraud_detection_pipeline() for fraud use cases
    • Configure MLflow for experiment tracking
    
    📞 SUPPORT:
    ───────────
    📆 Documentation: {__url__}
    🐛 Issues: {__url__}/issues
    📧 Email: {__email__}
    
    🚀 Python {sys.version.split()[0]} | Tree Models v{__version__}
    """
    print(info)


def get_version() -> str:
    """Get package version string.
    
    Returns:
        Package version string
        
    Example:
        >>> version = get_version()
        >>> print(f"Using Tree Models v{version}")
    """
    return __version__


def get_supported_models() -> List[str]:
    """Get list of supported model types.
    
    Returns:
        List of supported model type strings
        
    Example:
        >>> models = get_supported_models()
        >>> print(f"Supported models: {', '.join(models)}")
    """
    return ['xgboost', 'lightgbm', 'catboost']


def get_available_scorers() -> List[str]:
    """Get list of available scoring functions.
    
    Returns:
        List of available scoring function names
        
    Example:
        >>> scorers = get_available_scorers()
        >>> print(f"Available scorers: {', '.join(scorers)}")
    """
    return [
        'roc_auc', 'average_precision', 'accuracy', 'precision',
        'recall', 'f1', 'balanced_accuracy', 'matthews_corrcoef',
        'neg_log_loss', 'cohen_kappa'
    ]


def get_environment_info() -> dict:
    """Get comprehensive environment information.
    
    Returns:
        Dictionary with environment details
        
    Example:
        >>> env_info = get_environment_info()
        >>> print(f"Python: {env_info['python_version']}")
    """
    env_info = {
        'tree_models_version': __version__,
        'python_version': sys.version.split()[0],
        'platform': sys.platform,
        'supported_models': get_supported_models(),
        'available_scorers': get_available_scorers()
    }
    
    # Check for optional dependencies
    optional_deps = {}
    
    try:
        import xgboost
        optional_deps['xgboost'] = xgboost.__version__
    except ImportError:
        optional_deps['xgboost'] = 'Not installed'
    
    try:
        import lightgbm
        optional_deps['lightgbm'] = lightgbm.__version__
    except ImportError:
        optional_deps['lightgbm'] = 'Not installed'
    
    try:
        import catboost
        optional_deps['catboost'] = catboost.__version__
    except ImportError:
        optional_deps['catboost'] = 'Not installed'
    
    try:
        import shap
        optional_deps['shap'] = shap.__version__
    except ImportError:
        optional_deps['shap'] = 'Not installed'
    
    try:
        import mlflow
        optional_deps['mlflow'] = mlflow.__version__
    except ImportError:
        optional_deps['mlflow'] = 'Not installed'
    
    env_info['optional_dependencies'] = optional_deps
    
    return env_info


def show_installation_guide() -> None:
    """Display installation guide for the package."""
    guide = f"""
    📦 TREE MODELS INSTALLATION GUIDE
    ═══════════════════════════════════
    
    🚀 QUICK INSTALL:
    ───────────────
    pip install tree-models
    
    🔧 DEVELOPMENT INSTALL:
    ───────────────────────
    git clone https://github.com/your-username/tree-models.git
    cd tree-models
    pip install -e ".[dev,test,ml]"
    
    ⚙️ OPTIONAL DEPENDENCIES:
    ─────────────────────────
    # For XGBoost support
    pip install xgboost
    
    # For LightGBM support  
    pip install lightgbm
    
    # For CatBoost support
    pip install catboost
    
    # For explainability
    pip install shap
    
    # For experiment tracking
    pip install mlflow
    
    # Install all optional dependencies
    pip install tree-models[all]
    
    🗺️ INSTALLATION OPTIONS:
    ─────────────────────────
    pip install tree-models[minimal]      # Core functionality only
    pip install tree-models[ml]           # + ML frameworks
    pip install tree-models[explainability] # + SHAP
    pip install tree-models[mlops]        # + MLflow
    pip install tree-models[viz]          # + Plotly
    pip install tree-models[dev]          # + Development tools
    pip install tree-models[all]          # Everything
    
    ✅ VERIFY INSTALLATION:
    ───────────────────────
    python -c "import tree_models; tree_models.show_package_info()"
    """
    print(guide)


def check_installation() -> None:
    """Check installation status and show any issues."""
    print("🔍 Checking Tree Models installation...\n")
    
    # Get environment info
    env_info = get_environment_info()
    
    print(f"✅ Tree Models v{env_info['tree_models_version']} installed")
    print(f"✅ Python {env_info['python_version']} ({env_info['platform']})")
    
    # Check optional dependencies
    print("\n📦 Optional Dependencies:")
    for name, version in env_info['optional_dependencies'].items():
        if version != 'Not installed':
            print(f"  ✅ {name}: {version}")
        else:
            print(f"  ❌ {name}: Not installed")
    
    # Check for any missing critical dependencies
    missing_critical = []
    for dep in ['xgboost', 'lightgbm', 'catboost']:
        if env_info['optional_dependencies'][dep] == 'Not installed':
            missing_critical.append(dep)
    
    if missing_critical:
        print(f"\n⚠️  Warning: No ML frameworks installed ({', '.join(missing_critical)})")
        print("   Install at least one with: pip install xgboost lightgbm catboost")
    else:
        print("\n🎉 Installation looks good!")
    
    print(f"\n📚 For help: tree_models.show_package_info()")