# tree_models/__init__.py
"""Tree Models - Production-Ready ML Framework for Tree-Based Models.

A comprehensive machine learning framework for tree-based models with:
- Advanced hyperparameter tuning with Optuna
- Comprehensive explainability (SHAP, scorecards, reason codes)
- Robustness testing and stability analysis
- Type-safe configuration system
- MLOps integration with MLflow
- Production-ready features

Key Features:
- ⚖️ Full sample weights integration for imbalanced datasets
- 🎯 Optimized for fraud detection and risk modeling
- 🔍 Comprehensive explainability for regulatory compliance
- 🛡️ Production-ready robustness testing
- 📊 Business-friendly scorecards and reason codes
- 🚀 One-line complete analysis pipelines
- 🔧 Type-safe configuration with comprehensive validation

Quick Start:
    >>> import tree_models as tm
    >>> from tree_models.models import tune_hyperparameters
    >>> 
    >>> # Quick hyperparameter tuning
    >>> best_params, best_score = tune_hyperparameters(
    ...     'xgboost', X_train, y_train,
    ...     sample_weight=weights,
    ...     scoring_function='recall',
    ...     n_trials=100
    ... )
    >>> 
    >>> # Complete pipeline
    >>> from tree_models import complete_model_analysis
    >>> results = complete_model_analysis(
    ...     model_type='xgboost',
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     sample_weight_train=weights
    ... )

Advanced Usage:
    >>> from tree_models.models import OptunaHyperparameterTuner, StandardModelTrainer
    >>> from tree_models.explainability import SHAPExplainer, ScorecardConverter
    >>> from tree_models.config import ModelConfig
    >>> 
    >>> # Configure and train
    >>> config = ModelConfig.for_fraud_detection()
    >>> trainer = StandardModelTrainer('xgboost', config=config)
    >>> tuner = OptunaHyperparameterTuner(trainer, n_trials=200)
    >>> 
    >>> # Optimize and explain
    >>> best_params, score = tuner.optimize(X_train, y_train, sample_weight=weights)
    >>> model = trainer.get_model(best_params)
    >>> model.fit(X_train, y_train, sample_weight=weights)
    >>> 
    >>> # Explain predictions
    >>> explainer = SHAPExplainer(model)
    >>> shap_values = explainer.compute_shap_values(X_test)
    >>> 
    >>> # Convert to business scorecard
    >>> converter = ScorecardConverter()
    >>> y_pred_proba = model.predict_proba(X_test)[:, 1]
    >>> scores = converter.fit_transform(y_pred_proba, sample_weight=test_weights)
"""

# Package metadata
__version__ = "2.0.0"
__author__ = "Tree Models Development Team"
__email__ = "tree-models@company.com"
__license__ = "MIT"
__description__ = "Production-ready ML framework for tree-based models"
__url__ = "https://github.com/company/tree_models"

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

# Core imports - make key functionality available at package level
# Model training and tuning
from .models.trainer import StandardModelTrainer
from .models.tuner import (
    OptunaHyperparameterTuner,
    ScoringConfig,
    tune_hyperparameters
)
from .models.evaluator import StandardModelEvaluator
from .models.feature_selector import (
    RFECVFeatureSelector,
    BorutaFeatureSelector,
    ConsensusFeatureSelector
)
from .models.robustness import (
    SeedRobustnessTester,
    SensitivityAnalyzer,
    PopulationStabilityIndex,
    quick_robustness_test
)

# Explainability 
from .explainability.shap_explainer import SHAPExplainer, quick_shap_analysis
from .explainability.scorecard import ScorecardConverter, convert_to_scorecard
from .explainability.reason_codes import ReasonCodeGenerator
from .explainability.partial_dependence import PartialDependencePlotter

# Configuration system
from .config.model_config import (
    ModelConfig,
    XGBoostConfig,
    LightGBMConfig,
    CatBoostConfig
)
from .config.data_config import DataConfig, FeatureConfig
from .config.loader import load_config, save_config

# Data processing
from .data.validator import DataValidator, validate_dataset
from .data.preprocessor import DataPreprocessor

# Utilities
from .utils.logger import get_logger, configure_logging, set_log_level
from .utils.timer import timer, timed_operation, benchmark
from .utils.exceptions import (
    TreeModelsError,
    ModelTrainingError,
    ConfigurationError,
    DataValidationError
)

# Tracking and MLOps
from .tracking.mlflow_tracker import MLflowTracker

# Public API definition - what users should import
__all__ = [
    # Core workflow functions
    'complete_model_analysis',
    'fraud_detection_pipeline',
    
    # Model training and tuning
    'StandardModelTrainer',
    'OptunaHyperparameterTuner', 
    'ScoringConfig',
    'tune_hyperparameters',
    'StandardModelEvaluator',
    
    # Feature selection
    'RFECVFeatureSelector',
    'BorutaFeatureSelector', 
    'ConsensusFeatureSelector',
    
    # Robustness testing
    'SeedRobustnessTester',
    'SensitivityAnalyzer',
    'PopulationStabilityIndex',
    'quick_robustness_test',
    
    # Explainability
    'SHAPExplainer',
    'quick_shap_analysis',
    'ScorecardConverter',
    'convert_to_scorecard',
    'ReasonCodeGenerator',
    'PartialDependencePlotter',
    
    # Configuration
    'ModelConfig',
    'XGBoostConfig',
    'LightGBMConfig', 
    'CatBoostConfig',
    'DataConfig',
    'FeatureConfig',
    'load_config',
    'save_config',
    
    # Data processing
    'DataValidator',
    'validate_dataset',
    'DataPreprocessor',
    
    # Utilities
    'get_logger',
    'configure_logging',
    'set_log_level',
    'timer',
    'timed_operation',
    'benchmark',
    
    # Exceptions
    'TreeModelsError',
    'ModelTrainingError',
    'ConfigurationError',
    'DataValidationError',
    
    # Tracking
    'MLflowTracker',
    
    # Metadata
    '__version__',
]


def complete_model_analysis(
    model_type: str,
    X_train: 'pd.DataFrame',
    y_train: 'pd.Series',
    X_test: 'pd.DataFrame',
    y_test: 'pd.Series',
    sample_weight_train: Optional['np.ndarray'] = None,
    sample_weight_test: Optional['np.ndarray'] = None,
    config: Optional[ModelConfig] = None,
    n_trials: int = 100,
    scoring_function: str = "roc_auc",
    output_dir: str = "model_analysis",
    **kwargs
) -> Dict[str, Any]:
    """Complete end-to-end model analysis pipeline.
    
    Performs hyperparameter tuning, model training, evaluation, explainability
    analysis, and robustness testing in a single function call.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        sample_weight_train: Optional training sample weights
        sample_weight_test: Optional test sample weights
        config: Optional model configuration
        n_trials: Number of hyperparameter tuning trials
        scoring_function: Primary scoring metric
        output_dir: Directory to save analysis results
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary containing all analysis results
        
    Example:
        >>> import pandas as pd
        >>> import tree_models as tm
        >>> 
        >>> # Load your data
        >>> X_train, X_test, y_train, y_test = load_your_data()
        >>> weights_train = calculate_sample_weights(y_train)
        >>> 
        >>> # Run complete analysis
        >>> results = tm.complete_model_analysis(
        ...     model_type='xgboost',
        ...     X_train=X_train, y_train=y_train,
        ...     X_test=X_test, y_test=y_test,
        ...     sample_weight_train=weights_train,
        ...     scoring_function='recall',
        ...     n_trials=50,
        ...     output_dir='fraud_model_analysis'
        ... )
        >>> 
        >>> print(f"Best score: {results['tuning']['best_score']:.4f}")
        >>> print(f"Best parameters: {results['tuning']['best_params']}")
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from typing import Dict, Any, Optional
    
    logger.info(f"🚀 Starting complete model analysis for {model_type}")
    logger.info(f"   Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"   Testing: {X_test.shape[0]} samples")
    logger.info(f"   Scoring: {scoring_function}, Trials: {n_trials}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'metadata': {
            'model_type': model_type,
            'scoring_function': scoring_function,
            'n_trials': n_trials,
            'output_dir': str(output_path),
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'has_sample_weights': sample_weight_train is not None
        }
    }
    
    try:
        # Step 1: Hyperparameter Tuning
        logger.info("📊 Step 1/5: Hyperparameter tuning")
        with timed_operation("hyperparameter_tuning") as timing:
            best_params, best_score = tune_hyperparameters(
                model_type=model_type,
                X=X_train,
                y=y_train,
                sample_weight=sample_weight_train,
                scoring_function=scoring_function,
                n_trials=n_trials,
                **kwargs
            )
        
        results['tuning'] = {
            'best_params': best_params,
            'best_score': best_score,
            'duration': timing['duration']
        }
        logger.info(f"   ✅ Best {scoring_function}: {best_score:.4f}")
        
        # Step 2: Model Training with Best Parameters
        logger.info("🎯 Step 2/5: Final model training")
        with timed_operation("model_training") as timing:
            trainer = StandardModelTrainer(model_type, config=config)
            model = trainer.get_model(best_params)
            
            if sample_weight_train is not None:
                model.fit(X_train, y_train, sample_weight=sample_weight_train)
            else:
                model.fit(X_train, y_train)
        
        results['training'] = {
            'model': model,
            'duration': timing['duration']
        }
        logger.info(f"   ✅ Model trained in {timing['duration']:.2f}s")
        
        # Step 3: Model Evaluation
        logger.info("📈 Step 3/5: Model evaluation")
        with timed_operation("model_evaluation") as timing:
            evaluator = StandardModelEvaluator()
            eval_result = evaluator.evaluate(
                model, X_test, y_test, sample_weight_test
            )
        
        results['evaluation'] = {
            'metrics': eval_result.metrics,
            'predictions': eval_result.predictions,
            'probabilities': eval_result.probabilities,
            'duration': timing['duration']
        }
        logger.info(f"   ✅ Test {scoring_function}: {eval_result.metrics.get(scoring_function, 'N/A')}")
        
        # Step 4: Explainability Analysis
        logger.info("🔍 Step 4/5: Explainability analysis")
        with timed_operation("explainability") as timing:
            # SHAP analysis
            shap_results = quick_shap_analysis(
                model, X_test[:1000] if len(X_test) > 1000 else X_test,
                sample_weight=sample_weight_test[:1000] if sample_weight_test is not None and len(X_test) > 1000 else sample_weight_test,
                save_dir=output_path / "explainability"
            )
            
            # Scorecard conversion
            y_pred_proba = eval_result.probabilities
            if y_pred_proba is not None:
                scores, converter = convert_to_scorecard(y_pred_proba, sample_weight_test)
            else:
                scores, converter = None, None
        
        results['explainability'] = {
            'shap_results': shap_results,
            'scorecard_scores': scores,
            'scorecard_converter': converter,
            'duration': timing['duration']
        }
        logger.info(f"   ✅ Explainability completed")
        
        # Step 5: Robustness Testing
        logger.info("🛡️ Step 5/5: Robustness testing")
        with timed_operation("robustness") as timing:
            robustness_results = quick_robustness_test(
                model_type=model_type,
                best_params=best_params,
                X=X_train[:5000] if len(X_train) > 5000 else X_train,  # Sample for speed
                y=y_train[:5000] if len(y_train) > 5000 else y_train,
                sample_weight=sample_weight_train[:5000] if sample_weight_train is not None and len(X_train) > 5000 else sample_weight_train,
                n_seeds=5,
                scoring_function=scoring_function
            )
        
        results['robustness'] = {
            **robustness_results,
            'duration': timing['duration']
        }
        logger.info(f"   ✅ Robustness testing completed")
        
        # Save results summary
        summary_path = output_path / "analysis_summary.json"
        summary = {
            'metadata': results['metadata'],
            'tuning_summary': {
                'best_score': results['tuning']['best_score'],
                'best_params': results['tuning']['best_params']
            },
            'evaluation_summary': results['evaluation']['metrics'],
            'robustness_summary': {
                k: v for k, v in results['robustness'].items() 
                if isinstance(v, (int, float, str, bool, type(None)))
            }
        }
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"🎉 Complete model analysis finished!")
        logger.info(f"   📁 Results saved to: {output_path}")
        logger.info(f"   📋 Summary saved to: {summary_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Complete model analysis failed: {e}")
        raise


def fraud_detection_pipeline(
    model_type: str,
    X_train: 'pd.DataFrame', 
    y_train: 'pd.Series',
    X_test: 'pd.DataFrame',
    y_test: 'pd.Series',
    sample_weight_train: Optional['np.ndarray'] = None,
    sample_weight_test: Optional['np.ndarray'] = None,
    focus: str = "recall",
    output_dir: str = "fraud_detection_analysis",
    **kwargs
) -> Dict[str, Any]:
    """Specialized pipeline optimized for fraud detection use cases.
    
    Provides fraud-specific configurations, evaluation metrics, and
    analysis focused on maximizing fraud detection performance.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
        X_train: Training features
        y_train: Training targets (0=legitimate, 1=fraud)
        X_test: Test features
        y_test: Test targets
        sample_weight_train: Optional training sample weights
        sample_weight_test: Optional test sample weights  
        focus: Optimization focus ('recall', 'precision', 'balanced', 'pr_auc')
        output_dir: Directory to save analysis results
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary containing fraud-specific analysis results
        
    Example:
        >>> # Fraud detection with focus on maximizing fraud detection
        >>> results = tm.fraud_detection_pipeline(
        ...     model_type='xgboost',
        ...     X_train=X_train, y_train=y_train,
        ...     X_test=X_test, y_test=y_test,
        ...     sample_weight_train=fraud_weights,
        ...     focus='recall',  # Maximize fraud detection
        ...     output_dir='fraud_model_v2'
        ... )
    """
    # Map focus to appropriate scoring function
    focus_mapping = {
        'recall': 'recall',
        'precision': 'precision', 
        'balanced': 'f1',
        'pr_auc': 'average_precision'
    }
    
    if focus not in focus_mapping:
        raise ConfigurationError(f"Invalid focus '{focus}'. Must be one of: {list(focus_mapping.keys())}")
    
    scoring_function = focus_mapping[focus]
    
    logger.info(f"🕵️ Starting fraud detection pipeline (focus: {focus})")
    
    # Use fraud-specific configuration
    if model_type == 'xgboost':
        config = XGBoostConfig.for_fraud_detection()
    elif model_type == 'lightgbm':
        config = LightGBMConfig.for_fraud_detection()
    elif model_type == 'catboost':
        config = CatBoostConfig.for_fraud_detection()
    else:
        config = None
        logger.warning(f"No fraud-specific config available for {model_type}")
    
    # Add fraud-specific evaluation metrics
    additional_metrics = [
        'accuracy', 'precision', 'recall', 'f1', 
        'roc_auc', 'average_precision', 'balanced_accuracy'
    ]
    
    # Run complete analysis with fraud-specific settings
    results = complete_model_analysis(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train, 
        X_test=X_test,
        y_test=y_test,
        sample_weight_train=sample_weight_train,
        sample_weight_test=sample_weight_test,
        config=config,
        scoring_function=scoring_function,
        output_dir=output_dir,
        additional_metrics=additional_metrics,
        **kwargs
    )
    
    # Add fraud-specific analysis
    results['fraud_analysis'] = {
        'focus': focus,
        'fraud_rate_train': float(y_train.mean()),
        'fraud_rate_test': float(y_test.mean()),
        'class_balance': {
            'legitimate': int((y_train == 0).sum()),
            'fraud': int((y_train == 1).sum())
        }
    }
    
    # Calculate fraud-specific metrics if we have probabilities
    if results['evaluation']['probabilities'] is not None:
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        y_pred_proba = results['evaluation']['probabilities']
        
        # Precision-Recall curve analysis
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Find optimal threshold for different objectives
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_f1_idx = np.argmax(f1_scores)
        
        results['fraud_analysis']['optimal_thresholds'] = {
            'f1_score': {
                'threshold': float(pr_thresholds[optimal_f1_idx]) if optimal_f1_idx < len(pr_thresholds) else 0.5,
                'precision': float(precision[optimal_f1_idx]),
                'recall': float(recall[optimal_f1_idx]),
                'f1': float(f1_scores[optimal_f1_idx])
            }
        }
        
    logger.info(f"🎯 Fraud detection pipeline completed!")
    logger.info(f"   Focus: {focus} ({scoring_function})")
    logger.info(f"   Fraud rate: {results['fraud_analysis']['fraud_rate_test']:.2%}")
    
    return results


def show_package_info() -> None:
    """Display comprehensive package information and usage guide."""
    info = f"""
    ╔════════════════════════════════════════════════════════════════════╗
    ║           TREE MODELS - ML FRAMEWORK v{__version__}                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    
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
    • tree_models.models        - Training, tuning, evaluation
    • tree_models.explainability - SHAP, scorecards, reason codes
    • tree_models.data          - Data validation and preprocessing
    • tree_models.config        - Type-safe configuration system
    • tree_models.tracking      - MLflow experiment tracking
    • tree_models.utils         - Logging, timing, error handling
    
    🔧 CONFIGURATION:
    ─────────────────
    >>> from tree_models.config import XGBoostConfig
    >>> config = XGBoostConfig.for_fraud_detection()
    >>> config.n_estimators = 500
    >>> config.max_depth = 8
    
    📊 HYPERPARAMETER TUNING:
    ─────────────────────────
    >>> from tree_models.models import tune_hyperparameters
    >>> best_params, score = tune_hyperparameters(
    ...     'xgboost', X_train, y_train,
    ...     scoring_function='recall',
    ...     n_trials=100,
    ...     sample_weight=weights
    ... )
    
    🔍 EXPLAINABILITY:
    ──────────────────
    >>> from tree_models.explainability import quick_shap_analysis
    >>> shap_results = quick_shap_analysis(model, X_test)
    >>> 
    >>> from tree_models.explainability import convert_to_scorecard
    >>> scores, converter = convert_to_scorecard(probabilities)
    
    🛡️ ROBUSTNESS TESTING:
    ───────────────────────
    >>> from tree_models.models import quick_robustness_test
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
    
    📖 Documentation: {__url__}
    🐛 Issues: {__url__}/issues
    📧 Support: {__email__}
    """
    print(info)


def get_version() -> str:
    """Get package version string."""
    return __version__


def get_supported_models() -> List[str]:
    """Get list of supported model types."""
    return ['xgboost', 'lightgbm', 'catboost']


def get_available_scorers() -> List[str]:
    """Get list of available scoring functions.""" 
    return [
        'roc_auc', 'average_precision', 'accuracy', 'precision',
        'recall', 'f1', 'balanced_accuracy', 'matthews_corrcoef',
        'neg_log_loss', 'cohen_kappa'
    ]


# Add convenience functions to public API
__all__.extend([
    'show_package_info',
    'get_version', 
    'get_supported_models',
    'get_available_scorers'
])

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
    
    try:
        import optuna
    except ImportError:
        missing.append('optuna')
    
    if missing:
        import warnings
        warnings.warn(
            f"Missing required packages: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}",
            ImportWarning
        )
        logger.warning(f"Missing packages: {missing}")


# Perform environment validation
_validate_environment()

# Optional startup message
logger.info(f"Tree Models v{__version__} ready - {len(get_supported_models())} models, {len(get_available_scorers())} scorers")
logger.info(f"Quick start: tm.show_package_info() or help(tm.complete_model_analysis)")


# Type annotations for dynamic imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np