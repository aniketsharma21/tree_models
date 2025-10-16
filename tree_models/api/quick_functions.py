"""Quick utility functions for common ML tasks.

This module provides fast, easy-to-use functions for common machine learning
tasks with sensible defaults and minimal configuration required.
"""

from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.exceptions import ConfigurationError

logger = get_logger(__name__)


def tune_hyperparameters(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    scoring_function: str = "roc_auc",
    n_trials: int = 100,
    cv_folds: int = 5,
    **kwargs: Any
) -> Tuple[Dict[str, Any], float]:
    """Quick hyperparameter tuning with sensible defaults.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
        X: Training features
        y: Training targets
        sample_weight: Optional sample weights
        scoring_function: Scoring metric to optimize
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        **kwargs: Additional tuning parameters
        
    Returns:
        Tuple of (best_parameters, best_score)
        
    Example:
        >>> best_params, score = tune_hyperparameters(
        ...     model_type='xgboost',
        ...     X=X_train, y=y_train,
        ...     scoring_function='recall',
        ...     n_trials=50
        ... )
    """
    from ..models.trainer import StandardModelTrainer
    from ..models.tuner import OptunaHyperparameterTuner
    
    logger.info(f"🎯 Quick hyperparameter tuning: {model_type} with {n_trials} trials")
    
    # Create trainer and tuner
    trainer = StandardModelTrainer(model_type, **kwargs)
    tuner = OptunaHyperparameterTuner(
        trainer, 
        scoring_metric=scoring_function,
        cv_folds=cv_folds,
        **kwargs
    )
    
    # Get default search space for the model type
    search_space = _get_default_search_space(model_type)
    
    # Run optimization
    best_params, best_score = tuner.optimize(
        X=X, y=y,
        search_space=search_space,
        sample_weight=sample_weight,
        n_trials=n_trials
    )
    
    logger.info(f"✅ Best {scoring_function}: {best_score:.4f}")
    return best_params, best_score


def quick_shap_analysis(
    model: Any,
    X: pd.DataFrame,
    sample_weight: Optional[np.ndarray] = None,
    max_samples: int = 1000,
    save_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Quick SHAP analysis with automatic configuration.
    
    Args:
        model: Trained model to explain
        X: Data to generate explanations for
        sample_weight: Optional sample weights
        max_samples: Maximum number of samples to analyze
        save_dir: Optional directory to save plots
        **kwargs: Additional SHAP parameters
        
    Returns:
        Dictionary with SHAP analysis results
        
    Example:
        >>> shap_results = quick_shap_analysis(
        ...     model=trained_model,
        ...     X=X_test,
        ...     max_samples=500
        ... )
    """
    from ..explainability.shap_explainer import SHAPExplainer
    
    logger.info(f"🔍 Quick SHAP analysis for {len(X)} samples (max {max_samples})")
    
    # Limit samples for performance
    if len(X) > max_samples:
        X_sample = X.iloc[:max_samples]
        weight_sample = sample_weight[:max_samples] if sample_weight is not None else None
    else:
        X_sample = X
        weight_sample = sample_weight
    
    # Create explainer with automatic type detection
    explainer = SHAPExplainer(model, explainer_type='auto', **kwargs)
    
    # Generate explanations
    results = explainer.explain(
        X_sample, 
        sample_weight=weight_sample,
        **kwargs
    )
    
    # Save plots if directory specified
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Generate and save summary plot
        explainer.plot_summary(results.shap_values, save_path=save_path / "shap_summary.png")
        
        # Generate and save waterfall plot for first instance
        if len(X_sample) > 0:
            explainer.plot_waterfall(
                results.shap_values, 
                instance_idx=0,
                save_path=save_path / "shap_waterfall_0.png"
            )
    
    logger.info("✅ SHAP analysis completed")
    return results


def quick_robustness_test(
    model_type: str,
    best_params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    n_seeds: int = 5,
    scoring_function: str = "roc_auc",
    **kwargs: Any
) -> Dict[str, Any]:
    """Quick robustness testing with multiple seeds.
    
    Args:
        model_type: Type of model to test
        best_params: Best parameters from hyperparameter tuning
        X: Training features
        y: Training targets
        sample_weight: Optional sample weights
        n_seeds: Number of different random seeds to test
        scoring_function: Scoring metric to evaluate
        **kwargs: Additional test parameters
        
    Returns:
        Dictionary with robustness test results
        
    Example:
        >>> robustness = quick_robustness_test(
        ...     model_type='xgboost',
        ...     best_params=best_params,
        ...     X=X_train, y=y_train,
        ...     n_seeds=10
        ... )
    """
    from ..models.trainer import StandardModelTrainer
    from ..models.robustness import SeedRobustnessTester
    
    logger.info(f"🛡️ Quick robustness test: {n_seeds} seeds for {model_type}")
    
    # Create trainer and tester
    trainer = StandardModelTrainer(model_type, **kwargs)
    tester = SeedRobustnessTester(
        n_seeds=n_seeds,
        scoring_function=scoring_function,
        **kwargs
    )
    
    # Run robustness test
    results = tester.test_robustness(
        model_trainer=trainer,
        X=X, y=y,
        sample_weight=sample_weight,
        best_params=best_params
    )
    
    logger.info(f"✅ Robustness test completed")
    return results


def convert_to_scorecard(
    probabilities: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    base_score: int = 600,
    pdo: int = 20,
    **kwargs: Any
) -> Tuple[np.ndarray, Any]:
    """Quick conversion of probabilities to business scorecard.
    
    Args:
        probabilities: Model prediction probabilities
        sample_weight: Optional sample weights
        base_score: Base scorecard score
        pdo: Points to double odds
        **kwargs: Additional scorecard parameters
        
    Returns:
        Tuple of (scorecard_scores, converter_object)
        
    Example:
        >>> scores, converter = convert_to_scorecard(
        ...     probabilities=y_pred_proba,
        ...     base_score=650,
        ...     pdo=25
        ... )
    """
    from ..explainability.scorecard import ScorecardConverter
    
    logger.info(f"📊 Converting {len(probabilities)} probabilities to scorecard")
    
    # Create converter
    converter = ScorecardConverter(
        base_score=base_score,
        pdo=pdo,
        **kwargs
    )
    
    # Convert probabilities to scores
    scores = converter.fit_transform(
        probabilities,
        sample_weight=sample_weight
    )
    
    logger.info(f"✅ Scorecard conversion completed")
    logger.info(f"   Score range: {scores.min():.0f} - {scores.max():.0f}")
    
    return scores, converter


def quick_feature_importance(
    model: Any,
    feature_names: Optional[list] = None,
    importance_type: str = "gain",
    top_n: int = 20,
    **kwargs: Any
) -> pd.DataFrame:
    """Quick feature importance analysis.
    
    Args:
        model: Trained model
        feature_names: Optional feature names
        importance_type: Type of importance ('gain', 'weight', 'cover')
        top_n: Number of top features to return
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with feature importance scores
        
    Example:
        >>> importance_df = quick_feature_importance(
        ...     model=trained_model,
        ...     feature_names=X.columns.tolist(),
        ...     top_n=15
        ... )
    """
    logger.info(f"📈 Quick feature importance analysis (top {top_n})")
    
    # Get feature importance from model
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        # For CatBoost
        importance_scores = model.get_feature_importance(type=importance_type)
    elif hasattr(model, 'feature_importance'):
        # For LightGBM
        importance_scores = model.feature_importance(importance_type=importance_type)
    else:
        raise ConfigurationError(f"Model type {type(model)} does not support feature importance")
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    
    # Sort by importance and take top N
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    logger.info(f"✅ Feature importance analysis completed")
    return importance_df


def quick_model_comparison(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    metrics: Optional[list] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """Quick comparison of multiple trained models.
    
    Args:
        models: Dictionary of {model_name: model_object}
        X_test: Test features
        y_test: Test targets
        sample_weight: Optional sample weights
        metrics: List of metrics to compute
        **kwargs: Additional evaluation parameters
        
    Returns:
        DataFrame comparing model performance
        
    Example:
        >>> comparison = quick_model_comparison(
        ...     models={'xgb': xgb_model, 'lgb': lgb_model},
        ...     X_test=X_test, y_test=y_test
        ... )
    """
    from ..models.evaluator import StandardModelEvaluator
    
    logger.info(f"⚖️ Quick model comparison: {len(models)} models")
    
    # Default metrics if none provided
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Create evaluator
    evaluator = StandardModelEvaluator(metrics=metrics, **kwargs)
    
    # Evaluate each model
    results = []
    for name, model in models.items():
        eval_result = evaluator.evaluate(
            model=model,
            X=X_test,
            y=y_test,
            sample_weight=sample_weight
        )
        
        # Add model name to metrics
        model_metrics = {'model': name, **eval_result.metrics}
        results.append(model_metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model')
    
    logger.info(f"✅ Model comparison completed")
    return comparison_df


def _get_default_search_space(model_type: str) -> Dict[str, Any]:
    """Get default hyperparameter search space for model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dictionary defining parameter search space
    """
    if model_type.lower() == 'xgboost':
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 12},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0}
        }
    elif model_type.lower() == 'lightgbm':
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 12},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'num_leaves': {'type': 'int', 'low': 20, 'high': 200}
        }
    elif model_type.lower() == 'catboost':
        return {
            'iterations': {'type': 'int', 'low': 50, 'high': 500},
            'depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'l2_leaf_reg': {'type': 'float', 'low': 1, 'high': 10},
            'border_count': {'type': 'int', 'low': 32, 'high': 255}
        }
    else:
        raise ConfigurationError(f"Unknown model type: {model_type}")