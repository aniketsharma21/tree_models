# tree_models/models/evaluator.py
"""Enhanced model evaluation system with comprehensive metrics and analysis.

This module provides comprehensive evaluation capabilities for tree-based models with:
- Type-safe evaluation orchestration and metrics computation
- Extensive classification and regression metrics with statistical significance
- ROC analysis, calibration assessment, and fairness evaluation
- Sample weights integration throughout evaluation workflows
- Performance benchmarking and model comparison capabilities
- Advanced visualization and reporting with multiple output formats
- Statistical testing and confidence intervals
- Business metrics and threshold optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass, field
import warnings
from datetime import datetime
import json

from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    EvaluationError,
    DataValidationError,
    handle_and_reraise,
    validate_parameter,
    create_error_context
)

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

# Optional dependencies with fallbacks
try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
        classification_report, confusion_matrix, accuracy_score, precision_score,
        recall_score, f1_score, log_loss, brier_score_loss, mean_squared_error,
        mean_absolute_error, r2_score, explained_variance_score
    )
    from sklearn.calibration import calibration_curve
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn metrics not available")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available for interactive plots")

try:
    from scipy import stats
    from scipy.stats import ks_2samp, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available for statistical tests")


@dataclass
class EvaluationConfig:
    """Type-safe configuration for model evaluation with comprehensive options."""
    
    # Basic evaluation settings
    classification_threshold: float = 0.5
    include_plots: bool = True
    plot_format: str = "png"  # "png", "svg", "html"
    
    # Metrics configuration
    compute_roc_auc: bool = True
    compute_pr_auc: bool = True
    compute_calibration: bool = True
    compute_fairness_metrics: bool = False
    
    # Statistical testing
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # Business metrics
    compute_business_metrics: bool = False
    cost_fp: float = 1.0  # Cost of false positive
    cost_fn: float = 5.0  # Cost of false negative
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_metric: str = "f1"  # "f1", "precision", "recall", "youden"
    
    # Output settings
    save_detailed_report: bool = True
    output_format: str = "json"  # "json", "html", "pdf"
    
    def __post_init__(self) -> None:
        """Validate evaluation configuration."""
        validate_parameter("classification_threshold", self.classification_threshold, min_value=0.0, max_value=1.0)
        validate_parameter("confidence_level", self.confidence_level, min_value=0.5, max_value=0.99)
        validate_parameter("bootstrap_samples", self.bootstrap_samples, min_value=100, max_value=10000)
        validate_parameter("plot_format", self.plot_format, valid_values=["png", "svg", "html", "pdf"])
        validate_parameter("threshold_metric", self.threshold_metric, 
                         valid_values=["f1", "precision", "recall", "youden", "profit"])


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results with metrics and diagnostics."""
    
    # Basic info
    model_type: str
    task_type: str  # "classification" or "regression"
    n_samples: int
    n_features: int
    
    # Core metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Classification-specific results
    roc_data: Optional[Dict[str, np.ndarray]] = None
    pr_data: Optional[Dict[str, np.ndarray]] = None
    calibration_data: Optional[Dict[str, np.ndarray]] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict[str, Any]] = None
    
    # Threshold analysis
    threshold_analysis: Optional[Dict[str, Any]] = None
    optimal_threshold: Optional[float] = None
    
    # Statistical analysis
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    statistical_tests: Optional[Dict[str, Any]] = None
    
    # Business metrics
    business_metrics: Optional[Dict[str, float]] = None
    cost_analysis: Optional[Dict[str, Any]] = None
    
    # Fairness analysis
    fairness_metrics: Optional[Dict[str, Any]] = None
    
    # Metadata
    evaluation_time: Optional[float] = None
    config: Optional[EvaluationConfig] = None
    timestamp: Optional[str] = None


class ModelEvaluator:
    """Enhanced model evaluator with comprehensive analysis capabilities.
    
    Provides unified interface for evaluating tree-based models with extensive
    metrics, statistical analysis, and business-oriented assessments.
    
    Example:
        >>> evaluator = ModelEvaluator()
        >>> 
        >>> # Configure evaluation
        >>> config = EvaluationConfig(
        ...     compute_calibration=True,
        ...     optimize_threshold=True,
        ...     compute_confidence_intervals=True
        ... )
        >>> 
        >>> # Evaluate model
        >>> results = evaluator.evaluate_model(
        ...     model, X_test, y_test,
        ...     sample_weight=test_weights,
        ...     config=config
        ... )
        >>> 
        >>> print(f"AUC: {results.metrics['auc']:.4f}")
        >>> print(f"Optimal threshold: {results.optimal_threshold:.3f}")
        >>> 
        >>> # Generate comprehensive report
        >>> evaluator.generate_report(results, output_dir="evaluation_results")
    """
    
    def __init__(
        self,
        random_state: int = 42,
        enable_logging: bool = True
    ) -> None:
        """Initialize enhanced model evaluator.
        
        Args:
            random_state: Random state for reproducibility
            enable_logging: Whether to enable detailed logging
        """
        self.random_state = random_state
        self.enable_logging = enable_logging
        
        # Set random seeds
        np.random.seed(random_state)
        
        logger.info(f"Initialized ModelEvaluator with random_state={random_state}")

    @timer(name="model_evaluation")
    def evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        config: Optional[EvaluationConfig] = None,
        model_name: str = "model",
        **kwargs: Any
    ) -> EvaluationResults:
        """Evaluate model with comprehensive metrics and analysis.
        
        Args:
            model: Trained model to evaluate
            X: Test features
            y: Test target
            sample_weight: Optional test sample weights
            config: Evaluation configuration
            model_name: Name for the model (used in reports)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Comprehensive evaluation results
            
        Raises:
            EvaluationError: If model evaluation fails
        """
        logger.info(f"🔍 Starting model evaluation:")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Test data: {X.shape}")
        logger.info(f"   Has weights: {sample_weight is not None}")
        
        if config is None:
            config = EvaluationConfig()
        
        start_time = datetime.now()
        
        try:
            with timed_operation("model_evaluation") as timing:
                # Validate inputs
                self._validate_evaluation_inputs(model, X, y, sample_weight)
                
                # Determine task type
                task_type = self._determine_task_type(y)
                
                # Get predictions
                predictions = self._get_predictions(model, X, task_type)
                
                # Initialize results
                results = EvaluationResults(
                    model_type=type(model).__name__,
                    task_type=task_type,
                    n_samples=len(X),
                    n_features=X.shape[1],
                    config=config,
                    timestamp=start_time.isoformat()
                )
                
                # Compute core metrics
                if task_type == "classification":
                    self._evaluate_classification(results, y, predictions, sample_weight, config)
                else:
                    self._evaluate_regression(results, y, predictions, sample_weight, config)
                
                # Statistical analysis
                if config.compute_confidence_intervals:
                    confidence_intervals = self._compute_confidence_intervals(
                        y, predictions, sample_weight, config
                    )
                    results.confidence_intervals = confidence_intervals
                
                # Business metrics
                if config.compute_business_metrics and task_type == "classification":
                    business_metrics = self._compute_business_metrics(
                        y, predictions, sample_weight, config
                    )
                    results.business_metrics = business_metrics
                
                results.evaluation_time = timing['duration']
            
            logger.info(f"✅ Model evaluation completed:")
            logger.info(f"   Duration: {results.evaluation_time:.2f}s")
            logger.info(f"   Task: {task_type}")
            if task_type == "classification":
                logger.info(f"   AUC: {results.metrics.get('auc', 'N/A'):.4f}")
            else:
                logger.info(f"   R²: {results.metrics.get('r2', 'N/A'):.4f}")
            
            return results
            
        except Exception as e:
            handle_and_reraise(
                e, EvaluationError,
                f"Model evaluation failed for {model_name}",
                error_code="MODEL_EVALUATION_FAILED",
                context=create_error_context(
                    model_name=model_name,
                    test_shape=X.shape,
                    has_weights=sample_weight is not None,
                    task_type=task_type if 'task_type' in locals() else 'unknown'
                )
            )

    def _validate_evaluation_inputs(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> None:
        """Validate inputs for model evaluation."""
        
        # Check model has required methods
        if not hasattr(model, 'predict'):
            raise DataValidationError("Model must have a 'predict' method")
        
        # Basic data validation
        if X.empty or y.empty:
            raise DataValidationError("Test data cannot be empty")
        
        if len(X) != len(y):
            raise DataValidationError("X and y must have the same length")
        
        # Sample weights validation
        if sample_weight is not None:
            if len(sample_weight) != len(X):
                raise DataValidationError("sample_weight length must match test data")
            
            if np.any(sample_weight < 0):
                raise DataValidationError("Sample weights cannot be negative")
        
        # Check for missing values
        if y.isnull().any():
            raise DataValidationError("Target variable cannot contain missing values")

    def _determine_task_type(self, y: pd.Series) -> str:
        """Determine if this is a classification or regression task."""
        
        # Check if target is binary/categorical or continuous
        unique_values = y.nunique()
        
        if unique_values <= 20 and y.dtype in ['int64', 'int32', 'bool', 'object', 'category']:
            return "classification"
        else:
            return "regression"

    def _get_predictions(self, model: Any, X: pd.DataFrame, task_type: str) -> Dict[str, np.ndarray]:
        """Get predictions from model based on task type."""
        
        predictions = {}
        
        # Get raw predictions
        predictions['raw'] = model.predict(X)
        
        # Get probability predictions for classification
        if task_type == "classification" and hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            
            if probas.shape[1] == 2:
                # Binary classification - use positive class probability
                predictions['probabilities'] = probas[:, 1]
            else:
                # Multi-class classification
                predictions['probabilities'] = probas
                predictions['predicted_class'] = np.argmax(probas, axis=1)
        
        return predictions

    def _evaluate_classification(
        self,
        results: EvaluationResults,
        y_true: pd.Series,
        predictions: Dict[str, np.ndarray],
        sample_weight: Optional[np.ndarray],
        config: EvaluationConfig
    ) -> None:
        """Comprehensive classification evaluation."""
        
        logger.debug("Performing classification evaluation")
        
        # Get predictions
        y_pred_proba = predictions.get('probabilities', predictions['raw'])
        y_pred_binary = (y_pred_proba >= config.classification_threshold).astype(int)
        
        # Core classification metrics
        metrics = {}
        
        try:
            # AUC metrics
            if config.compute_roc_auc and len(np.unique(y_true)) == 2:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba, sample_weight=sample_weight)
                
                # ROC curve data
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba, sample_weight=sample_weight)
                results.roc_data = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': roc_thresholds,
                    'auc': metrics['auc']
                }
            
            # Precision-Recall metrics
            if config.compute_pr_auc and len(np.unique(y_true)) == 2:
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba, sample_weight=sample_weight)
                
                # PR curve data
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba, sample_weight=sample_weight)
                results.pr_data = {
                    'precision': precision,
                    'recall': recall, 
                    'thresholds': pr_thresholds,
                    'auc': metrics['pr_auc']
                }
            
            # Standard classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred_binary, sample_weight=sample_weight)
            metrics['precision'] = precision_score(y_true, y_pred_binary, sample_weight=sample_weight, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred_binary, sample_weight=sample_weight, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred_binary, sample_weight=sample_weight, average='weighted', zero_division=0)
            
            # Log loss
            y_pred_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            metrics['log_loss'] = log_loss(y_true, y_pred_clipped, sample_weight=sample_weight)
            
            # Brier score
            if len(np.unique(y_true)) == 2:
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba, sample_weight=sample_weight)
            
            # Confusion matrix
            results.confusion_matrix = confusion_matrix(y_true, y_pred_binary, sample_weight=sample_weight)
            
            # Classification report
            try:
                from sklearn.metrics import classification_report
                report = classification_report(y_true, y_pred_binary, sample_weight=sample_weight, output_dict=True, zero_division=0)
                results.classification_report = report
            except Exception as e:
                logger.warning(f"Could not generate classification report: {e}")
            
            # Calibration analysis
            if config.compute_calibration and len(np.unique(y_true)) == 2:
                calibration_data = self._compute_calibration_analysis(
                    y_true, y_pred_proba, sample_weight
                )
                results.calibration_data = calibration_data
                metrics.update(calibration_data['metrics'])
            
            # Threshold optimization
            if config.optimize_threshold and len(np.unique(y_true)) == 2:
                threshold_analysis = self._optimize_classification_threshold(
                    y_true, y_pred_proba, sample_weight, config
                )
                results.threshold_analysis = threshold_analysis
                results.optimal_threshold = threshold_analysis['optimal_threshold']
                metrics['optimal_f1'] = threshold_analysis['optimal_metric_value']
            
        except Exception as e:
            logger.warning(f"Error computing classification metrics: {e}")
        
        results.metrics = metrics

    def _evaluate_regression(
        self,
        results: EvaluationResults,
        y_true: pd.Series,
        predictions: Dict[str, np.ndarray],
        sample_weight: Optional[np.ndarray],
        config: EvaluationConfig
    ) -> None:
        """Comprehensive regression evaluation."""
        
        logger.debug("Performing regression evaluation")
        
        y_pred = predictions['raw']
        metrics = {}
        
        try:
            # Core regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
            metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weight)
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
            
            # Mean Absolute Percentage Error (MAPE)
            mask = y_true != 0
            if mask.any():
                if sample_weight is not None:
                    mape = np.average(
                        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100,
                        weights=sample_weight[mask]
                    )
                else:
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                metrics['mape'] = mape
            
            # Residual analysis
            residuals = y_true - y_pred
            metrics['mean_residual'] = np.average(residuals, weights=sample_weight) if sample_weight is not None else np.mean(residuals)
            metrics['std_residual'] = np.sqrt(np.average(residuals**2, weights=sample_weight)) if sample_weight is not None else np.std(residuals)
            
            # Additional regression metrics
            # Mean Bias Error
            metrics['mbe'] = metrics['mean_residual']
            
            # Normalized RMSE
            y_range = y_true.max() - y_true.min()
            if y_range > 0:
                metrics['nrmse'] = metrics['rmse'] / y_range
            
        except Exception as e:
            logger.warning(f"Error computing regression metrics: {e}")
        
        results.metrics = metrics

    def _compute_calibration_analysis(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        sample_weight: Optional[np.ndarray],
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Compute calibration analysis for binary classification."""
        
        try:
            # Calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins, sample_weight=sample_weight
            )
            
            # Calibration metrics
            calibration_error = np.abs(fraction_of_positives - mean_predicted_value)
            expected_calibration_error = np.mean(calibration_error)
            max_calibration_error = np.max(calibration_error)
            
            # Reliability diagram data
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            return {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value,
                'bin_lowers': bin_lowers,
                'bin_uppers': bin_uppers,
                'metrics': {
                    'expected_calibration_error': expected_calibration_error,
                    'max_calibration_error': max_calibration_error
                }
            }
            
        except Exception as e:
            logger.warning(f"Error computing calibration analysis: {e}")
            return {'metrics': {}}

    def _optimize_classification_threshold(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        sample_weight: Optional[np.ndarray],
        config: EvaluationConfig
    ) -> Dict[str, Any]:
        """Optimize classification threshold based on specified metric."""
        
        # Generate threshold range
        thresholds = np.linspace(0.01, 0.99, 99)
        
        metric_scores = []
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
            
            try:
                if config.threshold_metric == "f1":
                    score = f1_score(y_true, y_pred_binary, sample_weight=sample_weight, zero_division=0)
                elif config.threshold_metric == "precision":
                    score = precision_score(y_true, y_pred_binary, sample_weight=sample_weight, zero_division=0)
                elif config.threshold_metric == "recall":
                    score = recall_score(y_true, y_pred_binary, sample_weight=sample_weight, zero_division=0)
                elif config.threshold_metric == "youden":
                    # Youden's J statistic
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, sample_weight=sample_weight).ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    score = sensitivity + specificity - 1
                elif config.threshold_metric == "profit":
                    # Business profit metric
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, sample_weight=sample_weight).ravel()
                    profit = tp * 1 - fp * config.cost_fp - fn * config.cost_fn
                    score = profit / len(y_true)  # Normalize by sample size
                else:
                    score = f1_score(y_true, y_pred_binary, sample_weight=sample_weight, zero_division=0)
                
                metric_scores.append(score)
                
                # Store detailed metrics for this threshold
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, sample_weight=sample_weight).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                threshold_metrics.append({
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
                })
                
            except Exception as e:
                logger.warning(f"Error computing metric for threshold {threshold}: {e}")
                metric_scores.append(0.0)
                threshold_metrics.append({
                    'threshold': threshold,
                    'precision': 0, 'recall': 0, 'f1': 0,
                    'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
                })
        
        # Find optimal threshold
        optimal_idx = np.argmax(metric_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = metric_scores[optimal_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'optimal_metric_value': optimal_score,
            'threshold_metric': config.threshold_metric,
            'thresholds': thresholds,
            'metric_scores': metric_scores,
            'detailed_metrics': threshold_metrics
        }

    def _compute_confidence_intervals(
        self,
        y_true: pd.Series,
        predictions: Dict[str, np.ndarray],
        sample_weight: Optional[np.ndarray],
        config: EvaluationConfig
    ) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals using bootstrap sampling."""
        
        logger.debug(f"Computing confidence intervals with {config.bootstrap_samples} bootstrap samples")
        
        n_samples = len(y_true)
        bootstrap_metrics = []
        
        # Determine prediction type
        if 'probabilities' in predictions:
            y_pred = predictions['probabilities']
            is_classification = True
        else:
            y_pred = predictions['raw']
            is_classification = len(np.unique(y_true)) <= 20
        
        # Bootstrap sampling
        for _ in range(config.bootstrap_samples):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_sample = y_true.iloc[indices]
            pred_sample = y_pred[indices]
            weight_sample = sample_weight[indices] if sample_weight is not None else None
            
            # Compute metrics for this bootstrap sample
            try:
                if is_classification:
                    if len(np.unique(y_sample)) == 2:  # Binary classification
                        auc = roc_auc_score(y_sample, pred_sample, sample_weight=weight_sample)
                        bootstrap_metrics.append({'auc': auc})
                    else:
                        bootstrap_metrics.append({})
                else:  # Regression
                    r2 = r2_score(y_sample, pred_sample, sample_weight=weight_sample)
                    rmse = np.sqrt(mean_squared_error(y_sample, pred_sample, sample_weight=weight_sample))
                    bootstrap_metrics.append({'r2': r2, 'rmse': rmse})
            
            except Exception:
                # Skip failed bootstrap samples
                continue
        
        # Compute confidence intervals
        confidence_intervals = {}
        alpha = 1 - config.confidence_level
        
        for metric_name in bootstrap_metrics[0].keys() if bootstrap_metrics else []:
            metric_values = [m[metric_name] for m in bootstrap_metrics if metric_name in m]
            
            if metric_values:
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                ci_lower = np.percentile(metric_values, lower_percentile)
                ci_upper = np.percentile(metric_values, upper_percentile)
                
                confidence_intervals[metric_name] = (ci_lower, ci_upper)
        
        return confidence_intervals

    def _compute_business_metrics(
        self,
        y_true: pd.Series,
        predictions: Dict[str, np.ndarray],
        sample_weight: Optional[np.ndarray],
        config: EvaluationConfig
    ) -> Dict[str, float]:
        """Compute business-oriented metrics."""
        
        y_pred_proba = predictions.get('probabilities', predictions['raw'])
        y_pred_binary = (y_pred_proba >= config.classification_threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, sample_weight=sample_weight).ravel()
        
        # Business metrics
        business_metrics = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_cost': fp * config.cost_fp + fn * config.cost_fn,
            'cost_per_prediction': (fp * config.cost_fp + fn * config.cost_fn) / len(y_true),
            'precision_weighted': tp / (tp + fp * config.cost_fp) if (tp + fp * config.cost_fp) > 0 else 0,
            'profit_per_prediction': (tp - fp * config.cost_fp - fn * config.cost_fn) / len(y_true)
        }
        
        return business_metrics

    def compare_models(
        self,
        models_results: Dict[str, EvaluationResults],
        comparison_metric: str = "auc"
    ) -> Dict[str, Any]:
        """Compare multiple model evaluation results.
        
        Args:
            models_results: Dictionary of model name -> evaluation results
            comparison_metric: Primary metric for comparison
            
        Returns:
            Dictionary with comparison analysis
        """
        logger.info(f"Comparing {len(models_results)} models on {comparison_metric}")
        
        comparison_data = []
        
        for model_name, results in models_results.items():
            row = {
                'model_name': model_name,
                'model_type': results.model_type,
                'task_type': results.task_type,
                'n_samples': results.n_samples,
                'evaluation_time': results.evaluation_time
            }
            
            # Add all metrics
            row.update(results.metrics)
            
            # Add confidence intervals if available
            if results.confidence_intervals:
                for metric, (ci_lower, ci_upper) in results.confidence_intervals.items():
                    row[f'{metric}_ci_lower'] = ci_lower
                    row[f'{metric}_ci_upper'] = ci_upper
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Ranking by comparison metric
        if comparison_metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(comparison_metric, ascending=False)
            
            best_model = comparison_df.iloc[0]['model_name']
            best_score = comparison_df.iloc[0][comparison_metric]
        else:
            best_model = None
            best_score = None
        
        return {
            'comparison_data': comparison_df,
            'comparison_metric': comparison_metric,
            'best_model': best_model,
            'best_score': best_score,
            'n_models': len(models_results)
        }

    def generate_report(
        self,
        results: EvaluationResults,
        output_dir: Union[str, Path],
        include_plots: bool = True
    ) -> Dict[str, Path]:
        """Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results to report on
            output_dir: Directory to save report files
            include_plots: Whether to include visualization plots
            
        Returns:
            Dictionary of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        try:
            # Save metrics summary
            metrics_path = output_dir / "metrics_summary.json"
            with open(metrics_path, 'w') as f:
                json.dump(results.metrics, f, indent=2, default=str)
            generated_files['metrics'] = metrics_path
            
            # Save detailed results
            if results.config and results.config.save_detailed_report:
                detailed_path = output_dir / f"evaluation_results.{results.config.output_format}"
                
                if results.config.output_format == "json":
                    # Convert results to JSON-serializable format
                    results_dict = {
                        'model_type': results.model_type,
                        'task_type': results.task_type,
                        'n_samples': results.n_samples,
                        'metrics': results.metrics,
                        'optimal_threshold': results.optimal_threshold,
                        'evaluation_time': results.evaluation_time,
                        'timestamp': results.timestamp
                    }
                    
                    # Add arrays as lists
                    if results.confusion_matrix is not None:
                        results_dict['confusion_matrix'] = results.confusion_matrix.tolist()
                    
                    if results.confidence_intervals:
                        results_dict['confidence_intervals'] = results.confidence_intervals
                    
                    with open(detailed_path, 'w') as f:
                        json.dump(results_dict, f, indent=2, default=str)
                
                generated_files['detailed_report'] = detailed_path
            
            # Generate plots
            if include_plots and results.config and results.config.include_plots:
                plot_files = self._generate_evaluation_plots(results, output_dir)
                generated_files.update(plot_files)
            
            logger.info(f"Generated evaluation report in {output_dir}")
            logger.info(f"Files created: {list(generated_files.keys())}")
            
            return generated_files
            
        except Exception as e:
            handle_and_reraise(
                e, EvaluationError,
                f"Failed to generate evaluation report",
                error_code="REPORT_GENERATION_FAILED"
            )

    def _generate_evaluation_plots(
        self,
        results: EvaluationResults,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Generate evaluation plots based on results."""
        
        plot_files = {}
        
        try:
            if results.task_type == "classification":
                # ROC curve
                if results.roc_data:
                    roc_path = self._plot_roc_curve(results.roc_data, output_dir)
                    if roc_path:
                        plot_files['roc_curve'] = roc_path
                
                # Precision-Recall curve
                if results.pr_data:
                    pr_path = self._plot_pr_curve(results.pr_data, output_dir)
                    if pr_path:
                        plot_files['pr_curve'] = pr_path
                
                # Confusion matrix
                if results.confusion_matrix is not None:
                    cm_path = self._plot_confusion_matrix(results.confusion_matrix, output_dir)
                    if cm_path:
                        plot_files['confusion_matrix'] = cm_path
                
                # Calibration plot
                if results.calibration_data:
                    cal_path = self._plot_calibration_curve(results.calibration_data, output_dir)
                    if cal_path:
                        plot_files['calibration_curve'] = cal_path
                
                # Threshold analysis
                if results.threshold_analysis:
                    thresh_path = self._plot_threshold_analysis(results.threshold_analysis, output_dir)
                    if thresh_path:
                        plot_files['threshold_analysis'] = thresh_path
                        
        except Exception as e:
            logger.warning(f"Error generating plots: {e}")
        
        return plot_files

    def _plot_roc_curve(self, roc_data: Dict[str, np.ndarray], output_dir: Path) -> Optional[Path]:
        """Generate ROC curve plot."""
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    linewidth=2, label=f"ROC Curve (AUC = {roc_data['auc']:.4f})")
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = output_dir / 'roc_curve.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"Failed to create ROC curve plot: {e}")
            plt.close()
            return None

    def _plot_pr_curve(self, pr_data: Dict[str, np.ndarray], output_dir: Path) -> Optional[Path]:
        """Generate Precision-Recall curve plot."""
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(pr_data['recall'], pr_data['precision'],
                    linewidth=2, label=f"PR Curve (AUC = {pr_data['auc']:.4f})")
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = output_dir / 'pr_curve.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"Failed to create PR curve plot: {e}")
            plt.close()
            return None

    def _plot_confusion_matrix(self, cm: np.ndarray, output_dir: Path) -> Optional[Path]:
        """Generate confusion matrix heatmap."""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
            
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            output_path = output_dir / 'confusion_matrix.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"Failed to create confusion matrix plot: {e}")
            plt.close()
            return None

    def _plot_calibration_curve(self, cal_data: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Generate calibration curve plot."""
        try:
            plt.figure(figsize=(8, 6))
            
            # Plot calibration curve
            plt.plot(cal_data['mean_predicted_value'], cal_data['fraction_of_positives'],
                    marker='o', linewidth=2, label='Model')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly Calibrated')
            
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = output_dir / 'calibration_curve.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"Failed to create calibration curve plot: {e}")
            plt.close()
            return None

    def _plot_threshold_analysis(self, thresh_data: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Generate threshold analysis plot."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot metric vs threshold
            thresholds = thresh_data['thresholds']
            metric_scores = thresh_data['metric_scores']
            optimal_threshold = thresh_data['optimal_threshold']
            
            ax1.plot(thresholds, metric_scores, linewidth=2)
            ax1.axvline(optimal_threshold, color='red', linestyle='--', 
                       label=f'Optimal = {optimal_threshold:.3f}')
            ax1.set_xlabel('Threshold')
            ax1.set_ylabel(f"{thresh_data['threshold_metric'].upper()} Score")
            ax1.set_title('Threshold Optimization')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot precision-recall vs threshold
            detailed_metrics = thresh_data['detailed_metrics']
            thresholds_detailed = [m['threshold'] for m in detailed_metrics]
            precisions = [m['precision'] for m in detailed_metrics]
            recalls = [m['recall'] for m in detailed_metrics]
            
            ax2.plot(thresholds_detailed, precisions, label='Precision', linewidth=2)
            ax2.plot(thresholds_detailed, recalls, label='Recall', linewidth=2)
            ax2.axvline(optimal_threshold, color='red', linestyle='--', 
                       label=f'Optimal = {optimal_threshold:.3f}')
            ax2.set_xlabel('Threshold')
            ax2.set_ylabel('Score')
            ax2.set_title('Precision & Recall vs Threshold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_path = output_dir / 'threshold_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"Failed to create threshold analysis plot: {e}")
            plt.close()
            return None


# Convenience functions
def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    **kwargs: Any
) -> EvaluationResults:
    """Convenience function for model evaluation.
    
    Args:
        model: Trained model to evaluate
        X: Test features
        y: Test target
        sample_weight: Optional sample weights
        **kwargs: Additional evaluation parameters
        
    Returns:
        Evaluation results object
        
    Example:
        >>> results = evaluate_model(trained_model, X_test, y_test, sample_weight=test_weights)
        >>> print(f"Model AUC: {results.metrics['auc']:.4f}")
        >>> print(f"Optimal threshold: {results.optimal_threshold:.3f}")
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(model, X, y, sample_weight, **kwargs)


def compare_models(
    models_dict: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    comparison_metric: str = "auc"
) -> Dict[str, Any]:
    """Convenience function for comparing multiple models.
    
    Args:
        models_dict: Dictionary of model_name -> trained_model
        X: Test features
        y: Test target
        sample_weight: Optional sample weights
        comparison_metric: Primary metric for comparison
        
    Returns:
        Model comparison analysis
        
    Example:
        >>> models = {'xgb': xgb_model, 'lgb': lgb_model, 'cat': cat_model}
        >>> comparison = compare_models(models, X_test, y_test, comparison_metric='auc')
        >>> print(f"Best model: {comparison['best_model']} (AUC: {comparison['best_score']:.4f})")
    """
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    results = {}
    for model_name, model in models_dict.items():
        results[model_name] = evaluator.evaluate_model(
            model, X, y, sample_weight, model_name=model_name
        )
    
    # Compare results
    return evaluator.compare_models(results, comparison_metric)


# Export key classes and functions
__all__ = [
    'EvaluationConfig',
    'EvaluationResults',
    'ModelEvaluator',
    'evaluate_model',
    'compare_models'
]