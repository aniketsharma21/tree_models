"""Base class for model evaluators.

This module provides the abstract base class that defines the interface
for all model evaluation components in the framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ...utils.exceptions import ConfigurationError, ModelEvaluationError
from ...utils.logger import get_logger
from .protocols import ModelProtocol

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Standardized evaluation result container.
    
    Provides a consistent interface for evaluation results across
    all model types and evaluation metrics.
    
    Attributes:
        predictions: Model predictions
        probabilities: Model prediction probabilities (if available)
        metrics: Dictionary of computed metrics
        evaluation_time: Time taken for evaluation (seconds)
        metadata: Additional evaluation information
        confusion_matrix: Optional confusion matrix for classification
    """
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    metrics: Dict[str, float]
    evaluation_time: float
    metadata: Dict[str, Any]
    confusion_matrix: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """Validate evaluation result after initialization."""
        if self.evaluation_time < 0:
            raise ValueError("Evaluation time cannot be negative")
        
        if not isinstance(self.metrics, dict):
            raise TypeError("Metrics must be a dictionary")
        
        if not isinstance(self.metadata, dict):
            raise TypeError("Metadata must be a dictionary")
        
        # Validate predictions array
        if not isinstance(self.predictions, np.ndarray):
            raise TypeError("Predictions must be a numpy array")
        
        # Validate probabilities if present
        if self.probabilities is not None:
            if not isinstance(self.probabilities, np.ndarray):
                raise TypeError("Probabilities must be a numpy array")
            
            if len(self.predictions) != len(self.probabilities):
                raise ValueError(
                    "Predictions and probabilities must have same length"
                )


class BaseModelEvaluator(ABC):
    """Abstract base class for model evaluation.
    
    Defines the interface for evaluating trained models with
    different metrics and validation strategies.
    
    This class provides common functionality for model evaluation
    and enforces a consistent interface across implementations.
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize model evaluator.
        
        Args:
            metrics: List of evaluation metrics to compute
            **kwargs: Additional evaluator-specific parameters
        """
        self.metrics = metrics or ["accuracy", "precision", "recall", "f1", "roc_auc"]
        self.config = kwargs
        self._supported_metrics = self._get_supported_metrics()
        
        # Validate requested metrics
        self._validate_metrics()
        
        logger.info(f"Initialized {self.__class__.__name__} with metrics: {self.metrics}")
    
    @abstractmethod
    def evaluate(
        self,
        model: ModelProtocol,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate a trained model on test data.
        
        Args:
            model: Trained model to evaluate
            X: Test features
            y: Test targets
            sample_weight: Optional sample weights
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with evaluation metrics
            
        Raises:
            ModelEvaluationError: If evaluation fails
        """
        pass
    
    def validate_model(self, model: ModelProtocol) -> None:
        """Validate that model implements required interface.
        
        Args:
            model: Model to validate
            
        Raises:
            ConfigurationError: If model validation fails
        """
        self._check_required_methods(model)
        self._check_probabilistic_methods(model)
    
    def validate_evaluation_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> None:
        """Validate input data for evaluation.
        
        Args:
            X: Test features
            y: Test targets
            sample_weight: Optional sample weights
            
        Raises:
            ConfigurationError: If data validation fails
        """
        self._validate_dataframes(X, y)
        self._validate_sample_weights(sample_weight, len(X))
        self._validate_target_distribution(y)
    
    def _check_required_methods(self, model: ModelProtocol) -> None:
        """Check for required model methods.
        
        Args:
            model: Model to check
            
        Raises:
            ConfigurationError: If required methods are missing
        """
        required_methods = ['fit', 'predict']
        
        for method_name in required_methods:
            if not hasattr(model, method_name):
                raise ConfigurationError(
                    f"Model must implement {method_name} method"
                )
    
    def _check_probabilistic_methods(self, model: ModelProtocol) -> None:
        """Check for probabilistic methods if needed.
        
        Args:
            model: Model to check
        """
        probabilistic_metrics = ["roc_auc", "log_loss", "average_precision", "brier_score"]
        
        requested_prob_metrics = [
            metric for metric in self.metrics 
            if metric in probabilistic_metrics
        ]
        
        if requested_prob_metrics and not hasattr(model, 'predict_proba'):
            logger.warning(
                f"Model does not implement predict_proba but probabilistic "
                f"metrics requested: {requested_prob_metrics}. "
                f"These metrics will be skipped."
            )
    
    def _validate_dataframes(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate DataFrame and Series inputs.
        
        Args:
            X: Test features
            y: Test targets
            
        Raises:
            ConfigurationError: If validation fails
        """
        if X.empty:
            raise ConfigurationError("Test data X cannot be empty")
        
        if y.empty:
            raise ConfigurationError("Test targets y cannot be empty")
        
        if len(X) != len(y):
            raise ConfigurationError(
                f"X and y must have same length: {len(X)} vs {len(y)}"
            )
    
    def _validate_sample_weights(
        self, 
        sample_weight: Optional[np.ndarray], 
        expected_length: int
    ) -> None:
        """Validate sample weights if provided.
        
        Args:
            sample_weight: Sample weights array
            expected_length: Expected length of weights
            
        Raises:
            ConfigurationError: If validation fails
        """
        if sample_weight is None:
            return
            
        if len(sample_weight) != expected_length:
            raise ConfigurationError(
                f"sample_weight must have same length as data: "
                f"{len(sample_weight)} vs {expected_length}"
            )
        
        if np.any(sample_weight < 0):
            raise ConfigurationError("sample_weight cannot contain negative values")
        
        if np.any(np.isnan(sample_weight)):
            raise ConfigurationError("sample_weight cannot contain NaN values")
    
    def _validate_target_distribution(self, y: pd.Series) -> None:
        """Validate target distribution for evaluation.
        
        Args:
            y: Test targets
        """
        unique_values = y.unique()
        
        # Check for single class
        if len(unique_values) == 1:
            logger.warning(
                f"Test set contains only one class: {unique_values[0]}. "
                f"Some metrics may not be computable."
            )
        
        # Check class distribution for classification
        if y.dtype in ['int64', 'int32', 'bool'] or len(unique_values) <= 10:
            class_counts = y.value_counts().to_dict()
            logger.info(f"Class distribution: {class_counts}")
    
    def _validate_metrics(self) -> None:
        """Validate that requested metrics are supported.
        
        Raises:
            ConfigurationError: If unsupported metrics are requested
        """
        unsupported = set(self.metrics) - set(self._supported_metrics)
        
        if unsupported:
            raise ConfigurationError(
                f"Unsupported metrics: {list(unsupported)}. "
                f"Supported metrics: {self._supported_metrics}"
            )
    
    def _get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics.
        
        Returns:
            List of supported metric names
        """
        return [
            # Classification metrics
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
            'average_precision', 'balanced_accuracy', 'matthews_corrcoef',
            'log_loss', 'brier_score', 'cohen_kappa',
            
            # Regression metrics
            'mae', 'mse', 'rmse', 'r2', 'mean_absolute_percentage_error',
            'explained_variance', 'max_error', 'median_absolute_error'
        ]
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available metrics.
        
        Returns:
            Dictionary mapping metric names to descriptions
        """
        return {
            # Classification metrics
            'accuracy': 'Fraction of predictions that match true labels',
            'precision': 'True positives / (True positives + False positives)',
            'recall': 'True positives / (True positives + False negatives)',
            'f1': 'Harmonic mean of precision and recall',
            'roc_auc': 'Area under the ROC curve',
            'average_precision': 'Area under the precision-recall curve',
            'balanced_accuracy': 'Average of recall for each class',
            'matthews_corrcoef': 'Matthews correlation coefficient',
            'log_loss': 'Logarithmic loss (cross-entropy)',
            'brier_score': 'Mean squared difference between predicted probabilities and outcomes',
            'cohen_kappa': 'Cohen\'s kappa coefficient',
            
            # Regression metrics
            'mae': 'Mean absolute error',
            'mse': 'Mean squared error',
            'rmse': 'Root mean squared error',
            'r2': 'Coefficient of determination (R-squared)',
            'mean_absolute_percentage_error': 'Mean absolute percentage error',
            'explained_variance': 'Explained variance score',
            'max_error': 'Maximum absolute error',
            'median_absolute_error': 'Median absolute error'
        }
    
    def __repr__(self) -> str:
        """String representation of the evaluator."""
        return (
            f"{self.__class__.__name__}("
            f"metrics={self.metrics})"
        )