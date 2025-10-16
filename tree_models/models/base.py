# tree_models/models/base.py
"""Abstract base classes for tree_models package.

This module defines the core interfaces and abstract base classes that
provide extensibility and standardization across all model-related
functionality in the package.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.exceptions import (
    ModelTrainingError, 
    ModelEvaluationError,
    ConfigurationError,
    handle_and_reraise
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Standardized training result container.
    
    Provides a consistent interface for training results across
    all model types and training strategies.
    """
    model: Any
    best_params: Dict[str, Any]
    best_score: float
    training_time: float
    metadata: Dict[str, Any]
    validation_scores: Optional[Dict[str, float]] = None
    feature_importance: Optional[pd.DataFrame] = None


@dataclass
class EvaluationResult:
    """Standardized evaluation result container.
    
    Provides a consistent interface for evaluation results across
    all model types and evaluation metrics.
    """
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    metrics: Dict[str, float]
    evaluation_time: float
    metadata: Dict[str, Any]
    confusion_matrix: Optional[np.ndarray] = None


class ModelProtocol(Protocol):
    """Protocol defining the interface for ML models.
    
    This protocol ensures that all models used in the framework
    implement the required methods for training and prediction.
    """
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> 'ModelProtocol':
        """Fit the model to training data."""
        ...
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input data."""
        ...
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities (for classifiers)."""
        ...


class BaseModelTrainer(ABC):
    """Abstract base class for model trainers.
    
    Defines the interface for training models with different strategies
    such as basic training, cross-validation, or hyperparameter tuning.
    """
    
    def __init__(
        self,
        model_type: str,
        random_state: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize model trainer.
        
        Args:
            model_type: Type of model to train
            random_state: Random state for reproducibility
            **kwargs: Additional trainer-specific parameters
        """
        self.model_type = model_type
        self.random_state = random_state
        self.config = kwargs
        self.is_fitted = False
        
        logger.info(f"Initialized {self.__class__.__name__} for {model_type}")
    
    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> TrainingResult:
        """Train a model on the provided data.
        
        Args:
            X: Training features
            y: Training targets
            sample_weight: Optional sample weights
            **kwargs: Additional training parameters
            
        Returns:
            TrainingResult with training information
            
        Raises:
            ModelTrainingError: If training fails
        """
        pass
    
    @abstractmethod
    def get_model(self, params: Optional[Dict[str, Any]] = None) -> ModelProtocol:
        """Get a model instance with specified parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            Initialized model instance
        """
        pass
    
    def validate_input_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> None:
        """Validate input data for training.
        
        Args:
            X: Training features
            y: Training targets  
            sample_weight: Optional sample weights
            
        Raises:
            ConfigurationError: If data validation fails
        """
        if X.empty:
            raise ConfigurationError("Training data X cannot be empty")
        
        if y.empty:
            raise ConfigurationError("Training targets y cannot be empty")
        
        if len(X) != len(y):
            raise ConfigurationError(
                f"X and y must have same length: {len(X)} vs {len(y)}"
            )
        
        if sample_weight is not None:
            if len(sample_weight) != len(X):
                raise ConfigurationError(
                    f"sample_weight must have same length as X: {len(sample_weight)} vs {len(X)}"
                )
            
            if np.any(sample_weight < 0):
                raise ConfigurationError("sample_weight cannot contain negative values")
        
        # Check for missing values
        if X.isnull().any().any():
            null_cols = X.columns[X.isnull().any()].tolist()
            raise ConfigurationError(f"X contains null values in columns: {null_cols}")
        
        if y.isnull().any():
            raise ConfigurationError("y contains null values")


class BaseHyperparameterTuner(ABC):
    """Abstract base class for hyperparameter tuning.
    
    Defines the interface for different hyperparameter optimization
    strategies such as grid search, random search, or Bayesian optimization.
    """
    
    def __init__(
        self,
        model_trainer: BaseModelTrainer,
        scoring_metric: str = "roc_auc",
        cv_folds: int = 5,
        **kwargs: Any
    ) -> None:
        """Initialize hyperparameter tuner.
        
        Args:
            model_trainer: Model trainer instance
            scoring_metric: Primary metric for optimization
            cv_folds: Number of cross-validation folds
            **kwargs: Additional tuner-specific parameters
        """
        self.model_trainer = model_trainer
        self.scoring_metric = scoring_metric
        self.cv_folds = cv_folds
        self.config = kwargs
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized {self.__class__.__name__} with {scoring_metric} metric")
    
    @abstractmethod
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: Dict[str, Any],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters for the model.
        
        Args:
            X: Training features
            y: Training targets
            search_space: Parameter search space definition
            sample_weight: Optional sample weights
            **kwargs: Additional optimization parameters
            
        Returns:
            Tuple of (best_parameters, best_score)
            
        Raises:
            ModelTrainingError: If optimization fails
        """
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame.
        
        Returns:
            DataFrame with optimization trial results
        """
        pass


class BaseModelEvaluator(ABC):
    """Abstract base class for model evaluation.
    
    Defines the interface for evaluating trained models with
    different metrics and validation strategies.
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
        required_methods = ['fit', 'predict']
        
        for method_name in required_methods:
            if not hasattr(model, method_name):
                raise ConfigurationError(
                    f"Model must implement {method_name} method"
                )
        
        # Check if predict_proba is available for probabilistic metrics
        probabilistic_metrics = ["roc_auc", "log_loss", "average_precision"]
        if any(metric in self.metrics for metric in probabilistic_metrics):
            if not hasattr(model, 'predict_proba'):
                logger.warning(
                    f"Model does not implement predict_proba but probabilistic "
                    f"metrics requested: {[m for m in self.metrics if m in probabilistic_metrics]}"
                )


class BaseFeatureSelector(ABC):
    """Abstract base class for feature selection.
    
    Defines the interface for different feature selection strategies
    such as univariate selection, recursive elimination, or model-based selection.
    """
    
    def __init__(
        self,
        selection_method: str,
        max_features: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize feature selector.
        
        Args:
            selection_method: Feature selection method name
            max_features: Maximum number of features to select
            **kwargs: Additional selector-specific parameters
        """
        self.selection_method = selection_method
        self.max_features = max_features
        self.config = kwargs
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[pd.DataFrame] = None
        
        logger.info(f"Initialized {self.__class__.__name__} with {selection_method} method")
    
    @abstractmethod
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features from input data.
        
        Args:
            X: Input features
            y: Target variable
            sample_weight: Optional sample weights
            **kwargs: Additional selection parameters
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from selection process.
        
        Returns:
            DataFrame with feature importance information
        """
        pass
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features.
        
        Args:
            X: Input data to transform
            
        Returns:
            Transformed data with selected features only
            
        Raises:
            ConfigurationError: If feature selection hasn't been performed
        """
        if self.selected_features_ is None:
            raise ConfigurationError(
                "Must call select_features before transform"
            )
        
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise ConfigurationError(
                f"Input data missing selected features: {missing_features}"
            )
        
        return X[self.selected_features_]


class BaseRobustnessTester(ABC):
    """Abstract base class for model robustness testing.
    
    Defines the interface for testing model stability and robustness
    under different conditions such as data perturbations or seed variations.
    """
    
    def __init__(
        self,
        test_type: str,
        n_iterations: int = 10,
        **kwargs: Any
    ) -> None:
        """Initialize robustness tester.
        
        Args:
            test_type: Type of robustness test
            n_iterations: Number of test iterations
            **kwargs: Additional tester-specific parameters
        """
        self.test_type = test_type
        self.n_iterations = n_iterations
        self.config = kwargs
        self.test_results_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized {self.__class__.__name__} with {test_type} test")
    
    @abstractmethod
    def test_robustness(
        self,
        model_trainer: BaseModelTrainer,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Test model robustness.
        
        Args:
            model_trainer: Model trainer to test
            X: Training features
            y: Training targets
            sample_weight: Optional sample weights
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with robustness test results
        """
        pass
    
    @abstractmethod
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics from robustness tests.
        
        Returns:
            Dictionary with stability metric values
        """
        pass


# Factory functions for creating instances
def create_model_trainer(
    trainer_type: str,
    model_type: str,
    **kwargs: Any
) -> BaseModelTrainer:
    """Factory function for creating model trainers.
    
    Args:
        trainer_type: Type of trainer to create
        model_type: Type of model to train
        **kwargs: Additional trainer parameters
        
    Returns:
        Initialized model trainer instance
        
    Raises:
        ConfigurationError: If trainer type is not recognized
    """
    from .trainer import StandardModelTrainer  # Avoid circular import
    
    trainer_map = {
        'standard': StandardModelTrainer,
        # Add more trainer types here as they're implemented
    }
    
    if trainer_type not in trainer_map:
        raise ConfigurationError(
            f"Unknown trainer type: {trainer_type}. Available: {list(trainer_map.keys())}"
        )
    
    trainer_class = trainer_map[trainer_type]
    return trainer_class(model_type=model_type, **kwargs)


def create_hyperparameter_tuner(
    tuner_type: str,
    model_trainer: BaseModelTrainer,
    **kwargs: Any
) -> BaseHyperparameterTuner:
    """Factory function for creating hyperparameter tuners.
    
    Args:
        tuner_type: Type of tuner to create
        model_trainer: Model trainer instance
        **kwargs: Additional tuner parameters
        
    Returns:
        Initialized hyperparameter tuner instance
        
    Raises:
        ConfigurationError: If tuner type is not recognized
    """
    from .tuner import OptunaHyperparameterTuner  # Avoid circular import
    
    tuner_map = {
        'optuna': OptunaHyperparameterTuner,
        # Add more tuner types here as they're implemented
    }
    
    if tuner_type not in tuner_map:
        raise ConfigurationError(
            f"Unknown tuner type: {tuner_type}. Available: {list(tuner_map.keys())}"
        )
    
    tuner_class = tuner_map[tuner_type]
    return tuner_class(model_trainer=model_trainer, **kwargs)


# Plugin registry for extensibility
class PluginRegistry:
    """Registry for extending the framework with custom implementations.
    
    Allows users to register custom trainers, tuners, evaluators, etc.
    that conform to the base class interfaces.
    """
    
    _trainers: Dict[str, type] = {}
    _tuners: Dict[str, type] = {}
    _evaluators: Dict[str, type] = {}
    _selectors: Dict[str, type] = {}
    _testers: Dict[str, type] = {}
    
    @classmethod
    def register_trainer(cls, name: str, trainer_class: type) -> None:
        """Register a custom model trainer.
        
        Args:
            name: Name for the trainer
            trainer_class: Trainer class (must inherit from BaseModelTrainer)
            
        Raises:
            ConfigurationError: If trainer doesn't inherit from base class
        """
        if not issubclass(trainer_class, BaseModelTrainer):
            raise ConfigurationError(
                f"Trainer class must inherit from BaseModelTrainer"
            )
        
        cls._trainers[name] = trainer_class
        logger.info(f"Registered custom trainer: {name}")
    
    @classmethod
    def register_tuner(cls, name: str, tuner_class: type) -> None:
        """Register a custom hyperparameter tuner.
        
        Args:
            name: Name for the tuner
            tuner_class: Tuner class (must inherit from BaseHyperparameterTuner)
        """
        if not issubclass(tuner_class, BaseHyperparameterTuner):
            raise ConfigurationError(
                f"Tuner class must inherit from BaseHyperparameterTuner"
            )
        
        cls._tuners[name] = tuner_class
        logger.info(f"Registered custom tuner: {name}")
    
    @classmethod
    def get_trainer_class(cls, name: str) -> type:
        """Get registered trainer class by name.
        
        Args:
            name: Trainer name
            
        Returns:
            Trainer class
            
        Raises:
            ConfigurationError: If trainer is not registered
        """
        if name not in cls._trainers:
            raise ConfigurationError(f"Trainer '{name}' not registered")
        
        return cls._trainers[name]
    
    @classmethod
    def get_available_trainers(cls) -> List[str]:
        """Get list of available trainer names.
        
        Returns:
            List of registered trainer names
        """
        return list(cls._trainers.keys())
    
    @classmethod
    def get_available_tuners(cls) -> List[str]:
        """Get list of available tuner names.
        
        Returns:
            List of registered tuner names
        """
        return list(cls._tuners.keys())