"""Base class for model trainers.

This module provides the abstract base class that defines the interface
for all model training components in the framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ...utils.exceptions import ConfigurationError, ModelTrainingError
from ...utils.logger import get_logger
from .protocols import ModelProtocol

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Standardized training result container.
    
    Provides a consistent interface for training results across
    all model types and training strategies.
    
    Attributes:
        model: The trained model object
        best_params: Best parameters found during training
        best_score: Best score achieved during training
        training_time: Time taken for training (seconds)
        metadata: Additional training information
        validation_scores: Optional cross-validation scores
        feature_importance: Optional feature importance data
    """
    model: Any
    best_params: Dict[str, Any]
    best_score: float
    training_time: float
    metadata: Dict[str, Any]
    validation_scores: Optional[Dict[str, float]] = None
    feature_importance: Optional[pd.DataFrame] = None
    
    def __post_init__(self) -> None:
        """Validate training result after initialization."""
        if self.training_time < 0:
            raise ValueError("Training time cannot be negative")
        
        if not isinstance(self.metadata, dict):
            raise TypeError("Metadata must be a dictionary")


class BaseModelTrainer(ABC):
    """Abstract base class for model trainers.
    
    Defines the interface for training models with different strategies
    such as basic training, cross-validation, or hyperparameter tuning.
    
    This class provides common functionality and enforces a consistent
    interface across all trainer implementations.
    """
    
    def __init__(
        self,
        model_type: str,
        random_state: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize model trainer.
        
        Args:
            model_type: Type of model to train (e.g., 'xgboost', 'lightgbm')
            random_state: Random state for reproducibility
            **kwargs: Additional trainer-specific parameters
        """
        self.model_type = model_type
        self.random_state = random_state
        self.config = kwargs
        self.is_fitted = False
        self._last_training_result: Optional[TrainingResult] = None
        
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
        self._validate_dataframes(X, y)
        self._validate_sample_weights(sample_weight, len(X))
        self._validate_missing_values(X, y)
        self._validate_data_types(X, y)
    
    def _validate_dataframes(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate DataFrame and Series inputs.
        
        Args:
            X: Training features
            y: Training targets
            
        Raises:
            ConfigurationError: If validation fails
        """
        if X.empty:
            raise ConfigurationError("Training data X cannot be empty")
        
        if y.empty:
            raise ConfigurationError("Training targets y cannot be empty")
        
        if len(X) != len(y):
            raise ConfigurationError(
                f"X and y must have same length: {len(X)} vs {len(y)}"
            )
        
        if len(X) < 2:
            raise ConfigurationError(
                f"Need at least 2 samples for training, got {len(X)}"
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
        
        if np.sum(sample_weight) == 0:
            raise ConfigurationError("sample_weight cannot sum to zero")
        
        if np.any(np.isnan(sample_weight)):
            raise ConfigurationError("sample_weight cannot contain NaN values")
    
    def _validate_missing_values(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Check for missing values in training data.
        
        Args:
            X: Training features
            y: Training targets
            
        Raises:
            ConfigurationError: If missing values found
        """
        # Check for missing values in features
        if X.isnull().any().any():
            null_cols = X.columns[X.isnull().any()].tolist()
            null_counts = X[null_cols].isnull().sum().to_dict()
            raise ConfigurationError(
                f"X contains null values in columns: {null_counts}"
            )
        
        # Check for missing values in targets
        if y.isnull().any():
            null_count = y.isnull().sum()
            raise ConfigurationError(
                f"y contains {null_count} null values"
            )
    
    def _validate_data_types(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate data types are appropriate for training.
        
        Args:
            X: Training features
            y: Training targets
            
        Raises:
            ConfigurationError: If data types are invalid
        """
        # Check for object columns that might need encoding
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.warning(
                f"Found object columns that may need encoding: {object_cols}"
            )
        
        # Check target data type
        if y.dtype == 'object':
            unique_values = y.unique()
            logger.warning(
                f"Target has object dtype with values: {unique_values[:10]}..."
            )
    
    def get_last_training_result(self) -> Optional[TrainingResult]:
        """Get the result from the last training run.
        
        Returns:
            Last training result or None if no training has been performed
        """
        return self._last_training_result
    
    def _store_training_result(self, result: TrainingResult) -> None:
        """Store training result for later retrieval.
        
        Args:
            result: Training result to store
        """
        self._last_training_result = result
        self.is_fitted = True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trainer configuration.
        
        Returns:
            Dictionary with trainer information
        """
        return {
            'trainer_class': self.__class__.__name__,
            'model_type': self.model_type,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'config': self.config
        }
    
    def __repr__(self) -> str:
        """String representation of the trainer."""
        return (
            f"{self.__class__.__name__}("
            f"model_type='{self.model_type}', "
            f"random_state={self.random_state}, "
            f"is_fitted={self.is_fitted})"
        )