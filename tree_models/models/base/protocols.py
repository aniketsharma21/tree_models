"""Protocol definitions for tree_models framework.

This module defines the protocols (interfaces) that models and other
components must implement to work with the framework.
"""

from typing import Any, Protocol, runtime_checkable
import pandas as pd
import numpy as np


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol defining the interface for ML models.
    
    This protocol ensures that all models used in the framework
    implement the required methods for training and prediction.
    
    The @runtime_checkable decorator allows isinstance() checks
    to validate protocol compliance at runtime.
    """
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> 'ModelProtocol':
        """Fit the model to training data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional fitting parameters (e.g., sample_weight)
            
        Returns:
            Self (fitted model)
        """
        ...
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions as numpy array
        """
        ...
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities (for classifiers).
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities as numpy array
            Shape: (n_samples, n_classes)
            
        Note:
            This method is optional for regression models
        """
        ...


@runtime_checkable
class ExplainerProtocol(Protocol):
    """Protocol for model explainers.
    
    Defines the interface that explainability components
    must implement.
    """
    
    def explain(
        self, 
        X: pd.DataFrame, 
        **kwargs: Any
    ) -> Any:
        """Generate explanations for predictions.
        
        Args:
            X: Input data to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            Explanation results (format depends on explainer type)
        """
        ...


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for data validators.
    
    Defines the interface for data validation components.
    """
    
    def validate(
        self, 
        data: pd.DataFrame, 
        **kwargs: Any
    ) -> Any:
        """Validate input data.
        
        Args:
            data: Data to validate
            **kwargs: Additional validation parameters
            
        Returns:
            Validation results
        """
        ...


@runtime_checkable  
class TransformerProtocol(Protocol):
    """Protocol for data transformers.
    
    Defines the interface for data transformation components.
    """
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs: Any) -> 'TransformerProtocol':
        """Fit the transformer to training data.
        
        Args:
            X: Training features
            y: Optional training targets
            **kwargs: Additional fitting parameters
            
        Returns:
            Self (fitted transformer)
        """
        ...
    
    def transform(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Transform input data.
        
        Args:
            X: Data to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            Transformed data
        """
        ...
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: pd.Series = None, 
        **kwargs: Any
    ) -> pd.DataFrame:
        """Fit transformer and transform data in one step.
        
        Args:
            X: Training features
            y: Optional training targets  
            **kwargs: Additional parameters
            
        Returns:
            Transformed data
        """
        return self.fit(X, y, **kwargs).transform(X, **kwargs)


@runtime_checkable
class ScorerProtocol(Protocol):
    """Protocol for scoring functions.
    
    Defines the interface for custom scoring functions.
    """
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray = None,
        **kwargs: Any
    ) -> float:
        """Score predictions against true values.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            sample_weight: Optional sample weights
            **kwargs: Additional scoring parameters
            
        Returns:
            Score value (higher is better by convention)
        """
        ...


# Utility functions for protocol validation
def validate_model_protocol(model: Any) -> None:
    """Validate that a model implements the ModelProtocol.
    
    Args:
        model: Model object to validate
        
    Raises:
        TypeError: If model doesn't implement the protocol
        
    Example:
        >>> validate_model_protocol(my_model)
    """
    if not isinstance(model, ModelProtocol):
        missing_methods = []
        
        if not hasattr(model, 'fit'):
            missing_methods.append('fit')
        if not hasattr(model, 'predict'):
            missing_methods.append('predict')
            
        raise TypeError(
            f"Model must implement ModelProtocol. "
            f"Missing methods: {missing_methods}"
        )


def validate_transformer_protocol(transformer: Any) -> None:
    """Validate that a transformer implements the TransformerProtocol.
    
    Args:
        transformer: Transformer object to validate
        
    Raises:
        TypeError: If transformer doesn't implement the protocol
    """
    if not isinstance(transformer, TransformerProtocol):
        missing_methods = []
        
        if not hasattr(transformer, 'fit'):
            missing_methods.append('fit')
        if not hasattr(transformer, 'transform'):
            missing_methods.append('transform')
            
        raise TypeError(
            f"Transformer must implement TransformerProtocol. "
            f"Missing methods: {missing_methods}"
        )


def check_model_capabilities(model: Any) -> dict:
    """Check what capabilities a model has.
    
    Args:
        model: Model to check
        
    Returns:
        Dictionary with capability flags
        
    Example:
        >>> caps = check_model_capabilities(model)
        >>> if caps['supports_probabilities']:
        ...     probas = model.predict_proba(X)
    """
    capabilities = {
        'supports_fitting': hasattr(model, 'fit'),
        'supports_prediction': hasattr(model, 'predict'),
        'supports_probabilities': hasattr(model, 'predict_proba'),
        'supports_feature_importance': (
            hasattr(model, 'feature_importances_') or
            hasattr(model, 'get_feature_importance') or
            hasattr(model, 'feature_importance')
        ),
        'supports_early_stopping': hasattr(model, 'best_iteration'),
        'supports_gpu': hasattr(model, 'tree_method') and 'gpu' in str(getattr(model, 'tree_method', ''))
    }
    
    return capabilities