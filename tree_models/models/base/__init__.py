"""Base classes and interfaces for tree_models framework.

This module provides the core abstractions and protocols that define
the interfaces for all model-related functionality in the package.

Components:
- BaseModelTrainer: Abstract base class for model training
- BaseModelEvaluator: Abstract base class for model evaluation  
- BaseHyperparameterTuner: Abstract base class for hyperparameter optimization
- BaseFeatureSelector: Abstract base class for feature selection
- BaseRobustnessTester: Abstract base class for robustness testing
- ModelProtocol: Protocol defining model interface requirements
- Result containers: Standardized result data structures

Example:
    >>> from tree_models.models.base import BaseModelTrainer, TrainingResult
    >>> from tree_models.models.base import ModelProtocol
"""

# Core base classes
from .trainer import BaseModelTrainer, TrainingResult
from .evaluator import BaseModelEvaluator, EvaluationResult
from .tuner import BaseHyperparameterTuner
from .selector import BaseFeatureSelector
from .tester import BaseRobustnessTester
from .protocols import ModelProtocol

__all__ = [
    # Base classes
    'BaseModelTrainer',
    'BaseModelEvaluator', 
    'BaseHyperparameterTuner',
    'BaseFeatureSelector',
    'BaseRobustnessTester',
    
    # Protocols
    'ModelProtocol',
    
    # Result containers
    'TrainingResult',
    'EvaluationResult'
]