"""Tree Models - Model Training and Evaluation Components.

This module provides core model training, evaluation, hyperparameter tuning,
feature selection, and robustness testing functionality.

Key Components:
- StandardModelTrainer: Unified training interface for tree-based models
- StandardModelEvaluator: Comprehensive model evaluation with advanced metrics
- OptunaHyperparameterTuner: Bayesian hyperparameter optimization
- Feature selection algorithms (RFECV, Boruta, Consensus)
- Robustness testing utilities

Example:
    >>> from tree_models.models import StandardModelTrainer, StandardModelEvaluator
    >>> trainer = StandardModelTrainer('xgboost')
    >>> evaluator = StandardModelEvaluator()
"""

# Core model components
from .trainer import StandardModelTrainer
from .evaluator import StandardModelEvaluator
from .tuner import OptunaHyperparameterTuner, ScoringConfig
from .feature_selector import (
    RFECVFeatureSelector,
    BorutaFeatureSelector, 
    ConsensusFeatureSelector
)
from .robustness import (
    SeedRobustnessTester,
    SensitivityAnalyzer,
    PopulationStabilityIndex
)

# Base classes for extensibility
from .base import (
    BaseModelTrainer,
    BaseModelEvaluator,
    BaseHyperparameterTuner,
    BaseFeatureSelector,
    BaseRobustnessTester,
    TrainingResult,
    EvaluationResult,
    ModelProtocol
)

__all__ = [
    # Core implementations
    'StandardModelTrainer',
    'StandardModelEvaluator', 
    'OptunaHyperparameterTuner',
    'ScoringConfig',
    
    # Feature selection
    'RFECVFeatureSelector',
    'BorutaFeatureSelector',
    'ConsensusFeatureSelector',
    
    # Robustness testing
    'SeedRobustnessTester',
    'SensitivityAnalyzer',
    'PopulationStabilityIndex',
    
    # Base classes and protocols
    'BaseModelTrainer',
    'BaseModelEvaluator',
    'BaseHyperparameterTuner',
    'BaseFeatureSelector', 
    'BaseRobustnessTester',
    'TrainingResult',
    'EvaluationResult',
    'ModelProtocol'
]