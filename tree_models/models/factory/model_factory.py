"""Factory functions for creating model instances.

This module provides factory functions for creating various types of
model components to avoid circular imports and centralize creation logic.
"""

from typing import Any, Dict
from ...utils.exceptions import ConfigurationError
from ...utils.logger import get_logger

logger = get_logger(__name__)


def create_model_trainer(
    trainer_type: str,
    model_type: str,
    **kwargs: Any
):
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
    trainer_registry = _get_trainer_registry()
    
    if trainer_type not in trainer_registry:
        raise ConfigurationError(
            f"Unknown trainer type: {trainer_type}. "
            f"Available: {list(trainer_registry.keys())}"
        )
    
    trainer_class = trainer_registry[trainer_type]
    logger.info(f"Creating {trainer_type} trainer for {model_type}")
    return trainer_class(model_type=model_type, **kwargs)


def create_hyperparameter_tuner(
    tuner_type: str,
    model_trainer,
    **kwargs: Any
):
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
    tuner_registry = _get_tuner_registry()
    
    if tuner_type not in tuner_registry:
        raise ConfigurationError(
            f"Unknown tuner type: {tuner_type}. "
            f"Available: {list(tuner_registry.keys())}"
        )
    
    tuner_class = tuner_registry[tuner_type]
    logger.info(f"Creating {tuner_type} tuner")
    return tuner_class(model_trainer=model_trainer, **kwargs)


def create_model_evaluator(
    evaluator_type: str = "standard",
    **kwargs: Any
):
    """Factory function for creating model evaluators.
    
    Args:
        evaluator_type: Type of evaluator to create
        **kwargs: Additional evaluator parameters
        
    Returns:
        Initialized model evaluator instance
    """
    evaluator_registry = _get_evaluator_registry()
    
    if evaluator_type not in evaluator_registry:
        raise ConfigurationError(
            f"Unknown evaluator type: {evaluator_type}. "
            f"Available: {list(evaluator_registry.keys())}"
        )
    
    evaluator_class = evaluator_registry[evaluator_type]
    logger.info(f"Creating {evaluator_type} evaluator")
    return evaluator_class(**kwargs)


def create_feature_selector(
    selector_type: str,
    **kwargs: Any
):
    """Factory function for creating feature selectors.
    
    Args:
        selector_type: Type of selector to create
        **kwargs: Additional selector parameters
        
    Returns:
        Initialized feature selector instance
    """
    selector_registry = _get_selector_registry()
    
    if selector_type not in selector_registry:
        raise ConfigurationError(
            f"Unknown selector type: {selector_type}. "
            f"Available: {list(selector_registry.keys())}"
        )
    
    selector_class = selector_registry[selector_type]
    logger.info(f"Creating {selector_type} selector")
    return selector_class(**kwargs)


def create_robustness_tester(
    tester_type: str,
    **kwargs: Any
):
    """Factory function for creating robustness testers.
    
    Args:
        tester_type: Type of tester to create
        **kwargs: Additional tester parameters
        
    Returns:
        Initialized robustness tester instance
    """
    tester_registry = _get_tester_registry()
    
    if tester_type not in tester_registry:
        raise ConfigurationError(
            f"Unknown tester type: {tester_type}. "
            f"Available: {list(tester_registry.keys())}"
        )
    
    tester_class = tester_registry[tester_type]
    logger.info(f"Creating {tester_type} tester")
    return tester_class(**kwargs)


def _get_trainer_registry() -> Dict[str, type]:
    """Get trainer class registry."""
    # Import here to avoid circular imports
    from ..trainer import StandardModelTrainer
    
    return {
        'standard': StandardModelTrainer,
        # Add more trainer types here
    }


def _get_tuner_registry() -> Dict[str, type]:
    """Get tuner class registry."""
    # Import here to avoid circular imports
    from ..tuner import OptunaHyperparameterTuner
    
    return {
        'optuna': OptunaHyperparameterTuner,
        # Add more tuner types here
    }


def _get_evaluator_registry() -> Dict[str, type]:
    """Get evaluator class registry."""
    # Import here to avoid circular imports
    from ..evaluator import StandardModelEvaluator
    
    return {
        'standard': StandardModelEvaluator,
        # Add more evaluator types here
    }


def _get_selector_registry() -> Dict[str, type]:
    """Get selector class registry."""
    # Import here to avoid circular imports
    from ..feature_selector import (
        RFECVFeatureSelector,
        BorutaFeatureSelector,
        ConsensusFeatureSelector
    )
    
    return {
        'rfecv': RFECVFeatureSelector,
        'boruta': BorutaFeatureSelector,
        'consensus': ConsensusFeatureSelector,
        # Add more selector types here
    }


def _get_tester_registry() -> Dict[str, type]:
    """Get tester class registry."""
    # Import here to avoid circular imports
    from ..robustness import (
        SeedRobustnessTester,
        SensitivityAnalyzer,
        PopulationStabilityIndex
    )
    
    return {
        'seed': SeedRobustnessTester,
        'sensitivity': SensitivityAnalyzer,
        'psi': PopulationStabilityIndex,
        # Add more tester types here
    }