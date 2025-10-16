"""Thread-safe singleton plugin registry for extensibility.

This module provides a registry system that allows users to register
custom implementations of framework components.
"""

from typing import Dict, List, Type
from threading import Lock
from ...utils.exceptions import ConfigurationError
from ...utils.logger import get_logger

logger = get_logger(__name__)


class PluginRegistry:
    """Thread-safe singleton registry for custom implementations.
    
    This registry allows users to extend the framework by registering
    custom trainers, tuners, evaluators, selectors, and testers.
    
    The singleton pattern ensures there's only one registry instance
    across the entire application.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls) -> 'PluginRegistry':
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize registry if not already done."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._trainers: Dict[str, type] = {}
        self._tuners: Dict[str, type] = {}
        self._evaluators: Dict[str, type] = {}
        self._selectors: Dict[str, type] = {}
        self._testers: Dict[str, type] = {}
        self._initialized = True
        
        logger.info("Initialized PluginRegistry")
    
    def register_trainer(self, name: str, trainer_class: type) -> None:
        """Register a custom model trainer.
        
        Args:
            name: Name for the trainer
            trainer_class: Trainer class (must inherit from BaseModelTrainer)
            
        Raises:
            ConfigurationError: If trainer doesn't inherit from base class
            
        Example:
            >>> registry.register_trainer('my_trainer', MyCustomTrainer)
        """
        self._validate_trainer_class(trainer_class)
        
        with self._lock:
            if name in self._trainers:
                logger.warning(f"Overriding existing trainer: {name}")
            self._trainers[name] = trainer_class
            
        logger.info(f"Registered custom trainer: {name}")
    
    def register_tuner(self, name: str, tuner_class: type) -> None:
        """Register a custom hyperparameter tuner.
        
        Args:
            name: Name for the tuner
            tuner_class: Tuner class (must inherit from BaseHyperparameterTuner)
            
        Example:
            >>> registry.register_tuner('my_tuner', MyCustomTuner)
        """
        self._validate_tuner_class(tuner_class)
        
        with self._lock:
            if name in self._tuners:
                logger.warning(f"Overriding existing tuner: {name}")
            self._tuners[name] = tuner_class
            
        logger.info(f"Registered custom tuner: {name}")
    
    def register_evaluator(self, name: str, evaluator_class: type) -> None:
        """Register a custom model evaluator.
        
        Args:
            name: Name for the evaluator
            evaluator_class: Evaluator class (must inherit from BaseModelEvaluator)
        """
        self._validate_evaluator_class(evaluator_class)
        
        with self._lock:
            if name in self._evaluators:
                logger.warning(f"Overriding existing evaluator: {name}")
            self._evaluators[name] = evaluator_class
            
        logger.info(f"Registered custom evaluator: {name}")
    
    def register_selector(self, name: str, selector_class: type) -> None:
        """Register a custom feature selector.
        
        Args:
            name: Name for the selector
            selector_class: Selector class (must inherit from BaseFeatureSelector)
        """
        self._validate_selector_class(selector_class)
        
        with self._lock:
            if name in self._selectors:
                logger.warning(f"Overriding existing selector: {name}")
            self._selectors[name] = selector_class
            
        logger.info(f"Registered custom selector: {name}")
    
    def register_tester(self, name: str, tester_class: type) -> None:
        """Register a custom robustness tester.
        
        Args:
            name: Name for the tester
            tester_class: Tester class (must inherit from BaseRobustnessTester)
        """
        self._validate_tester_class(tester_class)
        
        with self._lock:
            if name in self._testers:
                logger.warning(f"Overriding existing tester: {name}")
            self._testers[name] = tester_class
            
        logger.info(f"Registered custom tester: {name}")
    
    def get_trainer_class(self, name: str) -> type:
        """Get registered trainer class by name.
        
        Args:
            name: Trainer name
            
        Returns:
            Trainer class
            
        Raises:
            ConfigurationError: If trainer is not registered
        """
        if name not in self._trainers:
            raise ConfigurationError(
                f"Trainer '{name}' not registered. Available: {self.get_available_trainers()}"
            )
        
        return self._trainers[name]
    
    def get_tuner_class(self, name: str) -> type:
        """Get registered tuner class by name."""
        if name not in self._tuners:
            raise ConfigurationError(
                f"Tuner '{name}' not registered. Available: {self.get_available_tuners()}"
            )
        
        return self._tuners[name]
    
    def get_evaluator_class(self, name: str) -> type:
        """Get registered evaluator class by name."""
        if name not in self._evaluators:
            raise ConfigurationError(
                f"Evaluator '{name}' not registered. Available: {self.get_available_evaluators()}"
            )
        
        return self._evaluators[name]
    
    def get_selector_class(self, name: str) -> type:
        """Get registered selector class by name."""
        if name not in self._selectors:
            raise ConfigurationError(
                f"Selector '{name}' not registered. Available: {self.get_available_selectors()}"
            )
        
        return self._selectors[name]
    
    def get_tester_class(self, name: str) -> type:
        """Get registered tester class by name."""
        if name not in self._testers:
            raise ConfigurationError(
                f"Tester '{name}' not registered. Available: {self.get_available_testers()}"
            )
        
        return self._testers[name]
    
    def get_available_trainers(self) -> List[str]:
        """Get list of available trainer names."""
        return list(self._trainers.keys())
    
    def get_available_tuners(self) -> List[str]:
        """Get list of available tuner names."""
        return list(self._tuners.keys())
    
    def get_available_evaluators(self) -> List[str]:
        """Get list of available evaluator names."""
        return list(self._evaluators.keys())
    
    def get_available_selectors(self) -> List[str]:
        """Get list of available selector names."""
        return list(self._selectors.keys())
    
    def get_available_testers(self) -> List[str]:
        """Get list of available tester names."""
        return list(self._testers.keys())
    
    def clear_registry(self) -> None:
        """Clear all registered components (mainly for testing)."""
        with self._lock:
            self._trainers.clear()
            self._tuners.clear()
            self._evaluators.clear()
            self._selectors.clear()
            self._testers.clear()
        
        logger.info("Cleared plugin registry")
    
    def _validate_trainer_class(self, trainer_class: type) -> None:
        """Validate trainer class inheritance."""
        # Import here to avoid circular imports
        from ..base.trainer import BaseModelTrainer
        
        if not issubclass(trainer_class, BaseModelTrainer):
            raise ConfigurationError(
                f"Trainer class must inherit from BaseModelTrainer"
            )
    
    def _validate_tuner_class(self, tuner_class: type) -> None:
        """Validate tuner class inheritance."""
        # Import here to avoid circular imports
        from ..base.tuner import BaseHyperparameterTuner
        
        if not issubclass(tuner_class, BaseHyperparameterTuner):
            raise ConfigurationError(
                f"Tuner class must inherit from BaseHyperparameterTuner"
            )
    
    def _validate_evaluator_class(self, evaluator_class: type) -> None:
        """Validate evaluator class inheritance."""
        # Import here to avoid circular imports
        from ..base.evaluator import BaseModelEvaluator
        
        if not issubclass(evaluator_class, BaseModelEvaluator):
            raise ConfigurationError(
                f"Evaluator class must inherit from BaseModelEvaluator"
            )
    
    def _validate_selector_class(self, selector_class: type) -> None:
        """Validate selector class inheritance."""
        # Import here to avoid circular imports
        from ..base.selector import BaseFeatureSelector
        
        if not issubclass(selector_class, BaseFeatureSelector):
            raise ConfigurationError(
                f"Selector class must inherit from BaseFeatureSelector"
            )
    
    def _validate_tester_class(self, tester_class: type) -> None:
        """Validate tester class inheritance."""
        # Import here to avoid circular imports
        from ..base.tester import BaseRobustnessTester
        
        if not issubclass(tester_class, BaseRobustnessTester):
            raise ConfigurationError(
                f"Tester class must inherit from BaseRobustnessTester"
            )
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"PluginRegistry("
            f"trainers={len(self._trainers)}, "
            f"tuners={len(self._tuners)}, "
            f"evaluators={len(self._evaluators)}, "
            f"selectors={len(self._selectors)}, "
            f"testers={len(self._testers)})"
        )


# Global registry instance
plugin_registry = PluginRegistry()