"""Base class for robustness testers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from ...utils.logger import get_logger
from .trainer import BaseModelTrainer

logger = get_logger(__name__)


class BaseRobustnessTester(ABC):
    """Abstract base class for model robustness testing."""
    
    def __init__(
        self,
        test_type: str,
        n_iterations: int = 10,
        **kwargs: Any
    ) -> None:
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
        """Test model robustness."""
        pass
    
    @abstractmethod
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics from robustness tests."""
        pass