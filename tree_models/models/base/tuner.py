"""Base class for hyperparameter tuners."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ...utils.logger import get_logger
from .trainer import BaseModelTrainer

logger = get_logger(__name__)


class BaseHyperparameterTuner(ABC):
    """Abstract base class for hyperparameter tuning."""
    
    def __init__(
        self,
        model_trainer: BaseModelTrainer,
        scoring_metric: str = "roc_auc",
        cv_folds: int = 5,
        **kwargs: Any
    ) -> None:
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
        """Optimize hyperparameters for the model."""
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        pass