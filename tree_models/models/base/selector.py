"""Base class for feature selectors."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from ...utils.logger import get_logger

logger = get_logger(__name__)


class BaseFeatureSelector(ABC):
    """Abstract base class for feature selection."""
    
    def __init__(
        self,
        selection_method: str,
        max_features: Optional[int] = None,
        **kwargs: Any
    ) -> None:
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
        """Select features from input data."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from selection process."""
        pass