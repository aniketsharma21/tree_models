import logging
from typing import Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class CatBoostAdapter:
    """Adapter wrapper for CatBoost training logic."""

    def train(
        self,
        trainer: Any,
        model_config: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame],
        y_valid: Optional[pd.Series],
        sample_weight: Optional[Any],
        validation_weight: Optional[Any],
        train_params: dict,
        training_config: Any,
    ) -> Any:
        logger.debug("Using CatBoostAdapter to train model")
        return trainer._train_catboost(
            None,
            model_config,
            X_train,
            y_train,
            X_valid,
            y_valid,
            sample_weight,
            validation_weight,
            train_params,
            training_config,
        )
