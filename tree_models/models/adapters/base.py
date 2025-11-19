from typing import Any, Optional
import pandas as pd


class ModelAdapter:
    """Base adapter interface for model training adapters.

    Adapter implementations expose a single `train` method that the
    `ModelTrainer` can call. Adapters accept the trainer instance as the
    first parameter so they can reuse utilities or logging from the trainer.
    """

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
        raise NotImplementedError()
