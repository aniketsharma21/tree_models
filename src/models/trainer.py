"""Unified training interface for tree-based models with sample weights support.

This module provides a unified training interface for XGBoost, LightGBM, and CatBoost
with support for early stopping, evaluation sets, sample weights, and model persistence.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

from ..config.model_config import (
    XGB_DEFAULT_PARAMS, LGBM_DEFAULT_PARAMS, CATBOOST_DEFAULT_PARAMS, SEED
)
from ..utils.logger import get_logger
from ..utils.timer import timer
from ..utils.io_utils import save_model, ensure_dir

logger = get_logger(__name__)


class BaseTrainer:
    """Base class for all model trainers with sample weights support."""

    def __init__(self, model_type: str, params: Optional[Dict[str, Any]] = None):
        """Initialize base trainer.

        Args:
            model_type: Type of model ("xgboost", "lightgbm", "catboost")
            params: Model hyperparameters
        """
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.training_history = {}
        self.feature_importance_ = None
        self.best_iteration = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None,
            sample_weight_train: Optional[np.ndarray] = None,
            sample_weight_valid: Optional[np.ndarray] = None,
            **kwargs) -> 'BaseTrainer':
        """Fit the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X, **kwargs)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            raise ValueError("Model must be fitted to get feature importance")
        return self.feature_importance_

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        save_model(self.model, filepath)
        logger.info(f"Saved {self.model_type} model to {filepath}")


class XGBoostTrainer(BaseTrainer):
    """XGBoost model trainer with sample weights support.

    Example:
        >>> trainer = XGBoostTrainer(params={"max_depth": 6, "learning_rate": 0.1})
        >>> trainer.fit(X_train, y_train, X_valid, y_valid, 
        ...              sample_weight_train=weights_train)
        >>> predictions = trainer.predict_proba(X_test)[:, 1]
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost trainer.

        Args:
            params: XGBoost hyperparameters
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed")

        # Merge with default parameters
        default_params = XGB_DEFAULT_PARAMS.copy()
        if params:
            default_params.update(params)

        super().__init__("xgboost", default_params)

    @timer
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None,
            sample_weight_train: Optional[np.ndarray] = None,
            sample_weight_valid: Optional[np.ndarray] = None,
            early_stopping_rounds: int = 50,
            verbose: bool = False,
            **kwargs) -> 'XGBoostTrainer':
        """Train XGBoost model with sample weights support.

        Args:
            X_train: Training features
            y_train: Training target
            X_valid: Validation features
            y_valid: Validation target
            sample_weight_train: Training sample weights
            sample_weight_valid: Validation sample weights
            early_stopping_rounds: Early stopping rounds
            verbose: Whether to print training progress
            **kwargs: Additional parameters

        Returns:
            Fitted trainer
        """
        logger.info(f"Training XGBoost model with {len(X_train)} samples")
        if sample_weight_train is not None:
            logger.info("Using sample weights for training")

        # Prepare training data with sample weights
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight_train)
        eval_set = [(dtrain, 'train')]

        if X_valid is not None and y_valid is not None:
            dvalid = xgb.DMatrix(X_valid, label=y_valid, weight=sample_weight_valid)
            eval_set.append((dvalid, 'valid'))

        # Extract n_estimators from params
        n_estimators = self.params.pop('n_estimators', 200)

        # Train model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=eval_set,
            early_stopping_rounds=early_stopping_rounds if X_valid is not None else None,
            verbose_eval=verbose,
            **kwargs
        )

        # Store training history
        self.training_history = self.model.evals_result()
        self.best_iteration = self.model.best_iteration

        # Get feature importance
        self.feature_importance_ = self.model.get_score(importance_type='gain')

        # Convert to array format
        importance_array = np.zeros(len(X_train.columns))
        for i, col in enumerate(X_train.columns):
            if col in self.feature_importance_:
                importance_array[i] = self.feature_importance_[col]
        self.feature_importance_ = importance_array

        logger.info(f"XGBoost training completed. Best iteration: {self.best_iteration}")

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict with XGBoost model."""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest, **kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict probabilities with XGBoost model."""
        predictions = self.predict(X, **kwargs)

        # For binary classification, XGBoost returns probabilities directly
        # Convert to sklearn format (n_samples, n_classes)
        if len(predictions.shape) == 1:
            proba = np.column_stack([1 - predictions, predictions])
        else:
            proba = predictions

        return proba


class LightGBMTrainer(BaseTrainer):
    """LightGBM model trainer with sample weights support.

    Example:
        >>> trainer = LightGBMTrainer(params={"max_depth": 6, "learning_rate": 0.1})
        >>> trainer.fit(X_train, y_train, X_valid, y_valid,
        ...              sample_weight_train=weights_train)
        >>> predictions = trainer.predict_proba(X_test)[:, 1]
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize LightGBM trainer.

        Args:
            params: LightGBM hyperparameters
        """
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM is not installed")

        # Merge with default parameters
        default_params = LGBM_DEFAULT_PARAMS.copy()
        if params:
            default_params.update(params)

        super().__init__("lightgbm", default_params)

    @timer
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None,
            sample_weight_train: Optional[np.ndarray] = None,
            sample_weight_valid: Optional[np.ndarray] = None,
            early_stopping_rounds: int = 50,
            verbose: bool = False,
            categorical_features: Optional[List[str]] = None,
            **kwargs) -> 'LightGBMTrainer':
        """Train LightGBM model with sample weights support.

        Args:
            X_train: Training features
            y_train: Training target
            X_valid: Validation features
            y_valid: Validation target
            sample_weight_train: Training sample weights
            sample_weight_valid: Validation sample weights
            early_stopping_rounds: Early stopping rounds
            verbose: Whether to print training progress
            categorical_features: List of categorical feature names
            **kwargs: Additional parameters

        Returns:
            Fitted trainer
        """
        logger.info(f"Training LightGBM model with {len(X_train)} samples")
        if sample_weight_train is not None:
            logger.info("Using sample weights for training")

        # Prepare training data with sample weights
        train_data = lgb.Dataset(
            X_train, 
            label=y_train, 
            weight=sample_weight_train,
            categorical_feature=categorical_features
        )
        valid_sets = [train_data]
        valid_names = ['train']

        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(
                X_valid, 
                label=y_valid,
                weight=sample_weight_valid,
                categorical_feature=categorical_features,
                reference=train_data
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')

        # Extract n_estimators from params
        n_estimators = self.params.pop('n_estimators', 200)

        # Train model
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds) if X_valid is not None else None,
                lgb.log_evaluation(period=0 if not verbose else 100)
            ],
            **kwargs
        )

        # Store training history
        self.training_history = self.model.evals_result_
        self.best_iteration = self.model.best_iteration

        # Get feature importance
        self.feature_importance_ = self.model.feature_importance(importance_type='gain')

        logger.info(f"LightGBM training completed. Best iteration: {self.best_iteration}")

        return self

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict probabilities with LightGBM model."""
        predictions = self.model.predict(X, **kwargs)

        # For binary classification, LightGBM returns probabilities directly
        # Convert to sklearn format (n_samples, n_classes)
        if len(predictions.shape) == 1:
            proba = np.column_stack([1 - predictions, predictions])
        else:
            proba = predictions

        return proba


class CatBoostTrainer(BaseTrainer):
    """CatBoost model trainer with sample weights support.

    Example:
        >>> trainer = CatBoostTrainer(params={"depth": 6, "learning_rate": 0.1})
        >>> trainer.fit(X_train, y_train, X_valid, y_valid,
        ...              sample_weight_train=weights_train)
        >>> predictions = trainer.predict_proba(X_test)[:, 1]
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize CatBoost trainer.

        Args:
            params: CatBoost hyperparameters
        """
        if not CB_AVAILABLE:
            raise ImportError("CatBoost is not installed")

        # Merge with default parameters
        default_params = CATBOOST_DEFAULT_PARAMS.copy()
        if params:
            default_params.update(params)

        super().__init__("catboost", default_params)

    @timer
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None,
            sample_weight_train: Optional[np.ndarray] = None,
            sample_weight_valid: Optional[np.ndarray] = None,
            early_stopping_rounds: int = 50,
            verbose: bool = False,
            categorical_features: Optional[List[str]] = None,
            **kwargs) -> 'CatBoostTrainer':
        """Train CatBoost model with sample weights support.

        Args:
            X_train: Training features
            y_train: Training target
            X_valid: Validation features
            y_valid: Validation target
            sample_weight_train: Training sample weights
            sample_weight_valid: Validation sample weights
            early_stopping_rounds: Early stopping rounds
            verbose: Whether to print training progress
            categorical_features: List of categorical feature names
            **kwargs: Additional parameters

        Returns:
            Fitted trainer
        """
        logger.info(f"Training CatBoost model with {len(X_train)} samples")
        if sample_weight_train is not None:
            logger.info("Using sample weights for training")

        # Prepare categorical features
        cat_features = None
        if categorical_features:
            # Convert feature names to indices
            cat_features = [X_train.columns.get_loc(col) for col in categorical_features 
                          if col in X_train.columns]

        # Create CatBoost model
        self.model = cb.CatBoostClassifier(**self.params)

        # Prepare evaluation set
        eval_set = None
        sample_weight_eval = None
        if X_valid is not None and y_valid is not None:
            eval_set = (X_valid, y_valid)
            sample_weight_eval = sample_weight_valid

        # Train model with sample weights
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight_train,
            eval_set=eval_set,
            cat_features=cat_features,
            early_stopping_rounds=early_stopping_rounds if X_valid is not None else None,
            verbose=verbose,
            **kwargs
        )

        # Store training history
        if hasattr(self.model, 'evals_result_'):
            self.training_history = self.model.evals_result_

        self.best_iteration = self.model.best_iteration_

        # Get feature importance
        self.feature_importance_ = self.model.get_feature_importance()

        logger.info(f"CatBoost training completed. Best iteration: {self.best_iteration}")

        return self


class ModelTrainer:
    """Unified interface for training different tree-based models with sample weights support.

    Example:
        >>> trainer = ModelTrainer()
        >>> model = trainer.train("xgboost", X_train, y_train, X_valid, y_valid,
        ...                      sample_weight_train=weights_train,
        ...                      params={"max_depth": 6})
        >>> predictions = model.predict_proba(X_test)[:, 1]
    """

    def __init__(self):
        """Initialize model trainer."""
        self.available_models = {}

        if XGB_AVAILABLE:
            self.available_models["xgboost"] = XGBoostTrainer
        if LGB_AVAILABLE:
            self.available_models["lightgbm"] = LightGBMTrainer  
        if CB_AVAILABLE:
            self.available_models["catboost"] = CatBoostTrainer

        logger.info(f"Available models: {list(self.available_models.keys())}")

    def train(self, model_type: str, 
             X_train: pd.DataFrame, y_train: pd.Series,
             X_valid: Optional[pd.DataFrame] = None,
             y_valid: Optional[pd.Series] = None,
             sample_weight_train: Optional[np.ndarray] = None,
             sample_weight_valid: Optional[np.ndarray] = None,
             params: Optional[Dict[str, Any]] = None,
             **kwargs) -> BaseTrainer:
        """Train a model of specified type with sample weights support.

        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training target
            X_valid: Validation features
            y_valid: Validation target
            sample_weight_train: Training sample weights
            sample_weight_valid: Validation sample weights
            params: Model hyperparameters
            **kwargs: Additional training parameters

        Returns:
            Trained model
        """
        if model_type not in self.available_models:
            raise ValueError(f"Model type '{model_type}' not available. "
                           f"Available: {list(self.available_models.keys())}")

        trainer_class = self.available_models[model_type]
        trainer = trainer_class(params=params)

        return trainer.fit(
            X_train, y_train, X_valid, y_valid, 
            sample_weight_train=sample_weight_train,
            sample_weight_valid=sample_weight_valid,
            **kwargs
        )

    def get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for a model type.

        Args:
            model_type: Type of model

        Returns:
            Default parameters dictionary
        """
        if model_type == "xgboost":
            return XGB_DEFAULT_PARAMS.copy()
        elif model_type == "lightgbm":
            return LGBM_DEFAULT_PARAMS.copy()
        elif model_type == "catboost":
            return CATBOOST_DEFAULT_PARAMS.copy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def train_model(model_type: str,
               X_train: pd.DataFrame, y_train: pd.Series,
               X_valid: Optional[pd.DataFrame] = None,
               y_valid: Optional[pd.Series] = None,
               sample_weight_train: Optional[np.ndarray] = None,
               sample_weight_valid: Optional[np.ndarray] = None,
               params: Optional[Dict[str, Any]] = None,
               save_path: Optional[Union[str, Path]] = None,
               **kwargs) -> BaseTrainer:
    """Convenience function to train and optionally save a model with sample weights.

    Args:
        model_type: Type of model to train
        X_train: Training features  
        y_train: Training target
        X_valid: Validation features
        y_valid: Validation target
        sample_weight_train: Training sample weights
        sample_weight_valid: Validation sample weights
        params: Model hyperparameters
        save_path: Optional path to save the model
        **kwargs: Additional training parameters

    Returns:
        Trained model

    Example:
        >>> model = train_model("xgboost", X_train, y_train, X_valid, y_valid,
        ...                    sample_weight_train=weights_train,
        ...                    params={"max_depth": 6}, 
        ...                    save_path="models/xgb_baseline.pkl")
    """
    trainer = ModelTrainer()
    model = trainer.train(
        model_type, X_train, y_train, X_valid, y_valid, 
        sample_weight_train=sample_weight_train,
        sample_weight_valid=sample_weight_valid,
        params=params, 
        **kwargs
    )

    if save_path:
        model.save(save_path)

    return model
