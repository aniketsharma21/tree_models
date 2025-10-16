# tree_models/models/trainer.py
"""Enhanced model training interface with comprehensive features.

This module provides the core training interface for tree-based models with:
- Type-safe training orchestration and pipeline management
- Support for XGBoost, LightGBM, and CatBoost with unified interface
- Advanced training strategies (early stopping, learning curves, cross-validation)
- Sample weights integration throughout training workflows
- Comprehensive hyperparameter validation and optimization hooks
- Memory management and performance monitoring
- Extensive logging and progress tracking
- Model checkpointing and recovery capabilities
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass
import warnings
from datetime import datetime
import json

from .base import BaseTreeModel
from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    TrainingError,
    ConfigurationError,
    DataValidationError,
    PerformanceError,
    handle_and_reraise,
    validate_parameter,
    create_error_context
)
from ..config.model_config import ModelConfig, XGBoostConfig, LightGBMConfig, CatBoostConfig

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

# Optional model imports with fallbacks
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available")

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


@dataclass
class TrainingConfig:
    """Type-safe configuration for model training with comprehensive options."""
    
    # Basic training settings
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 100
    validation_fraction: float = 0.2
    
    # Cross-validation settings
    enable_cross_validation: bool = False
    cv_folds: int = 5
    cv_scoring: str = 'roc_auc'
    
    # Performance settings
    enable_gpu: bool = False
    n_jobs: int = -1
    memory_limit_gb: Optional[float] = None
    
    # Monitoring settings
    verbose: bool = True
    log_evaluation: int = 100
    track_learning_curves: bool = True
    
    # Checkpointing
    enable_checkpoints: bool = True
    checkpoint_frequency: int = 1000
    checkpoint_dir: Optional[str] = None
    
    # Advanced options
    enable_feature_importance: bool = True
    compute_shap_values: bool = False
    validate_input_data: bool = True
    
    def __post_init__(self) -> None:
        """Validate training configuration."""
        validate_parameter("validation_fraction", self.validation_fraction, min_value=0.0, max_value=0.5)
        validate_parameter("cv_folds", self.cv_folds, min_value=2, max_value=20)
        validate_parameter("early_stopping_rounds", self.early_stopping_rounds, min_value=1, max_value=1000)
        
        if self.memory_limit_gb is not None:
            validate_parameter("memory_limit_gb", self.memory_limit_gb, min_value=0.1, max_value=1000.0)


@dataclass
class TrainingResults:
    """Comprehensive training results with metrics and diagnostics."""
    
    # Model and training info
    model: Any
    model_type: str
    training_time: float
    
    # Performance metrics
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    cv_metrics: Optional[Dict[str, float]] = None
    
    # Training history
    learning_curves: Optional[Dict[str, List[float]]] = None
    feature_importance: Optional[pd.DataFrame] = None
    
    # Diagnostics
    best_iteration: Optional[int] = None
    early_stopped: bool = False
    convergence_info: Optional[Dict[str, Any]] = None
    
    # Metadata
    training_config: Optional[TrainingConfig] = None
    model_config: Optional[ModelConfig] = None
    timestamp: Optional[str] = None


class ModelTrainer:
    """Enhanced model trainer with comprehensive training orchestration.
    
    Provides unified interface for training tree-based models with advanced
    features like early stopping, cross-validation, and performance monitoring.
    
    Example:
        >>> trainer = ModelTrainer()
        >>> 
        >>> # Configure training
        >>> config = TrainingConfig(
        ...     enable_early_stopping=True,
        ...     early_stopping_rounds=50,
        ...     track_learning_curves=True
        ... )
        >>> 
        >>> # Train model
        >>> results = trainer.train_model(
        ...     model_config, X_train, y_train,
        ...     X_valid=X_valid, y_valid=y_valid,
        ...     sample_weight=weights,
        ...     training_config=config
        ... )
        >>> 
        >>> print(f"Training completed in {results.training_time:.2f}s")
        >>> print(f"Validation AUC: {results.validation_metrics['auc']:.4f}")
    """
    
    def __init__(
        self,
        random_state: int = 42,
        enable_logging: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize enhanced model trainer.
        
        Args:
            random_state: Random state for reproducibility
            enable_logging: Whether to enable detailed logging
            checkpoint_dir: Directory for saving training checkpoints
        """
        self.random_state = random_state
        self.enable_logging = enable_logging
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Training state
        self._current_model = None
        self._training_history = []
        self._best_score = None
        self._best_iteration = None
        
        # Performance tracking
        self._training_times = {}
        self._memory_usage = {}
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized ModelTrainer:")
        logger.info(f"  Random state: {random_state}")
        logger.info(f"  Checkpoint dir: {checkpoint_dir}")

    @timer(name="model_training")
    def train_model(
        self,
        model_config: ModelConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        validation_weight: Optional[np.ndarray] = None,
        training_config: Optional[TrainingConfig] = None,
        **kwargs: Any
    ) -> TrainingResults:
        """Train a tree-based model with comprehensive configuration.
        
        Args:
            model_config: Model configuration object
            X_train: Training features
            y_train: Training target
            X_valid: Optional validation features
            y_valid: Optional validation target
            sample_weight: Optional training sample weights
            validation_weight: Optional validation sample weights
            training_config: Training configuration
            **kwargs: Additional training parameters
            
        Returns:
            Comprehensive training results object
            
        Raises:
            TrainingError: If model training fails
        """
        logger.info(f"🚀 Starting model training:")
        logger.info(f"   Model: {model_config.model_type}")
        logger.info(f"   Training data: {X_train.shape}")
        logger.info(f"   Validation data: {X_valid.shape if X_valid is not None else 'None'}")
        
        if training_config is None:
            training_config = TrainingConfig()
        
        start_time = datetime.now()
        
        try:
            with timed_operation("model_training") as timing:
                # Validate inputs
                if training_config.validate_input_data:
                    self._validate_training_inputs(
                        X_train, y_train, X_valid, y_valid, 
                        sample_weight, validation_weight
                    )
                
                # Prepare validation data if needed
                if X_valid is None and training_config.enable_early_stopping:
                    X_train, X_valid, y_train, y_valid, sample_weight, validation_weight = \
                        self._create_validation_split(
                            X_train, y_train, sample_weight, training_config.validation_fraction
                        )
                
                # Initialize model
                model = self._create_model(model_config, training_config)
                
                # Configure training parameters
                train_params = self._prepare_training_parameters(
                    model_config, training_config, X_valid is not None
                )
                
                # Train model based on type
                if model_config.model_type == 'xgboost':
                    results = self._train_xgboost(
                        model, model_config, X_train, y_train, X_valid, y_valid,
                        sample_weight, validation_weight, train_params, training_config
                    )
                elif model_config.model_type == 'lightgbm':
                    results = self._train_lightgbm(
                        model, model_config, X_train, y_train, X_valid, y_valid,
                        sample_weight, validation_weight, train_params, training_config
                    )
                elif model_config.model_type == 'catboost':
                    results = self._train_catboost(
                        model, model_config, X_train, y_train, X_valid, y_valid,
                        sample_weight, validation_weight, train_params, training_config
                    )
                else:
                    raise ConfigurationError(f"Unsupported model type: {model_config.model_type}")
                
                # Post-training analysis
                if training_config.enable_feature_importance:
                    feature_importance = self._compute_feature_importance(results.model, X_train.columns)
                    results.feature_importance = feature_importance
                
                # Cross-validation if enabled
                if training_config.enable_cross_validation:
                    cv_metrics = self._perform_cross_validation(
                        model_config, X_train, y_train, sample_weight, training_config
                    )
                    results.cv_metrics = cv_metrics
                
                # Add metadata
                results.training_config = training_config
                results.model_config = model_config
                results.timestamp = start_time.isoformat()
                results.training_time = timing['duration']
            
            logger.info(f"✅ Model training completed:")
            logger.info(f"   Duration: {results.training_time:.2f}s")
            logger.info(f"   Best iteration: {results.best_iteration}")
            if results.validation_metrics:
                logger.info(f"   Validation metrics: {results.validation_metrics}")
            
            return results
            
        except Exception as e:
            handle_and_reraise(
                e, TrainingError,
                f"Model training failed for {model_config.model_type}",
                error_code="MODEL_TRAINING_FAILED",
                context=create_error_context(
                    model_type=model_config.model_type,
                    train_shape=X_train.shape,
                    has_validation=X_valid is not None,
                    has_weights=sample_weight is not None
                )
            )

    def _validate_training_inputs(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame],
        y_valid: Optional[pd.Series],
        sample_weight: Optional[np.ndarray],
        validation_weight: Optional[np.ndarray]
    ) -> None:
        """Comprehensive input validation for training data."""
        
        # Basic data validation
        if X_train.empty or y_train.empty:
            raise DataValidationError("Training data cannot be empty")
        
        if len(X_train) != len(y_train):
            raise DataValidationError("X_train and y_train must have same length")
        
        # Check for missing values in target
        if y_train.isnull().any():
            raise DataValidationError("Target variable cannot contain missing values")
        
        # Validation data consistency
        if X_valid is not None:
            if y_valid is None:
                raise DataValidationError("y_valid required when X_valid is provided")
            
            if len(X_valid) != len(y_valid):
                raise DataValidationError("X_valid and y_valid must have same length")
            
            if list(X_train.columns) != list(X_valid.columns):
                raise DataValidationError("Training and validation features must match")
        
        # Sample weights validation
        if sample_weight is not None:
            if len(sample_weight) != len(X_train):
                raise DataValidationError("sample_weight length must match training data")
            
            if np.any(sample_weight < 0):
                raise DataValidationError("Sample weights cannot be negative")
        
        if validation_weight is not None:
            if X_valid is None:
                raise DataValidationError("validation_weight requires validation data")
            
            if len(validation_weight) != len(X_valid):
                raise DataValidationError("validation_weight length must match validation data")
        
        # Check for feature data types
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise DataValidationError("No numeric features found in training data")

    def _create_validation_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        validation_fraction: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, 
               Optional[np.ndarray], Optional[np.ndarray]]:
        """Create train/validation split from training data."""
        
        from sklearn.model_selection import train_test_split
        
        split_params = {
            'test_size': validation_fraction,
            'random_state': self.random_state,
            'stratify': y if len(y.unique()) <= 20 else None  # Stratify for classification
        }
        
        if sample_weight is not None:
            X_train, X_valid, y_train, y_valid, weight_train, weight_valid = train_test_split(
                X, y, sample_weight, **split_params
            )
            return X_train, X_valid, y_train, y_valid, weight_train, weight_valid
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, **split_params)
            return X_train, X_valid, y_train, y_valid, None, None

    def _create_model(self, model_config: ModelConfig, training_config: TrainingConfig) -> Any:
        """Create model instance based on configuration."""
        
        if model_config.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ConfigurationError("XGBoost is not installed")
            return None  # XGBoost uses different training pattern
        
        elif model_config.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ConfigurationError("LightGBM is not installed")
            return None  # LightGBM uses different training pattern
        
        elif model_config.model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ConfigurationError("CatBoost is not installed")
            
            # CatBoost can be initialized with parameters
            params = model_config.get_params()
            params.update({
                'random_seed': self.random_state,
                'verbose': training_config.log_evaluation if training_config.verbose else False
            })
            
            # Determine task type
            if hasattr(model_config, 'objective') and 'reg' in str(model_config.objective):
                return cb.CatBoostRegressor(**params)
            else:
                return cb.CatBoostClassifier(**params)
        
        else:
            raise ConfigurationError(f"Unknown model type: {model_config.model_type}")

    def _prepare_training_parameters(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        has_validation: bool
    ) -> Dict[str, Any]:
        """Prepare training parameters for the specific model type."""
        
        base_params = model_config.get_params()
        
        # Add training-specific parameters
        train_params = {
            'random_state': self.random_state,
            'n_jobs': training_config.n_jobs if training_config.n_jobs != -1 else None
        }
        
        # Early stopping configuration
        if training_config.enable_early_stopping and has_validation:
            if model_config.model_type == 'xgboost':
                train_params.update({
                    'early_stopping_rounds': training_config.early_stopping_rounds,
                    'verbose_eval': training_config.log_evaluation if training_config.verbose else False
                })
            elif model_config.model_type == 'lightgbm':
                train_params.update({
                    'early_stopping_rounds': training_config.early_stopping_rounds,
                    'verbose_eval': training_config.log_evaluation if training_config.verbose else False
                })
            elif model_config.model_type == 'catboost':
                train_params.update({
                    'early_stopping_rounds': training_config.early_stopping_rounds,
                    'verbose': training_config.log_evaluation if training_config.verbose else False
                })
        
        # GPU configuration
        if training_config.enable_gpu:
            if model_config.model_type == 'xgboost':
                train_params['tree_method'] = 'gpu_hist'
            elif model_config.model_type == 'lightgbm':
                train_params['device_type'] = 'gpu'
            elif model_config.model_type == 'catboost':
                train_params['task_type'] = 'GPU'
        
        # Memory limit
        if training_config.memory_limit_gb:
            memory_mb = int(training_config.memory_limit_gb * 1024)
            if model_config.model_type == 'catboost':
                train_params['used_ram_limit'] = f'{memory_mb}MB'
        
        return {**base_params, **train_params}

    def _train_xgboost(
        self,
        model: Any,
        model_config: XGBoostConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame],
        y_valid: Optional[pd.Series],
        sample_weight: Optional[np.ndarray],
        validation_weight: Optional[np.ndarray],
        train_params: Dict[str, Any],
        training_config: TrainingConfig
    ) -> TrainingResults:
        """Train XGBoost model with comprehensive monitoring."""
        
        logger.info("Training XGBoost model")
        
        # Prepare datasets
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        
        eval_list = [(dtrain, 'train')]
        if X_valid is not None:
            dvalid = xgb.DMatrix(X_valid, label=y_valid, weight=validation_weight)
            eval_list.append((dvalid, 'valid'))
        
        # Training parameters
        params = train_params.copy()
        num_round = params.pop('n_estimators', 100)
        
        # Callbacks for monitoring
        callbacks = []
        learning_curves = {'train': [], 'valid': []} if training_config.track_learning_curves else None
        
        if training_config.track_learning_curves:
            def learning_curve_callback(env):
                if learning_curves is not None:
                    for eval_name, eval_result in env.evaluation_result_list:
                        if eval_name in learning_curves:
                            learning_curves[eval_name].append(eval_result)
            callbacks.append(learning_curve_callback)
        
        # Train model
        evals_result = {}
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_round,
            evals=eval_list,
            evals_result=evals_result,
            callbacks=callbacks,
            verbose_eval=training_config.log_evaluation if training_config.verbose else False
        )
        
        # Compute metrics
        train_pred = model.predict(dtrain)
        train_metrics = self._compute_metrics(y_train, train_pred, sample_weight)
        
        validation_metrics = {}
        if X_valid is not None:
            valid_pred = model.predict(dvalid)
            validation_metrics = self._compute_metrics(y_valid, valid_pred, validation_weight)
        
        # Determine best iteration
        best_iteration = getattr(model, 'best_iteration', None)
        if best_iteration is None and evals_result:
            # Find best iteration from evaluation results
            if 'valid' in evals_result:
                eval_metric = list(evals_result['valid'].keys())[0]
                scores = evals_result['valid'][eval_metric]
                if 'error' in eval_metric or 'loss' in eval_metric:
                    best_iteration = np.argmin(scores)
                else:
                    best_iteration = np.argmax(scores)
            else:
                best_iteration = len(evals_result['train'][list(evals_result['train'].keys())[0]]) - 1
        
        return TrainingResults(
            model=model,
            model_type='xgboost',
            training_time=0.0,  # Will be set by caller
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            learning_curves=learning_curves,
            best_iteration=best_iteration,
            early_stopped=best_iteration is not None and best_iteration < num_round - 1,
            convergence_info={'evals_result': evals_result}
        )

    def _train_lightgbm(
        self,
        model: Any,
        model_config: LightGBMConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame],
        y_valid: Optional[pd.Series],
        sample_weight: Optional[np.ndarray],
        validation_weight: Optional[np.ndarray],
        train_params: Dict[str, Any],
        training_config: TrainingConfig
    ) -> TrainingResults:
        """Train LightGBM model with comprehensive monitoring."""
        
        logger.info("Training LightGBM model")
        
        # Prepare datasets
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid, weight=validation_weight, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Training parameters
        params = train_params.copy()
        num_iterations = params.pop('n_estimators', 100)
        
        # Callbacks
        callbacks = []
        learning_curves = {'train': [], 'valid': []} if training_config.track_learning_curves else None
        
        if training_config.track_learning_curves:
            def learning_curve_callback(env):
                if learning_curves is not None:
                    for eval_name, eval_results in env.evaluation_result_list:
                        if eval_name in learning_curves:
                            # Get the first metric value
                            metric_value = list(eval_results.values())[0][-1]
                            learning_curves[eval_name].append(metric_value)
            callbacks.append(learning_curve_callback)
        
        # Train model
        evals_result = {}
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=num_iterations,
            valid_sets=valid_sets,
            valid_names=valid_names,
            evals_result=evals_result,
            callbacks=callbacks
        )
        
        # Compute metrics
        train_pred = model.predict(X_train)
        train_metrics = self._compute_metrics(y_train, train_pred, sample_weight)
        
        validation_metrics = {}
        if X_valid is not None:
            valid_pred = model.predict(X_valid)
            validation_metrics = self._compute_metrics(y_valid, valid_pred, validation_weight)
        
        # Best iteration
        best_iteration = getattr(model, 'best_iteration', model.current_iteration())
        
        return TrainingResults(
            model=model,
            model_type='lightgbm',
            training_time=0.0,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            learning_curves=learning_curves,
            best_iteration=best_iteration,
            early_stopped=best_iteration < num_iterations,
            convergence_info={'evals_result': evals_result}
        )

    def _train_catboost(
        self,
        model: Any,
        model_config: CatBoostConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame],
        y_valid: Optional[pd.Series],
        sample_weight: Optional[np.ndarray],
        validation_weight: Optional[np.ndarray],
        train_params: Dict[str, Any],
        training_config: TrainingConfig
    ) -> TrainingResults:
        """Train CatBoost model with comprehensive monitoring."""
        
        logger.info("Training CatBoost model")
        
        # Prepare training parameters
        fit_params = {}
        
        if X_valid is not None:
            fit_params['eval_set'] = (X_valid, y_valid)
            
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
            
        if validation_weight is not None and X_valid is not None:
            # CatBoost doesn't directly support validation weights in eval_set
            logger.warning("CatBoost doesn't support validation sample weights directly")
        
        if training_config.enable_early_stopping and X_valid is not None:
            fit_params['early_stopping_rounds'] = training_config.early_stopping_rounds
        
        if training_config.verbose:
            fit_params['verbose_eval'] = training_config.log_evaluation
        
        # Train model
        model.fit(X_train, y_train, **fit_params)
        
        # Compute metrics
        train_pred = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_train)
        train_metrics = self._compute_metrics(y_train, train_pred, sample_weight)
        
        validation_metrics = {}
        if X_valid is not None:
            valid_pred = model.predict_proba(X_valid)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_valid)
            validation_metrics = self._compute_metrics(y_valid, valid_pred, validation_weight)
        
        # Get training info
        best_iteration = getattr(model, 'get_best_iteration', lambda: None)()
        tree_count = model.tree_count_
        
        return TrainingResults(
            model=model,
            model_type='catboost',
            training_time=0.0,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            best_iteration=best_iteration,
            early_stopped=best_iteration is not None and best_iteration < tree_count,
            convergence_info={'tree_count': tree_count}
        )

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        
        metrics = {}
        
        try:
            # Handle binary classification vs regression
            unique_values = y_true.nunique()
            
            if unique_values == 2:
                # Binary classification metrics
                y_pred_binary = (y_pred > 0.5).astype(int)
                
                metrics['auc'] = roc_auc_score(y_true, y_pred, sample_weight=sample_weight)
                metrics['accuracy'] = accuracy_score(y_true, y_pred_binary, sample_weight=sample_weight)
                metrics['precision'] = precision_score(y_true, y_pred_binary, sample_weight=sample_weight, zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred_binary, sample_weight=sample_weight, zero_division=0)
                
                # Log loss
                y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
                if sample_weight is not None:
                    log_loss = -np.average(
                        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped),
                        weights=sample_weight
                    )
                else:
                    log_loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
                metrics['logloss'] = log_loss
                
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                
                metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
                
                # R-squared
                if sample_weight is not None:
                    ss_res = np.average((y_true - y_pred) ** 2, weights=sample_weight)
                    ss_tot = np.average((y_true - np.average(y_true, weights=sample_weight)) ** 2, weights=sample_weight)
                else:
                    ss_res = np.mean((y_true - y_pred) ** 2)
                    ss_tot = np.mean((y_true - np.mean(y_true)) ** 2)
                
                metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        except Exception as e:
            logger.warning(f"Error computing metrics: {e}")
        
        return metrics

    def _compute_feature_importance(self, model: Any, feature_names: pd.Index) -> pd.DataFrame:
        """Compute feature importance from trained model."""
        
        try:
            # Get importance based on model type
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
            elif hasattr(model, 'get_score'):
                # XGBoost importance
                importance_dict = model.get_score(importance_type='weight')
                importance = np.array([importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))])
            else:
                logger.warning("Could not extract feature importance from model")
                return pd.DataFrame()
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Add relative importance
            total_importance = importance_df['importance'].sum()
            if total_importance > 0:
                importance_df['relative_importance'] = importance_df['importance'] / total_importance
            else:
                importance_df['relative_importance'] = 0.0
            
            return importance_df
            
        except Exception as e:
            logger.warning(f"Error computing feature importance: {e}")
            return pd.DataFrame()

    def _perform_cross_validation(
        self,
        model_config: ModelConfig,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        training_config: TrainingConfig
    ) -> Dict[str, float]:
        """Perform cross-validation evaluation."""
        
        logger.info(f"Performing {training_config.cv_folds}-fold cross-validation")
        
        try:
            # Create a simplified model for CV
            if model_config.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                params = model_config.get_params()
                estimator = xgb.XGBClassifier(**params) if y.nunique() == 2 else xgb.XGBRegressor(**params)
            elif model_config.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                params = model_config.get_params()
                estimator = lgb.LGBMClassifier(**params) if y.nunique() == 2 else lgb.LGBMRegressor(**params)
            elif model_config.model_type == 'catboost' and CATBOOST_AVAILABLE:
                params = model_config.get_params()
                estimator = cb.CatBoostClassifier(**params, verbose=False) if y.nunique() == 2 else cb.CatBoostRegressor(**params, verbose=False)
            else:
                logger.warning(f"Cross-validation not supported for {model_config.model_type}")
                return {}
            
            # Perform cross-validation
            cv_splitter = StratifiedKFold(
                n_splits=training_config.cv_folds,
                shuffle=True,
                random_state=self.random_state
            ) if y.nunique() == 2 else None
            
            cv_scores = cross_val_score(
                estimator, X, y,
                cv=cv_splitter or training_config.cv_folds,
                scoring=training_config.cv_scoring,
                n_jobs=1,  # Avoid nested parallelism
                fit_params={'sample_weight': sample_weight} if sample_weight is not None else None
            )
            
            return {
                f'cv_{training_config.cv_scoring}_mean': cv_scores.mean(),
                f'cv_{training_config.cv_scoring}_std': cv_scores.std(),
                f'cv_{training_config.cv_scoring}_min': cv_scores.min(),
                f'cv_{training_config.cv_scoring}_max': cv_scores.max()
            }
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {}

    def save_model(
        self,
        model: Any,
        model_path: Union[str, Path],
        results: Optional[TrainingResults] = None
    ) -> None:
        """Save trained model with metadata."""
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model based on type
            model_type = getattr(results, 'model_type', 'unknown') if results else 'unknown'
            
            if model_type == 'xgboost':
                model.save_model(str(model_path.with_suffix('.json')))
            elif model_type == 'lightgbm':
                model.save_model(str(model_path.with_suffix('.txt')))
            elif model_type == 'catboost':
                model.save_model(str(model_path.with_suffix('.cbm')))
            else:
                # Fallback to pickle
                with open(model_path.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(model, f)
            
            # Save metadata if available
            if results:
                metadata = {
                    'model_type': results.model_type,
                    'training_time': results.training_time,
                    'train_metrics': results.train_metrics,
                    'validation_metrics': results.validation_metrics,
                    'best_iteration': results.best_iteration,
                    'timestamp': results.timestamp
                }
                
                metadata_path = model_path.with_suffix('.metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            handle_and_reraise(
                e, TrainingError,
                f"Failed to save model to {model_path}",
                error_code="MODEL_SAVE_FAILED"
            )

    def load_model(self, model_path: Union[str, Path]) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Load trained model with metadata."""
        
        model_path = Path(model_path)
        
        try:
            # Try to load metadata first
            metadata = None
            metadata_path = model_path.with_suffix('.metadata.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                model_type = metadata.get('model_type', 'unknown')
            else:
                # Infer from file extension
                if model_path.suffix == '.json':
                    model_type = 'xgboost'
                elif model_path.suffix == '.txt':
                    model_type = 'lightgbm'
                elif model_path.suffix == '.cbm':
                    model_type = 'catboost'
                else:
                    model_type = 'pickle'
            
            # Load model
            if model_type == 'xgboost':
                model = xgb.Booster()
                model.load_model(str(model_path))
            elif model_type == 'lightgbm':
                model = lgb.Booster(model_file=str(model_path))
            elif model_type == 'catboost':
                model = cb.CatBoost()
                model.load_model(str(model_path))
            else:
                # Pickle fallback
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            logger.info(f"Model loaded from {model_path}")
            return model, metadata
            
        except Exception as e:
            handle_and_reraise(
                e, TrainingError,
                f"Failed to load model from {model_path}",
                error_code="MODEL_LOAD_FAILED"
            )


# Convenience functions
def train_model(
    model_config: ModelConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: Optional[pd.DataFrame] = None,
    y_valid: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
    **kwargs: Any
) -> TrainingResults:
    """Convenience function for model training.
    
    Args:
        model_config: Model configuration
        X_train: Training features
        y_train: Training target
        X_valid: Optional validation features
        y_valid: Optional validation target
        sample_weight: Optional sample weights
        **kwargs: Additional training parameters
        
    Returns:
        Training results object
        
    Example:
        >>> from tree_models.config.model_config import XGBoostConfig
        >>> config = XGBoostConfig(n_estimators=100, max_depth=6)
        >>> results = train_model(config, X_train, y_train, X_valid, y_valid)
        >>> print(f"Validation AUC: {results.validation_metrics['auc']:.4f}")
    """
    trainer = ModelTrainer()
    return trainer.train_model(
        model_config, X_train, y_train, X_valid, y_valid, sample_weight, **kwargs
    )


# Export key classes and functions
__all__ = [
    'TrainingConfig',
    'TrainingResults', 
    'ModelTrainer',
    'train_model'
]