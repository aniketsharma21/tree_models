# tree_models/models/tuner.py
"""Enhanced hyperparameter tuning with Optuna and comprehensive evaluation.

This module provides production-ready hyperparameter optimization with:
- Type-safe interfaces and comprehensive error handling
- Advanced Optuna integration with multiple sampling strategies
- Custom scoring functions with sample weights support
- Comprehensive metric logging and experiment tracking
- Performance monitoring and timeout management
"""

try:
    import optuna

    _OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    optuna = None
    _OPTUNA_AVAILABLE = False
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Protocol
from pathlib import Path
import warnings
import json
from dataclasses import dataclass, field
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
)
import joblib
from concurrent.futures import TimeoutError
import time

from .base import BaseHyperparameterTuner, BaseModelTrainer, TrainingResult
from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    ModelTrainingError,
    ConfigurationError,
    PerformanceError,
    handle_and_reraise,
    validate_parameter,
)

logger = get_logger(__name__)


@dataclass
class ScoringConfig:
    """Type-safe configuration for scoring functions in hyperparameter tuning.

    This class provides comprehensive configuration for evaluation metrics
    with proper validation and default values for common ML scenarios.
    """

    # Primary scoring function
    scoring_function: str = "roc_auc"
    direction: str = "maximize"  # "maximize" or "minimize"

    # Additional metrics to log for every trial
    additional_metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1", "average_precision"]
    )

    # Custom scoring function (if not using built-in)
    custom_scorer: Optional[Callable[[Any, pd.DataFrame, pd.Series], float]] = None

    # Cross-validation parameters
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified", "kfold", "time_series"
    cv_shuffle: bool = True

    # Evaluation parameters
    run_full_evaluation: bool = True
    evaluation_sample_size: Optional[int] = None

    # Performance constraints
    timeout_per_trial: Optional[float] = None  # seconds
    memory_limit_gb: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate scoring configuration after initialization."""
        validate_parameter("direction", self.direction, valid_values=["maximize", "minimize"])
        validate_parameter("cv_folds", self.cv_folds, min_value=2, max_value=20)
        validate_parameter("cv_strategy", self.cv_strategy, valid_values=["stratified", "kfold", "time_series"])

        if self.evaluation_sample_size is not None:
            validate_parameter("evaluation_sample_size", self.evaluation_sample_size, min_value=100)

        if self.timeout_per_trial is not None:
            validate_parameter("timeout_per_trial", self.timeout_per_trial, min_value=1.0)

        # Validate built-in scoring functions
        valid_scorers = {
            "roc_auc",
            "average_precision",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "neg_log_loss",
            "matthews_corrcoef",
            "balanced_accuracy",
            "cohen_kappa",
        }

        if self.custom_scorer is None and self.scoring_function not in valid_scorers:
            logger.warning(f"Unknown scoring function: {self.scoring_function}")

        logger.debug(f"ScoringConfig validated: {self.scoring_function} ({self.direction})")


class EnhancedCustomScorer:
    """Type-safe wrapper for custom scoring functions with comprehensive error handling.

    Provides robust scoring with sample weights support, error recovery,
    and performance monitoring for evaluation metrics.
    """

    def __init__(
        self,
        scoring_function: Union[str, Callable],
        sample_weight: Optional[np.ndarray] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize enhanced custom scorer.

        Args:
            scoring_function: Name of scoring function or callable
            sample_weight: Sample weights for evaluation
            timeout: Optional timeout for scoring computation
        """
        self.scoring_function = scoring_function
        self.sample_weight = sample_weight
        self.timeout = timeout

        # Map string names to sklearn functions with sample weight support info
        self.scorer_map = {
            "roc_auc": (roc_auc_score, True),
            "average_precision": (average_precision_score, True),
            "accuracy": (accuracy_score, True),
            "precision": (precision_score, True),
            "recall": (recall_score, True),
            "f1": (f1_score, True),
            "neg_log_loss": (lambda y_true, y_pred: -log_loss(y_true, y_pred), True),
            "matthews_corrcoef": (matthews_corrcoef, False),
            "balanced_accuracy": (balanced_accuracy_score, True),
            "cohen_kappa": (cohen_kappa_score, False),
        }

    def __call__(self, estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
        """Score the estimator on given data with comprehensive error handling.

        Args:
            estimator: Trained model
            X: Features
            y: Target

        Returns:
            Score value

        Raises:
            ModelEvaluationError: If scoring fails critically
        """
        try:
            with timed_operation(f"scoring_{self.scoring_function}", timeout=self.timeout) as timing:
                # Get predictions
                if hasattr(estimator, "predict_proba"):
                    y_pred = estimator.predict_proba(X)[:, 1]  # Positive class probability
                    use_proba = True
                else:
                    y_pred = estimator.predict(X)
                    use_proba = False

                # Get scorer function
                if callable(self.scoring_function):
                    scorer_func = self.scoring_function
                    supports_weights = True  # Assume custom functions support weights
                elif isinstance(self.scoring_function, str) and self.scoring_function in self.scorer_map:
                    scorer_func, supports_weights = self.scorer_map[self.scoring_function]
                else:
                    raise ConfigurationError(f"Unknown scoring function: {self.scoring_function}")

                # Calculate score with appropriate prediction type and sample weights
                score = self._compute_score_safely(scorer_func, y, y_pred, use_proba, supports_weights)

                logger.debug(f"Scoring completed in {timing['duration']:.3f}s: {score:.4f}")
                return float(score)

        except TimeoutError:
            raise PerformanceError(
                f"Scoring with {self.scoring_function} exceeded timeout of {self.timeout}s",
                error_code="SCORING_TIMEOUT",
            )
        except Exception as e:
            handle_and_reraise(
                e,
                ModelEvaluationError,
                f"Failed to compute score with {self.scoring_function}",
                error_code="SCORING_FAILED",
                context={"scorer": str(self.scoring_function), "X_shape": X.shape, "y_shape": y.shape},
            )

    def _compute_score_safely(
        self, scorer_func: Callable, y_true: pd.Series, y_pred: np.ndarray, use_proba: bool, supports_weights: bool
    ) -> float:
        """Safely compute score with fallback strategies."""
        # Determine if we need binary predictions for this scorer
        binary_scorers = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "matthews_corrcoef",
            "balanced_accuracy",
            "cohen_kappa",
        }
        needs_binary = any(scorer in str(self.scoring_function).lower() for scorer in binary_scorers)

        if needs_binary and use_proba:
            y_pred_binary = (y_pred > 0.5).astype(int)
        else:
            y_pred_binary = y_pred

        final_pred = y_pred_binary if needs_binary else y_pred

        # Try with sample weights first if supported
        if self.sample_weight is not None and supports_weights:
            try:
                return scorer_func(y_true, final_pred, sample_weight=self.sample_weight)
            except TypeError:
                logger.warning(f"Scorer {self.scoring_function} doesn't support sample_weight parameter")
                supports_weights = False

        # Fall back to unweighted scoring
        try:
            return scorer_func(y_true, final_pred)
        except Exception as e:
            # Final fallback - try with different prediction format
            if not needs_binary and use_proba:
                logger.warning(f"Retrying {self.scoring_function} with binary predictions")
                return scorer_func(y_true, (y_pred > 0.5).astype(int))
            raise


if _OPTUNA_AVAILABLE:

    class OptunaHyperparameterTuner(BaseHyperparameterTuner):
        """Production-ready Optuna-based hyperparameter tuner with comprehensive features.

        (Full implementation available when `optuna` is installed.)
        """

    """Production-ready Optuna-based hyperparameter tuner with comprehensive features.
    
    Features:
    - Type-safe configuration and error handling
    - Advanced sampling strategies (TPE, CMA-ES, Random, Grid)
    - Pruning for early trial termination
    - Comprehensive metric logging
    - Performance monitoring and timeouts
    - Memory usage tracking
    - Experiment reproducibility
    
    Example:
        >>> from tree_models.models import OptunaHyperparameterTuner, StandardModelTrainer
        >>> from tree_models.config import ScoringConfig
        >>> 
        >>> trainer = StandardModelTrainer("xgboost")
        >>> config = ScoringConfig(scoring_function="recall", direction="maximize")
        >>> tuner = OptunaHyperparameterTuner(trainer, scoring_config=config)
        >>> best_params, best_score = tuner.optimize(X_train, y_train, search_space)
    """

    def __init__(
        self,
        model_trainer: BaseModelTrainer,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        scoring_config: Optional[ScoringConfig] = None,
        search_space: Optional[Dict[str, Any]] = None,
        sampler: str = "tpe",
        pruner: str = "median",
        study_name: Optional[str] = None,
        storage_url: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Optuna hyperparameter tuner.

        Args:
            model_trainer: Model trainer instance
            n_trials: Number of optimization trials
            timeout: Global timeout in seconds
            scoring_config: Configuration for scoring and evaluation
            search_space: Custom search space (overrides defaults)
            sampler: Optuna sampler ('tpe', 'random', 'grid', 'cmaes')
            pruner: Optuna pruner ('median', 'successive_halving', 'hyperband', 'nop')
            study_name: Optional study name for persistence
            storage_url: Optional database URL for study persistence
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs (1 recommended for stability)
            **kwargs: Additional parameters
        """
        # Initialize base class
        scoring_metric = scoring_config.scoring_function if scoring_config else "roc_auc"
        cv_folds = scoring_config.cv_folds if scoring_config else 5
        super().__init__(model_trainer, scoring_metric, cv_folds, **kwargs)

        # Validate parameters
        validate_parameter("n_trials", n_trials, min_value=1, max_value=10000)
        validate_parameter("sampler", sampler, valid_values=["tpe", "random", "grid", "cmaes"])
        validate_parameter("pruner", pruner, valid_values=["median", "successive_halving", "hyperband", "nop"])
        validate_parameter("n_jobs", n_jobs, min_value=1, max_value=16)

        # Store configuration
        self.n_trials = n_trials
        self.timeout = timeout
        self.scoring_config = scoring_config or ScoringConfig()
        self.search_space = search_space
        self.sampler_name = sampler
        self.pruner_name = pruner
        self.study_name = study_name
        self.storage_url = storage_url
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Runtime state
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.trial_results: List[Dict[str, Any]] = []

        # Data storage for objective function (set during optimization)
        self._X_train: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series] = None
        self._sample_weight: Optional[np.ndarray] = None

        logger.info(f"Initialized OptunaHyperparameterTuner:")
        logger.info(f"  Model: {model_trainer.model_type}")
        logger.info(f"  Trials: {n_trials}, Sampler: {sampler}, Pruner: {pruner}")
        logger.info(f"  Scoring: {self.scoring_config.scoring_function} ({self.scoring_config.direction})")

    def _create_study(self) -> optuna.Study:
        """Create and configure Optuna study."""
        direction = (
            optuna.study.StudyDirection.MAXIMIZE
            if self.scoring_config.direction == "maximize"
            else optuna.study.StudyDirection.MINIMIZE
        )

        # Create sampler
        sampler = self._get_sampler()

        # Create pruner
        pruner = self._get_pruner()

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        logger.info(f"Created Optuna study with {len(study.trials)} existing trials")
        return study

    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        """Get configured Optuna sampler."""
        if self.sampler_name == "tpe":
            return optuna.samplers.TPESampler(
                seed=self.random_state, n_startup_trials=min(10, self.n_trials // 10), n_ei_candidates=24
            )
        elif self.sampler_name == "random":
            return optuna.samplers.RandomSampler(seed=self.random_state)
        elif self.sampler_name == "grid":
            return optuna.samplers.GridSampler()
        elif self.sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=self.random_state, n_startup_trials=min(10, self.n_trials // 10))
        else:
            raise ConfigurationError(f"Unknown sampler: {self.sampler_name}")

    def _get_pruner(self) -> optuna.pruners.BasePruner:
        """Get configured Optuna pruner."""
        if self.pruner_name == "median":
            return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
        elif self.pruner_name == "successive_halving":
            return optuna.pruners.SuccessiveHalvingPruner()
        elif self.pruner_name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        elif self.pruner_name == "nop":
            return optuna.pruners.NopPruner()
        else:
            raise ConfigurationError(f"Unknown pruner: {self.pruner_name}")

    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default search space for the model type."""
        model_type = self.model_trainer.model_type.lower()

        if model_type == "xgboost":
            return {
                "n_estimators": ("int", 50, 1000),
                "max_depth": ("int", 3, 15),
                "learning_rate": ("float", 0.01, 0.3, True),  # True for log scale
                "subsample": ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.6, 1.0),
                "reg_alpha": ("float", 0.0, 10.0, True),
                "reg_lambda": ("float", 0.0, 10.0, True),
                "gamma": ("float", 0.0, 5.0),
                "min_child_weight": ("int", 1, 10),
            }
        elif model_type == "lightgbm":
            return {
                "n_estimators": ("int", 50, 1000),
                "max_depth": ("int", 3, 15),
                "learning_rate": ("float", 0.01, 0.3, True),
                "feature_fraction": ("float", 0.6, 1.0),
                "bagging_fraction": ("float", 0.6, 1.0),
                "bagging_freq": ("int", 1, 10),
                "min_child_samples": ("int", 5, 200),
                "reg_alpha": ("float", 0.0, 10.0, True),
                "reg_lambda": ("float", 0.0, 10.0, True),
                "num_leaves": ("int", 15, 300),
            }
        elif model_type == "catboost":
            return {
                "iterations": ("int", 50, 1000),
                "depth": ("int", 3, 12),
                "learning_rate": ("float", 0.01, 0.3, True),
                "l2_leaf_reg": ("float", 1.0, 10.0, True),
                "subsample": ("float", 0.6, 1.0),
                "colsample_bylevel": ("float", 0.6, 1.0),
                "min_data_in_leaf": ("int", 1, 100),
            }
        else:
            logger.warning(f"No default search space for model type: {model_type}")
            return {}

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the trial with type safety."""
        search_space = self.search_space or self._get_default_search_space()
        params = {}

        for param_name, param_config in search_space.items():
            try:
                if isinstance(param_config, tuple):
                    if len(param_config) >= 3:
                        param_type, low, high = param_config[:3]
                        log_scale = param_config[3] if len(param_config) > 3 else False

                        if param_type == "int":
                            params[param_name] = trial.suggest_int(param_name, low, high)
                        elif param_type == "float":
                            params[param_name] = trial.suggest_float(param_name, low, high, log=log_scale)
                        else:
                            raise ConfigurationError(f"Unknown parameter type: {param_type}")
                    else:
                        raise ConfigurationError(f"Invalid parameter config for {param_name}: {param_config}")

                elif isinstance(param_config, (list, tuple)) and not isinstance(param_config[0], str):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
                else:
                    logger.warning(f"Skipping invalid parameter config for {param_name}: {param_config}")

            except Exception as e:
                logger.error(f"Failed to suggest parameter {param_name}: {e}")
                continue

        return params

    @timer(name="optuna_objective", log_result=False)
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization with comprehensive error handling."""
        trial_start_time = time.time()

        try:
            # Check trial timeout
            if self.scoring_config.timeout_per_trial:
                trial.set_user_attr("timeout", self.scoring_config.timeout_per_trial)

            # Suggest hyperparameters
            params = self._suggest_params(trial)
            if not params:
                raise ConfigurationError("No valid parameters suggested")

            # Log trial start
            logger.debug(f"Trial {trial.number} started with params: {params}")

            # Perform cross-validation
            cv_score, cv_std, additional_metrics = self._evaluate_params_with_cv(params)

            # Store trial results
            trial_duration = time.time() - trial_start_time

            # Set trial attributes
            trial.set_user_attr("cv_mean", cv_score)
            trial.set_user_attr("cv_std", cv_std)
            trial.set_user_attr("duration", trial_duration)
            trial.set_user_attr("params", params)

            # Set additional metrics as attributes
            for metric_name, metric_value in additional_metrics.items():
                trial.set_user_attr(f"metric_{metric_name}", metric_value)

            # Store comprehensive trial result
            trial_result = {
                "trial_number": trial.number,
                "params": params.copy(),
                "cv_mean": cv_score,
                "cv_std": cv_std,
                "duration": trial_duration,
                "additional_metrics": additional_metrics.copy(),
            }
            self.trial_results.append(trial_result)

            logger.info(
                f"Trial {trial.number}: {self.scoring_config.scoring_function}="
                f"{cv_score:.4f}±{cv_std:.4f} ({trial_duration:.1f}s)"
            )

            return cv_score

        except optuna.TrialPruned:
            logger.debug(f"Trial {trial.number} was pruned")
            raise

        except Exception as e:
            trial_duration = time.time() - trial_start_time
            logger.error(f"Trial {trial.number} failed after {trial_duration:.1f}s: {e}")

            # Return worst possible score to continue optimization
            return float("-inf") if self.scoring_config.direction == "maximize" else float("inf")

    def _evaluate_params_with_cv(self, params: Dict[str, Any]) -> Tuple[float, float, Dict[str, float]]:
        """Evaluate parameters using cross-validation with comprehensive metrics."""
        # Create cross-validation strategy
        if self.scoring_config.cv_strategy == "stratified":
            cv = StratifiedKFold(
                n_splits=self.scoring_config.cv_folds,
                shuffle=self.scoring_config.cv_shuffle,
                random_state=self.random_state,
            )
        else:
            from sklearn.model_selection import KFold

            cv = KFold(
                n_splits=self.scoring_config.cv_folds,
                shuffle=self.scoring_config.cv_shuffle,
                random_state=self.random_state,
            )

        # Create model with parameters
        model = self.model_trainer.get_model(params)

        # Create scorer
        scorer = EnhancedCustomScorer(
            self.scoring_config.scoring_function, self._sample_weight, self.scoring_config.timeout_per_trial
        )

        # Perform cross-validation
        try:
            cv_scores = cross_val_score(
                estimator=model,
                X=self._X_train,
                y=self._y_train,
                cv=cv,
                scoring=scorer,
                n_jobs=1,  # Avoid nested parallelism issues
                error_score="raise",
            )

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

        except Exception as e:
            handle_and_reraise(
                e, ModelTrainingError, f"Cross-validation failed for params: {params}", error_code="CV_FAILED"
            )

        # Evaluate additional metrics if requested
        additional_metrics = {}
        if self.scoring_config.run_full_evaluation and self.scoring_config.additional_metrics:
            additional_metrics = self._evaluate_additional_metrics(model, params)

        return cv_mean, cv_std, additional_metrics

    def _evaluate_additional_metrics(self, model: Any, params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate additional metrics on full dataset."""
        additional_metrics = {}

        # Sample data if needed for performance
        X_eval = self._X_train
        y_eval = self._y_train
        sample_weight_eval = self._sample_weight

        if self.scoring_config.evaluation_sample_size and len(X_eval) > self.scoring_config.evaluation_sample_size:

            indices = np.random.choice(len(X_eval), self.scoring_config.evaluation_sample_size, replace=False)
            X_eval = X_eval.iloc[indices]
            y_eval = y_eval.iloc[indices]
            if sample_weight_eval is not None:
                sample_weight_eval = sample_weight_eval[indices]

        # Train model on evaluation data
        try:
            eval_model = self.model_trainer.get_model(params)
            if sample_weight_eval is not None:
                eval_model.fit(X_eval, y_eval, sample_weight=sample_weight_eval)
            else:
                eval_model.fit(X_eval, y_eval)

        except Exception as e:
            logger.warning(f"Failed to train model for additional metrics: {e}")
            return additional_metrics

        # Evaluate each additional metric
        for metric_name in self.scoring_config.additional_metrics:
            if metric_name == self.scoring_config.scoring_function:
                continue  # Skip primary metric

            try:
                scorer = EnhancedCustomScorer(metric_name, sample_weight_eval)
                score = scorer(eval_model, X_eval, y_eval)
                additional_metrics[metric_name] = float(score)

            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")
                additional_metrics[metric_name] = 0.0

        return additional_metrics

    @timer(name="hyperparameter_optimization")
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: Optional[Dict[str, Any]] = None,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], float]:
        """Run hyperparameter optimization with comprehensive error handling.

        Args:
            X: Training features
            y: Training targets
            search_space: Optional custom search space
            sample_weight: Optional sample weights
            **kwargs: Additional optimization parameters

        Returns:
            Tuple of (best_parameters, best_score)

        Raises:
            ModelTrainingError: If optimization fails
        """
        logger.info(f"Starting hyperparameter optimization:")
        logger.info(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"  Trials: {self.n_trials}, Timeout: {self.timeout}s")
        logger.info(f"  Metric: {self.scoring_config.scoring_function} ({self.scoring_config.direction})")

        # Validate input data
        self.model_trainer.validate_input_data(X, y, sample_weight)

        # Store data for objective function
        self._X_train = X.copy()
        self._y_train = y.copy()
        self._sample_weight = sample_weight.copy() if sample_weight is not None else None

        # Use custom search space if provided
        if search_space:
            self.search_space = search_space

        try:
            # Create study
            self.study = self._create_study()

            # Run optimization
            self.study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=True,
                callbacks=[self._log_trial_callback],
            )

            # Extract results
            if not self.study.trials:
                raise ModelTrainingError("No successful trials completed")

            self.best_params = self.study.best_params.copy()
            self.best_score = self.study.best_value

            # Log summary
            logger.info(f"✅ Optimization completed!")
            logger.info(f"  Best {self.scoring_config.scoring_function}: {self.best_score:.4f}")
            logger.info(f"  Best parameters: {self.best_params}")
            logger.info(f"  Completed trials: {len(self.study.trials)}")

            return self.best_params, self.best_score

        except Exception as e:
            handle_and_reraise(
                e,
                ModelTrainingError,
                "Hyperparameter optimization failed",
                error_code="OPTIMIZATION_FAILED",
                context={"n_trials": self.n_trials, "model_type": self.model_trainer.model_type},
            )

        finally:
            # Clean up stored data
            self._X_train = None
            self._y_train = None
            self._sample_weight = None

    def _log_trial_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback for logging trial progress."""
        if trial.value is not None:
            logger.debug(f"Trial {trial.number} completed: value={trial.value:.4f}")

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame with comprehensive trial information.

        Returns:
            DataFrame with detailed trial results
        """
        if not self.trial_results:
            logger.warning("No trial results available")
            return pd.DataFrame()

        # Convert to DataFrame with flattened structure
        records = []

        for result in self.trial_results:
            record = {
                "trial_number": result["trial_number"],
                "cv_mean": result["cv_mean"],
                "cv_std": result["cv_std"],
                "duration": result["duration"],
            }

            # Add parameters with prefix
            for param_name, param_value in result["params"].items():
                record[f"param_{param_name}"] = param_value

            # Add additional metrics with prefix
            for metric_name, metric_value in result["additional_metrics"].items():
                record[f"metric_{metric_name}"] = metric_value

            records.append(record)

        df = pd.DataFrame(records)

        # Sort by primary metric
        sort_ascending = self.scoring_config.direction == "minimize"
        df = df.sort_values("cv_mean", ascending=sort_ascending)

        logger.info(f"Optimization history: {len(df)} trials, {len(df.columns)} columns")
        return df

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get parameter importance from optimization study.

        Returns:
            DataFrame with parameter importance scores or None if unavailable
        """
        if self.study is None or len(self.study.trials) < 10:
            logger.warning("Insufficient trials for parameter importance analysis")
            return None

        try:
            importance = optuna.importance.get_param_importances(self.study)

            df = pd.DataFrame(
                [{"parameter": param, "importance": imp} for param, imp in importance.items()]
            ).sort_values("importance", ascending=False)

            logger.info(f"Parameter importance calculated for {len(df)} parameters")
            return df

        except Exception as e:
            logger.error(f"Failed to calculate parameter importance: {e}")
            return None

    def save_study(self, filepath: Union[str, Path]) -> None:
        """Save Optuna study to file with error handling.

        Args:
            filepath: Path to save study

        Raises:
            FileOperationError: If save operation fails
        """
        if self.study is None:
            raise ConfigurationError("No study available to save")

        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save study using joblib for better compatibility
            joblib.dump(self.study, filepath)
            logger.info(f"Study saved to {filepath}")

        except Exception as e:
            from ..utils.exceptions import FileOperationError

            handle_and_reraise(
                e, FileOperationError, f"Failed to save study to {filepath}", error_code="STUDY_SAVE_FAILED"
            )

    def load_study(self, filepath: Union[str, Path]) -> None:
        """Load Optuna study from file with error handling.

        Args:
            filepath: Path to load study from

        Raises:
            FileOperationError: If load operation fails
        """
        try:
            self.study = joblib.load(filepath)

            if self.study.best_trial:
                self.best_params = self.study.best_params.copy()
                self.best_score = self.study.best_value

            logger.info(f"Study loaded from {filepath} with {len(self.study.trials)} trials")

        except Exception as e:
            from ..utils.exceptions import FileOperationError

            handle_and_reraise(
                e, FileOperationError, f"Failed to load study from {filepath}", error_code="STUDY_LOAD_FAILED"
            )

else:

    class OptunaHyperparameterTuner(BaseHyperparameterTuner):
        """Stub that raises a clear error when Optuna is not installed.

        This allows the package to be imported in environments where Optuna
        is not available (e.g. minimal CI environments). Attempting to
        instantiate or use this tuner will raise a ConfigurationError.
        """

        def __init__(self, *args, **kwargs):
            from ..utils.exceptions import ConfigurationError

            raise ConfigurationError(
                "Optuna is not installed. Install 'optuna' to use OptunaHyperparameterTuner",
                error_code="OPTUNA_NOT_AVAILABLE",
            )


# Convenience functions for backward compatibility and easy usage
def tune_hyperparameters(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    scoring_function: str = "roc_auc",
    additional_metrics: Optional[List[str]] = None,
    n_trials: int = 100,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], float]:
    """Convenience function for hyperparameter tuning.

    Args:
        model_type: Type of model to tune ('xgboost', 'lightgbm', 'catboost')
        X: Training features
        y: Training targets
        sample_weight: Optional sample weights
        scoring_function: Primary scoring function
        additional_metrics: Additional metrics to log
        n_trials: Number of optimization trials
        timeout: Optional timeout in seconds
        **kwargs: Additional tuner arguments

    Returns:
        Tuple of (best_parameters, best_score)

    Example:
        >>> best_params, best_score = tune_hyperparameters(
        ...     'xgboost', X_train, y_train,
        ...     sample_weight=weights,
        ...     scoring_function='recall',
        ...     additional_metrics=['precision', 'f1'],
        ...     n_trials=50,
        ...     timeout=3600
        ... )
    """
    from .trainer import StandardModelTrainer  # Import here to avoid circular imports

    # Create model trainer
    trainer = StandardModelTrainer(model_type, random_state=kwargs.get("random_state", 42))

    # Create scoring configuration
    additional_metrics = additional_metrics or ["accuracy", "precision", "recall", "f1"]
    scoring_config = ScoringConfig(
        scoring_function=scoring_function, additional_metrics=additional_metrics, run_full_evaluation=True
    )

    # Create and run tuner
    tuner = OptunaHyperparameterTuner(
        model_trainer=trainer, n_trials=n_trials, timeout=timeout, scoring_config=scoring_config, **kwargs
    )

    return tuner.optimize(X, y, sample_weight=sample_weight)
