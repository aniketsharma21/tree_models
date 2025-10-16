"""Enhanced hyperparameter tuning with Optuna, MLflow, and comprehensive evaluation.

This module provides advanced hyperparameter tuning capabilities with:
- Custom scoring functions with direction control
- Comprehensive metric logging for every trial
- Sample weights support throughout
- MLflow integration for experiment tracking
- Multi-model comparison capabilities
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from pathlib import Path
import warnings
import pickle
from dataclasses import dataclass, field
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score, 
    recall_score, f1_score, log_loss, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score
)
import joblib

from ..utils.logger import get_logger
from ..utils.timer import timer

logger = get_logger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for scoring functions in hyperparameter tuning."""

    # Primary scoring function
    scoring_function: str = "roc_auc"  # Main metric to optimize
    direction: str = "maximize"  # "maximize" or "minimize"

    # Additional metrics to log for every trial
    additional_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "average_precision"
    ])

    # Custom scoring function (if not using built-in)
    custom_scorer: Optional[Callable] = None

    # Cross-validation parameters
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified", "kfold", "time_series"

    # Evaluation parameters
    run_full_evaluation: bool = True  # Run complete evaluation on each trial
    evaluation_sample_size: Optional[int] = None  # Subsample for faster evaluation

    def __post_init__(self):
        """Validate scoring configuration."""
        valid_directions = ["maximize", "minimize"]
        if self.direction not in valid_directions:
            raise ValueError(f"direction must be one of {valid_directions}")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")

        # Validate built-in scoring functions
        valid_scorers = [
            "roc_auc", "average_precision", "accuracy", "precision", "recall", 
            "f1", "neg_log_loss", "matthews_corrcoef", "balanced_accuracy"
        ]

        if self.custom_scorer is None and self.scoring_function not in valid_scorers:
            logger.warning(f"Unknown scoring function: {self.scoring_function}")


class CustomScorer:
    """Wrapper for custom scoring functions with sample weights support."""

    def __init__(self, scoring_function: str, sample_weight: Optional[np.ndarray] = None):
        """Initialize custom scorer.

        Args:
            scoring_function: Name of scoring function or callable
            sample_weight: Sample weights for evaluation
        """
        self.scoring_function = scoring_function
        self.sample_weight = sample_weight

        # Map string names to sklearn functions
        self.scorer_map = {
            "roc_auc": roc_auc_score,
            "average_precision": average_precision_score,
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "neg_log_loss": lambda y_true, y_pred: -log_loss(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef,
            "balanced_accuracy": balanced_accuracy_score,
            "cohen_kappa": cohen_kappa_score
        }

    def __call__(self, estimator, X, y) -> float:
        """Score the estimator on given data.

        Args:
            estimator: Trained model
            X: Features
            y: Target

        Returns:
            Score value
        """
        try:
            # Get predictions
            if hasattr(estimator, "predict_proba"):
                y_pred = estimator.predict_proba(X)[:, 1]  # Probability of positive class
            else:
                y_pred = estimator.predict(X)

            # Get scorer function
            if callable(self.scoring_function):
                scorer_func = self.scoring_function
            elif self.scoring_function in self.scorer_map:
                scorer_func = self.scorer_map[self.scoring_function]
            else:
                raise ValueError(f"Unknown scoring function: {self.scoring_function}")

            # Calculate score with sample weights if supported
            try:
                if self.sample_weight is not None:
                    # Check if scorer supports sample weights
                    if self.scoring_function in ["roc_auc", "average_precision", "accuracy", 
                                                "precision", "recall", "f1", "balanced_accuracy"]:
                        score = scorer_func(y, y_pred, sample_weight=self.sample_weight)
                    else:
                        # Fallback to unweighted score
                        score = scorer_func(y, y_pred)
                        logger.warning(f"Sample weights not supported for {self.scoring_function}")
                else:
                    score = scorer_func(y, y_pred)

                return float(score)

            except Exception as e:
                logger.warning(f"Error computing weighted score, falling back to unweighted: {e}")
                # Fallback to basic scoring
                if self.scoring_function in ["roc_auc", "average_precision"] and hasattr(estimator, "predict_proba"):
                    score = scorer_func(y, y_pred)
                else:
                    y_pred_binary = (y_pred > 0.5).astype(int) if hasattr(estimator, "predict_proba") else y_pred
                    score = scorer_func(y, y_pred_binary)

                return float(score)

        except Exception as e:
            logger.error(f"Error in custom scorer: {e}")
            return 0.0  # Return neutral score on error


class EnhancedOptunaHyperparameterTuner:
    """Enhanced hyperparameter tuner with custom scoring and comprehensive evaluation.

    Features:
    - Custom scoring functions with direction control
    - Comprehensive metric logging for every trial
    - Sample weights support
    - MLflow integration
    - Multi-model comparison
    - Advanced pruning and sampling strategies

    Example:
        >>> tuner = EnhancedOptunaHyperparameterTuner(
        ...     model_type='xgboost',
        ...     n_trials=100,
        ...     scoring_config=ScoringConfig(
        ...         scoring_function='recall',
        ...         direction='maximize',
        ...         additional_metrics=['precision', 'f1', 'roc_auc']
        ...     )
        ... )
        >>> best_params, best_score = tuner.optimize(X, y, sample_weight=weights)
    """

    def __init__(self,
                 model_type: str = "xgboost",
                 n_trials: int = 100,
                 timeout: Optional[int] = None,
                 scoring_config: Optional[ScoringConfig] = None,
                 search_space: Optional[Dict[str, Any]] = None,
                 sampler: str = "tpe",
                 pruner: str = "median",
                 n_jobs: int = 1,
                 mlflow_logger = None,
                 random_state: int = 42):
        """Initialize enhanced hyperparameter tuner.

        Args:
            model_type: Type of model to tune ('xgboost', 'lightgbm', 'catboost')
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (optional)
            scoring_config: Configuration for scoring and evaluation
            search_space: Custom search space (overrides defaults)
            sampler: Optuna sampler ('tpe', 'random', 'grid', 'cmaes')
            pruner: Optuna pruner ('median', 'successive_halving', 'hyperband')
            n_jobs: Number of parallel jobs
            mlflow_logger: MLflow logger instance
            random_state: Random state for reproducibility
        """
        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.timeout = timeout
        self.scoring_config = scoring_config or ScoringConfig()
        self.search_space = search_space
        self.sampler_name = sampler
        self.pruner_name = pruner
        self.n_jobs = n_jobs
        self.mlflow_logger = mlflow_logger
        self.random_state = random_state

        # Study and results storage
        self.study = None
        self.best_params = None
        self.best_score = None
        self.trial_results = []

        # Validation
        valid_models = ["xgboost", "lightgbm", "catboost"]
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}")

        logger.info(f"Initialized enhanced tuner for {model_type}")
        logger.info(f"Scoring: {self.scoring_config.scoring_function} ({self.scoring_config.direction})")
        logger.info(f"Additional metrics: {self.scoring_config.additional_metrics}")

    def _get_sampler(self):
        """Get Optuna sampler based on configuration."""
        if self.sampler_name == "tpe":
            return optuna.samplers.TPESampler(seed=self.random_state)
        elif self.sampler_name == "random":
            return optuna.samplers.RandomSampler(seed=self.random_state)
        elif self.sampler_name == "grid":
            return optuna.samplers.GridSampler()
        elif self.sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=self.random_state)
        else:
            logger.warning(f"Unknown sampler {self.sampler_name}, using TPE")
            return optuna.samplers.TPESampler(seed=self.random_state)

    def _get_pruner(self):
        """Get Optuna pruner based on configuration."""
        if self.pruner_name == "median":
            return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.pruner_name == "successive_halving":
            return optuna.pruners.SuccessiveHalvingPruner()
        elif self.pruner_name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        elif self.pruner_name == "nop":
            return optuna.pruners.NopPruner()
        else:
            logger.warning(f"Unknown pruner {self.pruner_name}, using median")
            return optuna.pruners.MedianPruner()

    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default search space for the model type."""
        if self.model_type == "xgboost":
            return {
                "n_estimators": (50, 500),
                "max_depth": (3, 12), 
                "learning_rate": (0.01, 0.3),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0),
                "reg_alpha": (0.0, 10.0),
                "reg_lambda": (0.0, 10.0),
                "gamma": (0.0, 5.0)
            }
        elif self.model_type == "lightgbm":
            return {
                "n_estimators": (50, 500),
                "max_depth": (3, 12),
                "learning_rate": (0.01, 0.3),
                "feature_fraction": (0.6, 1.0),
                "bagging_fraction": (0.6, 1.0),
                "bagging_freq": (1, 10),
                "min_child_samples": (5, 100),
                "reg_alpha": (0.0, 10.0),
                "reg_lambda": (0.0, 10.0)
            }
        elif self.model_type == "catboost":
            return {
                "iterations": (50, 500),
                "depth": (3, 12),
                "learning_rate": (0.01, 0.3),
                "l2_leaf_reg": (1, 10),
                "subsample": (0.6, 1.0),
                "colsample_bylevel": (0.6, 1.0)
            }
        else:
            return {}

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the trial."""
        search_space = self.search_space or self._get_default_search_space()
        params = {}

        for param_name, param_range in search_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                low, high = param_range
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                elif isinstance(low, float) or isinstance(high, float):
                    params[param_name] = trial.suggest_float(param_name, low, high, log=param_name in ['learning_rate'])
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                logger.warning(f"Invalid parameter range for {param_name}: {param_range}")

        return params

    def _get_estimator(self, model_type: str, params: Dict[str, Any]):
        """Get estimator instance for given model type and parameters."""
        if model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(**params, random_state=self.random_state, 
                                    objective='binary:logistic', eval_metric='auc')
        elif model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params, random_state=self.random_state,
                                     objective='binary', metric='auc')
        elif model_type == "catboost":
            import catboost as cb
            return cb.CatBoostClassifier(**params, random_seed=self.random_state,
                                        loss_function='Logloss', eval_metric='AUC', verbose=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _evaluate_trial_comprehensive(self, params: Dict[str, Any], 
                                    X: pd.DataFrame, y: pd.Series,
                                    sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Comprehensive evaluation of a trial including all requested metrics.

        Args:
            params: Hyperparameters to evaluate
            X: Features
            y: Target
            sample_weight: Sample weights

        Returns:
            Dictionary of metric names to values
        """
        try:
            # Subsample for faster evaluation if requested
            if (self.scoring_config.evaluation_sample_size and 
                len(X) > self.scoring_config.evaluation_sample_size):
                indices = np.random.choice(len(X), self.scoring_config.evaluation_sample_size, replace=False)
                X_eval = X.iloc[indices]
                y_eval = y.iloc[indices]
                sample_weight_eval = sample_weight[indices] if sample_weight is not None else None
            else:
                X_eval, y_eval, sample_weight_eval = X, y, sample_weight

            # Train model with current parameters
            model = self._get_estimator(self.model_type, params)

            if sample_weight_eval is not None:
                model.fit(X_eval, y_eval, sample_weight=sample_weight_eval)
            else:
                model.fit(X_eval, y_eval)

            # Get predictions
            y_pred_proba = model.predict_proba(X_eval)[:, 1]
            y_pred_binary = (y_pred_proba > 0.5).astype(int)

            # Calculate all requested metrics
            metrics = {}

            # Primary scoring function
            if self.scoring_config.custom_scorer:
                primary_score = self.scoring_config.custom_scorer(y_eval, y_pred_proba, sample_weight_eval)
            else:
                scorer = CustomScorer(self.scoring_config.scoring_function, sample_weight_eval)
                primary_score = scorer(model, X_eval, y_eval)

            metrics[self.scoring_config.scoring_function] = primary_score

            # Additional metrics
            for metric_name in self.scoring_config.additional_metrics:
                try:
                    if metric_name == "roc_auc":
                        if sample_weight_eval is not None:
                            score = roc_auc_score(y_eval, y_pred_proba, sample_weight=sample_weight_eval)
                        else:
                            score = roc_auc_score(y_eval, y_pred_proba)
                    elif metric_name == "average_precision":
                        if sample_weight_eval is not None:
                            score = average_precision_score(y_eval, y_pred_proba, sample_weight=sample_weight_eval)
                        else:
                            score = average_precision_score(y_eval, y_pred_proba)
                    elif metric_name == "accuracy":
                        if sample_weight_eval is not None:
                            score = accuracy_score(y_eval, y_pred_binary, sample_weight=sample_weight_eval)
                        else:
                            score = accuracy_score(y_eval, y_pred_binary)
                    elif metric_name == "precision":
                        if sample_weight_eval is not None:
                            score = precision_score(y_eval, y_pred_binary, sample_weight=sample_weight_eval, zero_division=0)
                        else:
                            score = precision_score(y_eval, y_pred_binary, zero_division=0)
                    elif metric_name == "recall":
                        if sample_weight_eval is not None:
                            score = recall_score(y_eval, y_pred_binary, sample_weight=sample_weight_eval, zero_division=0)
                        else:
                            score = recall_score(y_eval, y_pred_binary, zero_division=0)
                    elif metric_name == "f1":
                        if sample_weight_eval is not None:
                            score = f1_score(y_eval, y_pred_binary, sample_weight=sample_weight_eval, zero_division=0)
                        else:
                            score = f1_score(y_eval, y_pred_binary, zero_division=0)
                    elif metric_name == "log_loss":
                        if sample_weight_eval is not None:
                            score = -log_loss(y_eval, y_pred_proba, sample_weight=sample_weight_eval)
                        else:
                            score = -log_loss(y_eval, y_pred_proba)
                    elif metric_name == "matthews_corrcoef":
                        score = matthews_corrcoef(y_eval, y_pred_binary)
                    elif metric_name == "balanced_accuracy":
                        if sample_weight_eval is not None:
                            score = balanced_accuracy_score(y_eval, y_pred_binary, sample_weight=sample_weight_eval)
                        else:
                            score = balanced_accuracy_score(y_eval, y_pred_binary)
                    else:
                        logger.warning(f"Unknown metric: {metric_name}")
                        continue

                    metrics[metric_name] = float(score)

                except Exception as e:
                    logger.warning(f"Error calculating metric {metric_name}: {e}")
                    metrics[metric_name] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            return {self.scoring_config.scoring_function: 0.0}

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        try:
            # Suggest hyperparameters
            params = self._suggest_params(trial)

            # Cross-validation score (primary optimization target)
            if self.scoring_config.cv_strategy == "stratified":
                cv = StratifiedKFold(n_splits=self.scoring_config.cv_folds, shuffle=True, random_state=self.random_state)
            else:
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=self.scoring_config.cv_folds, shuffle=True, random_state=self.random_state)

            # Create custom scorer
            scorer = CustomScorer(self.scoring_config.scoring_function, self._sample_weight)

            # Perform cross-validation
            estimator = self._get_estimator(self.model_type, params)

            scores = cross_val_score(
                estimator=estimator,
                X=self._X_train,
                y=self._y_train,
                cv=cv,
                scoring=scorer,
                n_jobs=1,  # Avoid nested parallelism
                error_score=0.0
            )

            mean_cv_score = np.mean(scores)
            std_cv_score = np.std(scores)

            # Log cross-validation metrics
            trial.set_user_attr("cv_mean", mean_cv_score)
            trial.set_user_attr("cv_std", std_cv_score)
            trial.set_user_attr("cv_scores", scores.tolist())

            # Comprehensive evaluation if requested
            if self.scoring_config.run_full_evaluation:
                comprehensive_metrics = self._evaluate_trial_comprehensive(
                    params, self._X_train, self._y_train, self._sample_weight
                )

                # Log all comprehensive metrics
                for metric_name, metric_value in comprehensive_metrics.items():
                    trial.set_user_attr(f"eval_{metric_name}", metric_value)

            # Log to MLflow if available
            if self.mlflow_logger:
                # Log parameters
                mlflow_params = {f"trial_{k}": v for k, v in params.items()}
                mlflow_params.update({
                    "trial_number": trial.number,
                    "cv_mean": mean_cv_score,
                    "cv_std": std_cv_score
                })
                self.mlflow_logger.log_params(mlflow_params)

                # Log metrics
                mlflow_metrics = {
                    f"trial_cv_{self.scoring_config.scoring_function}": mean_cv_score,
                    f"trial_cv_{self.scoring_config.scoring_function}_std": std_cv_score
                }

                if self.scoring_config.run_full_evaluation:
                    for metric_name, metric_value in comprehensive_metrics.items():
                        mlflow_metrics[f"trial_{metric_name}"] = metric_value

                self.mlflow_logger.log_metrics(mlflow_metrics)

            # Store trial results
            trial_result = {
                "trial_number": trial.number,
                "params": params.copy(),
                "cv_mean": mean_cv_score,
                "cv_std": std_cv_score
            }

            if self.scoring_config.run_full_evaluation:
                trial_result.update(comprehensive_metrics)

            self.trial_results.append(trial_result)

            logger.info(f"Trial {trial.number}: {self.scoring_config.scoring_function}={mean_cv_score:.4f}±{std_cv_score:.4f}")

            return mean_cv_score

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            return 0.0 if self.scoring_config.direction == "maximize" else float('inf')

    @timer
    def optimize(self, X: pd.DataFrame, y: pd.Series,
                sample_weight: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], float]:
        """Run hyperparameter optimization.

        Args:
            X: Training features
            y: Training target
            sample_weight: Sample weights (optional)

        Returns:
            Tuple of (best_parameters, best_score)
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        logger.info(f"Optimizing {self.scoring_config.scoring_function} ({self.scoring_config.direction})")

        # Store data for objective function
        self._X_train = X
        self._y_train = y
        self._sample_weight = sample_weight

        # Create study
        direction = "maximize" if self.scoring_config.direction == "maximize" else "minimize"

        self.study = optuna.create_study(
            direction=direction,
            sampler=self._get_sampler(),
            pruner=self._get_pruner()
        )

        # Run optimization
        try:
            self.study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=True
            )

            # Get best results
            self.best_params = self.study.best_params.copy()
            self.best_score = self.study.best_value

            logger.info(f"✅ Optimization completed!")
            logger.info(f"Best {self.scoring_config.scoring_function}: {self.best_score:.4f}")
            logger.info(f"Best parameters: {self.best_params}")

            # Log final results to MLflow
            if self.mlflow_logger:
                self.mlflow_logger.log_params({f"best_{k}": v for k, v in self.best_params.items()})
                self.mlflow_logger.log_metrics({
                    f"best_{self.scoring_config.scoring_function}": self.best_score,
                    "total_trials": len(self.study.trials)
                })

            return self.best_params, self.best_score

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
        finally:
            # Clean up stored data
            self._X_train = None
            self._y_train = None
            self._sample_weight = None

    def get_trial_results_dataframe(self) -> pd.DataFrame:
        """Get detailed results from all trials as DataFrame.

        Returns:
            DataFrame with trial results and all logged metrics
        """
        if not self.trial_results:
            logger.warning("No trial results available")
            return pd.DataFrame()

        # Convert nested dictionaries to flat structure
        flat_results = []
        for result in self.trial_results:
            flat_result = {}

            # Add basic trial info
            flat_result["trial_number"] = result["trial_number"]
            flat_result["cv_mean"] = result["cv_mean"]
            flat_result["cv_std"] = result["cv_std"]

            # Add parameters
            for param, value in result["params"].items():
                flat_result[f"param_{param}"] = value

            # Add evaluation metrics
            for key, value in result.items():
                if key not in ["trial_number", "params", "cv_mean", "cv_std"]:
                    flat_result[key] = value

            flat_results.append(flat_result)

        df = pd.DataFrame(flat_results)

        # Sort by primary metric
        sort_ascending = self.scoring_config.direction == "minimize"
        df = df.sort_values("cv_mean", ascending=sort_ascending)

        logger.info(f"Trial results DataFrame created: {len(df)} trials, {len(df.columns)} columns")

        return df

    def get_feature_importance(self) -> pd.DataFrame:
        """Get hyperparameter importance from optimization study.

        Returns:
            DataFrame with parameter importance scores
        """
        if self.study is None:
            raise ValueError("Must run optimization first")

        try:
            # Get parameter importance
            importance = optuna.importance.get_param_importances(self.study)

            # Convert to DataFrame
            df = pd.DataFrame([
                {"parameter": param, "importance": imp}
                for param, imp in importance.items()
            ]).sort_values("importance", ascending=False)

            logger.info(f"Parameter importance calculated for {len(df)} parameters")

            return df

        except Exception as e:
            logger.error(f"Error calculating parameter importance: {e}")
            return pd.DataFrame()

    def save_study(self, filepath: Union[str, Path]):
        """Save Optuna study to file.

        Args:
            filepath: Path to save study
        """
        if self.study is None:
            raise ValueError("Must run optimization first")

        with open(filepath, 'wb') as f:
            pickle.dump(self.study, f)

        logger.info(f"Study saved to {filepath}")

    def load_study(self, filepath: Union[str, Path]):
        """Load Optuna study from file.

        Args:
            filepath: Path to load study from
        """
        with open(filepath, 'rb') as f:
            self.study = pickle.load(f)

        if self.study.best_trial:
            self.best_params = self.study.best_params.copy()
            self.best_score = self.study.best_value

        logger.info(f"Study loaded from {filepath}")


# Legacy compatibility - keep original class name as alias
OptunaHyperparameterTuner = EnhancedOptunaHyperparameterTuner


class MultiModelTuner:
    """Enhanced multi-model tuner with comprehensive evaluation.

    Compares multiple model types with comprehensive metric logging.
    """

    def __init__(self, 
                 model_types: List[str] = ["xgboost", "lightgbm", "catboost"],
                 n_trials_per_model: int = 50,
                 scoring_config: Optional[ScoringConfig] = None,
                 mlflow_logger = None):
        """Initialize multi-model tuner.

        Args:
            model_types: List of model types to compare
            n_trials_per_model: Number of trials per model
            scoring_config: Scoring configuration
            mlflow_logger: MLflow logger
        """
        self.model_types = model_types
        self.n_trials_per_model = n_trials_per_model
        self.scoring_config = scoring_config or ScoringConfig()
        self.mlflow_logger = mlflow_logger

        self.tuners = {}
        self.results = {}

    def optimize_all(self, X: pd.DataFrame, y: pd.Series,
                    sample_weight: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
        """Optimize all model types and return comparison.

        Args:
            X: Training features
            y: Training target
            sample_weight: Sample weights

        Returns:
            Dictionary with results for each model type
        """
        logger.info(f"Starting multi-model optimization for {len(self.model_types)} models")

        for model_type in self.model_types:
            logger.info(f"\n🔧 Optimizing {model_type.upper()}...")

            # Create tuner for this model
            tuner = EnhancedOptunaHyperparameterTuner(
                model_type=model_type,
                n_trials=self.n_trials_per_model,
                scoring_config=self.scoring_config,
                mlflow_logger=self.mlflow_logger
            )

            # Optimize
            best_params, best_score = tuner.optimize(X, y, sample_weight)

            # Store results
            self.tuners[model_type] = tuner
            self.results[model_type] = {
                "best_params": best_params,
                "best_score": best_score,
                "trial_results": tuner.get_trial_results_dataframe()
            }

            logger.info(f"✅ {model_type.upper()}: {self.scoring_config.scoring_function}={best_score:.4f}")

        # Create comparison summary
        comparison_df = self._create_comparison_summary()
        logger.info(f"\n📊 Model Comparison Summary:")
        logger.info(f"\n{comparison_df.to_string()}")

        return self.results

    def _create_comparison_summary(self) -> pd.DataFrame:
        """Create summary comparison of all models."""
        summary_data = []

        for model_type, result in self.results.items():
            summary_data.append({
                "model_type": model_type,
                "best_score": result["best_score"],
                "n_trials": len(result["trial_results"])
            })

        df = pd.DataFrame(summary_data)
        sort_ascending = self.scoring_config.direction == "minimize"
        df = df.sort_values("best_score", ascending=sort_ascending)

        return df

    def get_best_model(self) -> Tuple[str, Dict[str, Any], float]:
        """Get the best performing model overall.

        Returns:
            Tuple of (model_type, best_params, best_score)
        """
        if not self.results:
            raise ValueError("Must run optimization first")

        best_model_type = None
        best_score = float('-inf') if self.scoring_config.direction == "maximize" else float('inf')
        best_params = None

        for model_type, result in self.results.items():
            score = result["best_score"]

            if self.scoring_config.direction == "maximize":
                if score > best_score:
                    best_score = score
                    best_model_type = model_type
                    best_params = result["best_params"]
            else:
                if score < best_score:
                    best_score = score
                    best_model_type = model_type  
                    best_params = result["best_params"]

        logger.info(f"Best overall model: {best_model_type} with {self.scoring_config.scoring_function}={best_score:.4f}")

        return best_model_type, best_params, best_score


# Convenience functions
def tune_hyperparameters(model_type: str,
                        X: pd.DataFrame, 
                        y: pd.Series,
                        sample_weight: Optional[np.ndarray] = None,
                        scoring_function: str = "roc_auc",
                        additional_metrics: List[str] = None,
                        n_trials: int = 100,
                        **kwargs) -> Tuple[Dict[str, Any], float]:
    """Convenient function for hyperparameter tuning.

    Args:
        model_type: Type of model to tune
        X: Training features
        y: Training target
        sample_weight: Sample weights
        scoring_function: Primary scoring function
        additional_metrics: Additional metrics to log
        n_trials: Number of trials
        **kwargs: Additional arguments for tuner

    Returns:
        Tuple of (best_params, best_score)

    Example:
        >>> best_params, best_score = tune_hyperparameters(
        ...     'xgboost', X, y, sample_weight=weights,
        ...     scoring_function='recall',
        ...     additional_metrics=['precision', 'f1', 'roc_auc'],
        ...     n_trials=50
        ... )
    """
    additional_metrics = additional_metrics or ["accuracy", "precision", "recall", "f1"]

    scoring_config = ScoringConfig(
        scoring_function=scoring_function,
        additional_metrics=additional_metrics,
        run_full_evaluation=True
    )

    tuner = EnhancedOptunaHyperparameterTuner(
        model_type=model_type,
        n_trials=n_trials,
        scoring_config=scoring_config,
        **kwargs
    )

    return tuner.optimize(X, y, sample_weight)


def compare_models(X: pd.DataFrame,
                  y: pd.Series,
                  sample_weight: Optional[np.ndarray] = None,
                  model_types: List[str] = None,
                  scoring_function: str = "roc_auc",
                  additional_metrics: List[str] = None,
                  n_trials_per_model: int = 50) -> Dict[str, Any]:
    """Compare multiple models with comprehensive evaluation.

    Args:
        X: Training features
        y: Training target
        sample_weight: Sample weights
        model_types: List of models to compare
        scoring_function: Primary scoring function
        additional_metrics: Additional metrics to log
        n_trials_per_model: Trials per model

    Returns:
        Dictionary with comparison results
    """
    model_types = model_types or ["xgboost", "lightgbm", "catboost"]
    additional_metrics = additional_metrics or ["accuracy", "precision", "recall", "f1"]

    scoring_config = ScoringConfig(
        scoring_function=scoring_function,
        additional_metrics=additional_metrics,
        run_full_evaluation=True
    )

    multi_tuner = MultiModelTuner(
        model_types=model_types,
        n_trials_per_model=n_trials_per_model,
        scoring_config=scoring_config
    )

    results = multi_tuner.optimize_all(X, y, sample_weight)
    best_model, best_params, best_score = multi_tuner.get_best_model()

    return {
        "all_results": results,
        "best_model": best_model,
        "best_params": best_params,
        "best_score": best_score,
        "comparison_summary": multi_tuner._create_comparison_summary()
    }
