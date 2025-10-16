"""Base configuration with type definitions and Python defaults.

This module provides type-safe configuration classes with default values
that can be overridden by YAML files or environment variables.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import os


@dataclass
class ModelConfig:
    """Base model configuration with type safety."""

    # Model type
    model_type: str = "xgboost"

    # Core hyperparameters
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42

    # Training parameters
    early_stopping_rounds: int = 50
    verbose: bool = False

    # Model-specific parameters (will be populated by subclasses)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model training."""
        base_dict = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state
        }
        base_dict.update(self.custom_params)
        return base_dict


@dataclass
class XGBoostConfig(ModelConfig):
    """XGBoost-specific configuration."""

    model_type: str = "xgboost"

    # XGBoost specific parameters
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    objective: str = "binary:logistic"
    eval_metric: str = "auc"

    def __post_init__(self):
        """Populate custom_params with XGBoost-specific values."""
        self.custom_params = {
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'objective': self.objective,
            'eval_metric': self.eval_metric
        }


@dataclass
class LightGBMConfig(ModelConfig):
    """LightGBM-specific configuration."""

    model_type: str = "lightgbm"

    # LightGBM specific parameters
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    objective: str = "binary"
    metric: str = "auc"
    boosting_type: str = "gbdt"

    def __post_init__(self):
        """Populate custom_params with LightGBM-specific values."""
        self.custom_params = {
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'min_child_samples': self.min_child_samples,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'objective': self.objective,
            'metric': self.metric,
            'boosting_type': self.boosting_type
        }


@dataclass 
class CatBoostConfig(ModelConfig):
    """CatBoost-specific configuration."""

    model_type: str = "catboost"

    # CatBoost specific parameters (using 'depth' instead of 'max_depth')
    depth: int = 6
    l2_leaf_reg: float = 3.0
    subsample: float = 0.8
    colsample_bylevel: float = 1.0
    loss_function: str = "Logloss"
    eval_metric: str = "AUC"
    verbose: bool = False

    def __post_init__(self):
        """Populate custom_params with CatBoost-specific values."""
        # CatBoost uses 'iterations' instead of 'n_estimators'
        # and 'depth' instead of 'max_depth'
        self.custom_params = {
            'iterations': self.n_estimators,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'l2_leaf_reg': self.l2_leaf_reg,
            'subsample': self.subsample,
            'colsample_bylevel': self.colsample_bylevel,
            'loss_function': self.loss_function,
            'eval_metric': self.eval_metric,
            'verbose': self.verbose,
            'random_seed': self.random_state
        }


@dataclass
class DataConfig:
    """Data processing configuration."""

    # Data paths
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    output_dir: str = "output"

    # Column names
    target_col: str = "target"
    weight_col: Optional[str] = None
    id_col: Optional[str] = None

    # Data splitting
    test_size: float = 0.2
    valid_size: float = 0.2
    stratify: bool = True

    # Sampling
    sample_size: Optional[int] = None
    random_state: int = 42

    # Missing value handling
    missing_strategy: str = "median"
    categorical_strategy: str = "most_frequent"

    # Feature engineering
    encoding_strategy: str = "label"  # "label", "onehot", "target"
    scaling_strategy: Optional[str] = None  # None, "standard", "minmax", "robust"
    use_knn_imputation: bool = False


@dataclass
class FeatureSelectionConfig:
    """Feature selection configuration."""

    # General settings
    enable_feature_selection: bool = True
    save_results: bool = True

    # Variance filtering
    variance_threshold: float = 0.01

    # RFECV settings
    rfecv_enabled: bool = True
    rfecv_cv_folds: int = 5
    rfecv_scoring: str = "roc_auc"
    rfecv_step: int = 1
    rfecv_min_features: int = 1

    # Boruta settings
    boruta_enabled: bool = True
    boruta_max_iter: int = 100
    boruta_alpha: float = 0.05
    boruta_two_step: bool = True

    # Consensus settings
    min_agreement: int = 2


@dataclass
class TuningConfig:
    """Hyperparameter tuning configuration."""

    # General settings
    enable_tuning: bool = True
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds

    # Cross-validation
    cv_folds: int = 5
    scoring: str = "roc_auc"

    # Optuna settings
    sampler: str = "tpe"  # "tpe", "random", "grid"
    pruner: str = "median"  # "median", "successive_halving", "hyperband"
    n_startup_trials: int = 10
    n_warmup_steps: int = 10

    # Parallelization
    n_jobs: int = 1

    # MLflow integration
    log_to_mlflow: bool = True

    # Custom search space (will be populated from YAML)
    custom_search_space: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Model evaluation configuration."""

    # General settings
    generate_plots: bool = True
    save_results: bool = True
    plot_format: str = "png"  # "png", "pdf", "svg"
    plot_dpi: int = 300

    # Metrics
    primary_metric: str = "auc_roc"
    threshold: float = 0.5

    # Threshold optimization
    optimize_threshold: bool = True
    threshold_metrics: List[str] = field(default_factory=lambda: ["f1", "precision", "recall", "youden"])

    # Gains analysis
    gains_bins: int = 10

    # Cross-validation for evaluation
    eval_cv_folds: int = 5


@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration."""

    # MLflow settings
    tracking_uri: Optional[str] = None
    experiment_name: str = "tree_model_experiment"
    run_name: Optional[str] = None

    # Logging settings
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    log_model: bool = True
    log_plots: bool = True

    # Auto-logging
    auto_log_system_info: bool = True
    log_git_commit: bool = True
    log_environment: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all components."""

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # Global settings
    seed: int = 42
    verbose: bool = True
    debug: bool = False

    # Execution settings
    parallel_jobs: int = 1
    memory_limit: Optional[str] = None  # e.g., "4GB"

    def get_model_config(self, model_type: str) -> ModelConfig:
        """Get appropriate model config based on model type."""
        if model_type.lower() == "xgboost":
            return XGBoostConfig()
        elif model_type.lower() == "lightgbm":
            return LightGBMConfig()
        elif model_type.lower() == "catboost":
            return CatBoostConfig()
        else:
            return ModelConfig(model_type=model_type)


# Environment-based configuration
class EnvironmentConfig:
    """Environment-specific configuration management."""

    @staticmethod
    def get_env_var(key: str, default: Any = None) -> Any:
        """Get environment variable with type conversion."""
        value = os.getenv(key, default)

        # Convert string representations to appropriate types
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            try:
                # Try int first
                if '.' not in value:
                    return int(value)
                else:
                    return float(value)
            except ValueError:
                pass

        return value

    @classmethod
    def from_environment(cls) -> Dict[str, Any]:
        """Extract configuration from environment variables."""
        return {
            # Data configuration
            'data': {
                'train_path': cls.get_env_var('TRAIN_PATH'),
                'test_path': cls.get_env_var('TEST_PATH'),
                'output_dir': cls.get_env_var('OUTPUT_DIR', 'output'),
                'target_col': cls.get_env_var('TARGET_COL', 'target'),
                'weight_col': cls.get_env_var('WEIGHT_COL'),
            },

            # Model configuration
            'model': {
                'model_type': cls.get_env_var('MODEL_TYPE', 'xgboost'),
                'n_estimators': cls.get_env_var('N_ESTIMATORS', 200),
                'max_depth': cls.get_env_var('MAX_DEPTH', 6),
                'learning_rate': cls.get_env_var('LEARNING_RATE', 0.1),
            },

            # Tuning configuration
            'tuning': {
                'enable_tuning': cls.get_env_var('ENABLE_TUNING', True),
                'n_trials': cls.get_env_var('N_TRIALS', 100),
                'cv_folds': cls.get_env_var('CV_FOLDS', 5),
            },

            # MLflow configuration
            'mlflow': {
                'tracking_uri': cls.get_env_var('MLFLOW_TRACKING_URI'),
                'experiment_name': cls.get_env_var('MLFLOW_EXPERIMENT', 'tree_model_experiment'),
                'run_name': cls.get_env_var('MLFLOW_RUN_NAME'),
            },

            # Global settings
            'seed': cls.get_env_var('RANDOM_SEED', 42),
            'verbose': cls.get_env_var('VERBOSE', True),
            'debug': cls.get_env_var('DEBUG', False),
        }


# Default configurations for easy access
DEFAULT_XGBOOST_CONFIG = XGBoostConfig()
DEFAULT_LIGHTGBM_CONFIG = LightGBMConfig()
DEFAULT_CATBOOST_CONFIG = CatBoostConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()

# For backward compatibility with existing code
SEED = DEFAULT_EXPERIMENT_CONFIG.seed
TEST_SIZE = DEFAULT_EXPERIMENT_CONFIG.data.test_size
VALID_SIZE = DEFAULT_EXPERIMENT_CONFIG.data.valid_size

# Legacy parameter dictionaries for backward compatibility
XGB_DEFAULT_PARAMS = DEFAULT_XGBOOST_CONFIG.to_dict()
LGBM_DEFAULT_PARAMS = DEFAULT_LIGHTGBM_CONFIG.to_dict()
CATBOOST_DEFAULT_PARAMS = DEFAULT_CATBOOST_CONFIG.to_dict()

# Optuna search spaces
XGB_OPTUNA_SPACE = {
    "max_depth": (3, 12),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (0.0, 10.0),
    "n_estimators": (100, 500)
}

LGBM_OPTUNA_SPACE = {
    "max_depth": (3, 12),
    "learning_rate": (0.01, 0.3),
    "feature_fraction": (0.6, 1.0),
    "bagging_fraction": (0.6, 1.0),
    "min_child_samples": (5, 100),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (0.0, 10.0),
    "n_estimators": (100, 500)
}

CATBOOST_OPTUNA_SPACE = {
    "depth": (3, 12),
    "learning_rate": (0.01, 0.3),
    "l2_leaf_reg": (1, 10),
    "subsample": (0.6, 1.0),
    "colsample_bylevel": (0.6, 1.0),
    "iterations": (100, 500)
}
