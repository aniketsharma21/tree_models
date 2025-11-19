# tree_models/config/model_config.py
"""Type-safe model configuration system with validation and presets.

This module provides comprehensive configuration classes for different
tree-based models with built-in validation, fraud detection presets,
and environment variable override support.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ConfigurationError, validate_parameter
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    """Supported model types."""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


class OptimizationObjective(Enum):
    """Common optimization objectives."""

    BINARY_CLASSIFICATION = "binary:logistic"
    MULTICLASS_CLASSIFICATION = "multi:softmax"
    REGRESSION = "reg:squarederror"
    RANKING = "rank:pairwise"


@dataclass
class BaseModelConfig:
    """Base configuration class for all tree-based models.

    Provides common parameters and validation logic that applies
    to all supported tree-based model types.
    """

    # Core training parameters
    model_type: str = "base"
    random_state: Optional[int] = 42
    n_jobs: int = -1
    verbose: int = 0

    # Early stopping
    early_stopping_rounds: Optional[int] = None

    # Validation parameters
    eval_metric: Optional[str] = None

    # Custom parameters (for extensibility)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.random_state is not None:
            validate_parameter("random_state", self.random_state, min_value=0)

        if self.n_jobs != -1:
            validate_parameter("n_jobs", self.n_jobs, min_value=1)

        if self.early_stopping_rounds is not None:
            validate_parameter("early_stopping_rounds", self.early_stopping_rounds, min_value=1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns:
            Dictionary representation of configuration
        """
        config_dict = {}

        # Add non-None values
        for field_name, field_value in self.__dict__.items():
            if field_value is not None and not field_name.startswith("_"):
                if field_name == "custom_params":
                    config_dict.update(field_value)
                else:
                    config_dict[field_name] = field_value

        return config_dict

    def get_params(self) -> Dict[str, Any]:
        """Backward-compatible alias for parameter retrieval used by trainers.

        Returns the configuration as a plain dict suitable for passing to
        model constructors or libraries.
        """
        return self.to_dict()

    def update_from_env(self, prefix: str = "") -> None:
        """Update configuration from environment variables.

        Args:
            prefix: Environment variable prefix (e.g., "XGBOOST_")
        """
        for field_name in self.__dataclass_fields__:
            env_name = f"{prefix}{field_name.upper()}"
            if env_name in os.environ:
                try:
                    env_value = os.environ[env_name]

                    # Convert string to appropriate type
                    field_type = self.__dataclass_fields__[field_name].type

                    if field_type == int or field_type == Optional[int]:
                        converted_value = int(env_value)
                    elif field_type == float or field_type == Optional[float]:
                        converted_value = float(env_value)
                    elif field_type == bool or field_type == Optional[bool]:
                        converted_value = env_value.lower() in ("true", "1", "yes", "on")
                    else:
                        converted_value = env_value

                    setattr(self, field_name, converted_value)
                    logger.info(f"Updated {field_name} from environment: {converted_value}")

                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse environment variable {env_name}: {e}")

    @classmethod
    def for_fraud_detection(cls) -> "BaseModelConfig":
        """Create configuration optimized for fraud detection.

        Returns:
            Configuration instance with fraud detection presets
        """
        # This will be overridden by subclasses
        return cls()


@dataclass
class XGBoostConfig(BaseModelConfig):
    """Type-safe configuration for XGBoost models.

    Provides comprehensive parameter validation and fraud detection
    presets optimized for tree-based gradient boosting.
    """

    # Model type
    model_type: str = "xgboost"

    # Boosting parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.3
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    colsample_bylevel: float = 1.0
    colsample_bynode: float = 1.0

    # Regularization
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    gamma: float = 0.0  # Minimum split loss
    min_child_weight: int = 1

    # Tree construction
    max_delta_step: float = 0.0
    grow_policy: str = "depthwise"  # "depthwise" or "lossguide"
    max_leaves: int = 0

    # Objective and evaluation
    objective: str = "binary:logistic"
    eval_metric: str = "auc"

    # Performance
    tree_method: str = "auto"  # "auto", "exact", "approx", "hist", "gpu_hist"

    def __post_init__(self) -> None:
        """Validate XGBoost-specific parameters."""
        super().__post_init__()

        # Validate core parameters
        validate_parameter("n_estimators", self.n_estimators, min_value=1, max_value=10000)
        validate_parameter("max_depth", self.max_depth, min_value=1, max_value=20)
        validate_parameter("learning_rate", self.learning_rate, min_value=0.001, max_value=1.0)

        # Validate sampling parameters
        validate_parameter("subsample", self.subsample, min_value=0.1, max_value=1.0)
        validate_parameter("colsample_bytree", self.colsample_bytree, min_value=0.1, max_value=1.0)
        validate_parameter("colsample_bylevel", self.colsample_bylevel, min_value=0.1, max_value=1.0)
        validate_parameter("colsample_bynode", self.colsample_bynode, min_value=0.1, max_value=1.0)

        # Validate regularization parameters
        validate_parameter("reg_alpha", self.reg_alpha, min_value=0.0)
        validate_parameter("reg_lambda", self.reg_lambda, min_value=0.0)
        validate_parameter("gamma", self.gamma, min_value=0.0)
        validate_parameter("min_child_weight", self.min_child_weight, min_value=0)

        # Validate categorical parameters
        validate_parameter("grow_policy", self.grow_policy, valid_values=["depthwise", "lossguide"])
        validate_parameter(
            "tree_method", self.tree_method, valid_values=["auto", "exact", "approx", "hist", "gpu_hist"]
        )

        # Validate objective
        valid_objectives = [
            "binary:logistic",
            "binary:logitraw",
            "binary:hinge",
            "multi:softmax",
            "multi:softprob",
            "reg:squarederror",
            "reg:squaredlogerror",
            "reg:logistic",
            "reg:pseudohubererror",
            "reg:absoluteerror",
            "reg:quantileerror",
            "rank:pairwise",
            "rank:ndcg",
            "rank:map",
        ]
        if self.objective not in valid_objectives:
            logger.warning(f"Uncommon objective: {self.objective}")

    @classmethod
    def for_fraud_detection(cls) -> "XGBoostConfig":
        """Create XGBoost configuration optimized for fraud detection.

        Returns:
            XGBoostConfig with fraud detection presets
        """
        return cls(
            # Increased complexity for fraud patterns
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,  # Lower for stability
            # Robust sampling for imbalanced data
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            # Higher regularization for generalization
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            min_child_weight=3,
            # Fraud detection specific
            objective="binary:logistic",
            eval_metric="auc",
            # Performance optimization
            tree_method="hist",
            grow_policy="lossguide",
            max_leaves=255,
        )

    @classmethod
    def for_high_cardinality(cls) -> "XGBoostConfig":
        """Create configuration for high cardinality features.

        Returns:
            XGBoostConfig optimized for datasets with many categorical features
        """
        return cls(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=2.0,
            tree_method="hist",
            grow_policy="lossguide",
            max_leaves=511,
        )


@dataclass
class LightGBMConfig(BaseModelConfig):
    """Type-safe configuration for LightGBM models.

    Provides LightGBM-specific parameter validation and optimization
    presets for different use cases.
    """

    # Model type
    model_type: str = "lightgbm"

    # Boosting parameters
    n_estimators: int = 100
    max_depth: int = -1  # -1 means no limit
    learning_rate: float = 0.1
    num_leaves: int = 31

    # Sampling parameters
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0

    # Regularization
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    min_child_samples: int = 20
    min_child_weight: float = 1e-3
    min_split_gain: float = 0.0

    # Categorical features
    cat_smooth: float = 10.0
    cat_l2: float = 10.0

    # Objective and metric
    objective: str = "binary"
    metric: str = "auc"
    boosting_type: str = "gbdt"  # "gbdt", "dart", "goss", "rf"

    # Performance
    device_type: str = "cpu"  # "cpu" or "gpu"

    def __post_init__(self) -> None:
        """Validate LightGBM-specific parameters."""
        super().__post_init__()

        # Validate core parameters
        validate_parameter("n_estimators", self.n_estimators, min_value=1, max_value=10000)
        if self.max_depth != -1:
            validate_parameter("max_depth", self.max_depth, min_value=1, max_value=20)
        validate_parameter("learning_rate", self.learning_rate, min_value=0.001, max_value=1.0)
        validate_parameter("num_leaves", self.num_leaves, min_value=2, max_value=1000)

        # Validate sampling parameters
        validate_parameter("feature_fraction", self.feature_fraction, min_value=0.1, max_value=1.0)
        validate_parameter("bagging_fraction", self.bagging_fraction, min_value=0.1, max_value=1.0)
        validate_parameter("bagging_freq", self.bagging_freq, min_value=0)

        # Validate regularization
        validate_parameter("reg_alpha", self.reg_alpha, min_value=0.0)
        validate_parameter("reg_lambda", self.reg_lambda, min_value=0.0)
        validate_parameter("min_child_samples", self.min_child_samples, min_value=1)
        validate_parameter("min_child_weight", self.min_child_weight, min_value=0.0)
        validate_parameter("min_split_gain", self.min_split_gain, min_value=0.0)

        # Validate categorical parameters
        validate_parameter("cat_smooth", self.cat_smooth, min_value=0.0)
        validate_parameter("cat_l2", self.cat_l2, min_value=0.0)

        # Validate boosting type
        validate_parameter("boosting_type", self.boosting_type, valid_values=["gbdt", "dart", "goss", "rf"])
        validate_parameter("device_type", self.device_type, valid_values=["cpu", "gpu"])

    @classmethod
    def for_fraud_detection(cls) -> "LightGBMConfig":
        """Create LightGBM configuration optimized for fraud detection."""
        return cls(
            # Increased complexity for fraud patterns
            n_estimators=750,
            max_depth=12,
            learning_rate=0.05,
            num_leaves=127,
            # Robust sampling
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            # Higher regularization
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=50,
            min_split_gain=0.01,
            # Fraud detection specific
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            # Categorical feature optimization
            cat_smooth=15.0,
            cat_l2=15.0,
        )

    @classmethod
    def for_fast_training(cls) -> "LightGBMConfig":
        """Create configuration for fast training with reasonable performance."""
        return cls(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=63,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            boosting_type="gbdt",
        )


@dataclass
class CatBoostConfig(BaseModelConfig):
    """Type-safe configuration for CatBoost models.

    Provides CatBoost-specific parameter validation and presets
    optimized for categorical feature handling.
    """

    # Model type
    model_type: str = "catboost"

    # Boosting parameters
    iterations: int = 1000  # CatBoost uses 'iterations' instead of 'n_estimators'
    depth: int = 6
    learning_rate: float = 0.03

    # Regularization
    l2_leaf_reg: float = 3.0
    bagging_temperature: float = 1.0

    # Sampling
    subsample: float = 1.0
    colsample_bylevel: float = 1.0

    # Categorical features
    one_hot_max_size: int = 2
    max_ctr_complexity: int = 4

    # Objective and metric
    loss_function: str = "Logloss"
    eval_metric: str = "AUC"

    # Performance
    task_type: str = "CPU"  # "CPU" or "GPU"
    thread_count: int = -1

    # Training behavior
    use_best_model: bool = True
    verbose: int = 0

    def __post_init__(self) -> None:
        """Validate CatBoost-specific parameters."""
        super().__post_init__()

        # Validate core parameters
        validate_parameter("iterations", self.iterations, min_value=1, max_value=100000)
        validate_parameter("depth", self.depth, min_value=1, max_value=16)
        validate_parameter("learning_rate", self.learning_rate, min_value=0.001, max_value=1.0)

        # Validate regularization
        validate_parameter("l2_leaf_reg", self.l2_leaf_reg, min_value=0.0)
        validate_parameter("bagging_temperature", self.bagging_temperature, min_value=0.0)

        # Validate sampling
        validate_parameter("subsample", self.subsample, min_value=0.1, max_value=1.0)
        validate_parameter("colsample_bylevel", self.colsample_bylevel, min_value=0.1, max_value=1.0)

        # Validate categorical parameters
        validate_parameter("one_hot_max_size", self.one_hot_max_size, min_value=0, max_value=255)
        validate_parameter("max_ctr_complexity", self.max_ctr_complexity, min_value=1, max_value=10)

        # Validate task type
        validate_parameter("task_type", self.task_type, valid_values=["CPU", "GPU"])

    @classmethod
    def for_fraud_detection(cls) -> "CatBoostConfig":
        """Create CatBoost configuration optimized for fraud detection."""
        return cls(
            # Increased complexity
            iterations=2000,
            depth=10,
            learning_rate=0.02,
            # Higher regularization
            l2_leaf_reg=5.0,
            bagging_temperature=0.5,
            # Robust sampling
            subsample=0.8,
            colsample_bylevel=0.8,
            # Categorical optimization
            one_hot_max_size=10,
            max_ctr_complexity=6,
            # Fraud detection specific
            loss_function="Logloss",
            eval_metric="AUC",
            use_best_model=True,
        )

    @classmethod
    def for_categorical_heavy(cls) -> "CatBoostConfig":
        """Create configuration for datasets with many categorical features."""
        return cls(
            iterations=1500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=2.0,
            one_hot_max_size=20,
            max_ctr_complexity=8,
            use_best_model=True,
        )


# Model configuration factory
class ModelConfigFactory:
    """Factory for creating model configurations based on use case."""

    _config_classes = {
        ModelType.XGBOOST: XGBoostConfig,
        ModelType.LIGHTGBM: LightGBMConfig,
        ModelType.CATBOOST: CatBoostConfig,
    }

    @classmethod
    def create_config(
        cls, model_type: Union[str, ModelType], use_case: str = "default", **kwargs: Any
    ) -> BaseModelConfig:
        """Create model configuration for specific use case.

        Args:
            model_type: Type of model
            use_case: Use case preset ("default", "fraud_detection", "fast_training", etc.)
            **kwargs: Additional configuration parameters

        Returns:
            Configured model config instance

        Raises:
            ConfigurationError: If model type or use case is invalid
        """
        # Convert string to enum if needed
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError:
                raise ConfigurationError(f"Unknown model type: {model_type}")

        # Get configuration class
        if model_type not in cls._config_classes:
            raise ConfigurationError(f"No configuration class for model type: {model_type}")

        config_class = cls._config_classes[model_type]

        # Create configuration based on use case
        if use_case == "fraud_detection":
            config = config_class.for_fraud_detection()
        elif use_case == "fast_training" and hasattr(config_class, "for_fast_training"):
            config = config_class.for_fast_training()
        elif use_case == "categorical_heavy" and hasattr(config_class, "for_categorical_heavy"):
            config = config_class.for_categorical_heavy()
        elif use_case == "high_cardinality" and hasattr(config_class, "for_high_cardinality"):
            config = config_class.for_high_cardinality()
        else:
            config = config_class()

        # Apply any additional parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown parameter for {model_type}: {key}")

        return config

    @classmethod
    def get_available_presets(cls, model_type: Union[str, ModelType]) -> List[str]:
        """Get available configuration presets for model type.

        Args:
            model_type: Type of model

        Returns:
            List of available preset names
        """
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError:
                return []

        if model_type not in cls._config_classes:
            return []

        config_class = cls._config_classes[model_type]
        presets = ["default"]

        # Check for available class methods
        if hasattr(config_class, "for_fraud_detection"):
            presets.append("fraud_detection")
        if hasattr(config_class, "for_fast_training"):
            presets.append("fast_training")
        if hasattr(config_class, "for_categorical_heavy"):
            presets.append("categorical_heavy")
        if hasattr(config_class, "for_high_cardinality"):
            presets.append("high_cardinality")

        return presets


# Convenience function
def create_model_config(model_type: str, use_case: str = "default", **kwargs: Any) -> BaseModelConfig:
    """Convenience function to create model configuration.

    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
        use_case: Use case preset
        **kwargs: Additional parameters

    Returns:
        Configured model config instance

    Example:
        >>> config = create_model_config('xgboost', 'fraud_detection')
        >>> config.n_estimators = 1000
        >>> model_params = config.to_dict()
    """
    return ModelConfigFactory.create_config(model_type, use_case, **kwargs)


# Type aliases for convenience
ModelConfig = BaseModelConfig
XGBConfig = XGBoostConfig
LGBMConfig = LightGBMConfig
CBConfig = CatBoostConfig


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment."""

    model: BaseModelConfig
    data: Any
    experiment_name: str = "default_experiment"
    description: str = ""
    random_state: int = 42


# Default configurations
DEFAULT_CONFIGS = {"xgboost": XGBoostConfig(), "lightgbm": LightGBMConfig(), "catboost": CatBoostConfig()}
