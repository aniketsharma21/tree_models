# tree_models/config/data_config.py
"""Enhanced data configuration classes with comprehensive settings.

This module provides type-safe configuration classes for data handling with:
- Type-safe data loading and processing configurations
- Feature engineering and selection configuration management
- Preprocessing pipeline configuration with validation
- Training/validation/test split configuration with stratification
- Sample weights and imbalanced data handling configurations
- Advanced data validation and quality assurance settings
- Cross-validation and time series split configurations
- Integration with data validation and preprocessing modules
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..utils.exceptions import ConfigurationError, validate_parameter
from ..utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


@dataclass
class DataSourceConfig:
    """Configuration for data sources and loading parameters."""

    # File paths
    train_path: Optional[Union[str, Path]] = None
    test_path: Optional[Union[str, Path]] = None
    validation_path: Optional[Union[str, Path]] = None

    # Data format settings
    file_format: str = "csv"  # "csv", "parquet", "json", "excel"
    encoding: str = "utf-8"
    separator: str = ","

    # Loading parameters
    header: Union[int, str, None] = "infer"
    index_col: Optional[Union[int, str, List[Union[int, str]]]] = None
    usecols: Optional[List[Union[int, str]]] = None

    # Memory optimization
    low_memory: bool = False
    chunksize: Optional[int] = None
    nrows: Optional[int] = None

    # Data types
    dtype: Optional[Dict[str, str]] = None
    parse_dates: Optional[List[str]] = None
    date_parser: Optional[Callable] = None

    def __post_init__(self) -> None:
        """Validate data source configuration."""
        valid_formats = ["csv", "parquet", "json", "excel", "hdf", "feather"]
        validate_parameter("file_format", self.file_format, valid_values=valid_formats)

        if self.chunksize is not None:
            validate_parameter("chunksize", self.chunksize, min_value=1, max_value=1000000)

        if self.nrows is not None:
            validate_parameter("nrows", self.nrows, min_value=1)


@dataclass
class DataSplitConfig:
    """Configuration for train/validation/test data splitting."""

    # Split ratios
    test_size: float = 0.2
    validation_size: float = 0.2  # From remaining data after test split

    # Splitting strategy
    stratify: bool = True
    shuffle: bool = True
    random_state: int = 42

    # Time series splitting
    time_series_split: bool = False
    time_column: Optional[str] = None
    time_split_method: str = "chronological"  # "chronological", "gap", "expanding"

    # Cross-validation settings
    cv_folds: int = 5
    cv_strategy: str = "kfold"  # "kfold", "stratified", "group", "time_series"
    group_column: Optional[str] = None

    # Advanced splitting
    ensure_minimum_samples: bool = True
    minimum_samples_per_class: int = 10

    def __post_init__(self) -> None:
        """Validate data split configuration."""
        validate_parameter("test_size", self.test_size, min_value=0.0, max_value=0.8)
        validate_parameter("validation_size", self.validation_size, min_value=0.0, max_value=0.8)

        total_split = self.test_size + self.validation_size
        if total_split >= 1.0:
            raise ConfigurationError("Combined test_size and validation_size must be < 1.0")

        validate_parameter("cv_folds", self.cv_folds, min_value=2, max_value=20)
        validate_parameter("minimum_samples_per_class", self.minimum_samples_per_class, min_value=1)

        valid_cv_strategies = ["kfold", "stratified", "group", "time_series"]
        validate_parameter("cv_strategy", self.cv_strategy, valid_values=valid_cv_strategies)

        valid_time_methods = ["chronological", "gap", "expanding"]
        validate_parameter("time_split_method", self.time_split_method, valid_values=valid_time_methods)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering and selection."""

    # Target column
    target_column: str = "target"

    # Feature selection
    feature_columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None

    # Feature types
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    text_features: Optional[List[str]] = None
    date_features: Optional[List[str]] = None

    # Feature engineering
    create_polynomial_features: bool = False
    polynomial_degree: int = 2
    include_interaction_terms: bool = False

    # Categorical encoding preferences
    high_cardinality_threshold: int = 50
    rare_category_threshold: float = 0.01

    # Feature selection parameters
    enable_feature_selection: bool = False
    selection_method: str = "variance"  # "variance", "correlation", "mutual_info", "rfe"
    n_features_to_select: Optional[int] = None
    feature_selection_cv: int = 5

    def __post_init__(self) -> None:
        """Validate feature configuration."""
        if not self.target_column:
            raise ConfigurationError("target_column cannot be empty")

        validate_parameter("polynomial_degree", self.polynomial_degree, min_value=1, max_value=5)
        validate_parameter("high_cardinality_threshold", self.high_cardinality_threshold, min_value=2)
        validate_parameter("rare_category_threshold", self.rare_category_threshold, min_value=0.0, max_value=0.5)

        valid_selection_methods = ["variance", "correlation", "mutual_info", "rfe", "boruta"]
        validate_parameter("selection_method", self.selection_method, valid_values=valid_selection_methods)

        if self.n_features_to_select is not None:
            validate_parameter("n_features_to_select", self.n_features_to_select, min_value=1)


@dataclass
class FeatureEngineeringConfig:
    """Type-safe configuration for feature engineering operations."""

    # Mathematical transformations
    log_transform_cols: Optional[List[str]] = None
    sqrt_transform_cols: Optional[List[str]] = None
    box_cox_transform_cols: Optional[List[str]] = None

    # Polynomial and interaction features
    create_polynomial_features: bool = False
    polynomial_degree: int = 2
    polynomial_include_bias: bool = False
    create_interaction_features: bool = False
    interaction_pairs: Optional[List[Tuple[str, str]]] = None

    # Binning and discretization
    create_bins: Optional[Dict[str, Union[int, List[float]]]] = None
    quantile_bins: Optional[Dict[str, int]] = None

    # Ratio and difference features
    create_ratios: Optional[List[Tuple[str, str]]] = None
    create_differences: Optional[List[Tuple[str, str]]] = None

    # Aggregation features
    groupby_aggregations: Optional[Dict[str, Dict[str, List[str]]]] = None
    rolling_windows: Optional[Dict[str, Dict[str, Union[int, str]]]] = None

    # Text processing
    text_vectorization: Optional[Dict[str, str]] = None  # column -> method
    max_text_features: int = 1000
    text_ngram_range: Tuple[int, int] = (1, 2)

    # Time series features
    extract_date_features: Optional[List[str]] = None
    create_lag_features: Optional[Dict[str, List[int]]] = None

    # Feature validation
    validate_features: bool = True
    remove_correlated_features: bool = False
    correlation_threshold: float = 0.95

    # Performance settings
    memory_efficient: bool = False
    chunk_size: int = 10000

    def __post_init__(self) -> None:
        """Validate feature engineering configuration."""
        validate_parameter("polynomial_degree", self.polynomial_degree, min_value=1, max_value=5)
        validate_parameter("max_text_features", self.max_text_features, min_value=10, max_value=100000)
        validate_parameter("correlation_threshold", self.correlation_threshold, min_value=0.0, max_value=1.0)
        validate_parameter("chunk_size", self.chunk_size, min_value=100, max_value=100000)

        # Validate n-gram range
        if len(self.text_ngram_range) != 2 or self.text_ngram_range[0] > self.text_ngram_range[1]:
            raise ConfigurationError("Invalid text_ngram_range")


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing pipeline."""

    # Missing value handling
    missing_value_strategy: Dict[str, str] = field(
        default_factory=lambda: {"numeric": "median", "categorical": "most_frequent"}
    )

    # Scaling and normalization
    scaling_strategy: Optional[str] = "standard"  # "standard", "minmax", "robust", None
    normalize_features: bool = False

    # Categorical encoding
    categorical_encoding: Dict[str, str] = field(
        default_factory=lambda: {"low_cardinality": "onehot", "high_cardinality": "target"}
    )

    # Outlier handling
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation"
    outlier_threshold: float = 3.0
    outlier_action: str = "clip"  # "clip", "remove", "flag"

    # Feature transformation
    log_transform_features: Optional[List[str]] = None
    sqrt_transform_features: Optional[List[str]] = None

    # Data validation
    enable_validation: bool = True
    validation_rules: Optional[Dict[str, Any]] = None

    # Processing options
    handle_unknown_categories: str = "ignore"  # "ignore", "error", "encode"
    preserve_dtypes: bool = True

    def __post_init__(self) -> None:
        """Validate preprocessing configuration."""
        valid_missing_strategies = ["mean", "median", "mode", "most_frequent", "constant", "knn", "drop"]
        for data_type, strategy in self.missing_value_strategy.items():
            validate_parameter(f"missing_value_strategy[{data_type}]", strategy, valid_values=valid_missing_strategies)

        valid_scaling = [None, "standard", "minmax", "robust", "quantile"]
        validate_parameter("scaling_strategy", self.scaling_strategy, valid_values=valid_scaling)

        valid_encoding = ["onehot", "label", "target", "woe", "ordinal"]
        for cardinality, encoding in self.categorical_encoding.items():
            validate_parameter(f"categorical_encoding[{cardinality}]", encoding, valid_values=valid_encoding)

        valid_outlier_methods = ["iqr", "zscore", "isolation", "elliptic"]
        validate_parameter("outlier_method", self.outlier_method, valid_values=valid_outlier_methods)

        validate_parameter("outlier_threshold", self.outlier_threshold, min_value=0.5, max_value=10.0)

        valid_outlier_actions = ["clip", "remove", "flag"]
        validate_parameter("outlier_action", self.outlier_action, valid_values=valid_outlier_actions)

        valid_unknown_handling = ["ignore", "error", "encode"]
        validate_parameter(
            "handle_unknown_categories", self.handle_unknown_categories, valid_values=valid_unknown_handling
        )


@dataclass
class SampleWeightConfig:
    """Configuration for sample weights and class balancing."""

    # Sample weights
    use_sample_weights: bool = False
    weight_column: Optional[str] = None

    # Class balancing
    balance_classes: bool = False
    balancing_strategy: str = "auto"  # "auto", "balanced", "balanced_subsample", "custom"
    class_weights: Optional[Dict[Any, float]] = None

    # Imbalanced data handling
    handle_imbalanced_data: bool = False
    imbalance_method: str = "smote"  # "smote", "adasyn", "random_oversample", "random_undersample"
    sampling_ratio: Union[str, float] = "auto"

    # Weight computation
    weight_computation_method: str = "inverse_frequency"  # "inverse_frequency", "log_inverse", "custom"
    weight_smoothing: float = 0.0

    # Validation
    validate_weights: bool = True
    normalize_weights: bool = True

    def __post_init__(self) -> None:
        """Validate sample weight configuration."""
        valid_balancing_strategies = ["auto", "balanced", "balanced_subsample", "custom"]
        validate_parameter("balancing_strategy", self.balancing_strategy, valid_values=valid_balancing_strategies)

        valid_imbalance_methods = ["smote", "adasyn", "random_oversample", "random_undersample", "tomek", "enn"]
        validate_parameter("imbalance_method", self.imbalance_method, valid_values=valid_imbalance_methods)

        valid_weight_methods = ["inverse_frequency", "log_inverse", "custom"]
        validate_parameter(
            "weight_computation_method", self.weight_computation_method, valid_values=valid_weight_methods
        )

        validate_parameter("weight_smoothing", self.weight_smoothing, min_value=0.0, max_value=10.0)

        if isinstance(self.sampling_ratio, float):
            validate_parameter("sampling_ratio", self.sampling_ratio, min_value=0.1, max_value=10.0)


@dataclass
class ValidationConfig:
    """Configuration for data validation and quality checks."""

    # Basic validation
    check_missing_values: bool = True
    check_duplicates: bool = True
    check_outliers: bool = True
    check_data_types: bool = True

    # Advanced validation
    check_distributions: bool = False
    check_correlations: bool = False
    check_data_drift: bool = False

    # Thresholds
    missing_threshold: float = 0.5  # Flag if >50% missing
    correlation_threshold: float = 0.95
    outlier_threshold: float = 3.0

    # Validation rules
    custom_validation_rules: Optional[Dict[str, Callable]] = None

    # Reporting
    generate_report: bool = True
    report_format: str = "json"  # "json", "html", "pdf"
    include_plots: bool = False

    def __post_init__(self) -> None:
        """Validate validation configuration."""
        validate_parameter("missing_threshold", self.missing_threshold, min_value=0.0, max_value=1.0)
        validate_parameter("correlation_threshold", self.correlation_threshold, min_value=0.0, max_value=1.0)
        validate_parameter("outlier_threshold", self.outlier_threshold, min_value=0.5, max_value=10.0)

        valid_report_formats = ["json", "html", "pdf"]
        validate_parameter("report_format", self.report_format, valid_values=valid_report_formats)


@dataclass
class DataConfig:
    """Comprehensive data configuration combining all data-related settings.

    This is the main configuration class that combines all data-related
    configurations for a complete ML pipeline setup.

    Example:
        >>> data_config = DataConfig(
        ...     source=DataSourceConfig(
        ...         train_path="data/train.csv",
        ...         test_path="data/test.csv"
        ...     ),
        ...     features=FeatureConfig(
        ...         target_column="target",
        ...         enable_feature_selection=True
        ...     ),
        ...     preprocessing=PreprocessingConfig(
        ...         scaling_strategy="standard",
        ...         outlier_detection=True
        ...     )
        ... )
    """

    # Core configurations
    source: DataSourceConfig = field(default_factory=DataSourceConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    splitting: DataSplitConfig = field(default_factory=DataSplitConfig)

    # Optional configurations
    sample_weights: Optional[SampleWeightConfig] = None
    validation: Optional[ValidationConfig] = None

    # Output settings
    output_dir: Union[str, Path] = "data_output"
    save_processed_data: bool = True
    data_version: str = "1.0"

    # Metadata
    description: str = ""
    created_by: str = ""
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and initialize data configuration."""

        # Set default timestamp
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

        # Initialize optional configurations if not provided
        if self.sample_weights is None:
            self.sample_weights = SampleWeightConfig()

        if self.validation is None:
            self.validation = ValidationConfig()

        # Validate output directory
        self.output_dir = Path(self.output_dir)

        # Cross-validate configurations
        self._validate_configuration_consistency()

    def _validate_configuration_consistency(self) -> None:
        """Validate consistency across different configuration sections."""

        # Check if target column is not in excluded columns
        if self.features.exclude_columns and self.features.target_column in self.features.exclude_columns:
            raise ConfigurationError("Target column cannot be in excluded columns")

        # Check sample weights column consistency
        if (
            self.sample_weights.use_sample_weights
            and self.sample_weights.weight_column
            and self.features.exclude_columns
            and self.sample_weights.weight_column in self.features.exclude_columns
        ):
            raise ConfigurationError("Sample weight column cannot be in excluded columns")

        # Validate time series configuration consistency
        if self.splitting.time_series_split and not self.splitting.time_column:
            raise ConfigurationError("time_column must be specified for time series splitting")

        # Check feature selection consistency
        if (
            self.features.enable_feature_selection
            and self.features.n_features_to_select
            and self.features.feature_columns
            and self.features.n_features_to_select > len(self.features.feature_columns)
        ):
            raise ConfigurationError("n_features_to_select cannot exceed available features")

    def get_feature_columns(self) -> Optional[List[str]]:
        """Get list of feature columns to use."""
        return self.features.feature_columns

    def get_target_column(self) -> str:
        """Get target column name."""
        return self.features.target_column

    def should_validate_data(self) -> bool:
        """Check if data validation should be performed."""
        return self.validation is not None and any(
            [
                self.validation.check_missing_values,
                self.validation.check_duplicates,
                self.validation.check_outliers,
                self.validation.check_data_types,
            ]
        )

    def should_use_sample_weights(self) -> bool:
        """Check if sample weights should be used."""
        return self.sample_weights is not None and self.sample_weights.use_sample_weights

    def should_balance_classes(self) -> bool:
        """Check if class balancing should be performed."""
        return self.sample_weights is not None and self.sample_weights.balance_classes

    def get_preprocessing_steps(self) -> List[str]:
        """Get list of preprocessing steps to perform."""
        steps = []

        if self.preprocessing.outlier_detection:
            steps.append("outlier_detection")

        if self.preprocessing.missing_value_strategy:
            steps.append("missing_value_imputation")

        if self.preprocessing.scaling_strategy:
            steps.append("feature_scaling")

        if self.preprocessing.categorical_encoding:
            steps.append("categorical_encoding")

        if self.preprocessing.log_transform_features:
            steps.append("log_transformation")

        if self.features.enable_feature_selection:
            steps.append("feature_selection")

        return steps

    def get_split_strategy(self) -> Dict[str, Any]:
        """Get data splitting strategy configuration."""
        return {
            "test_size": self.splitting.test_size,
            "validation_size": self.splitting.validation_size,
            "stratify": self.splitting.stratify,
            "time_series": self.splitting.time_series_split,
            "cv_strategy": self.splitting.cv_strategy,
            "cv_folds": self.splitting.cv_folds,
            "random_state": self.splitting.random_state,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        """Create DataConfig from dictionary."""

        # Extract nested configurations
        source_config = DataSourceConfig(**config_dict.get("source", {}))
        features_config = FeatureConfig(**config_dict.get("features", {}))
        preprocessing_config = PreprocessingConfig(**config_dict.get("preprocessing", {}))
        splitting_config = DataSplitConfig(**config_dict.get("splitting", {}))

        # Optional configurations
        sample_weights_config = None
        if "sample_weights" in config_dict and config_dict["sample_weights"]:
            sample_weights_config = SampleWeightConfig(**config_dict["sample_weights"])

        validation_config = None
        if "validation" in config_dict and config_dict["validation"]:
            validation_config = ValidationConfig(**config_dict["validation"])

        # Create main config
        main_config = {
            k: v
            for k, v in config_dict.items()
            if k not in ["source", "features", "preprocessing", "splitting", "sample_weights", "validation"]
        }

        return cls(
            source=source_config,
            features=features_config,
            preprocessing=preprocessing_config,
            splitting=splitting_config,
            sample_weights=sample_weights_config,
            validation=validation_config,
            **main_config,
        )

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        import json

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        # Handle Path objects for JSON serialization
        def path_converter(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object {obj} is not JSON serializable")

        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=path_converter)

        logger.info(f"Data configuration saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "DataConfig":
        """Load configuration from file."""
        import json

        file_path = Path(file_path)

        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        with open(file_path) as f:
            config_dict = json.load(f)

        logger.info(f"Data configuration loaded from {file_path}")
        return cls.from_dict(config_dict)


# Utility functions for creating common configurations


def create_basic_data_config(
    train_path: Union[str, Path], target_column: str, test_path: Optional[Union[str, Path]] = None, **kwargs: Any
) -> DataConfig:
    """Create a basic data configuration for simple use cases.

    Args:
        train_path: Path to training data
        target_column: Name of target column
        test_path: Optional path to test data
        **kwargs: Additional configuration parameters

    Returns:
        DataConfig object with basic settings

    Example:
        >>> config = create_basic_data_config(
        ...     train_path="data/train.csv",
        ...     target_column="target",
        ...     test_path="data/test.csv"
        ... )
    """
    source_config = DataSourceConfig(train_path=train_path, test_path=test_path)

    features_config = FeatureConfig(target_column=target_column)

    return DataConfig(source=source_config, features=features_config, **kwargs)


def create_advanced_data_config(
    train_path: Union[str, Path],
    target_column: str,
    enable_feature_selection: bool = True,
    enable_validation: bool = True,
    balance_classes: bool = False,
    **kwargs: Any,
) -> DataConfig:
    """Create an advanced data configuration with comprehensive settings.

    Args:
        train_path: Path to training data
        target_column: Name of target column
        enable_feature_selection: Whether to enable feature selection
        enable_validation: Whether to enable data validation
        balance_classes: Whether to balance classes
        **kwargs: Additional configuration parameters

    Returns:
        DataConfig object with advanced settings

    Example:
        >>> config = create_advanced_data_config(
        ...     train_path="data/train.csv",
        ...     target_column="target",
        ...     enable_feature_selection=True,
        ...     enable_validation=True,
        ...     balance_classes=True
        ... )
    """
    source_config = DataSourceConfig(
        train_path=train_path, **{k: v for k, v in kwargs.items() if k.startswith("source_")}
    )

    features_config = FeatureConfig(
        target_column=target_column,
        enable_feature_selection=enable_feature_selection,
        **{k: v for k, v in kwargs.items() if k.startswith("features_")},
    )

    preprocessing_config = PreprocessingConfig(
        enable_validation=enable_validation, **{k: v for k, v in kwargs.items() if k.startswith("preprocessing_")}
    )

    sample_weights_config = SampleWeightConfig(
        balance_classes=balance_classes, **{k: v for k, v in kwargs.items() if k.startswith("weights_")}
    )

    validation_config = ValidationConfig(
        generate_report=enable_validation, **{k: v for k, v in kwargs.items() if k.startswith("validation_")}
    )

    # Filter out prefixed kwargs
    main_kwargs = {
        k: v
        for k, v in kwargs.items()
        if not any(
            k.startswith(prefix) for prefix in ["source_", "features_", "preprocessing_", "weights_", "validation_"]
        )
    }

    return DataConfig(
        source=source_config,
        features=features_config,
        preprocessing=preprocessing_config,
        sample_weights=sample_weights_config,
        validation=validation_config,
        **main_kwargs,
    )


def create_time_series_data_config(
    train_path: Union[str, Path], target_column: str, time_column: str, **kwargs: Any
) -> DataConfig:
    """Create a data configuration for time series data.

    Args:
        train_path: Path to training data
        target_column: Name of target column
        time_column: Name of time/date column
        **kwargs: Additional configuration parameters

    Returns:
        DataConfig object configured for time series

    Example:
        >>> config = create_time_series_data_config(
        ...     train_path="data/timeseries.csv",
        ...     target_column="value",
        ...     time_column="timestamp"
        ... )
    """
    source_config = DataSourceConfig(train_path=train_path, parse_dates=[time_column])

    features_config = FeatureConfig(target_column=target_column, date_features=[time_column])

    splitting_config = DataSplitConfig(
        time_series_split=True,
        time_column=time_column,
        cv_strategy="time_series",
        stratify=False,  # Usually not applicable for time series
    )

    return DataConfig(source=source_config, features=features_config, splitting=splitting_config, **kwargs)


# Export key classes and functions
__all__ = [
    "DataSourceConfig",
    "DataSplitConfig",
    "FeatureConfig",
    "FeatureEngineeringConfig",
    "PreprocessingConfig",
    "SampleWeightConfig",
    "ValidationConfig",
    "DataConfig",
    "create_basic_data_config",
    "create_advanced_data_config",
    "create_time_series_data_config",
]
