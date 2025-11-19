# tree_models/data/preprocessor.py
"""Enhanced data preprocessing with production-ready features.

This module provides comprehensive data preprocessing capabilities with:
- Type-safe interfaces and comprehensive validation
- Weight of Evidence (WoE) encoding with sample weights support
- Multiple imputation strategies with statistical approaches
- Column-specific processing configurations
- Advanced categorical encoding (target, WoE, one-hot)
- Robust scaling and transformation pipelines
- Mapping persistence and version control
- Performance optimization and memory management
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from ..utils.exceptions import (
    DataProcessingError,
    DataValidationError,
    create_error_context,
    handle_and_reraise,
    validate_parameter,
)
from ..utils.logger import get_logger
from ..utils.timer import timed_operation, timer

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


@dataclass
class ColumnConfig:
    """Type-safe configuration for column-specific preprocessing with validation."""

    # Missing value imputation
    missing_strategy: str = "median"  # "mean", "median", "most_frequent", "constant", "knn", "drop"
    missing_constant: Union[str, int, float] = -999999  # Value for constant strategy

    # Categorical encoding
    encoding_strategy: str = "label"  # "label", "onehot", "target", "woe", "ordinal"

    # Scaling and transformation
    scaling_strategy: Optional[str] = None  # "standard", "minmax", "robust", "quantile"
    transform_strategy: Optional[str] = None  # "log", "sqrt", "box-cox", "yeo-johnson"

    # Advanced options
    handle_unknown: str = "error"  # "error", "ignore", "use_encoded_value", "drop"
    unknown_value: Union[str, int, float] = -1  # Value for unknown categories

    # Outlier handling
    outlier_method: Optional[str] = None  # "iqr", "zscore", "isolation", "clip"
    outlier_threshold: float = 3.0  # Threshold for outlier detection

    # Feature engineering
    create_missing_indicator: bool = False  # Create binary indicator for missing values

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        valid_missing = ["mean", "median", "most_frequent", "constant", "knn", "drop"]
        valid_encoding = ["label", "onehot", "target", "woe", "ordinal"]
        valid_scaling = [None, "standard", "minmax", "robust", "quantile"]
        valid_transform = [None, "log", "sqrt", "box-cox", "yeo-johnson"]
        valid_unknown = ["error", "ignore", "use_encoded_value", "drop"]
        valid_outlier = [None, "iqr", "zscore", "isolation", "clip"]

        validate_parameter("missing_strategy", self.missing_strategy, valid_values=valid_missing)
        validate_parameter("encoding_strategy", self.encoding_strategy, valid_values=valid_encoding)
        validate_parameter("scaling_strategy", self.scaling_strategy, valid_values=valid_scaling)
        validate_parameter("transform_strategy", self.transform_strategy, valid_values=valid_transform)
        validate_parameter("handle_unknown", self.handle_unknown, valid_values=valid_unknown)
        validate_parameter("outlier_method", self.outlier_method, valid_values=valid_outlier)
        validate_parameter("outlier_threshold", self.outlier_threshold, min_value=0.1, max_value=10.0)


class WeightOfEvidenceEncoder:
    """Enhanced Weight of Evidence encoder with comprehensive features.

    Weight of Evidence measures the strength of a categorical variable's
    relationship with a binary target variable:
    WoE = ln(% of non-events / % of events)

    Example:
        >>> encoder = WeightOfEvidenceEncoder(smoothing=0.5)
        >>> X_encoded = encoder.fit_transform(X['category'], y, sample_weight=weights)
        >>> woe_mapping = encoder.get_mapping()
        >>> encoder.save_mapping('woe_mappings.json')
    """

    def __init__(
        self,
        smoothing: float = 0.5,
        handle_unknown: str = "use_encoded_value",
        min_samples_leaf: int = 1,
        regularization: float = 0.0,
    ) -> None:
        """Initialize enhanced WoE encoder.

        Args:
            smoothing: Smoothing factor to prevent division by zero
            handle_unknown: How to handle unknown categories during transform
            min_samples_leaf: Minimum samples required for a category
            regularization: L2 regularization factor for WoE values
        """
        validate_parameter("smoothing", smoothing, min_value=0.0, max_value=10.0)
        validate_parameter("min_samples_leaf", min_samples_leaf, min_value=1)
        validate_parameter("regularization", regularization, min_value=0.0, max_value=1.0)

        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        self.min_samples_leaf = min_samples_leaf
        self.regularization = regularization

        # Fitted parameters
        self.woe_mapping_: Dict[Any, float] = {}
        self.global_woe_: float = 0.0
        self.category_counts_: Dict[Any, int] = {}
        self.iv_value_: float = 0.0  # Information Value
        self.is_fitted_: bool = False

    @timer(name="woe_encoder_fitting")
    def fit(self, X: pd.Series, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "WeightOfEvidenceEncoder":
        """Fit the WoE encoder with enhanced statistics.

        Args:
            X: Categorical feature series
            y: Binary target series (0/1)
            sample_weight: Optional sample weights

        Returns:
            Fitted encoder

        Raises:
            DataProcessingError: If fitting fails
        """
        logger.info(f"🔍 Fitting WoE encoder: {X.nunique()} categories, {len(X)} samples")

        try:
            with timed_operation("woe_encoding_fit"):
                # Validate inputs
                self._validate_inputs(X, y, sample_weight)

                if sample_weight is None:
                    sample_weight = np.ones(len(X))

                # Create working DataFrame
                df = pd.DataFrame({"feature": X, "target": y, "weight": sample_weight}).dropna()

                # Calculate global statistics
                total_events = df[df["target"] == 1]["weight"].sum()
                total_non_events = df[df["target"] == 0]["weight"].sum()
                global_event_rate = total_events / (total_events + total_non_events)

                # Calculate WoE for each category
                woe_mapping = {}
                category_counts = {}
                iv_components = []

                for category in df["feature"].unique():
                    cat_data = df[df["feature"] == category]

                    # Check minimum sample requirement
                    if len(cat_data) < self.min_samples_leaf:
                        logger.debug(f"Category '{category}' has insufficient samples ({len(cat_data)}), skipping")
                        continue

                    # Calculate weighted counts
                    events_weight = cat_data[cat_data["target"] == 1]["weight"].sum()
                    non_events_weight = cat_data[cat_data["target"] == 0]["weight"].sum()
                    total_weight = cat_data["weight"].sum()

                    # Apply smoothing
                    events_weight_smooth = events_weight + self.smoothing
                    non_events_weight_smooth = non_events_weight + self.smoothing
                    total_events_smooth = total_events + self.smoothing
                    total_non_events_smooth = total_non_events + self.smoothing

                    # Calculate percentages
                    event_rate = events_weight_smooth / total_events_smooth
                    non_event_rate = non_events_weight_smooth / total_non_events_smooth

                    # Calculate WoE
                    if event_rate > 0 and non_event_rate > 0:
                        woe = np.log(non_event_rate / event_rate)

                        # Apply regularization
                        if self.regularization > 0:
                            woe = woe / (1 + self.regularization * np.abs(woe))
                    else:
                        woe = 0.0

                    woe_mapping[category] = woe
                    category_counts[category] = len(cat_data)

                    # Calculate Information Value component
                    dist_events = events_weight / total_events if total_events > 0 else 0
                    dist_non_events = non_events_weight / total_non_events if total_non_events > 0 else 0

                    if dist_events > 0 and dist_non_events > 0:
                        iv_component = (dist_non_events - dist_events) * woe
                        iv_components.append(iv_component)

                # Calculate global WoE for missing/unknown values
                self.global_woe_ = (
                    np.log((1 - global_event_rate) / global_event_rate)
                    if global_event_rate > 0 and global_event_rate < 1
                    else 0.0
                )

                # Store results
                self.woe_mapping_ = woe_mapping
                self.category_counts_ = category_counts
                self.iv_value_ = sum(iv_components)
                self.is_fitted_ = True

            logger.info("✅ WoE encoder fitted:")
            logger.info(f"   Categories: {len(woe_mapping)}")
            if woe_mapping:
                woe_vals = list(woe_mapping.values())
                logger.info(f"   WoE range: [{min(woe_vals):.3f}, {max(woe_vals):.3f}]")
            else:
                logger.info("   WoE range: [N/A, N/A] (no categories fitted)")
            logger.info(f"   Information Value: {self.iv_value_:.4f}")

            return self

        except Exception as e:
            handle_and_reraise(
                e,
                DataProcessingError,
                "WoE encoder fitting failed",
                error_code="WOE_FITTING_FAILED",
                context=create_error_context(
                    n_categories=X.nunique(), n_samples=len(X), has_weights=sample_weight is not None
                ),
            )

    def _validate_inputs(self, X: pd.Series, y: pd.Series, sample_weight: Optional[np.ndarray]) -> None:
        """Validate inputs for WoE encoding."""

        if len(X) != len(y):
            raise DataValidationError("X and y must have the same length")

        if sample_weight is not None and len(sample_weight) != len(X):
            raise DataValidationError("sample_weight must have the same length as X")

        # Check target is binary
        unique_targets = y.dropna().unique()
        if len(unique_targets) != 2 or not all(t in [0, 1] for t in unique_targets):
            raise DataValidationError("Target must be binary (0/1) for WoE encoding")

        if sample_weight is not None and np.any(sample_weight < 0):
            raise DataValidationError("Sample weights cannot be negative")

    def transform(self, X: pd.Series) -> np.ndarray:
        """Transform categorical series using fitted WoE mapping.

        Args:
            X: Categorical feature series to transform

        Returns:
            WoE encoded array

        Raises:
            DataProcessingError: If transformation fails
        """
        if not self.is_fitted_:
            raise DataProcessingError("WoE encoder must be fitted before transform", error_code="WOE_NOT_FITTED")

        try:
            # Handle unknown categories based on strategy
            if self.handle_unknown == "error":
                unknown_cats = set(X.dropna().unique()) - set(self.woe_mapping_.keys())
                if unknown_cats:
                    raise DataValidationError(f"Unknown categories found: {list(unknown_cats)}")

            # Transform using mapping
            transformed = X.map(self.woe_mapping_)

            # Handle missing/unknown values
            if self.handle_unknown == "use_encoded_value":
                transformed = transformed.fillna(self.global_woe_)
            elif self.handle_unknown == "ignore":
                transformed = transformed.fillna(0.0)
            elif self.handle_unknown == "drop":
                # This would require returning indices, not implemented here
                transformed = transformed.fillna(self.global_woe_)

            return transformed.values

        except Exception as e:
            handle_and_reraise(e, DataProcessingError, "WoE transformation failed", error_code="WOE_TRANSFORM_FAILED")

    def fit_transform(self, X: pd.Series, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit encoder and transform in one step."""
        return self.fit(X, y, sample_weight).transform(X)

    def get_mapping(self) -> Dict[Any, float]:
        """Get the WoE mapping dictionary."""
        if not self.is_fitted_:
            raise DataProcessingError("Encoder must be fitted first")
        return self.woe_mapping_.copy()

    def get_feature_strength(self) -> Dict[str, float]:
        """Get feature strength metrics.

        Returns:
            Dictionary with Information Value and other strength metrics
        """
        if not self.is_fitted_:
            raise DataProcessingError("Encoder must be fitted first")

        return {
            "information_value": self.iv_value_,
            "n_categories": len(self.woe_mapping_),
            "woe_range": max(self.woe_mapping_.values()) - min(self.woe_mapping_.values()) if self.woe_mapping_ else 0,
            "predictive_strength": "High" if self.iv_value_ > 0.3 else "Medium" if self.iv_value_ > 0.1 else "Low",
        }

    def save_mapping(self, filepath: Union[str, Path]) -> None:
        """Save WoE mapping to file with metadata."""
        if not self.is_fitted_:
            raise DataProcessingError("Encoder must be fitted before saving")

        try:
            mapping_data = {
                "woe_mapping": {str(k): float(v) for k, v in self.woe_mapping_.items()},
                "global_woe": float(self.global_woe_),
                "iv_value": float(self.iv_value_),
                "category_counts": {str(k): int(v) for k, v in self.category_counts_.items()},
                "config": {
                    "smoothing": self.smoothing,
                    "handle_unknown": self.handle_unknown,
                    "min_samples_leaf": self.min_samples_leaf,
                    "regularization": self.regularization,
                },
                "version": "2.0",
            }

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w") as f:
                json.dump(mapping_data, f, indent=2)

            logger.info(f"WoE mapping saved to {filepath}")

        except Exception as e:
            handle_and_reraise(
                e, DataProcessingError, f"Failed to save WoE mapping to {filepath}", error_code="WOE_SAVE_FAILED"
            )

    def load_mapping(self, filepath: Union[str, Path]) -> "WeightOfEvidenceEncoder":
        """Load WoE mapping from file."""
        try:
            with open(filepath) as f:
                mapping_data = json.load(f)

            self.woe_mapping_ = mapping_data["woe_mapping"]
            self.global_woe_ = mapping_data["global_woe"]
            self.iv_value_ = mapping_data.get("iv_value", 0.0)
            self.category_counts_ = mapping_data.get("category_counts", {})

            # Load config if available
            if "config" in mapping_data:
                config = mapping_data["config"]
                self.smoothing = config.get("smoothing", self.smoothing)
                self.handle_unknown = config.get("handle_unknown", self.handle_unknown)
                self.min_samples_leaf = config.get("min_samples_leaf", self.min_samples_leaf)
                self.regularization = config.get("regularization", self.regularization)

            self.is_fitted_ = True
            logger.info(f"WoE mapping loaded from {filepath}")

            return self

        except Exception as e:
            handle_and_reraise(
                e, DataProcessingError, f"Failed to load WoE mapping from {filepath}", error_code="WOE_LOAD_FAILED"
            )


class AdvancedDataPreprocessor:
    """Enhanced data preprocessor with production-ready features.

    Supports column-specific preprocessing strategies with comprehensive
    validation, error handling, and performance optimization.

    Example:
        >>> preprocessor = AdvancedDataPreprocessor()
        >>>
        >>> # Configure different strategies for different columns
        >>> preprocessor.set_column_config('age', ColumnConfig(
        ...     missing_strategy='median',
        ...     scaling_strategy='standard',
        ...     outlier_method='iqr'
        ... ))
        >>> preprocessor.set_column_config('category', ColumnConfig(
        ...     encoding_strategy='woe',
        ...     missing_strategy='constant',
        ...     missing_constant='unknown'
        ... ))
        >>>
        >>> X_processed = preprocessor.fit_transform(X_train, y_train, sample_weight=weights)
        >>> X_test_processed = preprocessor.transform(X_test)
    """

    def __init__(
        self,
        default_config: Optional[ColumnConfig] = None,
        mapping_save_dir: Optional[Union[str, Path]] = None,
        validation_enabled: bool = True,
        memory_efficient: bool = False,
    ) -> None:
        """Initialize advanced preprocessor.

        Args:
            default_config: Default configuration for all columns
            mapping_save_dir: Directory to save categorical mappings
            validation_enabled: Whether to perform comprehensive validation
            memory_efficient: Whether to optimize for memory usage
        """
        self.default_config = default_config or ColumnConfig()
        self.column_configs: Dict[str, ColumnConfig] = {}
        self.mapping_save_dir = Path(mapping_save_dir) if mapping_save_dir else None
        self.validation_enabled = validation_enabled
        self.memory_efficient = memory_efficient

        # Fitted components storage
        self.fitted_transformers: Dict[str, Any] = {}
        self.categorical_mappings: Dict[str, Any] = {}
        self.feature_names_in_: Optional[List[str]] = None
        self.feature_names_out_: Optional[List[str]] = None
        self.preprocessing_stats_: Dict[str, Any] = {}
        self.is_fitted_: bool = False

        # Performance tracking
        self.processing_times_: Dict[str, float] = {}

        # Create mapping directory if specified
        if self.mapping_save_dir:
            self.mapping_save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized AdvancedDataPreprocessor:")
        logger.info(f"  Validation: {validation_enabled}, Memory efficient: {memory_efficient}")

    def set_column_config(self, column: str, config: ColumnConfig) -> None:
        """Set configuration for a specific column."""
        self.column_configs[column] = config
        logger.debug(f"Set config for column '{column}'")

    def set_column_configs(self, configs: Dict[str, ColumnConfig]) -> None:
        """Set configurations for multiple columns."""
        self.column_configs.update(configs)
        logger.info(f"Set configs for {len(configs)} columns")

    def get_column_config(self, column: str) -> ColumnConfig:
        """Get configuration for a column (default if not specified)."""
        return self.column_configs.get(column, self.default_config)

    def _compute_weighted_statistics(self, values: np.ndarray, weights: np.ndarray, stat_type: str) -> float:
        """Compute weighted statistics."""

        if stat_type == "mean":
            return np.average(values, weights=weights)
        elif stat_type == "median":
            return self._weighted_percentile(values, weights, 50)
        elif stat_type == "mode":
            unique_vals, inverse_indices = np.unique(values, return_inverse=True)
            weighted_counts = np.bincount(inverse_indices, weights=weights)
            mode_idx = np.argmax(weighted_counts)
            return unique_vals[mode_idx]
        else:
            raise ValueError(f"Unknown statistic type: {stat_type}")

    def _weighted_percentile(self, values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
        """Calculate weighted percentile."""
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumsum = np.cumsum(sorted_weights)
        cutoff = cumsum[-1] * percentile / 100.0

        idx = np.searchsorted(cumsum, cutoff)
        if idx < len(sorted_values):
            return sorted_values[idx]
        else:
            return sorted_values[-1]

    @timer(name="missing_value_imputation")
    def _impute_missing_values(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, sample_weight: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Impute missing values using column-specific strategies."""

        X_imputed = X.copy()

        for column in X.columns:
            missing_count = X[column].isnull().sum()
            if missing_count == 0:
                continue

            config = self.get_column_config(column)
            strategy = config.missing_strategy

            logger.debug(f"Imputing {missing_count} missing values in '{column}' using '{strategy}'")

            try:
                with timed_operation(f"impute_{column}") as timing:
                    if strategy == "drop":
                        # Skip imputation, will be handled later
                        continue

                    elif strategy in ["mean", "median"]:
                        if pd.api.types.is_numeric_dtype(X[column]):
                            non_null_mask = ~X[column].isnull()
                            if sample_weight is not None and non_null_mask.sum() > 0:
                                fill_value = self._compute_weighted_statistics(
                                    X[column][non_null_mask].values, sample_weight[non_null_mask], strategy
                                )
                            else:
                                fill_value = X[column].mean() if strategy == "mean" else X[column].median()
                        else:
                            logger.warning(f"Cannot use {strategy} for non-numeric column '{column}', using mode")
                            fill_value = X[column].mode()[0] if len(X[column].mode()) > 0 else "unknown"

                        X_imputed[column] = X_imputed[column].fillna(fill_value)
                        self.fitted_transformers[f"{column}_impute_value"] = fill_value

                    elif strategy == "most_frequent":
                        non_null_mask = ~X[column].isnull()
                        if sample_weight is not None and non_null_mask.sum() > 0:
                            fill_value = self._compute_weighted_statistics(
                                X[column][non_null_mask].values, sample_weight[non_null_mask], "mode"
                            )
                        else:
                            fill_value = X[column].mode()[0] if len(X[column].mode()) > 0 else "unknown"

                        X_imputed[column] = X_imputed[column].fillna(fill_value)
                        self.fitted_transformers[f"{column}_impute_value"] = fill_value

                    elif strategy == "constant":
                        fill_value = config.missing_constant
                        X_imputed[column] = X_imputed[column].fillna(fill_value)
                        self.fitted_transformers[f"{column}_impute_value"] = fill_value

                    elif strategy == "knn":
                        if pd.api.types.is_numeric_dtype(X[column]):
                            imputer = KNNImputer(n_neighbors=5)
                            X_imputed[column] = imputer.fit_transform(X[[column]]).ravel()
                            self.fitted_transformers[f"{column}_knn_imputer"] = imputer
                        else:
                            logger.warning(f"KNN not suitable for non-numeric column '{column}', using mode")
                            fill_value = X[column].mode()[0] if len(X[column].mode()) > 0 else "unknown"
                            X_imputed[column] = X_imputed[column].fillna(fill_value)
                            self.fitted_transformers[f"{column}_impute_value"] = fill_value

                    # Create missing indicator if requested
                    if config.create_missing_indicator and missing_count > 0:
                        indicator_name = f"{column}_was_missing"
                        X_imputed[indicator_name] = X[column].isnull().astype(int)
                        logger.debug(f"Created missing indicator: {indicator_name}")

                self.processing_times_[f"impute_{column}"] = timing["duration"]

            except Exception as e:
                logger.error(f"Failed to impute column '{column}': {e}")
                # Fallback to simple constant imputation
                X_imputed[column] = X_imputed[column].fillna("unknown" if X[column].dtype == "object" else -999999)

        return X_imputed

    @timer(name="categorical_encoding")
    def _encode_categorical(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, sample_weight: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Encode categorical variables using column-specific strategies."""

        X_encoded = X.copy()
        columns_to_drop = []
        columns_to_add = {}

        categorical_columns = X.select_dtypes(include=["object", "category"]).columns

        for column in categorical_columns:
            config = self.get_column_config(column)
            strategy = config.encoding_strategy

            logger.debug(f"Encoding '{column}' using '{strategy}' strategy")

            try:
                with timed_operation(f"encode_{column}") as timing:
                    if strategy == "label":
                        encoder = LabelEncoder()
                        non_null_mask = ~X[column].isnull()

                        if non_null_mask.sum() > 0:
                            # Fit on non-null values
                            encoder.fit(X[column].dropna().astype(str))

                            # Vectorized mapping: build mapping from class -> label and apply via Series.map
                            mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                            X_str = X[column].astype(str)
                            mapped = X_str.map(mapping).astype(float)

                            # Fill unknowns with configured unknown value
                            X_encoded[column] = mapped.fillna(config.unknown_value)

                            # Save encoder and mapping
                            self.fitted_transformers[f"{column}_label_encoder"] = encoder
                            self.categorical_mappings[column] = mapping

                    elif strategy == "onehot":
                        # One-hot encoding with better handling
                        dummies = pd.get_dummies(X[column], prefix=column, dummy_na=False, dtype=int)

                        columns_to_drop.append(column)
                        for col in dummies.columns:
                            columns_to_add[col] = dummies[col]

                        # Save information for transform
                        self.fitted_transformers[f"{column}_onehot_columns"] = list(dummies.columns)
                        self.categorical_mappings[column] = {
                            "type": "onehot",
                            "columns": list(dummies.columns),
                            "categories": X[column].dropna().unique().tolist(),
                        }

                    elif strategy == "target":
                        if y is None:
                            raise DataProcessingError("Target encoding requires target variable y")

                        # Target encoding with regularization
                        target_means = {}
                        global_mean = np.average(y, weights=sample_weight) if sample_weight is not None else y.mean()

                        for category in X[column].dropna().unique():
                            mask = X[column] == category
                            if mask.sum() > 0:
                                if sample_weight is not None:
                                    cat_mean = np.average(y[mask], weights=sample_weight[mask])
                                else:
                                    cat_mean = y[mask].mean()

                                # Apply smoothing based on sample size
                                n_samples = mask.sum()
                                smoothing_factor = n_samples / (n_samples + 100)  # Adjust smoothing
                                target_means[category] = (
                                    smoothing_factor * cat_mean + (1 - smoothing_factor) * global_mean
                                )

                        # Apply encoding
                        X_encoded[column] = X[column].map(target_means).fillna(global_mean)

                        # Save mapping
                        target_means["__global_mean__"] = global_mean
                        self.fitted_transformers[f"{column}_target_encoder"] = target_means
                        self.categorical_mappings[column] = target_means

                    elif strategy == "woe":
                        if y is None:
                            raise DataProcessingError("WoE encoding requires target variable y")

                        # Use enhanced WoE encoder
                        woe_encoder = WeightOfEvidenceEncoder(
                            handle_unknown=config.handle_unknown,
                            min_samples_leaf=5,  # Minimum samples for robust WoE
                            regularization=0.1,  # Small regularization
                        )

                        X_encoded[column] = woe_encoder.fit_transform(X[column], y, sample_weight)

                        # Save encoder and mapping
                        self.fitted_transformers[f"{column}_woe_encoder"] = woe_encoder
                        self.categorical_mappings[column] = woe_encoder.get_mapping()

                        # Log feature strength
                        strength_metrics = woe_encoder.get_feature_strength()
                        logger.info(
                            f"WoE encoding for '{column}': IV={strength_metrics['information_value']:.4f}, "
                            f"Strength={strength_metrics['predictive_strength']}"
                        )

                    # Save mapping to file if directory specified
                    if self.mapping_save_dir and column in self.categorical_mappings:
                        mapping_file = self.mapping_save_dir / f"{column}_{strategy}_mapping.json"
                        try:
                            with open(mapping_file, "w") as f:
                                json.dump(
                                    {str(k): v for k, v in self.categorical_mappings[column].items()},
                                    f,
                                    indent=2,
                                    default=str,
                                )
                        except Exception as e:
                            logger.warning(f"Failed to save mapping for {column}: {e}")

                self.processing_times_[f"encode_{column}"] = timing["duration"]

            except Exception as e:
                logger.error(f"Failed to encode column '{column}': {e}")
                # Fallback to simple label encoding
                try:
                    encoder = LabelEncoder()
                    X_encoded[column] = encoder.fit_transform(X[column].astype(str).fillna("unknown"))
                    self.fitted_transformers[f"{column}_label_encoder"] = encoder
                except Exception:
                    logger.error(f"Fallback encoding failed for '{column}', dropping column")
                    columns_to_drop.append(column)

        # Apply column modifications for one-hot encoding
        if columns_to_drop:
            X_encoded = X_encoded.drop(columns=columns_to_drop)

        for col_name, col_data in columns_to_add.items():
            X_encoded[col_name] = col_data

        return X_encoded

    @timer(name="feature_scaling")
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using column-specific strategies."""

        X_scaled = X.copy()
        numeric_columns = X.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            config = self.get_column_config(column)
            strategy = config.scaling_strategy

            if strategy is None:
                continue

            logger.debug(f"Scaling '{column}' using '{strategy}' strategy")

            try:
                # Create appropriate scaler
                if strategy == "standard":
                    scaler = StandardScaler()
                elif strategy == "minmax":
                    scaler = MinMaxScaler()
                elif strategy == "robust":
                    scaler = RobustScaler()
                else:
                    logger.warning(f"Unknown scaling strategy '{strategy}' for column '{column}'")
                    continue

                # Fit and transform
                X_scaled[column] = scaler.fit_transform(X[[column]]).ravel()

                # Save scaler
                self.fitted_transformers[f"{column}_{strategy}_scaler"] = scaler

            except Exception as e:
                logger.error(f"Failed to scale column '{column}': {e}")

        return X_scaled

    @timer(name="data_preprocessing_fit")
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, sample_weight: Optional[np.ndarray] = None
    ) -> "AdvancedDataPreprocessor":
        """Fit the preprocessor on training data.

        Args:
            X: Training features
            y: Training target (needed for target/WoE encoding)
            sample_weight: Training sample weights

        Returns:
            Fitted preprocessor

        Raises:
            DataProcessingError: If fitting fails
        """
        logger.info("🔧 Fitting AdvancedDataPreprocessor:")
        logger.info(f"   Data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"   Has target: {y is not None}, Has weights: {sample_weight is not None}")

        try:
            with timed_operation("preprocessing_fit") as timing:
                # Validate inputs
                if self.validation_enabled:
                    self._validate_fit_inputs(X, y, sample_weight)

                # Store input feature names
                self.feature_names_in_ = list(X.columns)

                # Create working copy
                X_work = X.copy()

                # Step 1: Handle missing values
                X_work = self._impute_missing_values(X_work, y, sample_weight)

                # Step 2: Encode categorical variables
                X_work = self._encode_categorical(X_work, y, sample_weight)

                # Step 3: Scale features
                # Assign the scaled DataFrame back so downstream steps use scaled values
                X_work = self._scale_features(X_work)

                # Store output feature names
                self.feature_names_out_ = list(X_work.columns)
                self.is_fitted_ = True

                self.processing_times_["total_fit_time"] = timing["duration"]

                # Log summary
                logger.info("✅ Preprocessor fitted successfully")
                logger.info(f"   Input features: {len(self.feature_names_in_)}")
                logger.info(f"   Output features: {len(self.feature_names_out_)}")
                logger.info(f"   Transformers: {len(self.fitted_transformers)}")

                return self

        except Exception as e:
            handle_and_reraise(
                e, DataProcessingError, "Preprocessor fitting failed", error_code="PREPROCESS_FIT_FAILED"
            )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline.

        Args:
            X: Features to transform

        Returns:
            Transformed DataFrame

        Raises:
            DataProcessingError: If transformation fails
        """
        if not self.is_fitted_:
            raise DataProcessingError("Preprocessor must be fitted before transform", error_code="NOT_FITTED")

        logger.info(f"🔄 Transforming data: {X.shape[0]} samples")

        try:
            with timed_operation("preprocessing_transform"):
                X_trans = X.copy()

                # Check input features
                missing_cols = set(self.feature_names_in_) - set(X.columns)
                if missing_cols:
                    if self.validation_enabled:
                        raise DataValidationError(f"Missing columns in input: {missing_cols}")
                    else:
                        # Add missing columns with NaNs if validation disabled
                        for col in missing_cols:
                            X_trans[col] = np.nan

                # Apply transformations in order

                # 1. Imputation (using fitted values)
                for col in self.feature_names_in_:
                    if col not in X_trans.columns:
                        continue

                    # Apply constant/mode/median from fit
                    if f"{col}_impute_value" in self.fitted_transformers:
                        fill_val = self.fitted_transformers[f"{col}_impute_value"]
                        X_trans[col] = X_trans[col].fillna(fill_val)
                    elif f"{col}_knn_imputer" in self.fitted_transformers:
                        imputer = self.fitted_transformers[f"{col}_knn_imputer"]
                        X_trans[col] = imputer.transform(X_trans[[col]]).ravel()

                # 2. Categorical Encoding
                for col, mapping in self.categorical_mappings.items():
                    if col not in X_trans.columns:
                        continue

                    if isinstance(mapping, dict) and "type" in mapping and mapping["type"] == "onehot":
                        # One-hot encoding
                        dummies = pd.get_dummies(X_trans[col], prefix=col, dummy_na=False, dtype=int)
                        expected_cols = mapping["columns"]

                        # Add missing dummy columns as 0
                        for dummy_col in expected_cols:
                            if dummy_col not in dummies.columns:
                                dummies[dummy_col] = 0

                        # Drop original and add dummies
                        X_trans = X_trans.drop(columns=[col])
                        for dummy_col in expected_cols:
                            X_trans[dummy_col] = dummies[dummy_col]

                    elif f"{col}_label_encoder" in self.fitted_transformers:
                        # Label encoding
                        encoder = self.fitted_transformers[f"{col}_label_encoder"]
                        # Handle unknown values
                        X_trans[col] = (
                            X_trans[col]
                            .astype(str)
                            .apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
                        )

                    elif f"{col}_target_encoder" in self.fitted_transformers:
                        # Target encoding
                        mapping = self.fitted_transformers[f"{col}_target_encoder"]
                        global_mean = mapping["__global_mean__"]
                        X_trans[col] = X_trans[col].map(mapping).fillna(global_mean)

                    elif f"{col}_woe_encoder" in self.fitted_transformers:
                        # WoE encoding
                        encoder = self.fitted_transformers[f"{col}_woe_encoder"]
                        X_trans[col] = encoder.transform(X_trans[col])

                # 3. Scaling
                for col in X_trans.columns:
                    if f"{col}_standard_scaler" in self.fitted_transformers:
                        scaler = self.fitted_transformers[f"{col}_standard_scaler"]
                        X_trans[col] = scaler.transform(X_trans[[col]]).ravel()
                    elif f"{col}_minmax_scaler" in self.fitted_transformers:
                        scaler = self.fitted_transformers[f"{col}_minmax_scaler"]
                        X_trans[col] = scaler.transform(X_trans[[col]]).ravel()
                    elif f"{col}_robust_scaler" in self.fitted_transformers:
                        scaler = self.fitted_transformers[f"{col}_robust_scaler"]
                        X_trans[col] = scaler.transform(X_trans[[col]]).ravel()

                # Ensure output columns match fit
                # Reorder and select columns
                if self.feature_names_out_:
                    # Add any missing columns as 0 (e.g. one-hot levels not seen)
                    for col in self.feature_names_out_:
                        if col not in X_trans.columns:
                            X_trans[col] = 0

                    # Select only expected columns in correct order
                    X_trans = X_trans[self.feature_names_out_]

                return X_trans

        except Exception as e:
            handle_and_reraise(e, DataProcessingError, "Transformation failed", error_code="TRANSFORM_FAILED")

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, sample_weight: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y, sample_weight).transform(X)

    def _validate_fit_inputs(
        self, X: pd.DataFrame, y: Optional[pd.Series], sample_weight: Optional[np.ndarray]
    ) -> None:
        """Validate inputs for fitting."""
        if X.empty:
            raise DataValidationError("Input DataFrame is empty")

        if y is not None and len(X) != len(y):
            raise DataValidationError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        if sample_weight is not None and len(X) != len(sample_weight):
            raise DataValidationError(f"X and sample_weight length mismatch: {len(X)} vs {len(sample_weight)}")

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        if not self.is_fitted_:
            raise DataProcessingError("Preprocessor must be fitted first")

        return self.feature_names_out_.copy() if self.feature_names_out_ else []

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics and performance metrics."""
        if not self.is_fitted_:
            raise DataProcessingError("Preprocessor must be fitted first")

        return self.preprocessing_stats_.copy()

    def save_mappings(self, base_path: Union[str, Path]) -> None:
        """Save all categorical mappings and transformers to files."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Save categorical mappings
        for column, mapping in self.categorical_mappings.items():
            save_path = base_path / f"{column}_mapping.json"
            try:
                with open(save_path, "w") as f:
                    json.dump({str(k): v for k, v in mapping.items()}, f, indent=2, default=str)
                logger.info(f"Saved mapping for column '{column}' to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save mapping for {column}: {e}")

        # Save preprocessor configuration and metadata
        config_data = {
            "feature_names_in": self.feature_names_in_,
            "feature_names_out": self.feature_names_out_,
            "preprocessing_stats": self.preprocessing_stats_,
            "column_configs": {
                col: {
                    "missing_strategy": config.missing_strategy,
                    "encoding_strategy": config.encoding_strategy,
                    "scaling_strategy": config.scaling_strategy,
                    "handle_unknown": config.handle_unknown,
                }
                for col, config in self.column_configs.items()
            },
        }

        config_path = base_path / "preprocessor_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2, default=str)

        logger.info(f"Saved preprocessor configuration to {config_path}")


# Convenience functions for backward compatibility and quick usage
def preprocess_data(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
    column_configs: Optional[Dict[str, ColumnConfig]] = None,
    default_config: Optional[ColumnConfig] = None,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, AdvancedDataPreprocessor]:
    """Quick data preprocessing with advanced options.

    Args:
        X: Feature DataFrame
        y: Target series
        sample_weight: Sample weights
        column_configs: Dictionary of column-specific configurations
        default_config: Default configuration for all columns
        **kwargs: Additional preprocessor parameters

    Returns:
        Tuple of (processed_dataframe, fitted_preprocessor)

    Example:
        >>> configs = {
        ...     'age': ColumnConfig(missing_strategy='median', scaling_strategy='standard'),
        ...     'category': ColumnConfig(encoding_strategy='woe', missing_strategy='constant')
        ... }
        >>> X_processed, preprocessor = preprocess_data(
        ...     X, y, sample_weight=weights, column_configs=configs
        ... )
    """
    preprocessor = AdvancedDataPreprocessor(default_config=default_config, **kwargs)

    if column_configs:
        preprocessor.set_column_configs(column_configs)

    X_processed = preprocessor.fit_transform(X, y, sample_weight)

    return X_processed, preprocessor


def create_woe_encoding(
    X_cat: pd.Series,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, WeightOfEvidenceEncoder]:
    """Create Weight of Evidence encoding for a categorical feature.

    Args:
        X_cat: Categorical feature series
        y: Binary target series
        sample_weight: Optional sample weights
        save_path: Optional path to save the WoE mapping
        **kwargs: Additional encoder parameters

    Returns:
        Tuple of (woe_encoded_values, fitted_encoder)

    Example:
        >>> woe_values, encoder = create_woe_encoding(
        ...     X['category'], y, sample_weight=weights,
        ...     save_path='category_woe.json'
        ... )
        >>> strength_metrics = encoder.get_feature_strength()
        >>> print(f"Information Value: {strength_metrics['information_value']:.4f}")
    """
    encoder = WeightOfEvidenceEncoder(**kwargs)
    woe_values = encoder.fit_transform(X_cat, y, sample_weight)

    if save_path:
        encoder.save_mapping(save_path)

    return woe_values, encoder


# Alias for backward compatibility
DataPreprocessor = AdvancedDataPreprocessor

# Export key classes and functions
__all__ = [
    "ColumnConfig",
    "WeightOfEvidenceEncoder",
    "AdvancedDataPreprocessor",
    "DataPreprocessor",
    "preprocess_data",
    "create_woe_encoding",
]
