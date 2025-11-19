# tree_models/data/feature_engineer.py
"""Enhanced feature engineering tools with comprehensive transformations.

This module provides comprehensive feature engineering capabilities with:
- Type-safe feature transformation and creation pipelines
- Advanced mathematical and statistical transformations
- Time series and temporal feature engineering
- Categorical feature engineering with encoding strategies
- Text and NLP feature extraction capabilities
- Interaction and polynomial feature creation
- Domain-specific feature engineering (financial, behavioral, etc.)
- Feature validation and quality assessment
- Performance optimization for large datasets
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config.data_config import FeatureEngineeringConfig
from ..utils.exceptions import (
    FeatureEngineeringError,
    create_error_context,
    handle_and_reraise,
)
from ..utils.logger import get_logger
from ..utils.timer import timed_operation, timer

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

# Optional dependencies with fallbacks
try:
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available for advanced feature engineering")

try:
    from scipy import stats
    from scipy.special import boxcox1p

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available for statistical transformations")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available for text processing")


@dataclass
class FeatureEngineeringResults:
    """Results from feature engineering operations."""

    # Transformed data
    data: pd.DataFrame

    # Feature metadata
    original_features: List[str]
    new_features: List[str]
    dropped_features: List[str] = field(default_factory=list)

    # Transformation info
    transformations_applied: List[str] = field(default_factory=list)
    feature_importance: Optional[Dict[str, float]] = None

    # Statistics
    n_features_before: int = 0
    n_features_after: int = 0
    processing_time: float = 0.0

    # Metadata
    config_used: Optional[FeatureEngineeringConfig] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class FeatureEngineer:
    """Enhanced feature engineering with comprehensive transformation capabilities.

    Provides systematic feature engineering with mathematical transformations,
    interaction features, time series features, and text processing.

    Example:
        >>> engineer = FeatureEngineer()
        >>>
        >>> # Configure feature engineering
        >>> config = FeatureEngineeringConfig(
        ...     log_transform_cols=['income', 'amount'],
        ...     create_polynomial_features=True,
        ...     polynomial_degree=2,
        ...     create_ratios=[('income', 'expenses'), ('assets', 'liabilities')],
        ...     extract_date_features=['transaction_date']
        ... )
        >>>
        >>> # Engineer features
        >>> results = engineer.engineer_features(df, config=config)
        >>> print(f"Features: {results.n_features_before} → {results.n_features_after}")
        >>>
        >>> # Get transformed data
        >>> df_transformed = results.data
    """

    def __init__(self, random_state: int = 42, enable_logging: bool = True) -> None:
        """Initialize feature engineer.

        Args:
            random_state: Random state for reproducibility
            enable_logging: Whether to enable detailed logging
        """
        self.random_state = random_state
        self.enable_logging = enable_logging

        # Fitted transformers for consistency
        self.fitted_transformers_: Dict[str, Any] = {}
        self.feature_names_: Optional[List[str]] = None

        # Set random seeds
        np.random.seed(random_state)

        logger.info(f"Initialized FeatureEngineer with random_state={random_state}")

    @timer(name="feature_engineering")
    def engineer_features(
        self, data: pd.DataFrame, config: Optional[FeatureEngineeringConfig] = None, target_column: Optional[str] = None
    ) -> FeatureEngineeringResults:
        """Engineer features with comprehensive transformations.

        Args:
            data: Input DataFrame
            config: Feature engineering configuration
            target_column: Target column (excluded from transformations)

        Returns:
            Feature engineering results with transformed data

        Raises:
            FeatureEngineeringError: If feature engineering fails
        """
        logger.info("🔧 Starting feature engineering:")
        logger.info(f"   Input shape: {data.shape}")
        logger.info(f"   Target column: {target_column}")

        if config is None:
            config = FeatureEngineeringConfig()

        start_time = datetime.now()

        try:
            with timed_operation("feature_engineering") as timing:
                # Initialize results
                results = FeatureEngineeringResults(
                    data=data.copy(),
                    original_features=list(data.columns),
                    new_features=[],
                    n_features_before=len(data.columns),
                    config_used=config,
                )

                # Exclude target column from transformations
                feature_cols = [col for col in data.columns if col != target_column]

                # Apply transformations in order

                # 1. Mathematical transformations
                if any([config.log_transform_cols, config.sqrt_transform_cols, config.box_cox_transform_cols]):
                    results = self._apply_mathematical_transforms(results, config)

                # 2. Date/time features
                if config.extract_date_features:
                    results = self._extract_date_features(results, config)

                # 3. Binning and discretization
                if config.create_bins or config.quantile_bins:
                    results = self._create_binned_features(results, config)

                # 4. Ratio and difference features
                if config.create_ratios or config.create_differences:
                    results = self._create_ratio_difference_features(results, config)

                # 5. Aggregation features
                if config.groupby_aggregations:
                    results = self._create_aggregation_features(results, config)

                # 6. Rolling window features
                if config.rolling_windows:
                    results = self._create_rolling_features(results, config)

                # 7. Lag features
                if config.create_lag_features:
                    results = self._create_lag_features(results, config)

                # 8. Text features
                if config.text_vectorization:
                    results = self._create_text_features(results, config)

                # 9. Polynomial and interaction features
                if config.create_polynomial_features or config.create_interaction_features:
                    results = self._create_polynomial_interaction_features(results, config, target_column)

                # 10. Feature validation and cleanup
                if config.validate_features:
                    results = self._validate_and_cleanup_features(results, config)

                # Update final statistics
                results.n_features_after = len(results.data.columns)
                results.processing_time = timing["duration"]

                # Calculate new features
                results.new_features = [col for col in results.data.columns if col not in results.original_features]

            logger.info("✅ Feature engineering completed:")
            logger.info(f"   Duration: {results.processing_time:.2f}s")
            logger.info(f"   Features: {results.n_features_before} → {results.n_features_after}")
            logger.info(f"   New features: {len(results.new_features)}")
            logger.info(f"   Transformations: {len(results.transformations_applied)}")

            return results

        except Exception as e:
            handle_and_reraise(
                e,
                FeatureEngineeringError,
                "Feature engineering failed",
                error_code="FEATURE_ENGINEERING_FAILED",
                context=create_error_context(input_shape=data.shape, target_column=target_column),
            )

    def _apply_mathematical_transforms(
        self, results: FeatureEngineeringResults, config: FeatureEngineeringConfig
    ) -> FeatureEngineeringResults:
        """Apply mathematical transformations (log, sqrt, box-cox)."""

        logger.debug("Applying mathematical transformations")

        data = results.data

        # Log transformations
        if config.log_transform_cols:
            for col in config.log_transform_cols:
                if col in data.columns:
                    # Ensure positive values for log transformation
                    min_val = data[col].min()
                    if min_val <= 0:
                        # Add constant to make all values positive
                        shift_val = abs(min_val) + 1
                        data[f"{col}_log"] = np.log1p(data[col] + shift_val)
                        logger.debug(f"Applied log transform to {col} with shift {shift_val}")
                    else:
                        data[f"{col}_log"] = np.log1p(data[col])
                        logger.debug(f"Applied log transform to {col}")

                    results.transformations_applied.append(f"log_transform_{col}")

        # Square root transformations
        if config.sqrt_transform_cols:
            for col in config.sqrt_transform_cols:
                if col in data.columns:
                    # Ensure non-negative values
                    if data[col].min() >= 0:
                        data[f"{col}_sqrt"] = np.sqrt(data[col])
                        results.transformations_applied.append(f"sqrt_transform_{col}")
                        logger.debug(f"Applied sqrt transform to {col}")
                    else:
                        logger.warning(f"Cannot apply sqrt transform to {col}: contains negative values")

        # Box-Cox transformations
        if config.box_cox_transform_cols and SCIPY_AVAILABLE:
            for col in config.box_cox_transform_cols:
                if col in data.columns:
                    try:
                        # Box-Cox requires positive values
                        if data[col].min() > 0:
                            data[f"{col}_boxcox"] = boxcox1p(data[col], 0.15)  # lambda=0.15
                            results.transformations_applied.append(f"boxcox_transform_{col}")
                            logger.debug(f"Applied Box-Cox transform to {col}")
                        else:
                            logger.warning(f"Cannot apply Box-Cox transform to {col}: contains non-positive values")
                    except Exception as e:
                        logger.warning(f"Box-Cox transform failed for {col}: {e}")

        results.data = data
        return results

    def _extract_date_features(
        self, results: FeatureEngineeringResults, config: FeatureEngineeringConfig
    ) -> FeatureEngineeringResults:
        """Extract date and time features."""

        logger.debug("Extracting date/time features")

        data = results.data

        for col in config.extract_date_features:
            if col in data.columns:
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(data[col]):
                        date_col = pd.to_datetime(data[col], errors="coerce")
                    else:
                        date_col = data[col]

                    # Extract various date components
                    data[f"{col}_year"] = date_col.dt.year
                    data[f"{col}_month"] = date_col.dt.month
                    data[f"{col}_day"] = date_col.dt.day
                    data[f"{col}_dayofweek"] = date_col.dt.dayofweek
                    data[f"{col}_quarter"] = date_col.dt.quarter
                    data[f"{col}_week"] = date_col.dt.isocalendar().week
                    data[f"{col}_is_weekend"] = (date_col.dt.dayofweek >= 5).astype(int)

                    # Additional time-based features
                    if date_col.dt.hour.notna().any():  # If time information is available
                        data[f"{col}_hour"] = date_col.dt.hour
                        data[f"{col}_is_business_hours"] = ((date_col.dt.hour >= 9) & (date_col.dt.hour <= 17)).astype(
                            int
                        )

                    results.transformations_applied.append(f"date_features_{col}")
                    logger.debug(f"Extracted date features from {col}")

                except Exception as e:
                    logger.warning(f"Failed to extract date features from {col}: {e}")

        results.data = data
        return results

    def _create_binned_features(
        self, results: FeatureEngineeringResults, config: FeatureEngineeringConfig
    ) -> FeatureEngineeringResults:
        """Create binned/discretized features."""

        logger.debug("Creating binned features")

        data = results.data

        # Custom bins
        if config.create_bins:
            for col, bins in config.create_bins.items():
                if col in data.columns:
                    try:
                        if isinstance(bins, int):
                            # Equal-width binning
                            data[f"{col}_binned"] = pd.cut(data[col], bins=bins, labels=False)
                        else:
                            # Custom bin edges
                            data[f"{col}_binned"] = pd.cut(data[col], bins=bins, labels=False)

                        results.transformations_applied.append(f"binning_{col}")
                        logger.debug(f"Created bins for {col}")
                    except Exception as e:
                        logger.warning(f"Failed to create bins for {col}: {e}")

        # Quantile-based bins
        if config.quantile_bins:
            for col, n_bins in config.quantile_bins.items():
                if col in data.columns:
                    try:
                        data[f"{col}_qbinned"] = pd.qcut(data[col], q=n_bins, labels=False, duplicates="drop")
                        results.transformations_applied.append(f"quantile_binning_{col}")
                        logger.debug(f"Created quantile bins for {col}")
                    except Exception as e:
                        logger.warning(f"Failed to create quantile bins for {col}: {e}")

        results.data = data
        return results

    def _create_ratio_difference_features(
        self, results: FeatureEngineeringResults, config: FeatureEngineeringConfig
    ) -> FeatureEngineeringResults:
        """Create ratio and difference features."""

        logger.debug("Creating ratio and difference features")

        data = results.data

        # Ratio features
        if config.create_ratios:
            for col1, col2 in config.create_ratios:
                if col1 in data.columns and col2 in data.columns:
                    try:
                        # Avoid division by zero
                        ratio = data[col1] / (data[col2] + 1e-8)
                        data[f"{col1}_div_{col2}"] = ratio.replace([np.inf, -np.inf], np.nan)

                        results.transformations_applied.append(f"ratio_{col1}_{col2}")
                        logger.debug(f"Created ratio feature: {col1}/{col2}")
                    except Exception as e:
                        logger.warning(f"Failed to create ratio {col1}/{col2}: {e}")

        # Difference features
        if config.create_differences:
            for col1, col2 in config.create_differences:
                if col1 in data.columns and col2 in data.columns:
                    try:
                        data[f"{col1}_minus_{col2}"] = data[col1] - data[col2]

                        results.transformations_applied.append(f"difference_{col1}_{col2}")
                        logger.debug(f"Created difference feature: {col1}-{col2}")
                    except Exception as e:
                        logger.warning(f"Failed to create difference {col1}-{col2}: {e}")

        results.data = data
        return results

    def _create_aggregation_features(
        self, results: FeatureEngineeringResults, config: FeatureEngineeringConfig
    ) -> FeatureEngineeringResults:
        """Create aggregation features based on groupings."""

        logger.debug("Creating aggregation features")

        data = results.data

        for group_col, agg_config in config.groupby_aggregations.items():
            if group_col in data.columns:
                try:
                    for target_col, agg_funcs in agg_config.items():
                        if target_col in data.columns:
                            grouped = data.groupby(group_col)[target_col]

                            for agg_func in agg_funcs:
                                if agg_func in ["mean", "median", "std", "min", "max", "sum", "count"]:
                                    agg_result = getattr(grouped, agg_func)()
                                    feature_name = f"{target_col}_{agg_func}_by_{group_col}"
                                    data[feature_name] = data[group_col].map(agg_result)

                                    results.transformations_applied.append(f"aggregation_{feature_name}")
                                    logger.debug(f"Created aggregation feature: {feature_name}")

                except Exception as e:
                    logger.warning(f"Failed to create aggregation features for {group_col}: {e}")

        results.data = data
        return results

    def _create_rolling_features(
        self, results: FeatureEngineeringResults, config: FeatureEngineeringConfig
    ) -> FeatureEngineeringResults:
        """Create rolling window features."""

        logger.debug("Creating rolling window features")

        data = results.data

        for col, window_config in config.rolling_windows.items():
            if col in data.columns:
                try:
                    window_size = window_config.get("window", 3)
                    agg_func = window_config.get("agg", "mean")

                    if agg_func == "mean":
                        data[f"{col}_rolling_mean_{window_size}"] = data[col].rolling(window=window_size).mean()
                    elif agg_func == "std":
                        data[f"{col}_rolling_std_{window_size}"] = data[col].rolling(window=window_size).std()
                    elif agg_func == "sum":
                        data[f"{col}_rolling_sum_{window_size}"] = data[col].rolling(window=window_size).sum()
                    elif agg_func == "min":
                        data[f"{col}_rolling_min_{window_size}"] = data[col].rolling(window=window_size).min()
                    elif agg_func == "max":
                        data[f"{col}_rolling_max_{window_size}"] = data[col].rolling(window=window_size).max()

                    results.transformations_applied.append(f"rolling_{col}_{agg_func}_{window_size}")
                    logger.debug(f"Created rolling {agg_func} feature for {col} (window={window_size})")

                except Exception as e:
                    logger.warning(f"Failed to create rolling features for {col}: {e}")

        results.data = data
        return results

    def _create_lag_features(
        self, results: FeatureEngineeringResults, config: FeatureEngineeringConfig
    ) -> FeatureEngineeringResults:
        """Create lag features for time series data."""

        logger.debug("Creating lag features")

        data = results.data

        for col, lag_periods in config.create_lag_features.items():
            if col in data.columns:
                try:
                    for lag in lag_periods:
                        data[f"{col}_lag_{lag}"] = data[col].shift(lag)

                        results.transformations_applied.append(f"lag_{col}_{lag}")
                        logger.debug(f"Created lag feature: {col}_lag_{lag}")

                except Exception as e:
                    logger.warning(f"Failed to create lag features for {col}: {e}")

        results.data = data
        return results

    def _create_text_features(
        self, results: FeatureEngineeringResults, config: FeatureEngineeringConfig
    ) -> FeatureEngineeringResults:
        """Create text-based features."""

        logger.debug("Creating text features")

        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for text vectorization")
            return results

        data = results.data

        for col, method in config.text_vectorization.items():
            if col in data.columns:
                try:
                    # Basic text preprocessing
                    text_data = data[col].fillna("").astype(str)

                    # Create basic text features
                    data[f"{col}_length"] = text_data.str.len()
                    data[f"{col}_word_count"] = text_data.str.split().str.len()
                    data[f"{col}_char_count"] = text_data.str.replace(" ", "").str.len()

                    # Vectorization
                    if method == "tfidf":
                        vectorizer = TfidfVectorizer(
                            max_features=config.max_text_features,
                            ngram_range=config.text_ngram_range,
                            stop_words="english",
                        )
                    elif method == "count":
                        vectorizer = CountVectorizer(
                            max_features=config.max_text_features,
                            ngram_range=config.text_ngram_range,
                            stop_words="english",
                        )
                    else:
                        continue

                    # Fit and transform
                    text_features = vectorizer.fit_transform(text_data)

                    # Add vectorized features to dataframe
                    feature_names = [f"{col}_{method}_{i}" for i in range(text_features.shape[1])]
                    text_df = pd.DataFrame(text_features.toarray(), columns=feature_names, index=data.index)

                    # Concatenate with main dataframe
                    data = pd.concat([data, text_df], axis=1)

                    # Store vectorizer for future use
                    self.fitted_transformers_[f"{col}_{method}_vectorizer"] = vectorizer

                    results.transformations_applied.append(f"text_vectorization_{col}_{method}")
                    logger.debug(f"Created {method} text features for {col} ({text_features.shape[1]} features)")

                except Exception as e:
                    logger.warning(f"Failed to create text features for {col}: {e}")

        results.data = data
        return results

    def _create_polynomial_interaction_features(
        self, results: FeatureEngineeringResults, config: FeatureEngineeringConfig, target_column: Optional[str]
    ) -> FeatureEngineeringResults:
        """Create polynomial and interaction features."""

        logger.debug("Creating polynomial and interaction features")

        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for polynomial features")
            return results

        data = results.data

        # Get numeric columns for polynomial features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)

        # Limit number of features to prevent explosion
        max_features_for_poly = 10
        if len(numeric_cols) > max_features_for_poly:
            logger.warning(
                f"Too many numeric features ({len(numeric_cols)}) for polynomial features. Using first {max_features_for_poly}."
            )
            numeric_cols = numeric_cols[:max_features_for_poly]

        if len(numeric_cols) == 0:
            logger.warning("No numeric columns available for polynomial features")
            return results

        self.feature_names_ = numeric_cols.tolist()

        try:
            # Create polynomial features
            if config.create_polynomial_features:
                poly = PolynomialFeatures(
                    degree=config.polynomial_degree, include_bias=config.polynomial_include_bias, interaction_only=False
                )

                poly_features = poly.fit_transform(data[numeric_cols])

                # Get feature names
                poly_feature_names = [f"poly_{name}" for name in poly.get_feature_names_out(numeric_cols)]

                # Create DataFrame for polynomial features (excluding original features)
                n_original = len(numeric_cols)
                new_poly_features = poly_features[:, n_original:]  # Exclude original features
                new_poly_names = poly_feature_names[n_original:]

                if len(new_poly_names) > 0:
                    poly_df = pd.DataFrame(new_poly_features, columns=new_poly_names, index=data.index)
                    data = pd.concat([data, poly_df], axis=1)

                    # Store transformer
                    self.fitted_transformers_["polynomial_features"] = poly

                    results.transformations_applied.append("polynomial_features")
                    logger.debug(f"Created {len(new_poly_names)} polynomial features")

            # Create specific interaction features
            if config.create_interaction_features and config.interaction_pairs:
                for col1, col2 in config.interaction_pairs:
                    if col1 in data.columns and col2 in data.columns:
                        if pd.api.types.is_numeric_dtype(data[col1]) and pd.api.types.is_numeric_dtype(data[col2]):
                            data[f"{col1}_x_{col2}"] = data[col1] * data[col2]

                            results.transformations_applied.append(f"interaction_{col1}_{col2}")
                            logger.debug(f"Created interaction feature: {col1} x {col2}")

        except Exception as e:
            logger.warning(f"Failed to create polynomial/interaction features: {e}")

        results.data = data
        return results

    def _validate_and_cleanup_features(
        self, results: FeatureEngineeringResults, config: FeatureEngineeringConfig
    ) -> FeatureEngineeringResults:
        """Validate and cleanup engineered features."""

        logger.debug("Validating and cleaning up features")

        data = results.data
        initial_columns = len(data.columns)

        # Remove features with too many missing values
        missing_threshold = 0.95
        missing_rates = data.isnull().mean()
        high_missing_cols = missing_rates[missing_rates > missing_threshold].index.tolist()

        if high_missing_cols:
            data = data.drop(columns=high_missing_cols)
            results.dropped_features.extend(high_missing_cols)
            logger.debug(f"Dropped {len(high_missing_cols)} features with >95% missing values")

        # Remove constant features
        constant_cols = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            data = data.drop(columns=constant_cols)
            results.dropped_features.extend(constant_cols)
            logger.debug(f"Dropped {len(constant_cols)} constant features")

        # Remove highly correlated features
        if config.remove_correlated_features:
            numeric_data = data.select_dtypes(include=[np.number])

            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr().abs()

                # Find highly correlated pairs
                upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

                high_corr_cols = [
                    col for col in upper_triangle.columns if any(upper_triangle[col] > config.correlation_threshold)
                ]

                if high_corr_cols:
                    data = data.drop(columns=high_corr_cols)
                    results.dropped_features.extend(high_corr_cols)
                    logger.debug(f"Dropped {len(high_corr_cols)} highly correlated features")

        # Replace infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(data[col]).any():
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)

        results.data = data

        final_columns = len(data.columns)
        if initial_columns != final_columns:
            logger.debug(f"Feature cleanup: {initial_columns} → {final_columns} columns")

        return results

    def transform_new_data(self, data: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
        """Transform new data using fitted transformers.

        Args:
            data: New data to transform
            config: Same configuration used for fitting

        Returns:
            Transformed data
        """
        logger.info(f"Transforming new data with shape: {data.shape}")

        try:
            # Apply the same transformations but using fitted transformers
            transformed_data = data.copy()

            # Mathematical transformations (these don't need fitted transformers)
            if config.log_transform_cols:
                for col in config.log_transform_cols:
                    if col in transformed_data.columns:
                        min_val = transformed_data[col].min()
                        if min_val <= 0:
                            shift_val = abs(min_val) + 1
                            transformed_data[f"{col}_log"] = np.log1p(transformed_data[col] + shift_val)
                        else:
                            transformed_data[f"{col}_log"] = np.log1p(transformed_data[col])

            # Apply fitted transformers
            for transformer_name, transformer in self.fitted_transformers_.items():
                if "vectorizer" in transformer_name:
                    # Text vectorization
                    col_name = transformer_name.split("_")[0]
                    method = transformer_name.split("_")[1]

                    if col_name in transformed_data.columns:
                        text_data = transformed_data[col_name].fillna("").astype(str)
                        text_features = transformer.transform(text_data)

                        feature_names = [f"{col_name}_{method}_{i}" for i in range(text_features.shape[1])]
                        text_df = pd.DataFrame(
                            text_features.toarray(), columns=feature_names, index=transformed_data.index
                        )

                        transformed_data = pd.concat([transformed_data, text_df], axis=1)

                elif transformer_name == "polynomial_features":
                    # Polynomial features
                    numeric_cols = self.feature_names_ if self.feature_names_ else []
                    available_cols = [col for col in numeric_cols if col in transformed_data.columns]

                    if available_cols:
                        poly_features = transformer.transform(transformed_data[available_cols])
                        poly_feature_names = [
                            f"poly_{name}" for name in transformer.get_feature_names_out(available_cols)
                        ]

                        # Exclude original features
                        n_original = len(available_cols)
                        new_poly_features = poly_features[:, n_original:]
                        new_poly_names = poly_feature_names[n_original:]

                        if len(new_poly_names) > 0:
                            poly_df = pd.DataFrame(
                                new_poly_features, columns=new_poly_names, index=transformed_data.index
                            )
                            transformed_data = pd.concat([transformed_data, poly_df], axis=1)

            logger.info(f"Data transformation completed: {data.shape} → {transformed_data.shape}")
            return transformed_data

        except Exception as e:
            handle_and_reraise(
                e, FeatureEngineeringError, "Failed to transform new data", error_code="TRANSFORM_NEW_DATA_FAILED"
            )

    def get_feature_importance(
        self, data: pd.DataFrame, target: pd.Series, method: str = "mutual_info"
    ) -> Dict[str, float]:
        """Calculate feature importance for engineered features.

        Args:
            data: Features DataFrame
            target: Target variable
            method: Importance calculation method

        Returns:
            Dictionary of feature importances
        """
        try:
            if method == "mutual_info" and SKLEARN_AVAILABLE:
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

                numeric_data = data.select_dtypes(include=[np.number]).fillna(0)

                if len(target.unique()) <= 20:  # Classification
                    importances = mutual_info_classif(numeric_data, target, random_state=self.random_state)
                else:  # Regression
                    importances = mutual_info_regression(numeric_data, target, random_state=self.random_state)

                return dict(zip(numeric_data.columns, importances))

            elif method == "correlation":
                numeric_data = data.select_dtypes(include=[np.number])
                correlations = numeric_data.corrwith(target).abs()
                return correlations.to_dict()

            else:
                logger.warning(f"Unknown importance method: {method}")
                return {}

        except Exception as e:
            logger.warning(f"Failed to calculate feature importance: {e}")
            return {}


# Convenience functions
def create_basic_features(
    data: pd.DataFrame, target_column: Optional[str] = None, **kwargs: Any
) -> FeatureEngineeringResults:
    """Create basic engineered features with default settings.

    Args:
        data: Input DataFrame
        target_column: Target column name
        **kwargs: Additional configuration parameters

    Returns:
        Feature engineering results

    Example:
        >>> results = create_basic_features(
        ...     df,
        ...     target_column='target',
        ...     log_transform_cols=['income', 'amount']
        ... )
        >>> df_transformed = results.data
    """
    config = FeatureEngineeringConfig(**kwargs)
    engineer = FeatureEngineer()
    return engineer.engineer_features(data, config, target_column)


def create_time_series_features(
    data: pd.DataFrame,
    date_columns: List[str],
    value_columns: List[str],
    target_column: Optional[str] = None,
    **kwargs: Any,
) -> FeatureEngineeringResults:
    """Create time series specific features.

    Args:
        data: Input DataFrame
        date_columns: List of date column names
        value_columns: List of value columns for lag/rolling features
        target_column: Target column name
        **kwargs: Additional configuration parameters

    Returns:
        Feature engineering results

    Example:
        >>> results = create_time_series_features(
        ...     df,
        ...     date_columns=['timestamp'],
        ...     value_columns=['price', 'volume'],
        ...     target_column='target'
        ... )
    """
    # Configure time series specific features
    time_config = {
        "extract_date_features": date_columns,
        "create_lag_features": {col: [1, 2, 3, 7] for col in value_columns},
        "rolling_windows": {col: {"window": 7, "agg": "mean"} for col in value_columns},
    }
    time_config.update(kwargs)

    config = FeatureEngineeringConfig(**time_config)
    engineer = FeatureEngineer()
    return engineer.engineer_features(data, config, target_column)


def create_financial_features(
    data: pd.DataFrame, amount_columns: List[str], target_column: Optional[str] = None, **kwargs: Any
) -> FeatureEngineeringResults:
    """Create financial domain specific features.

    Args:
        data: Input DataFrame
        amount_columns: List of amount/value columns
        target_column: Target column name
        **kwargs: Additional configuration parameters

    Returns:
        Feature engineering results

    Example:
        >>> results = create_financial_features(
        ...     df,
        ...     amount_columns=['income', 'expenses', 'assets'],
        ...     target_column='default_risk'
        ... )
    """
    # Configure financial specific features
    financial_config = {
        "log_transform_cols": amount_columns,
        "create_ratios": [("income", "expenses"), ("assets", "liabilities")] if "expenses" in data.columns else [],
        "create_bins": dict.fromkeys(amount_columns[:3], 5),  # Limit to first 3 columns
    }
    financial_config.update(kwargs)

    config = FeatureEngineeringConfig(**financial_config)
    engineer = FeatureEngineer()
    return engineer.engineer_features(data, config, target_column)


# Export key classes and functions
__all__ = [
    "FeatureEngineeringConfig",
    "FeatureEngineeringResults",
    "FeatureEngineer",
    "create_basic_features",
    "create_time_series_features",
    "create_financial_features",
]
