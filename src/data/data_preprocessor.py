"""Enhanced data preprocessing utilities with advanced encoding and imputation.

This module provides comprehensive data preprocessing capabilities including:
- Weight of Evidence (WoE) encoding with sample weights support
- Multiple imputation strategies with custom constants
- Column-specific processing for different strategies
- Categorical mapping persistence
- Advanced scaling and encoding options
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
import warnings

from ..utils.logger import get_logger
from ..utils.timer import timer
from ..utils.io_utils import save_json, load_json

logger = get_logger(__name__)


@dataclass
class ColumnConfig:
    """Configuration for column-specific preprocessing."""

    # Missing value imputation
    missing_strategy: str = "median"  # "mean", "median", "most_frequent", "constant", "knn"
    missing_constant: Union[str, int, float] = -999999  # Value for constant strategy

    # Categorical encoding
    encoding_strategy: str = "label"  # "label", "onehot", "target", "woe"

    # Scaling
    scaling_strategy: Optional[str] = None  # "standard", "minmax", "robust"

    # Advanced options
    handle_unknown: str = "error"  # "error", "ignore", "use_encoded_value"
    unknown_value: Union[str, int, float] = -1  # Value for unknown categories

    def __post_init__(self):
        """Validate configuration."""
        valid_missing = ["mean", "median", "most_frequent", "constant", "knn"]
        valid_encoding = ["label", "onehot", "target", "woe"]
        valid_scaling = [None, "standard", "minmax", "robust"]
        valid_unknown = ["error", "ignore", "use_encoded_value"]

        if self.missing_strategy not in valid_missing:
            raise ValueError(f"missing_strategy must be one of {valid_missing}")
        if self.encoding_strategy not in valid_encoding:
            raise ValueError(f"encoding_strategy must be one of {valid_encoding}")
        if self.scaling_strategy not in valid_scaling:
            raise ValueError(f"scaling_strategy must be one of {valid_scaling}")
        if self.handle_unknown not in valid_unknown:
            raise ValueError(f"handle_unknown must be one of {valid_unknown}")


class WeightOfEvidenceEncoder:
    """Weight of Evidence encoder with sample weights support.

    Weight of Evidence measures the strength of a categorical variable's
    relationship with a binary target variable.

    WoE = ln(% of non-events / % of events)

    Example:
        >>> encoder = WeightOfEvidenceEncoder()
        >>> X_encoded = encoder.fit_transform(X['category'], y, sample_weight=weights)
        >>> encoder.save_mapping('woe_mappings.json')
    """

    def __init__(self, smoothing: float = 0.5, handle_unknown: str = "use_encoded_value"):
        """Initialize WoE encoder.

        Args:
            smoothing: Smoothing factor to prevent division by zero
            handle_unknown: How to handle unknown categories during transform
        """
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        self.woe_mapping_ = {}
        self.global_woe_ = 0.0
        self.fitted_ = False

    @timer
    def fit(self, X: pd.Series, y: pd.Series, 
           sample_weight: Optional[np.ndarray] = None) -> 'WeightOfEvidenceEncoder':
        """Fit the WoE encoder.

        Args:
            X: Categorical feature series
            y: Binary target series (0/1)
            sample_weight: Optional sample weights

        Returns:
            Fitted encoder
        """
        logger.info(f"Fitting WoE encoder for feature with {X.nunique()} unique categories")

        if sample_weight is None:
            sample_weight = np.ones(len(X))

        # Validate target is binary
        unique_targets = y.unique()
        if len(unique_targets) != 2 or not all(t in [0, 1] for t in unique_targets):
            raise ValueError("Target must be binary (0/1) for WoE encoding")

        # Create DataFrame for calculations
        df = pd.DataFrame({
            'feature': X,
            'target': y,
            'weight': sample_weight
        })

        # Calculate WoE for each category
        woe_mapping = {}

        for category in df['feature'].unique():
            if pd.isna(category):
                continue

            # Get weighted counts for this category
            cat_data = df[df['feature'] == category]

            # Events (target = 1) and non-events (target = 0) with weights
            events_weight = cat_data[cat_data['target'] == 1]['weight'].sum()
            non_events_weight = cat_data[cat_data['target'] == 0]['weight'].sum()

            # Total events and non-events with weights
            total_events_weight = df[df['target'] == 1]['weight'].sum()
            total_non_events_weight = df[df['target'] == 0]['weight'].sum()

            # Calculate percentages with smoothing
            event_rate = (events_weight + self.smoothing) / (total_events_weight + self.smoothing)
            non_event_rate = (non_events_weight + self.smoothing) / (total_non_events_weight + self.smoothing)

            # Calculate WoE
            if event_rate > 0 and non_event_rate > 0:
                woe = np.log(non_event_rate / event_rate)
            else:
                woe = 0.0

            woe_mapping[category] = woe

        # Handle missing values - assign global WoE
        total_events_weight = df[df['target'] == 1]['weight'].sum()
        total_non_events_weight = df[df['target'] == 0]['weight'].sum()

        if total_events_weight > 0 and total_non_events_weight > 0:
            global_event_rate = total_events_weight / (total_events_weight + total_non_events_weight)
            global_non_event_rate = total_non_events_weight / (total_events_weight + total_non_events_weight)
            self.global_woe_ = np.log(global_non_event_rate / global_event_rate) if global_event_rate > 0 else 0.0
        else:
            self.global_woe_ = 0.0

        self.woe_mapping_ = woe_mapping
        self.fitted_ = True

        logger.info(f"WoE encoder fitted with {len(woe_mapping)} categories")
        logger.info(f"WoE range: [{min(woe_mapping.values()):.3f}, {max(woe_mapping.values()):.3f}]")

        return self

    def transform(self, X: pd.Series) -> np.ndarray:
        """Transform categorical series using fitted WoE mapping.

        Args:
            X: Categorical feature series

        Returns:
            WoE encoded array
        """
        if not self.fitted_:
            raise ValueError("Encoder must be fitted before transform")

        # Handle different scenarios for unknown categories
        if self.handle_unknown == "error":
            unknown_cats = set(X.unique()) - set(self.woe_mapping_.keys()) - {np.nan}
            if unknown_cats:
                raise ValueError(f"Unknown categories found: {unknown_cats}")

        # Transform using mapping
        transformed = X.map(self.woe_mapping_)

        # Handle missing values and unknown categories
        if self.handle_unknown == "use_encoded_value":
            transformed = transformed.fillna(self.global_woe_)  # Unknown categories get global WoE
        elif self.handle_unknown == "ignore":
            transformed = transformed.fillna(0.0)

        # Handle any remaining NaN values
        transformed = transformed.fillna(self.global_woe_)

        return transformed.values

    def fit_transform(self, X: pd.Series, y: pd.Series,
                     sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit encoder and transform in one step.

        Args:
            X: Categorical feature series
            y: Binary target series
            sample_weight: Optional sample weights

        Returns:
            WoE encoded array
        """
        return self.fit(X, y, sample_weight).transform(X)

    def get_mapping(self) -> Dict[Any, float]:
        """Get the WoE mapping dictionary.

        Returns:
            Dictionary mapping categories to WoE values
        """
        if not self.fitted_:
            raise ValueError("Encoder must be fitted first")
        return self.woe_mapping_.copy()

    def save_mapping(self, filepath: Union[str, Path]) -> None:
        """Save WoE mapping to file.

        Args:
            filepath: Path to save the mapping
        """
        if not self.fitted_:
            raise ValueError("Encoder must be fitted before saving")

        mapping_data = {
            'woe_mapping': {str(k): v for k, v in self.woe_mapping_.items()},
            'global_woe': self.global_woe_,
            'smoothing': self.smoothing,
            'handle_unknown': self.handle_unknown
        }

        save_json(mapping_data, filepath)
        logger.info(f"WoE mapping saved to {filepath}")

    def load_mapping(self, filepath: Union[str, Path]) -> 'WeightOfEvidenceEncoder':
        """Load WoE mapping from file.

        Args:
            filepath: Path to load the mapping from

        Returns:
            Loaded encoder
        """
        mapping_data = load_json(filepath)

        self.woe_mapping_ = mapping_data['woe_mapping']
        self.global_woe_ = mapping_data['global_woe']
        self.smoothing = mapping_data['smoothing']
        self.handle_unknown = mapping_data['handle_unknown']
        self.fitted_ = True

        logger.info(f"WoE mapping loaded from {filepath}")
        return self


class AdvancedDataPreprocessor:
    """Advanced data preprocessor with column-specific strategies.

    Supports different preprocessing strategies for different columns,
    including advanced encoding methods and sample weights.

    Example:
        >>> preprocessor = AdvancedDataPreprocessor()
        >>> 
        >>> # Configure different strategies for different columns
        >>> preprocessor.set_column_config('age', ColumnConfig(
        ...     missing_strategy='median',
        ...     scaling_strategy='standard'
        ... ))
        >>> preprocessor.set_column_config('category', ColumnConfig(
        ...     encoding_strategy='woe',
        ...     missing_strategy='constant',
        ...     missing_constant='unknown'
        ... ))
        >>> 
        >>> X_processed = preprocessor.fit_transform(X_train, y_train, sample_weight=weights)
    """

    def __init__(self, 
                 default_config: Optional[ColumnConfig] = None,
                 mapping_save_dir: Optional[Union[str, Path]] = None):
        """Initialize advanced preprocessor.

        Args:
            default_config: Default configuration for all columns
            mapping_save_dir: Directory to save categorical mappings
        """
        self.default_config = default_config or ColumnConfig()
        self.column_configs = {}
        self.mapping_save_dir = Path(mapping_save_dir) if mapping_save_dir else None

        # Fitted transformers storage
        self.fitted_transformers = {}
        self.categorical_mappings = {}
        self.feature_names_ = None
        self.target_column_ = None
        self.fitted_ = False

        # Create mapping directory if specified
        if self.mapping_save_dir:
            self.mapping_save_dir.mkdir(parents=True, exist_ok=True)

    def set_column_config(self, column: str, config: ColumnConfig) -> None:
        """Set configuration for a specific column.

        Args:
            column: Column name
            config: Column configuration
        """
        self.column_configs[column] = config
        logger.info(f"Set config for column '{column}': {config}")

    def set_column_configs(self, configs: Dict[str, ColumnConfig]) -> None:
        """Set configurations for multiple columns.

        Args:
            configs: Dictionary mapping column names to configurations
        """
        self.column_configs.update(configs)
        logger.info(f"Set configs for {len(configs)} columns")

    def get_column_config(self, column: str) -> ColumnConfig:
        """Get configuration for a column (or default if not specified).

        Args:
            column: Column name

        Returns:
            Column configuration
        """
        return self.column_configs.get(column, self.default_config)

    def _weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted median."""
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumsum = np.cumsum(sorted_weights)
        cutoff = cumsum[-1] / 2.0

        median_idx = np.searchsorted(cumsum, cutoff)
        return sorted_values[median_idx]

    def _weighted_mode(self, values: np.ndarray, weights: np.ndarray) -> Any:
        """Calculate weighted mode."""
        unique_values, indices = np.unique(values, return_inverse=True)
        weighted_counts = np.bincount(indices, weights=weights)
        mode_idx = np.argmax(weighted_counts)
        return unique_values[mode_idx]

    def _impute_missing_values(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                              sample_weight: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Impute missing values using column-specific strategies.

        Args:
            X: Feature DataFrame
            y: Target series (needed for some strategies)
            sample_weight: Sample weights

        Returns:
            DataFrame with imputed values
        """
        X_imputed = X.copy()

        for column in X.columns:
            if X[column].isnull().sum() == 0:
                continue

            config = self.get_column_config(column)
            strategy = config.missing_strategy

            logger.info(f"Imputing column '{column}' using '{strategy}' strategy")

            if strategy == "mean":
                if pd.api.types.is_numeric_dtype(X[column]):
                    if sample_weight is not None:
                        # Weighted mean
                        weights_sum = sample_weight[~X[column].isnull()].sum()
                        if weights_sum > 0:
                            fill_value = np.average(X[column].dropna(), 
                                                  weights=sample_weight[~X[column].isnull()])
                        else:
                            fill_value = X[column].mean()
                    else:
                        fill_value = X[column].mean()
                    X_imputed[column] = X_imputed[column].fillna(fill_value)
                    self.fitted_transformers[f'{column}_impute_value'] = fill_value
                else:
                    logger.warning(f"Cannot use mean strategy for non-numeric column '{column}', using most_frequent")
                    fill_value = X[column].mode()[0] if len(X[column].mode()) > 0 else "unknown"
                    X_imputed[column] = X_imputed[column].fillna(fill_value)
                    self.fitted_transformers[f'{column}_impute_value'] = fill_value

            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(X[column]):
                    if sample_weight is not None:
                        # Weighted median approximation
                        fill_value = self._weighted_median(X[column].dropna().values,
                                                         sample_weight[~X[column].isnull()])
                    else:
                        fill_value = X[column].median()
                    X_imputed[column] = X_imputed[column].fillna(fill_value)
                    self.fitted_transformers[f'{column}_impute_value'] = fill_value
                else:
                    logger.warning(f"Cannot use median strategy for non-numeric column '{column}', using most_frequent")
                    fill_value = X[column].mode()[0] if len(X[column].mode()) > 0 else "unknown"
                    X_imputed[column] = X_imputed[column].fillna(fill_value)
                    self.fitted_transformers[f'{column}_impute_value'] = fill_value

            elif strategy == "most_frequent":
                if sample_weight is not None:
                    # Weighted mode
                    fill_value = self._weighted_mode(X[column].dropna().values,
                                                   sample_weight[~X[column].isnull()])
                else:
                    fill_value = X[column].mode()[0] if len(X[column].mode()) > 0 else "unknown"
                X_imputed[column] = X_imputed[column].fillna(fill_value)
                self.fitted_transformers[f'{column}_impute_value'] = fill_value

            elif strategy == "constant":
                fill_value = config.missing_constant
                X_imputed[column] = X_imputed[column].fillna(fill_value)
                self.fitted_transformers[f'{column}_impute_value'] = fill_value

            elif strategy == "knn":
                # KNN imputation (ignores sample weights for now)
                if f'{column}_knn_imputer' not in self.fitted_transformers:
                    imputer = KNNImputer(n_neighbors=5)
                    # For KNN, we need numeric data
                    if pd.api.types.is_numeric_dtype(X[column]):
                        X_imputed[column] = imputer.fit_transform(X[[column]]).ravel()
                        self.fitted_transformers[f'{column}_knn_imputer'] = imputer
                    else:
                        logger.warning(f"KNN imputation not suitable for non-numeric column '{column}', using most_frequent")
                        fill_value = X[column].mode()[0] if len(X[column].mode()) > 0 else "unknown"
                        X_imputed[column] = X_imputed[column].fillna(fill_value)
                        self.fitted_transformers[f'{column}_impute_value'] = fill_value

        return X_imputed

    def _encode_categorical(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                          sample_weight: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Encode categorical variables using column-specific strategies.

        Args:
            X: Feature DataFrame
            y: Target series (needed for target/WoE encoding)
            sample_weight: Sample weights

        Returns:
            DataFrame with encoded categorical variables
        """
        X_encoded = X.copy()
        encoded_columns = []

        for column in X.columns:
            if not pd.api.types.is_object_dtype(X[column]) and not pd.api.types.is_categorical_dtype(X[column]):
                continue

            config = self.get_column_config(column)
            strategy = config.encoding_strategy

            logger.info(f"Encoding column '{column}' using '{strategy}' strategy")

            if strategy == "label":
                encoder = LabelEncoder()
                # Handle missing values before encoding
                non_null_mask = ~X[column].isnull()
                if non_null_mask.sum() > 0:
                    encoded_values = encoder.fit_transform(X[column].dropna())
                    X_encoded[column] = X_encoded[column].astype(str)  # Convert to string first
                    X_encoded.loc[non_null_mask, column] = encoded_values
                    X_encoded[column] = pd.to_numeric(X_encoded[column], errors='coerce')

                    # Save encoder and mapping
                    self.fitted_transformers[f'{column}_label_encoder'] = encoder
                    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                    self.categorical_mappings[column] = mapping

                    # Save mapping to file if directory specified
                    if self.mapping_save_dir:
                        save_json(mapping, self.mapping_save_dir / f"{column}_label_mapping.json")

            elif strategy == "onehot":
                # One-hot encoding
                dummies = pd.get_dummies(X[column], prefix=column, dummy_na=False)
                encoded_columns.extend(dummies.columns.tolist())

                # Remove original column and add dummy columns
                X_encoded = X_encoded.drop(columns=[column])
                X_encoded = pd.concat([X_encoded, dummies], axis=1)

                # Save column information for transform
                self.fitted_transformers[f'{column}_onehot_columns'] = dummies.columns.tolist()
                self.categorical_mappings[column] = {
                    'type': 'onehot',
                    'columns': dummies.columns.tolist(),
                    'categories': X[column].unique().tolist()
                }

                # Save mapping to file if directory specified
                if self.mapping_save_dir:
                    save_json(self.categorical_mappings[column], 
                             self.mapping_save_dir / f"{column}_onehot_mapping.json")

            elif strategy == "target":
                if y is None:
                    raise ValueError("Target encoding requires target variable y")

                # Target encoding (mean of target for each category)
                target_means = {}

                for category in X[column].unique():
                    if pd.isna(category):
                        continue
                    mask = X[column] == category
                    if sample_weight is not None:
                        target_mean = np.average(y[mask], weights=sample_weight[mask])
                    else:
                        target_mean = y[mask].mean()
                    target_means[category] = target_mean

                # Global mean for unknown categories
                global_mean = np.average(y, weights=sample_weight) if sample_weight is not None else y.mean()

                # Apply encoding
                X_encoded[column] = X[column].map(target_means).fillna(global_mean)

                # Save mapping
                target_means['__global_mean__'] = global_mean
                self.fitted_transformers[f'{column}_target_encoder'] = target_means
                self.categorical_mappings[column] = target_means

                # Save mapping to file if directory specified
                if self.mapping_save_dir:
                    save_json(target_means, self.mapping_save_dir / f"{column}_target_mapping.json")

            elif strategy == "woe":
                if y is None:
                    raise ValueError("WoE encoding requires target variable y")

                # Weight of Evidence encoding
                woe_encoder = WeightOfEvidenceEncoder(
                    handle_unknown=config.handle_unknown
                )
                X_encoded[column] = woe_encoder.fit_transform(X[column], y, sample_weight)

                # Save encoder and mapping
                self.fitted_transformers[f'{column}_woe_encoder'] = woe_encoder
                self.categorical_mappings[column] = woe_encoder.get_mapping()

                # Save mapping to file if directory specified
                if self.mapping_save_dir:
                    woe_encoder.save_mapping(self.mapping_save_dir / f"{column}_woe_mapping.json")

        return X_encoded

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using column-specific strategies.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with scaled features
        """
        X_scaled = X.copy()

        for column in X.columns:
            if not pd.api.types.is_numeric_dtype(X[column]):
                continue

            config = self.get_column_config(column)
            strategy = config.scaling_strategy

            if strategy is None:
                continue

            logger.info(f"Scaling column '{column}' using '{strategy}' strategy")

            if strategy == "standard":
                scaler = StandardScaler()
            elif strategy == "minmax":
                scaler = MinMaxScaler()
            elif strategy == "robust":
                scaler = RobustScaler()
            else:
                continue

            # Fit and transform
            X_scaled[column] = scaler.fit_transform(X[[column]]).ravel()

            # Save scaler
            self.fitted_transformers[f'{column}_{strategy}_scaler'] = scaler

        return X_scaled

    @timer
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
           sample_weight: Optional[np.ndarray] = None) -> 'AdvancedDataPreprocessor':
        """Fit the preprocessor on training data.

        Args:
            X: Training features
            y: Training target (needed for target/WoE encoding)
            sample_weight: Training sample weights

        Returns:
            Fitted preprocessor
        """
        logger.info(f"Fitting advanced preprocessor on {X.shape[0]} samples, {X.shape[1]} features")

        X_work = X.copy()

        # Store feature names and target info
        self.feature_names_ = X.columns.tolist()
        if y is not None:
            self.target_column_ = y.name

        # Step 1: Handle missing values
        X_work = self._impute_missing_values(X_work, y, sample_weight)

        # Step 2: Encode categorical variables
        X_work = self._encode_categorical(X_work, y, sample_weight)

        # Step 3: Scale features
        X_work = self._scale_features(X_work)

        self.fitted_ = True
        logger.info("Advanced preprocessor fitted successfully")
        logger.info(f"Output shape: {X_work.shape}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor.

        Args:
            X: Data to transform

        Returns:
            Transformed DataFrame
        """
        if not self.fitted_:
            raise ValueError("Preprocessor must be fitted before transform")

        logger.info(f"Transforming data: {X.shape[0]} samples, {X.shape[1]} features")

        X_work = X.copy()

        # Step 1: Handle missing values
        for column in X.columns:
            if column not in self.feature_names_:
                continue

            config = self.get_column_config(column)

            if X[column].isnull().sum() > 0:
                if f'{column}_impute_value' in self.fitted_transformers:
                    fill_value = self.fitted_transformers[f'{column}_impute_value']
                    X_work[column] = X_work[column].fillna(fill_value)
                elif f'{column}_knn_imputer' in self.fitted_transformers:
                    imputer = self.fitted_transformers[f'{column}_knn_imputer']
                    X_work[column] = imputer.transform(X_work[[column]]).ravel()

        # Step 2: Encode categorical variables
        columns_to_drop = []
        columns_to_add = {}

        for column in X.columns:
            if column not in self.feature_names_:
                continue

            config = self.get_column_config(column)

            if pd.api.types.is_object_dtype(X[column]) or pd.api.types.is_categorical_dtype(X[column]):
                if config.encoding_strategy == "label":
                    if f'{column}_label_encoder' in self.fitted_transformers:
                        encoder = self.fitted_transformers[f'{column}_label_encoder']
                        # Handle unknown categories
                        try:
                            X_work[column] = encoder.transform(X[column].fillna('__missing__'))
                        except ValueError:
                            # Handle unknown categories
                            if config.handle_unknown == "use_encoded_value":
                                # Map known categories, use unknown_value for unknown
                                mapping = self.categorical_mappings.get(column, {})
                                X_work[column] = X[column].map(mapping).fillna(config.unknown_value)
                            else:
                                raise ValueError(f"Unknown categories found in column '{column}'")

                elif config.encoding_strategy == "onehot":
                    if f'{column}_onehot_columns' in self.fitted_transformers:
                        expected_columns = self.fitted_transformers[f'{column}_onehot_columns']
                        dummies = pd.get_dummies(X[column], prefix=column, dummy_na=False)

                        # Ensure all expected columns are present
                        for col in expected_columns:
                            if col not in dummies.columns:
                                dummies[col] = 0

                        # Keep only expected columns
                        dummies = dummies[expected_columns]

                        columns_to_drop.append(column)
                        for col in expected_columns:
                            columns_to_add[col] = dummies[col]

                elif config.encoding_strategy == "target":
                    if f'{column}_target_encoder' in self.fitted_transformers:
                        mapping = self.fitted_transformers[f'{column}_target_encoder']
                        global_mean = mapping.get('__global_mean__', 0)
                        X_work[column] = X[column].map(mapping).fillna(global_mean)

                elif config.encoding_strategy == "woe":
                    if f'{column}_woe_encoder' in self.fitted_transformers:
                        woe_encoder = self.fitted_transformers[f'{column}_woe_encoder']
                        X_work[column] = woe_encoder.transform(X[column])

        # Drop and add columns for one-hot encoding
        X_work = X_work.drop(columns=columns_to_drop)
        for col_name, col_data in columns_to_add.items():
            X_work[col_name] = col_data

        # Step 3: Scale features
        for column in X_work.columns:
            if not pd.api.types.is_numeric_dtype(X_work[column]):
                continue

            config = self.get_column_config(column)
            strategy = config.scaling_strategy

            if strategy and f'{column}_{strategy}_scaler' in self.fitted_transformers:
                scaler = self.fitted_transformers[f'{column}_{strategy}_scaler']
                X_work[column] = scaler.transform(X_work[[column]]).ravel()

        logger.info(f"Transformation completed: {X_work.shape}")
        return X_work

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                     sample_weight: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Fit preprocessor and transform data in one step.

        Args:
            X: Training features
            y: Training target
            sample_weight: Training sample weights

        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y, sample_weight).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation.

        Returns:
            List of output feature names
        """
        if not self.fitted_:
            raise ValueError("Preprocessor must be fitted first")

        # This would need to be implemented based on the transformations applied
        # For now, return original feature names (simplified)
        return self.feature_names_

    def save_mappings(self, base_path: Union[str, Path]) -> None:
        """Save all categorical mappings to files.

        Args:
            base_path: Base directory path to save mappings
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        for column, mapping in self.categorical_mappings.items():
            save_path = base_path / f"{column}_mapping.json"
            save_json(mapping, save_path)
            logger.info(f"Saved mapping for column '{column}' to {save_path}")

    def load_mappings(self, base_path: Union[str, Path]) -> None:
        """Load categorical mappings from files.

        Args:
            base_path: Base directory path to load mappings from
        """
        base_path = Path(base_path)

        for mapping_file in base_path.glob("*_mapping.json"):
            column = mapping_file.stem.replace("_mapping", "")
            mapping = load_json(mapping_file)
            self.categorical_mappings[column] = mapping
            logger.info(f"Loaded mapping for column '{column}' from {mapping_file}")


# Convenience functions for quick preprocessing
def preprocess_data(X: pd.DataFrame, 
                   y: Optional[pd.Series] = None,
                   sample_weight: Optional[np.ndarray] = None,
                   column_configs: Optional[Dict[str, ColumnConfig]] = None,
                   default_config: Optional[ColumnConfig] = None) -> Tuple[pd.DataFrame, AdvancedDataPreprocessor]:
    """Quick data preprocessing with advanced options.

    Args:
        X: Feature DataFrame
        y: Target series
        sample_weight: Sample weights
        column_configs: Dictionary of column-specific configurations
        default_config: Default configuration for all columns

    Returns:
        Tuple of (processed_dataframe, fitted_preprocessor)

    Example:
        >>> configs = {
        ...     'age': ColumnConfig(missing_strategy='median', scaling_strategy='standard'),
        ...     'category': ColumnConfig(encoding_strategy='woe', missing_strategy='constant', missing_constant='unknown')
        ... }
        >>> X_processed, preprocessor = preprocess_data(X, y, sample_weight=weights, column_configs=configs)
    """
    preprocessor = AdvancedDataPreprocessor(default_config=default_config)

    if column_configs:
        preprocessor.set_column_configs(column_configs)

    X_processed = preprocessor.fit_transform(X, y, sample_weight)

    return X_processed, preprocessor


def create_woe_encoding(X_cat: pd.Series, y: pd.Series, 
                       sample_weight: Optional[np.ndarray] = None,
                       save_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, WeightOfEvidenceEncoder]:
    """Create Weight of Evidence encoding for a categorical feature.

    Args:
        X_cat: Categorical feature series
        y: Binary target series
        sample_weight: Optional sample weights
        save_path: Optional path to save the WoE mapping

    Returns:
        Tuple of (woe_encoded_values, fitted_encoder)

    Example:
        >>> woe_values, encoder = create_woe_encoding(X['category'], y, sample_weight=weights)
        >>> print(f"WoE range: [{woe_values.min():.3f}, {woe_values.max():.3f}]")
    """
    encoder = WeightOfEvidenceEncoder()
    woe_values = encoder.fit_transform(X_cat, y, sample_weight)

    if save_path:
        encoder.save_mapping(save_path)

    return woe_values, encoder


# Legacy compatibility - update existing split_data function to include sample weights
def split_data(X: pd.DataFrame, y: pd.Series,
              test_size: float = 0.2, 
              valid_size: float = 0.2,
              stratify: bool = True,
              random_state: int = 42,
              sample_weight: Optional[np.ndarray] = None) -> Union[
                  Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series],
                  Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, 
                        np.ndarray, np.ndarray, np.ndarray]
              ]:
    """Split data into train/validation/test sets with optional sample weights.

    Args:
        X: Feature DataFrame
        y: Target series
        test_size: Proportion for test set
        valid_size: Proportion for validation set (from remaining data after test split)
        stratify: Whether to stratify splits
        random_state: Random state for reproducibility
        sample_weight: Optional sample weights to split accordingly

    Returns:
        If sample_weight is None: (X_train, X_valid, X_test, y_train, y_valid, y_test)
        If sample_weight provided: (..., weights_train, weights_valid, weights_test)
    """
    from sklearn.model_selection import train_test_split

    # First split: separate test set
    stratify_first = y if stratify else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, 
        stratify=stratify_first, 
        random_state=random_state
    )

    if sample_weight is not None:
        # Get indices for weight splitting
        temp_indices = X_temp.index
        test_indices = X_test.index

        weights_temp = sample_weight[temp_indices]
        weights_test = sample_weight[test_indices]

    # Second split: separate train and validation from temp
    valid_size_adjusted = valid_size / (1 - test_size)  # Adjust for remaining data
    stratify_second = y_temp if stratify else None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, 
        test_size=valid_size_adjusted,
        stratify=stratify_second,
        random_state=random_state
    )

    if sample_weight is not None:
        # Split weights accordingly
        train_indices = X_train.index
        valid_indices = X_valid.index

        weights_train = sample_weight[train_indices]
        weights_valid = sample_weight[valid_indices]

        logger.info(f"Data split with weights: Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")

        return (X_train, X_valid, X_test, y_train, y_valid, y_test,
                weights_train, weights_valid, weights_test)
    else:
        logger.info(f"Data split: Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")

        return X_train, X_valid, X_test, y_train, y_valid, y_test
