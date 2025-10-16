"""Enhanced data loading utilities compatible with advanced preprocessing.

This module provides data loading functions that work seamlessly with
the enhanced preprocessing system, including sample weights handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import warnings

from ..utils.logger import get_logger
from ..utils.timer import timer

logger = get_logger(__name__)


@timer
def load_csv(filepath: Union[str, Path], 
            sample_weight_col: Optional[str] = None,
            target_col: Optional[str] = None,
            **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series, np.ndarray]]:
    """Load CSV file with optional automatic separation of features, target, and weights.

    Args:
        filepath: Path to CSV file
        sample_weight_col: Name of sample weight column (optional)
        target_col: Name of target column (optional) 
        **kwargs: Additional arguments for pd.read_csv()

    Returns:
        If target_col and sample_weight_col are None: DataFrame
        If target_col provided: (features_df, target_series)
        If sample_weight_col provided: (features_df, target_series, weights_array)

    Example:
        >>> df = load_csv('data.csv')
        >>> X, y, weights = load_csv('data.csv', target_col='is_fraud', sample_weight_col='weight')
    """
    logger.info(f"Loading CSV file: {filepath}")

    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Display basic info
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # If no special columns specified, return full DataFrame
        if target_col is None and sample_weight_col is None:
            return df

        # Extract target column if specified
        if target_col is not None:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")

            y = df[target_col]
            X = df.drop(columns=[target_col])

            logger.info(f"Target column '{target_col}': {y.nunique()} unique values")
            if pd.api.types.is_numeric_dtype(y):
                logger.info(f"Target statistics: mean={y.mean():.3f}, std={y.std():.3f}")
            else:
                logger.info(f"Target value counts: {y.value_counts().to_dict()}")
        else:
            X = df
            y = None

        # Extract sample weights if specified
        if sample_weight_col is not None:
            if sample_weight_col not in df.columns:
                raise ValueError(f"Sample weight column '{sample_weight_col}' not found in data")

            weights = df[sample_weight_col].values
            if target_col is None:
                X = df.drop(columns=[sample_weight_col])
            else:
                X = X.drop(columns=[sample_weight_col])

            logger.info(f"Sample weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")

            # Return appropriate tuple
            if y is not None:
                return X, y, weights
            else:
                return X, weights

        # Return X and y if only target specified
        return X, y

    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise


def load_multiple_files(file_paths: Dict[str, Union[str, Path]],
                       **kwargs) -> Dict[str, pd.DataFrame]:
    """Load multiple CSV files.

    Args:
        file_paths: Dictionary mapping names to file paths
        **kwargs: Arguments passed to load_csv

    Returns:
        Dictionary mapping names to loaded DataFrames

    Example:
        >>> files = {'train': 'train.csv', 'test': 'test.csv'}
        >>> datasets = load_multiple_files(files)
    """
    logger.info(f"Loading {len(file_paths)} files")

    datasets = {}
    for name, path in file_paths.items():
        logger.info(f"Loading {name} from {path}")
        datasets[name] = load_csv(path, **kwargs)

    return datasets


def detect_data_types(df: pd.DataFrame, 
                     categorical_threshold: int = 50) -> Dict[str, list]:
    """Detect and categorize column data types for preprocessing.

    Args:
        df: DataFrame to analyze
        categorical_threshold: Max unique values to consider categorical

    Returns:
        Dictionary with 'numeric', 'categorical', and 'datetime' column lists

    Example:
        >>> types = detect_data_types(df)
        >>> print(f"Numeric: {types['numeric']}")
    """
    logger.info(f"Detecting data types for {len(df.columns)} columns")

    numeric_cols = []
    categorical_cols = []
    datetime_cols = []

    for col in df.columns:
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        # Check if numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's really categorical (few unique values)
            if df[col].nunique() <= categorical_threshold:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        # Check if object/string type
        else:
            categorical_cols.append(col)

    result = {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }

    logger.info(f"Data types detected:")
    logger.info(f"  Numeric: {len(numeric_cols)} columns")
    logger.info(f"  Categorical: {len(categorical_cols)} columns") 
    logger.info(f"  Datetime: {len(datetime_cols)} columns")

    return result


def create_sample_weights(y: pd.Series, 
                         strategy: str = 'balanced',
                         custom_weights: Optional[Dict] = None) -> np.ndarray:
    """Create sample weights for imbalanced data.

    Args:
        y: Target variable
        strategy: Weighting strategy ('balanced', 'sqrt', 'log', 'custom')
        custom_weights: Custom weights dictionary for 'custom' strategy

    Returns:
        Array of sample weights

    Example:
        >>> weights = create_sample_weights(y, strategy='balanced')
        >>> # Use with preprocessing
        >>> X_processed = preprocessor.fit_transform(X, y, sample_weight=weights)
    """
    logger.info(f"Creating sample weights using '{strategy}' strategy")

    if strategy == 'balanced':
        # Inverse frequency weighting
        class_counts = y.value_counts()
        total_samples = len(y)
        n_classes = len(class_counts)

        weights = np.zeros(len(y))
        for class_label in class_counts.index:
            class_weight = total_samples / (n_classes * class_counts[class_label])
            weights[y == class_label] = class_weight

    elif strategy == 'sqrt':
        # Square root of inverse frequency
        class_counts = y.value_counts()
        total_samples = len(y)

        weights = np.zeros(len(y))
        for class_label in class_counts.index:
            class_weight = np.sqrt(total_samples / class_counts[class_label])
            weights[y == class_label] = class_weight

    elif strategy == 'log':
        # Logarithmic weighting
        class_counts = y.value_counts()
        total_samples = len(y)

        weights = np.zeros(len(y))
        for class_label in class_counts.index:
            class_weight = np.log(total_samples / class_counts[class_label])
            weights[y == class_label] = max(class_weight, 1.0)  # Ensure positive

    elif strategy == 'custom':
        if custom_weights is None:
            raise ValueError("custom_weights must be provided for 'custom' strategy")

        weights = np.zeros(len(y))
        for class_label, weight in custom_weights.items():
            weights[y == class_label] = weight

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    logger.info(f"Sample weights created:")
    logger.info(f"  Range: [{weights.min():.3f}, {weights.max():.3f}]")
    logger.info(f"  Mean: {weights.mean():.3f}")

    unique_classes = y.unique()
    for class_label in unique_classes:
        class_weight = weights[y == class_label][0]  # All samples of same class have same weight
        class_count = (y == class_label).sum()
        logger.info(f"  Class {class_label}: weight={class_weight:.3f}, count={class_count}")

    return weights


def validate_data_quality(df: pd.DataFrame, 
                         max_missing_ratio: float = 0.5,
                         min_samples: int = 100) -> Dict[str, Any]:
    """Validate data quality and provide recommendations.

    Args:
        df: DataFrame to validate
        max_missing_ratio: Maximum allowed missing value ratio per column
        min_samples: Minimum required samples

    Returns:
        Dictionary with validation results and recommendations

    Example:
        >>> quality_report = validate_data_quality(df)
        >>> print(quality_report['recommendations'])
    """
    logger.info("Validating data quality")

    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': [],
        'statistics': {}
    }

    # Check minimum samples
    if len(df) < min_samples:
        validation_results['errors'].append(
            f"Insufficient samples: {len(df)} < {min_samples}"
        )
        validation_results['valid'] = False

    # Check missing values
    missing_ratios = df.isnull().sum() / len(df)
    high_missing = missing_ratios[missing_ratios > max_missing_ratio]

    if len(high_missing) > 0:
        validation_results['warnings'].append(
            f"High missing value ratio in columns: {high_missing.index.tolist()}"
        )
        validation_results['recommendations'].append(
            "Consider dropping columns with high missing ratios or using advanced imputation"
        )

    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        validation_results['warnings'].append(
            f"Constant columns found: {constant_cols}"
        )
        validation_results['recommendations'].append(
            "Remove constant columns as they provide no information"
        )

    # Check for high cardinality categorical columns
    high_card_cols = []
    for col in df.columns:
        if (pd.api.types.is_object_dtype(df[col]) and 
            df[col].nunique() > len(df) * 0.8):
            high_card_cols.append(col)

    if high_card_cols:
        validation_results['warnings'].append(
            f"High cardinality categorical columns: {high_card_cols}"
        )
        validation_results['recommendations'].append(
            "Consider grouping rare categories or using different encoding strategies"
        )

    # Collect statistics
    validation_results['statistics'] = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'missing_values_total': df.isnull().sum().sum(),
        'missing_ratio_mean': missing_ratios.mean(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_rows': df.duplicated().sum()
    }

    if validation_results['valid'] and len(validation_results['warnings']) == 0:
        logger.info("✅ Data quality validation passed")
    else:
        logger.warning(f"⚠️ Data quality issues found: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings")

    return validation_results


def load_and_validate(filepath: Union[str, Path],
                     target_col: Optional[str] = None,
                     sample_weight_col: Optional[str] = None,
                     validate_quality: bool = True,
                     **kwargs) -> Dict[str, Any]:
    """Load data and perform quality validation in one step.

    Args:
        filepath: Path to data file
        target_col: Target column name
        sample_weight_col: Sample weight column name
        validate_quality: Whether to perform quality validation
        **kwargs: Additional arguments for load_csv

    Returns:
        Dictionary containing loaded data and validation results

    Example:
        >>> result = load_and_validate('data.csv', target_col='fraud', validate_quality=True)
        >>> X, y = result['features'], result['target']
    """
    logger.info(f"Loading and validating data from {filepath}")

    # Load data
    loaded_data = load_csv(filepath, target_col=target_col, 
                          sample_weight_col=sample_weight_col, **kwargs)

    # Unpack loaded data
    if isinstance(loaded_data, tuple):
        if len(loaded_data) == 2:
            if target_col:
                X, y = loaded_data
                weights = None
            else:
                X, weights = loaded_data
                y = None
        elif len(loaded_data) == 3:
            X, y, weights = loaded_data
        else:
            X = loaded_data[0]
            y = weights = None
    else:
        X = loaded_data
        y = weights = None

    result = {
        'features': X,
        'target': y,
        'sample_weights': weights,
        'raw_data': loaded_data if not isinstance(loaded_data, tuple) else None
    }

    # Validate quality if requested
    if validate_quality:
        validation_results = validate_data_quality(X if y is None else pd.concat([X, pd.Series(y, name='target')], axis=1))
        result['validation'] = validation_results

        # Print recommendations
        if validation_results['recommendations']:
            logger.info("💡 Recommendations:")
            for rec in validation_results['recommendations']:
                logger.info(f"   {rec}")

    # Detect data types
    data_types = detect_data_types(X)
    result['data_types'] = data_types

    return result


# Convenience function for fraud detection datasets
def load_fraud_dataset(filepath: Union[str, Path],
                      target_col: str = 'is_fraud',
                      create_weights: bool = True,
                      weight_strategy: str = 'balanced') -> Dict[str, Any]:
    """Load fraud detection dataset with automatic weight creation.

    Args:
        filepath: Path to fraud dataset
        target_col: Name of fraud target column
        create_weights: Whether to create sample weights automatically
        weight_strategy: Strategy for weight creation

    Returns:
        Dictionary with features, target, weights, and metadata

    Example:
        >>> fraud_data = load_fraud_dataset('fraud_data.csv')
        >>> X, y, weights = fraud_data['features'], fraud_data['target'], fraud_data['weights']
    """
    logger.info(f"Loading fraud detection dataset from {filepath}")

    # Load basic data
    result = load_and_validate(filepath, target_col=target_col, validate_quality=True)

    X = result['features']
    y = result['target']

    # Create sample weights if requested
    if create_weights:
        weights = create_sample_weights(y, strategy=weight_strategy)
        result['sample_weights'] = weights
        result['weights'] = weights  # Alias for convenience

    # Add fraud-specific statistics
    fraud_stats = {
        'fraud_rate': y.mean(),
        'fraud_count': y.sum(),
        'non_fraud_count': (y == 0).sum(),
        'imbalance_ratio': (y == 0).sum() / y.sum() if y.sum() > 0 else float('inf')
    }

    result['fraud_statistics'] = fraud_stats

    logger.info(f"📊 Fraud Dataset Statistics:")
    logger.info(f"   Total samples: {len(y)}")
    logger.info(f"   Fraud cases: {fraud_stats['fraud_count']} ({fraud_stats['fraud_rate']:.3%})")
    logger.info(f"   Non-fraud cases: {fraud_stats['non_fraud_count']}")
    logger.info(f"   Imbalance ratio: {fraud_stats['imbalance_ratio']:.1f}:1")

    return result
