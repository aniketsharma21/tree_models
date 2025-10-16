"""Enhanced data processing module for tree-based models.

This module provides comprehensive data loading and preprocessing capabilities
optimized for fraud detection and imbalanced datasets with sample weights support.

Key Features:
- Advanced data loading with automatic type detection
- Weight of Evidence (WoE) encoding with sample weights
- Column-specific preprocessing strategies
- Custom constant imputation (e.g., -999999)
- Categorical mapping persistence
- Sample weights integration throughout
- Data quality validation
- Fraud dataset utilities

Basic Usage:
    >>> from tree_model_helper.data import load_csv, ColumnConfig, preprocess_data
    >>> 
    >>> # Load data with automatic separation
    >>> X, y, weights = load_csv('fraud_data.csv', target_col='is_fraud', 
    ...                          sample_weight_col='weight')
    >>> 
    >>> # Configure preprocessing
    >>> configs = {
    ...     'age': ColumnConfig(missing_strategy='median', scaling_strategy='standard'),
    ...     'category': ColumnConfig(encoding_strategy='woe', missing_constant='Unknown')
    ... }
    >>> 
    >>> # Preprocess with sample weights
    >>> X_processed, preprocessor = preprocess_data(X, y, sample_weight=weights, 
    ...                                           column_configs=configs)

Advanced Usage:
    >>> from tree_model_helper.data import AdvancedDataPreprocessor, WeightOfEvidenceEncoder
    >>> 
    >>> # Advanced preprocessor with mapping persistence
    >>> preprocessor = AdvancedDataPreprocessor(mapping_save_dir='mappings/')
    >>> preprocessor.set_column_config('income', ColumnConfig(
    ...     missing_strategy='constant', 
    ...     missing_constant=-999999,
    ...     scaling_strategy='robust'
    ... ))
    >>> 
    >>> X_processed = preprocessor.fit_transform(X, y, sample_weights)
    >>> 
    >>> # Save mappings for production
    >>> preprocessor.save_mappings('production_mappings/')
"""

from .data_loader import (
    # Basic loading functions
    load_csv,
    load_multiple_files,
    load_and_validate,

    # Fraud-specific utilities
    load_fraud_dataset,

    # Data analysis utilities
    detect_data_types,
    validate_data_quality,
    create_sample_weights,
)

from .data_preprocessor import (
    # Main preprocessing classes
    AdvancedDataPreprocessor,
    WeightOfEvidenceEncoder,
    ColumnConfig,

    # Convenience functions
    preprocess_data,
    create_woe_encoding,
    split_data,
)

# Version and metadata
__version__ = '0.2.0'
__author__ = 'Tree Model Helper'

# Main exports for easy access
__all__ = [
    # Loading functions
    'load_csv',
    'load_multiple_files', 
    'load_and_validate',
    'load_fraud_dataset',

    # Data analysis
    'detect_data_types',
    'validate_data_quality',
    'create_sample_weights',

    # Preprocessing classes
    'AdvancedDataPreprocessor',
    'WeightOfEvidenceEncoder',
    'ColumnConfig',

    # Preprocessing functions
    'preprocess_data',
    'create_woe_encoding',
    'split_data',
]

# Quick access functions
def quick_load_and_preprocess(filepath: str,
                             target_col: str,
                             sample_weight_col: str = None,
                             woe_columns: list = None,
                             constant_columns: dict = None,
                             scaling_columns: list = None) -> tuple:
    """Quick load and preprocess in one function call.

    Args:
        filepath: Path to data file
        target_col: Target column name
        sample_weight_col: Sample weight column name (optional)
        woe_columns: Columns to apply WoE encoding
        constant_columns: Dict of column->constant for missing value imputation
        scaling_columns: Columns to apply standard scaling

    Returns:
        Tuple of (X_processed, y, sample_weights, preprocessor)

    Example:
        >>> X, y, weights, preprocessor = quick_load_and_preprocess(
        ...     'fraud_data.csv',
        ...     target_col='is_fraud',
        ...     woe_columns=['category', 'region'],
        ...     constant_columns={'income': -999999},
        ...     scaling_columns=['age', 'amount']
        ... )
    """
    # Load data
    result = load_and_validate(filepath, target_col=target_col, 
                              sample_weight_col=sample_weight_col)

    X = result['features']
    y = result['target']
    weights = result.get('sample_weights')

    # Create sample weights if not provided
    if weights is None and y is not None:
        weights = create_sample_weights(y, strategy='balanced')

    # Configure column-specific processing
    column_configs = {}

    # WoE encoding columns
    if woe_columns:
        for col in woe_columns:
            column_configs[col] = ColumnConfig(
                encoding_strategy='woe',
                missing_strategy='constant',
                missing_constant='Unknown'
            )

    # Constant imputation columns
    if constant_columns:
        for col, constant in constant_columns.items():
            if col in column_configs:
                column_configs[col].missing_strategy = 'constant'
                column_configs[col].missing_constant = constant
            else:
                column_configs[col] = ColumnConfig(
                    missing_strategy='constant',
                    missing_constant=constant
                )

    # Scaling columns
    if scaling_columns:
        for col in scaling_columns:
            if col in column_configs:
                column_configs[col].scaling_strategy = 'standard'
            else:
                column_configs[col] = ColumnConfig(
                    missing_strategy='median',
                    scaling_strategy='standard'
                )

    # Preprocess
    X_processed, preprocessor = preprocess_data(
        X, y, sample_weight=weights, column_configs=column_configs
    )

    return X_processed, y, weights, preprocessor


def create_fraud_pipeline(filepath: str,
                         target_col: str = 'is_fraud',
                         categorical_woe: bool = True,
                         balance_weights: bool = True) -> dict:
    """Create a complete fraud detection data pipeline.

    Args:
        filepath: Path to fraud dataset
        target_col: Target column name
        categorical_woe: Use WoE encoding for categorical variables
        balance_weights: Create balanced sample weights

    Returns:
        Dictionary with processed data and metadata

    Example:
        >>> pipeline = create_fraud_pipeline('fraud_data.csv')
        >>> X_train, y_train, weights = pipeline['train_data']
        >>> preprocessor = pipeline['preprocessor']
    """
    # Load fraud dataset
    fraud_data = load_fraud_dataset(filepath, target_col=target_col,
                                   create_weights=balance_weights)

    X = fraud_data['features']
    y = fraud_data['target']
    weights = fraud_data.get('sample_weights')
    data_types = fraud_data['data_types']

    # Configure preprocessing based on data types
    column_configs = {}

    # Configure categorical columns
    for col in data_types['categorical']:
        if categorical_woe:
            column_configs[col] = ColumnConfig(
                encoding_strategy='woe',
                missing_strategy='constant',
                missing_constant='Unknown',
                handle_unknown='use_encoded_value'
            )
        else:
            column_configs[col] = ColumnConfig(
                encoding_strategy='target',
                missing_strategy='most_frequent'
            )

    # Configure numeric columns
    for col in data_types['numeric']:
        # Use different strategies based on column characteristics
        if 'amount' in col.lower() or 'value' in col.lower():
            # Amount columns - use constant for missing, robust scaling
            column_configs[col] = ColumnConfig(
                missing_strategy='constant',
                missing_constant=-999999,
                scaling_strategy='robust'
            )
        elif 'age' in col.lower() or 'days' in col.lower():
            # Age/time columns - use median, standard scaling
            column_configs[col] = ColumnConfig(
                missing_strategy='median',
                scaling_strategy='standard'
            )
        else:
            # Default numeric processing
            column_configs[col] = ColumnConfig(
                missing_strategy='median',
                scaling_strategy='robust'
            )

    # Create preprocessor with mapping persistence
    preprocessor = AdvancedDataPreprocessor(
        mapping_save_dir=f'fraud_mappings_{target_col}'
    )
    preprocessor.set_column_configs(column_configs)

    # Split data with sample weights
    if weights is not None:
        X_train, X_valid, X_test, y_train, y_valid, y_test, w_train, w_valid, w_test = split_data(
            X, y, test_size=0.2, valid_size=0.2, stratify=True, sample_weight=weights
        )
    else:
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
            X, y, test_size=0.2, valid_size=0.2, stratify=True
        )
        w_train = w_valid = w_test = None

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train, y_train, w_train)
    X_valid_processed = preprocessor.transform(X_valid)
    X_test_processed = preprocessor.transform(X_test)

    # Return complete pipeline
    pipeline = {
        'train_data': (X_train_processed, y_train, w_train),
        'valid_data': (X_valid_processed, y_valid, w_valid),
        'test_data': (X_test_processed, y_test, w_test),
        'preprocessor': preprocessor,
        'original_data': (X, y, weights),
        'fraud_statistics': fraud_data['fraud_statistics'],
        'data_types': data_types,
        'validation_results': fraud_data.get('validation'),
        'column_configs': column_configs
    }

    print(f"🎯 Fraud Detection Pipeline Created:")
    print(f"   Training samples: {len(X_train)} (fraud rate: {y_train.mean():.3%})")
    print(f"   Validation samples: {len(X_valid)} (fraud rate: {y_valid.mean():.3%})")
    print(f"   Test samples: {len(X_test)} (fraud rate: {y_test.mean():.3%})")
    print(f"   Features: {X_train_processed.shape[1]} (after preprocessing)")
    print(f"   Sample weights: {'✅ Created' if balance_weights else '❌ Not used'}")
    print(f"   WoE encoding: {'✅ Enabled' if categorical_woe else '❌ Disabled'}")

    return pipeline


# Configuration shortcuts for common use cases
class CommonConfigs:
    """Pre-defined configuration templates for common scenarios."""

    @staticmethod
    def fraud_detection_numeric():
        """Configuration for numeric features in fraud detection."""
        return ColumnConfig(
            missing_strategy='constant',
            missing_constant=-999999,
            scaling_strategy='robust'
        )

    @staticmethod  
    def fraud_detection_categorical():
        """Configuration for categorical features in fraud detection."""
        return ColumnConfig(
            encoding_strategy='woe',
            missing_strategy='constant', 
            missing_constant='Unknown',
            handle_unknown='use_encoded_value'
        )

    @staticmethod
    def high_cardinality_categorical():
        """Configuration for high cardinality categorical features."""
        return ColumnConfig(
            encoding_strategy='target',
            missing_strategy='most_frequent',
            handle_unknown='use_encoded_value'
        )

    @staticmethod
    def sensitive_numeric():
        """Configuration for sensitive numeric features (age, income)."""
        return ColumnConfig(
            missing_strategy='median',
            scaling_strategy='standard'
        )

    @staticmethod
    def amount_features():
        """Configuration for monetary amount features."""
        return ColumnConfig(
            missing_strategy='constant',
            missing_constant=0,  # Zero for missing amounts
            scaling_strategy='robust'  # Robust to outliers
        )


# Export common configs
__all__.extend([
    'quick_load_and_preprocess',
    'create_fraud_pipeline', 
    'CommonConfigs'
])

# Display help function
def show_usage_examples():
    """Display usage examples for the enhanced data processing module."""
    examples = """
    🎯 ENHANCED DATA PROCESSING - USAGE EXAMPLES
    ============================================

    1. 📂 BASIC DATA LOADING:

       from tree_model_helper.data import load_csv

       # Load with automatic separation
       X, y, weights = load_csv('data.csv', target_col='fraud', sample_weight_col='weight')

    2. 🔧 COLUMN-SPECIFIC PREPROCESSING:

       from tree_model_helper.data import ColumnConfig, preprocess_data

       configs = {
           'age': ColumnConfig(missing_strategy='median', scaling_strategy='standard'),
           'category': ColumnConfig(encoding_strategy='woe', missing_constant='Unknown'),
           'income': ColumnConfig(missing_constant=-999999, scaling_strategy='robust')
       }

       X_processed, preprocessor = preprocess_data(X, y, sample_weight=weights, 
                                                  column_configs=configs)

    3. 🎯 FRAUD DETECTION PIPELINE:

       from tree_model_helper.data import create_fraud_pipeline

       pipeline = create_fraud_pipeline('fraud_data.csv')
       X_train, y_train, weights = pipeline['train_data']
       preprocessor = pipeline['preprocessor']

    4. 🔍 WEIGHT OF EVIDENCE ENCODING:

       from tree_model_helper.data import WeightOfEvidenceEncoder

       encoder = WeightOfEvidenceEncoder()
       woe_values = encoder.fit_transform(X['category'], y, sample_weight=weights)
       encoder.save_mapping('category_woe.json')

    5. ⚡ QUICK PREPROCESSING:

       from tree_model_helper.data import quick_load_and_preprocess

       X, y, weights, preprocessor = quick_load_and_preprocess(
           'data.csv', target_col='fraud',
           woe_columns=['category', 'region'],
           constant_columns={'income': -999999},
           scaling_columns=['age', 'amount']
       )

    6. 🏭 PRODUCTION DEPLOYMENT:

       from tree_model_helper.data import AdvancedDataPreprocessor

       # Save mappings during training
       preprocessor = AdvancedDataPreprocessor(mapping_save_dir='prod_mappings/')
       X_processed = preprocessor.fit_transform(X_train, y_train, weights_train)
       preprocessor.save_mappings('production/')

       # Load mappings in production
       prod_preprocessor = AdvancedDataPreprocessor()
       prod_preprocessor.load_mappings('production/')
       X_new_processed = prod_preprocessor.transform(X_new)
    """

    print(examples)

# Add to exports
__all__.append('show_usage_examples')
