"""Tree Models - Data Processing and Validation Components.

This module provides comprehensive data processing utilities including
data validation, preprocessing, and feature engineering capabilities.

Key Components:
- DataValidator: Comprehensive data quality assessment and anomaly detection
- DataPreprocessor: Advanced preprocessing pipeline with type safety
- FeatureEngineer: 15+ transformation types with domain-specific helpers

Example:
    >>> from tree_models.data import DataValidator, FeatureEngineer
    >>> validator = DataValidator()
    >>> engineer = FeatureEngineer()
    >>> validation_results = validator.validate(df)
"""

# Core data components
from .validator import DataValidator, validate_dataset
from .preprocessor import DataPreprocessor, AdvancedDataPreprocessor
from .feature_engineer import FeatureEngineer, create_financial_features

__all__ = [
    # Core implementations
    'DataValidator',
    'DataPreprocessor',
    'AdvancedDataPreprocessor',
    'FeatureEngineer',
    
    # Quick utility functions
    'validate_dataset',
    'create_financial_features'
]