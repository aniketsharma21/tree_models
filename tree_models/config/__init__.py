"""Tree Models - Configuration Management Components.

This module provides type-safe configuration management for all aspects
of the ML pipeline including model parameters, data processing, and experiments.

Key Components:
- ModelConfig: Base configuration for all model types
- XGBoostConfig, LightGBMConfig, CatBoostConfig: Model-specific configurations
- DataConfig: Data processing and feature configuration
- ConfigLoader: YAML/JSON configuration loading with validation

Example:
    >>> from tree_models.config import XGBoostConfig, load_config
    >>> config = XGBoostConfig.for_fraud_detection()
    >>> config = load_config('config/model.yaml')
"""

# Configuration classes
from .model_config import (
    ModelConfig,
    XGBoostConfig,
    LightGBMConfig, 
    CatBoostConfig
)
from .data_config import (
    DataConfig,
    FeatureConfig,
    FeatureEngineeringConfig,
    create_advanced_data_config
)

# Configuration utilities
from .loader import (
    ConfigLoader,
    load_config,
    save_config,
    validate_config
)

__all__ = [
    # Model configurations
    'ModelConfig',
    'XGBoostConfig',
    'LightGBMConfig',
    'CatBoostConfig',
    
    # Data configurations
    'DataConfig',
    'FeatureConfig',
    'FeatureEngineeringConfig',
    'create_advanced_data_config',
    
    # Configuration utilities
    'ConfigLoader',
    'load_config',
    'save_config',
    'validate_config'
]