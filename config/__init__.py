"""Tree Model Helper Configuration System.

This package provides a hybrid configuration system combining the benefits of
Python's type safety with YAML's flexibility for machine learning experiments.

Key Features:
- Type-safe configuration classes with dataclasses
- YAML file support for easy configuration management
- Environment variable overrides
- Configuration validation with Pydantic (optional)
- Model-specific configuration templates
- MLflow integration settings
- Production-ready configurations

Basic Usage:
    >>> from tree_model_helper.config import load_config, XGBoostConfig
    >>> 
    >>> # Load from YAML file
    >>> config = load_config('config/xgboost_default.yaml')
    >>> print(f"Using {config.model.model_type} with {config.model.n_estimators} trees")
    >>> 
    >>> # Create from Python objects
    >>> model_config = XGBoostConfig(n_estimators=500, max_depth=8)
    >>> print(model_config.to_dict())

Advanced Usage:
    >>> from tree_model_helper.config import ConfigLoader, get_config_template
    >>> 
    >>> # Create custom loader
    >>> loader = ConfigLoader(config_dir='custom_configs/', validate=True)
    >>> config = loader.load_config('my_experiment.yaml', model_type='lightgbm')
    >>> 
    >>> # Generate template
    >>> template = get_config_template('catboost')
    >>> print(template)
"""

from .base_config import (
    # Configuration classes
    ExperimentConfig,
    ModelConfig,
    XGBoostConfig,
    LightGBMConfig,
    CatBoostConfig,
    DataConfig,
    FeatureSelectionConfig,
    TuningConfig,
    EvaluationConfig,
    MLflowConfig,
    EnvironmentConfig,

    # Default instances
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_XGBOOST_CONFIG,
    DEFAULT_LIGHTGBM_CONFIG,
    DEFAULT_CATBOOST_CONFIG,

    # Legacy compatibility
    SEED,
    TEST_SIZE,
    VALID_SIZE,
    XGB_DEFAULT_PARAMS,
    LGBM_DEFAULT_PARAMS,
    CATBOOST_DEFAULT_PARAMS,
    XGB_OPTUNA_SPACE,
    LGBM_OPTUNA_SPACE,
    CATBOOST_OPTUNA_SPACE,
)

from .config_loader import (
    # Main loader class
    ConfigLoader,
    ConfigurationError,

    # Convenience functions
    load_config,
    create_config_from_dict,
    merge_config_files,
    get_config_template,
    validate_config_file,

    # Discovery utilities
    find_config_files,
    get_available_configs,
    load_environment_config,

    # Caching utilities
    get_cached_config,
    clear_config_cache,
)

from .config_schema import (
    # Schema classes (if Pydantic available)
    ModelConfigSchema,
    XGBoostConfigSchema,
    LightGBMConfigSchema,
    CatBoostConfigSchema,
    DataConfigSchema,
    FeatureSelectionConfigSchema,
    TuningConfigSchema,
    EvaluationConfigSchema,
    MLflowConfigSchema,
    ExperimentConfigSchema,

    # Validation functions
    validate_config_dict,
    validate_model_config,
    validate_experiment_config,
    get_validation_schema,
    validate_file_path,
    validate_model_parameters,
)

# Version info
__version__ = '0.1.0'

# Quick access to common configurations
def get_default_config(model_type: str = 'xgboost') -> ExperimentConfig:
    """Get default configuration for a model type.

    Args:
        model_type: Model type ('xgboost', 'lightgbm', 'catboost')

    Returns:
        Default experiment configuration

    Example:
        >>> config = get_default_config('lightgbm')
        >>> print(config.model.model_type)  # 'lightgbm'
    """
    config = DEFAULT_EXPERIMENT_CONFIG
    config.model = config.get_model_config(model_type)
    return config


def quick_config(model_type: str = 'xgboost', 
                n_estimators: int = 200,
                max_depth: int = 6,
                learning_rate: float = 0.1,
                **kwargs) -> ModelConfig:
    """Quick model configuration creation.

    Args:
        model_type: Model type
        n_estimators: Number of estimators
        max_depth: Maximum depth
        learning_rate: Learning rate
        **kwargs: Additional model parameters

    Returns:
        Model configuration object

    Example:
        >>> model_config = quick_config('xgboost', n_estimators=500, reg_alpha=0.1)
        >>> params = model_config.to_dict()
    """
    if model_type == 'xgboost':
        return XGBoostConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            **kwargs
        )
    elif model_type == 'lightgbm':
        return LightGBMConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            **kwargs
        )
    elif model_type == 'catboost':
        return CatBoostConfig(
            n_estimators=n_estimators,
            depth=max_depth,  # CatBoost uses 'depth'
            learning_rate=learning_rate,
            **kwargs
        )
    else:
        return ModelConfig(
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            **kwargs
        )


# Configuration helpers
class ConfigManager:
    """High-level configuration manager for experiments.

    Example:
        >>> manager = ConfigManager()
        >>> 
        >>> # Load production config
        >>> config = manager.get_production_config('xgboost')
        >>> 
        >>> # Create experiment config
        >>> config = manager.create_experiment_config(
        ...     model_type='lightgbm',
        ...     data_path='data/fraud_data.csv',
        ...     experiment_name='fraud_detection_v2'
        ... )
    """

    def __init__(self, config_dir: str = None):
        """Initialize configuration manager."""
        self.loader = ConfigLoader(config_dir=config_dir)

    def get_production_config(self, model_type: str = None) -> ExperimentConfig:
        """Get production-ready configuration."""
        try:
            config = self.loader.load_config('production.yaml', model_type=model_type)
            return config
        except Exception:
            # Fallback to default with production-like settings
            config = get_default_config(model_type or 'xgboost')
            config.model.n_estimators = 500
            config.model.early_stopping_rounds = 100
            config.tuning.enable_tuning = False
            config.evaluation.generate_plots = False
            return config

    def create_experiment_config(self, 
                               model_type: str = 'xgboost',
                               data_path: str = None,
                               experiment_name: str = 'ml_experiment',
                               enable_tuning: bool = True,
                               n_trials: int = 100,
                               **kwargs) -> ExperimentConfig:
        """Create experiment configuration with common settings.

        Args:
            model_type: Type of model to use
            data_path: Path to training data
            experiment_name: MLflow experiment name
            enable_tuning: Whether to enable hyperparameter tuning
            n_trials: Number of tuning trials
            **kwargs: Additional configuration overrides

        Returns:
            Configured experiment object
        """
        config = get_default_config(model_type)

        # Set data path
        if data_path:
            config.data.train_path = data_path

        # Set experiment name
        config.mlflow.experiment_name = experiment_name

        # Configure tuning
        config.tuning.enable_tuning = enable_tuning
        config.tuning.n_trials = n_trials

        # Apply any additional overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)

        return config

    def compare_configs(self, config_files: list) -> dict:
        """Compare multiple configuration files.

        Args:
            config_files: List of configuration file paths

        Returns:
            Dictionary with comparison results
        """
        configs = {}
        comparison = {}

        # Load all configs
        for file_path in config_files:
            try:
                config_name = Path(file_path).stem
                configs[config_name] = self.loader.load_config(file_path)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

        # Compare key settings
        comparison['model_types'] = {name: config.model.model_type for name, config in configs.items()}
        comparison['n_estimators'] = {name: config.model.n_estimators for name, config in configs.items()}
        comparison['tuning_enabled'] = {name: config.tuning.enable_tuning for name, config in configs.items()}
        comparison['feature_selection'] = {name: config.feature_selection.enable_feature_selection for name, config in configs.items()}

        return comparison

    def validate_configs(self, config_dir: str = None) -> dict:
        """Validate all configuration files in a directory.

        Args:
            config_dir: Directory containing config files

        Returns:
            Dictionary with validation results
        """
        if config_dir is None:
            config_dir = self.loader.config_dir / "defaults"

        results = {}
        config_files = find_config_files(config_dir)

        for config_file in config_files:
            config_name = config_file.stem
            errors = validate_config_file(config_file)
            results[config_name] = {
                'valid': len(errors) == 0,
                'errors': errors
            }

        return results


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience shortcuts
get_production_config = config_manager.get_production_config
create_experiment_config = config_manager.create_experiment_config

# Available configurations discovery
def list_available_configs() -> dict:
    """List all available configuration files.

    Returns:
        Dictionary of available configurations

    Example:
        >>> configs = list_available_configs()
        >>> print(f"Available configs: {list(configs.keys())}")
    """
    return get_available_configs()


def show_config_info(config: ExperimentConfig) -> str:
    """Get human-readable configuration information.

    Args:
        config: Configuration object

    Returns:
        Formatted configuration summary
    """
    info = f"""Configuration Summary:
{'='*50}
Model: {config.model.model_type.upper()}
- Estimators: {config.model.n_estimators}
- Max Depth: {config.model.max_depth}
- Learning Rate: {config.model.learning_rate}

Data:
- Target Column: {config.data.target_col}
- Weight Column: {config.data.weight_col or 'None'}
- Test Size: {config.data.test_size}
- Valid Size: {config.data.valid_size}

Feature Selection: {'Enabled' if config.feature_selection.enable_feature_selection else 'Disabled'}
Hyperparameter Tuning: {'Enabled' if config.tuning.enable_tuning else 'Disabled'}
- Trials: {config.tuning.n_trials if config.tuning.enable_tuning else 'N/A'}

MLflow:
- Experiment: {config.mlflow.experiment_name}
- Tracking URI: {config.mlflow.tracking_uri or 'Default'}

Output Directory: {config.data.output_dir}
Random Seed: {config.seed}
"""
    return info


# Export main components for easy access
__all__ = [
    # Configuration classes
    'ExperimentConfig',
    'ModelConfig', 'XGBoostConfig', 'LightGBMConfig', 'CatBoostConfig',
    'DataConfig', 'FeatureSelectionConfig', 'TuningConfig', 'EvaluationConfig', 'MLflowConfig',

    # Loading functions
    'load_config', 'ConfigLoader', 'ConfigurationError',
    'create_config_from_dict', 'get_config_template',

    # Defaults
    'DEFAULT_EXPERIMENT_CONFIG', 'get_default_config', 'quick_config',

    # Management
    'ConfigManager', 'config_manager',
    'get_production_config', 'create_experiment_config',

    # Utilities
    'list_available_configs', 'show_config_info', 'validate_config_file',

    # Legacy compatibility
    'SEED', 'XGB_DEFAULT_PARAMS', 'LGBM_DEFAULT_PARAMS', 'CATBOOST_DEFAULT_PARAMS',
]
