"""Configuration loading utilities with YAML support.

This module provides utilities to load configuration from YAML files,
merge with Python defaults, validate, and handle environment overrides.
"""

import yaml
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import asdict

from .base_config import (
    ExperimentConfig, ModelConfig, XGBoostConfig, LightGBMConfig, CatBoostConfig,
    DataConfig, FeatureSelectionConfig, TuningConfig, EvaluationConfig, MLflowConfig,
    EnvironmentConfig, DEFAULT_EXPERIMENT_CONFIG
)
from .config_schema import (
    validate_experiment_config, validate_model_config,
    get_validation_schema, validate_file_path
)


class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass


class ConfigLoader:
    """Configuration loader with YAML support and validation.

    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load_config('config/production.yaml')
        >>> model_config = loader.get_model_config(config, 'xgboost')
    """

    def __init__(self, 
                 config_dir: Optional[Union[str, Path]] = None,
                 validate: bool = True,
                 allow_environment_override: bool = True):
        """Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files
            validate: Whether to validate loaded configuration
            allow_environment_override: Whether to allow environment variable overrides
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.validate = validate
        self.allow_environment_override = allow_environment_override
        self._cache = {}

    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML file and return as dictionary.

        Args:
            file_path: Path to YAML file

        Returns:
            Dictionary with loaded configuration

        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        file_path = Path(file_path)

        # Check if path is relative and make it relative to config_dir
        if not file_path.is_absolute():
            file_path = self.config_dir / file_path

        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if config is None:
                config = {}

            return config

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file {file_path}: {e}")

    def save_yaml(self, config: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration dictionary to save
            file_path: Path where to save the file
        """
        file_path = Path(file_path)

        # Create parent directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration to {file_path}: {e}")

    def merge_configs(self, base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries.

        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if (key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)):
                # Recursively merge nested dictionaries
                merged[key] = self.merge_configs(merged[key], value)
            else:
                # Override or add new key
                merged[key] = value

        return merged

    def load_config(self, 
                   config_file: Optional[Union[str, Path]] = None,
                   base_config: Optional[Dict[str, Any]] = None,
                   model_type: Optional[str] = None) -> ExperimentConfig:
        """Load complete experiment configuration.

        Args:
            config_file: Path to YAML configuration file
            base_config: Base configuration to start with
            model_type: Specific model type to load config for

        Returns:
            Complete experiment configuration object

        Example:
            >>> loader = ConfigLoader()
            >>> config = loader.load_config('experiments/xgboost_tuning.yaml')
            >>> print(f"Using model: {config.model.model_type}")
        """
        # Start with default configuration
        if base_config is None:
            base_config = asdict(DEFAULT_EXPERIMENT_CONFIG)

        # Load from YAML file if provided
        if config_file:
            file_config = self.load_yaml(config_file)
            base_config = self.merge_configs(base_config, file_config)

        # Apply environment overrides if enabled
        if self.allow_environment_override:
            env_config = EnvironmentConfig.from_environment()
            base_config = self.merge_configs(base_config, env_config)

        # Validate configuration if enabled
        if self.validate:
            try:
                base_config = validate_experiment_config(base_config)
            except Exception as e:
                raise ConfigurationError(f"Configuration validation failed: {e}")

        # Convert to ExperimentConfig object
        try:
            # Handle model type specific configuration
            if model_type or 'model' in base_config:
                model_config = base_config.get('model', {})
                if model_type:
                    model_config['model_type'] = model_type

                model_type = model_config.get('model_type', 'xgboost').lower()

                # Create appropriate model config
                if model_type == 'xgboost':
                    model_obj = XGBoostConfig(**model_config)
                elif model_type == 'lightgbm':
                    model_obj = LightGBMConfig(**model_config)
                elif model_type == 'catboost':
                    model_obj = CatBoostConfig(**model_config)
                else:
                    model_obj = ModelConfig(**model_config)

                base_config['model'] = model_obj

            # Create other config objects
            if 'data' in base_config:
                base_config['data'] = DataConfig(**base_config['data'])

            if 'feature_selection' in base_config:
                base_config['feature_selection'] = FeatureSelectionConfig(**base_config['feature_selection'])

            if 'tuning' in base_config:
                base_config['tuning'] = TuningConfig(**base_config['tuning'])

            if 'evaluation' in base_config:
                base_config['evaluation'] = EvaluationConfig(**base_config['evaluation'])

            if 'mlflow' in base_config:
                base_config['mlflow'] = MLflowConfig(**base_config['mlflow'])

            # Create final experiment config
            experiment_config = ExperimentConfig(**base_config)

            return experiment_config

        except Exception as e:
            raise ConfigurationError(f"Error creating configuration objects: {e}")

    def get_model_config(self, config: Union[ExperimentConfig, Dict[str, Any]], 
                        model_type: Optional[str] = None) -> ModelConfig:
        """Get model configuration for specific model type.

        Args:
            config: Experiment configuration or dictionary
            model_type: Model type to get config for

        Returns:
            Model configuration object
        """
        if isinstance(config, ExperimentConfig):
            if model_type and model_type != config.model.model_type:
                return config.get_model_config(model_type)
            return config.model

        elif isinstance(config, dict):
            model_config_dict = config.get('model', {})
            if model_type:
                model_config_dict['model_type'] = model_type

            model_type = model_config_dict.get('model_type', 'xgboost').lower()

            if model_type == 'xgboost':
                return XGBoostConfig(**model_config_dict)
            elif model_type == 'lightgbm':
                return LightGBMConfig(**model_config_dict)
            elif model_type == 'catboost':
                return CatBoostConfig(**model_config_dict)
            else:
                return ModelConfig(**model_config_dict)

        else:
            raise ConfigurationError(f"Invalid config type: {type(config)}")

    def load_multiple_configs(self, config_files: List[Union[str, Path]]) -> Dict[str, ExperimentConfig]:
        """Load multiple configuration files.

        Args:
            config_files: List of configuration file paths

        Returns:
            Dictionary mapping file names to configuration objects
        """
        configs = {}

        for config_file in config_files:
            file_path = Path(config_file)
            config_name = file_path.stem

            try:
                config = self.load_config(config_file)
                configs[config_name] = config
            except Exception as e:
                print(f"Warning: Could not load config {config_file}: {e}")

        return configs

    def export_config(self, config: ExperimentConfig, 
                     output_path: Union[str, Path],
                     format: str = 'yaml') -> None:
        """Export configuration to file.

        Args:
            config: Configuration to export
            output_path: Path to save the configuration
            format: Export format ('yaml', 'json')
        """
        output_path = Path(output_path)
        config_dict = asdict(config)

        if format.lower() == 'yaml':
            self.save_yaml(config_dict, output_path)
        elif format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, default=str)
        else:
            raise ConfigurationError(f"Unsupported export format: {format}")

    def create_config_template(self, model_type: str = 'xgboost', 
                              output_path: Optional[Union[str, Path]] = None) -> str:
        """Create a configuration template YAML file.

        Args:
            model_type: Model type for the template
            output_path: Where to save the template

        Returns:
            Template content as string
        """
        # Create default config for the model type
        base_config = DEFAULT_EXPERIMENT_CONFIG
        base_config.model = base_config.get_model_config(model_type)

        config_dict = asdict(base_config)

        # Convert to YAML string
        template_content = yaml.dump(config_dict, default_flow_style=False, sort_keys=False, indent=2)

        # Add comments
        commented_template = f"""# Configuration template for {model_type.upper()} model
# Generated automatically - customize as needed

# Model configuration
{template_content}

# Additional notes:
# - Set data.train_path and data.test_path to your data files
# - Adjust model parameters in the 'model' section
# - Enable/disable features in other sections as needed
# - Use environment variables for sensitive values (MLFLOW_TRACKING_URI, etc.)
"""

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(commented_template)

        return commented_template

    def validate_config_file(self, config_file: Union[str, Path]) -> List[str]:
        """Validate a configuration file and return any errors.

        Args:
            config_file: Path to configuration file

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            config = self.load_config(config_file)
        except ConfigurationError as e:
            errors.append(str(e))
            return errors
        except Exception as e:
            errors.append(f"Unexpected error: {e}")
            return errors

        # Additional custom validations
        try:
            # Check file paths exist
            if config.data.train_path and not validate_file_path(config.data.train_path, must_exist=True):
                errors.append(f"Training data file not found: {config.data.train_path}")

            if config.data.test_path and not validate_file_path(config.data.test_path, must_exist=True):
                errors.append(f"Test data file not found: {config.data.test_path}")

            # Check output directory can be created
            if not validate_file_path(config.data.output_dir):
                errors.append(f"Cannot create output directory: {config.data.output_dir}")

            # Check model-specific parameters
            model_params = config.model.to_dict()
            from .config_schema import validate_model_parameters
            model_errors = validate_model_parameters(config.model.model_type, model_params)
            errors.extend(model_errors)

        except Exception as e:
            errors.append(f"Validation error: {e}")

        return errors


# Convenience functions
def load_config(config_file: Union[str, Path], 
               model_type: Optional[str] = None,
               validate: bool = True) -> ExperimentConfig:
    """Load configuration from YAML file.

    Args:
        config_file: Path to configuration file
        model_type: Specific model type to use
        validate: Whether to validate configuration

    Returns:
        Experiment configuration object

    Example:
        >>> config = load_config('config/xgboost_default.yaml')
        >>> print(f"Model: {config.model.model_type}")
    """
    loader = ConfigLoader(validate=validate)
    return loader.load_config(config_file, model_type=model_type)


def create_config_from_dict(config_dict: Dict[str, Any], 
                           model_type: Optional[str] = None) -> ExperimentConfig:
    """Create configuration object from dictionary.

    Args:
        config_dict: Configuration dictionary
        model_type: Model type to use

    Returns:
        Experiment configuration object
    """
    loader = ConfigLoader()
    return loader.load_config(base_config=config_dict, model_type=model_type)


def merge_config_files(base_file: Union[str, Path], 
                      override_file: Union[str, Path],
                      output_file: Optional[Union[str, Path]] = None) -> ExperimentConfig:
    """Merge two configuration files.

    Args:
        base_file: Base configuration file
        override_file: Override configuration file
        output_file: Optional output file to save merged config

    Returns:
        Merged configuration object
    """
    loader = ConfigLoader()

    base_config = loader.load_yaml(base_file)
    override_config = loader.load_yaml(override_file)

    merged_config = loader.merge_configs(base_config, override_config)

    if output_file:
        loader.save_yaml(merged_config, output_file)

    return create_config_from_dict(merged_config)


def get_config_template(model_type: str = 'xgboost') -> str:
    """Get configuration template for a model type.

    Args:
        model_type: Model type for template

    Returns:
        YAML template string
    """
    loader = ConfigLoader()
    return loader.create_config_template(model_type)


def validate_config_file(config_file: Union[str, Path]) -> List[str]:
    """Validate a configuration file.

    Args:
        config_file: Path to configuration file

    Returns:
        List of validation errors (empty if valid)
    """
    loader = ConfigLoader()
    return loader.validate_config_file(config_file)


# Configuration discovery utilities
def find_config_files(directory: Union[str, Path], 
                     pattern: str = "*.yaml") -> List[Path]:
    """Find configuration files in a directory.

    Args:
        directory: Directory to search in
        pattern: File pattern to match

    Returns:
        List of found configuration files
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    return list(directory.glob(pattern))


def get_available_configs(config_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """Get all available configuration files.

    Args:
        config_dir: Directory to search in (defaults to package config dir)

    Returns:
        Dictionary mapping config names to file paths
    """
    if config_dir is None:
        config_dir = Path(__file__).parent / "defaults"
    else:
        config_dir = Path(config_dir)

    configs = {}

    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            config_name = config_file.stem
            configs[config_name] = config_file

    return configs


# Environment-specific loading
def load_environment_config(environment: str = "development") -> ExperimentConfig:
    """Load configuration for a specific environment.

    Args:
        environment: Environment name (development, staging, production)

    Returns:
        Environment-specific configuration
    """
    config_file = Path(__file__).parent / "defaults" / f"{environment}.yaml"

    if config_file.exists():
        return load_config(config_file)
    else:
        # Fall back to default configuration with environment overrides
        loader = ConfigLoader()
        return loader.load_config()


# Configuration caching for performance
_config_cache = {}

def get_cached_config(config_file: Union[str, Path], 
                     cache_key: Optional[str] = None) -> ExperimentConfig:
    """Get configuration with caching for better performance.

    Args:
        config_file: Path to configuration file
        cache_key: Optional cache key (defaults to file path)

    Returns:
        Cached configuration object
    """
    if cache_key is None:
        cache_key = str(Path(config_file).resolve())

    if cache_key not in _config_cache:
        _config_cache[cache_key] = load_config(config_file)

    return _config_cache[cache_key]


def clear_config_cache():
    """Clear the configuration cache."""
    global _config_cache
    _config_cache.clear()
