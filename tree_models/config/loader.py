# tree_models/config/loader.py
"""Enhanced configuration loading with production-ready features.

This module provides comprehensive configuration management with:
- Type-safe YAML configuration loading and validation
- Environment variable integration and overrides
- Configuration merging and templating capabilities
- Comprehensive error handling and validation
- Caching and performance optimization
- Support for multiple configuration formats
"""

import yaml
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import asdict, fields
import warnings

from .model_config import (
    ModelConfig, XGBoostConfig, LightGBMConfig, CatBoostConfig,
    ExperimentConfig, DEFAULT_CONFIGS
)
from .data_config import DataConfig, FeatureConfig, ProcessingConfig
from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    ConfigurationError,
    DataValidationError,
    handle_and_reraise,
    validate_parameter,
    create_error_context
)

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class ConfigLoader:
    """Enhanced configuration loader with comprehensive features.

    Provides type-safe configuration loading, validation, environment
    integration, and advanced configuration management capabilities.

    Example:
        >>> loader = ConfigLoader('config/')
        >>> config = loader.load_experiment_config('production.yaml')
        >>> model_config = loader.get_model_config('xgboost', config)
        >>> 
        >>> # Environment-specific loading
        >>> prod_config = loader.load_environment_config('production')
        >>> 
        >>> # Configuration validation
        >>> errors = loader.validate_config_file('staging.yaml')
        >>> if not errors:
        ...     config = loader.load_config('staging.yaml')
    """

    def __init__(
        self, 
        config_dir: Optional[Union[str, Path]] = None,
        validate: bool = True,
        allow_environment_override: bool = True,
        cache_enabled: bool = True,
        encoding: str = 'utf-8'
    ) -> None:
        """Initialize enhanced configuration loader.

        Args:
            config_dir: Directory containing configuration files
            validate: Whether to validate loaded configuration
            allow_environment_override: Whether to allow environment variable overrides
            cache_enabled: Whether to cache loaded configurations
            encoding: File encoding for configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / 'defaults'
        self.validate = validate
        self.allow_environment_override = allow_environment_override
        self.cache_enabled = cache_enabled
        self.encoding = encoding
        
        # Configuration cache
        self._cache: Dict[str, Any] = {}
        self._file_timestamps: Dict[str, float] = {}
        
        # Validation schemas
        self._validation_schemas: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized ConfigLoader:")
        logger.info(f"  Config dir: {self.config_dir}")
        logger.info(f"  Validation: {validate}, Environment override: {allow_environment_override}")

    @timer(name="config_loading")
    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML file with comprehensive error handling.

        Args:
            file_path: Path to YAML file

        Returns:
            Dictionary with loaded configuration

        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        file_path = self._resolve_config_path(file_path)
        
        try:
            # Check cache first if enabled
            if self.cache_enabled:
                cache_key = str(file_path)
                file_mtime = file_path.stat().st_mtime
                
                if (cache_key in self._cache and 
                    cache_key in self._file_timestamps and
                    self._file_timestamps[cache_key] == file_mtime):
                    logger.debug(f"Loading config from cache: {file_path.name}")
                    return self._cache[cache_key].copy()

            # Load and parse YAML
            with open(file_path, 'r', encoding=self.encoding) as f:
                config = yaml.safe_load(f)

            if config is None:
                config = {}

            # Validate YAML structure
            if not isinstance(config, dict):
                raise ConfigurationError(f"Configuration file must contain a dictionary, got {type(config)}")

            # Cache the result if enabled
            if self.cache_enabled:
                cache_key = str(file_path)
                self._cache[cache_key] = config.copy()
                self._file_timestamps[cache_key] = file_path.stat().st_mtime

            logger.debug(f"Loaded configuration from {file_path}")
            return config

        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file {file_path}: {e}")
        except Exception as e:
            handle_and_reraise(
                e, ConfigurationError,
                f"Error loading configuration file {file_path}",
                error_code="CONFIG_LOAD_FAILED",
                context=create_error_context(file_path=str(file_path))
            )

    def _resolve_config_path(self, file_path: Union[str, Path]) -> Path:
        """Resolve configuration file path."""
        file_path = Path(file_path)

        # Check if path is relative and make it relative to config_dir
        if not file_path.is_absolute():
            file_path = self.config_dir / file_path

        if not file_path.exists():
            # Try with .yaml extension if not present
            if file_path.suffix not in ['.yaml', '.yml']:
                yaml_path = file_path.with_suffix('.yaml')
                if yaml_path.exists():
                    return yaml_path
                
                yml_path = file_path.with_suffix('.yml')
                if yml_path.exists():
                    return yml_path

            raise ConfigurationError(f"Configuration file not found: {file_path}")

        return file_path

    def save_yaml(self, config: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file with validation."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding=self.encoding) as f:
                yaml.dump(
                    config, f, 
                    default_flow_style=False, 
                    sort_keys=False, 
                    indent=2,
                    allow_unicode=True
                )
            
            logger.info(f"Configuration saved to {file_path}")

        except Exception as e:
            handle_and_reraise(
                e, ConfigurationError,
                f"Error saving configuration to {file_path}",
                error_code="CONFIG_SAVE_FAILED"
            )

    def merge_configs(
        self, 
        base_config: Dict[str, Any], 
        override_config: Dict[str, Any],
        merge_strategy: str = 'deep'
    ) -> Dict[str, Any]:
        """Deep merge configuration dictionaries with multiple strategies.

        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            merge_strategy: Merging strategy ('deep', 'shallow', 'replace')

        Returns:
            Merged configuration dictionary
        """
        if merge_strategy == 'replace':
            return override_config.copy()
        elif merge_strategy == 'shallow':
            merged = base_config.copy()
            merged.update(override_config)
            return merged
        elif merge_strategy == 'deep':
            return self._deep_merge(base_config, override_config)
        else:
            raise ConfigurationError(f"Unknown merge strategy: {merge_strategy}")

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dictionaries."""
        merged = base.copy()

        for key, value in override.items():
            if (key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)):
                # Recursively merge nested dictionaries
                merged[key] = self._deep_merge(merged[key], value)
            else:
                # Override or add new key
                merged[key] = value

        return merged

    def apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.
        
        Environment variables are expected in the format:
        TREE_MODELS_<SECTION>_<KEY>=value
        
        Example: TREE_MODELS_MODEL_N_ESTIMATORS=200
        """
        if not self.allow_environment_override:
            return config

        env_overrides = {}
        prefix = "TREE_MODELS_"

        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Parse environment variable
                config_path = env_key[len(prefix):].lower().split('_')
                
                # Convert string value to appropriate type
                typed_value = self._parse_env_value(env_value)
                
                # Apply to config
                self._set_nested_config(env_overrides, config_path, typed_value)

        if env_overrides:
            logger.info(f"Applying environment overrides: {list(env_overrides.keys())}")
            config = self.merge_configs(config, env_overrides)

        return config

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first (handles lists, dicts, booleans, etc.)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Handle boolean strings
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Return as string
        return value

    def _set_nested_config(self, config: Dict[str, Any], path: List[str], value: Any) -> None:
        """Set nested configuration value using dot notation path."""
        current = config
        
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[path[-1]] = value

    @timer(name="experiment_config_loading")
    def load_experiment_config(
        self, 
        config_file: Optional[Union[str, Path]] = None,
        base_config: Optional[Dict[str, Any]] = None,
        model_type: Optional[str] = None,
        environment: Optional[str] = None
    ) -> ExperimentConfig:
        """Load complete experiment configuration with validation.

        Args:
            config_file: Path to YAML configuration file
            base_config: Base configuration to start with
            model_type: Specific model type to configure
            environment: Environment-specific overrides to apply

        Returns:
            Complete experiment configuration object

        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        logger.info(f"🔧 Loading experiment configuration:")
        logger.info(f"   File: {config_file}, Model: {model_type}, Environment: {environment}")

        try:
            with timed_operation("config_processing"):
                # Start with default configuration
                if base_config is None:
                    base_config = self._get_default_config()

                # Load from YAML file if provided
                if config_file:
                    file_config = self.load_yaml(config_file)
                    base_config = self.merge_configs(base_config, file_config)

                # Apply environment-specific overrides
                if environment:
                    env_config = self._load_environment_overrides(environment)
                    if env_config:
                        base_config = self.merge_configs(base_config, env_config)

                # Apply environment variable overrides
                base_config = self.apply_environment_overrides(base_config)

                # Validate configuration if enabled
                if self.validate:
                    base_config = self._validate_experiment_config(base_config)

                # Create experiment configuration object
                experiment_config = self._create_experiment_config(base_config, model_type)

            logger.info(f"✅ Experiment configuration loaded successfully")
            return experiment_config

        except Exception as e:
            handle_and_reraise(
                e, ConfigurationError,
                "Failed to load experiment configuration",
                error_code="EXPERIMENT_CONFIG_FAILED",
                context=create_error_context(
                    config_file=str(config_file) if config_file else None,
                    model_type=model_type,
                    environment=environment
                )
            )

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default experiment configuration."""
        return {
            'model': {
                'model_type': 'xgboost',
                'random_state': 42
            },
            'data': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42
            },
            'experiment': {
                'name': 'default_experiment',
                'description': 'Default experiment configuration'
            }
        }

    def _load_environment_overrides(self, environment: str) -> Optional[Dict[str, Any]]:
        """Load environment-specific configuration overrides."""
        env_file = self.config_dir / f"environments/{environment}.yaml"
        
        if env_file.exists():
            logger.debug(f"Loading environment overrides from {env_file}")
            return self.load_yaml(env_file)
        
        return None

    def _validate_experiment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experiment configuration."""
        
        # Basic structure validation
        required_sections = ['model']
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing required section '{section}', using defaults")
                config[section] = {}

        # Model configuration validation
        if 'model' in config:
            model_type = config['model'].get('model_type', 'xgboost')
            config['model'] = self._validate_model_config(config['model'], model_type)

        # Data configuration validation
        if 'data' in config:
            config['data'] = self._validate_data_config(config['data'])

        return config

    def _validate_model_config(self, model_config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Validate model-specific configuration."""
        
        # Get default config for model type
        if model_type in DEFAULT_CONFIGS:
            defaults = asdict(DEFAULT_CONFIGS[model_type])
            model_config = self.merge_configs(defaults, model_config)
        
        # Validate parameters
        valid_params = self._get_valid_model_params(model_type)
        
        # Remove invalid parameters with warning
        invalid_params = set(model_config.keys()) - valid_params
        for param in invalid_params:
            if param != 'model_type':  # Allow model_type
                logger.warning(f"Removing invalid parameter for {model_type}: {param}")
                del model_config[param]

        return model_config

    def _get_valid_model_params(self, model_type: str) -> set:
        """Get valid parameter names for model type."""
        
        param_sets = {
            'xgboost': {
                'model_type', 'n_estimators', 'max_depth', 'learning_rate', 'subsample',
                'colsample_bytree', 'gamma', 'min_child_weight', 'reg_alpha', 'reg_lambda',
                'random_state', 'n_jobs', 'tree_method', 'gpu_id'
            },
            'lightgbm': {
                'model_type', 'n_estimators', 'max_depth', 'learning_rate', 'subsample',
                'colsample_bytree', 'min_child_samples', 'min_child_weight', 'reg_alpha',
                'reg_lambda', 'random_state', 'n_jobs', 'device_type'
            },
            'catboost': {
                'model_type', 'iterations', 'depth', 'learning_rate', 'l2_leaf_reg',
                'border_count', 'random_seed', 'task_type', 'devices'
            }
        }
        
        return param_sets.get(model_type, set())

    def _validate_data_config(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data configuration."""
        
        # Validate test_size and validation_size
        for size_param in ['test_size', 'validation_size']:
            if size_param in data_config:
                size_value = data_config[size_param]
                if not (0.0 < size_value < 1.0):
                    logger.warning(f"Invalid {size_param}: {size_value}, using default 0.2")
                    data_config[size_param] = 0.2

        return data_config

    def _create_experiment_config(
        self, 
        config_dict: Dict[str, Any], 
        model_type: Optional[str] = None
    ) -> ExperimentConfig:
        """Create experiment configuration object from dictionary."""
        
        # Handle model type specification
        if model_type:
            config_dict.setdefault('model', {})['model_type'] = model_type

        # Create model configuration object
        model_config_dict = config_dict.get('model', {})
        model_type = model_config_dict.get('model_type', 'xgboost')
        
        if model_type == 'xgboost':
            model_config = XGBoostConfig(**model_config_dict)
        elif model_type == 'lightgbm':
            model_config = LightGBMConfig(**model_config_dict)
        elif model_type == 'catboost':
            model_config = CatBoostConfig(**model_config_dict)
        else:
            model_config = ModelConfig(**model_config_dict)

        # Create data configuration object
        data_config_dict = config_dict.get('data', {})
        data_config = DataConfig(**data_config_dict)

        # Create experiment configuration
        experiment_config = ExperimentConfig(
            model=model_config,
            data=data_config,
            experiment_name=config_dict.get('experiment', {}).get('name', 'default_experiment'),
            description=config_dict.get('experiment', {}).get('description', ''),
            random_state=config_dict.get('random_state', 42)
        )

        return experiment_config

    def get_model_config(
        self, 
        model_type: str, 
        config: Optional[Union[ExperimentConfig, Dict[str, Any]]] = None,
        **overrides: Any
    ) -> ModelConfig:
        """Get model configuration for specific model type.

        Args:
            model_type: Model type to get config for
            config: Experiment configuration or dictionary
            **overrides: Parameter overrides

        Returns:
            Model configuration object

        Example:
            >>> loader = ConfigLoader()
            >>> xgb_config = loader.get_model_config(
            ...     'xgboost', 
            ...     n_estimators=200,
            ...     max_depth=8
            ... )
        """
        try:
            # Start with default config for model type
            if model_type in DEFAULT_CONFIGS:
                base_config = asdict(DEFAULT_CONFIGS[model_type])
            else:
                base_config = {'model_type': model_type}

            # Merge with provided config
            if isinstance(config, ExperimentConfig):
                if config.model.model_type == model_type:
                    model_dict = asdict(config.model)
                    base_config = self.merge_configs(base_config, model_dict)
            elif isinstance(config, dict):
                model_dict = config.get('model', {})
                base_config = self.merge_configs(base_config, model_dict)

            # Apply overrides
            if overrides:
                base_config.update(overrides)

            # Create appropriate model config object
            if model_type == 'xgboost':
                return XGBoostConfig(**base_config)
            elif model_type == 'lightgbm':
                return LightGBMConfig(**base_config)
            elif model_type == 'catboost':
                return CatBoostConfig(**base_config)
            else:
                return ModelConfig(**base_config)

        except Exception as e:
            handle_and_reraise(
                e, ConfigurationError,
                f"Failed to create model config for {model_type}",
                error_code="MODEL_CONFIG_FAILED"
            )

    def load_environment_config(self, environment: str = "development") -> ExperimentConfig:
        """Load configuration for a specific environment.

        Args:
            environment: Environment name (development, staging, production)

        Returns:
            Environment-specific configuration
        """
        config_file = f"environments/{environment}.yaml"
        
        try:
            return self.load_experiment_config(config_file, environment=environment)
        except ConfigurationError:
            logger.warning(f"Environment config not found: {environment}, using defaults")
            return self.load_experiment_config(environment=environment)

    def validate_config_file(self, config_file: Union[str, Path]) -> List[str]:
        """Validate a configuration file and return any errors.

        Args:
            config_file: Path to configuration file

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            # Load configuration
            config_dict = self.load_yaml(config_file)
            
            # Attempt to create experiment config
            experiment_config = self._create_experiment_config(config_dict)
            
            # Validate file paths if specified
            if hasattr(experiment_config.data, 'train_path') and experiment_config.data.train_path:
                train_path = Path(experiment_config.data.train_path)
                if not train_path.exists():
                    errors.append(f"Training data file not found: {train_path}")

            if hasattr(experiment_config.data, 'test_path') and experiment_config.data.test_path:
                test_path = Path(experiment_config.data.test_path)
                if not test_path.exists():
                    errors.append(f"Test data file not found: {test_path}")

        except ConfigurationError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Validation error: {e}")

        return errors

    def create_config_template(
        self, 
        model_type: str = 'xgboost', 
        output_path: Optional[Union[str, Path]] = None,
        include_comments: bool = True
    ) -> str:
        """Create a configuration template YAML file.

        Args:
            model_type: Model type for the template
            output_path: Where to save the template
            include_comments: Whether to include explanatory comments

        Returns:
            Template content as string
        """
        try:
            # Get default model configuration
            model_config = self.get_model_config(model_type)
            
            # Create template structure
            template_dict = {
                'experiment': {
                    'name': f'{model_type}_experiment',
                    'description': f'Template configuration for {model_type.upper()} model'
                },
                'model': asdict(model_config),
                'data': {
                    'train_path': 'data/train.csv',
                    'test_path': 'data/test.csv', 
                    'target_column': 'target',
                    'test_size': 0.2,
                    'validation_size': 0.2,
                    'random_state': 42
                }
            }

            # Convert to YAML string
            template_content = yaml.dump(
                template_dict, 
                default_flow_style=False, 
                sort_keys=False, 
                indent=2
            )

            # Add comments if requested
            if include_comments:
                commented_template = f"""# Configuration template for {model_type.upper()} model
# Generated automatically - customize as needed

{template_content}
# Additional notes:
# - Set data.train_path and data.test_path to your data files
# - Adjust model parameters in the 'model' section
# - Use environment variables for sensitive values
# - Available model types: xgboost, lightgbm, catboost
"""
                template_content = commented_template

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding=self.encoding) as f:
                    f.write(template_content)
                logger.info(f"Configuration template saved to {output_path}")

            return template_content

        except Exception as e:
            handle_and_reraise(
                e, ConfigurationError,
                f"Failed to create config template for {model_type}",
                error_code="TEMPLATE_CREATION_FAILED"
            )

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
        self._file_timestamps.clear()
        logger.debug("Configuration cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached configurations."""
        return {
            'cached_configs': len(self._cache),
            'cache_enabled': self.cache_enabled,
            'cached_files': list(self._cache.keys())
        }


# Convenience functions for easy usage
def load_config(
    config_file: Union[str, Path], 
    model_type: Optional[str] = None,
    validate: bool = True,
    **kwargs: Any
) -> ExperimentConfig:
    """Load configuration from YAML file with convenience interface.

    Args:
        config_file: Path to configuration file
        model_type: Specific model type to use
        validate: Whether to validate configuration
        **kwargs: Additional loader parameters

    Returns:
        Experiment configuration object

    Example:
        >>> config = load_config('config/xgboost_default.yaml')
        >>> print(f"Model: {config.model.model_type}")
        >>> print(f"Estimators: {config.model.n_estimators}")
    """
    loader = ConfigLoader(validate=validate, **kwargs)
    return loader.load_experiment_config(config_file, model_type=model_type)


def create_config_from_dict(
    config_dict: Dict[str, Any], 
    model_type: Optional[str] = None,
    validate: bool = True
) -> ExperimentConfig:
    """Create configuration object from dictionary.

    Args:
        config_dict: Configuration dictionary
        model_type: Model type to use
        validate: Whether to validate configuration

    Returns:
        Experiment configuration object
    """
    loader = ConfigLoader(validate=validate)
    return loader.load_experiment_config(base_config=config_dict, model_type=model_type)


def merge_config_files(
    base_file: Union[str, Path], 
    override_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    merge_strategy: str = 'deep'
) -> ExperimentConfig:
    """Merge two configuration files.

    Args:
        base_file: Base configuration file
        override_file: Override configuration file
        output_file: Optional output file to save merged config
        merge_strategy: Merging strategy to use

    Returns:
        Merged configuration object
    """
    loader = ConfigLoader()

    base_config = loader.load_yaml(base_file)
    override_config = loader.load_yaml(override_file)

    merged_config = loader.merge_configs(base_config, override_config, merge_strategy)

    if output_file:
        loader.save_yaml(merged_config, output_file)

    return create_config_from_dict(merged_config)


def get_model_config(
    model_type: str, 
    **overrides: Any
) -> ModelConfig:
    """Get model configuration with parameter overrides.

    Args:
        model_type: Model type to configure
        **overrides: Parameter overrides

    Returns:
        Model configuration object

    Example:
        >>> xgb_config = get_model_config('xgboost', n_estimators=200, max_depth=8)
        >>> lgb_config = get_model_config('lightgbm', learning_rate=0.05)
    """
    loader = ConfigLoader()
    return loader.get_model_config(model_type, **overrides)


def validate_config_file(config_file: Union[str, Path]) -> List[str]:
    """Validate a configuration file.

    Args:
        config_file: Path to configuration file

    Returns:
        List of validation errors (empty if valid)

    Example:
        >>> errors = validate_config_file('config/production.yaml')
        >>> if errors:
        ...     print("Configuration errors found:")
        ...     for error in errors:
        ...         print(f"  - {error}")
        >>> else:
        ...     print("Configuration is valid!")
    """
    loader = ConfigLoader()
    return loader.validate_config_file(config_file)


def get_config_template(
    model_type: str = 'xgboost',
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """Get configuration template for a model type.

    Args:
        model_type: Model type for template
        output_path: Optional path to save template

    Returns:
        YAML template string
    """
    loader = ConfigLoader()
    return loader.create_config_template(model_type, output_path)


def find_config_files(
    directory: Union[str, Path], 
    pattern: str = "*.yaml"
) -> List[Path]:
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

    return sorted(directory.glob(pattern))


def load_environment_config(environment: str = "development") -> ExperimentConfig:
    """Load configuration for a specific environment.

    Args:
        environment: Environment name (development, staging, production)

    Returns:
        Environment-specific configuration

    Example:
        >>> dev_config = load_environment_config('development')
        >>> prod_config = load_environment_config('production')
    """
    loader = ConfigLoader()
    return loader.load_environment_config(environment)


# Configuration caching utilities
_global_config_cache: Dict[str, ExperimentConfig] = {}

def get_cached_config(
    config_file: Union[str, Path], 
    cache_key: Optional[str] = None,
    **kwargs: Any
) -> ExperimentConfig:
    """Get configuration with caching for better performance.

    Args:
        config_file: Path to configuration file
        cache_key: Optional cache key (defaults to file path)
        **kwargs: Additional loader parameters

    Returns:
        Cached configuration object
    """
    if cache_key is None:
        cache_key = str(Path(config_file).resolve())

    if cache_key not in _global_config_cache:
        _global_config_cache[cache_key] = load_config(config_file, **kwargs)

    return _global_config_cache[cache_key]


def clear_global_config_cache() -> None:
    """Clear the global configuration cache."""
    global _global_config_cache
    _global_config_cache.clear()


# Export key classes and functions
__all__ = [
    'ConfigLoader',
    'load_config',
    'create_config_from_dict',
    'merge_config_files',
    'get_model_config',
    'validate_config_file',
    'get_config_template',
    'find_config_files',
    'load_environment_config',
    'get_cached_config',
    'clear_global_config_cache'
]