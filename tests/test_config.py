"""Unit tests for configuration system."""
import os
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
import yaml

from config.base_config import (
    ModelConfig, XGBoostConfig, LightGBMConfig, CatBoostConfig,
    DataConfig, ExperimentConfig, EnvironmentConfig
)
from config.config_loader import ConfigLoader, ConfigurationError, load_config
from config.config_schema import validate_model_config, validate_experiment_config


class TestBaseConfig:
    """Test base configuration classes."""

    @pytest.mark.unit
    def test_model_config_creation(self):
        """Test ModelConfig creation and to_dict method."""
        config = ModelConfig(
            model_type="test_model",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05
        )
        
        assert config.model_type == "test_model"
        assert config.n_estimators == 100
        assert config.max_depth == 5
        assert config.learning_rate == 0.05
        
        params_dict = config.to_dict()
        assert params_dict['n_estimators'] == 100
        assert params_dict['max_depth'] == 5
        assert params_dict['learning_rate'] == 0.05
        assert params_dict['random_state'] == 42

    @pytest.mark.unit
    def test_xgboost_config_creation(self):
        """Test XGBoostConfig creation and custom parameters."""
        config = XGBoostConfig(
            n_estimators=200,
            max_depth=8,
            subsample=0.9,
            reg_alpha=0.1
        )
        
        assert config.model_type == "xgboost"
        assert config.n_estimators == 200
        assert config.subsample == 0.9
        assert config.reg_alpha == 0.1
        
        params_dict = config.to_dict()
        assert params_dict['subsample'] == 0.9
        assert params_dict['objective'] == "binary:logistic"
        assert 'reg_alpha' in params_dict

    @pytest.mark.unit
    def test_lightgbm_config_creation(self):
        """Test LightGBMConfig creation."""
        config = LightGBMConfig(
            n_estimators=150,
            feature_fraction=0.7,
            min_child_samples=25
        )
        
        assert config.model_type == "lightgbm"
        assert config.feature_fraction == 0.7
        assert config.min_child_samples == 25
        
        params_dict = config.to_dict()
        assert params_dict['feature_fraction'] == 0.7
        assert params_dict['boosting_type'] == "gbdt"

    @pytest.mark.unit
    def test_catboost_config_creation(self):
        """Test CatBoostConfig creation."""
        config = CatBoostConfig(
            n_estimators=100,
            depth=8,
            l2_leaf_reg=2.0
        )
        
        assert config.model_type == "catboost"
        assert config.depth == 8
        assert config.l2_leaf_reg == 2.0
        
        params_dict = config.to_dict()
        assert params_dict['iterations'] == 100  # CatBoost uses 'iterations'
        assert params_dict['depth'] == 8  # CatBoost uses 'depth' not 'max_depth'

    @pytest.mark.unit
    def test_data_config_creation(self):
        """Test DataConfig creation."""
        config = DataConfig(
            train_path="data/train.csv",
            target_col="fraud_label",
            test_size=0.25,
            stratify=True
        )
        
        assert config.train_path == "data/train.csv"
        assert config.target_col == "fraud_label"
        assert config.test_size == 0.25
        assert config.stratify is True

    @pytest.mark.unit
    def test_experiment_config_creation(self):
        """Test ExperimentConfig creation and model retrieval."""
        model_config = XGBoostConfig(n_estimators=300)
        data_config = DataConfig(target_col="target")
        
        config = ExperimentConfig(
            model=model_config,
            data=data_config,
            seed=123
        )
        
        assert config.seed == 123
        assert config.model.model_type == "xgboost"
        assert config.data.target_col == "target"
        
        # Test get_model_config method
        lgbm_config = config.get_model_config("lightgbm")
        assert lgbm_config.model_type == "lightgbm"


class TestEnvironmentConfig:
    """Test environment configuration management."""

    @pytest.mark.unit
    def test_get_env_var_type_conversion(self):
        """Test environment variable type conversion."""
        # Test boolean conversion
        os.environ['TEST_BOOL_TRUE'] = 'true'
        os.environ['TEST_BOOL_FALSE'] = 'false'
        assert EnvironmentConfig.get_env_var('TEST_BOOL_TRUE') is True
        assert EnvironmentConfig.get_env_var('TEST_BOOL_FALSE') is False
        
        # Test integer conversion
        os.environ['TEST_INT'] = '42'
        assert EnvironmentConfig.get_env_var('TEST_INT') == 42
        
        # Test float conversion
        os.environ['TEST_FLOAT'] = '3.14'
        assert EnvironmentConfig.get_env_var('TEST_FLOAT') == 3.14
        
        # Test string (no conversion)
        os.environ['TEST_STRING'] = 'hello'
        assert EnvironmentConfig.get_env_var('TEST_STRING') == 'hello'

    @pytest.mark.unit 
    def test_from_environment(self):
        """Test extracting configuration from environment variables."""
        os.environ['MODEL_TYPE'] = 'lightgbm'
        os.environ['N_ESTIMATORS'] = '500'
        os.environ['LEARNING_RATE'] = '0.05'
        os.environ['TRAIN_PATH'] = '/data/train.csv'
        os.environ['ENABLE_TUNING'] = 'false'
        
        env_config = EnvironmentConfig.from_environment()
        
        assert env_config['model']['model_type'] == 'lightgbm'
        assert env_config['model']['n_estimators'] == 500
        assert env_config['model']['learning_rate'] == 0.05
        assert env_config['data']['train_path'] == '/data/train.csv'
        assert env_config['tuning']['enable_tuning'] is False


class TestConfigLoader:
    """Test configuration loading utilities."""

    @pytest.mark.unit
    def test_config_loader_initialization(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader()
        assert loader.validate is True
        assert loader.allow_environment_override is True
        
        loader = ConfigLoader(validate=False, allow_environment_override=False)
        assert loader.validate is False
        assert loader.allow_environment_override is False

    @pytest.mark.unit
    def test_load_yaml_file(self, temp_config_file: Path):
        """Test loading YAML configuration file."""
        loader = ConfigLoader()
        config_dict = loader.load_yaml(temp_config_file)
        
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert config_dict['model']['model_type'] == 'xgboost'

    @pytest.mark.unit
    def test_load_yaml_nonexistent_file(self):
        """Test loading non-existent YAML file raises error."""
        loader = ConfigLoader()
        
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            loader.load_yaml("nonexistent_file.yaml")

    @pytest.mark.unit
    def test_merge_configs(self):
        """Test merging configuration dictionaries."""
        loader = ConfigLoader()
        
        base_config = {
            'model': {'n_estimators': 100, 'max_depth': 6},
            'data': {'target_col': 'target'}
        }
        
        override_config = {
            'model': {'n_estimators': 200, 'learning_rate': 0.05},
            'tuning': {'enable_tuning': True}
        }
        
        merged = loader.merge_configs(base_config, override_config)
        
        assert merged['model']['n_estimators'] == 200  # overridden
        assert merged['model']['max_depth'] == 6  # preserved
        assert merged['model']['learning_rate'] == 0.05  # added
        assert merged['data']['target_col'] == 'target'  # preserved
        assert merged['tuning']['enable_tuning'] is True  # added

    @pytest.mark.unit
    def test_load_config_from_dict(self, sample_config_dict: Dict[str, Any]):
        """Test loading configuration from dictionary."""
        loader = ConfigLoader(validate=False)  # Skip validation for simplicity
        
        config = loader.load_config(base_config=sample_config_dict)
        
        assert isinstance(config, ExperimentConfig)
        assert config.model.model_type == 'xgboost'
        assert config.model.n_estimators == 100
        assert config.data.target_col == 'target'

    @pytest.mark.unit
    def test_load_config_with_model_type_override(self, sample_config_dict: Dict[str, Any]):
        """Test loading configuration with model type override."""
        loader = ConfigLoader(validate=False)
        
        config = loader.load_config(
            base_config=sample_config_dict,
            model_type='lightgbm'
        )
        
        assert config.model.model_type == 'lightgbm'
        assert isinstance(config.model, LightGBMConfig)

    @pytest.mark.unit
    def test_get_model_config(self, sample_config_dict: Dict[str, Any]):
        """Test getting specific model configuration."""
        loader = ConfigLoader(validate=False)
        
        # Test with ExperimentConfig
        experiment_config = loader.load_config(base_config=sample_config_dict)
        model_config = loader.get_model_config(experiment_config, 'catboost')
        assert isinstance(model_config, CatBoostConfig)
        
        # Test with dictionary
        model_config = loader.get_model_config(sample_config_dict, 'lightgbm')
        assert isinstance(model_config, LightGBMConfig)

    @pytest.mark.unit 
    def test_save_and_load_yaml(self, test_data_dir: Path, sample_config_dict: Dict[str, Any]):
        """Test saving and loading YAML configuration."""
        loader = ConfigLoader()
        
        # Save configuration
        output_file = test_data_dir / "saved_config.yaml"
        loader.save_yaml(sample_config_dict, output_file)
        
        assert output_file.exists()
        
        # Load saved configuration
        loaded_config = loader.load_yaml(output_file)
        assert loaded_config == sample_config_dict

    @pytest.mark.unit
    def test_load_config_with_environment_override(self, temp_config_file: Path):
        """Test configuration loading with environment overrides."""
        os.environ['N_ESTIMATORS'] = '999'
        os.environ['LEARNING_RATE'] = '0.001'
        
        loader = ConfigLoader(validate=False, allow_environment_override=True)
        config = loader.load_config(temp_config_file)
        
        assert config.model.n_estimators == 999
        assert config.model.learning_rate == 0.001

    @pytest.mark.unit
    def test_validation_error_handling(self, temp_config_file: Path):
        """Test that validation errors are properly handled."""
        # Create invalid config
        invalid_config = {'model': {'model_type': 'invalid_model'}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            invalid_file = Path(f.name)
        
        try:
            loader = ConfigLoader(validate=True)
            
            with pytest.raises(ConfigurationError, match="Configuration validation failed"):
                loader.load_config(invalid_file)
        finally:
            invalid_file.unlink()  # Clean up


class TestConvenienceFunctions:
    """Test convenience functions for configuration loading."""

    @pytest.mark.unit
    def test_load_config_function(self, temp_config_file: Path):
        """Test load_config convenience function."""
        config = load_config(temp_config_file, validate=False)
        
        assert isinstance(config, ExperimentConfig)
        assert config.model.model_type == 'xgboost'

    @pytest.mark.unit
    def test_load_config_with_model_type(self, temp_config_file: Path):
        """Test load_config with model type override."""
        config = load_config(temp_config_file, model_type='lightgbm', validate=False)
        
        assert config.model.model_type == 'lightgbm'
        assert isinstance(config.model, LightGBMConfig)


class TestConfigValidation:
    """Test configuration validation functionality."""

    @pytest.mark.unit
    @pytest.mark.skipif(
        not pytest.importorskip("pydantic", minversion="1.10.0"), 
        reason="Pydantic not available"
    )
    def test_validate_model_config(self):
        """Test model configuration validation."""
        valid_config = {
            'model_type': 'xgboost',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
        
        validated = validate_model_config(valid_config)
        assert validated['model_type'] == 'xgboost'
        assert validated['n_estimators'] == 100

    @pytest.mark.unit
    @pytest.mark.skipif(
        not pytest.importorskip("pydantic", minversion="1.10.0"),
        reason="Pydantic not available" 
    )
    def test_validate_invalid_model_config(self):
        """Test validation of invalid model configuration."""
        invalid_config = {
            'model_type': 'invalid_model',
            'n_estimators': -1,  # Invalid negative value
            'learning_rate': 2.0  # Invalid > 1.0
        }
        
        with pytest.raises(Exception):  # Should raise validation error
            validate_model_config(invalid_config)

    @pytest.mark.unit
    @pytest.mark.skipif(
        not pytest.importorskip("pydantic", minversion="1.10.0"),
        reason="Pydantic not available"
    )
    def test_validate_experiment_config(self, sample_config_dict: Dict[str, Any]):
        """Test experiment configuration validation."""
        validated = validate_experiment_config(sample_config_dict)
        
        assert 'model' in validated
        assert 'data' in validated
        assert validated['model']['model_type'] == 'xgboost'


class TestConfigCaching:
    """Test configuration caching functionality."""

    @pytest.mark.unit
    def test_config_caching(self, temp_config_file: Path):
        """Test configuration caching works correctly."""
        from config.config_loader import get_cached_config, clear_config_cache
        
        clear_config_cache()
        
        # First load
        config1 = get_cached_config(temp_config_file)
        
        # Second load should use cache
        config2 = get_cached_config(temp_config_file)
        
        # Should be the same object (cached)
        assert config1 is config2

    @pytest.mark.unit
    def test_cache_clearing(self, temp_config_file: Path):
        """Test cache clearing functionality."""
        from config.config_loader import get_cached_config, clear_config_cache
        
        # Load config to cache
        config1 = get_cached_config(temp_config_file)
        
        # Clear cache
        clear_config_cache()
        
        # Load again - should be different object
        config2 = get_cached_config(temp_config_file)
        
        assert config1 is not config2