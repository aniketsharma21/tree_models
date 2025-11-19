# tests/unit/test_trainer.py
"""Unit tests for the ModelTrainer class and training functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from tree_models.models.trainer import ModelTrainer, TrainingConfig, TrainingResults, train_model
from tree_models.config.model_config import XGBoostConfig, LightGBMConfig


class TestTrainingConfig:
    """Test cases for TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.enable_early_stopping == True
        assert config.early_stopping_rounds == 100
        assert config.validation_fraction == 0.2
        assert config.cv_folds == 5
        assert config.verbose == True

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid config
        config = TrainingConfig(validation_fraction=0.3, cv_folds=10, early_stopping_rounds=50)
        assert config.validation_fraction == 0.3

        # Invalid validation fraction
        with pytest.raises(Exception):
            TrainingConfig(validation_fraction=0.8)  # Too high

        # Invalid cv_folds
        with pytest.raises(Exception):
            TrainingConfig(cv_folds=1)  # Too low


class TestTrainingResults:
    """Test cases for TrainingResults dataclass."""

    def test_results_creation(self):
        """Test TrainingResults creation and attributes."""
        mock_model = Mock()

        results = TrainingResults(
            model=mock_model,
            model_type="xgboost",
            training_time=120.5,
            train_metrics={"auc": 0.85, "accuracy": 0.78},
            validation_metrics={"auc": 0.82, "accuracy": 0.75},
        )

        assert results.model == mock_model
        assert results.model_type == "xgboost"
        assert results.training_time == 120.5
        assert results.train_metrics["auc"] == 0.85
        assert results.validation_metrics["auc"] == 0.82
        assert results.best_iteration is None  # Default value
        assert results.early_stopped == False  # Default value


class SimpleModel:
    """Simple model class for testing pickling."""
    def save_model(self, path):
        pass

class TestModelTrainer:
    """Test cases for ModelTrainer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5

        X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)])
        y = pd.Series(np.random.randint(0, 2, n_samples), name="target")

        # Create sample weights
        sample_weight = np.random.uniform(0.5, 2.0, n_samples)

        return X, y, sample_weight

    @pytest.fixture
    def trainer(self):
        """Create a ModelTrainer instance for testing."""
        return ModelTrainer(random_state=42)

    def test_trainer_initialization(self, trainer):
        """Test ModelTrainer initialization."""
        assert trainer.random_state == 42
        assert trainer.enable_logging == True
        assert trainer.checkpoint_dir is None
        assert trainer._current_model is None

    def test_trainer_with_checkpoint_dir(self):
        """Test ModelTrainer with checkpoint directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(checkpoint_dir=temp_dir)
            assert trainer.checkpoint_dir == Path(temp_dir)
            assert trainer.checkpoint_dir is not None
            assert trainer.checkpoint_dir.exists()

    def test_validate_training_inputs_valid(self, trainer, sample_data):
        """Test input validation with valid data."""
        X, y, sample_weight = sample_data
        X_valid = X.iloc[:100].copy()
        y_valid = y.iloc[:100].copy()
        weight_valid = sample_weight[:100]

        # Should not raise any exception
        trainer._validate_training_inputs(X, y, X_valid, y_valid, sample_weight, weight_valid)

    def test_validate_training_inputs_invalid(self, trainer, sample_data):
        """Test input validation with invalid data."""
        X, y, sample_weight = sample_data

        # Empty data
        with pytest.raises(Exception):
            trainer._validate_training_inputs(pd.DataFrame(), pd.Series(dtype=float), None, None, None, None)

        # Mismatched lengths
        with pytest.raises(Exception):
            trainer._validate_training_inputs(X, y.iloc[:-1], None, None, None, None)

        # Negative sample weights
        bad_weights = sample_weight.copy()
        bad_weights[0] = -1.0
        with pytest.raises(Exception):
            trainer._validate_training_inputs(X, y, None, None, bad_weights, None)

    def test_create_validation_split(self, trainer, sample_data):
        """Test validation split creation."""
        X, y, sample_weight = sample_data

        # Without sample weights
        X_train, X_valid, y_train, y_valid, w_train, w_valid = trainer._create_validation_split(X, y, None, 0.2)

        assert len(X_train) + len(X_valid) == len(X)
        assert len(y_train) + len(y_valid) == len(y)
        assert w_train is None and w_valid is None

        # With sample weights
        X_train, X_valid, y_train, y_valid, w_train, w_valid = trainer._create_validation_split(
            X, y, sample_weight, 0.2
        )

        assert len(w_train) + len(w_valid) == len(sample_weight)
        assert w_train is not None and w_valid is not None

    def test_compute_metrics_classification(self, trainer, sample_data):
        """Test metrics computation for classification."""
        X, y, sample_weight = sample_data

        # Mock predictions
        y_pred = np.random.uniform(0, 1, len(y))

        metrics = trainer._compute_metrics(y, y_pred, sample_weight)

        # Check that required metrics are present
        expected_metrics = ["auc", "accuracy", "precision", "recall", "logloss"]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert not np.isnan(metrics[metric])

    def test_compute_metrics_regression(self, trainer):
        """Test metrics computation for regression."""
        # Create regression data
        y_true = pd.Series(np.random.randn(100))
        y_pred = y_true + np.random.randn(100) * 0.1  # Add some noise

        metrics = trainer._compute_metrics(y_true, y_pred)

        # Check that required metrics are present
        expected_metrics = ["mse", "rmse", "mae", "r2"]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert not np.isnan(metrics[metric])

    @patch("tree_models.models.trainer.xgb")
    def test_train_xgboost_mock(self, mock_xgb, trainer, sample_data):
        """Test XGBoost training with mocked XGBoost."""
        X, y, sample_weight = sample_data

        # Mock XGBoost components
        mock_model = Mock()
        mock_model.predict.return_value = np.random.uniform(0, 1, len(y))
        mock_model.best_iteration = 50  # Set a valid integer
        mock_xgb.train.return_value = mock_model
        mock_xgb.DMatrix = Mock()

        # Mock model config
        model_config = Mock()
        model_config.model_type = "xgboost"
        model_config.get_params.return_value = {"n_estimators": 100, "max_depth": 6}

        # Mock training config
        training_config = TrainingConfig()

        # Call the method
        results = trainer._train_xgboost(
            mock_model, model_config, X, y, None, None, sample_weight, None, {"n_estimators": 100}, training_config
        )

        # Verify results
        assert isinstance(results, TrainingResults)
        assert results.model_type == "xgboost"
        assert "auc" in results.train_metrics

    def test_save_and_load_model(self, trainer):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model"

            # Mock model and results
            mock_model = SimpleModel()
            
            mock_results = TrainingResults(
                model=mock_model,
                model_type="test",
                training_time=100.0,
                train_metrics={"accuracy": 0.85},
                validation_metrics={"accuracy": 0.80},
            )

            # Test saving
            trainer.save_model(mock_model, model_path, mock_results)

            # Check that files were created
            assert (model_path.parent / "test_model.pkl").exists()
            assert (model_path.parent / "test_model.metadata.json").exists()


class TestTrainingConvenienceFunctions:
    """Test convenience functions for training."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y

    @patch("tree_models.models.trainer.ModelTrainer")
    def test_train_model_function(self, mock_trainer_class, sample_data):
        """Test the train_model convenience function."""
        X, y = sample_data

        # Mock trainer and its methods
        mock_trainer = Mock()
        mock_results = Mock()
        mock_trainer.train_model.return_value = mock_results
        mock_trainer_class.return_value = mock_trainer

        # Mock model config
        mock_model_config = Mock()

        # Call function
        result = train_model(mock_model_config, X, y)

        # Verify that trainer was created and called correctly
        mock_trainer_class.assert_called_once()
        mock_trainer.train_model.assert_called_once()
        assert result == mock_results


# Integration test placeholders (would require actual model libraries)
class TestModelTrainerIntegration:
    """Integration tests that would require actual ML libraries."""

    @pytest.mark.skipif(True, reason="Requires XGBoost installation")
    def test_real_xgboost_training(self):
        """Test with real XGBoost (skipped by default)."""
        # This would test actual XGBoost training
        # Only runs if XGBoost is available and test is explicitly enabled
        pass

    @pytest.mark.skipif(True, reason="Requires LightGBM installation")
    def test_real_lightgbm_training(self):
        """Test with real LightGBM (skipped by default)."""
        # This would test actual LightGBM training
        pass

    @pytest.mark.skipif(True, reason="Requires CatBoost installation")
    def test_real_catboost_training(self):
        """Test with real CatBoost (skipped by default)."""
        # This would test actual CatBoost training
        pass


if __name__ == "__main__":
    pytest.main([__file__])
