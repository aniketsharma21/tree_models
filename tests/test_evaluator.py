# tests/unit/test_evaluator.py
"""Unit tests for the ModelEvaluator class and evaluation functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from tree_models.models.evaluator import (
    ModelEvaluator, EvaluationConfig, EvaluationResults, evaluate_model, compare_models
)


class TestEvaluationConfig:
    """Test cases for EvaluationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        
        assert config.classification_threshold == 0.5
        assert config.compute_roc_auc == True
        assert config.compute_pr_auc == True
        assert config.confidence_level == 0.95
        assert config.bootstrap_samples == 1000
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid config
        config = EvaluationConfig(
            classification_threshold=0.7,
            confidence_level=0.90,
            bootstrap_samples=500
        )
        assert config.classification_threshold == 0.7
        
        # Invalid threshold
        with pytest.raises(Exception):
            EvaluationConfig(classification_threshold=1.5)
        
        # Invalid confidence level
        with pytest.raises(Exception):
            EvaluationConfig(confidence_level=1.1)


class TestEvaluationResults:
    """Test cases for EvaluationResults dataclass."""
    
    def test_results_creation(self):
        """Test EvaluationResults creation."""
        results = EvaluationResults(
            model_type="XGBClassifier",
            task_type="classification",
            n_samples=1000,
            n_features=10,
            metrics={"auc": 0.85, "accuracy": 0.78}
        )
        
        assert results.model_type == "XGBClassifier"
        assert results.task_type == "classification"
        assert results.n_samples == 1000
        assert results.metrics["auc"] == 0.85


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples), name='target')
        sample_weight = np.random.uniform(0.5, 2.0, n_samples)
        
        return X, y, sample_weight
    
    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples), name='target')
        sample_weight = np.random.uniform(0.5, 2.0, n_samples)
        
        return X, y, sample_weight
    
    @pytest.fixture
    def mock_classification_model(self):
        """Create a mock classification model."""
        model = Mock()
        model.predict.return_value = np.random.randint(0, 2, 1000)
        model.predict_proba.return_value = np.column_stack([
            np.random.uniform(0, 1, 1000),
            np.random.uniform(0, 1, 1000)
        ])
        return model
    
    @pytest.fixture
    def mock_regression_model(self):
        """Create a mock regression model."""
        model = Mock()
        model.predict.return_value = np.random.randn(1000)
        return model
    
    @pytest.fixture
    def evaluator(self):
        """Create a ModelEvaluator instance."""
        return ModelEvaluator(random_state=42)
    
    def test_evaluator_initialization(self, evaluator):
        """Test ModelEvaluator initialization."""
        assert evaluator.random_state == 42
        assert evaluator.enable_logging == True
    
    def test_validate_evaluation_inputs_valid(self, evaluator, sample_classification_data, mock_classification_model):
        """Test input validation with valid data."""
        X, y, sample_weight = sample_classification_data
        
        # Should not raise any exception
        evaluator._validate_evaluation_inputs(mock_classification_model, X, y, sample_weight)
    
    def test_validate_evaluation_inputs_invalid(self, evaluator, sample_classification_data):
        """Test input validation with invalid data."""
        X, y, sample_weight = sample_classification_data
        
        # Model without predict method
        bad_model = Mock()
        del bad_model.predict
        
        with pytest.raises(Exception):
            evaluator._validate_evaluation_inputs(bad_model, X, y, sample_weight)
        
        # Empty data
        with pytest.raises(Exception):
            evaluator._validate_evaluation_inputs(
                Mock(), pd.DataFrame(), pd.Series(dtype=float), None
            )
        
        # Mismatched lengths
        with pytest.raises(Exception):
            evaluator._validate_evaluation_inputs(
                Mock(), X, y.iloc[:-1], sample_weight
            )
    
    def test_determine_task_type(self, evaluator):
        """Test task type determination."""
        # Classification (binary)
        y_binary = pd.Series([0, 1, 0, 1, 1])
        assert evaluator._determine_task_type(y_binary) == "classification"
        
        # Classification (categorical)
        y_categorical = pd.Series(['A', 'B', 'A', 'C', 'B'])
        assert evaluator._determine_task_type(y_categorical) == "classification"
        
        # Regression (continuous)
        y_continuous = pd.Series([1.5, 2.7, -0.3, 4.2, 1.8])
        assert evaluator._determine_task_type(y_continuous) == "regression"
    
    def test_get_predictions_classification(self, evaluator, mock_classification_model, sample_classification_data):
        """Test prediction extraction for classification."""
        X, _, _ = sample_classification_data
        
        predictions = evaluator._get_predictions(mock_classification_model, X, "classification")
        
        assert 'raw' in predictions
        assert 'probabilities' in predictions
        assert len(predictions['raw']) == len(X)
        assert len(predictions['probabilities']) == len(X)
    
    def test_get_predictions_regression(self, evaluator, mock_regression_model, sample_regression_data):
        """Test prediction extraction for regression."""
        X, _, _ = sample_regression_data
        
        predictions = evaluator._get_predictions(mock_regression_model, X, "regression")
        
        assert 'raw' in predictions
        assert len(predictions['raw']) == len(X)
        assert 'probabilities' not in predictions
    
    def test_evaluate_classification(self, evaluator, mock_classification_model, sample_classification_data):
        """Test classification evaluation."""
        X, y, sample_weight = sample_classification_data
        config = EvaluationConfig()
        
        results = EvaluationResults(
            model_type="Mock",
            task_type="classification",
            n_samples=len(X),
            n_features=X.shape[1]
        )
        
        # Get predictions
        predictions = evaluator._get_predictions(mock_classification_model, X, "classification")
        
        # Run evaluation
        evaluator._evaluate_classification(results, y, predictions, sample_weight, config)
        
        # Check that metrics were computed
        expected_metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
        for metric in expected_metrics:
            if metric in results.metrics:
                assert isinstance(results.metrics[metric], float)
                assert not np.isnan(results.metrics[metric])
    
    def test_evaluate_regression(self, evaluator, mock_regression_model, sample_regression_data):
        """Test regression evaluation."""
        X, y, sample_weight = sample_regression_data
        config = EvaluationConfig()
        
        results = EvaluationResults(
            model_type="Mock",
            task_type="regression", 
            n_samples=len(X),
            n_features=X.shape[1]
        )
        
        # Get predictions
        predictions = evaluator._get_predictions(mock_regression_model, X, "regression")
        
        # Run evaluation  
        evaluator._evaluate_regression(results, y, predictions, sample_weight, config)
        
        # Check that metrics were computed
        expected_metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric in expected_metrics:
            if metric in results.metrics:
                assert isinstance(results.metrics[metric], float)
                assert not np.isnan(results.metrics[metric])
    
    def test_compute_confidence_intervals(self, evaluator, sample_classification_data):
        """Test confidence interval computation."""
        X, y, sample_weight = sample_classification_data
        
        # Mock predictions
        predictions = {
            'probabilities': np.random.uniform(0, 1, len(y))
        }
        
        config = EvaluationConfig(bootstrap_samples=100)  # Small number for testing
        
        ci = evaluator._compute_confidence_intervals(y, predictions, sample_weight, config)
        
        # Check that confidence intervals were computed
        assert isinstance(ci, dict)
        if 'auc' in ci:
            assert len(ci['auc']) == 2  # Lower and upper bounds
            assert ci['auc'][0] <= ci['auc'][1]  # Lower <= Upper
    
    def test_optimize_classification_threshold(self, evaluator, sample_classification_data):
        """Test threshold optimization."""
        X, y, sample_weight = sample_classification_data
        
        # Create more realistic predictions
        y_pred_proba = np.where(y == 1, 
                               np.random.uniform(0.6, 0.9, len(y)),
                               np.random.uniform(0.1, 0.4, len(y)))
        
        config = EvaluationConfig(threshold_metric="f1")
        
        threshold_analysis = evaluator._optimize_classification_threshold(
            y, y_pred_proba, sample_weight, config
        )
        
        assert 'optimal_threshold' in threshold_analysis
        assert 'optimal_metric_value' in threshold_analysis
        assert 0 <= threshold_analysis['optimal_threshold'] <= 1
        assert threshold_analysis['optimal_metric_value'] >= 0
    
    @patch('tree_models.models.evaluator.plt')
    def test_generate_report(self, mock_plt, evaluator, mock_classification_model, sample_classification_data):
        """Test report generation."""
        X, y, sample_weight = sample_classification_data
        
        # Create evaluation results
        results = evaluator.evaluate_model(
            mock_classification_model, X, y, sample_weight,
            config=EvaluationConfig(include_plots=False)  # Disable plots for testing
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generated_files = evaluator.generate_report(results, temp_dir, include_plots=False)
            
            # Check that files were created
            assert 'metrics' in generated_files
            assert generated_files['metrics'].exists()


class TestEvaluationConvenienceFunctions:
    """Test convenience functions for evaluation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y
    
    @patch('tree_models.models.evaluator.ModelEvaluator')
    def test_evaluate_model_function(self, mock_evaluator_class, sample_data):
        """Test the evaluate_model convenience function."""
        X, y = sample_data
        
        # Mock evaluator and results
        mock_evaluator = Mock()
        mock_results = Mock()
        mock_evaluator.evaluate_model.return_value = mock_results
        mock_evaluator_class.return_value = mock_evaluator
        
        # Mock model
        mock_model = Mock()
        
        # Call function
        result = evaluate_model(mock_model, X, y)
        
        # Verify
        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate_model.assert_called_once()
        assert result == mock_results
    
    @patch('tree_models.models.evaluator.ModelEvaluator')
    def test_compare_models_function(self, mock_evaluator_class, sample_data):
        """Test the compare_models convenience function."""
        X, y = sample_data
        
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        # Mock evaluation results
        mock_result1 = Mock()
        mock_result2 = Mock()
        mock_evaluator.evaluate_model.side_effect = [mock_result1, mock_result2]
        
        # Mock comparison result
        mock_comparison = {'best_model': 'model1', 'best_score': 0.85}
        mock_evaluator.compare_models.return_value = mock_comparison
        
        # Mock models
        models = {'model1': Mock(), 'model2': Mock()}
        
        # Call function
        result = compare_models(models, X, y)
        
        # Verify
        assert mock_evaluator.evaluate_model.call_count == 2
        mock_evaluator.compare_models.assert_called_once()
        assert result == mock_comparison


if __name__ == "__main__":
    pytest.main([__file__])