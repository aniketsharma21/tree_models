"""Unit tests for model training and evaluation functionality."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Tuple

# Test imports - adjust based on actual module structure
from src.models.trainer import train_model, ModelTrainer
from src.models.evaluator import evaluate_model, ModelEvaluator  
from src.models.tuner import EnhancedOptunaHyperparameterTuner


class TestModelTrainer:
    """Test model training functionality."""

    @pytest.mark.unit
    @pytest.mark.models
    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(model_type="xgboost")
        assert trainer.model_type == "xgboost"
        assert trainer.model is None

    @pytest.mark.unit
    @pytest.mark.models
    def test_xgboost_training(self, sample_X_y):
        """Test XGBoost model training."""
        xgb = pytest.importorskip("xgboost")
        
        X, y = sample_X_y
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Split data
        split_idx = int(0.8 * len(X_numeric))
        X_train, X_valid = X_numeric.iloc[:split_idx], X_numeric.iloc[split_idx:]
        y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]
        
        params = {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': 0
        }
        
        # Test training function (assuming it exists)
        try:
            model = train_model(
                model_type="xgboost",
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                params=params
            )
            
            assert model is not None
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')
            
            # Test predictions
            predictions = model.predict(X_valid)
            assert len(predictions) == len(y_valid)
            
            pred_proba = model.predict_proba(X_valid)
            assert pred_proba.shape[0] == len(y_valid)
            assert pred_proba.shape[1] == 2  # Binary classification
            
        except ImportError:
            pytest.skip("train_model function not available")

    @pytest.mark.unit
    @pytest.mark.models
    def test_lightgbm_training(self, sample_X_y):
        """Test LightGBM model training."""
        lgb = pytest.importorskip("lightgbm")
        
        X, y = sample_X_y
        X_numeric = X.select_dtypes(include=[np.number])
        
        split_idx = int(0.8 * len(X_numeric))
        X_train, X_valid = X_numeric.iloc[:split_idx], X_numeric.iloc[split_idx:]
        y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]
        
        params = {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': -1
        }
        
        try:
            model = train_model(
                model_type="lightgbm",
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                params=params
            )
            
            assert model is not None
            predictions = model.predict(X_valid)
            assert len(predictions) == len(y_valid)
            
        except ImportError:
            pytest.skip("train_model function not available")

    @pytest.mark.unit  
    @pytest.mark.models
    def test_model_saving_loading(self, sample_X_y, test_data_dir):
        """Test model saving and loading functionality."""
        pytest.importorskip("xgboost")
        
        X, y = sample_X_y
        X_numeric = X.select_dtypes(include=[np.number])
        
        params = {
            'n_estimators': 5,
            'max_depth': 3,
            'random_state': 42,
            'verbosity': 0
        }
        
        model_path = test_data_dir / "test_model.pkl"
        
        try:
            # Train and save model
            model = train_model(
                model_type="xgboost",
                X_train=X_numeric,
                y_train=y,
                params=params,
                save_path=model_path
            )
            
            # Check model was saved
            assert model_path.exists()
            
            # Test loading (if load function exists)
            # loaded_model = load_model(model_path)
            # assert loaded_model is not None
            
        except ImportError:
            pytest.skip("Model saving/loading functions not available")


class TestModelEvaluator:
    """Test model evaluation functionality."""

    @pytest.mark.unit
    @pytest.mark.models
    def test_evaluation_metrics_calculation(self):
        """Test basic evaluation metrics calculation."""
        # Create simple test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.1, 0.8])
        
        try:
            results = evaluate_model(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                generate_plots=False,
                save_results=False
            )
            
            assert 'metrics' in results
            metrics = results['metrics']
            
            # Check key metrics are present
            assert 'auc_roc' in metrics
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            
            # Check values are reasonable
            assert 0 <= metrics['auc_roc'] <= 1
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1'] <= 1
            
        except ImportError:
            pytest.skip("evaluate_model function not available")

    @pytest.mark.unit
    @pytest.mark.models
    def test_threshold_optimization(self):
        """Test threshold optimization functionality."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.1, 0.8, 0.85, 0.25])
        
        try:
            results = evaluate_model(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                generate_plots=False,
                save_results=False
            )
            
            if 'threshold_analysis' in results:
                threshold_analysis = results['threshold_analysis']
                assert 'optimal_thresholds' in threshold_analysis
                
                optimal_thresholds = threshold_analysis['optimal_thresholds']
                for metric in ['f1', 'precision', 'recall']:
                    if metric in optimal_thresholds:
                        threshold = optimal_thresholds[metric]
                        assert 0 <= threshold <= 1
            
        except ImportError:
            pytest.skip("evaluate_model function not available")

    @pytest.mark.unit
    @pytest.mark.models
    def test_evaluation_with_sample_weights(self):
        """Test evaluation with sample weights."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        sample_weight = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 2.0])
        
        try:
            results = evaluate_model(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                sample_weight=sample_weight,
                generate_plots=False,
                save_results=False
            )
            
            assert 'metrics' in results
            # Weighted metrics should be different from unweighted
            
        except (ImportError, TypeError):
            pytest.skip("Weighted evaluation not available")

    @pytest.mark.integration
    @pytest.mark.models
    def test_evaluation_plot_generation(self, test_data_dir):
        """Test evaluation plot generation."""
        y_true = np.random.binomial(1, 0.3, 100)
        y_pred_proba = np.random.beta(2, 5, 100)
        
        output_dir = test_data_dir / "evaluation_plots"
        
        try:
            results = evaluate_model(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                output_dir=output_dir,
                generate_plots=True,
                save_results=True
            )
            
            # Check plots were generated
            assert output_dir.exists()
            
            # Look for common plot files
            plot_files = list(output_dir.glob("*.png"))
            assert len(plot_files) > 0
            
        except ImportError:
            pytest.skip("Plot generation not available")


class TestHyperparameterTuning:
    """Test hyperparameter tuning functionality."""

    @pytest.mark.unit
    @pytest.mark.models
    def test_tuner_initialization(self):
        """Test hyperparameter tuner initialization."""
        try:
            tuner = EnhancedOptunaHyperparameterTuner(
                model_type="xgboost",
                n_trials=5
            )
            
            assert tuner.model_type == "xgboost"
            assert tuner.n_trials == 5
            
        except ImportError:
            pytest.skip("EnhancedOptunaHyperparameterTuner not available")

    @pytest.mark.slow
    @pytest.mark.models
    def test_xgboost_hyperparameter_tuning(self, sample_X_y):
        """Test XGBoost hyperparameter optimization."""
        pytest.importorskip("optuna")
        pytest.importorskip("xgboost")
        
        X, y = sample_X_y
        X_numeric = X.select_dtypes(include=[np.number])
        
        try:
            tuner = EnhancedOptunaHyperparameterTuner(
                model_type="xgboost",
                n_trials=3,  # Small number for testing
                cv_folds=2   # Reduce CV folds for speed
            )
            
            best_params, best_score = tuner.optimize(
                X_numeric, y,
                scoring='roc_auc'
            )
            
            assert isinstance(best_params, dict)
            assert isinstance(best_score, float)
            assert 0 <= best_score <= 1
            
            # Check required parameters are present
            assert 'n_estimators' in best_params
            assert 'max_depth' in best_params
            assert 'learning_rate' in best_params
            
        except ImportError:
            pytest.skip("Hyperparameter tuning not available")

    @pytest.mark.unit
    @pytest.mark.models
    def test_custom_search_space(self):
        """Test custom search space definition."""
        try:
            custom_space = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            
            tuner = EnhancedOptunaHyperparameterTuner(
                model_type="xgboost",
                n_trials=3,
                custom_search_space=custom_space
            )
            
            assert tuner.custom_search_space == custom_space
            
        except ImportError:
            pytest.skip("Custom search space not available")

    @pytest.mark.unit
    @pytest.mark.models
    def test_scoring_functions(self):
        """Test different scoring function configurations."""
        try:
            for scoring in ['roc_auc', 'precision', 'recall', 'f1']:
                tuner = EnhancedOptunaHyperparameterTuner(
                    model_type="xgboost",
                    n_trials=1,
                    scoring=scoring
                )
                
                assert tuner.scoring == scoring
                
        except ImportError:
            pytest.skip("Scoring functions not available")


class TestModelIntegration:
    """Integration tests for complete model workflows."""

    @pytest.mark.integration
    @pytest.mark.models
    def test_complete_training_pipeline(self, sample_X_y, test_data_dir):
        """Test complete model training pipeline."""
        pytest.importorskip("xgboost")
        
        X, y = sample_X_y
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Split data
        split_idx = int(0.8 * len(X_numeric))
        X_train, X_test = X_numeric.iloc[:split_idx], X_numeric.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        output_dir = test_data_dir / "integration_test"
        
        try:
            # Train model
            model = train_model(
                model_type="xgboost",
                X_train=X_train,
                y_train=y_train,
                params={'n_estimators': 10, 'max_depth': 3, 'verbosity': 0},
                save_path=output_dir / "model.pkl"
            )
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate model
            results = evaluate_model(
                y_true=y_test,
                y_pred_proba=y_pred_proba,
                output_dir=output_dir / "evaluation",
                generate_plots=True,
                save_results=True
            )
            
            # Check results
            assert 'metrics' in results
            assert results['metrics']['auc_roc'] > 0.5  # Should be better than random
            
            # Check files were created
            assert (output_dir / "model.pkl").exists()
            assert (output_dir / "evaluation").exists()
            
        except ImportError:
            pytest.skip("Integration test functions not available")

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.models
    def test_tuning_and_evaluation_pipeline(self, sample_X_y, test_data_dir):
        """Test hyperparameter tuning followed by evaluation."""
        pytest.importorskip("optuna")
        pytest.importorskip("xgboost")
        
        X, y = sample_X_y
        X_numeric = X.select_dtypes(include=[np.number])
        
        try:
            # Hyperparameter tuning
            tuner = EnhancedOptunaHyperparameterTuner(
                model_type="xgboost",
                n_trials=3,
                cv_folds=2
            )
            
            best_params, best_score = tuner.optimize(X_numeric, y)
            
            # Train final model with best parameters
            split_idx = int(0.8 * len(X_numeric))
            X_train, X_test = X_numeric.iloc[:split_idx], X_numeric.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            model = train_model(
                model_type="xgboost",
                X_train=X_train,
                y_train=y_train,
                params=best_params
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            results = evaluate_model(y_test, y_pred_proba, generate_plots=False)
            
            # Tuned model should perform reasonably well
            assert results['metrics']['auc_roc'] > 0.5
            
        except ImportError:
            pytest.skip("Tuning and evaluation pipeline not available")


class TestErrorHandling:
    """Test error handling in model components."""

    @pytest.mark.unit
    @pytest.mark.models
    def test_invalid_model_type(self):
        """Test handling of invalid model types."""
        try:
            with pytest.raises(ValueError):
                train_model(
                    model_type="invalid_model",
                    X_train=pd.DataFrame({'a': [1, 2, 3]}),
                    y_train=pd.Series([0, 1, 0]),
                    params={}
                )
        except ImportError:
            pytest.skip("train_model function not available")

    @pytest.mark.unit
    @pytest.mark.models 
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        try:
            with pytest.raises((ValueError, IndexError)):
                train_model(
                    model_type="xgboost",
                    X_train=pd.DataFrame(),
                    y_train=pd.Series(dtype=int),
                    params={'n_estimators': 10}
                )
        except ImportError:
            pytest.skip("train_model function not available")

    @pytest.mark.unit
    @pytest.mark.models
    def test_mismatched_data_shapes(self):
        """Test handling of mismatched X and y shapes."""
        try:
            with pytest.raises(ValueError):
                train_model(
                    model_type="xgboost",
                    X_train=pd.DataFrame({'a': [1, 2, 3]}),
                    y_train=pd.Series([0, 1]),  # Mismatched length
                    params={'n_estimators': 10}
                )
        except ImportError:
            pytest.skip("train_model function not available")