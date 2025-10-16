# tests/integration/test_end_to_end_pipeline.py
"""Integration tests for complete ML pipeline workflows."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from tree_models.config.data_config import DataConfig, FeatureConfig, DataSourceConfig
from tree_models.data.validator import DataValidator, ValidationConfig
from tree_models.data.preprocessor import AdvancedDataPreprocessor, ColumnConfig
from tree_models.data.feature_engineer import FeatureEngineer, FeatureEngineeringConfig
from tree_models.models.trainer import ModelTrainer, TrainingConfig
from tree_models.models.evaluator import ModelEvaluator, EvaluationConfig


class TestEndToEndPipeline:
    """Test complete ML pipeline from data loading to model evaluation."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a comprehensive sample dataset for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create diverse feature types
        data = {
            # Numeric features
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.lognormal(10, 1, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            
            # Categorical features
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'occupation': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Artist', 'Other'], n_samples),
            'city': np.random.choice([f'City_{i}' for i in range(20)], n_samples),
            
            # Date feature
            'application_date': pd.date_range('2020-01-01', periods=n_samples, freq='D')[:n_samples],
            
            # Target variable (binary classification)
            'approved': np.random.binomial(1, 0.3, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Introduce some missing values
        missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        df.loc[missing_indices, 'income'] = np.nan
        
        # Add some outliers
        outlier_indices = np.random.choice(n_samples, size=10, replace=False)
        df.loc[outlier_indices, 'credit_score'] = np.random.uniform(300, 350, 10)
        
        return df
    
    @pytest.fixture
    def data_config(self):
        """Create a comprehensive data configuration."""
        return DataConfig(
            source=DataSourceConfig(file_format="dataframe"),
            features=FeatureConfig(
                target_column="approved",
                categorical_features=['education', 'occupation', 'city'],
                numeric_features=['age', 'income', 'credit_score'],
                date_features=['application_date']
            )
        )
    
    def test_data_validation_pipeline(self, sample_dataset):
        """Test the data validation pipeline."""
        # Initialize validator
        validator = DataValidator(random_state=42)
        
        # Configure validation
        config = ValidationConfig(
            check_missing_values=True,
            check_outliers=True,
            check_distributions=True,
            generate_report=False
        )
        
        # Run validation
        results = validator.validate_dataset(
            sample_dataset,
            target_column='approved',
            config=config
        )
        
        # Verify validation results
        assert isinstance(results.data_quality_score, float)
        assert 0 <= results.data_quality_score <= 100
        assert results.n_samples == len(sample_dataset)
        assert results.n_features == len(sample_dataset.columns)
        
        # Check that validation found issues (we introduced missing values and outliers)
        assert len(results.validation_warnings) > 0
        assert 'missing_value_analysis' in results.__dict__
        assert 'outlier_analysis' in results.__dict__
    
    def test_preprocessing_pipeline(self, sample_dataset, data_config):
        """Test the data preprocessing pipeline."""
        # Configure preprocessing
        column_configs = {
            'income': ColumnConfig(
                missing_strategy='median',
                scaling_strategy='standard',
                outlier_method='clip'
            ),
            'education': ColumnConfig(
                encoding_strategy='onehot',
                missing_strategy='constant',
                missing_constant='Unknown'
            ),
            'city': ColumnConfig(
                encoding_strategy='target',
                missing_strategy='most_frequent'
            )
        }
        
        # Initialize preprocessor
        preprocessor = AdvancedDataPreprocessor()
        preprocessor.set_column_configs(column_configs)
        
        # Prepare data
        X = sample_dataset.drop('approved', axis=1)
        y = sample_dataset['approved']
        
        # Fit and transform
        X_processed = preprocessor.fit_transform(X, y)
        
        # Verify preprocessing results
        assert len(X_processed) == len(X)
        assert X_processed.shape[1] >= X.shape[1]  # Should have same or more columns due to encoding
        
        # Check that missing values were handled
        assert X_processed.isnull().sum().sum() == 0  # No missing values should remain
        
        # Transform new data (test consistency)
        X_new_processed = preprocessor.transform(X.iloc[:100])
        assert X_new_processed.shape[1] == X_processed.shape[1]
    
    def test_feature_engineering_pipeline(self, sample_dataset):
        """Test the feature engineering pipeline."""
        # Configure feature engineering
        config = FeatureEngineeringConfig(
            extract_date_features=['application_date'],
            log_transform_cols=['income'],
            create_ratios=[('income', 'age')],
            create_bins={'credit_score': 5}
        )
        
        # Initialize feature engineer
        engineer = FeatureEngineer(random_state=42)
        
        # Engineer features
        results = engineer.engineer_features(
            sample_dataset,
            config=config,
            target_column='approved'
        )
        
        # Verify feature engineering results
        assert results.n_features_after > results.n_features_before
        assert len(results.new_features) > 0
        assert len(results.transformations_applied) > 0
        
        # Check specific transformations
        expected_features = [
            'application_date_year', 'application_date_month',
            'income_log', 'income_div_age', 'credit_score_binned'
        ]
        
        for feature in expected_features:
            if feature in results.data.columns:
                assert not results.data[feature].isnull().all()
    
    @patch('tree_models.models.trainer.xgb')
    def test_training_pipeline_mock(self, mock_xgb, sample_dataset, data_config):
        """Test the model training pipeline with mocked XGBoost."""
        # Mock XGBoost components
        mock_model = Mock()
        mock_model.predict.return_value = np.random.uniform(0, 1, len(sample_dataset))
        mock_xgb.train.return_value = mock_model
        mock_xgb.DMatrix = Mock()
        
        # Prepare data
        X = sample_dataset.drop('approved', axis=1).select_dtypes(include=[np.number])
        y = sample_dataset['approved']
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Configure training
        from tree_models.config.model_config import XGBoostConfig
        model_config = XGBoostConfig(n_estimators=10, max_depth=3)
        training_config = TrainingConfig(
            enable_early_stopping=False,  # Simplify for testing
            enable_cross_validation=False,
            verbose=False
        )
        
        # Initialize trainer
        trainer = ModelTrainer(random_state=42)
        
        # Train model
        results = trainer.train_model(
            model_config, X_train, y_train,
            X_test, y_test,
            training_config=training_config
        )
        
        # Verify training results
        assert results.model_type == 'xgboost'
        assert isinstance(results.training_time, float)
        assert 'auc' in results.train_metrics
        assert 'auc' in results.validation_metrics
    
    def test_evaluation_pipeline(self, sample_dataset):
        """Test the model evaluation pipeline."""
        # Create mock model with realistic behavior
        mock_model = Mock()
        
        # Create predictions that correlate with actual target
        X = sample_dataset.drop('approved', axis=1).select_dtypes(include=[np.number])
        y = sample_dataset['approved']
        
        # Generate somewhat realistic predictions
        predictions = np.where(
            y == 1,
            np.random.uniform(0.6, 0.9, len(y)),  # Higher scores for positive class
            np.random.uniform(0.1, 0.4, len(y))   # Lower scores for negative class
        )
        
        mock_model.predict.return_value = (predictions > 0.5).astype(int)
        mock_model.predict_proba.return_value = np.column_stack([1 - predictions, predictions])
        
        # Configure evaluation
        config = EvaluationConfig(
            compute_roc_auc=True,
            compute_pr_auc=True,
            optimize_threshold=True,
            compute_confidence_intervals=False,  # Skip for speed
            include_plots=False
        )
        
        # Initialize evaluator
        evaluator = ModelEvaluator(random_state=42)
        
        # Evaluate model
        results = evaluator.evaluate_model(
            mock_model, X, y,
            config=config
        )
        
        # Verify evaluation results
        assert results.task_type == "classification"
        assert 'auc' in results.metrics
        assert 'accuracy' in results.metrics
        assert results.optimal_threshold is not None
        assert 0 <= results.optimal_threshold <= 1
        
        # Check ROC data
        if results.roc_data:
            assert 'fpr' in results.roc_data
            assert 'tpr' in results.roc_data
            assert 'auc' in results.roc_data
    
    def test_complete_pipeline_integration(self, sample_dataset):
        """Test complete pipeline from validation to evaluation."""
        # Step 1: Data Validation
        validator = DataValidator()
        validation_results = validator.validate_dataset(
            sample_dataset,
            target_column='approved',
            config=ValidationConfig(generate_report=False)
        )
        
        assert validation_results.is_valid or len(validation_results.validation_errors) == 0
        
        # Step 2: Data Preprocessing
        X = sample_dataset.drop('approved', axis=1)
        y = sample_dataset['approved']
        
        # Simple preprocessing for integration test
        preprocessor = AdvancedDataPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)
        
        # Step 3: Feature Engineering (simple)
        engineer = FeatureEngineer()
        fe_config = FeatureEngineeringConfig(
            extract_date_features=['application_date'] if 'application_date' in X.columns else None
        )
        fe_results = engineer.engineer_features(X_processed, fe_config, target_column=None)
        X_final = fe_results.data.select_dtypes(include=[np.number])  # Keep only numeric for simplicity
        
        # Step 4: Mock Training (simplified)
        mock_model = Mock()
        predictions = np.random.uniform(0.3, 0.7, len(y))
        mock_model.predict_proba.return_value = np.column_stack([1 - predictions, predictions])
        mock_model.predict.return_value = (predictions > 0.5).astype(int)
        
        # Step 5: Evaluation
        evaluator = ModelEvaluator()
        eval_results = evaluator.evaluate_model(
            mock_model, X_final, y,
            config=EvaluationConfig(include_plots=False)
        )
        
        # Verify complete pipeline
        assert eval_results.n_samples == len(sample_dataset)
        assert 'auc' in eval_results.metrics
        
        # Log pipeline success metrics
        pipeline_metrics = {
            'data_quality_score': validation_results.data_quality_score,
            'features_engineered': fe_results.n_features_after - fe_results.n_features_before,
            'final_auc': eval_results.metrics.get('auc', 0),
            'pipeline_success': True
        }
        
        assert pipeline_metrics['pipeline_success']
        assert isinstance(pipeline_metrics['data_quality_score'], (int, float))


class TestPipelineErrorHandling:
    """Test error handling in pipeline components."""
    
    def test_invalid_data_handling(self):
        """Test pipeline behavior with invalid data."""
        # Create problematic dataset
        problematic_data = pd.DataFrame({
            'feature1': [np.inf, -np.inf, np.nan, 1, 2],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Test validator handles problematic data
        validator = DataValidator()
        
        try:
            results = validator.validate_dataset(problematic_data, target_column='target')
            # Should complete but report issues
            assert len(results.validation_warnings) > 0 or len(results.validation_errors) > 0
        except Exception as e:
            # Should handle gracefully
            assert "validation" in str(e).lower() or "data" in str(e).lower()
    
    def test_empty_data_handling(self):
        """Test pipeline behavior with empty data."""
        empty_data = pd.DataFrame()
        
        validator = DataValidator()
        
        with pytest.raises(Exception):
            validator.validate_dataset(empty_data)
    
    def test_mismatched_data_handling(self):
        """Test pipeline behavior with mismatched features."""
        # Training data
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Test data with different features
        test_data = pd.DataFrame({
            'feature1': [6, 7, 8],
            'feature3': ['f', 'g', 'h']  # Different feature name
        })
        
        # Preprocessor should handle gracefully
        preprocessor = AdvancedDataPreprocessor()
        
        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        
        # Fit on training data
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        
        # Transform test data (should handle missing features)
        try:
            X_test_processed = preprocessor.transform(test_data)
            # Should complete with appropriate handling of missing features
        except Exception as e:
            # Should provide informative error
            assert "feature" in str(e).lower() or "column" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])