"""Unit tests for data processing functionality."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Tuple

from src.data.data_loader import load_csv, DataLoader
from src.data.data_preprocessor import DataPreprocessor, split_data


class TestDataLoader:
    """Test data loading functionality."""

    @pytest.mark.unit
    @pytest.mark.data
    def test_load_csv_basic(self, temp_csv_file: Path):
        """Test basic CSV loading functionality."""
        try:
            df = load_csv(temp_csv_file)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'target' in df.columns
            
        except ImportError:
            pytest.skip("load_csv function not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_load_csv_with_sample_size(self, temp_csv_file: Path):
        """Test CSV loading with sample size limit."""
        try:
            sample_size = 100
            df = load_csv(temp_csv_file, sample_size=sample_size)
            
            assert len(df) <= sample_size
            
        except ImportError:
            pytest.skip("load_csv function not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_load_csv_nonexistent_file(self):
        """Test error handling for non-existent files."""
        try:
            with pytest.raises(FileNotFoundError):
                load_csv("nonexistent_file.csv")
                
        except ImportError:
            pytest.skip("load_csv function not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_data_loader_initialization(self):
        """Test DataLoader class initialization."""
        try:
            loader = DataLoader()
            assert loader is not None
            
            loader_with_params = DataLoader(
                sample_size=1000,
                random_state=42
            )
            assert loader_with_params.sample_size == 1000
            assert loader_with_params.random_state == 42
            
        except ImportError:
            pytest.skip("DataLoader class not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_load_multiple_formats(self, test_data_dir: Path, sample_dataset: pd.DataFrame):
        """Test loading different file formats."""
        # Create test files in different formats
        csv_file = test_data_dir / "test.csv"
        parquet_file = test_data_dir / "test.parquet"
        
        sample_dataset.to_csv(csv_file, index=False)
        
        try:
            sample_dataset.to_parquet(parquet_file, index=False)
        except ImportError:
            pytest.skip("Parquet not available")
        
        try:
            # Test CSV loading
            df_csv = load_csv(csv_file)
            assert isinstance(df_csv, pd.DataFrame)
            
            # Test parquet loading (if supported)
            if hasattr(pd, 'read_parquet'):
                df_parquet = pd.read_parquet(parquet_file)
                assert isinstance(df_parquet, pd.DataFrame)
                
        except ImportError:
            pytest.skip("File loading functions not available")


class TestDataPreprocessor:
    """Test data preprocessing functionality."""

    @pytest.mark.unit
    @pytest.mark.data
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        try:
            preprocessor = DataPreprocessor()
            assert preprocessor.missing_strategy == "median"
            assert preprocessor.encoding_strategy == "label"
            
            preprocessor_custom = DataPreprocessor(
                missing_strategy="mean",
                encoding_strategy="onehot",
                scaling_strategy="standard"
            )
            assert preprocessor_custom.missing_strategy == "mean"
            assert preprocessor_custom.encoding_strategy == "onehot"
            assert preprocessor_custom.scaling_strategy == "standard"
            
        except ImportError:
            pytest.skip("DataPreprocessor class not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_missing_value_imputation(self, sample_dataset: pd.DataFrame):
        """Test missing value imputation."""
        try:
            preprocessor = DataPreprocessor(missing_strategy="median")
            
            # Create data with missing values
            X = sample_dataset.drop('target', axis=1)
            y = sample_dataset['target']
            
            # Add some missing values
            X_with_missing = X.copy()
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_with_missing.loc[0, numeric_cols[0]] = np.nan
            
            # Fit and transform
            X_transformed = preprocessor.fit_transform(X_with_missing, y)
            
            # Check no missing values remain
            assert not X_transformed.isnull().any().any()
            assert X_transformed.shape[0] == X_with_missing.shape[0]
            
        except ImportError:
            pytest.skip("DataPreprocessor not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_categorical_encoding(self):
        """Test categorical variable encoding."""
        try:
            # Create test data with categorical variables
            df = pd.DataFrame({
                'cat_feature': ['A', 'B', 'C', 'A', 'B'],
                'num_feature': [1, 2, 3, 4, 5],
                'target': [0, 1, 0, 1, 0]
            })
            
            X = df[['cat_feature', 'num_feature']]
            y = df['target']
            
            # Test label encoding
            preprocessor_label = DataPreprocessor(encoding_strategy="label")
            X_label = preprocessor_label.fit_transform(X, y)
            
            assert 'cat_feature' in X_label.columns
            assert X_label['cat_feature'].dtype in [np.int64, np.int32, np.float64]
            
            # Test one-hot encoding
            preprocessor_onehot = DataPreprocessor(encoding_strategy="onehot")
            X_onehot = preprocessor_onehot.fit_transform(X, y)
            
            # Should have more columns after one-hot encoding
            assert X_onehot.shape[1] > X.shape[1]
            
        except ImportError:
            pytest.skip("Categorical encoding not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_feature_scaling(self):
        """Test feature scaling functionality."""
        try:
            # Create test data
            df = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'feature2': [10, 20, 30, 40, 50],
                'target': [0, 1, 0, 1, 0]
            })
            
            X = df[['feature1', 'feature2']]
            y = df['target']
            
            # Test standard scaling
            preprocessor_std = DataPreprocessor(scaling_strategy="standard")
            X_scaled = preprocessor_std.fit_transform(X, y)
            
            # Check scaled features have mean ~0 and std ~1
            assert abs(X_scaled['feature1'].mean()) < 1e-10
            assert abs(X_scaled['feature1'].std() - 1.0) < 1e-10
            
            # Test min-max scaling
            preprocessor_minmax = DataPreprocessor(scaling_strategy="minmax")
            X_minmax = preprocessor_minmax.fit_transform(X, y)
            
            # Check scaled features are between 0 and 1
            assert X_minmax.min().min() >= 0
            assert X_minmax.max().max() <= 1
            
        except ImportError:
            pytest.skip("Feature scaling not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_fit_transform_consistency(self, sample_dataset: pd.DataFrame):
        """Test fit_transform vs fit + transform consistency."""
        try:
            preprocessor1 = DataPreprocessor()
            preprocessor2 = DataPreprocessor()
            
            X = sample_dataset.drop('target', axis=1)
            y = sample_dataset['target']
            
            # Method 1: fit_transform
            X_transformed1 = preprocessor1.fit_transform(X, y)
            
            # Method 2: fit then transform
            preprocessor2.fit(X, y)
            X_transformed2 = preprocessor2.transform(X)
            
            # Results should be identical
            pd.testing.assert_frame_equal(X_transformed1, X_transformed2)
            
        except (ImportError, AttributeError):
            pytest.skip("Preprocessor fit/transform methods not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_transform_new_data(self, sample_dataset: pd.DataFrame):
        """Test transforming new data with fitted preprocessor."""
        try:
            preprocessor = DataPreprocessor()
            
            # Split data
            split_idx = len(sample_dataset) // 2
            train_data = sample_dataset.iloc[:split_idx]
            test_data = sample_dataset.iloc[split_idx:]
            
            X_train = train_data.drop('target', axis=1)
            y_train = train_data['target']
            X_test = test_data.drop('target', axis=1)
            
            # Fit on training data
            X_train_transformed = preprocessor.fit_transform(X_train, y_train)
            
            # Transform test data
            X_test_transformed = preprocessor.transform(X_test)
            
            # Check same number of features
            assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
            
            # Check column names match
            assert list(X_train_transformed.columns) == list(X_test_transformed.columns)
            
        except ImportError:
            pytest.skip("Preprocessor transform not available")


class TestDataSplitting:
    """Test data splitting functionality."""

    @pytest.mark.unit
    @pytest.mark.data
    def test_basic_data_splitting(self, sample_X_y):
        """Test basic train/validation/test split."""
        try:
            X, y = sample_X_y
            
            X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
                X, y, test_size=0.2, valid_size=0.2, stratify=True
            )
            
            # Check sizes
            total_size = len(X)
            expected_train_size = int(total_size * 0.6)  # Remaining after test and valid
            expected_test_size = int(total_size * 0.2)
            expected_valid_size = int(total_size * 0.2)
            
            assert len(X_train) == len(y_train)
            assert len(X_valid) == len(y_valid)
            assert len(X_test) == len(y_test)
            
            # Check total size is preserved
            assert len(X_train) + len(X_valid) + len(X_test) == total_size
            
            # Check no data leakage (no overlap in indices)
            train_idx = set(X_train.index)
            valid_idx = set(X_valid.index)
            test_idx = set(X_test.index)
            
            assert len(train_idx & valid_idx) == 0
            assert len(train_idx & test_idx) == 0
            assert len(valid_idx & test_idx) == 0
            
        except ImportError:
            pytest.skip("split_data function not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_stratified_splitting(self, sample_X_y):
        """Test stratified splitting preserves class distribution."""
        try:
            X, y = sample_X_y
            
            X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
                X, y, test_size=0.2, valid_size=0.2, stratify=True
            )
            
            # Check class distributions are similar
            original_ratio = y.mean()
            train_ratio = y_train.mean()
            valid_ratio = y_valid.mean()
            test_ratio = y_test.mean()
            
            tolerance = 0.1  # Allow 10% deviation
            assert abs(train_ratio - original_ratio) < tolerance
            assert abs(valid_ratio - original_ratio) < tolerance  
            assert abs(test_ratio - original_ratio) < tolerance
            
        except ImportError:
            pytest.skip("Stratified splitting not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_random_state_reproducibility(self, sample_X_y):
        """Test random state ensures reproducible splits."""
        try:
            X, y = sample_X_y
            
            # First split
            X_train1, X_valid1, X_test1, y_train1, y_valid1, y_test1 = split_data(
                X, y, test_size=0.2, valid_size=0.2, random_state=42
            )
            
            # Second split with same random state
            X_train2, X_valid2, X_test2, y_train2, y_valid2, y_test2 = split_data(
                X, y, test_size=0.2, valid_size=0.2, random_state=42
            )
            
            # Results should be identical
            pd.testing.assert_frame_equal(X_train1, X_train2)
            pd.testing.assert_series_equal(y_train1, y_train2)
            pd.testing.assert_frame_equal(X_valid1, X_valid2)
            pd.testing.assert_series_equal(y_valid1, y_valid2)
            
        except ImportError:
            pytest.skip("Random state splitting not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_different_split_sizes(self, sample_X_y):
        """Test different train/valid/test split proportions."""
        try:
            X, y = sample_X_y
            total_size = len(X)
            
            # Test different configurations
            test_configs = [
                (0.1, 0.1),  # 80/10/10 split
                (0.2, 0.2),  # 60/20/20 split
                (0.3, 0.1),  # 60/30/10 split
            ]
            
            for test_size, valid_size in test_configs:
                X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
                    X, y, test_size=test_size, valid_size=valid_size
                )
                
                # Check approximate sizes
                expected_test = int(total_size * test_size)
                expected_valid = int(total_size * valid_size)
                
                assert abs(len(X_test) - expected_test) <= 1  # Allow off-by-one
                assert abs(len(X_valid) - expected_valid) <= 1
                
                # Check total size preserved
                assert len(X_train) + len(X_valid) + len(X_test) == total_size
                
        except ImportError:
            pytest.skip("Different split sizes not available")


class TestDataValidation:
    """Test data validation and error handling."""

    @pytest.mark.unit
    @pytest.mark.data
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        try:
            preprocessor = DataPreprocessor()
            empty_df = pd.DataFrame()
            empty_series = pd.Series(dtype=int)
            
            with pytest.raises((ValueError, IndexError)):
                preprocessor.fit_transform(empty_df, empty_series)
                
        except ImportError:
            pytest.skip("DataPreprocessor error handling not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_mismatched_data_shapes(self):
        """Test handling of mismatched X and y shapes."""
        try:
            X = pd.DataFrame({'feature': [1, 2, 3]})
            y = pd.Series([0, 1])  # Different length
            
            preprocessor = DataPreprocessor()
            
            with pytest.raises(ValueError):
                preprocessor.fit_transform(X, y)
                
        except ImportError:
            pytest.skip("Shape validation not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_invalid_column_names(self):
        """Test handling of invalid column names."""
        try:
            # Create DataFrame with problematic column names
            df = pd.DataFrame({
                'feature with spaces': [1, 2, 3],
                'feature-with-dashes': [4, 5, 6],
                'target': [0, 1, 0]
            })
            
            X = df.drop('target', axis=1)
            y = df['target']
            
            preprocessor = DataPreprocessor()
            X_transformed = preprocessor.fit_transform(X, y)
            
            # Should handle column names gracefully
            assert isinstance(X_transformed, pd.DataFrame)
            
        except ImportError:
            pytest.skip("Column name handling not available")

    @pytest.mark.unit
    @pytest.mark.data
    def test_all_missing_column(self):
        """Test handling of columns with all missing values."""
        try:
            df = pd.DataFrame({
                'good_feature': [1, 2, 3, 4],
                'all_missing': [np.nan, np.nan, np.nan, np.nan],
                'target': [0, 1, 0, 1]
            })
            
            X = df.drop('target', axis=1)
            y = df['target']
            
            preprocessor = DataPreprocessor()
            X_transformed = preprocessor.fit_transform(X, y)
            
            # Should handle all-missing columns appropriately
            assert isinstance(X_transformed, pd.DataFrame)
            
        except ImportError:
            pytest.skip("Missing value handling not available")


class TestDataIntegration:
    """Integration tests for complete data processing workflows."""

    @pytest.mark.integration
    @pytest.mark.data
    def test_complete_data_pipeline(self, temp_csv_file: Path, test_data_dir: Path):
        """Test complete data processing pipeline."""
        try:
            # Load data
            df = load_csv(temp_csv_file)
            
            # Split features and target
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Split data
            X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
                X, y, test_size=0.2, valid_size=0.2, stratify=True
            )
            
            # Preprocess data
            preprocessor = DataPreprocessor(
                missing_strategy="median",
                encoding_strategy="label",
                scaling_strategy="standard"
            )
            
            X_train_processed = preprocessor.fit_transform(X_train, y_train)
            X_valid_processed = preprocessor.transform(X_valid)
            X_test_processed = preprocessor.transform(X_test)
            
            # Verify processed data
            assert isinstance(X_train_processed, pd.DataFrame)
            assert isinstance(X_valid_processed, pd.DataFrame)
            assert isinstance(X_test_processed, pd.DataFrame)
            
            # Check no missing values
            assert not X_train_processed.isnull().any().any()
            assert not X_valid_processed.isnull().any().any()
            assert not X_test_processed.isnull().any().any()
            
            # Check consistent features
            assert list(X_train_processed.columns) == list(X_valid_processed.columns)
            assert list(X_valid_processed.columns) == list(X_test_processed.columns)
            
        except ImportError:
            pytest.skip("Complete data pipeline not available")

    @pytest.mark.integration
    @pytest.mark.data
    def test_data_persistence(self, temp_csv_file: Path, test_data_dir: Path):
        """Test saving and loading processed data."""
        try:
            # Load and process data
            df = load_csv(temp_csv_file)
            X = df.drop('target', axis=1)
            y = df['target']
            
            preprocessor = DataPreprocessor()
            X_processed = preprocessor.fit_transform(X, y)
            
            # Save processed data
            processed_file = test_data_dir / "processed_data.csv"
            X_processed.to_csv(processed_file, index=False)
            
            # Load saved data
            X_loaded = pd.read_csv(processed_file)
            
            # Should be identical
            pd.testing.assert_frame_equal(X_processed, X_loaded)
            
        except ImportError:
            pytest.skip("Data persistence not available")