# tests/conftest.py
"""Pytest configuration and shared fixtures for tree_models tests."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import warnings

# Suppress warnings during testing
warnings.filterwarnings('ignore')


@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture(autouse=True)
def setup_test_environment(random_seed):
    """Setup test environment with consistent random state."""
    np.random.seed(random_seed)


@pytest.fixture
def sample_binary_classification_data():
    """Create sample binary classification dataset."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create somewhat realistic binary target
    weights = np.random.randn(n_features)
    linear_combo = X.values @ weights
    probabilities = 1 / (1 + np.exp(-linear_combo))
    y = pd.Series(np.random.binomial(1, probabilities), name='target')
    
    # Add sample weights
    sample_weight = np.random.uniform(0.5, 2.0, n_samples)
    
    return X, y, sample_weight


@pytest.fixture 
def sample_regression_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some relationship to features
    weights = np.random.randn(n_features)
    y = pd.Series(
        X.values @ weights + np.random.randn(n_samples) * 0.1,
        name='target'
    )
    
    sample_weight = np.random.uniform(0.5, 2.0, n_samples)
    
    return X, y, sample_weight


@pytest.fixture
def sample_mixed_data():
    """Create dataset with mixed feature types."""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        # Numeric features
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'score': np.random.normal(650, 100, n_samples),
        
        # Categorical features
        'category_A': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'category_B': np.random.choice([f'Group_{i}' for i in range(5)], n_samples),
        
        # Date feature
        'date_col': pd.date_range('2020-01-01', periods=n_samples, freq='D')[:n_samples],
        
        # Text feature
        'text_col': [f'sample text {i} with words' for i in range(n_samples)],
        
        # Target
        'target': np.random.binomial(1, 0.3, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    return df


@pytest.fixture
def mock_xgboost_model():
    """Create a mock XGBoost model."""
    model = Mock()
    model.predict.return_value = np.random.uniform(0, 1, 100)
    model.predict_proba.return_value = np.column_stack([
        np.random.uniform(0, 1, 100),
        np.random.uniform(0, 1, 100)
    ])
    model.__class__.__name__ = 'XGBClassifier'
    return model


@pytest.fixture
def mock_lightgbm_model():
    """Create a mock LightGBM model."""
    model = Mock()
    model.predict.return_value = np.random.uniform(0, 1, 100)
    model.predict_proba.return_value = np.column_stack([
        np.random.uniform(0, 1, 100),
        np.random.uniform(0, 1, 100)
    ])
    model.__class__.__name__ = 'LGBMClassifier'
    return model


@pytest.fixture
def temporary_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config_files(temporary_directory):
    """Create sample configuration files for testing."""
    config_dir = temporary_directory / 'configs'
    config_dir.mkdir()
    
    # Basic model config
    model_config = {
        'model_type': 'xgboost',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
    
    # Data config
    data_config = {
        'source': {
            'train_path': 'data/train.csv',
            'test_path': 'data/test.csv',
            'file_format': 'csv'
        },
        'features': {
            'target_column': 'target',
            'numeric_features': ['age', 'income', 'score'],
            'categorical_features': ['category_A', 'category_B']
        }
    }
    
    # Save configs as JSON
    import json
    
    model_config_path = config_dir / 'model_config.json'
    with open(model_config_path, 'w') as f:
        json.dump(model_config, f)
    
    data_config_path = config_dir / 'data_config.json'
    with open(data_config_path, 'w') as f:
        json.dump(data_config, f)
    
    return {
        'config_dir': config_dir,
        'model_config_path': model_config_path,
        'data_config_path': data_config_path,
        'model_config': model_config,
        'data_config': data_config
    }


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "requires_ml_libs: marks tests that require ML libraries (xgboost, lightgbm, etc.)"
    )


# Skip conditions for optional dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle optional dependencies."""
    
    # Check for available libraries
    try:
        import xgboost
        has_xgboost = True
    except ImportError:
        has_xgboost = False
    
    try:
        import lightgbm
        has_lightgbm = True
    except ImportError:
        has_lightgbm = False
    
    try:
        import catboost
        has_catboost = True
    except ImportError:
        has_catboost = False
    
    # Add skip markers based on available libraries
    for item in items:
        if "test_xgboost" in item.nodeid and not has_xgboost:
            item.add_marker(pytest.mark.skip(reason="XGBoost not available"))
        
        if "test_lightgbm" in item.nodeid and not has_lightgbm:
            item.add_marker(pytest.mark.skip(reason="LightGBM not available"))
        
        if "test_catboost" in item.nodeid and not has_catboost:
            item.add_marker(pytest.mark.skip(reason="CatBoost not available"))
        
        if item.get_closest_marker("requires_ml_libs"):
            if not (has_xgboost or has_lightgbm or has_catboost):
                item.add_marker(pytest.mark.skip(reason="No ML libraries available"))


@pytest.fixture
def mock_ml_libraries():
    """Mock ML libraries for testing without requiring actual installations."""
    
    class MockXGBoost:
        class XGBClassifier:
            def __init__(self, **kwargs):
                self.params = kwargs
            
            def fit(self, X, y, **kwargs):
                return self
            
            def predict(self, X):
                return np.random.randint(0, 2, len(X))
            
            def predict_proba(self, X):
                proba = np.random.uniform(0, 1, (len(X), 2))
                proba = proba / proba.sum(axis=1, keepdims=True)
                return proba
        
        class DMatrix:
            def __init__(self, data, label=None, weight=None):
                self.data = data
                self.label = label
                self.weight = weight
        
        @staticmethod
        def train(params, dtrain, **kwargs):
            model = Mock()
            model.predict.return_value = np.random.uniform(0, 1, 100)
            return model
    
    class MockLightGBM:
        class LGBMClassifier:
            def __init__(self, **kwargs):
                self.params = kwargs
            
            def fit(self, X, y, **kwargs):
                return self
            
            def predict(self, X):
                return np.random.randint(0, 2, len(X))
            
            def predict_proba(self, X):
                proba = np.random.uniform(0, 1, (len(X), 2))
                return proba / proba.sum(axis=1, keepdims=True)
        
        class Dataset:
            def __init__(self, data, label=None, weight=None, **kwargs):
                self.data = data
                self.label = label
                self.weight = weight
    
    return {
        'xgboost': MockXGBoost(),
        'lightgbm': MockLightGBM()
    }


# Fixtures for performance testing
@pytest.fixture(scope="session")
def performance_datasets():
    """Create datasets for performance testing."""
    datasets = {}
    
    # Small dataset
    np.random.seed(42)
    datasets['small'] = pd.DataFrame({
        f'feature_{i}': np.random.randn(1000) 
        for i in range(10)
    })
    datasets['small']['target'] = np.random.randint(0, 2, 1000)
    
    # Medium dataset  
    datasets['medium'] = pd.DataFrame({
        f'feature_{i}': np.random.randn(10000)
        for i in range(50)
    })
    datasets['medium']['target'] = np.random.randint(0, 2, 10000)
    
    # Large dataset (only create if needed)
    def create_large():
        return pd.DataFrame({
            f'feature_{i}': np.random.randn(50000)
            for i in range(100)
        })
    
    datasets['large_factory'] = create_large
    
    return datasets


# Utility functions for testing
def assert_dataframe_equal(df1, df2, check_dtype=True, check_names=True):
    """Enhanced DataFrame equality assertion."""
    pd.testing.assert_frame_equal(
        df1, df2,
        check_dtype=check_dtype,
        check_names=check_names,
        check_categorical=False,
        check_index_type=False
    )


def assert_series_equal(s1, s2, check_dtype=True, check_names=True):
    """Enhanced Series equality assertion."""
    pd.testing.assert_series_equal(
        s1, s2,
        check_dtype=check_dtype,
        check_names=check_names,
        check_categorical=False
    )


# Performance measurement utilities
@pytest.fixture
def measure_time():
    """Context manager for measuring execution time."""
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def timer():
        start_time = time.time()
        yield
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
    
    return timer