"""Test configuration for pytest."""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from typing import Generator, Dict, Any

import pytest
import pandas as pd
import numpy as np

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session") 
def sample_dataset() -> pd.DataFrame:
    """Create sample binary classification dataset for testing."""
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Add correlation structure
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
    X[:, 2] = -X[:, 0] + 0.3 * np.random.randn(n_samples)
    
    # Generate target with signal
    linear_combination = (
        0.5 * X[:, 0] + 
        0.3 * X[:, 1] - 
        0.2 * X[:, 2] + 
        0.1 * X[:, 3] * X[:, 4]
    )
    
    probabilities = 1 / (1 + np.exp(-(linear_combination + 0.2 * np.random.randn(n_samples))))
    y = np.random.binomial(1, probabilities)
    
    # Create DataFrame
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add categorical features
    df['category_A'] = np.random.choice(['cat_1', 'cat_2', 'cat_3'], n_samples)
    df['category_B'] = np.random.choice(['type_X', 'type_Y'], n_samples, p=[0.3, 0.7])
    
    # Add some missing values
    missing_mask = np.random.random((n_samples, n_features)) < 0.05
    df.iloc[:, :n_features] = df.iloc[:, :n_features].mask(missing_mask)
    
    return df


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        'model': {
            'model_type': 'xgboost',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'data': {
            'target_col': 'target',
            'test_size': 0.2,
            'valid_size': 0.2,
            'stratify': True
        },
        'tuning': {
            'enable_tuning': False,
            'n_trials': 10,
            'cv_folds': 3
        },
        'evaluation': {
            'generate_plots': False,
            'save_results': False
        },
        'mlflow': {
            'experiment_name': 'test_experiment',
            'log_params': True
        }
    }


@pytest.fixture
def temp_config_file(test_data_dir: Path, sample_config_dict: Dict[str, Any]) -> Path:
    """Create temporary YAML config file for testing."""
    import yaml
    
    config_file = test_data_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config_dict, f)
    
    return config_file


@pytest.fixture
def temp_csv_file(test_data_dir: Path, sample_dataset: pd.DataFrame) -> Path:
    """Create temporary CSV file for testing."""
    csv_file = test_data_dir / "test_data.csv"
    sample_dataset.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing without actual MLflow server."""
    with pytest.MonkeyPatch().context() as m:
        # Mock MLflow modules
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()
        mock_mlflow.log_param = MagicMock()
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.log_artifact = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        
        m.setattr("mlflow.start_run", mock_mlflow.start_run)
        m.setattr("mlflow.log_param", mock_mlflow.log_param)
        m.setattr("mlflow.log_metric", mock_mlflow.log_metric)
        m.setattr("mlflow.log_artifact", mock_mlflow.log_artifact)
        m.setattr("mlflow.set_experiment", mock_mlflow.set_experiment)
        
        yield mock_mlflow


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment variables after each test."""
    # Store original environment
    original_env = dict(os.environ)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_X_y(sample_dataset: pd.DataFrame):
    """Split sample dataset into features and target."""
    X = sample_dataset.drop('target', axis=1)
    y = sample_dataset['target']
    return X, y


@pytest.fixture
def trained_xgb_model(sample_X_y):
    """Train a simple XGBoost model for testing."""
    try:
        import xgboost as xgb
    except ImportError:
        pytest.skip("XGBoost not available")
    
    X, y = sample_X_y
    
    # Select only numeric features for simplicity
    X_numeric = X.select_dtypes(include=[np.number])
    
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        verbosity=0
    )
    model.fit(X_numeric, y)
    
    return model, X_numeric, y


# Marks for test categorization
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "config: mark test as configuration-related"
    )
    config.addinivalue_line(
        "markers", "models: mark test as model-related"
    )
    config.addinivalue_line(
        "markers", "data: mark test as data-related"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their paths."""
    for item in items:
        # Auto-mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark based on test file names
        if "config" in str(item.fspath):
            item.add_marker(pytest.mark.config)
        elif "model" in str(item.fspath):
            item.add_marker(pytest.mark.models)
        elif "data" in str(item.fspath):
            item.add_marker(pytest.mark.data)