# Tree Models - ML Framework

A comprehensive machine learning framework for tree-based models (XGBoost, LightGBM, CatBoost) with advanced configuration management, MLOps integration, and production-ready features.

## Features

- **Hybrid Configuration System**: Combines Python type safety with YAML flexibility
- **Multiple Model Support**: XGBoost, LightGBM, CatBoost with unified interface
- **MLOps Integration**: Comprehensive MLflow tracking and experiment management
- **Feature Selection**: RFECV, Boruta, and consensus-based selection
- **Model Explainability**: SHAP integration and business scorecards
- **Production Ready**: Sample weights, robustness testing, drift detection

## Quick Start

### Installation

```bash
pip install tree-models
```

### Basic Usage

```python
from tree_models.config import load_config
from tree_models.models import train_model

# Load configuration
config = load_config('config/xgboost_default.yaml')

# Train model
model = train_model(
    model_type="xgboost",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    params=config.model.to_dict()
)
```

### Advanced Usage

```python
from tree_models import fraud_detection_pipeline

# Complete fraud detection pipeline
results = fraud_detection_pipeline(
    model_class=xgb.XGBClassifier,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    sample_weight_train=weights_train,
    focus='recall',  # Maximize fraud detection
    output_dir='fraud_model_analysis'
)
```

## Configuration

The framework uses a hybrid configuration system that combines Python type safety with YAML flexibility:

### Python Configuration

```python
from tree_models.config import XGBoostConfig

config = XGBoostConfig(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    reg_alpha=0.1
)
```

### YAML Configuration

```yaml
model:
  model_type: xgboost
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.1
  
data:
  train_path: "data/train.csv"
  target_col: "target"
  test_size: 0.2

tuning:
  enable_tuning: true
  n_trials: 100
```

### Environment Overrides

```bash
export MODEL_TYPE=lightgbm
export N_ESTIMATORS=300
export TRAIN_PATH=/data/fraud_data.csv
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/aniketsharma21/tree_models.git
cd tree_models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tree_models --cov-report=html

# Run specific test file
pytest tests/test_config.py -v
```

### Code Quality

```bash
# Format code
black tree_models tests
isort tree_models tests

# Lint code
flake8 tree_models tests
mypy tree_models

# Run all quality checks
make lint
```

## Project Structure

```
tree_models/
├── config/                 # Configuration system
│   ├── base_config.py     # Python configuration classes
│   ├── config_loader.py   # YAML loading utilities
│   ├── config_schema.py   # Validation schemas
│   └── defaults/          # Default configuration files
├── src/                   # Source code
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model training, tuning, evaluation
│   ├── tracking/         # MLflow integration
│   └── utils/            # Utility functions
├── tests/                # Unit tests
├── notebooks/            # Example notebooks
└── docs/                # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: [docs.tree-models.com](https://docs.tree-models.com)
- Issues: [GitHub Issues](https://github.com/aniketsharma21/tree_models/issues)
- Discussions: [GitHub Discussions](https://github.com/aniketsharma21/tree_models/discussions)