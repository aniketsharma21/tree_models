# Tree Models 🌳

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, production-ready Python package for tree-based machine learning with advanced features for data validation, preprocessing, feature engineering, model training, evaluation, and explainability.

## 🚀 **Key Features**

### **🔧 Core Capabilities**
- **Unified Training Interface**: Support for XGBoost, LightGBM, and CatBoost
- **Comprehensive Evaluation**: Advanced metrics, ROC analysis, calibration, threshold optimization
- **Data Validation**: Comprehensive quality assessment and anomaly detection
- **Feature Engineering**: 15+ transformation types with domain-specific helpers
- **Model Explainability**: SHAP integration, scorecards, reason codes, partial dependence

### **🏭 Production-Ready Features**
- **Type Safety**: Full type hints and validation throughout
- **Error Handling**: Robust exception handling with detailed contexts
- **Performance Optimization**: Memory-efficient processing for large datasets
- **Configuration Management**: Type-safe YAML/JSON configuration system
- **MLOps Integration**: MLflow tracking with automatic logging
- **Comprehensive Testing**: Unit, integration, and performance tests

### **📊 Advanced Analytics**
- **Sample Weights**: Full integration throughout all workflows
- **Statistical Validation**: Confidence intervals, significance testing
- **Business Metrics**: Cost-based evaluation and threshold optimization
- **Regulatory Compliance**: FCRA/GDPR compliant explanations
- **Data Drift Detection**: Statistical tests for distribution changes

## 📦 **Installation**

### **Quick Install**
```bash
pip install tree-models
```

### **Development Installation**
```bash
git clone https://github.com/your-username/tree-models.git
cd tree-models
make install
```

### **Optional Dependencies**
```bash
# For XGBoost support
pip install xgboost

# For LightGBM support  
pip install lightgbm

# For CatBoost support
pip install catboost

# For advanced text processing
pip install nltk

# For interactive plotting
pip install plotly
```

## 🏃‍♂️ **Quick Start**

### **Simple Model Training**
```python
import pandas as pd
from tree_models import train_model, evaluate_model
from tree_models.config import XGBoostConfig

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Configure and train model
config = XGBoostConfig(n_estimators=100, max_depth=6)
results = train_model(config, X_train, y_train, X_valid, y_valid)

# Evaluate model
evaluation = evaluate_model(results.model, X_test, y_test)
print(f"AUC: {evaluation.metrics['auc']:.4f}")
```

### **Complete ML Pipeline**
```python
from tree_models.data import validate_dataset, FeatureEngineer
from tree_models.config import create_advanced_data_config

# 1. Validate data quality
validation_results = validate_dataset(df, target_column='target')
print(f"Data quality score: {validation_results.data_quality_score:.2f}/100")

# 2. Engineer features
engineer = FeatureEngineer()
fe_results = engineer.engineer_features(
    df, 
    config=FeatureEngineeringConfig(
        log_transform_cols=['income', 'amount'],
        extract_date_features=['transaction_date'],
        create_ratios=[('income', 'expenses')]
    )
)

# 3. Train with MLflow tracking
from tree_models.tracking import start_experiment

with start_experiment("fraud_detection", "baseline_v1") as tracker:
    results = train_model(config, X_train, y_train)
    tracker.log_model(results.model, "model")
    tracker.log_metrics(results.validation_metrics)
```

## 📚 **Documentation**

### **Core Modules**

#### **🤖 Models**
- `ModelTrainer`: Unified training interface for tree-based models
- `ModelEvaluator`: Comprehensive evaluation with advanced metrics
- `FeatureSelector`: Multiple feature selection algorithms
- `RobustnessAnalyzer`: Model stability and robustness testing

#### **🔍 Explainability**
- `SHAPExplainer`: Enhanced SHAP integration with performance optimization
- `ScorecardConverter`: Business scorecard generation with calibration
- `ReasonCodeGenerator`: Regulatory-compliant explanations
- `PartialDependencePlotter`: PD and ICE plots with interactions

#### **📊 Data Processing**
- `DataValidator`: Comprehensive data quality assessment
- `AdvancedDataPreprocessor`: Production-ready preprocessing pipeline
- `FeatureEngineer`: 15+ transformation types with domain helpers

#### **⚙️ Configuration**
- `DataConfig`: Type-safe data processing configuration
- `ModelConfig`: Model-specific configurations (XGBoost, LightGBM, CatBoost)
- `ConfigLoader`: YAML/JSON configuration loading with validation

#### **📈 Tracking**
- `MLflowTracker`: Enhanced MLflow integration with auto-logging
- `ExperimentTracker`: High-level experiment management

### **Usage Examples**

#### **Advanced Feature Engineering**
```python
from tree_models.data.feature_engineer import create_financial_features

# Domain-specific feature engineering
results = create_financial_features(
    df,
    amount_columns=['income', 'expenses', 'assets'],
    target_column='default_risk'
)
```

#### **Model Comparison**
```python
from tree_models.models.evaluator import compare_models

models = {
    'xgb': xgb_model,
    'lgb': lgb_model,
    'cat': cat_model
}

comparison = compare_models(models, X_test, y_test, comparison_metric='auc')
print(f"Best model: {comparison['best_model']} (AUC: {comparison['best_score']:.4f})")
```

#### **SHAP Explanations**
```python
from tree_models.explainability import SHAPExplainer

explainer = SHAPExplainer(model, explainer_type='auto')
results = explainer.explain(X_test, max_samples=1000)

# Generate explanations
explainer.plot_summary(results.shap_values)
explainer.plot_waterfall(results.shap_values, instance_idx=0)
```

## 🧪 **Testing**

### **Run Tests**
```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test types
make test-unit
make test-integration
make test-performance

# Run linting and formatting
make lint
make format
```

### **Test Categories**
- **Unit Tests**: Component-level testing with mocking
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Benchmarks and scalability validation

## 🚀 **Development**

### **Setup Development Environment**
```bash
# Clone repository
git clone https://github.com/your-username/tree-models.git
cd tree-models

# Setup environment
make setup

# Install in development mode
make install-dev

# Run pre-commit hooks
make setup-hooks
```

### **Project Structure**
```
tree_models/
├── models/                 # Core ML components
├── explainability/        # Model explanation tools
├── data/                  # Data processing utilities
├── config/                # Configuration management
├── tracking/              # Experiment tracking
└── utils/                 # Shared utilities

tests/
├── unit/                  # Unit tests
├── integration/           # Integration tests
└── performance/           # Performance benchmarks

examples/
├── quickstart/            # Beginner examples
└── advanced/              # Advanced use cases
```

### **Contributing Guidelines**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`make test`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### **Code Standards**
- **Type Hints**: All functions must have type hints
- **Docstrings**: Google-style docstrings required
- **Testing**: 90%+ code coverage required
- **Formatting**: Black code formatting enforced
- **Linting**: Flake8 and mypy validation

## 📋 **Requirements**

### **Core Dependencies**
- **Python**: 3.8+
- **pandas**: >=1.3.0
- **numpy**: >=1.21.0
- **scikit-learn**: >=1.0.0
- **pydantic**: >=1.8.0

### **Optional Dependencies**
- **xgboost**: >=1.5.0 (for XGBoost models)
- **lightgbm**: >=3.2.0 (for LightGBM models)
- **catboost**: >=1.0.0 (for CatBoost models)
- **mlflow**: >=1.20.0 (for experiment tracking)
- **shap**: >=0.40.0 (for model explanations)
- **plotly**: >=5.0.0 (for interactive plots)

## 🔧 **Configuration**

### **YAML Configuration Example**
```yaml
# config/xgboost_config.yaml
experiment:
  name: fraud_detection_v1
  description: Fraud detection with XGBoost

model:
  model_type: xgboost
  n_estimators: 200
  max_depth: 8
  learning_rate: 0.1
  subsample: 0.8

data:
  train_path: data/train.csv
  test_path: data/test.csv
  target_column: is_fraud
  test_size: 0.2
  validation_size: 0.2

preprocessing:
  scaling_strategy: standard
  outlier_detection: true
  missing_value_strategy:
    numeric: median
    categorical: most_frequent
```

### **Loading Configuration**
```python
from tree_models.config import load_config

config = load_config('config/xgboost_config.yaml')
results = train_model(config.model, X_train, y_train)
```

## 📊 **Performance Benchmarks**

### **Scalability** (tested on standard hardware)
- **Small datasets** (1K samples, 10 features): <1s processing
- **Medium datasets** (100K samples, 50 features): <30s processing  
- **Large datasets** (1M+ samples): Memory-efficient chunked processing

### **Memory Efficiency**
- **Validation**: <2x data size memory overhead
- **Preprocessing**: Streaming processing for large datasets
- **Feature Engineering**: Configurable memory limits

## 🤝 **Community & Support**

### **Getting Help**
- 📖 **Documentation**: [https://tree-models.readthedocs.io](https://tree-models.readthedocs.io)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/tree-models/discussions)  
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-username/tree-models/issues)
- 📧 **Email**: support@tree-models.dev

### **Contributing**
We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### **Citation**
If you use tree-models in your research, please cite:
```bibtex
@software{tree_models,
  title = {Tree Models: Production-Ready Tree-Based Machine Learning},
  author = {Your Name},
  url = {https://github.com/your-username/tree-models},
  year = {2025}
}
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 **Roadmap**

### **Current Version (1.0.0)**
- ✅ Core training and evaluation
- ✅ Comprehensive data processing
- ✅ Basic explainability features
- ✅ Configuration management

### **Planned Features (1.1.0)**
- 🔄 AutoML capabilities
- 🔄 Advanced ensemble methods
- 🔄 Real-time inference optimization
- 🔄 Cloud deployment utilities

### **Future Enhancements**
- 🔮 Deep learning integration
- 🔮 Distributed training support
- 🔮 Advanced fairness metrics
- 🔮 Edge deployment capabilities

---

## 🌟 **Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/tree-models&type=Date)](https://star-history.com/#your-username/tree-models&Date)

---

**Made with ❤️ for the ML community**