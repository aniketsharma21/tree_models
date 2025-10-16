# Tree Model Helper - Hybrid Configuration System

## 🎯 Answer: Use BOTH Config.py AND YAML!

This implementation provides a **hybrid configuration system** that combines the benefits of both approaches:

- **🔒 Type Safety**: Python dataclasses with type hints for IDE support
- **📝 Flexibility**: YAML files for easy configuration management 
- **🌍 Environment**: Support for environment variable overrides
- **✅ Validation**: Built-in schema validation with Pydantic

## 📁 File Structure

```
tree_model_helper/config/
├── __init__.py                    # Main configuration exports
├── base_config.py                 # Python type definitions & defaults  
├── config_schema.py               # Pydantic validation schemas
├── config_loader.py               # YAML loading & merging utilities
└── defaults/
    ├── xgboost_default.yaml       # XGBoost configuration template
    ├── lightgbm_default.yaml      # LightGBM configuration template
    └── production.yaml             # Production-ready configuration
```

## 🚀 Quick Start

### 1. Load from YAML (Best for Experiments)
```python
from tree_model_helper.config import load_config

config = load_config('config/xgboost_default.yaml')
print(f"Model: {config.model.model_type}")
print(f"N_estimators: {config.model.n_estimators}")
```

### 2. Python Objects (Best for Programming)
```python
from tree_model_helper.config import XGBoostConfig

model_config = XGBoostConfig(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05
)
params = model_config.to_dict()
```

### 3. Production Config (Best for Deployment)
```python
from tree_model_helper.config import get_production_config

# Override with environment variables
import os
os.environ['MODEL_TYPE'] = 'xgboost'
os.environ['N_ESTIMATORS'] = '300'

config = get_production_config('xgboost')
```

### 4. Environment Overrides
```bash
export MODEL_TYPE=lightgbm
export N_ESTIMATORS=300
export TRAIN_PATH=/data/fraud_data.csv
export MLFLOW_EXPERIMENT=fraud_detection_v2
```

## 🎯 Use Cases

| Scenario | Best Approach | Example |
|----------|---------------|---------|
| **Development** | YAML files | `config = load_config('dev.yaml')` |
| **Production** | Environment vars | `config = get_production_config()` |
| **Experimentation** | YAML + overrides | Load base YAML, modify parameters |
| **Programmatic** | Python objects | `XGBoostConfig(n_estimators=500)` |
| **Team collaboration** | YAML templates | Share standardized configurations |

## 🏆 Why Hybrid is Better

### ❌ Problems with Config.py Only:
- Need code changes for configuration updates
- Harder for non-programmers to modify
- Security risk with code execution
- Complex deployment process

### ❌ Problems with YAML Only:
- No type safety or validation
- No IDE autocomplete support
- Limited logic capabilities
- Runtime parsing overhead

### ✅ Hybrid Solution Benefits:
- **Type safety** from Python + **flexibility** from YAML
- **Environment overrides** for production
- **Schema validation** for robustness
- **Backwards compatibility** with existing code
- **Model-specific** optimizations
- **Production-ready** security settings

## 📊 Configuration Sections

The system includes comprehensive configuration for:

- **Model**: XGBoost/LightGBM/CatBoost parameters
- **Data**: Paths, preprocessing, splitting
- **Feature Selection**: Variance, RFECV, Boruta
- **Tuning**: Optuna hyperparameter optimization
- **Evaluation**: Metrics, plots, thresholds
- **MLflow**: Experiment tracking
- **Production**: Security, monitoring, alerts

## 🧪 Run the Demo

```python
python tree_model_helper/config_demo.py
```

This will demonstrate all configuration features with examples.

## 🎯 Recommendation

**Use the hybrid approach** for production ML systems:

1. **Ship with Python defaults** for type safety
2. **Use YAML files** for experiments and customization
3. **Environment variables** for production secrets
4. **Validation schemas** for robustness
5. **Model-specific configs** for optimization

This gives you the best of both worlds without the downsides of either approach alone.
