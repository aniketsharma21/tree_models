# Tree Models - High-Level API

This directory contains the high-level API for the `tree_models` package, designed to provide a simple and intuitive interface for common machine learning tasks. It abstracts away much of the underlying complexity, allowing users to perform complex analyses with just a few lines of code.

The API is organized into three main modules:

1.  [`workflows.py`](./workflows.py): Complete end-to-end analysis pipelines.
2.  [`quick_functions.py`](./quick_functions.py): Fast utility functions for common, standalone tasks.
3.  [`info.py`](./info.py): Package information and help utilities.

---

### 🚀 `workflows.py`

This module provides comprehensive, multi-step workflows that orchestrate various components of the `tree_models` library to deliver a complete analysis.

**Key Functions:**

-   `complete_model_analysis()`: An all-in-one pipeline that performs hyperparameter tuning, model training, evaluation, explainability analysis, and robustness testing. It's the easiest way to get a thorough understanding of your model's performance.
-   `fraud_detection_pipeline()`: A specialized workflow tailored for fraud detection use cases. It uses fraud-specific configurations and evaluation metrics to maximize the detection of fraudulent activities.

**Example:**

```python
from tree_models.api import complete_model_analysis

# Run a full analysis with one function call
results = complete_model_analysis(
    model_type='xgboost',
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    scoring_function='recall',
    n_trials=50
)
```

---

### ⚡ `quick_functions.py`

This module contains a collection of standalone functions that simplify common ML tasks. These functions are designed for quick, focused analyses with sensible defaults.

**Key Functions:**

-   `tune_hyperparameters()`: Quickly find the best hyperparameters for your model using Optuna.
-   `quick_shap_analysis()`: Generate SHAP-based explainability plots (summary, waterfall) with a single command.
-   `quick_robustness_test()`: Assess your model's stability by training it with multiple random seeds.
-   `convert_to_scorecard()`: Convert model probabilities into a business-friendly scorecard.
-   `quick_feature_importance()`: Get the most important features from your trained model.
-   `quick_model_comparison()`: Compare the performance of multiple models on a test set.

**Example:**

```python
from tree_models.api import tune_hyperparameters, quick_shap_analysis

# Tune hyperparameters
best_params, score = tune_hyperparameters(
    'xgboost', X_train, y_train, n_trials=100
)

# Explain a trained model
shap_results = quick_shap_analysis(model, X_test)
```

---

### ℹ️ `info.py`

This module provides utility functions to get information about the `tree_models` package, its environment, and supported features.

**Key Functions:**

-   `show_package_info()`: Displays a comprehensive overview of the package, including a quick start guide and key features.
-   `get_version()`: Returns the current version of the package.
-   `get_supported_models()`: Lists the model types supported by the framework (e.g., 'xgboost', 'lightgbm').
-   `get_available_scorers()`: Lists the available scoring functions for evaluation and tuning.
-   `check_installation()`: Verifies that the package and its optional dependencies are installed correctly.

**Example:**

```python
import tree_models as tm

# Display package information
tm.show_package_info()

# Check which models are available
supported_models = tm.get_supported_models()
print(f"Supported models: {supported_models}")
```
