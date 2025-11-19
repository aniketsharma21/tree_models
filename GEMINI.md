# Project Overview

This project, `tree-models`, is a comprehensive Python package for tree-based machine learning. It provides a production-ready framework with advanced features for data validation, preprocessing, feature engineering, model training, evaluation, and explainability. The library supports popular tree-based models like XGBoost, LightGBM, and CatBoost.

The project is well-structured with a clear separation of concerns. It includes a `models` module for core machine learning components, an `explainability` module for model interpretation, a `data` module for data processing, a `config` module for configuration management, and a `tracking` module for experiment tracking. The project also has a comprehensive test suite, including unit, integration, and performance tests.

# Building and Running

The project uses `make` for task automation. Here are some key commands:

*   **Installation:**
    *   `make install`: Install the package in production mode.
    *   `make install-dev`: Install the package in development mode with all dependencies.

*   **Testing:**
    *   `make test`: Run all tests.
    *   `make test-unit`: Run unit tests.
    *   `make test-integration`: Run integration tests.
    *   `make test-performance`: Run performance benchmarks.
    *   `make test-coverage`: Run tests with a coverage report.

*   **Linting and Formatting:**
    *   `make lint`: Run linting checks.
    *   `make format`: Format the code.

*   **Building:**
    *   `make build`: Build the source and wheel distributions.

# Development Conventions

The project follows a set of development conventions to ensure code quality and consistency:

*   **Type Hinting:** All functions must have type hints.
*   **Docstrings:** Google-style docstrings are required.
*   **Testing:** A high code coverage of 90%+ is required.
*   **Formatting:** The code is formatted using `black`.
*   **Linting:** `flake8` and `mypy` are used for linting.
*   **Pre-commit Hooks:** Pre-commit hooks are used to enforce code standards.
