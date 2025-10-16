"""Tree Models - Experiment Tracking and MLOps Components.

This module provides MLOps integration with experiment tracking,
model versioning, and deployment utilities.

Key Components:
- MLflowTracker: Enhanced MLflow integration with auto-logging
- ExperimentTracker: High-level experiment management

Example:
    >>> from tree_models.tracking import MLflowTracker, start_experiment
    >>> tracker = MLflowTracker()
    >>> with start_experiment("fraud_detection", "baseline_v1") as exp:
    ...     exp.log_model(model, "model")
"""

# MLOps and tracking components
from .mlflow_tracker import (
    MLflowTracker,
    ExperimentTracker,
    start_experiment
)

__all__ = [
    'MLflowTracker',
    'ExperimentTracker', 
    'start_experiment'
]