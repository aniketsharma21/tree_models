import random
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tree_models.models.trainer import ModelTrainer


class TestAdapter:
    """A lightweight adapter used only for testing reproducibility.

    It reads the current numpy and python RNG state and returns those
    values in a TrainingResults-like SimpleNamespace so the test can
    verify deterministic behavior.
    """

    def train(
        self,
        trainer,
        model_config,
        X_train,
        y_train,
        X_valid,
        y_valid,
        sample_weight,
        validation_weight,
        train_params,
        training_config,
    ):
        # Produce deterministic outputs derived from RNG
        a = float(np.random.rand())
        b = float(random.random())

        return SimpleNamespace(
            model=None,
            model_type=getattr(model_config, "model_type", "test"),
            training_time=0.0,
            train_metrics={"rand_np": a},
            validation_metrics={"rand_py": b},
            cv_metrics=None,
            learning_curves=None,
            feature_importance=None,
            best_iteration=None,
            early_stopped=False,
            convergence_info={},
        )


def test_trainer_respects_random_state(monkeypatch):
    # Create tiny dataset
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])

    # Patch adapter factory to return our TestAdapter
    import tree_models.models.adapters.factory as factory_module

    monkeypatch.setattr(factory_module, "get_adapter", lambda model_type: TestAdapter())

    # Minimal model_config substitute
    model_config = SimpleNamespace(model_type="xgboost", get_params=lambda: {})

    # Train twice with the same seed and assert deterministic outputs
    trainer1 = ModelTrainer(random_state=123)
    res1 = trainer1.train_model(model_config, X, y, X_valid=X, y_valid=y)

    trainer2 = ModelTrainer(random_state=123)
    res2 = trainer2.train_model(model_config, X, y, X_valid=X, y_valid=y)

    assert res1.train_metrics["rand_np"] == res2.train_metrics["rand_np"]
    assert res1.validation_metrics["rand_py"] == res2.validation_metrics["rand_py"]

    # Different seed should (very likely) produce different values
    trainer3 = ModelTrainer(random_state=456)
    res3 = trainer3.train_model(model_config, X, y, X_valid=X, y_valid=y)

    assert res3.train_metrics["rand_np"] != res1.train_metrics["rand_np"]
