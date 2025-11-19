import os
from pathlib import Path
import pandas as pd

import importlib.util
from pathlib import Path as _Path

# Dynamically load the example module by file path to avoid import issues
repo_root = _Path(__file__).resolve().parents[1]
example_path = repo_root / "examples" / "run_quick_example.py"
spec = importlib.util.spec_from_file_location("examples.run_quick_example", str(example_path))
quick_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quick_example)


def test_run_quick_example_monkeypatch_train(monkeypatch, tmp_path):
    # Prepare a small sample CSV in the expected location
    repo_root = Path(__file__).resolve().parents[1]
    sample_dir = repo_root / "notebooks" / "examples_output" / "data"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "sample_dataset.csv"

    df = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [0.1, 0.2, 0.3, 0.4], "target": [0, 1, 0, 1]})
    df.to_csv(sample_path, index=False)

    # Monkeypatch ModelTrainer.train_model to a lightweight stub so example runs quickly
    from tree_models.models.trainer import ModelTrainer

    def fake_train(self, *args, **kwargs):
        class R:
            model = None
            model_type = getattr(args[0], "model_type", "xgboost") if args else "xgboost"
            train_metrics = {}
            validation_metrics = {}

        return R()

    monkeypatch.setattr(ModelTrainer, "train_model", fake_train)

    # Run main (should complete without raising)
    quick_example.main()
