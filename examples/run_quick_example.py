"""Quick runnable example using the sample dataset included in the repo.

This script demonstrates using `DataValidator` (with sampling) and a quick
training run using `ModelTrainer` with `random_state` propagation to
show reproducible behavior.

Run:
    python -m examples.run_quick_example
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tree_models.data.validator import DataValidator, ValidationConfig
from tree_models.models.trainer import ModelTrainer
from tree_models.config.model_config import XGBoostConfig


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sample_path = repo_root / "notebooks" / "examples_output" / "data" / "sample_dataset.csv"

    if not sample_path.exists():
        print(f"Sample dataset not found at {sample_path}. Please provide a CSV to run the quick example.")
        return

    df = pd.read_csv(sample_path)
    if "target" not in df.columns:
        # Try to heuristically pick last column as target
        df.columns = [str(c) for c in df.columns]
        df["target"] = df.iloc[:, -1]

    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"Loaded sample dataset: {len(df)} rows, {X.shape[1]} features")

    # Run data validation with sampling (safe for large sample files)
    val_config = ValidationConfig(max_rows_limit=1000, sampling_strategy="random", generate_report=False)
    validator = DataValidator(random_state=42)
    results = validator.validate_dataset(df, target_column="target", config=val_config)

    print("Data validation completed:")
    print(f"  Was sampled: {results.was_sampled}")
    print(f"  Sampled from: {results.sampled_from_n}")
    print(f"  Sampled n: {results.sampled_n}")
    print(f"  Quality score: {results.data_quality_score}")

    # Quick train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Create trainer and config
    config = XGBoostConfig.for_fast_training() if hasattr(XGBoostConfig, "for_fast_training") else XGBoostConfig()
    trainer = ModelTrainer(random_state=42)

    # Run a lightweight training (small n_estimators)
    config.n_estimators = 20
    training_config = None

    results = trainer.train_model(
        config, X_train, y_train, X_valid=X_test, y_valid=y_test, training_config=training_config
    )

    print("Training completed:")
    print(f"  Model type: {results.model_type}")
    print(f"  Train metrics: {results.train_metrics}")
    print(f"  Validation metrics: {results.validation_metrics}")


if __name__ == "__main__":
    main()
