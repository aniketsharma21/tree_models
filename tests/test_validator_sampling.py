import pandas as pd
import numpy as np

from tree_models.data.validator import DataValidator, ValidationConfig


def test_validator_sampling_behavior():
    # Create a DataFrame larger than the max_rows_limit
    n = 2000
    df = pd.DataFrame({"x": np.arange(n), "y": np.random.randn(n)})

    config = ValidationConfig(max_rows_limit=500, sampling_strategy="random", generate_report=False)
    validator = DataValidator(random_state=42)
    results = validator.validate_dataset(df, target_column=None, config=config)

    assert results.was_sampled is True
    assert results.sampled_from_n == n
    assert results.sampled_n == 500
