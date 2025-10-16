"""Input/Output utilities for models, configs, and data.

This module provides abstracted I/O functions for saving and loading various
file types used in the ML pipeline.
"""

import json
import pickle
import joblib
import os
from pathlib import Path
from typing import Any, Dict, Union, Optional
import pandas as pd

from .logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object of the directory

    Example:
        >>> output_dir = ensure_dir("models/experiments/run_001")
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Output file path

    Example:
        >>> metrics = {"auc": 0.85, "precision": 0.78}
        >>> save_json(metrics, "results/metrics.json")
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded dictionary

    Example:
        >>> metrics = load_json("results/metrics.json")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded JSON from {filepath}")
    return data


def save_model(model: Any, filepath: Union[str, Path], method: str = "joblib") -> None:
    """Save ML model to disk.

    Args:
        model: Model object to save
        filepath: Output file path
        method: Serialization method ("joblib", "pickle")

    Example:
        >>> save_model(trained_model, "models/xgb_baseline.pkl")
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if method == "joblib":
        joblib.dump(model, filepath)
    elif method == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unsupported method: {method}")

    logger.info(f"Saved model to {filepath} using {method}")


def load_model(filepath: Union[str, Path], method: str = "joblib") -> Any:
    """Load ML model from disk.

    Args:
        filepath: Model file path
        method: Serialization method ("joblib", "pickle")

    Returns:
        Loaded model object

    Example:
        >>> model = load_model("models/xgb_baseline.pkl")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    if method == "joblib":
        model = joblib.load(filepath)
    elif method == "pickle":
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Unsupported method: {method}")

    logger.info(f"Loaded model from {filepath} using {method}")
    return model


def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], 
                  format: str = "csv", **kwargs) -> None:
    """Save pandas DataFrame to file.

    Args:
        df: DataFrame to save
        filepath: Output file path
        format: File format ("csv", "parquet", "feather")
        **kwargs: Additional arguments for pandas save methods

    Example:
        >>> save_dataframe(results_df, "data/processed/results.csv", index=False)
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if format == "csv":
        df.to_csv(filepath, **kwargs)
    elif format == "parquet":
        df.to_parquet(filepath, **kwargs)
    elif format == "feather":
        df.to_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved DataFrame ({df.shape[0]}x{df.shape[1]}) to {filepath}")


def load_dataframe(filepath: Union[str, Path], format: str = "csv", 
                  **kwargs) -> pd.DataFrame:
    """Load pandas DataFrame from file.

    Args:
        filepath: Input file path
        format: File format ("csv", "parquet", "feather")
        **kwargs: Additional arguments for pandas load methods

    Returns:
        Loaded DataFrame

    Example:
        >>> df = load_dataframe("data/processed/results.csv")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if format == "csv":
        df = pd.read_csv(filepath, **kwargs)
    elif format == "parquet":
        df = pd.read_parquet(filepath, **kwargs)
    elif format == "feather":
        df = pd.read_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Loaded DataFrame ({df.shape[0]}x{df.shape[1]}) from {filepath}")
    return df


def get_file_size(filepath: Union[str, Path]) -> str:
    """Get human-readable file size.

    Args:
        filepath: File path

    Returns:
        File size as string (e.g., "1.2 MB")

    Example:
        >>> size = get_file_size("models/large_model.pkl")
        >>> print(f"Model size: {size}")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return "File not found"

    size_bytes = filepath.stat().st_size

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} TB"


def list_files(directory: Union[str, Path], pattern: str = "*") -> list:
    """List files in directory matching pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern (default: "*")

    Returns:
        List of matching file paths

    Example:
        >>> model_files = list_files("models/", "*.pkl")
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    files = list(directory.glob(pattern))
    logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")

    return [str(f) for f in files]
