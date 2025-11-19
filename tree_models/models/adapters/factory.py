from typing import Any

from .xgboost_adapter import XGBoostAdapter
from .lightgbm_adapter import LightGBMAdapter
from .catboost_adapter import CatBoostAdapter


def get_adapter(model_type: str) -> Any:
    """Return an adapter instance for the requested model type."""
    model_type = str(model_type).lower()

    if model_type == "xgboost":
        return XGBoostAdapter()
    if model_type == "lightgbm":
        return LightGBMAdapter()
    if model_type == "catboost":
        return CatBoostAdapter()

    raise ValueError(f"Unsupported model type for adapter: {model_type}")
