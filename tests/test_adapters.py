from tree_models.models.adapters import factory
from tree_models.models.adapters.xgboost_adapter import XGBoostAdapter
from tree_models.models.adapters.lightgbm_adapter import LightGBMAdapter
from tree_models.models.adapters.catboost_adapter import CatBoostAdapter


def test_get_adapter_instances():
    xa = factory.get_adapter("xgboost")
    assert isinstance(xa, XGBoostAdapter)

    la = factory.get_adapter("lightgbm")
    assert isinstance(la, LightGBMAdapter)

    ca = factory.get_adapter("catboost")
    assert isinstance(ca, CatBoostAdapter)

    try:
        factory.get_adapter("unsupported_model")
    except ValueError:
        # Expected
        pass
    else:
        raise AssertionError("Unsupported model did not raise ValueError")
