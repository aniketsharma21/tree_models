"""Configuration validation schemas using Pydantic.

This module provides validation schemas to ensure configuration loaded
from YAML files meets expected types and constraints.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import re

try:
    from pydantic import BaseModel, Field, validator, root_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback validation without Pydantic
    BaseModel = object
    PYDANTIC_AVAILABLE = False


def validate_config_dict(config_dict: Dict[str, Any], schema_class) -> Dict[str, Any]:
    """Validate configuration dictionary against schema.

    Args:
        config_dict: Configuration dictionary to validate
        schema_class: Schema class to validate against

    Returns:
        Validated configuration dictionary

    Raises:
        ValidationError: If configuration is invalid
    """
    if PYDANTIC_AVAILABLE:
        validated = schema_class(**config_dict)
        return validated.dict()
    else:
        # Basic validation without Pydantic
        return _basic_validation(config_dict, schema_class)


def _basic_validation(config_dict: Dict[str, Any], schema_class) -> Dict[str, Any]:
    """Basic validation without Pydantic."""
    # This is a simplified validation - in production, you'd want Pydantic
    return config_dict


if PYDANTIC_AVAILABLE:

    class ModelConfigSchema(BaseModel):
        """Schema for model configuration validation."""

        model_type: str = Field(..., regex=r"^(xgboost|lightgbm|catboost)$")
        n_estimators: int = Field(50, ge=1, le=10000)
        max_depth: int = Field(6, ge=1, le=20)
        learning_rate: float = Field(0.1, gt=0.0, le=1.0)
        random_state: int = Field(42, ge=0)
        early_stopping_rounds: int = Field(50, ge=1)
        verbose: bool = False
        custom_params: Dict[str, Any] = Field(default_factory=dict)

        @validator('model_type')
        def validate_model_type(cls, v):
            """Validate model type."""
            valid_types = ['xgboost', 'lightgbm', 'catboost']
            if v.lower() not in valid_types:
                raise ValueError(f'model_type must be one of {valid_types}')
            return v.lower()


    class XGBoostConfigSchema(ModelConfigSchema):
        """Schema for XGBoost configuration validation."""

        model_type: str = Field('xgboost', const=True)
        subsample: float = Field(0.8, gt=0.0, le=1.0)
        colsample_bytree: float = Field(0.8, gt=0.0, le=1.0)
        reg_alpha: float = Field(0.0, ge=0.0)
        reg_lambda: float = Field(1.0, ge=0.0)
        objective: str = Field('binary:logistic')
        eval_metric: str = Field('auc')

        @validator('objective')
        def validate_objective(cls, v):
            """Validate XGBoost objective."""
            valid_objectives = [
                'binary:logistic', 'binary:hinge', 'reg:squarederror', 
                'multi:softmax', 'multi:softprob'
            ]
            if v not in valid_objectives:
                raise ValueError(f'objective must be one of {valid_objectives}')
            return v


    class LightGBMConfigSchema(ModelConfigSchema):
        """Schema for LightGBM configuration validation."""

        model_type: str = Field('lightgbm', const=True)
        feature_fraction: float = Field(0.8, gt=0.0, le=1.0)
        bagging_fraction: float = Field(0.8, gt=0.0, le=1.0)
        bagging_freq: int = Field(5, ge=0)
        min_child_samples: int = Field(20, ge=1)
        reg_alpha: float = Field(0.0, ge=0.0)
        reg_lambda: float = Field(0.0, ge=0.0)
        objective: str = Field('binary')
        metric: str = Field('auc')
        boosting_type: str = Field('gbdt')

        @validator('boosting_type')
        def validate_boosting_type(cls, v):
            """Validate LightGBM boosting type."""
            valid_types = ['gbdt', 'dart', 'goss', 'rf']
            if v not in valid_types:
                raise ValueError(f'boosting_type must be one of {valid_types}')
            return v


    class CatBoostConfigSchema(ModelConfigSchema):
        """Schema for CatBoost configuration validation."""

        model_type: str = Field('catboost', const=True)
        depth: int = Field(6, ge=1, le=16)
        l2_leaf_reg: float = Field(3.0, ge=0.0)
        subsample: float = Field(0.8, gt=0.0, le=1.0)
        colsample_bylevel: float = Field(1.0, gt=0.0, le=1.0)
        loss_function: str = Field('Logloss')
        eval_metric: str = Field('AUC')


    class DataConfigSchema(BaseModel):
        """Schema for data configuration validation."""

        train_path: Optional[str] = None
        test_path: Optional[str] = None
        output_dir: str = Field('output')
        target_col: str = Field('target', min_length=1)
        weight_col: Optional[str] = None
        id_col: Optional[str] = None
        test_size: float = Field(0.2, gt=0.0, lt=1.0)
        valid_size: float = Field(0.2, gt=0.0, lt=1.0)
        stratify: bool = True
        sample_size: Optional[int] = Field(None, gt=0)
        random_state: int = Field(42, ge=0)
        missing_strategy: str = Field('median')
        categorical_strategy: str = Field('most_frequent')
        encoding_strategy: str = Field('label')
        scaling_strategy: Optional[str] = None
        use_knn_imputation: bool = False

        @validator('missing_strategy')
        def validate_missing_strategy(cls, v):
            """Validate missing value strategy."""
            valid_strategies = ['mean', 'median', 'most_frequent', 'constant']
            if v not in valid_strategies:
                raise ValueError(f'missing_strategy must be one of {valid_strategies}')
            return v

        @validator('encoding_strategy')
        def validate_encoding_strategy(cls, v):
            """Validate encoding strategy."""
            valid_strategies = ['label', 'onehot', 'target']
            if v not in valid_strategies:
                raise ValueError(f'encoding_strategy must be one of {valid_strategies}')
            return v

        @validator('scaling_strategy')
        def validate_scaling_strategy(cls, v):
            """Validate scaling strategy."""
            if v is not None:
                valid_strategies = ['standard', 'minmax', 'robust']
                if v not in valid_strategies:
                    raise ValueError(f'scaling_strategy must be one of {valid_strategies}')
            return v

        @root_validator
        def validate_sizes(cls, values):
            """Validate that test_size + valid_size < 1.0."""
            test_size = values.get('test_size', 0.2)
            valid_size = values.get('valid_size', 0.2)
            if test_size + valid_size >= 1.0:
                raise ValueError('test_size + valid_size must be less than 1.0')
            return values


    class FeatureSelectionConfigSchema(BaseModel):
        """Schema for feature selection configuration validation."""

        enable_feature_selection: bool = True
        save_results: bool = True
        variance_threshold: float = Field(0.01, ge=0.0)
        rfecv_enabled: bool = True
        rfecv_cv_folds: int = Field(5, ge=2, le=20)
        rfecv_scoring: str = Field('roc_auc')
        rfecv_step: int = Field(1, ge=1)
        rfecv_min_features: int = Field(1, ge=1)
        boruta_enabled: bool = True
        boruta_max_iter: int = Field(100, ge=10, le=1000)
        boruta_alpha: float = Field(0.05, gt=0.0, lt=1.0)
        boruta_two_step: bool = True
        min_agreement: int = Field(2, ge=1)

        @validator('rfecv_scoring')
        def validate_scoring(cls, v):
            """Validate scoring metric."""
            valid_metrics = [
                'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
                'average_precision', 'neg_log_loss'
            ]
            if v not in valid_metrics:
                raise ValueError(f'rfecv_scoring must be one of {valid_metrics}')
            return v


    class TuningConfigSchema(BaseModel):
        """Schema for hyperparameter tuning configuration validation."""

        enable_tuning: bool = True
        n_trials: int = Field(100, ge=1, le=10000)
        timeout: Optional[int] = Field(None, gt=0)
        cv_folds: int = Field(5, ge=2, le=20)
        scoring: str = Field('roc_auc')
        sampler: str = Field('tpe')
        pruner: str = Field('median')
        n_startup_trials: int = Field(10, ge=1)
        n_warmup_steps: int = Field(10, ge=0)
        n_jobs: int = Field(1, ge=-1)
        log_to_mlflow: bool = True
        custom_search_space: Dict[str, Any] = Field(default_factory=dict)

        @validator('sampler')
        def validate_sampler(cls, v):
            """Validate Optuna sampler."""
            valid_samplers = ['tpe', 'random', 'grid', 'cmaes']
            if v not in valid_samplers:
                raise ValueError(f'sampler must be one of {valid_samplers}')
            return v

        @validator('pruner')
        def validate_pruner(cls, v):
            """Validate Optuna pruner."""
            valid_pruners = ['median', 'successive_halving', 'hyperband', 'nop']
            if v not in valid_pruners:
                raise ValueError(f'pruner must be one of {valid_pruners}')
            return v

        @validator('scoring')
        def validate_scoring(cls, v):
            """Validate scoring metric."""
            valid_metrics = [
                'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
                'average_precision', 'neg_log_loss'
            ]
            if v not in valid_metrics:
                raise ValueError(f'scoring must be one of {valid_metrics}')
            return v


    class EvaluationConfigSchema(BaseModel):
        """Schema for evaluation configuration validation."""

        generate_plots: bool = True
        save_results: bool = True
        plot_format: str = Field('png')
        plot_dpi: int = Field(300, ge=72, le=600)
        primary_metric: str = Field('auc_roc')
        threshold: float = Field(0.5, gt=0.0, lt=1.0)
        optimize_threshold: bool = True
        threshold_metrics: List[str] = Field(default_factory=lambda: ['f1', 'precision', 'recall', 'youden'])
        gains_bins: int = Field(10, ge=2, le=100)
        eval_cv_folds: int = Field(5, ge=2, le=20)

        @validator('plot_format')
        def validate_plot_format(cls, v):
            """Validate plot format."""
            valid_formats = ['png', 'pdf', 'svg', 'jpg', 'eps']
            if v not in valid_formats:
                raise ValueError(f'plot_format must be one of {valid_formats}')
            return v

        @validator('threshold_metrics')
        def validate_threshold_metrics(cls, v):
            """Validate threshold optimization metrics."""
            valid_metrics = ['f1', 'precision', 'recall', 'youden', 'accuracy']
            for metric in v:
                if metric not in valid_metrics:
                    raise ValueError(f'threshold_metrics must contain only {valid_metrics}')
            return v


    class MLflowConfigSchema(BaseModel):
        """Schema for MLflow configuration validation."""

        tracking_uri: Optional[str] = None
        experiment_name: str = Field('tree_model_experiment', min_length=1)
        run_name: Optional[str] = None
        log_params: bool = True
        log_metrics: bool = True
        log_artifacts: bool = True
        log_model: bool = True
        log_plots: bool = True
        auto_log_system_info: bool = True
        log_git_commit: bool = True
        log_environment: bool = True

        @validator('experiment_name')
        def validate_experiment_name(cls, v):
            """Validate experiment name format."""
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError('experiment_name must contain only alphanumeric characters, underscores, and hyphens')
            return v


    class ExperimentConfigSchema(BaseModel):
        """Schema for complete experiment configuration validation."""

        model: Union[ModelConfigSchema, XGBoostConfigSchema, LightGBMConfigSchema, CatBoostConfigSchema] = Field(default_factory=ModelConfigSchema)
        data: DataConfigSchema = Field(default_factory=DataConfigSchema)
        feature_selection: FeatureSelectionConfigSchema = Field(default_factory=FeatureSelectionConfigSchema)
        tuning: TuningConfigSchema = Field(default_factory=TuningConfigSchema)
        evaluation: EvaluationConfigSchema = Field(default_factory=EvaluationConfigSchema)
        mlflow: MLflowConfigSchema = Field(default_factory=MLflowConfigSchema)
        seed: int = Field(42, ge=0)
        verbose: bool = True
        debug: bool = False
        parallel_jobs: int = Field(1, ge=-1)
        memory_limit: Optional[str] = None

        @validator('memory_limit')
        def validate_memory_limit(cls, v):
            """Validate memory limit format."""
            if v is not None:
                if not re.match(r'^\d+[KMGT]?B?$', v.upper()):
                    raise ValueError('memory_limit must be in format like "4GB", "512MB", "1024"')
            return v

        @root_validator
        def validate_model_specific_config(cls, values):
            """Validate that model config matches model type."""
            model_config = values.get('model', {})
            if isinstance(model_config, dict):
                model_type = model_config.get('model_type', 'xgboost')
            else:
                model_type = getattr(model_config, 'model_type', 'xgboost')

            # Ensure we use the appropriate schema for the model type
            if model_type == 'xgboost' and not isinstance(model_config, XGBoostConfigSchema):
                values['model'] = XGBoostConfigSchema(**model_config if isinstance(model_config, dict) else model_config.dict())
            elif model_type == 'lightgbm' and not isinstance(model_config, LightGBMConfigSchema):
                values['model'] = LightGBMConfigSchema(**model_config if isinstance(model_config, dict) else model_config.dict())
            elif model_type == 'catboost' and not isinstance(model_config, CatBoostConfigSchema):
                values['model'] = CatBoostConfigSchema(**model_config if isinstance(model_config, dict) else model_config.dict())

            return values

else:
    # Fallback schemas without Pydantic
    class ModelConfigSchema:
        pass

    class XGBoostConfigSchema:
        pass

    class LightGBMConfigSchema:
        pass

    class CatBoostConfigSchema:
        pass

    class DataConfigSchema:
        pass

    class FeatureSelectionConfigSchema:
        pass

    class TuningConfigSchema:
        pass

    class EvaluationConfigSchema:
        pass

    class MLflowConfigSchema:
        pass

    class ExperimentConfigSchema:
        pass


# Configuration validation functions
def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate model configuration."""
    model_type = config.get('model_type', 'xgboost').lower()

    if model_type == 'xgboost':
        return validate_config_dict(config, XGBoostConfigSchema)
    elif model_type == 'lightgbm':
        return validate_config_dict(config, LightGBMConfigSchema)
    elif model_type == 'catboost':
        return validate_config_dict(config, CatBoostConfigSchema)
    else:
        return validate_config_dict(config, ModelConfigSchema)


def validate_experiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate complete experiment configuration."""
    return validate_config_dict(config, ExperimentConfigSchema)


def get_validation_schema(config_type: str):
    """Get validation schema for a specific configuration type.

    Args:
        config_type: Type of configuration ('model', 'data', 'tuning', etc.)

    Returns:
        Appropriate validation schema class
    """
    schema_map = {
        'model': ModelConfigSchema,
        'xgboost': XGBoostConfigSchema,
        'lightgbm': LightGBMConfigSchema,
        'catboost': CatBoostConfigSchema,
        'data': DataConfigSchema,
        'feature_selection': FeatureSelectionConfigSchema,
        'tuning': TuningConfigSchema,
        'evaluation': EvaluationConfigSchema,
        'mlflow': MLflowConfigSchema,
        'experiment': ExperimentConfigSchema
    }

    return schema_map.get(config_type, ModelConfigSchema)


# Validation utilities
def validate_file_path(path: str, must_exist: bool = False) -> bool:
    """Validate file path format and existence.

    Args:
        path: File path to validate
        must_exist: Whether file must already exist

    Returns:
        True if path is valid
    """
    try:
        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            return False

        # Check if parent directory exists or can be created
        if not must_exist:
            path_obj.parent.mkdir(parents=True, exist_ok=True)

        return True
    except Exception:
        return False


def validate_model_parameters(model_type: str, params: Dict[str, Any]) -> List[str]:
    """Validate model-specific parameters.

    Args:
        model_type: Type of model
        params: Parameters to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if model_type == 'xgboost':
        # XGBoost-specific validations
        if 'n_estimators' in params and (params['n_estimators'] < 1 or params['n_estimators'] > 10000):
            errors.append("n_estimators must be between 1 and 10000")

        if 'max_depth' in params and (params['max_depth'] < 1 or params['max_depth'] > 20):
            errors.append("max_depth must be between 1 and 20")

        if 'learning_rate' in params and (params['learning_rate'] <= 0 or params['learning_rate'] > 1):
            errors.append("learning_rate must be between 0 and 1")

    elif model_type == 'lightgbm':
        # LightGBM-specific validations
        if 'num_leaves' in params and params['num_leaves'] < 2:
            errors.append("num_leaves must be at least 2")

        if 'min_child_samples' in params and params['min_child_samples'] < 1:
            errors.append("min_child_samples must be at least 1")

    elif model_type == 'catboost':
        # CatBoost-specific validations
        if 'depth' in params and (params['depth'] < 1 or params['depth'] > 16):
            errors.append("depth must be between 1 and 16")

    return errors
