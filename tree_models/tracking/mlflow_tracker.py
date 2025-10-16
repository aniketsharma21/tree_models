# tree_models/tracking/mlflow_tracker.py
"""Enhanced MLflow experiment tracking with production-ready features.

This module provides comprehensive MLflow integration with:
- Type-safe experiment tracking and logging
- Automatic system and environment information capture
- Enhanced artifact management and versioning
- Model lifecycle management and deployment support
- Comprehensive error handling and validation
- Performance monitoring and resource tracking
- Integration with tree_models configuration system
"""

import mlflow
import mlflow.sklearn
import tempfile
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import asdict
import warnings

from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    TrackingError,
    ConfigurationError,
    handle_and_reraise,
    validate_parameter,
    create_error_context
)
from ..config.model_config import ModelConfig, ExperimentConfig

logger = get_logger(__name__)

# Optional MLflow integrations
try:
    import mlflow.xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import mlflow.lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import mlflow.catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


class MLflowTracker:
    """Enhanced MLflow experiment tracking with production features.

    Provides comprehensive experiment tracking with automatic logging,
    model versioning, and integration with tree_models configurations.

    Example:
        >>> tracker = MLflowTracker(
        ...     experiment_name="fraud_detection_v2",
        ...     tracking_uri="http://localhost:5000"
        ... )
        >>> 
        >>> with tracker.start_run("xgb_baseline") as run:
        ...     run.log_experiment_config(config)
        ...     run.log_model_training(model, metrics, artifacts)
        ...     run.log_evaluation_results(eval_results)
        ...     run.register_model("fraud_model", "Production")
    """

    def __init__(
        self, 
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        auto_log_system_info: bool = True,
        enable_system_metrics: bool = True,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Initialize enhanced MLflow tracker.

        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for the run
            tracking_uri: MLflow tracking server URI
            artifact_location: Custom artifact storage location
            auto_log_system_info: Whether to log system information automatically
            enable_system_metrics: Whether to track system resource metrics
            tags: Optional tags to set for the experiment

        Raises:
            TrackingError: If tracker initialization fails
        """
        validate_parameter("experiment_name", experiment_name, min_length=1)
        
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.auto_log_system_info = auto_log_system_info
        self.enable_system_metrics = enable_system_metrics
        
        # Tracking state
        self.run_id: Optional[str] = None
        self.experiment_id: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._active_run: Optional[mlflow.ActiveRun] = None
        
        # Performance tracking
        self._logged_metrics_count = 0
        self._logged_params_count = 0
        self._logged_artifacts_count = 0

        try:
            # Set tracking URI if provided
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"Set MLflow tracking URI: {tracking_uri}")

            # Setup experiment
            self._setup_experiment(artifact_location)
            
            # Set experiment-level tags
            if tags:
                self._set_experiment_tags(tags)

            logger.info(f"Initialized MLflowTracker for experiment: {experiment_name}")

        except Exception as e:
            handle_and_reraise(
                e, TrackingError,
                f"Failed to initialize MLflow tracker",
                error_code="TRACKER_INIT_FAILED",
                context=create_error_context(
                    experiment_name=experiment_name,
                    tracking_uri=tracking_uri
                )
            )

    def _setup_experiment(self, artifact_location: Optional[str] = None) -> None:
        """Setup MLflow experiment with error handling."""
        
        try:
            # Try to create new experiment
            if artifact_location:
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name, 
                    artifact_location=artifact_location
                )
            else:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            
            logger.info(f"Created new experiment: {self.experiment_name} (ID: {self.experiment_id})")

        except Exception:
            # Experiment might already exist
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment and experiment.lifecycle_stage != "deleted":
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name} (ID: {self.experiment_id})")
            else:
                raise TrackingError(f"Could not create or access experiment: {self.experiment_name}")

    def _set_experiment_tags(self, tags: Dict[str, str]) -> None:
        """Set tags at the experiment level."""
        try:
            mlflow.set_experiment_tags(tags)
            logger.debug(f"Set {len(tags)} experiment-level tags")
        except Exception as e:
            logger.warning(f"Failed to set experiment tags: {e}")

    def __enter__(self) -> 'MLflowTracker':
        """Start MLflow run context with comprehensive initialization."""
        
        try:
            self._start_time = datetime.now(timezone.utc)
            
            self._active_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name
            )
            
            self.run_id = self._active_run.info.run_id
            
            logger.info(f"🚀 Started MLflow run: {self.run_id}")
            logger.info(f"   Experiment: {self.experiment_name}")
            logger.info(f"   Run name: {self.run_name or 'auto-generated'}")

            # Log system information if enabled
            if self.auto_log_system_info:
                self._log_system_info()

            return self

        except Exception as e:
            handle_and_reraise(
                e, TrackingError,
                "Failed to start MLflow run",
                error_code="RUN_START_FAILED"
            )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End MLflow run context with summary logging."""
        
        try:
            # Log run summary metrics
            if self._start_time:
                duration = (datetime.now(timezone.utc) - self._start_time).total_seconds()
                self.log_metrics({
                    "run_duration_seconds": duration,
                    "logged_metrics_count": self._logged_metrics_count,
                    "logged_params_count": self._logged_params_count,
                    "logged_artifacts_count": self._logged_artifacts_count
                })

            # Log exception information if run failed
            if exc_type is not None:
                self.set_tags({
                    "run_status": "FAILED",
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val) if exc_val else "Unknown error"
                })
                logger.error(f"Run failed with {exc_type.__name__}: {exc_val}")
            else:
                self.set_tags({"run_status": "COMPLETED"})

            mlflow.end_run()
            
            logger.info(f"✅ Ended MLflow run: {self.run_id}")
            logger.info(f"   Duration: {duration:.2f}s")
            logger.info(f"   Logged: {self._logged_params_count} params, "
                       f"{self._logged_metrics_count} metrics, "
                       f"{self._logged_artifacts_count} artifacts")

        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")

    def _log_system_info(self) -> None:
        """Log comprehensive system and environment information."""
        
        try:
            with timed_operation("system_info_logging"):
                system_info = self._gather_system_info()
                
                # Log as parameters and tags
                mlflow.log_params(system_info['params'])
                self.set_tags(system_info['tags'])
                
                # Log detailed system info as artifact
                self.log_dict(system_info, "system_info.json", "metadata")
                
                logger.debug("Logged system information")

        except Exception as e:
            logger.warning(f"Could not log system info: {e}")

    def _gather_system_info(self) -> Dict[str, Dict[str, Any]]:
        """Gather comprehensive system information."""
        
        import sys
        import platform
        import psutil
        
        params = {}
        tags = {}
        
        # Python and system info
        params['python_version'] = sys.version.split()[0]
        params['platform'] = platform.platform()
        params['cpu_count'] = psutil.cpu_count()
        params['memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
        
        tags['python_version'] = sys.version.split()[0]
        tags['platform'] = platform.system()
        tags['start_time'] = self._start_time.isoformat()
        tags['environment'] = os.environ.get("ENVIRONMENT", "development")
        tags['user'] = os.environ.get("USER", "unknown")
        
        # Git information
        try:
            git_info = self._get_git_info()
            params.update(git_info['params'])
            tags.update(git_info['tags'])
        except Exception:
            pass
        
        # Package versions
        try:
            package_info = self._get_package_versions()
            params.update(package_info)
        except Exception:
            pass
        
        return {
            'params': params,
            'tags': tags,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'hostname': platform.node()
        }

    def _get_git_info(self) -> Dict[str, Dict[str, str]]:
        """Get git repository information."""
        
        try:
            # Git commit hash
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            
            # Git branch
            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            
            # Git status (check if dirty)
            git_status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            
            is_dirty = len(git_status) > 0
            
            return {
                'params': {
                    'git_commit_hash': git_commit,
                    'git_branch': git_branch,
                    'git_is_dirty': is_dirty
                },
                'tags': {
                    'git_commit': git_commit[:8],  # Short hash for tags
                    'git_branch': git_branch,
                    'git_dirty': str(is_dirty).lower()
                }
            }
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {'params': {}, 'tags': {}}

    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        
        packages = {}
        
        try:
            import sklearn
            packages['sklearn_version'] = sklearn.__version__
        except ImportError:
            pass
        
        try:
            import xgboost
            packages['xgboost_version'] = xgboost.__version__
        except ImportError:
            pass
        
        try:
            import lightgbm
            packages['lightgbm_version'] = lightgbm.__version__
        except ImportError:
            pass
        
        try:
            import catboost
            packages['catboost_version'] = catboost.__version__
        except ImportError:
            pass
        
        try:
            import pandas
            packages['pandas_version'] = pandas.__version__
        except ImportError:
            pass
        
        try:
            import numpy
            packages['numpy_version'] = numpy.__version__
        except ImportError:
            pass
        
        return packages

    @timer(name="mlflow_params_logging")
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow with type conversion and validation.

        Args:
            params: Dictionary of parameters to log

        Example:
            >>> tracker.log_params({
            ...     "max_depth": 6, 
            ...     "learning_rate": 0.1,
            ...     "model_type": "xgboost"
            ... })
        """
        try:
            # Convert and validate parameters
            converted_params = self._convert_params(params)
            
            # MLflow has a limit on parameter key/value length
            validated_params = self._validate_params(converted_params)
            
            mlflow.log_params(validated_params)
            
            self._logged_params_count += len(validated_params)
            logger.debug(f"Logged {len(validated_params)} parameters")

        except Exception as e:
            handle_and_reraise(
                e, TrackingError,
                "Failed to log parameters",
                error_code="PARAMS_LOG_FAILED",
                context={"n_params": len(params)}
            )

    def _convert_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameters to MLflow-compatible types."""
        
        converted = {}
        
        for key, value in params.items():
            # Convert numpy types
            if isinstance(value, np.integer):
                converted[key] = int(value)
            elif isinstance(value, np.floating):
                converted[key] = float(value)
            elif isinstance(value, np.bool_):
                converted[key] = bool(value)
            elif isinstance(value, np.ndarray):
                # Convert small arrays to strings
                if value.size <= 10:
                    converted[key] = str(value.tolist())
                else:
                    converted[key] = f"array_shape_{value.shape}"
            elif isinstance(value, (list, tuple)):
                # Convert to string if reasonable length
                if len(str(value)) <= 500:
                    converted[key] = str(value)
                else:
                    converted[key] = f"list_length_{len(value)}"
            elif isinstance(value, dict):
                # Flatten simple dicts or convert to string
                if len(str(value)) <= 500:
                    converted[key] = json.dumps(value)
                else:
                    converted[key] = f"dict_keys_{len(value)}"
            else:
                converted[key] = str(value)
        
        return converted

    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for MLflow constraints."""
        
        validated = {}
        
        for key, value in params.items():
            # MLflow parameter key limit: 250 characters
            if len(key) > 250:
                short_key = key[:247] + "..."
                logger.warning(f"Parameter key truncated: {key} -> {short_key}")
                key = short_key
            
            # MLflow parameter value limit: 500 characters  
            value_str = str(value)
            if len(value_str) > 500:
                short_value = value_str[:497] + "..."
                logger.warning(f"Parameter value truncated for {key}")
                value_str = short_value
            
            validated[key] = value_str
        
        return validated

    @timer(name="mlflow_metrics_logging")
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow with validation and error handling.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time series metrics

        Example:
            >>> tracker.log_metrics({
            ...     "train_auc": 0.85, 
            ...     "valid_auc": 0.82,
            ...     "train_loss": 0.25
            ... })
        """
        try:
            # Convert and validate metrics
            converted_metrics = self._convert_metrics(metrics)
            
            # Log metrics
            if step is not None:
                for key, value in converted_metrics.items():
                    mlflow.log_metric(key, value, step)
            else:
                mlflow.log_metrics(converted_metrics)

            self._logged_metrics_count += len(converted_metrics)
            logger.debug(f"Logged {len(converted_metrics)} metrics")

        except Exception as e:
            handle_and_reraise(
                e, TrackingError,
                "Failed to log metrics",
                error_code="METRICS_LOG_FAILED",
                context={"n_metrics": len(metrics)}
            )

    def _convert_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Convert metrics to float values with validation."""
        
        converted = {}
        
        for key, value in metrics.items():
            try:
                # Convert to float
                if isinstance(value, (np.integer, np.floating, int, float)):
                    float_value = float(value)
                    
                    # Check for invalid values
                    if np.isnan(float_value):
                        logger.warning(f"Metric '{key}' is NaN, skipping")
                        continue
                    elif np.isinf(float_value):
                        logger.warning(f"Metric '{key}' is infinite, skipping")
                        continue
                    
                    converted[key] = float_value
                else:
                    logger.warning(f"Metric '{key}' is not numeric: {type(value)}, skipping")
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert metric '{key}': {e}")
        
        return converted

    def log_model(
        self, 
        model: Any, 
        artifact_path: str,
        model_type: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        registered_model_name: Optional[str] = None
    ) -> None:
        """Log model to MLflow with enhanced model type detection.

        Args:
            model: Trained model object
            artifact_path: Path within the run's artifact URI
            model_type: Model type override ("sklearn", "xgboost", "lightgbm", "catboost")
            signature: Model signature for input/output schema
            input_example: Example input for the model
            registered_model_name: Name to register the model under

        Example:
            >>> tracker.log_model(
            ...     xgb_model, 
            ...     "model",
            ...     registered_model_name="fraud_detection_model"
            ... )
        """
        try:
            # Auto-detect model type if not provided
            if model_type is None:
                model_type = self._detect_model_type(model)

            logger.info(f"Logging {model_type} model to {artifact_path}")

            # Log model using appropriate MLflow method
            if model_type == "xgboost" and XGBOOST_AVAILABLE:
                mlflow.xgboost.log_model(
                    model, artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
                mlflow.lightgbm.log_model(
                    model, artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            elif model_type == "catboost" and CATBOOST_AVAILABLE:
                mlflow.catboost.log_model(
                    model, artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            else:
                # Fallback to sklearn or generic pickle
                mlflow.sklearn.log_model(
                    model, artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )

            self._logged_artifacts_count += 1
            logger.info(f"Successfully logged {model_type} model")

        except Exception as e:
            handle_and_reraise(
                e, TrackingError,
                f"Failed to log model",
                error_code="MODEL_LOG_FAILED",
                context={"model_type": model_type, "artifact_path": artifact_path}
            )

    def _detect_model_type(self, model: Any) -> str:
        """Auto-detect model type from model object."""
        
        model_class_name = model.__class__.__name__.lower()
        model_module = model.__class__.__module__.lower()
        
        if "xgb" in model_class_name or "xgboost" in model_module:
            return "xgboost"
        elif "lgb" in model_class_name or "lightgbm" in model_module:
            return "lightgbm"
        elif "catboost" in model_class_name or "catboost" in model_module:
            return "catboost"
        else:
            return "sklearn"

    def log_artifact(
        self, 
        local_path: Union[str, Path], 
        artifact_path: Optional[str] = None
    ) -> None:
        """Log artifact file to MLflow with validation."""
        
        try:
            local_path = Path(local_path)
            
            if not local_path.exists():
                raise TrackingError(f"Artifact file not found: {local_path}")
            
            mlflow.log_artifact(str(local_path), artifact_path)
            
            self._logged_artifacts_count += 1
            logger.debug(f"Logged artifact: {local_path.name}")

        except Exception as e:
            handle_and_reraise(
                e, TrackingError,
                f"Failed to log artifact {local_path}",
                error_code="ARTIFACT_LOG_FAILED"
            )

    def log_dataframe(
        self, 
        df: pd.DataFrame, 
        filename: str,
        artifact_path: Optional[str] = None,
        format: str = "csv"
    ) -> None:
        """Log pandas DataFrame as artifact with multiple format support.

        Args:
            df: DataFrame to log
            filename: Filename for the artifact
            artifact_path: Optional path within run's artifact directory
            format: File format ("csv", "parquet", "json")

        Example:
            >>> tracker.log_dataframe(results_df, "evaluation_results.csv")
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / filename

                if format == "csv" or filename.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                elif format == "parquet" or filename.endswith('.parquet'):
                    df.to_parquet(file_path, index=False)
                elif format == "json" or filename.endswith('.json'):
                    df.to_json(file_path, orient='records', indent=2)
                else:
                    # Default to CSV
                    df.to_csv(file_path, index=False)

                self.log_artifact(file_path, artifact_path)

        except Exception as e:
            handle_and_reraise(
                e, TrackingError,
                f"Failed to log DataFrame as {filename}",
                error_code="DATAFRAME_LOG_FAILED"
            )

    def log_dict(
        self, 
        dictionary: Dict[str, Any], 
        filename: str,
        artifact_path: Optional[str] = None,
        indent: int = 2
    ) -> None:
        """Log dictionary as JSON artifact with enhanced serialization."""
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / filename

                # Custom JSON encoder for numpy types and other objects
                def json_encoder(obj):
                    if isinstance(obj, (np.integer, np.floating, np.bool_)):
                        return obj.item()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    else:
                        return str(obj)

                with open(file_path, 'w') as f:
                    json.dump(dictionary, f, indent=indent, default=json_encoder, ensure_ascii=False)

                self.log_artifact(file_path, artifact_path)

        except Exception as e:
            handle_and_reraise(
                e, TrackingError,
                f"Failed to log dictionary as {filename}",
                error_code="DICT_LOG_FAILED"
            )

    def log_experiment_config(self, config: Union[ExperimentConfig, Dict[str, Any]]) -> None:
        """Log experiment configuration with structured format.

        Args:
            config: Experiment configuration object or dictionary

        Example:
            >>> tracker.log_experiment_config(experiment_config)
        """
        try:
            if isinstance(config, ExperimentConfig):
                config_dict = asdict(config)
            else:
                config_dict = config

            # Log model parameters
            if 'model' in config_dict:
                model_params = config_dict['model']
                self.log_params({f"model_{k}": v for k, v in model_params.items()})

            # Log data configuration
            if 'data' in config_dict:
                data_params = config_dict['data']
                self.log_params({f"data_{k}": v for k, v in data_params.items()})

            # Log full config as artifact
            self.log_dict(config_dict, "experiment_config.json", "config")

            logger.info("Logged experiment configuration")

        except Exception as e:
            handle_and_reraise(
                e, TrackingError,
                "Failed to log experiment configuration",
                error_code="CONFIG_LOG_FAILED"
            )

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags for the run with validation."""
        
        try:
            # Validate and convert tags
            validated_tags = {}
            
            for key, value in tags.items():
                # Ensure both key and value are strings and within limits
                key_str = str(key)[:250]  # MLflow tag key limit
                value_str = str(value)[:5000]  # MLflow tag value limit
                
                validated_tags[key_str] = value_str

            # Set tags
            for key, value in validated_tags.items():
                mlflow.set_tag(key, value)

            logger.debug(f"Set {len(validated_tags)} tags")

        except Exception as e:
            logger.warning(f"Failed to set tags: {e}")

    def get_run_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current run."""
        
        if not self.run_id:
            return {"status": "not_started"}

        try:
            run = mlflow.get_run(self.run_id)
            
            return {
                "run_id": self.run_id,
                "experiment_id": self.experiment_id,
                "experiment_name": self.experiment_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "lifecycle_stage": run.info.lifecycle_stage,
                "user_id": run.info.user_id,
                "logged_metrics": self._logged_metrics_count,
                "logged_params": self._logged_params_count,
                "logged_artifacts": self._logged_artifacts_count
            }
            
        except Exception as e:
            logger.warning(f"Could not get run info: {e}")
            return {"error": str(e)}


# High-level experiment management
class ExperimentTracker:
    """High-level experiment tracking interface with enhanced features.

    Provides simplified experiment management with automatic configuration
    handling and model lifecycle support.

    Example:
        >>> tracker = ExperimentTracker("fraud_detection_v2")
        >>> 
        >>> # Run multiple experiments
        >>> with tracker.start_run("xgb_baseline") as run:
        ...     run.log_experiment_config(config)
        ...     # ... training code ...
        ...     run.log_model(model, "model")
        >>> 
        >>> # Compare experiments
        >>> best_run = tracker.get_best_run("valid_auc", ascending=False)
        >>> tracker.promote_model(best_run.info.run_id, "Production")
    """

    def __init__(
        self, 
        experiment_name: str, 
        tracking_uri: Optional[str] = None,
        default_tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI
            default_tags: Default tags to apply to all runs
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.default_tags = default_tags or {}

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        logger.info(f"Initialized ExperimentTracker: {experiment_name}")

    def start_run(
        self, 
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> MLflowTracker:
        """Start a new experiment run.

        Args:
            run_name: Optional name for the run
            tags: Additional tags for this run

        Returns:
            MLflow tracker context manager
        """
        # Merge default tags with run-specific tags
        all_tags = self.default_tags.copy()
        if tags:
            all_tags.update(tags)

        return MLflowTracker(
            experiment_name=self.experiment_name,
            run_name=run_name,
            tracking_uri=self.tracking_uri,
            tags=all_tags
        )

    def list_runs(
        self, 
        max_results: int = 100,
        order_by: Optional[List[str]] = None,
        filter_string: Optional[str] = None
    ) -> pd.DataFrame:
        """List runs from the experiment with filtering and ordering.

        Args:
            max_results: Maximum number of runs to return
            order_by: List of columns to order by (e.g., ["metrics.auc DESC"])
            filter_string: MLflow filter string

        Returns:
            DataFrame with run information
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                logger.warning(f"Experiment '{self.experiment_name}' not found")
                return pd.DataFrame()

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=order_by,
                filter_string=filter_string
            )

            return runs

        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            return pd.DataFrame()

    def get_best_run(
        self, 
        metric_name: str, 
        ascending: bool = False,
        filter_string: Optional[str] = None
    ) -> Optional[mlflow.entities.Run]:
        """Get the best run based on a metric.

        Args:
            metric_name: Name of the metric to optimize
            ascending: Whether lower values are better
            filter_string: Optional filter to apply

        Returns:
            Best run object or None
        """
        try:
            order_direction = "ASC" if ascending else "DESC"
            order_by = [f"metrics.{metric_name} {order_direction}"]

            runs_df = self.list_runs(
                max_results=1,
                order_by=order_by,
                filter_string=filter_string
            )

            if runs_df.empty:
                return None

            best_run_id = runs_df.iloc[0]['run_id']
            return mlflow.get_run(best_run_id)

        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None

    def compare_runs(
        self, 
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare multiple runs across specified metrics.

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to include in comparison

        Returns:
            DataFrame comparing the runs
        """
        try:
            comparison_data = []

            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                
                row = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', ''),
                    'status': run.info.status,
                    'start_time': run.info.start_time
                }

                # Add metrics
                if metrics:
                    for metric in metrics:
                        row[f'metrics.{metric}'] = run.data.metrics.get(metric)
                else:
                    # Include all metrics
                    for metric, value in run.data.metrics.items():
                        row[f'metrics.{metric}'] = value

                comparison_data.append(row)

            return pd.DataFrame(comparison_data)

        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return pd.DataFrame()


# Convenience functions
def start_experiment(
    experiment_name: str,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    **kwargs
) -> MLflowTracker:
    """Start an experiment run with convenience interface.

    Args:
        experiment_name: Name of the experiment
        run_name: Optional name for the run
        tracking_uri: MLflow tracking server URI
        **kwargs: Additional tracker parameters

    Returns:
        MLflow tracker context manager

    Example:
        >>> with start_experiment("fraud_detection", "baseline_run") as tracker:
        ...     tracker.log_params(model_params)
        ...     tracker.log_metrics(results)
        ...     tracker.log_model(model, "model")
    """
    return MLflowTracker(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        **kwargs
    )


# Export key classes and functions
__all__ = [
    'MLflowTracker',
    'ExperimentTracker',
    'start_experiment'
]