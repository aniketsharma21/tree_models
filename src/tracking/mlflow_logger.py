"""MLflow experiment tracking utilities.

This module provides a wrapper for MLflow logging with automatic tracking
of parameters, metrics, artifacts, and models.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import tempfile
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.io_utils import ensure_dir

logger = get_logger(__name__)


class MLflowLogger:
    """MLflow experiment tracking wrapper.

    Example:
        >>> with MLflowLogger(experiment_name="fraud_detection") as mlf:
        ...     mlf.log_params({"max_depth": 6, "learning_rate": 0.1})
        ...     mlf.log_metrics({"auc": 0.85, "precision": 0.78})
        ...     mlf.log_model(trained_model, "xgboost_model")
    """

    def __init__(self, 
                 experiment_name: str,
                 run_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None,
                 auto_log_system_info: bool = True):
        """Initialize MLflow logger.

        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for the run
            tracking_uri: MLflow tracking server URI
            auto_log_system_info: Whether to log system information automatically
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.auto_log_system_info = auto_log_system_info
        self.run_id = None
        self._start_time = None

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
            else:
                raise ValueError(f"Could not create or find experiment: {experiment_name}")

    def __enter__(self) -> 'MLflowLogger':
        """Start MLflow run context."""
        self._start_time = datetime.now()
        self.run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=self.run_name
        )
        self.run_id = self.run.info.run_id

        logger.info(f"Started MLflow run: {self.run_id}")

        if self.auto_log_system_info:
            self._log_system_info()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run context."""
        if self._start_time:
            duration = (datetime.now() - self._start_time).total_seconds()
            self.log_metrics({"run_duration_seconds": duration})

        mlflow.end_run()
        logger.info(f"Ended MLflow run: {self.run_id}")

    def _log_system_info(self):
        """Log system and environment information."""
        try:
            # Git commit hash
            try:
                git_commit = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD']
                ).decode('ascii').strip()
                mlflow.set_tag("git_commit", git_commit)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            # Python version
            import sys
            mlflow.set_tag("python_version", sys.version)

            # Timestamp
            mlflow.set_tag("start_time", self._start_time.isoformat())

            # Environment
            mlflow.set_tag("environment", os.environ.get("ENVIRONMENT", "development"))

            logger.info("Logged system information")

        except Exception as e:
            logger.warning(f"Could not log system info: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log

        Example:
            >>> mlf.log_params({"max_depth": 6, "n_estimators": 100})
        """
        # Convert numpy types to native Python types
        converted_params = {}
        for key, value in params.items():
            if isinstance(value, np.integer):
                converted_params[key] = int(value)
            elif isinstance(value, np.floating):
                converted_params[key] = float(value)
            elif isinstance(value, np.ndarray):
                converted_params[key] = value.tolist()
            else:
                converted_params[key] = value

        mlflow.log_params(converted_params)
        logger.info(f"Logged {len(converted_params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for tracking metrics over time

        Example:
            >>> mlf.log_metrics({"auc": 0.85, "precision": 0.78})
        """
        # Convert numpy types
        converted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                converted_metrics[key] = float(value)
            else:
                converted_metrics[key] = value

        if step is not None:
            for key, value in converted_metrics.items():
                mlflow.log_metric(key, value, step)
        else:
            mlflow.log_metrics(converted_metrics)

        logger.info(f"Logged {len(converted_metrics)} metrics")

    def log_model(self, model: Any, artifact_path: str, 
                 model_type: str = "sklearn"):
        """Log model to MLflow.

        Args:
            model: Trained model object
            artifact_path: Path within the run's artifact URI
            model_type: Type of model ("sklearn", "xgboost", "lightgbm")

        Example:
            >>> mlf.log_model(xgb_model, "model", model_type="xgboost")
        """
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path)
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path)
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, artifact_path)
            else:
                # Fallback to pickle
                mlflow.sklearn.log_model(model, artifact_path)

            logger.info(f"Logged {model_type} model to {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_artifact(self, local_path: Union[str, Path], 
                    artifact_path: Optional[str] = None):
        """Log artifact file to MLflow.

        Args:
            local_path: Local file path
            artifact_path: Optional path within run's artifact directory

        Example:
            >>> mlf.log_artifact("plots/roc_curve.png", "plots")
        """
        mlflow.log_artifact(str(local_path), artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def log_artifacts(self, local_dir: Union[str, Path],
                     artifact_path: Optional[str] = None):
        """Log directory of artifacts to MLflow.

        Args:
            local_dir: Local directory path
            artifact_path: Optional path within run's artifact directory

        Example:
            >>> mlf.log_artifacts("output/plots/", "visualization")
        """
        mlflow.log_artifacts(str(local_dir), artifact_path)
        logger.info(f"Logged artifacts from: {local_dir}")

    def log_dataframe(self, df: pd.DataFrame, filename: str,
                     artifact_path: Optional[str] = None):
        """Log pandas DataFrame as artifact.

        Args:
            df: DataFrame to log
            filename: Filename for the artifact
            artifact_path: Optional path within run's artifact directory

        Example:
            >>> mlf.log_dataframe(results_df, "evaluation_results.csv")
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / filename

            if filename.endswith('.csv'):
                df.to_csv(file_path, index=False)
            elif filename.endswith('.parquet'):
                df.to_parquet(file_path, index=False)
            else:
                # Default to CSV
                df.to_csv(file_path, index=False)

            self.log_artifact(file_path, artifact_path)

    def log_dict(self, dictionary: Dict, filename: str,
                artifact_path: Optional[str] = None):
        """Log dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            filename: Filename for the artifact
            artifact_path: Optional path within run's artifact directory

        Example:
            >>> mlf.log_dict(model_config, "config.json")
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / filename

            with open(file_path, 'w') as f:
                json.dump(dictionary, f, indent=2, default=str)

            self.log_artifact(file_path, artifact_path)

    def log_text(self, text: str, filename: str,
                artifact_path: Optional[str] = None):
        """Log text as artifact.

        Args:
            text: Text content to log
            filename: Filename for the artifact
            artifact_path: Optional path within run's artifact directory

        Example:
            >>> mlf.log_text(model_summary, "model_summary.txt")
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / filename

            with open(file_path, 'w') as f:
                f.write(text)

            self.log_artifact(file_path, artifact_path)

    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the run.

        Args:
            tags: Dictionary of tags to set

        Example:
            >>> mlf.set_tags({"model_type": "xgboost", "version": "v1.0"})
        """
        for key, value in tags.items():
            mlflow.set_tag(key, str(value))

        logger.info(f"Set {len(tags)} tags")

    def get_run_info(self) -> Dict[str, Any]:
        """Get information about the current run.

        Returns:
            Dictionary with run information
        """
        if self.run_id:
            run = mlflow.get_run(self.run_id)
            return {
                "run_id": self.run_id,
                "experiment_id": self.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri
            }
        return {}


def start_mlflow_server(host: str = "localhost", 
                       port: int = 5000,
                       backend_store_uri: Optional[str] = None) -> None:
    """Start MLflow tracking server.

    Args:
        host: Host address
        port: Port number
        backend_store_uri: Backend store URI for MLflow

    Example:
        >>> start_mlflow_server(host="0.0.0.0", port=5000)
    """
    cmd = [
        "mlflow", "server",
        "--host", host,
        "--port", str(port)
    ]

    if backend_store_uri:
        cmd.extend(["--backend-store-uri", backend_store_uri])

    logger.info(f"Starting MLflow server on {host}:{port}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start MLflow server: {e}")
    except KeyboardInterrupt:
        logger.info("MLflow server stopped")


class ExperimentTracker:
    """High-level experiment tracking interface.

    Example:
        >>> tracker = ExperimentTracker("fraud_detection_v2")
        >>> with tracker.start_run("xgb_baseline") as run:
        ...     run.log_params(model_params)
        ...     run.log_metrics(evaluation_metrics)
        ...     run.log_model(trained_model)
    """

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def start_run(self, run_name: Optional[str] = None) -> MLflowLogger:
        """Start a new run.

        Args:
            run_name: Optional name for the run

        Returns:
            MLflow logger context manager
        """
        return MLflowLogger(
            experiment_name=self.experiment_name,
            run_name=run_name,
            tracking_uri=self.tracking_uri
        )

    def list_runs(self, max_results: int = 100) -> pd.DataFrame:
        """List runs from the experiment.

        Args:
            max_results: Maximum number of runs to return

        Returns:
            DataFrame with run information
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            logger.warning(f"Experiment '{self.experiment_name}' not found")
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results
        )

        return runs

    def get_best_run(self, metric_name: str, 
                    ascending: bool = False) -> Optional[mlflow.entities.Run]:
        """Get the best run based on a metric.

        Args:
            metric_name: Name of the metric to optimize
            ascending: Whether to sort in ascending order

        Returns:
            Best run object or None
        """
        runs_df = self.list_runs()

        if runs_df.empty:
            return None

        metric_col = f"metrics.{metric_name}"
        if metric_col not in runs_df.columns:
            logger.warning(f"Metric '{metric_name}' not found in runs")
            return None

        # Sort by metric
        sorted_runs = runs_df.sort_values(metric_col, ascending=ascending)
        best_run_id = sorted_runs.iloc[0]['run_id']

        return mlflow.get_run(best_run_id)
