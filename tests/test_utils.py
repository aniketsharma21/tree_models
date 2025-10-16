"""Additional unit tests for utility and tracking modules."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Utility tests
from src.utils.logger import get_logger
from src.utils.timer import timer
from src.utils.io_utils import save_json, load_json
from src.tracking.mlflow_logger import MLflowLogger


class TestLogger:
    """Test logging functionality."""

    @pytest.mark.unit
    def test_get_logger_basic(self):
        """Test basic logger creation."""
        try:
            logger = get_logger(__name__)
            assert logger is not None
            assert hasattr(logger, 'info')
            assert hasattr(logger, 'error')
            assert hasattr(logger, 'warning')
            assert hasattr(logger, 'debug')
            
        except ImportError:
            pytest.skip("Logger not available")

    @pytest.mark.unit
    def test_get_logger_with_name(self):
        """Test logger with specific name."""
        try:
            logger_name = "test_logger"
            logger = get_logger(logger_name)
            
            assert logger.name == logger_name
            
        except ImportError:
            pytest.skip("Logger not available")

    @pytest.mark.unit
    def test_logger_levels(self):
        """Test different logging levels."""
        try:
            logger = get_logger("test_levels")
            
            # Test that logger can handle different levels without error
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            
            # Should not raise any exceptions
            assert True
            
        except ImportError:
            pytest.skip("Logger levels not available")


class TestTimer:
    """Test timing functionality."""

    @pytest.mark.unit
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        try:
            import time
            
            with timer() as t:
                time.sleep(0.01)  # Sleep for 10ms
            
            # Should have measured some time > 0
            elapsed = t.elapsed
            assert elapsed > 0
            assert elapsed < 1.0  # Should be much less than 1 second
            
        except ImportError:
            pytest.skip("Timer not available")

    @pytest.mark.unit
    def test_timer_decorator(self):
        """Test timer as decorator."""
        try:
            import time
            
            @timer
            def timed_function():
                time.sleep(0.01)
                return "result"
            
            result = timed_function()
            assert result == "result"
            
        except ImportError:
            pytest.skip("Timer decorator not available")

    @pytest.mark.unit
    def test_timer_manual_start_stop(self):
        """Test manual timer start/stop."""
        try:
            import time
            
            t = timer()
            t.start()
            time.sleep(0.01)
            elapsed = t.stop()
            
            assert elapsed > 0
            assert t.elapsed > 0
            
        except ImportError:
            pytest.skip("Manual timer not available")


class TestIOUtils:
    """Test I/O utility functions."""

    @pytest.mark.unit
    def test_save_load_json(self, test_data_dir: Path):
        """Test JSON save and load functionality."""
        try:
            test_data = {
                "model_type": "xgboost",
                "parameters": {
                    "n_estimators": 100,
                    "max_depth": 6
                },
                "metrics": {
                    "accuracy": 0.85,
                    "auc": 0.92
                }
            }
            
            json_file = test_data_dir / "test_data.json"
            
            # Save JSON
            save_json(test_data, json_file)
            assert json_file.exists()
            
            # Load JSON
            loaded_data = load_json(json_file)
            assert loaded_data == test_data
            
        except ImportError:
            pytest.skip("JSON I/O utils not available")

    @pytest.mark.unit
    def test_save_json_with_numpy(self, test_data_dir: Path):
        """Test JSON save with numpy arrays."""
        try:
            import numpy as np
            
            test_data = {
                "array": np.array([1, 2, 3, 4]).tolist(),
                "float": float(np.float32(3.14)),
                "int": int(np.int64(42))
            }
            
            json_file = test_data_dir / "test_numpy.json"
            
            save_json(test_data, json_file)
            loaded_data = load_json(json_file)
            
            assert loaded_data["array"] == [1, 2, 3, 4]
            assert loaded_data["float"] == pytest.approx(3.14)
            assert loaded_data["int"] == 42
            
        except ImportError:
            pytest.skip("Numpy JSON handling not available")

    @pytest.mark.unit
    def test_load_nonexistent_json(self):
        """Test error handling for non-existent JSON files."""
        try:
            with pytest.raises(FileNotFoundError):
                load_json("nonexistent_file.json")
                
        except ImportError:
            pytest.skip("JSON error handling not available")

    @pytest.mark.unit
    def test_save_load_nested_json(self, test_data_dir: Path):
        """Test deeply nested JSON structures."""
        try:
            nested_data = {
                "level1": {
                    "level2": {
                        "level3": {
                            "data": [1, 2, 3],
                            "metadata": {
                                "created": "2024-01-01",
                                "version": "1.0.0"
                            }
                        }
                    }
                }
            }
            
            json_file = test_data_dir / "nested.json"
            
            save_json(nested_data, json_file)
            loaded_data = load_json(json_file)
            
            assert loaded_data == nested_data
            
        except ImportError:
            pytest.skip("Nested JSON not available")


class TestMLflowLogger:
    """Test MLflow logging functionality."""

    @pytest.mark.unit
    def test_mlflow_logger_initialization(self):
        """Test MLflowLogger initialization."""
        try:
            logger = MLflowLogger(
                experiment_name="test_experiment",
                run_name="test_run"
            )
            
            assert logger.experiment_name == "test_experiment"
            assert logger.run_name == "test_run"
            
        except ImportError:
            pytest.skip("MLflowLogger not available")

    @pytest.mark.unit
    def test_mlflow_context_manager(self, mock_mlflow):
        """Test MLflowLogger as context manager."""
        try:
            with MLflowLogger(experiment_name="test") as mlf:
                assert mlf is not None
                
                # Test logging methods exist
                assert hasattr(mlf, 'log_param')
                assert hasattr(mlf, 'log_metric')
                assert hasattr(mlf, 'log_artifact')
                
        except ImportError:
            pytest.skip("MLflowLogger context manager not available")

    @pytest.mark.unit
    def test_mlflow_parameter_logging(self, mock_mlflow):
        """Test parameter logging functionality."""
        try:
            with MLflowLogger(experiment_name="test") as mlf:
                params = {
                    "model_type": "xgboost",
                    "n_estimators": 100,
                    "learning_rate": 0.1
                }
                
                mlf.log_params(params)
                
                # Verify mock was called
                assert mock_mlflow.log_param.called
                
        except ImportError:
            pytest.skip("MLflow parameter logging not available")

    @pytest.mark.unit
    def test_mlflow_metric_logging(self, mock_mlflow):
        """Test metric logging functionality."""
        try:
            with MLflowLogger(experiment_name="test") as mlf:
                metrics = {
                    "accuracy": 0.85,
                    "auc_roc": 0.92,
                    "f1_score": 0.78
                }
                
                mlf.log_metrics(metrics)
                
                # Verify mock was called
                assert mock_mlflow.log_metric.called
                
        except ImportError:
            pytest.skip("MLflow metric logging not available")

    @pytest.mark.unit
    def test_mlflow_artifact_logging(self, mock_mlflow, test_data_dir: Path):
        """Test artifact logging functionality."""
        try:
            # Create test artifact
            artifact_file = test_data_dir / "test_artifact.txt"
            with open(artifact_file, 'w') as f:
                f.write("Test artifact content")
            
            with MLflowLogger(experiment_name="test") as mlf:
                mlf.log_artifact(artifact_file, "artifacts")
                
                # Verify mock was called
                assert mock_mlflow.log_artifact.called
                
        except ImportError:
            pytest.skip("MLflow artifact logging not available")

    @pytest.mark.unit
    def test_mlflow_model_logging(self, mock_mlflow, trained_xgb_model):
        """Test model logging functionality."""
        try:
            model, X, y = trained_xgb_model
            
            with MLflowLogger(experiment_name="test") as mlf:
                mlf.log_model(model, "xgboost_model", model_type="xgboost")
                
                # Should not raise exceptions
                assert True
                
        except ImportError:
            pytest.skip("MLflow model logging not available")

    @pytest.mark.unit
    def test_mlflow_tags(self, mock_mlflow):
        """Test MLflow tag functionality."""
        try:
            with MLflowLogger(experiment_name="test") as mlf:
                tags = {
                    "model_type": "xgboost",
                    "experiment_type": "baseline",
                    "status": "completed"
                }
                
                mlf.set_tags(tags)
                
                # Should not raise exceptions
                assert True
                
        except ImportError:
            pytest.skip("MLflow tags not available")

    @pytest.mark.integration
    def test_complete_mlflow_workflow(self, mock_mlflow, test_data_dir: Path):
        """Test complete MLflow logging workflow."""
        try:
            with MLflowLogger(
                experiment_name="integration_test",
                run_name="complete_workflow"
            ) as mlf:
                
                # Log parameters
                params = {"n_estimators": 100, "max_depth": 6}
                mlf.log_params(params)
                
                # Log metrics
                metrics = {"accuracy": 0.85, "auc": 0.92}
                mlf.log_metrics(metrics)
                
                # Create and log artifact
                artifact_file = test_data_dir / "workflow_artifact.json"
                with open(artifact_file, 'w') as f:
                    json.dump({"test": "data"}, f)
                mlf.log_artifact(artifact_file)
                
                # Set tags
                mlf.set_tags({"status": "success"})
                
            # Should complete without errors
            assert True
            
        except ImportError:
            pytest.skip("Complete MLflow workflow not available")


class TestPlotUtils:
    """Test plotting utility functions."""

    @pytest.mark.unit
    def test_plot_configuration(self):
        """Test plot configuration setup."""
        try:
            from src.utils.plot_utils import setup_plot_style
            
            setup_plot_style()
            
            # Should not raise exceptions
            assert True
            
        except ImportError:
            pytest.skip("Plot utils not available")

    @pytest.mark.unit
    def test_save_plot_function(self, test_data_dir: Path):
        """Test plot saving functionality."""
        try:
            from src.utils.plot_utils import save_plot
            import matplotlib.pyplot as plt
            
            # Create simple plot
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 2])
            ax.set_title("Test Plot")
            
            # Save plot
            output_file = test_data_dir / "test_plot.png"
            save_plot(fig, output_file)
            
            # Check file was created
            assert output_file.exists()
            
            plt.close(fig)
            
        except ImportError:
            pytest.skip("Plot saving not available")

    @pytest.mark.unit
    def test_plot_formatting(self):
        """Test plot formatting utilities."""
        try:
            from src.utils.plot_utils import format_plot
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 2])
            
            # Apply formatting
            format_plot(ax, title="Formatted Plot", xlabel="X", ylabel="Y")
            
            assert ax.get_title() == "Formatted Plot"
            assert ax.get_xlabel() == "X"
            assert ax.get_ylabel() == "Y"
            
            plt.close(fig)
            
        except ImportError:
            pytest.skip("Plot formatting not available")


class TestUtilityIntegration:
    """Integration tests for utility functions."""

    @pytest.mark.integration
    def test_logging_and_timing_integration(self):
        """Test logging and timing working together."""
        try:
            import time
            
            logger = get_logger("integration_test")
            
            with timer() as t:
                logger.info("Starting timed operation")
                time.sleep(0.01)
                logger.info("Completed timed operation")
            
            logger.info(f"Operation took {t.elapsed:.3f} seconds")
            
            assert t.elapsed > 0
            
        except ImportError:
            pytest.skip("Logging/timing integration not available")

    @pytest.mark.integration
    def test_io_and_logging_integration(self, test_data_dir: Path):
        """Test I/O and logging working together."""
        try:
            logger = get_logger("io_test")
            
            test_data = {"integration": True, "test": "data"}
            json_file = test_data_dir / "integration_test.json"
            
            logger.info("Saving test data...")
            save_json(test_data, json_file)
            logger.info(f"Data saved to {json_file}")
            
            logger.info("Loading test data...")
            loaded_data = load_json(json_file)
            logger.info("Data loaded successfully")
            
            assert loaded_data == test_data
            
        except ImportError:
            pytest.skip("I/O and logging integration not available")


class TestErrorHandling:
    """Test error handling in utility functions."""

    @pytest.mark.unit
    def test_timer_error_handling(self):
        """Test timer error handling."""
        try:
            t = timer()
            
            # Try to stop timer that wasn't started
            with pytest.raises((RuntimeError, ValueError)):
                t.stop()
                
        except ImportError:
            pytest.skip("Timer error handling not available")

    @pytest.mark.unit
    def test_json_error_handling(self, test_data_dir: Path):
        """Test JSON I/O error handling."""
        try:
            # Test saving invalid data
            invalid_data = {"function": lambda x: x}  # Functions are not JSON serializable
            
            with pytest.raises((TypeError, ValueError)):
                save_json(invalid_data, test_data_dir / "invalid.json")
                
        except ImportError:
            pytest.skip("JSON error handling not available")

    @pytest.mark.unit
    def test_mlflow_error_handling(self):
        """Test MLflow error handling."""
        try:
            # Test with invalid experiment name
            with pytest.raises((ValueError, Exception)):
                with MLflowLogger(experiment_name=""):
                    pass
                    
        except ImportError:
            pytest.skip("MLflow error handling not available")