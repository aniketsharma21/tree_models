# tests/performance/test_benchmarks.py
"""Performance benchmarks and stress tests for tree_models components."""

import pytest
import numpy as np
import pandas as pd
import time
from memory_profiler import memory_usage
from unittest.mock import Mock
import psutil
import os

from tree_models.data.validator import DataValidator, ValidationConfig
from tree_models.data.preprocessor import AdvancedDataPreprocessor, ColumnConfig
from tree_models.data.feature_engineer import FeatureEngineer, FeatureEngineeringConfig
from tree_models.models.evaluator import ModelEvaluator, EvaluationConfig


class TestPerformanceBenchmarks:
    """Performance benchmarks for core components."""

    @pytest.fixture(params=[(1000, 10), (10000, 50), (50000, 100)])
    def benchmark_datasets(self, request):
        """Create datasets of varying sizes for benchmarking."""
        n_samples, n_features = request.param
        np.random.seed(42)

        # Create mixed data types
        data = {}

        # Numeric features (70%)
        n_numeric = int(n_features * 0.7)
        for i in range(n_numeric):
            data[f"numeric_{i}"] = np.random.randn(n_samples)

        # Categorical features (20%)
        n_categorical = int(n_features * 0.2)
        for i in range(n_categorical):
            n_categories = np.random.randint(3, 20)
            categories = [f"cat_{j}" for j in range(n_categories)]
            data[f"categorical_{i}"] = np.random.choice(categories, n_samples)

        # Date features (10%)
        n_date = n_features - n_numeric - n_categorical
        for i in range(n_date):
            base_date = pd.Timestamp("2020-01-01")
            days_to_add = np.random.randint(0, 365, size=n_samples)
            data[f"date_{i}"] = base_date + pd.to_timedelta(days_to_add, unit="D")

        # Target variable
        data["target"] = np.random.binomial(1, 0.3, n_samples).astype(np.int64)

        # Introduce missing values (5-15%)
        df = pd.DataFrame(data)
        missing_rate = np.random.uniform(0.05, 0.15)
        n_missing = int(len(df) * missing_rate)

        for col in df.columns[:-1]:  # Exclude target
            missing_idx = np.random.choice(len(df), size=n_missing // len(df.columns), replace=False)
            df.loc[list(missing_idx), col] = np.nan

        return df, n_samples, n_features

    def test_data_validator_performance(self, benchmark_datasets):
        """Benchmark data validation performance."""
        df, n_samples, n_features = benchmark_datasets

        validator = DataValidator()
        config = ValidationConfig(
            check_missing_values=True,
            check_outliers=True,
            check_distributions=False,  # Skip expensive operations for large datasets
            generate_report=False,
        )

        # Measure time and memory
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        results = validator.validate_dataset(df, target_column="target", config=config)

        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        # Performance assertions
        # Time should scale reasonably with data size
        expected_max_time = (n_samples * n_features) / 10000  # Rough heuristic
        assert execution_time < max(expected_max_time, 30), f"Validation too slow: {execution_time:.2f}s"

        # Memory usage should be reasonable
        expected_max_memory = (n_samples * n_features * 8) / (1024 * 1024) * 3  # 3x data size
        # Allow a reasonable minimum threshold for Python process overhead on various platforms
        expected_max_memory = max(expected_max_memory, 4.0)
        assert memory_used < expected_max_memory, f"Memory usage too high: {memory_used:.2f}MB"

        # Log performance metrics
        print(f"\nData Validation Performance ({n_samples}x{n_features}):")
        print(f"  Time: {execution_time:.2f}s")
        print(f"  Memory: {memory_used:.2f}MB")
        print(f"  Quality Score: {results.data_quality_score:.2f}")

    def test_preprocessor_performance(self, benchmark_datasets):
        """Benchmark data preprocessing performance."""
        df, n_samples, n_features = benchmark_datasets

        # Configure preprocessing
        preprocessor = AdvancedDataPreprocessor()

        # Set reasonable configs for different column types
        for col in df.columns:
            if col == "target":
                continue
            elif df[col].dtype in ["object", "category"]:
                config = ColumnConfig(
                    missing_strategy="most_frequent", encoding_strategy="label"  # Faster than onehot for benchmarking
                )
            else:
                config = ColumnConfig(missing_strategy="median", scaling_strategy="standard")
            preprocessor.set_column_config(col, config)

        X = df.drop("target", axis=1)
        y = df["target"]

        # Measure performance
        start_time = time.time()

        X_processed = preprocessor.fit_transform(X, y)

        end_time = time.time()
        execution_time = end_time - start_time

        # Performance assertions
        expected_max_time = (n_samples * n_features) / 5000  # Rough heuristic
        assert execution_time < max(expected_max_time, 60), f"Preprocessing too slow: {execution_time:.2f}s"

        # Verify output integrity
        assert len(X_processed) == len(X)
        assert X_processed.shape[1] >= X.shape[1]  # May add features due to encoding
        assert X_processed.isnull().sum().sum() == 0  # No missing values

        print(f"\nPreprocessing Performance ({n_samples}x{n_features}):")
        print(f"  Time: {execution_time:.2f}s")
        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {X_processed.shape}")

    def test_feature_engineer_performance(self, benchmark_datasets):
        """Benchmark feature engineering performance."""
        df, n_samples, n_features = benchmark_datasets

        # Configure feature engineering (limited for performance)
        config = FeatureEngineeringConfig(
            log_transform_cols=[col for col in df.columns if "numeric" in col][:3],  # Limit to 3
            extract_date_features=[col for col in df.columns if "date" in col][:1],  # Limit to 1
            create_polynomial_features=False,  # Skip expensive operations
            create_interaction_features=False,
        )

        engineer = FeatureEngineer()

        # Measure performance
        start_time = time.time()

        results = engineer.engineer_features(df, config, target_column="target")

        end_time = time.time()
        execution_time = end_time - start_time

        # Performance assertions
        expected_max_time = (n_samples * n_features) / 2000
        assert execution_time < max(expected_max_time, 120), f"Feature engineering too slow: {execution_time:.2f}s"

        # Verify output
        assert results.n_features_after >= results.n_features_before
        assert len(results.new_features) > 0

        print(f"\nFeature Engineering Performance ({n_samples}x{n_features}):")
        print(f"  Time: {execution_time:.2f}s")
        print(f"  Features: {results.n_features_before} → {results.n_features_after}")
        print(f"  New features: {len(results.new_features)}")

    def test_evaluator_performance(self, benchmark_datasets):
        """Benchmark model evaluation performance."""
        df, n_samples, n_features = benchmark_datasets

        # Create mock model for consistent predictions
        mock_model = Mock()

        # Generate realistic predictions
        X = df.drop("target", axis=1).select_dtypes(include=[np.number])
        if X.empty:
            # If no numeric columns, create some
            X = pd.DataFrame(np.random.randn(n_samples, 5))

        y = df["target"]

        # Create predictions that somewhat correlate with target
        base_predictions = np.random.uniform(0.2, 0.8, len(y))
        predictions = np.where(
            y == 1, np.minimum(base_predictions + 0.3, 0.95), np.maximum(base_predictions - 0.3, 0.05)
        )

        mock_model.predict.return_value = (predictions > 0.5).astype(np.int64)
        mock_model.predict_proba.return_value = np.column_stack([1 - predictions, predictions])

        # Configure evaluation
        config = EvaluationConfig(
            compute_roc_auc=True,
            compute_pr_auc=True,
            compute_calibration=True,
            optimize_threshold=True,
            compute_confidence_intervals=False,  # Skip expensive bootstrap
            include_plots=False,
        )

        evaluator = ModelEvaluator()

        # Measure performance
        start_time = time.time()

        results = evaluator.evaluate_model(mock_model, X, y, config=config)

        end_time = time.time()
        execution_time = end_time - start_time

        # Performance assertions
        expected_max_time = n_samples / 1000  # Should be fast
        assert execution_time < max(expected_max_time, 30), f"Evaluation too slow: {execution_time:.2f}s"

        # Verify results
        assert "auc" in results.metrics
        assert results.optimal_threshold is not None

        print(f"\nEvaluation Performance ({n_samples}x{X.shape[1]}):")
        print(f"  Time: {execution_time:.2f}s")
        print(f"  AUC: {results.metrics.get('auc', 0):.3f}")
        print(f"  Optimal threshold: {results.optimal_threshold:.3f}")


class TestMemoryEfficiency:
    """Test memory efficiency for large datasets."""

    @pytest.mark.slow
    def test_large_dataset_memory_usage(self):
        """Test memory usage with large datasets."""
        # Create large dataset
        n_samples = 100000
        n_features = 100

        print(f"\nTesting memory efficiency with {n_samples:,} samples, {n_features} features")

        # Monitor memory before
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create data in chunks to avoid memory spike during creation
        chunk_size = 10000
        chunks = []

        for i in range(0, n_samples, chunk_size):
            current_chunk_size = min(chunk_size, n_samples - i)
            chunk_data = {f"feature_{j}": np.random.randn(current_chunk_size) for j in range(n_features)}
            chunk_data["target"] = np.random.randint(0, 2, current_chunk_size).astype(np.int64)
            chunks.append(pd.DataFrame(chunk_data))

        df = pd.concat(chunks, ignore_index=True)
        del chunks  # Free memory

        after_creation_memory = process.memory_info().rss / 1024 / 1024

        # Test validator memory efficiency
        validator = DataValidator()
        config = ValidationConfig(
            check_missing_values=True,
            check_outliers=False,  # Skip expensive operations
            check_distributions=False,
            generate_report=False,
        )

        results = validator.validate_dataset(df, target_column="target", config=config)

        after_validation_memory = process.memory_info().rss / 1024 / 1024

        # Memory usage should be reasonable
        data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        validation_overhead = after_validation_memory - after_creation_memory

        # Validation shouldn't use more than 2x the data size in additional memory
        assert (
            validation_overhead < data_size_mb * 2
        ), f"Validation memory overhead too high: {validation_overhead:.2f}MB"

        print(f"Memory Usage:")
        print(f"  Initial: {initial_memory:.2f}MB")
        print(f"  After data creation: {after_creation_memory:.2f}MB")
        print(f"  After validation: {after_validation_memory:.2f}MB")
        print(f"  Data size: {data_size_mb:.2f}MB")
        print(f"  Validation overhead: {validation_overhead:.2f}MB")


class TestScalabilityLimits:
    """Test scalability limits and edge cases."""

    def test_wide_dataset_performance(self):
        """Test performance with wide datasets (many features)."""
        n_samples = 1000
        n_features = 1000  # Wide dataset

        print(f"\nTesting wide dataset: {n_samples} samples, {n_features} features")

        # Create wide dataset
        data = {f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)}
        data["target"] = np.random.randint(0, 2, n_samples).astype(np.int64)
        df = pd.DataFrame(data)

        # Test validator
        validator = DataValidator()
        config = ValidationConfig(check_correlation=False, generate_report=False)  # Would be very expensive

        start_time = time.time()
        results = validator.validate_dataset(df, target_column="target", config=config)
        execution_time = time.time() - start_time

        # Should complete in reasonable time
        assert execution_time < 300, f"Wide dataset validation too slow: {execution_time:.2f}s"

        print(f"  Validation time: {execution_time:.2f}s")
        print(f"  Quality score: {results.data_quality_score:.2f}")

    def test_tall_dataset_performance(self):
        """Test performance with tall datasets (many samples)."""
        n_samples = 100000  # Tall dataset
        n_features = 20

        print(f"\nTesting tall dataset: {n_samples:,} samples, {n_features} features")

        # Create tall dataset
        data = {f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)}
        data["target"] = np.random.randint(0, 2, n_samples).astype(np.int64)
        df = pd.DataFrame(data)

        # Test preprocessor with chunking
        preprocessor = AdvancedDataPreprocessor(memory_efficient=True)

        X = df.drop("target", axis=1)
        y = df["target"]

        start_time = time.time()
        X_processed = preprocessor.fit_transform(X, y)
        execution_time = time.time() - start_time

        # Should complete in reasonable time
        assert execution_time < 180, f"Tall dataset preprocessing too slow: {execution_time:.2f}s"

        print(f"  Preprocessing time: {execution_time:.2f}s")
        print(f"  Output shape: {X_processed.shape}")


class TestResourceMonitoring:
    """Test resource usage monitoring."""

    def test_cpu_usage_monitoring(self):
        """Monitor CPU usage during operations."""
        # Create moderate dataset
        n_samples = 10000
        df = pd.DataFrame({f"feature_{i}": np.random.randn(n_samples) for i in range(20)})
        df["target"] = np.random.randint(0, 2, n_samples).astype(np.int64)

        # Monitor CPU usage
        process = psutil.Process(os.getpid())

        # Baseline CPU
        time.sleep(1)  # Let system stabilize
        baseline_cpu = process.cpu_percent(interval=1)

        # Test operation
        validator = DataValidator()
        start_time = time.time()

        results = validator.validate_dataset(df, target_column="target")

        execution_time = time.time() - start_time
        final_cpu = process.cpu_percent(interval=None)

        print(f"\nCPU Usage Monitoring:")
        print(f"  Baseline CPU: {baseline_cpu:.2f}%")
        print(f"  Peak CPU during operation: {final_cpu:.2f}%")
        print(f"  Execution time: {execution_time:.2f}s")

        # CPU usage should be reasonable (not maxing out single core continuously)
        # This is informational rather than a hard test

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss

        # Perform repeated operations
        for i in range(10):
            # Create small dataset
            df = pd.DataFrame({f"feature_{j}": np.random.randn(1000) for j in range(10)})
            df["target"] = np.random.randint(0, 2, 1000).astype(np.int64)

            # Validate
            validator = DataValidator()
            results = validator.validate_dataset(df, target_column="target")

            # Clear references
            del df, validator, results

        final_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB

        print(f"\nMemory Leak Detection:")
        print(f"  Initial memory: {initial_memory / 1024 / 1024:.2f}MB")
        print(f"  Final memory: {final_memory / 1024 / 1024:.2f}MB")
        print(f"  Memory growth: {memory_growth:.2f}MB")

        # Memory growth should be minimal (< 50MB for 10 iterations)
        assert memory_growth < 50, f"Possible memory leak detected: {memory_growth:.2f}MB growth"


if __name__ == "__main__":
    # Run with verbose output and timing
    pytest.main([__file__, "-v", "-s", "--tb=short"])
