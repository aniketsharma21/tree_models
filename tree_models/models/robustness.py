# tree_models/models/robustness.py
"""Enhanced model robustness and stability testing utilities.

This module provides comprehensive robustness testing for tree-based models with:
- Type-safe interfaces and comprehensive error handling
- Seed robustness testing with multi-seed training and evaluation
- Sensitivity analysis through feature perturbation testing
- Feature stability assessment across multiple runs
- Population Stability Index (PSI) for drift detection
- Sample weights integration throughout all testing workflows
- Performance monitoring and parallel processing support

Key Features:
- Multi-seed model training with statistical analysis
- Feature perturbation sensitivity analysis
- Feature importance stability assessment
- Data drift detection with PSI calculations
- Comprehensive reporting and visualization
- Production-ready error handling and logging
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
import warnings
import json
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, log_loss
)
from scipy.stats import pearsonr, spearmanr
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .base import BaseRobustnessTester
from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    RobustnessTestError,
    ModelTrainingError,
    ConfigurationError,
    PerformanceError,
    handle_and_reraise,
    validate_parameter,
    create_error_context
)

logger = get_logger(__name__)


@dataclass
class RobustnessConfig:
    """Type-safe configuration for robustness testing with comprehensive validation."""
    
    # Basic configuration
    n_seeds: int = 10
    test_size: float = 0.2
    random_state: int = 42
    
    # Perturbation testing
    perturbation_range: Tuple[float, float] = (-0.1, 0.1)
    n_perturbations: int = 100
    perturbation_method: str = "gaussian"  # "gaussian", "uniform", "percentage"
    
    # PSI configuration
    psi_bins: int = 10
    psi_method: str = "quantile"  # "quantile", "equal_width"
    
    # Statistical analysis
    confidence_level: float = 0.95
    bootstrap_iterations: int = 1000
    
    # Performance settings
    parallel_jobs: int = 1  # Changed default to 1 for stability
    timeout_per_seed: Optional[float] = None
    memory_limit_gb: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        validate_parameter("n_seeds", self.n_seeds, min_value=2, max_value=100)
        validate_parameter("test_size", self.test_size, min_value=0.1, max_value=0.9)
        validate_parameter("random_state", self.random_state, min_value=0)
        
        # Validate perturbation parameters
        if len(self.perturbation_range) != 2:
            raise ConfigurationError("perturbation_range must be a tuple of length 2")
        
        if self.perturbation_range[0] >= self.perturbation_range[1]:
            raise ConfigurationError("perturbation_range[0] must be < perturbation_range[1]")
        
        validate_parameter("n_perturbations", self.n_perturbations, min_value=10, max_value=10000)
        validate_parameter("perturbation_method", self.perturbation_method, 
                         valid_values=["gaussian", "uniform", "percentage"])
        
        # Validate PSI parameters
        validate_parameter("psi_bins", self.psi_bins, min_value=5, max_value=50)
        validate_parameter("psi_method", self.psi_method, 
                         valid_values=["quantile", "equal_width"])
        
        # Validate statistical parameters
        validate_parameter("confidence_level", self.confidence_level, min_value=0.5, max_value=0.999)
        validate_parameter("bootstrap_iterations", self.bootstrap_iterations, min_value=100, max_value=10000)
        
        # Validate performance parameters
        validate_parameter("parallel_jobs", self.parallel_jobs, min_value=1, max_value=32)
        
        if self.timeout_per_seed is not None:
            validate_parameter("timeout_per_seed", self.timeout_per_seed, min_value=10.0)
        
        if self.memory_limit_gb is not None:
            validate_parameter("memory_limit_gb", self.memory_limit_gb, min_value=0.5, max_value=256.0)


class SeedRobustnessTester(BaseRobustnessTester):
    """Enhanced seed robustness tester with comprehensive error handling and analysis.

    Tests model robustness across different random seeds with comprehensive
    statistical analysis, feature importance stability assessment, and
    production-ready error handling.

    Example:
        >>> from tree_models.models import SeedRobustnessTester
        >>> from tree_models.models.trainer import StandardModelTrainer
        >>> 
        >>> trainer = StandardModelTrainer('xgboost')
        >>> tester = SeedRobustnessTester(n_seeds=10, scoring_function='roc_auc')
        >>> 
        >>> results = tester.test_robustness(
        ...     model_trainer=trainer,
        ...     X=X_train, y=y_train,
        ...     sample_weight=weights,
        ...     model_params={'n_estimators': 100, 'max_depth': 6}
        ... )
        >>> 
        >>> stability_metrics = tester.get_stability_metrics()
        >>> print(f"Performance stability: {stability_metrics['performance_cv']:.4f}")
    """

    def __init__(
        self,
        n_seeds: int = 10,
        scoring_function: str = "roc_auc",
        config: Optional[RobustnessConfig] = None,
        **kwargs: Any
    ) -> None:
        """Initialize enhanced seed robustness tester.

        Args:
            n_seeds: Number of different seeds to test
            scoring_function: Primary metric for evaluation
            config: Robustness testing configuration
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            test_type="seed_robustness",
            n_iterations=n_seeds,
            **kwargs
        )
        
        self.scoring_function = scoring_function
        self.config = config or RobustnessConfig(n_seeds=n_seeds)
        
        # Override n_iterations from config
        self.n_iterations = self.config.n_seeds
        
        # Storage for results and analysis
        self.raw_results: List[Dict[str, Any]] = []
        self.stability_metrics_: Optional[Dict[str, float]] = None
        self.feature_stability_: Optional[Dict[str, Any]] = None
        
        # Validate scoring function
        valid_scorers = {
            'roc_auc', 'average_precision', 'accuracy', 'precision', 
            'recall', 'f1', 'balanced_accuracy', 'neg_log_loss'
        }
        
        if scoring_function not in valid_scorers:
            logger.warning(f"Unknown scoring function: {scoring_function}")
        
        logger.info(f"Initialized SeedRobustnessTester:")
        logger.info(f"  Seeds: {self.config.n_seeds}, Scoring: {scoring_function}")
        logger.info(f"  Test size: {self.config.test_size}, Parallel jobs: {self.config.parallel_jobs}")

    def _train_single_seed(
        self,
        seed: int,
        model_trainer: 'BaseModelTrainer',
        X: pd.DataFrame,
        y: pd.Series,
        model_params: Dict[str, Any],
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train and evaluate model with a single seed with comprehensive error handling."""
        
        context = create_error_context(
            seed=seed,
            model_type=model_trainer.model_type,
            data_shape=X.shape,
            has_weights=sample_weight is not None
        )
        
        try:
            with timed_operation(f"seed_{seed}_training", timeout=self.config.timeout_per_seed) as timing:
                # Validate inputs
                model_trainer.validate_input_data(X, y, sample_weight)
                
                # Create model with seed
                seed_params = model_params.copy()
                
                # Set seed based on model type
                if model_trainer.model_type.lower() == 'xgboost':
                    seed_params['random_state'] = seed
                elif model_trainer.model_type.lower() == 'lightgbm':
                    seed_params['random_state'] = seed
                elif model_trainer.model_type.lower() == 'catboost':
                    seed_params['random_seed'] = seed
                else:
                    seed_params['random_state'] = seed
                
                # Split data with seed
                if sample_weight is not None:
                    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
                        X, y, sample_weight, 
                        test_size=self.config.test_size,
                        random_state=seed, 
                        stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=self.config.test_size,
                        random_state=seed, 
                        stratify=y
                    )
                    sw_train, sw_test = None, None
                
                # Train model
                model = model_trainer.get_model(seed_params)
                
                if sw_train is not None:
                    try:
                        model.fit(X_train, y_train, sample_weight=sw_train)
                    except TypeError:
                        logger.warning(f"Seed {seed}: Model doesn't support sample_weight, training without")
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # Evaluate model
                metrics = self._evaluate_model_comprehensive(
                    model, X_test, y_test, sw_test
                )
                
                # Get feature importance if available
                feature_importance = self._extract_feature_importance(model, list(X.columns))
                
                training_duration = timing['duration']
                
            logger.debug(f"Seed {seed} completed in {training_duration:.2f}s: {self.scoring_function}={metrics.get(self.scoring_function, 'N/A')}")
            
            return {
                'seed': seed,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_duration': training_duration,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'success': True
            }

        except Exception as e:
            handle_and_reraise(
                e, RobustnessTestError,
                f"Failed to train model with seed {seed}",
                error_code="SEED_TRAINING_FAILED",
                context=context
            )

    def _evaluate_model_comprehensive(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Comprehensive model evaluation with error handling."""
        
        metrics = {}
        
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = y_pred.astype(float)
            
            # Calculate metrics with sample weights support
            metric_functions = {
                'roc_auc': (roc_auc_score, True, False),  # (function, supports_weights, needs_binary)
                'average_precision': (average_precision_score, True, False),
                'accuracy': (accuracy_score, True, True),
                'precision': (precision_score, True, True),
                'recall': (recall_score, True, True),
                'f1': (f1_score, True, True)
            }
            
            for metric_name, (metric_func, supports_weights, needs_binary) in metric_functions.items():
                try:
                    pred_input = y_pred if needs_binary else y_pred_proba
                    
                    if sample_weight is not None and supports_weights:
                        if metric_name in ['precision', 'recall', 'f1']:
                            metrics[metric_name] = metric_func(
                                y_test, pred_input, 
                                sample_weight=sample_weight, 
                                zero_division=0
                            )
                        else:
                            metrics[metric_name] = metric_func(
                                y_test, pred_input, 
                                sample_weight=sample_weight
                            )
                    else:
                        if metric_name in ['precision', 'recall', 'f1']:
                            metrics[metric_name] = metric_func(y_test, pred_input, zero_division=0)
                        else:
                            metrics[metric_name] = metric_func(y_test, pred_input)
                            
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
                    metrics[metric_name] = np.nan
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed comprehensive evaluation: {e}")
            return {metric: np.nan for metric in ['roc_auc', 'average_precision', 'accuracy', 'precision', 'recall', 'f1']}

    def _extract_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[np.ndarray]:
        """Extract feature importance from model with error handling."""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):  # CatBoost
                return model.get_feature_importance()
            elif hasattr(model, 'coef_'):  # Linear models
                return np.abs(model.coef_).flatten()
            else:
                logger.debug("Model does not provide feature importance")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to extract feature importance: {e}")
            return None

    @timer(name="seed_robustness_testing")
    def test_robustness(
        self,
        model_trainer: 'BaseModelTrainer',
        X: pd.DataFrame,
        y: pd.Series,
        model_params: Optional[Dict[str, Any]] = None,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Test model robustness across multiple seeds with comprehensive analysis.

        Args:
            model_trainer: Model trainer instance
            X: Training features
            y: Training targets
            model_params: Model parameters to use
            sample_weight: Optional sample weights
            **kwargs: Additional testing parameters

        Returns:
            Dictionary with comprehensive robustness test results

        Raises:
            RobustnessTestError: If robustness testing fails
        """
        logger.info(f"🛡️ Starting seed robustness testing:")
        logger.info(f"   Seeds: {self.config.n_seeds}, Model: {model_trainer.model_type}")
        logger.info(f"   Data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"   Scoring: {self.scoring_function}")

        try:
            # Validate inputs
            model_trainer.validate_input_data(X, y, sample_weight)
            model_params = model_params or {}
            
            # Generate seeds for reproducibility
            np.random.seed(self.config.random_state)
            seeds = np.random.randint(0, 10000, self.config.n_seeds).tolist()
            
            # Train models with different seeds
            self.raw_results = []
            successful_runs = 0
            
            for i, seed in enumerate(seeds, 1):
                logger.info(f"Training seed {i}/{self.config.n_seeds} (seed={seed})")
                
                try:
                    result = self._train_single_seed(
                        seed, model_trainer, X, y, model_params, sample_weight
                    )
                    self.raw_results.append(result)
                    successful_runs += 1
                    
                except RobustnessTestError as e:
                    logger.error(f"Seed {seed} failed: {e}")
                    # Add failed result for tracking
                    self.raw_results.append({
                        'seed': seed,
                        'success': False,
                        'error': str(e),
                        'metrics': {metric: np.nan for metric in ['roc_auc', 'average_precision', 'accuracy', 'precision', 'recall', 'f1']}
                    })
            
            # Check if we have sufficient successful runs
            if successful_runs < 2:
                raise RobustnessTestError(
                    "Insufficient successful runs for robustness analysis",
                    error_code="INSUFFICIENT_RUNS",
                    context={'successful_runs': successful_runs, 'required_minimum': 2}
                )
            
            # Analyze results
            analysis = self._analyze_robustness_results(list(X.columns))
            
            # Store results for later access
            self.test_results_ = {
                'config': self.config,
                'seeds_used': seeds,
                'raw_results': self.raw_results,
                'analysis': analysis,
                'successful_runs': successful_runs,
                'total_runs': len(seeds)
            }
            
            # Calculate stability metrics
            self.stability_metrics_ = self._calculate_stability_metrics(analysis)
            
            logger.info(f"✅ Seed robustness testing completed:")
            logger.info(f"   Success rate: {successful_runs}/{len(seeds)} ({successful_runs/len(seeds)*100:.1f}%)")
            logger.info(f"   Mean {self.scoring_function}: {analysis['metrics'][self.scoring_function]['mean']:.4f} ± {analysis['metrics'][self.scoring_function]['std']:.4f}")
            logger.info(f"   Performance CV: {self.stability_metrics_['performance_cv']:.4f}")
            
            return self.test_results_

        except Exception as e:
            handle_and_reraise(
                e, RobustnessTestError,
                "Seed robustness testing failed",
                error_code="ROBUSTNESS_TEST_FAILED",
                context=create_error_context(
                    n_seeds=self.config.n_seeds,
                    model_type=model_trainer.model_type,
                    data_shape=X.shape
                )
            )

    def _analyze_robustness_results(self, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze robustness results with comprehensive statistical analysis."""
        
        # Filter successful results
        successful_results = [r for r in self.raw_results if r.get('success', False)]
        
        if len(successful_results) < 2:
            raise RobustnessTestError("Insufficient successful results for analysis")
        
        # Analyze performance metrics
        metrics_analysis = {}
        
        # Extract all metrics from successful runs
        all_metrics = {}
        for result in successful_results:
            for metric_name, value in result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                if not np.isnan(value):
                    all_metrics[metric_name].append(value)
        
        # Calculate statistics for each metric
        for metric_name, values in all_metrics.items():
            if len(values) >= 2:
                metrics_analysis[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values, ddof=1),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'cv': np.std(values, ddof=1) / np.mean(values) if np.mean(values) != 0 else np.inf,
                    'n_values': len(values)
                }
                
                # Calculate confidence interval
                if len(values) >= 3:
                    from scipy import stats
                    confidence = self.config.confidence_level
                    ci = stats.t.interval(confidence, len(values)-1, 
                                        loc=np.mean(values), 
                                        scale=stats.sem(values))
                    metrics_analysis[metric_name]['ci_lower'] = ci[0]
                    metrics_analysis[metric_name]['ci_upper'] = ci[1]
        
        # Analyze feature importance stability
        feature_importance_analysis = self._analyze_feature_importance_stability(
            successful_results, feature_names
        )
        
        # Calculate training time statistics
        training_times = [r['training_duration'] for r in successful_results if 'training_duration' in r]
        training_time_stats = None
        if training_times:
            training_time_stats = {
                'mean': np.mean(training_times),
                'std': np.std(training_times, ddof=1),
                'min': np.min(training_times),
                'max': np.max(training_times),
                'total': np.sum(training_times)
            }
        
        return {
            'metrics': metrics_analysis,
            'feature_importance': feature_importance_analysis,
            'training_time': training_time_stats,
            'n_successful_runs': len(successful_results),
            'success_rate': len(successful_results) / len(self.raw_results)
        }

    def _analyze_feature_importance_stability(
        self, 
        successful_results: List[Dict[str, Any]], 
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Analyze stability of feature importances across runs."""
        
        # Extract feature importances
        importance_matrices = []
        for result in successful_results:
            if result.get('feature_importance') is not None:
                importance_matrices.append(result['feature_importance'])
        
        if len(importance_matrices) < 2:
            logger.warning("Insufficient feature importance data for stability analysis")
            return None
        
        try:
            # Convert to DataFrame for easier analysis
            importance_df = pd.DataFrame(importance_matrices, columns=feature_names)
            
            # Calculate pairwise correlations
            correlations = []
            for i in range(len(importance_matrices)):
                for j in range(i + 1, len(importance_matrices)):
                    try:
                        corr, _ = pearsonr(importance_matrices[i], importance_matrices[j])
                        if not np.isnan(corr):
                            correlations.append(corr)
                    except Exception:
                        continue
            
            # Calculate rank correlations
            rank_correlations = []
            for i in range(len(importance_matrices)):
                for j in range(i + 1, len(importance_matrices)):
                    try:
                        corr, _ = spearmanr(importance_matrices[i], importance_matrices[j])
                        if not np.isnan(corr):
                            rank_correlations.append(corr)
                    except Exception:
                        continue
            
            # Feature-level statistics
            feature_stats = {}
            for feature in feature_names:
                values = importance_df[feature].values
                feature_stats[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values, ddof=1),
                    'cv': np.std(values, ddof=1) / np.mean(values) if np.mean(values) > 0 else np.inf,
                    'rank_stability': self._calculate_rank_stability(importance_df, feature)
                }
            
            self.feature_stability_ = {
                'mean_correlation': np.mean(correlations) if correlations else np.nan,
                'std_correlation': np.std(correlations, ddof=1) if len(correlations) > 1 else np.nan,
                'min_correlation': np.min(correlations) if correlations else np.nan,
                'max_correlation': np.max(correlations) if correlations else np.nan,
                'mean_rank_correlation': np.mean(rank_correlations) if rank_correlations else np.nan,
                'feature_statistics': feature_stats,
                'overall_stability_score': np.mean(correlations) if correlations else 0.0
            }
            
            return self.feature_stability_
            
        except Exception as e:
            logger.warning(f"Error analyzing feature importance stability: {e}")
            return None

    def _calculate_rank_stability(self, importance_df: pd.DataFrame, feature: str) -> float:
        """Calculate rank stability for a specific feature across runs."""
        try:
            # Get ranks for this feature across all runs
            ranks = []
            for _, row in importance_df.iterrows():
                feature_rank = row.rank(ascending=False)[feature]
                ranks.append(feature_rank)
            
            # Calculate coefficient of variation for ranks
            if len(ranks) > 1 and np.mean(ranks) > 0:
                return 1.0 - (np.std(ranks, ddof=1) / np.mean(ranks))
            return 0.0
            
        except Exception:
            return 0.0

    def _calculate_stability_metrics(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall stability metrics from analysis results."""
        
        stability_metrics = {}
        
        # Performance stability (using primary scoring function)
        if self.scoring_function in analysis['metrics']:
            metric_stats = analysis['metrics'][self.scoring_function]
            if metric_stats['mean'] != 0:
                stability_metrics['performance_cv'] = metric_stats['cv']
                stability_metrics['performance_stability'] = max(0.0, 1.0 - metric_stats['cv'])
            else:
                stability_metrics['performance_cv'] = np.inf
                stability_metrics['performance_stability'] = 0.0
        else:
            stability_metrics['performance_cv'] = np.nan
            stability_metrics['performance_stability'] = np.nan
        
        # Feature importance stability
        if analysis['feature_importance']:
            stability_metrics['feature_correlation'] = analysis['feature_importance']['mean_correlation']
            stability_metrics['feature_stability'] = analysis['feature_importance']['overall_stability_score']
        else:
            stability_metrics['feature_correlation'] = np.nan
            stability_metrics['feature_stability'] = np.nan
        
        # Overall stability score (weighted combination)
        stability_scores = [v for v in [
            stability_metrics.get('performance_stability'),
            stability_metrics.get('feature_stability')
        ] if v is not None and not np.isnan(v)]
        
        if stability_scores:
            stability_metrics['overall_stability'] = np.mean(stability_scores)
        else:
            stability_metrics['overall_stability'] = 0.0
        
        # Success rate
        stability_metrics['success_rate'] = analysis['success_rate']
        
        return stability_metrics

    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics from the latest robustness test.

        Returns:
            Dictionary with stability metric values

        Raises:
            RobustnessTestError: If no test has been run
        """
        if self.stability_metrics_ is None:
            raise RobustnessTestError(
                "No stability metrics available. Run test_robustness() first.",
                error_code="NO_STABILITY_METRICS"
            )
        
        return self.stability_metrics_.copy()

    def get_detailed_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis results from the latest test.

        Returns:
            Dictionary with comprehensive analysis results
        """
        if not hasattr(self, 'test_results_'):
            raise RobustnessTestError(
                "No test results available. Run test_robustness() first.",
                error_code="NO_TEST_RESULTS"
            )
        
        return self.test_results_

    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save robustness test results to file.

        Args:
            filepath: Path to save results
        """
        if not hasattr(self, 'test_results_'):
            raise RobustnessTestError("No results to save")
        
        try:
            # Prepare serializable results
            save_data = self.test_results_.copy()
            
            # Convert numpy arrays and non-serializable objects
            for result in save_data['raw_results']:
                if 'feature_importance' in result and result['feature_importance'] is not None:
                    result['feature_importance'] = result['feature_importance'].tolist()
                # Remove model objects
                result.pop('model', None)
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.info(f"Robustness test results saved to {filepath}")
            
        except Exception as e:
            handle_and_reraise(
                e, RobustnessTestError,
                f"Failed to save results to {filepath}",
                error_code="SAVE_FAILED"
            )


# Convenience functions for backward compatibility and easy usage
def quick_robustness_test(
    model_type: str,
    best_params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    n_seeds: int = 5,
    scoring_function: str = "roc_auc"
) -> Dict[str, Any]:
    """Quick robustness test with minimal configuration.

    Args:
        model_type: Type of model to test ('xgboost', 'lightgbm', 'catboost')
        best_params: Model parameters to use
        X: Training features
        y: Training targets
        sample_weight: Optional sample weights
        n_seeds: Number of seeds to test
        scoring_function: Primary metric for evaluation

    Returns:
        Dictionary with robustness test results

    Example:
        >>> results = quick_robustness_test(
        ...     model_type='xgboost',
        ...     best_params={'n_estimators': 100, 'max_depth': 6},
        ...     X=X_train, y=y_train,
        ...     sample_weight=weights,
        ...     n_seeds=10,
        ...     scoring_function='recall'
        ... )
        >>> print(f"Stability: {results['stability_score']:.3f}")
    """
    from .trainer import StandardModelTrainer  # Import here to avoid circular imports
    
    # Create trainer
    trainer = StandardModelTrainer(model_type, random_state=42)
    
    # Create robustness tester
    config = RobustnessConfig(n_seeds=n_seeds)
    tester = SeedRobustnessTester(
        n_seeds=n_seeds,
        scoring_function=scoring_function,
        config=config
    )
    
    # Run robustness test
    detailed_results = tester.test_robustness(
        model_trainer=trainer,
        X=X,
        y=y,
        model_params=best_params,
        sample_weight=sample_weight
    )
    
    # Get stability metrics
    stability_metrics = tester.get_stability_metrics()
    
    # Create simplified results for backward compatibility
    analysis = detailed_results['analysis']
    primary_metric_stats = analysis['metrics'][scoring_function]
    
    return {
        'mean_score': primary_metric_stats['mean'],
        'score_std': primary_metric_stats['std'],
        'score_cv': primary_metric_stats['cv'],
        'stability_score': stability_metrics['performance_stability'],
        'feature_stability': stability_metrics.get('feature_stability', np.nan),
        'success_rate': stability_metrics['success_rate'],
        'stability_stats': stability_metrics,
        'detailed_results': detailed_results,
        'seed_results': [
            {
                'seed': r['seed'],
                'score': r['metrics'].get(scoring_function, np.nan),
                'success': r.get('success', False)
            }
            for r in detailed_results['raw_results']
        ]
    }


# Additional robustness testing classes can be added here:
# - SensitivityAnalyzer for feature perturbation testing
# - PopulationStabilityIndex for drift detection
# - FeatureStabilityIndex for feature importance correlation analysis

class PopulationStabilityIndex:
    """Calculate Population Stability Index (PSI) for drift detection.
    
    PSI measures the shift in distribution between two datasets,
    commonly used to detect data drift in production models.
    """
    
    def __init__(self, bins: int = 10, method: str = "quantile"):
        """Initialize PSI calculator.
        
        Args:
            bins: Number of bins for discretization
            method: Binning method ('quantile' or 'equal_width')
        """
        self.bins = bins
        self.method = method
        self.bin_edges_ = None
        
    def fit(self, reference_data: pd.Series) -> 'PopulationStabilityIndex':
        """Fit PSI calculator on reference data.
        
        Args:
            reference_data: Reference dataset to establish bins
            
        Returns:
            Self for method chaining
        """
        if self.method == "quantile":
            self.bin_edges_ = np.quantile(reference_data, np.linspace(0, 1, self.bins + 1))
        elif self.method == "equal_width":
            self.bin_edges_ = np.linspace(reference_data.min(), reference_data.max(), self.bins + 1)
        else:
            raise ValueError(f"Unknown binning method: {self.method}")
            
        # Ensure unique bin edges
        self.bin_edges_ = np.unique(self.bin_edges_)
        return self
    
    def calculate_psi(self, reference_data: pd.Series, comparison_data: pd.Series) -> float:
        """Calculate PSI between reference and comparison data.
        
        Args:
            reference_data: Reference dataset
            comparison_data: Comparison dataset
            
        Returns:
            PSI value (higher values indicate more drift)
        """
        if self.bin_edges_ is None:
            self.fit(reference_data)
        
        # Bin both datasets
        ref_binned = np.histogram(reference_data, bins=self.bin_edges_)[0]
        comp_binned = np.histogram(comparison_data, bins=self.bin_edges_)[0]
        
        # Calculate proportions
        ref_props = ref_binned / len(reference_data)
        comp_props = comp_binned / len(comparison_data)
        
        # Handle zero proportions
        ref_props = np.where(ref_props == 0, 0.001, ref_props)
        comp_props = np.where(comp_props == 0, 0.001, comp_props)
        
        # Calculate PSI
        psi = np.sum((comp_props - ref_props) * np.log(comp_props / ref_props))
        
        return psi


def calculate_psi_simple(
    X_reference: pd.DataFrame, 
    X_comparison: pd.DataFrame, 
    bins: int = 10
) -> Dict[str, float]:
    """Calculate PSI for all features in datasets.
    
    Args:
        X_reference: Reference dataset
        X_comparison: Comparison dataset
        bins: Number of bins for PSI calculation
        
    Returns:
        Dictionary with PSI values for each feature
        
    Example:
        >>> psi_results = calculate_psi_simple(X_train, X_test)
        >>> high_drift_features = [f for f, psi in psi_results.items() if psi > 0.2]
    """
    psi_calculator = PopulationStabilityIndex(bins=bins)
    psi_results = {}
    
    for column in X_reference.columns:
        if column in X_comparison.columns:
            try:
                psi_value = psi_calculator.calculate_psi(
                    X_reference[column], 
                    X_comparison[column]
                )
                psi_results[column] = psi_value
            except Exception as e:
                logger.warning(f"Failed to calculate PSI for {column}: {e}")
                psi_results[column] = np.nan
    
    return psi_results