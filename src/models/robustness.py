"""Model robustness and stability testing utilities.

This module provides comprehensive robustness testing for tree-based models including:
- Seed robustness testing (retrain with multiple seeds)
- Sensitivity analysis (feature perturbation testing)
- Feature stability index (correlation of feature importances across runs)
- Distribution drift detection (Population Stability Index - PSI)
- Sample weights integration throughout all tests

Key Features:
- Multi-seed model training and evaluation
- Feature perturbation sensitivity analysis
- Feature importance stability assessment
- Data drift detection with PSI calculations
- Comprehensive reporting and visualization
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

from ..utils.logger import get_logger
from ..utils.timer import timer

logger = get_logger(__name__)


@dataclass
class RobustnessConfig:
    """Configuration for robustness testing."""
    n_seeds: int = 10
    test_size: float = 0.2
    perturbation_range: Tuple[float, float] = (-0.1, 0.1)
    n_perturbations: int = 100
    psi_bins: int = 10
    confidence_level: float = 0.95
    parallel_jobs: int = -1

    def __post_init__(self):
        """Validate configuration."""
        if self.n_seeds < 2:
            raise ValueError("n_seeds must be at least 2")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.n_perturbations < 10:
            raise ValueError("n_perturbations must be at least 10")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")


class SeedRobustnessTester:
    """Test model robustness across different random seeds.

    Trains models with different random seeds and evaluates the stability
    of performance metrics and feature importances.

    Example:
        >>> tester = SeedRobustnessTester(model_class=xgb.XGBClassifier, model_params=params)
        >>> results = tester.test_robustness(X, y, sample_weight=weights)
        >>> print(f"Mean AUC: {results['metrics']['roc_auc']['mean']:.4f} ± {results['metrics']['roc_auc']['std']:.4f}")
    """

    def __init__(self, model_class, model_params: Dict[str, Any], 
                 config: Optional[RobustnessConfig] = None):
        """Initialize seed robustness tester.

        Args:
            model_class: Model class (e.g., xgb.XGBClassifier)
            model_params: Parameters for model initialization
            config: Robustness testing configuration
        """
        self.model_class = model_class
        self.model_params = model_params.copy()
        self.config = config or RobustnessConfig()

        # Remove random_state/seed parameters to allow manual setting
        seed_params = ['random_state', 'random_seed', 'seed']
        for param in seed_params:
            self.model_params.pop(param, None)

        logger.info(f"Initialized seed robustness tester with {self.config.n_seeds} seeds")

    def _train_and_evaluate_single_seed(self, seed: int, X: pd.DataFrame, y: pd.Series,
                                      sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train and evaluate model with single seed."""
        try:
            # Set seed in model parameters
            model_params = self.model_params.copy()
            if 'XGBClassifier' in str(self.model_class):
                model_params['random_state'] = seed
            elif 'LGBMClassifier' in str(self.model_class):
                model_params['random_state'] = seed
            elif 'CatBoostClassifier' in str(self.model_class):
                model_params['random_seed'] = seed
            else:
                model_params['random_state'] = seed

            # Split data with seed
            if sample_weight is not None:
                X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
                    X, y, sample_weight, test_size=self.config.test_size,
                    random_state=seed, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.config.test_size,
                    random_state=seed, stratify=y
                )
                sw_train, sw_test = None, None

            # Train model
            model = self.model_class(**model_params)

            if sw_train is not None and hasattr(model, 'fit'):
                # Check if model supports sample_weight
                try:
                    model.fit(X_train, y_train, sample_weight=sw_train)
                except TypeError:
                    logger.warning(f"Model does not support sample_weight, training without it")
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = y_pred  # Binary predictions

            # Calculate metrics
            metrics = {}

            # ROC AUC
            try:
                if sw_test is not None:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, sample_weight=sw_test)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = np.nan

            # Average Precision
            try:
                if sw_test is not None:
                    metrics['average_precision'] = average_precision_score(y_test, y_pred_proba, sample_weight=sw_test)
                else:
                    metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
            except ValueError:
                metrics['average_precision'] = np.nan

            # Other metrics
            try:
                if sw_test is not None:
                    metrics['accuracy'] = accuracy_score(y_test, y_pred, sample_weight=sw_test)
                    metrics['precision'] = precision_score(y_test, y_pred, sample_weight=sw_test, zero_division=0)
                    metrics['recall'] = recall_score(y_test, y_pred, sample_weight=sw_test, zero_division=0)
                    metrics['f1'] = f1_score(y_test, y_pred, sample_weight=sw_test, zero_division=0)
                else:
                    metrics['accuracy'] = accuracy_score(y_test, y_pred)
                    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
                    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
                    metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
            except Exception as e:
                logger.warning(f"Error calculating metrics for seed {seed}: {e}")
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    metrics[metric] = np.nan

            # Get feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):  # CatBoost
                feature_importance = model.get_feature_importance()

            return {
                'seed': seed,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'model': model,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }

        except Exception as e:
            logger.error(f"Error training model with seed {seed}: {e}")
            return {
                'seed': seed,
                'error': str(e),
                'metrics': {metric: np.nan for metric in ['roc_auc', 'average_precision', 'accuracy', 'precision', 'recall', 'f1']},
                'feature_importance': None
            }

    @timer
    def test_robustness(self, X: pd.DataFrame, y: pd.Series,
                       sample_weight: Optional[np.ndarray] = None,
                       parallel: bool = False) -> Dict[str, Any]:
        """Test model robustness across multiple seeds.

        Args:
            X: Features
            y: Target
            sample_weight: Sample weights
            parallel: Whether to use parallel processing

        Returns:
            Dictionary with robustness test results
        """
        logger.info(f"Testing seed robustness with {self.config.n_seeds} seeds")

        # Generate seeds
        np.random.seed(42)
        seeds = np.random.randint(0, 10000, self.config.n_seeds)

        # Train models
        results = []

        # Sequential processing (avoiding multiprocessing issues)
        for seed in seeds:
            result = self._train_and_evaluate_single_seed(seed, X, y, sample_weight)
            results.append(result)
            logger.info(f"Completed seed {seed}")

        # Analyze results
        analysis = self._analyze_seed_results(results, list(X.columns))

        logger.info(f"✅ Seed robustness testing completed")
        logger.info(f"   Mean AUC: {analysis['metrics']['roc_auc']['mean']:.4f} ± {analysis['metrics']['roc_auc']['std']:.4f}")
        logger.info(f"   Mean F1: {analysis['metrics']['f1']['mean']:.4f} ± {analysis['metrics']['f1']['std']:.4f}")

        return {
            'config': self.config,
            'raw_results': results,
            'analysis': analysis,
            'seeds_used': seeds.tolist()
        }

    def _analyze_seed_results(self, results: List[Dict[str, Any]], 
                            feature_names: List[str]) -> Dict[str, Any]:
        """Analyze results from multiple seed runs."""
        # Extract metrics
        metrics_data = {}
        for result in results:
            if 'error' not in result:
                for metric, value in result['metrics'].items():
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    metrics_data[metric].append(value)

        # Calculate metric statistics
        metrics_stats = {}
        for metric, values in metrics_data.items():
            values = [v for v in values if not np.isnan(v)]
            if values:
                metrics_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
                }
            else:
                metrics_stats[metric] = {
                    'mean': np.nan, 'std': np.nan, 'min': np.nan,
                    'max': np.nan, 'median': np.nan, 'cv': np.nan
                }

        # Analyze feature importance stability
        feature_importance_stability = None
        importance_matrices = []

        for result in results:
            if result.get('feature_importance') is not None and 'error' not in result:
                importance_matrices.append(result['feature_importance'])

        if len(importance_matrices) >= 2:
            importance_df = pd.DataFrame(importance_matrices, columns=feature_names)

            # Calculate correlations between runs
            correlations = []
            for i in range(len(importance_matrices)):
                for j in range(i + 1, len(importance_matrices)):
                    corr, _ = pearsonr(importance_matrices[i], importance_matrices[j])
                    correlations.append(corr)

            feature_importance_stability = {
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'min_correlation': np.min(correlations),
                'max_correlation': np.max(correlations),
                'importance_means': importance_df.mean().to_dict(),
                'importance_stds': importance_df.std().to_dict(),
                'importance_cv': (importance_df.std() / importance_df.mean()).to_dict()
            }

        # Calculate success rate
        successful_runs = len([r for r in results if 'error' not in r])
        success_rate = successful_runs / len(results)

        return {
            'metrics': metrics_stats,
            'feature_importance_stability': feature_importance_stability,
            'success_rate': success_rate,
            'total_runs': len(results),
            'successful_runs': successful_runs,
            'failed_runs': len(results) - successful_runs
        }
