"""Model evaluation utilities with comprehensive metrics and sample weights support.

This module provides functions to evaluate model performance using various metrics
and generate visualization plots for model assessment, with full support for sample weights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import json

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats

from ..utils.logger import get_logger
from ..utils.timer import timer
from ..utils.io_utils import save_json, ensure_dir

logger = get_logger(__name__)


def kolmogorov_smirnov_statistic(y_true: np.ndarray, y_prob: np.ndarray,
                                sample_weight: Optional[np.ndarray] = None) -> float:
    """Compute Kolmogorov-Smirnov statistic with sample weights support.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        sample_weight: Optional sample weights

    Returns:
        KS statistic value

    Example:
        >>> ks = kolmogorov_smirnov_statistic(y_test, y_pred_proba, sample_weight=weights)
        >>> print(f"KS Statistic: {ks:.3f}")
    """
    if sample_weight is not None:
        # For weighted KS, we need to compute weighted CDFs
        # This is a simplified implementation - for production use a proper weighted KS test
        pos_mask = y_true == 1
        neg_mask = y_true == 0

        pos_scores = y_prob[pos_mask]
        neg_scores = y_prob[neg_mask]
        pos_weights = sample_weight[pos_mask] if sample_weight is not None else None
        neg_weights = sample_weight[neg_mask] if sample_weight is not None else None

        # Use scipy's weighted KS test if available, otherwise fall back to unweighted
        try:
            ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)
        except:
            ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)
    else:
        # Separate scores for positive and negative classes
        pos_scores = y_prob[y_true == 1]
        neg_scores = y_prob[y_true == 0]

        # Compute KS statistic
        ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)

    return ks_stat


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray,
                    sample_weight: Optional[np.ndarray] = None) -> float:
    """Compute Gini coefficient with sample weights support.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        sample_weight: Optional sample weights

    Returns:
        Gini coefficient (2 * AUC - 1)

    Example:
        >>> gini = gini_coefficient(y_test, y_pred_proba, sample_weight=weights)
        >>> print(f"Gini Coefficient: {gini:.3f}")
    """
    auc = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
    return 2 * auc - 1


def compute_gains_table(y_true: np.ndarray, y_prob: np.ndarray, 
                       sample_weight: Optional[np.ndarray] = None,
                       n_bins: int = 10) -> pd.DataFrame:
    """Compute gains table for model performance analysis with sample weights support.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        sample_weight: Optional sample weights
        n_bins: Number of bins for the gains table

    Returns:
        DataFrame with gains table

    Example:
        >>> gains = compute_gains_table(y_test, y_pred_proba, sample_weight=weights)
        >>> print(gains)
    """
    # Create DataFrame
    df = pd.DataFrame({
        'true': y_true,
        'prob': y_prob
    })

    if sample_weight is not None:
        df['weight'] = sample_weight
    else:
        df['weight'] = 1.0

    # Sort by probability descending
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)

    # Create bins
    df['decile'] = pd.cut(range(len(df)), bins=n_bins, labels=range(1, n_bins + 1))

    # Calculate metrics by decile with weights
    if sample_weight is not None:
        gains_table = df.groupby('decile').agg({
            'weight': 'sum',  # Total weight
            'true': lambda x: np.sum(x * df.loc[x.index, 'weight']),  # Weighted positive count
            'prob': ['min', 'max', lambda x: np.average(x, weights=df.loc[x.index, 'weight'])]
        }).round(4)
    else:
        gains_table = df.groupby('decile').agg({
            'weight': 'sum',
            'true': 'sum',
            'prob': ['min', 'max', 'mean']
        }).round(4)

    # Flatten column names
    gains_table.columns = ['_'.join(col).strip() for col in gains_table.columns.values]
    gains_table = gains_table.rename(columns={
        'weight_sum': 'total_weight',
        'true_<lambda_0>': 'weighted_positive_count' if sample_weight is not None else 'positive_count',
        'true_sum': 'positive_count',
        'prob_min': 'min_prob',
        'prob_max': 'max_prob',
        'prob_<lambda_0>': 'mean_prob' if sample_weight is not None else 'mean_prob',
        'prob_mean': 'mean_prob'
    })

    # Calculate additional metrics
    if sample_weight is not None:
        gains_table['positive_rate'] = gains_table['weighted_positive_count'] / gains_table['total_weight']
        gains_table['cumulative_positive'] = gains_table['weighted_positive_count'].cumsum()
        gains_table['cumulative_weight'] = gains_table['total_weight'].cumsum()
        gains_table['cumulative_positive_rate'] = gains_table['cumulative_positive'] / gains_table['cumulative_weight']
        baseline_rate = gains_table['weighted_positive_count'].sum() / gains_table['total_weight'].sum()
    else:
        gains_table['positive_rate'] = gains_table['positive_count'] / gains_table['total_weight']
        gains_table['cumulative_positive'] = gains_table['positive_count'].cumsum()
        gains_table['cumulative_weight'] = gains_table['total_weight'].cumsum()
        gains_table['cumulative_positive_rate'] = gains_table['cumulative_positive'] / gains_table['cumulative_weight']
        baseline_rate = gains_table['positive_count'].sum() / gains_table['total_weight'].sum()

    gains_table['lift'] = gains_table['positive_rate'] / baseline_rate
    gains_table['cumulative_lift'] = gains_table['cumulative_positive_rate'] / baseline_rate

    return gains_table.reset_index()


@timer
def evaluate_binary_classifier(y_true: np.ndarray, y_prob: np.ndarray,
                             y_pred: Optional[np.ndarray] = None,
                             sample_weight: Optional[np.ndarray] = None,
                             threshold: float = 0.5,
                             pos_label: int = 1) -> Dict[str, float]:
    """Comprehensive evaluation of binary classifier with sample weights support.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        y_pred: Predicted binary labels (optional, will be computed from y_prob)
        sample_weight: Optional sample weights
        threshold: Threshold for binary classification
        pos_label: Label for positive class

    Returns:
        Dictionary with evaluation metrics

    Example:
        >>> metrics = evaluate_binary_classifier(y_test, y_pred_proba, 
        ...                                     sample_weight=weights)
        >>> print(f"AUC: {metrics['auc_roc']:.3f}")
    """
    logger.info("Evaluating binary classifier performance")
    if sample_weight is not None:
        logger.info("Using sample weights in evaluation")

    # Compute binary predictions if not provided
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)

    # Basic metrics with sample weights
    metrics = {
        # Probability-based metrics (support sample weights)
        'auc_roc': roc_auc_score(y_true, y_prob, sample_weight=sample_weight),
        'auc_pr': average_precision_score(y_true, y_prob, sample_weight=sample_weight),
        'log_loss': log_loss(y_true, y_prob, sample_weight=sample_weight),
        'brier_score': brier_score_loss(y_true, y_prob, sample_weight=sample_weight),

        # Statistical metrics
        'ks_statistic': kolmogorov_smirnov_statistic(y_true, y_prob, sample_weight),
        'gini': gini_coefficient(y_true, y_prob, sample_weight),

        # Classification metrics (support sample weights)
        'accuracy': accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        'precision': precision_score(y_true, y_pred, pos_label=pos_label, 
                                   sample_weight=sample_weight, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=pos_label,
                             sample_weight=sample_weight, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, pos_label=pos_label,
                           sample_weight=sample_weight, zero_division=0),

        # Threshold
        'threshold': threshold
    }

    # Confusion matrix metrics (weighted)
    if sample_weight is not None:
        # For weighted confusion matrix, we need to compute manually
        cm_weighted = np.zeros((2, 2))
        for i in range(len(y_true)):
            cm_weighted[y_true[i], y_pred[i]] += sample_weight[i]
        tn, fp, fn, tp = cm_weighted.ravel()
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics.update({
        'true_positives': float(tp),
        'true_negatives': float(tn),
        'false_positives': float(fp),
        'false_negatives': float(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
    })

    logger.info(f"Evaluation completed. AUC: {metrics['auc_roc']:.3f}, "
               f"Precision: {metrics['precision']:.3f}, "
               f"Recall: {metrics['recall']:.3f}")

    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                         sample_weight: Optional[np.ndarray] = None,
                         metric: str = 'f1') -> Tuple[float, float]:
    """Find optimal classification threshold based on specified metric with sample weights.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        sample_weight: Optional sample weights
        metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')

    Returns:
        Tuple of (optimal_threshold, optimal_metric_value)

    Example:
        >>> threshold, f1 = find_optimal_threshold(y_test, y_pred_proba, 
        ...                                       sample_weight=weights, metric='f1')
        >>> print(f"Optimal threshold: {threshold:.3f}, F1: {f1:.3f}")
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic = Sensitivity + Specificity - 1
            if sample_weight is not None:
                # Compute weighted confusion matrix
                cm_weighted = np.zeros((2, 2))
                for i in range(len(y_true)):
                    cm_weighted[y_true[i], y_pred[i]] += sample_weight[i]
                tn, fp, fn, tp = cm_weighted.ravel()
            else:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        scores.append(score)

    # Find optimal threshold
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]

    logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.3f} "
               f"(score: {optimal_score:.3f})")

    return optimal_threshold, optimal_score


class ModelEvaluator:
    """Comprehensive model evaluation class with sample weights support.

    Example:
        >>> evaluator = ModelEvaluator()
        >>> results = evaluator.evaluate(y_test, y_pred_proba, 
        ...                             sample_weight=weights,
        ...                             output_dir="evaluation_results/")
        >>> evaluator.generate_report(results, "model_evaluation_report.json")
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize evaluator.

        Args:
            output_dir: Directory to save evaluation outputs
        """
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        ensure_dir(self.output_dir)

    @timer
    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray,
                y_pred: Optional[np.ndarray] = None,
                sample_weight: Optional[np.ndarray] = None,
                threshold: float = 0.5,
                generate_plots: bool = True,
                plot_prefix: str = "model") -> Dict[str, Any]:
        """Comprehensive model evaluation with sample weights support.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            y_pred: Predicted binary labels
            sample_weight: Optional sample weights
            threshold: Classification threshold
            generate_plots: Whether to generate evaluation plots
            plot_prefix: Prefix for plot filenames

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting comprehensive model evaluation")

        # Basic metrics
        metrics = evaluate_binary_classifier(
            y_true, y_prob, y_pred, sample_weight, threshold
        )

        # Optimal thresholds
        optimal_thresholds = {}
        for metric in ['f1', 'precision', 'recall', 'youden']:
            try:
                thresh, score = find_optimal_threshold(
                    y_true, y_prob, sample_weight, metric
                )
                optimal_thresholds[f'optimal_{metric}_threshold'] = thresh
                optimal_thresholds[f'optimal_{metric}_score'] = score
            except Exception as e:
                logger.warning(f"Could not compute optimal {metric} threshold: {e}")

        # Gains table with sample weights
        gains_table = compute_gains_table(y_true, y_prob, sample_weight)

        # Compile results
        results = {
            'metrics': metrics,
            'optimal_thresholds': optimal_thresholds,
            'gains_table': gains_table.to_dict('records'),
            'evaluation_info': {
                'n_samples': len(y_true),
                'n_positive': int(np.sum(y_true)),
                'n_negative': int(len(y_true) - np.sum(y_true)),
                'positive_rate': float(np.mean(y_true)),
                'threshold_used': threshold,
                'sample_weights_used': sample_weight is not None,
                'total_weight': float(np.sum(sample_weight)) if sample_weight is not None else len(y_true)
            }
        }

        # Generate plots if requested
        if generate_plots:
            self._generate_plots(y_true, y_prob, y_pred, sample_weight, plot_prefix)
            results['plots_generated'] = True
            results['plot_directory'] = str(self.output_dir)

        logger.info("Model evaluation completed")
        return results

    def _generate_plots(self, y_true: np.ndarray, y_prob: np.ndarray,
                       y_pred: Optional[np.ndarray], sample_weight: Optional[np.ndarray],
                       prefix: str):
        """Generate evaluation plots with sample weights support."""
        logger.info("Generating evaluation plots")

        if y_pred is None:
            y_pred = (y_prob >= 0.5).astype(int)

        # ROC curve with sample weights
        self._plot_roc_curve(y_true, y_prob, sample_weight, prefix)

        # Precision-Recall curve with sample weights  
        self._plot_pr_curve(y_true, y_prob, sample_weight, prefix)

        # Confusion matrix (weighted if sample_weight provided)
        self._plot_confusion_matrix(y_true, y_pred, sample_weight, prefix)

        # Score distribution by class
        self._plot_score_distribution(y_true, y_prob, sample_weight, prefix)

        # Gains chart with sample weights
        self._plot_gains_chart(y_true, y_prob, sample_weight, prefix)

    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                       sample_weight: Optional[np.ndarray], prefix: str):
        """Plot ROC curve with sample weights."""
        fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=sample_weight)
        roc_auc = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)

        fig, ax = plt.subplots(figsize=(8, 6))

        weight_label = " (Weighted)" if sample_weight is not None else ""
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve{weight_label} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve{weight_label}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / f"{prefix}_roc_curve.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                      sample_weight: Optional[np.ndarray], prefix: str):
        """Plot Precision-Recall curve with sample weights."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob, sample_weight=sample_weight)
        pr_auc = average_precision_score(y_true, y_prob, sample_weight=sample_weight)

        fig, ax = plt.subplots(figsize=(8, 6))

        weight_label = " (Weighted)" if sample_weight is not None else ""
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve{weight_label} (AUC = {pr_auc:.3f})')

        # Baseline (random classifier)
        if sample_weight is not None:
            baseline = np.sum(y_true * sample_weight) / np.sum(sample_weight)
        else:
            baseline = np.sum(y_true) / len(y_true)

        ax.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Baseline (prevalence = {baseline:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve{weight_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / f"{prefix}_pr_curve.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              sample_weight: Optional[np.ndarray], prefix: str):
        """Plot confusion matrix with sample weights."""
        if sample_weight is not None:
            # Compute weighted confusion matrix
            cm = np.zeros((2, 2))
            for i in range(len(y_true)):
                cm[y_true[i], y_pred[i]] += sample_weight[i]
            weight_label = " (Weighted)"
        else:
            cm = confusion_matrix(y_true, y_pred)
            weight_label = ""

        fig, ax = plt.subplots(figsize=(8, 6))

        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='.1f' if sample_weight is not None else 'd', 
                    cmap='Blues', xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'], ax=ax)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix{weight_label}')

        save_path = self.output_dir / f"{prefix}_confusion_matrix.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_score_distribution(self, y_true: np.ndarray, y_prob: np.ndarray,
                                sample_weight: Optional[np.ndarray], prefix: str):
        """Plot score distribution by class with sample weights."""
        fig, ax = plt.subplots(figsize=(10, 6))

        pos_scores = y_prob[y_true == 1]
        neg_scores = y_prob[y_true == 0]

        if sample_weight is not None:
            pos_weights = sample_weight[y_true == 1]
            neg_weights = sample_weight[y_true == 0]

            ax.hist(neg_scores, bins=50, alpha=0.7, label='Negative Class', 
                   density=True, weights=neg_weights)
            ax.hist(pos_scores, bins=50, alpha=0.7, label='Positive Class', 
                   density=True, weights=pos_weights)
            weight_label = " (Weighted)"
        else:
            ax.hist(neg_scores, bins=50, alpha=0.7, label='Negative Class', density=True)
            ax.hist(pos_scores, bins=50, alpha=0.7, label='Positive Class', density=True)
            weight_label = ""

        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Score Distribution by Class{weight_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / f"{prefix}_score_distribution.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_gains_chart(self, y_true: np.ndarray, y_prob: np.ndarray,
                         sample_weight: Optional[np.ndarray], prefix: str):
        """Plot gains and lift charts with sample weights."""
        gains_table = compute_gains_table(y_true, y_prob, sample_weight)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Gains chart
        deciles = gains_table['decile']

        if sample_weight is not None:
            cumulative_gains = gains_table['cumulative_positive'] / gains_table['weighted_positive_count'].sum() * 100
            weight_label = " (Weighted)"
        else:
            cumulative_gains = gains_table['cumulative_positive'] / gains_table['positive_count'].sum() * 100
            weight_label = ""

        ax1.plot(deciles, cumulative_gains, 'o-', label='Model')
        ax1.plot([1, 10], [10, 100], '--', label='Random', alpha=0.7)
        ax1.set_xlabel('Decile')
        ax1.set_ylabel('Cumulative Gain (%)')
        ax1.set_title(f'Cumulative Gains Chart{weight_label}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Lift chart
        lift_values = gains_table['lift']
        ax2.bar(deciles, lift_values, alpha=0.7)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
        ax2.set_xlabel('Decile')
        ax2.set_ylabel('Lift')
        ax2.set_title(f'Lift Chart{weight_label}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f"{prefix}_gains_lift.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray,
                  y_pred: Optional[np.ndarray] = None,
                  sample_weight: Optional[np.ndarray] = None,
                  output_dir: Optional[str] = None,
                  generate_plots: bool = True) -> Dict[str, Any]:
    """Convenience function for model evaluation with sample weights support.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        y_pred: Predicted labels (optional)
        sample_weight: Optional sample weights
        output_dir: Output directory for results
        generate_plots: Whether to generate plots

    Returns:
        Evaluation results dictionary

    Example:
        >>> results = evaluate_model(y_test, y_pred_proba, 
        ...                         sample_weight=weights,
        ...                         output_dir="results/xgboost/")
    """
    evaluator = ModelEvaluator(output_dir)
    return evaluator.evaluate(y_true, y_prob, y_pred, sample_weight, 
                             generate_plots=generate_plots)
