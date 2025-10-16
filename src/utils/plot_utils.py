"""Plotting utilities for model evaluation and analysis.

This module provides standardized plotting functions for ML model evaluation,
feature importance, and SHAP analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve

from ..config.model_config import PLOT_CONFIG
from .logger import get_logger
from .io_utils import ensure_dir

logger = get_logger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette(PLOT_CONFIG["color_palette"])


def setup_plot(figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Set up a matplotlib figure with consistent styling.

    Args:
        figsize: Figure size tuple (width, height)

    Returns:
        Figure and axes objects

    Example:
        >>> fig, ax = setup_plot((12, 8))
        >>> ax.plot(x, y)
        >>> plt.show()
    """
    if figsize is None:
        figsize = PLOT_CONFIG["figsize"]

    fig, ax = plt.subplots(figsize=figsize, dpi=PLOT_CONFIG["dpi"])
    return fig, ax


def save_plot(fig: plt.Figure, filepath: Union[str, Path], 
              close_after_save: bool = True) -> None:
    """Save matplotlib figure to file.

    Args:
        fig: Matplotlib figure
        filepath: Output file path
        close_after_save: Whether to close figure after saving

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)
        >>> save_plot(fig, "plots/my_plot.png")
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    fig.savefig(filepath, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight', 
                format=PLOT_CONFIG["save_format"])

    logger.info(f"Saved plot to {filepath}")

    if close_after_save:
        plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                   title: str = "ROC Curve", 
                   save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Plot ROC curve.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        title: Plot title
        save_path: Optional path to save plot

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_roc_curve(y_test, y_pred_proba)
        >>> plt.show()
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)

    fig, ax = setup_plot()

    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if save_path:
        save_plot(fig, save_path, close_after_save=False)

    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray,
                               title: str = "Precision-Recall Curve",
                               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Plot Precision-Recall curve.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        title: Plot title
        save_path: Optional path to save plot

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_precision_recall_curve(y_test, y_pred_proba)
        >>> plt.show()
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    from sklearn.metrics import auc
    pr_auc = auc(recall, precision)

    fig, ax = setup_plot()

    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='red', linestyle='--', 
               label=f'Baseline (prevalence = {baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        save_plot(fig, save_path, close_after_save=False)

    return fig


def plot_feature_importance(feature_names: List[str], importance_values: np.ndarray,
                          title: str = "Feature Importance", top_k: int = 20,
                          save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Plot feature importance.

    Args:
        feature_names: List of feature names
        importance_values: Feature importance values
        title: Plot title
        top_k: Number of top features to display
        save_path: Optional path to save plot

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_feature_importance(feature_names, model.feature_importances_)
        >>> plt.show()
    """
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=True).tail(top_k)

    fig, ax = setup_plot(figsize=(10, max(8, top_k * 0.4)))

    bars = ax.barh(importance_df['feature'], importance_df['importance'])

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')

    ax.set_xlabel('Importance')
    ax.set_title(f'{title} (Top {top_k})')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path, close_after_save=False)

    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: Optional[List[str]] = None,
                         title: str = "Confusion Matrix",
                         save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        labels: Class labels
        title: Plot title
        save_path: Optional path to save plot

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_confusion_matrix(y_test, y_pred_binary)
        >>> plt.show()
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = setup_plot(figsize=(8, 6))

    if labels is None:
        labels = ['Negative', 'Positive']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    if save_path:
        save_plot(fig, save_path, close_after_save=False)

    return fig


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                          n_bins: int = 10, title: str = "Calibration Plot",
                          save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Plot calibration curve (reliability diagram).

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
        title: Plot title
        save_path: Optional path to save plot

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_calibration_curve(y_test, y_pred_proba)
        >>> plt.show()
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins)

    fig, ax = setup_plot()

    ax.plot(mean_predicted_value, fraction_of_positives, "s-",
            label="Model", linewidth=2)
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        save_plot(fig, save_path, close_after_save=False)

    return fig


def plot_distribution_comparison(data1: np.ndarray, data2: np.ndarray,
                               labels: Tuple[str, str] = ("Dataset 1", "Dataset 2"),
                               title: str = "Distribution Comparison",
                               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Plot distribution comparison between two datasets.

    Args:
        data1: First dataset
        data2: Second dataset
        labels: Labels for the datasets
        title: Plot title
        save_path: Optional path to save plot

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_distribution_comparison(train_scores, test_scores,
        ...                                  ("Train", "Test"))
        >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Histograms
    ax1.hist(data1, bins=50, alpha=0.7, label=labels[0], density=True)
    ax1.hist(data2, bins=50, alpha=0.7, label=labels[1], density=True)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title} - Histograms')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plots
    ax2.boxplot([data1, data2], labels=labels)
    ax2.set_ylabel('Value')
    ax2.set_title(f'{title} - Box Plots')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path, close_after_save=False)

    return fig


def create_subplots_grid(n_plots: int, ncols: int = 2, 
                        figsize_per_plot: Tuple[int, int] = (6, 4)) -> Tuple[plt.Figure, np.ndarray]:
    """Create a grid of subplots for multiple plots.

    Args:
        n_plots: Number of plots needed
        ncols: Number of columns in the grid
        figsize_per_plot: Size of each individual plot

    Returns:
        Figure and array of axes

    Example:
        >>> fig, axes = create_subplots_grid(4, ncols=2)
        >>> for i, ax in enumerate(axes.flat):
        ...     ax.plot(x, y_data[i])
    """
    nrows = (n_plots + ncols - 1) // ncols
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                            dpi=PLOT_CONFIG["dpi"])

    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Hide unused subplots
    for i in range(n_plots, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    return fig, axes
