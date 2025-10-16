"""Model explainability and interpretability utilities.

This module provides comprehensive explainability tools for tree-based models including:
- SHAP (SHapley Additive exPlanations) analysis
- Scorecard conversion for predicted probabilities
- Partial dependence and ICE plots
- Reason code generation for individual predictions
- Visual summaries with PNG/HTML export capabilities
- Sample weights integration throughout

Key Features:
- SHAP explainability with sample weights support
- Business-friendly scorecard conversion
- Interactive and static visualizations
- Reason code generation for compliance
- Production-ready export formats
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import warnings
import json
from dataclasses import dataclass
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import train_test_split

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from ..utils.logger import get_logger
from ..utils.timer import timer

logger = get_logger(__name__)


@dataclass
class ScorecardConfig:
    """Configuration for scorecard conversion."""
    min_score: int = 300
    max_score: int = 850
    odds_at_min: float = 50.0  # 50:1 odds at min score (2% probability)
    odds_at_max: float = 1/50.0  # 1:50 odds at max score (98% probability)  
    pdo: int = 20  # Points to double the odds

    def __post_init__(self):
        """Validate scorecard configuration."""
        if self.min_score >= self.max_score:
            raise ValueError("min_score must be less than max_score")
        if self.pdo <= 0:
            raise ValueError("pdo must be positive")


class ScorecardConverter:
    """Convert predicted probabilities to business-friendly scorecards.

    Scorecards are widely used in financial services to present
    risk scores in an intuitive format (e.g., 300-850 like FICO).

    Example:
        >>> converter = ScorecardConverter()
        >>> scores = converter.fit_transform(probabilities)
        >>> print(f"Fraud risk score: {scores[0]:.0f}/850")
    """

    def __init__(self, config: Optional[ScorecardConfig] = None):
        """Initialize scorecard converter.

        Args:
            config: Scorecard configuration
        """
        self.config = config or ScorecardConfig()
        self.fitted = False
        self.offset = None
        self.factor = None

    def fit(self, probabilities: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fit scorecard parameters to probability distribution.

        Args:
            probabilities: Predicted probabilities [0, 1]
            sample_weight: Sample weights (optional)

        Returns:
            Self
        """
        logger.info("Fitting scorecard converter")

        # Convert probabilities to odds
        probabilities = np.clip(probabilities, 1e-7, 1-1e-7)  # Avoid division by zero
        odds = probabilities / (1 - probabilities)

        # Calculate scorecard parameters
        # Score = offset + factor * log(odds)
        # At min_score: odds = odds_at_min
        # At max_score: odds = odds_at_max

        log_odds_min = np.log(self.config.odds_at_min)
        log_odds_max = np.log(self.config.odds_at_max)

        # Solve linear system for offset and factor
        self.factor = (self.config.max_score - self.config.min_score) / (log_odds_max - log_odds_min)
        self.offset = self.config.min_score - self.factor * log_odds_min

        self.fitted = True

        logger.info(f"Scorecard fitted: offset={self.offset:.2f}, factor={self.factor:.2f}")
        logger.info(f"Score range: {self.config.min_score}-{self.config.max_score}")

        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform probabilities to scorecard scores.

        Args:
            probabilities: Predicted probabilities [0, 1]

        Returns:
            Scorecard scores
        """
        if not self.fitted:
            raise ValueError("Must fit scorecard converter first")

        # Convert probabilities to odds
        probabilities = np.clip(probabilities, 1e-7, 1-1e-7)
        odds = probabilities / (1 - probabilities)

        # Calculate scores
        scores = self.offset + self.factor * np.log(odds)

        # Clip to valid range
        scores = np.clip(scores, self.config.min_score, self.config.max_score)

        return scores

    def fit_transform(self, probabilities: np.ndarray, 
                     sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit converter and transform probabilities in one step.

        Args:
            probabilities: Predicted probabilities
            sample_weight: Sample weights

        Returns:
            Scorecard scores
        """
        return self.fit(probabilities, sample_weight).transform(probabilities)

    def interpret_score(self, score: float) -> Dict[str, Any]:
        """Interpret a scorecard score in business terms.

        Args:
            score: Scorecard score

        Returns:
            Dictionary with interpretation
        """
        # Convert back to probability
        log_odds = (score - self.offset) / self.factor
        odds = np.exp(log_odds)
        probability = odds / (1 + odds)

        # Risk category
        if score >= 750:
            risk_category = "Very Low Risk"
            color = "green"
        elif score >= 650:
            risk_category = "Low Risk"
            color = "lightgreen"
        elif score >= 550:
            risk_category = "Medium Risk"
            color = "yellow"
        elif score >= 450:
            risk_category = "High Risk"
            color = "orange"
        else:
            risk_category = "Very High Risk"
            color = "red"

        return {
            "score": score,
            "probability": probability,
            "odds": odds,
            "risk_category": risk_category,
            "color": color,
            "percentile": self._score_to_percentile(score)
        }

    def _score_to_percentile(self, score: float) -> float:
        """Convert score to approximate percentile (simplified)."""
        normalized = (score - self.config.min_score) / (self.config.max_score - self.config.min_score)
        return normalized * 100


class SHAPExplainer:
    """SHAP-based model explainability with sample weights support.

    Provides comprehensive SHAP analysis including:
    - SHAP value computation with various explainer types
    - Summary plots and dependence plots
    - Individual prediction explanations
    - Feature importance ranking
    - Export to PNG/HTML formats

    Example:
        >>> explainer = SHAPExplainer(model)
        >>> shap_values = explainer.compute_shap_values(X, sample_weight=weights)
        >>> explainer.plot_shap_summary(save_path='shap_summary.png')
    """

    def __init__(self, model, explainer_type: str = "auto"):
        """Initialize SHAP explainer.

        Args:
            model: Trained model
            explainer_type: Type of SHAP explainer ('tree', 'linear', 'kernel', 'auto')
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")

        self.model = model
        self.explainer_type = explainer_type
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.feature_names = None

        logger.info(f"Initialized SHAP explainer with type: {explainer_type}")

    def _create_explainer(self, X: pd.DataFrame):
        """Create appropriate SHAP explainer for the model."""
        if self.explainer_type == "auto":
            # Auto-detect best explainer type
            if hasattr(self.model, 'get_booster'):  # XGBoost
                explainer_type = "tree"
            elif hasattr(self.model, '_Booster'):  # LightGBM
                explainer_type = "tree"
            elif hasattr(self.model, 'get_feature_importance'):  # CatBoost
                explainer_type = "tree"
            else:
                explainer_type = "kernel"
        else:
            explainer_type = self.explainer_type

        logger.info(f"Creating {explainer_type} SHAP explainer")

        if explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, X)
        elif explainer_type == "kernel":
            # Use a sample for kernel explainer to avoid long computation
            background_sample = X.sample(min(100, len(X)), random_state=42)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background_sample)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

        return self.explainer

    @timer
    def compute_shap_values(self, X: pd.DataFrame, 
                          sample_weight: Optional[np.ndarray] = None,
                          max_samples: Optional[int] = None) -> np.ndarray:
        """Compute SHAP values for dataset.

        Args:
            X: Feature DataFrame
            sample_weight: Sample weights (used for sampling if max_samples specified)
            max_samples: Maximum number of samples to compute (for performance)

        Returns:
            SHAP values array
        """
        logger.info(f"Computing SHAP values for {len(X)} samples")

        # Store feature names
        self.feature_names = list(X.columns)

        # Create explainer if not exists
        if self.explainer is None:
            self._create_explainer(X)

        # Sample data if requested
        if max_samples and len(X) > max_samples:
            if sample_weight is not None:
                # Weighted sampling
                sample_probs = sample_weight / sample_weight.sum()
                indices = np.random.choice(len(X), size=max_samples, replace=False, p=sample_probs)
            else:
                # Random sampling
                indices = np.random.choice(len(X), size=max_samples, replace=False)

            X_sample = X.iloc[indices]
            logger.info(f"Sampled {len(X_sample)} samples for SHAP computation")
        else:
            X_sample = X
            indices = None

        # Compute SHAP values
        if hasattr(self.explainer, 'shap_values'):
            # TreeExplainer and others
            shap_values = self.explainer.shap_values(X_sample)

            # Handle multi-class case (take positive class for binary classification)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # Positive class

        else:
            # KernelExplainer
            shap_values = self.explainer(X_sample)
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values

        self.shap_values = shap_values
        self.expected_value = self.explainer.expected_value

        # Handle expected value for multi-class
        if isinstance(self.expected_value, (list, np.ndarray)) and len(self.expected_value) == 2:
            self.expected_value = self.expected_value[1]  # Positive class

        logger.info(f"SHAP values computed: {shap_values.shape}")
        logger.info(f"Expected value: {self.expected_value:.4f}")

        return shap_values

    def plot_shap_summary(self, max_display: int = 20, 
                         save_path: Optional[Union[str, Path]] = None,
                         plot_type: str = "dot") -> None:
        """Create SHAP summary plot.

        Args:
            max_display: Maximum number of features to display
            save_path: Path to save plot (PNG format)
            plot_type: Type of plot ('dot', 'bar', 'violin')
        """
        if self.shap_values is None:
            raise ValueError("Must compute SHAP values first")

        logger.info(f"Creating SHAP summary plot (type: {plot_type})")

        plt.figure(figsize=(10, 8))

        try:
            if plot_type == "dot":
                shap.summary_plot(self.shap_values, 
                                feature_names=self.feature_names,
                                max_display=max_display, 
                                show=False)
            elif plot_type == "bar":
                shap.summary_plot(self.shap_values,
                                feature_names=self.feature_names,
                                plot_type="bar",
                                max_display=max_display,
                                show=False)
            elif plot_type == "violin":
                shap.summary_plot(self.shap_values,
                                feature_names=self.feature_names,
                                plot_type="violin", 
                                max_display=max_display,
                                show=False)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            plt.title(f"SHAP Summary Plot ({plot_type.title()})", fontsize=14, pad=20)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
            plt.close()
            raise

        plt.close()

    def plot_dependence(self, feature_idx: Union[int, str], 
                       interaction_feature: Optional[Union[int, str]] = None,
                       save_path: Optional[Union[str, Path]] = None) -> None:
        """Create SHAP dependence plot.

        Args:
            feature_idx: Feature index or name
            interaction_feature: Feature to color points by (auto-selected if None)
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("Must compute SHAP values first")

        # Convert feature name to index if needed
        if isinstance(feature_idx, str):
            if feature_idx not in self.feature_names:
                raise ValueError(f"Feature '{feature_idx}' not found")
            feature_name = feature_idx
            feature_idx = self.feature_names.index(feature_idx)
        else:
            feature_name = self.feature_names[feature_idx]

        logger.info(f"Creating SHAP dependence plot for feature: {feature_name}")

        plt.figure(figsize=(10, 6))

        try:
            shap.dependence_plot(feature_idx, self.shap_values,
                               feature_names=self.feature_names,
                               interaction_index=interaction_feature,
                               show=False)

            plt.title(f"SHAP Dependence Plot: {feature_name}", fontsize=14, pad=20)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP dependence plot saved to {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {e}")
            plt.close()
            raise

        plt.close()

    def plot_force(self, instance_idx: int, 
                  save_path: Optional[Union[str, Path]] = None) -> None:
        """Create SHAP force plot for individual prediction.

        Args:
            instance_idx: Index of instance to explain
            save_path: Path to save plot (HTML format)
        """
        if self.shap_values is None:
            raise ValueError("Must compute SHAP values first")

        logger.info(f"Creating SHAP force plot for instance {instance_idx}")

        try:
            force_plot = shap.force_plot(
                self.expected_value,
                self.shap_values[instance_idx],
                feature_names=self.feature_names,
                show=False
            )

            if save_path:
                shap.save_html(save_path, force_plot)
                logger.info(f"SHAP force plot saved to {save_path}")
            else:
                force_plot.show()

        except Exception as e:
            logger.error(f"Error creating SHAP force plot: {e}")
            raise

    def get_feature_importance(self, importance_type: str = "mean_abs") -> pd.DataFrame:
        """Get feature importance from SHAP values.

        Args:
            importance_type: Type of importance ('mean_abs', 'mean', 'std')

        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("Must compute SHAP values first")

        if importance_type == "mean_abs":
            importance = np.abs(self.shap_values).mean(axis=0)
        elif importance_type == "mean":
            importance = self.shap_values.mean(axis=0)
        elif importance_type == "std":
            importance = self.shap_values.std(axis=0)
        else:
            raise ValueError(f"Unknown importance type: {importance_type}")

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info(f"Generated {importance_type} feature importance")

        return df


class ReasonCodeGenerator:
    """Generate reason codes for individual predictions.

    Provides business-friendly explanations for model predictions
    by identifying top contributing features and their impacts.

    Example:
        >>> generator = ReasonCodeGenerator()
        >>> reasons = generator.generate_reason_codes(shap_values, feature_names, X)
        >>> print(reasons[0]['top_reasons'])
    """

    def __init__(self, max_reasons: int = 5, min_impact: float = 0.01):
        """Initialize reason code generator.

        Args:
            max_reasons: Maximum number of reason codes per prediction
            min_impact: Minimum SHAP impact to include in reasons
        """
        self.max_reasons = max_reasons
        self.min_impact = min_impact

    def generate_reason_codes(self, shap_values: np.ndarray,
                            feature_names: List[str],
                            X: pd.DataFrame,
                            instance_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Generate reason codes for predictions.

        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            X: Original features DataFrame
            instance_indices: Specific instances to explain (all if None)

        Returns:
            List of reason code dictionaries
        """
        if instance_indices is None:
            instance_indices = range(len(shap_values))

        logger.info(f"Generating reason codes for {len(instance_indices)} instances")

        reason_codes = []

        for idx in instance_indices:
            instance_shap = shap_values[idx]
            instance_features = X.iloc[idx]

            # Get feature contributions sorted by absolute impact
            contributions = []
            for i, (feature_name, shap_value) in enumerate(zip(feature_names, instance_shap)):
                if abs(shap_value) >= self.min_impact:
                    contributions.append({
                        'feature': feature_name,
                        'shap_value': shap_value,
                        'feature_value': instance_features[feature_name],
                        'impact': 'increases' if shap_value > 0 else 'decreases',
                        'abs_impact': abs(shap_value)
                    })

            # Sort by absolute impact and take top reasons
            contributions.sort(key=lambda x: x['abs_impact'], reverse=True)
            top_reasons = contributions[:self.max_reasons]

            # Generate human-readable reasons
            readable_reasons = []
            for reason in top_reasons:
                readable_text = self._format_reason(reason)
                readable_reasons.append({
                    'text': readable_text,
                    'feature': reason['feature'],
                    'impact': reason['shap_value'],
                    'value': reason['feature_value']
                })

            reason_codes.append({
                'instance_index': idx,
                'total_shap_impact': sum(instance_shap),
                'top_reasons': readable_reasons,
                'all_contributions': contributions
            })

        logger.info(f"Generated reason codes with average {np.mean([len(rc['top_reasons']) for rc in reason_codes]):.1f} reasons per instance")

        return reason_codes

    def _format_reason(self, contribution: Dict[str, Any]) -> str:
        """Format a contribution as human-readable reason."""
        feature = contribution['feature']
        value = contribution['feature_value']
        impact = contribution['impact']

        # Create readable feature name
        readable_feature = feature.replace('_', ' ').title()

        # Format value based on type
        if isinstance(value, (int, float)):
            if abs(value) < 0.01:
                formatted_value = f"{value:.4f}"
            elif abs(value) < 1:
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value:.1f}"
        else:
            formatted_value = str(value)

        return f"{readable_feature} ({formatted_value}) {impact} fraud risk"

    def export_reason_codes(self, reason_codes: List[Dict[str, Any]], 
                          save_path: Union[str, Path]) -> None:
        """Export reason codes to JSON file.

        Args:
            reason_codes: Generated reason codes
            save_path: Path to save JSON file
        """
        # Convert numpy types to native Python types for JSON serialization
        serializable_codes = []
        for code in reason_codes:
            serializable_code = {
                'instance_index': int(code['instance_index']),
                'total_shap_impact': float(code['total_shap_impact']),
                'top_reasons': [
                    {
                        'text': reason['text'],
                        'feature': reason['feature'],
                        'impact': float(reason['impact']),
                        'value': float(reason['value']) if isinstance(reason['value'], (int, float, np.number)) else str(reason['value'])
                    }
                    for reason in code['top_reasons']
                ]
            }
            serializable_codes.append(serializable_code)

        # Simple JSON save
        with open(save_path, 'w') as f:
            json.dump(serializable_codes, f, indent=2)

        logger.info(f"Reason codes exported to {save_path}")


class PartialDependencePlotter:
    """Create partial dependence and ICE plots.

    Partial dependence plots show the marginal effect of features on predictions.
    ICE (Individual Conditional Expectation) plots show prediction variation
    for individual instances as features change.
    """

    def __init__(self, model, feature_names: List[str]):
        """Initialize PD plotter.

        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names

    @timer
    def plot_partial_dependence(self, X: pd.DataFrame, 
                               features: Union[List[str], List[int]],
                               sample_weight: Optional[np.ndarray] = None,
                               grid_resolution: int = 100,
                               save_path: Optional[Union[str, Path]] = None) -> None:
        """Create partial dependence plots.

        Args:
            X: Feature DataFrame
            features: Features to plot (names or indices)
            sample_weight: Sample weights
            grid_resolution: Number of points in grid
            save_path: Path to save plot
        """
        logger.info(f"Creating partial dependence plots for {len(features)} features")

        # Convert feature names to indices if needed
        if isinstance(features[0], str):
            feature_indices = [self.feature_names.index(f) for f in features]
            feature_names_plot = features
        else:
            feature_indices = features
            feature_names_plot = [self.feature_names[i] for i in features]

        # Create subplots
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, (feature_idx, feature_name) in enumerate(zip(feature_indices, feature_names_plot)):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            try:
                # Compute partial dependence
                pd_result = partial_dependence(
                    self.model, X, [feature_idx],
                    grid_resolution=grid_resolution,
                    method='auto'
                )

                # Plot
                ax.plot(pd_result[1][0], pd_result[0][0], linewidth=2, color='blue')
                ax.set_xlabel(feature_name)
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'Partial Dependence: {feature_name}')
                ax.grid(True, alpha=0.3)

            except Exception as e:
                logger.warning(f"Could not create PD plot for {feature_name}: {e}")
                ax.text(0.5, 0.5, f'Error: {feature_name}', 
                       ha='center', va='center', transform=ax.transAxes)

        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.remove()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Partial dependence plots saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_ice(self, X: pd.DataFrame, feature: Union[str, int],
                sample_weight: Optional[np.ndarray] = None,
                max_samples: int = 100,
                grid_resolution: int = 50,
                save_path: Optional[Union[str, Path]] = None) -> None:
        """Create ICE (Individual Conditional Expectation) plot.

        Args:
            X: Feature DataFrame
            feature: Feature to plot (name or index)
            sample_weight: Sample weights (used for sampling)
            max_samples: Maximum samples to show
            grid_resolution: Number of points in grid
            save_path: Path to save plot
        """
        # Convert feature name to index if needed
        if isinstance(feature, str):
            feature_idx = self.feature_names.index(feature)
            feature_name = feature
        else:
            feature_idx = feature
            feature_name = self.feature_names[feature_idx]

        logger.info(f"Creating ICE plot for feature: {feature_name}")

        # Sample instances for ICE plot
        if len(X) > max_samples:
            if sample_weight is not None:
                sample_probs = sample_weight / sample_weight.sum()
                indices = np.random.choice(len(X), size=max_samples, replace=False, p=sample_probs)
            else:
                indices = np.random.choice(len(X), size=max_samples, replace=False)
            X_sample = X.iloc[indices]
        else:
            X_sample = X

        # Create feature grid
        feature_values = X[X.columns[feature_idx]]
        feature_grid = np.linspace(feature_values.min(), feature_values.max(), grid_resolution)

        # Compute ICE lines
        ice_lines = []

        for idx in range(len(X_sample)):
            instance = X_sample.iloc[idx:idx+1].copy()
            line_predictions = []

            for grid_value in feature_grid:
                instance_modified = instance.copy()
                instance_modified.iloc[0, feature_idx] = grid_value

                if hasattr(self.model, 'predict_proba'):
                    pred = self.model.predict_proba(instance_modified)[0, 1]
                else:
                    pred = self.model.predict(instance_modified)[0]

                line_predictions.append(pred)

            ice_lines.append(line_predictions)

        # Plot ICE lines and average
        plt.figure(figsize=(10, 6))

        # Plot individual ICE lines
        for line in ice_lines:
            plt.plot(feature_grid, line, alpha=0.3, color='blue', linewidth=0.5)

        # Plot average (PDP)
        avg_line = np.mean(ice_lines, axis=0)
        plt.plot(feature_grid, avg_line, color='red', linewidth=3, label='Average (PDP)')

        plt.xlabel(feature_name)
        plt.ylabel('Prediction')
        plt.title(f'ICE Plot: {feature_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ICE plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


def ensure_dir(path: Union[str, Path]):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Union[str, Path]):
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


class ModelExplainerSuite:
    """Comprehensive model explainability suite.

    Combines all explainability tools into a single interface
    for comprehensive model interpretation and reporting.

    Example:
        >>> suite = ModelExplainerSuite(model, X_train, y_train)
        >>> suite.generate_full_report(X_test, sample_weight=weights, output_dir='explanations/')
    """

    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series):
        """Initialize explainer suite.

        Args:
            model: Trained model
            X_train: Training features (for background)
            y_train: Training target
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = list(X_train.columns)

        # Initialize components
        self.shap_explainer = SHAPExplainer(model)
        self.scorecard_converter = ScorecardConverter()
        self.reason_generator = ReasonCodeGenerator()
        self.pd_plotter = PartialDependencePlotter(model, self.feature_names)

        logger.info(f"Initialized explainer suite with {len(self.feature_names)} features")

    @timer
    def generate_full_report(self, X_test: pd.DataFrame,
                           sample_weight: Optional[np.ndarray] = None,
                           output_dir: Union[str, Path] = "explanations",
                           max_samples_shap: int = 1000,
                           max_samples_ice: int = 100) -> Dict[str, Any]:
        """Generate comprehensive explainability report.

        Args:
            X_test: Test features
            sample_weight: Sample weights
            output_dir: Directory to save outputs
            max_samples_shap: Maximum samples for SHAP computation
            max_samples_ice: Maximum samples for ICE plots

        Returns:
            Dictionary with report summary
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        logger.info(f"Generating comprehensive explainability report in {output_dir}")

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_samples": len(X_test),
            "n_features": len(self.feature_names),
            "files_generated": []
        }

        # 1. Get predictions and convert to scorecard
        logger.info("Step 1: Generating predictions and scorecard scores")
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = self.model.predict(X_test)

        # Fit scorecard on predictions
        scorecard_scores = self.scorecard_converter.fit_transform(y_pred_proba, sample_weight)

        # Save scorecard results
        scorecard_df = pd.DataFrame({
            'probability': y_pred_proba,
            'scorecard_score': scorecard_scores,
            'risk_category': [self.scorecard_converter.interpret_score(score)['risk_category'] 
                            for score in scorecard_scores]
        })
        scorecard_path = output_dir / "scorecard_results.csv"
        scorecard_df.to_csv(scorecard_path, index=False)
        report["files_generated"].append(str(scorecard_path))

        # 2. Compute SHAP values
        logger.info("Step 2: Computing SHAP values")
        shap_values = self.shap_explainer.compute_shap_values(
            X_test, sample_weight, max_samples=max_samples_shap
        )

        # 3. Generate SHAP plots
        logger.info("Step 3: Generating SHAP visualizations")

        # Summary plot (dot)
        summary_dot_path = output_dir / "shap_summary_dot.png"
        self.shap_explainer.plot_shap_summary(save_path=summary_dot_path, plot_type="dot")
        report["files_generated"].append(str(summary_dot_path))

        # Summary plot (bar)
        summary_bar_path = output_dir / "shap_summary_bar.png"
        self.shap_explainer.plot_shap_summary(save_path=summary_bar_path, plot_type="bar")
        report["files_generated"].append(str(summary_bar_path))

        # Feature importance
        feature_importance = self.shap_explainer.get_feature_importance()
        importance_path = output_dir / "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        report["files_generated"].append(str(importance_path))

        # Top feature dependence plots
        top_features = feature_importance.head(5)['feature'].tolist()
        for i, feature in enumerate(top_features):
            dep_path = output_dir / f"shap_dependence_{feature}.png"
            self.shap_explainer.plot_dependence(feature, save_path=dep_path)
            report["files_generated"].append(str(dep_path))

        # 4. Generate reason codes
        logger.info("Step 4: Generating reason codes")
        reason_codes = self.reason_generator.generate_reason_codes(
            shap_values, self.feature_names, X_test
        )

        reason_codes_path = output_dir / "reason_codes.json"
        self.reason_generator.export_reason_codes(reason_codes, reason_codes_path)
        report["files_generated"].append(str(reason_codes_path))

        # 5. Create partial dependence plots
        logger.info("Step 5: Creating partial dependence plots")
        pd_path = output_dir / "partial_dependence.png"
        self.pd_plotter.plot_partial_dependence(
            X_test, top_features[:4], sample_weight, save_path=pd_path
        )
        report["files_generated"].append(str(pd_path))

        # 6. Create ICE plots for top 2 features
        logger.info("Step 6: Creating ICE plots")
        for i, feature in enumerate(top_features[:2]):
            ice_path = output_dir / f"ice_plot_{feature}.png"
            self.pd_plotter.plot_ice(
                X_test, feature, sample_weight, 
                max_samples=max_samples_ice, save_path=ice_path
            )
            report["files_generated"].append(str(ice_path))

        # 7. Generate individual explanations for high-risk instances
        logger.info("Step 7: Generating individual explanations")
        high_risk_indices = np.where(scorecard_scores < 500)[0][:5]  # Top 5 high-risk

        for i, idx in enumerate(high_risk_indices):
            force_path = output_dir / f"force_plot_highrisk_{i}.html"
            self.shap_explainer.plot_force(idx, save_path=force_path)
            report["files_generated"].append(str(force_path))

        # 8. Create summary statistics
        report["summary_stats"] = {
            "scorecard_stats": {
                "mean_score": float(scorecard_scores.mean()),
                "std_score": float(scorecard_scores.std()),
                "min_score": float(scorecard_scores.min()),
                "max_score": float(scorecard_scores.max()),
                "high_risk_pct": float((scorecard_scores < 500).mean() * 100)
            },
            "shap_stats": {
                "mean_abs_impact": float(np.abs(shap_values).mean()),
                "top_features": top_features,
                "expected_value": float(self.shap_explainer.expected_value)
            }
        }

        # Save report summary
        report_path = output_dir / "explainability_report.json"
        save_json(report, report_path)
        report["files_generated"].append(str(report_path))

        logger.info(f"✅ Explainability report completed! Generated {len(report['files_generated'])} files")

        return report


# Convenience functions
def quick_shap_analysis(model, X: pd.DataFrame, 
                       sample_weight: Optional[np.ndarray] = None,
                       save_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Quick SHAP analysis for a model.

    Args:
        model: Trained model
        X: Features
        sample_weight: Sample weights
        save_dir: Directory to save plots

    Returns:
        Dictionary with SHAP results

    Example:
        >>> results = quick_shap_analysis(model, X_test, sample_weight=weights)
        >>> print(f"Top feature: {results['feature_importance'].iloc[0]['feature']}")
    """
    explainer = SHAPExplainer(model)
    shap_values = explainer.compute_shap_values(X, sample_weight, max_samples=min(1000, len(X)))

    results = {
        'shap_values': shap_values,
        'expected_value': explainer.expected_value,
        'feature_importance': explainer.get_feature_importance()
    }

    if save_dir:
        save_dir = Path(save_dir)
        ensure_dir(save_dir)

        explainer.plot_shap_summary(save_path=save_dir / "shap_summary.png")
        results['feature_importance'].to_csv(save_dir / "feature_importance.csv", index=False)

    return results


def convert_to_scorecard(probabilities: np.ndarray,
                        sample_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ScorecardConverter]:
    """Convert probabilities to scorecard scores.

    Args:
        probabilities: Predicted probabilities
        sample_weight: Sample weights

    Returns:
        Tuple of (scores, fitted_converter)

    Example:
        >>> scores, converter = convert_to_scorecard(y_pred_proba)
        >>> interpretation = converter.interpret_score(scores[0])
        >>> print(f"Risk: {interpretation['risk_category']}")
    """
    converter = ScorecardConverter()
    scores = converter.fit_transform(probabilities, sample_weight)

    return scores, converter
