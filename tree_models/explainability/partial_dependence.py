# tree_models/explainability/partial_dependence.py
"""Enhanced partial dependence and ICE plotting with comprehensive features.

This module provides production-ready partial dependence analysis with:
- Type-safe interfaces and comprehensive error handling
- Partial Dependence Plots (PDP) for feature effect analysis
- Individual Conditional Expectation (ICE) plots for instance-level analysis
- Interactive plotting capabilities with Plotly integration
- Sample weights integration for representative analysis
- Performance optimization for large datasets
- Comprehensive visualization and export options
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
import warnings
from dataclasses import dataclass

from .base import BaseVisualizer, ensure_dir
from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    ExplainabilityError,
    ConfigurationError,
    DataValidationError,
    PerformanceError,
    handle_and_reraise,
    validate_parameter,
    create_error_context
)

logger = get_logger(__name__)

# Check for optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Interactive plots will be disabled.")

try:
    from sklearn.inspection import partial_dependence, PermutationImportance
    SKLEARN_PD_AVAILABLE = True
except ImportError:
    SKLEARN_PD_AVAILABLE = False
    logger.warning("sklearn.inspection not available. Using fallback implementation.")

warnings.filterwarnings('ignore')


@dataclass
class PDPlotConfig:
    """Configuration for partial dependence plotting."""
    
    # Grid resolution
    grid_resolution: int = 100
    percentiles: Tuple[float, float] = (0.05, 0.95)
    
    # Performance settings
    max_samples: int = 1000
    n_jobs: int = 1
    
    # Visualization settings
    figsize: Tuple[int, int] = (10, 6)
    show_ice: bool = True
    ice_alpha: float = 0.3
    ice_max_lines: int = 50
    
    # Interactive settings
    interactive: bool = False
    
    def __post_init__(self) -> None:
        """Validate PD plot configuration."""
        validate_parameter("grid_resolution", self.grid_resolution, min_value=10, max_value=1000)
        validate_parameter("max_samples", self.max_samples, min_value=50, max_value=10000)
        validate_parameter("ice_alpha", self.ice_alpha, min_value=0.0, max_value=1.0)
        validate_parameter("ice_max_lines", self.ice_max_lines, min_value=5, max_value=500)
        
        if not (0 < self.percentiles[0] < self.percentiles[1] < 1):
            raise ConfigurationError("Percentiles must be in (0, 1) with first < second")


class PartialDependencePlotter(BaseVisualizer):
    """Enhanced partial dependence and ICE plotter.
    
    Creates partial dependence plots showing marginal effects of features
    and Individual Conditional Expectation (ICE) plots showing per-instance
    feature effects.
    
    Example:
        >>> plotter = PartialDependencePlotter(model, feature_names)
        >>> pd_results = plotter.plot_partial_dependence(
        ...     X_test, ['feature_1', 'feature_2'], 
        ...     sample_weight=weights, save_path='pd_plots.png'
        ... )
        >>> ice_results = plotter.plot_ice(
        ...     X_test, 'feature_1', max_samples=100,
        ...     save_path='ice_feature_1.png'
        ... )
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        config: Optional[PDPlotConfig] = None,
        **kwargs: Any
    ) -> None:
        """Initialize enhanced PD plotter.
        
        Args:
            model: Trained model with predict or predict_proba method
            feature_names: List of feature names
            config: Plotting configuration
            **kwargs: Additional visualization parameters
        """
        super().__init__(**kwargs)
        
        self.model = model
        self.feature_names = feature_names
        self.config = config or PDPlotConfig()
        
        # Validate model
        self._validate_model()
        
        # Determine prediction function
        self.predict_fn = self._get_prediction_function()
        
        logger.info(f"Initialized PartialDependencePlotter:")
        logger.info(f"  Model: {type(model).__name__}, Features: {len(feature_names)}")
        logger.info(f"  Grid resolution: {self.config.grid_resolution}, Interactive: {self.config.interactive}")

    def _validate_model(self) -> None:
        """Validate that model has required prediction methods."""
        
        if not hasattr(self.model, 'predict') and not hasattr(self.model, 'predict_proba'):
            raise ConfigurationError("Model must have predict or predict_proba method")
        
        logger.debug("Model validation passed")

    def _get_prediction_function(self) -> Callable:
        """Get appropriate prediction function from model."""
        
        if hasattr(self.model, 'predict_proba'):
            # Use probability for positive class (binary classification)
            def predict_fn(X):
                proba = self.model.predict_proba(X)
                return proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
            return predict_fn
        else:
            # Use raw predictions
            return self.model.predict

    @timer(name="partial_dependence_computation")
    def compute_partial_dependence(
        self,
        X: pd.DataFrame,
        features: Union[List[str], List[int]],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Compute partial dependence values for specified features.
        
        Args:
            X: Input features
            features: Features to compute PD for (names or indices)
            sample_weight: Optional sample weights
            **kwargs: Additional computation parameters
            
        Returns:
            Dictionary with PD computation results
            
        Raises:
            ExplainabilityError: If PD computation fails
        """
        logger.info(f"🔍 Computing partial dependence for {len(features)} features")
        
        try:
            with timed_operation("pd_computation"):
                # Validate inputs
                self._validate_pd_inputs(X, features)
                
                # Convert feature names to indices if needed
                feature_indices, feature_names = self._resolve_feature_references(features)
                
                # Sample data if needed for performance
                X_sample, sample_indices = self._sample_data(X, sample_weight)
                
                # Compute PD for each feature
                pd_results = {}
                
                for feature_idx, feature_name in zip(feature_indices, feature_names):
                    logger.debug(f"Computing PD for feature: {feature_name}")
                    
                    try:
                        # Use scikit-learn implementation if available
                        if SKLEARN_PD_AVAILABLE:
                            pd_result = partial_dependence(
                                self.model, 
                                X_sample,
                                [feature_idx],
                                grid_resolution=self.config.grid_resolution,
                                percentiles=self.config.percentiles,
                                method='auto'
                            )
                            grid_values = pd_result['grid_values'][0]
                            pd_values = pd_result['average'][0]
                        else:
                            # Fallback implementation
                            grid_values, pd_values = self._compute_pd_manual(
                                X_sample, feature_idx, feature_name
                            )
                        
                        pd_results[feature_name] = {
                            'grid_values': grid_values,
                            'pd_values': pd_values,
                            'feature_index': feature_idx,
                            'feature_name': feature_name
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to compute PD for {feature_name}: {e}")
                        pd_results[feature_name] = {
                            'error': str(e),
                            'feature_name': feature_name
                        }
            
            logger.info(f"✅ Partial dependence computed for {len(pd_results)} features")
            
            return {
                'pd_results': pd_results,
                'feature_names': feature_names,
                'n_samples_used': len(X_sample),
                'sample_indices': sample_indices,
                'config': self.config
            }
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                "Partial dependence computation failed",
                error_code="PD_COMPUTATION_FAILED",
                context=create_error_context(
                    n_features=len(features),
                    n_samples=len(X),
                    grid_resolution=self.config.grid_resolution
                )
            )

    def _validate_pd_inputs(self, X: pd.DataFrame, features: Union[List[str], List[int]]) -> None:
        """Validate inputs for PD computation."""
        
        if X.empty:
            raise DataValidationError("Input DataFrame cannot be empty")
        
        if len(features) == 0:
            raise DataValidationError("Features list cannot be empty")
        
        # Check feature references
        for feature in features:
            if isinstance(feature, str):
                if feature not in X.columns:
                    raise DataValidationError(f"Feature '{feature}' not found in data")
            elif isinstance(feature, int):
                if feature < 0 or feature >= len(X.columns):
                    raise DataValidationError(f"Feature index {feature} out of range")
            else:
                raise DataValidationError(f"Invalid feature reference: {feature}")

    def _resolve_feature_references(
        self, 
        features: Union[List[str], List[int]]
    ) -> Tuple[List[int], List[str]]:
        """Resolve feature references to indices and names."""
        
        feature_indices = []
        feature_names = []
        
        for feature in features:
            if isinstance(feature, str):
                feature_idx = self.feature_names.index(feature)
                feature_indices.append(feature_idx)
                feature_names.append(feature)
            else:  # int
                feature_indices.append(feature)
                feature_names.append(self.feature_names[feature])
        
        return feature_indices, feature_names

    def _sample_data(
        self,
        X: pd.DataFrame,
        sample_weight: Optional[np.ndarray]
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Sample data for performance if needed."""
        
        if len(X) <= self.config.max_samples:
            return X, None
        
        logger.info(f"Sampling {self.config.max_samples} from {len(X)} samples for PD computation")
        
        np.random.seed(42)  # For reproducibility
        
        if sample_weight is not None:
            # Weighted sampling
            try:
                sample_probs = sample_weight / sample_weight.sum()
                indices = np.random.choice(
                    len(X), size=self.config.max_samples, replace=False, p=sample_probs
                )
            except Exception as e:
                logger.warning(f"Weighted sampling failed: {e}, using random sampling")
                indices = np.random.choice(len(X), size=self.config.max_samples, replace=False)
        else:
            # Random sampling
            indices = np.random.choice(len(X), size=self.config.max_samples, replace=False)
        
        return X.iloc[indices], indices

    def _compute_pd_manual(
        self,
        X: pd.DataFrame,
        feature_idx: int,
        feature_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Manual PD computation fallback."""
        
        feature_values = X.iloc[:, feature_idx]
        
        # Create grid
        if pd.api.types.is_numeric_dtype(feature_values):
            # Numeric feature
            percentile_min = np.percentile(feature_values, self.config.percentiles[0] * 100)
            percentile_max = np.percentile(feature_values, self.config.percentiles[1] * 100)
            grid_values = np.linspace(percentile_min, percentile_max, self.config.grid_resolution)
        else:
            # Categorical feature
            unique_values = feature_values.unique()
            grid_values = unique_values[:self.config.grid_resolution]
        
        # Compute PD values
        pd_values = []
        
        for grid_value in grid_values:
            # Create modified dataset with feature set to grid value
            X_modified = X.copy()
            X_modified.iloc[:, feature_idx] = grid_value
            
            # Get predictions
            predictions = self.predict_fn(X_modified)
            pd_values.append(np.mean(predictions))
        
        return grid_values, np.array(pd_values)

    @timer(name="ice_computation")
    def compute_ice(
        self,
        X: pd.DataFrame,
        feature: Union[str, int],
        sample_weight: Optional[np.ndarray] = None,
        max_samples: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Compute Individual Conditional Expectation (ICE) values.
        
        Args:
            X: Input features
            feature: Feature to compute ICE for
            sample_weight: Optional sample weights (used for sampling)
            max_samples: Maximum samples for ICE (overrides config)
            **kwargs: Additional computation parameters
            
        Returns:
            Dictionary with ICE computation results
        """
        if max_samples is None:
            max_samples = self.config.ice_max_lines
        
        logger.info(f"🧊 Computing ICE for feature: {feature}")
        
        try:
            with timed_operation("ice_computation"):
                # Resolve feature reference
                if isinstance(feature, str):
                    feature_idx = self.feature_names.index(feature)
                    feature_name = feature
                else:
                    feature_idx = feature
                    feature_name = self.feature_names[feature_idx]
                
                # Sample instances for ICE
                if len(X) > max_samples:
                    X_sample, sample_indices = self._sample_data_for_ice(
                        X, sample_weight, max_samples
                    )
                else:
                    X_sample = X
                    sample_indices = None
                
                # Get feature values and create grid
                feature_values = X.iloc[:, feature_idx]
                
                if pd.api.types.is_numeric_dtype(feature_values):
                    percentile_min = np.percentile(feature_values, self.config.percentiles[0] * 100)
                    percentile_max = np.percentile(feature_values, self.config.percentiles[1] * 100)
                    grid_values = np.linspace(percentile_min, percentile_max, self.config.grid_resolution)
                else:
                    unique_values = feature_values.unique()
                    grid_values = unique_values[:self.config.grid_resolution]
                
                # Compute ICE lines
                ice_lines = []
                
                for instance_idx in range(len(X_sample)):
                    instance = X_sample.iloc[instance_idx:instance_idx+1].copy()
                    ice_line = []
                    
                    for grid_value in grid_values:
                        instance_modified = instance.copy()
                        instance_modified.iloc[0, feature_idx] = grid_value
                        
                        prediction = self.predict_fn(instance_modified)[0]
                        ice_line.append(prediction)
                    
                    ice_lines.append(ice_line)
                
                ice_lines = np.array(ice_lines)
                
                # Compute average (PDP)
                pd_line = np.mean(ice_lines, axis=0)
            
            logger.info(f"✅ ICE computed for {len(ice_lines)} instances")
            
            return {
                'ice_lines': ice_lines,
                'pd_line': pd_line,
                'grid_values': grid_values,
                'feature_name': feature_name,
                'feature_index': feature_idx,
                'n_instances': len(ice_lines),
                'sample_indices': sample_indices
            }
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                f"ICE computation failed for feature {feature}",
                error_code="ICE_COMPUTATION_FAILED"
            )

    def _sample_data_for_ice(
        self,
        X: pd.DataFrame,
        sample_weight: Optional[np.ndarray],
        max_samples: int
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Sample data specifically for ICE computation."""
        
        logger.debug(f"Sampling {max_samples} instances for ICE computation")
        
        np.random.seed(42)
        
        if sample_weight is not None and len(sample_weight) == len(X):
            # Weighted sampling
            try:
                sample_probs = sample_weight / sample_weight.sum()
                indices = np.random.choice(
                    len(X), size=max_samples, replace=False, p=sample_probs
                )
            except Exception:
                indices = np.random.choice(len(X), size=max_samples, replace=False)
        else:
            # Random sampling
            indices = np.random.choice(len(X), size=max_samples, replace=False)
        
        return X.iloc[indices], indices

    def plot_partial_dependence(
        self,
        X: pd.DataFrame,
        features: Union[List[str], List[int]],
        sample_weight: Optional[np.ndarray] = None,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Create partial dependence plots for multiple features.
        
        Args:
            X: Input features
            features: Features to plot
            sample_weight: Optional sample weights
            save_path: Optional path to save plot
            **kwargs: Additional plotting parameters
            
        Returns:
            Plot figure object
        """
        logger.info(f"📊 Creating partial dependence plots for {len(features)} features")
        
        try:
            # Compute PD values
            pd_computation = self.compute_partial_dependence(X, features, sample_weight)
            pd_results = pd_computation['pd_results']
            
            # Create plots
            if self.config.interactive and PLOTLY_AVAILABLE:
                fig = self._create_interactive_pd_plots(pd_results)
            else:
                fig = self._create_static_pd_plots(pd_results)
            
            # Save if requested
            if save_path:
                self.save_plot(fig, save_path)
            
            return fig
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                f"Failed to create PD plots for {len(features)} features",
                error_code="PD_PLOT_CREATION_FAILED"
            )

    def _create_static_pd_plots(self, pd_results: Dict[str, Any]) -> Any:
        """Create static matplotlib PD plots."""
        
        n_features = len([r for r in pd_results.values() if 'grid_values' in r])
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        for feature_name, result in pd_results.items():
            if 'error' in result:
                continue
                
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            try:
                grid_values = result['grid_values']
                pd_values = result['pd_values']
                
                ax.plot(grid_values, pd_values, linewidth=2, color='blue', marker='o', markersize=3)
                ax.set_xlabel(feature_name)
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'Partial Dependence: {feature_name}')
                ax.grid(True, alpha=0.3)
                
                # Format x-axis for categorical features
                if not pd.api.types.is_numeric_dtype(grid_values):
                    ax.tick_params(axis='x', rotation=45)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {feature_name}\\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
            
            plot_idx += 1
        
        # Remove empty subplots
        for i in range(plot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        return fig

    def _create_interactive_pd_plots(self, pd_results: Dict[str, Any]) -> Any:
        """Create interactive Plotly PD plots."""
        
        n_features = len([r for r in pd_results.values() if 'grid_values' in r])
        
        # Create subplots
        fig = make_subplots(
            rows=(n_features + 2) // 3,
            cols=min(3, n_features),
            subplot_titles=list(pd_results.keys()),
            vertical_spacing=0.1
        )
        
        plot_idx = 0
        for feature_name, result in pd_results.items():
            if 'error' in result:
                continue
                
            row = plot_idx // 3 + 1
            col = plot_idx % 3 + 1
            
            try:
                grid_values = result['grid_values']
                pd_values = result['pd_values']
                
                trace = go.Scatter(
                    x=grid_values,
                    y=pd_values,
                    mode='lines+markers',
                    name=feature_name,
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                )
                
                fig.add_trace(trace, row=row, col=col)
                
                fig.update_xaxes(title_text=feature_name, row=row, col=col)
                fig.update_yaxes(title_text='Partial Dependence', row=row, col=col)
                
            except Exception as e:
                logger.warning(f"Failed to create interactive plot for {feature_name}: {e}")
            
            plot_idx += 1
        
        fig.update_layout(
            title="Partial Dependence Plots",
            showlegend=False,
            height=400 * ((n_features + 2) // 3)
        )
        
        return fig

    def plot_ice(
        self,
        X: pd.DataFrame,
        feature: Union[str, int],
        sample_weight: Optional[np.ndarray] = None,
        max_samples: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Create ICE plot for a single feature.
        
        Args:
            X: Input features
            feature: Feature to plot
            sample_weight: Optional sample weights
            max_samples: Maximum samples to show
            save_path: Optional path to save plot
            **kwargs: Additional plotting parameters
            
        Returns:
            Plot figure object
        """
        logger.info(f"🧊 Creating ICE plot for feature: {feature}")
        
        try:
            # Compute ICE values
            ice_result = self.compute_ice(X, feature, sample_weight, max_samples)
            
            # Create plot
            if self.config.interactive and PLOTLY_AVAILABLE:
                fig = self._create_interactive_ice_plot(ice_result)
            else:
                fig = self._create_static_ice_plot(ice_result)
            
            # Save if requested
            if save_path:
                self.save_plot(fig, save_path)
            
            return fig
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                f"Failed to create ICE plot for feature {feature}",
                error_code="ICE_PLOT_CREATION_FAILED"
            )

    def _create_static_ice_plot(self, ice_result: Dict[str, Any]) -> Any:
        """Create static matplotlib ICE plot."""
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        grid_values = ice_result['grid_values']
        ice_lines = ice_result['ice_lines']
        pd_line = ice_result['pd_line']
        feature_name = ice_result['feature_name']
        
        # Plot ICE lines
        for ice_line in ice_lines:
            ax.plot(grid_values, ice_line, alpha=self.config.ice_alpha, 
                   color='lightblue', linewidth=0.5)
        
        # Plot average (PDP)
        ax.plot(grid_values, pd_line, color='red', linewidth=3, 
               label='Average (PDP)', zorder=10)
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Prediction')
        ax.set_title(f'ICE Plot: {feature_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for categorical features
        if not pd.api.types.is_numeric_dtype(grid_values):
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

    def _create_interactive_ice_plot(self, ice_result: Dict[str, Any]) -> Any:
        """Create interactive Plotly ICE plot."""
        
        grid_values = ice_result['grid_values']
        ice_lines = ice_result['ice_lines']
        pd_line = ice_result['pd_line']
        feature_name = ice_result['feature_name']
        
        fig = go.Figure()
        
        # Add ICE lines
        for i, ice_line in enumerate(ice_lines):
            fig.add_trace(go.Scatter(
                x=grid_values,
                y=ice_line,
                mode='lines',
                line=dict(color='lightblue', width=1),
                opacity=self.config.ice_alpha,
                showlegend=False,
                hovertemplate=f'Instance {i}<br>%{{x}}: %{{y:.3f}}<extra></extra>'
            ))
        
        # Add PDP line
        fig.add_trace(go.Scatter(
            x=grid_values,
            y=pd_line,
            mode='lines',
            line=dict(color='red', width=3),
            name='Average (PDP)',
            hovertemplate='Average<br>%{x}: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'ICE Plot: {feature_name}',
            xaxis_title=feature_name,
            yaxis_title='Prediction',
            showlegend=True,
            hovermode='closest'
        )
        
        return fig

    def plot(
        self,
        data: Any,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Generic plot method for base class compatibility."""
        
        if isinstance(data, dict):
            if 'pd_results' in data:
                # PD plot
                return self._create_static_pd_plots(data['pd_results'])
            elif 'ice_lines' in data:
                # ICE plot  
                return self._create_static_ice_plot(data)
        
        raise ExplainabilityError("Invalid data format for PD plotting")


# Convenience functions
def create_partial_dependence_plots(
    model: Any,
    X: pd.DataFrame,
    features: Union[List[str], List[int]],
    feature_names: Optional[List[str]] = None,
    sample_weight: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> Any:
    """Create partial dependence plots with default settings.
    
    Args:
        model: Trained model
        X: Input features
        features: Features to plot
        feature_names: Feature names (uses X.columns if None)
        sample_weight: Optional sample weights
        save_path: Optional path to save plot
        **kwargs: Additional plotting parameters
        
    Returns:
        Plot figure object
        
    Example:
        >>> fig = create_partial_dependence_plots(
        ...     model, X_test, ['feature_1', 'feature_2'],
        ...     sample_weight=weights, save_path='pd_plots.png'
        ... )
    """
    if feature_names is None:
        feature_names = list(X.columns)
    
    plotter = PartialDependencePlotter(model, feature_names, **kwargs)
    return plotter.plot_partial_dependence(X, features, sample_weight, save_path)


def create_ice_plot(
    model: Any,
    X: pd.DataFrame,
    feature: Union[str, int],
    feature_names: Optional[List[str]] = None,
    sample_weight: Optional[np.ndarray] = None,
    max_samples: int = 50,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> Any:
    """Create ICE plot with default settings.
    
    Args:
        model: Trained model
        X: Input features
        feature: Feature to plot
        feature_names: Feature names (uses X.columns if None)
        sample_weight: Optional sample weights
        max_samples: Maximum samples for ICE lines
        save_path: Optional path to save plot
        **kwargs: Additional plotting parameters
        
    Returns:
        Plot figure object
        
    Example:
        >>> fig = create_ice_plot(
        ...     model, X_test, 'important_feature',
        ...     max_samples=100, save_path='ice_plot.png'
        ... )
    """
    if feature_names is None:
        feature_names = list(X.columns)
    
    plotter = PartialDependencePlotter(model, feature_names, **kwargs)
    return plotter.plot_ice(X, feature, sample_weight, max_samples, save_path)


# Export key classes and functions
__all__ = [
    'PDPlotConfig',
    'PartialDependencePlotter',
    'create_partial_dependence_plots',
    'create_ice_plot'
]