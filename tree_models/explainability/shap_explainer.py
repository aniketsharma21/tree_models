# tree_models/explainability/shap_explainer.py
"""Enhanced SHAP explainer with comprehensive features and error handling.

This module provides production-ready SHAP (SHapley Additive exPlanations) 
integration with:
- Type-safe interfaces and comprehensive error handling  
- Multiple SHAP explainer types with auto-detection
- Sample weights integration where possible
- Performance optimization for large datasets
- Rich visualization capabilities with export options
- Memory management and timeout handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import warnings

from .base import BaseExplainer, BaseVisualizer, ensure_dir, validate_explanation_data
from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    ExplainabilityError,
    ConfigurationError,
    PerformanceError,
    handle_and_reraise,
    validate_parameter,
    create_error_context
)

logger = get_logger(__name__)

# Check SHAP availability
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP explainability available")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

warnings.filterwarnings('ignore')


class SHAPExplainer(BaseExplainer):
    """Enhanced SHAP explainer with comprehensive features and error handling.
    
    Provides production-ready SHAP analysis with automatic explainer selection,
    performance optimization, and rich visualization capabilities.
    
    Example:
        >>> explainer = SHAPExplainer(model, explainer_type='auto')
        >>> results = explainer.explain(X_test, sample_weight=weights, max_samples=1000)
        >>> importance_df = explainer.get_feature_importance()
        >>> explainer.plot_summary(save_path='shap_summary.png')
    """
    
    def __init__(
        self,
        model: Any,
        explainer_type: str = "auto",
        feature_names: Optional[List[str]] = None,
        background_samples: int = 100,
        random_state: int = 42,
        **kwargs: Any
    ) -> None:
        """Initialize enhanced SHAP explainer.
        
        Args:
            model: Trained model to explain
            explainer_type: Type of SHAP explainer ('auto', 'tree', 'linear', 'kernel', 'deep')
            feature_names: Names of input features
            background_samples: Number of background samples for kernel explainer
            random_state: Random state for reproducibility
            **kwargs: Additional explainer parameters
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        super().__init__(model, feature_names, **kwargs)
        
        validate_parameter("explainer_type", explainer_type, 
                         valid_values=["auto", "tree", "linear", "kernel", "deep", "permutation"])
        validate_parameter("background_samples", background_samples, min_value=10, max_value=1000)
        
        self.explainer_type = explainer_type
        self.background_samples = background_samples
        self.random_state = random_state
        
        # SHAP components
        self.shap_explainer: Optional[Any] = None
        self.shap_values_: Optional[np.ndarray] = None
        self.expected_value_: Optional[float] = None
        self.background_data_: Optional[pd.DataFrame] = None
        
        # Visualization helper
        self.visualizer = SHAPVisualizer()
        
        logger.info(f"Initialized SHAPExplainer:")
        logger.info(f"  Type: {explainer_type}, Background samples: {background_samples}")

    def _detect_explainer_type(self, X: pd.DataFrame) -> str:
        """Auto-detect the best SHAP explainer type for the model."""
        
        # Tree-based models
        if hasattr(self.model, 'get_booster'):  # XGBoost
            return "tree"
        elif hasattr(self.model, '_Booster'):  # LightGBM  
            return "tree"
        elif hasattr(self.model, 'get_feature_importance'):  # CatBoost
            return "tree"
        elif hasattr(self.model, 'estimators_'):  # Random Forest, etc.
            return "tree"
        
        # Linear models
        elif hasattr(self.model, 'coef_'):
            return "linear"
        
        # Deep learning models
        elif hasattr(self.model, 'layers'):  # Keras/TensorFlow
            return "deep"
        elif hasattr(self.model, 'modules'):  # PyTorch
            return "deep"
        
        # Default to kernel explainer
        else:
            logger.info(f"Unknown model type {type(self.model)}, using kernel explainer")
            return "kernel"

    def _create_explainer(self, X: pd.DataFrame) -> Any:
        """Create appropriate SHAP explainer for the model."""
        
        explainer_type = (self.explainer_type if self.explainer_type != "auto" 
                         else self._detect_explainer_type(X))
        
        logger.info(f"Creating {explainer_type} SHAP explainer")
        
        try:
            if explainer_type == "tree":
                # Tree explainer for tree-based models
                explainer = shap.TreeExplainer(
                    self.model,
                    model_output='probability' if hasattr(self.model, 'predict_proba') else 'raw'
                )
                
            elif explainer_type == "linear":
                # Linear explainer for linear models
                explainer = shap.LinearExplainer(self.model, X)
                
            elif explainer_type == "kernel":
                # Kernel explainer (model-agnostic)
                # Create background data
                if len(X) > self.background_samples:
                    np.random.seed(self.random_state)
                    background_indices = np.random.choice(
                        len(X), size=self.background_samples, replace=False
                    )
                    background = X.iloc[background_indices]
                else:
                    background = X
                
                self.background_data_ = background
                
                # Use predict_proba if available
                if hasattr(self.model, 'predict_proba'):
                    def predict_fn(x):
                        return self.model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1]
                else:
                    def predict_fn(x):
                        return self.model.predict(pd.DataFrame(x, columns=X.columns))
                
                explainer = shap.KernelExplainer(predict_fn, background)
                
            elif explainer_type == "deep":
                # Deep explainer for neural networks
                if len(X) > self.background_samples:
                    background = X.sample(n=self.background_samples, random_state=self.random_state)
                else:
                    background = X
                
                explainer = shap.DeepExplainer(self.model, background.values)
                
            elif explainer_type == "permutation":
                # Permutation explainer
                explainer = shap.PermutationExplainer(
                    lambda x: self.model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1]
                    if hasattr(self.model, 'predict_proba') 
                    else self.model.predict(pd.DataFrame(x, columns=X.columns)),
                    X
                )
                
            else:
                raise ConfigurationError(f"Unknown explainer type: {explainer_type}")
            
            self.shap_explainer = explainer
            logger.info(f"✅ {explainer_type.title()} SHAP explainer created successfully")
            
            return explainer
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                f"Failed to create {explainer_type} SHAP explainer",
                error_code="EXPLAINER_CREATION_FAILED",
                context=create_error_context(explainer_type=explainer_type, model_type=type(self.model).__name__)
            )

    @timer(name="shap_explanation")
    def explain(
        self,
        X: pd.DataFrame,
        sample_weight: Optional[np.ndarray] = None,
        max_samples: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Compute SHAP explanations with comprehensive error handling.
        
        Args:
            X: Features to explain
            sample_weight: Sample weights (used for sampling if max_samples specified)
            max_samples: Maximum samples to compute (for performance)
            timeout: Optional timeout in seconds
            **kwargs: Additional explanation parameters
            
        Returns:
            Dictionary with SHAP explanation results
            
        Raises:
            ExplainabilityError: If SHAP computation fails
        """
        logger.info(f"🔍 Computing SHAP explanations:")
        logger.info(f"   Samples: {len(X)}, Features: {X.shape[1]}")
        
        try:
            # Validate inputs
            self.validate_features(X)
            
            if sample_weight is not None and len(sample_weight) != len(X):
                raise ExplainabilityError("sample_weight length must match X")
            
            # Store feature names if not already set
            if self.feature_names is None:
                self.feature_names = list(X.columns)
            
            with timed_operation("shap_computation", timeout=timeout) as timing:
                # Sample data if requested for performance
                X_sample, sample_indices = self._sample_data_for_shap(
                    X, sample_weight, max_samples
                )
                
                # Create explainer if needed
                if self.shap_explainer is None:
                    self._create_explainer(X_sample)
                
                # Compute SHAP values
                self.shap_values_ = self._compute_shap_values_safe(X_sample)
                
                # Handle expected value
                self.expected_value_ = self._extract_expected_value()
                
                # Create feature importance
                feature_importance = self._compute_feature_importance()
            
            computation_time = timing['duration']
            
            logger.info(f"✅ SHAP computation completed:")
            logger.info(f"   Computed for: {len(X_sample)} samples")
            logger.info(f"   Duration: {computation_time:.2f}s")
            logger.info(f"   Expected value: {self.expected_value_:.4f}")
            
            return {
                'shap_values': self.shap_values_,
                'expected_value': self.expected_value_,
                'feature_importance': feature_importance,
                'feature_names': self.feature_names,
                'computation_time': computation_time,
                'n_samples_computed': len(X_sample),
                'sample_indices': sample_indices
            }
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                "SHAP explanation computation failed",
                error_code="SHAP_COMPUTATION_FAILED",
                context=create_error_context(
                    n_samples=len(X),
                    n_features=X.shape[1],
                    explainer_type=self.explainer_type,
                    max_samples=max_samples
                )
            )

    def _sample_data_for_shap(
        self,
        X: pd.DataFrame,
        sample_weight: Optional[np.ndarray],
        max_samples: Optional[int]
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Sample data for SHAP computation with proper weighting."""
        
        if max_samples is None or len(X) <= max_samples:
            return X, None
        
        logger.info(f"Sampling {max_samples} from {len(X)} samples for SHAP computation")
        
        np.random.seed(self.random_state)
        
        if sample_weight is not None:
            # Weighted sampling
            try:
                sample_probs = sample_weight / sample_weight.sum()
                indices = np.random.choice(
                    len(X), size=max_samples, replace=False, p=sample_probs
                )
            except Exception as e:
                logger.warning(f"Weighted sampling failed: {e}, using random sampling")
                indices = np.random.choice(len(X), size=max_samples, replace=False)
        else:
            # Random sampling
            indices = np.random.choice(len(X), size=max_samples, replace=False)
        
        return X.iloc[indices], indices

    def _compute_shap_values_safe(self, X: pd.DataFrame) -> np.ndarray:
        """Safely compute SHAP values with error handling."""
        
        try:
            # Different explainers have different interfaces
            if hasattr(self.shap_explainer, 'shap_values'):
                # TreeExplainer, LinearExplainer
                shap_values = self.shap_explainer.shap_values(X)
            else:
                # KernelExplainer, PermutationExplainer (newer interface)
                shap_values = self.shap_explainer(X)
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
            
            # Handle multi-output case (binary classification)
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    # Binary classification - take positive class
                    shap_values = shap_values[1]
                    logger.debug("Using positive class SHAP values for binary classification")
                else:
                    # Multi-class - this is more complex, for now use first class
                    logger.warning(f"Multi-class SHAP values ({len(shap_values)} classes), using first class")
                    shap_values = shap_values[0]
            
            # Validate shape
            expected_shape = (len(X), len(self.feature_names))
            if shap_values.shape != expected_shape:
                logger.warning(f"SHAP values shape {shap_values.shape} != expected {expected_shape}")
            
            validate_explanation_data(shap_values, expected_shape)
            
            return shap_values
            
        except Exception as e:
            raise ExplainabilityError(f"SHAP values computation failed: {e}")

    def _extract_expected_value(self) -> float:
        """Extract expected value from SHAP explainer."""
        
        try:
            expected_value = self.shap_explainer.expected_value
            
            # Handle multi-output case
            if isinstance(expected_value, (list, np.ndarray)):
                if len(expected_value) == 2:
                    # Binary classification - take positive class
                    expected_value = expected_value[1]
                else:
                    # Multi-class - use first class
                    expected_value = expected_value[0]
            
            return float(expected_value)
            
        except Exception as e:
            logger.warning(f"Could not extract expected value: {e}")
            return 0.0

    def _compute_feature_importance(self) -> pd.DataFrame:
        """Compute feature importance from SHAP values."""
        
        if self.shap_values_ is None:
            raise ExplainabilityError("No SHAP values available")
        
        # Mean absolute SHAP values
        importance_values = np.abs(self.shap_values_).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_values,
            'importance_std': np.abs(self.shap_values_).std(axis=0),
            'mean_shap': self.shap_values_.mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Add relative importance
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['relative_importance'] = importance_df['importance'] / total_importance
        else:
            importance_df['relative_importance'] = 0.0
        
        logger.debug(f"Computed feature importance for {len(importance_df)} features")
        
        return importance_df

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from SHAP values.
        
        Returns:
            DataFrame with feature importance scores
            
        Raises:
            ExplainabilityError: If no explanations have been computed
        """
        if self.shap_values_ is None:
            raise ExplainabilityError(
                "No SHAP explanations available. Run explain() first.",
                error_code="NO_EXPLANATIONS_AVAILABLE"
            )
        
        return self._compute_feature_importance()

    def get_shap_values(self) -> np.ndarray:
        """Get computed SHAP values.
        
        Returns:
            SHAP values array
            
        Raises:
            ExplainabilityError: If no SHAP values have been computed
        """
        if self.shap_values_ is None:
            raise ExplainabilityError("No SHAP values available. Run explain() first.")
        
        return self.shap_values_.copy()

    def get_expected_value(self) -> float:
        """Get expected value from SHAP explainer.
        
        Returns:
            Expected value
        """
        if self.expected_value_ is None:
            raise ExplainabilityError("No expected value available. Run explain() first.")
        
        return self.expected_value_

    # Plotting methods (delegated to visualizer)
    def plot_summary(
        self,
        plot_type: str = "dot",
        max_display: int = 20,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Create SHAP summary plot.
        
        Args:
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum features to display
            save_path: Optional path to save plot
            **kwargs: Additional plotting parameters
            
        Returns:
            Plot figure
        """
        return self.visualizer.plot_summary(
            shap_values=self.shap_values_,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            save_path=save_path,
            **kwargs
        )

    def plot_dependence(
        self,
        feature: Union[str, int],
        interaction_feature: Optional[Union[str, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Create SHAP dependence plot.
        
        Args:
            feature: Feature to plot
            interaction_feature: Optional interaction feature
            save_path: Optional path to save plot
            **kwargs: Additional plotting parameters
            
        Returns:
            Plot figure
        """
        return self.visualizer.plot_dependence(
            shap_values=self.shap_values_,
            feature=feature,
            feature_names=self.feature_names,
            interaction_feature=interaction_feature,
            save_path=save_path,
            **kwargs
        )

    def plot_force(
        self,
        instance_idx: int,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Create SHAP force plot for individual prediction.
        
        Args:
            instance_idx: Index of instance to explain
            save_path: Optional path to save plot (HTML format)
            **kwargs: Additional plotting parameters
            
        Returns:
            Force plot object
        """
        return self.visualizer.plot_force(
            shap_values=self.shap_values_,
            expected_value=self.expected_value_,
            instance_idx=instance_idx,
            feature_names=self.feature_names,
            save_path=save_path,
            **kwargs
        )


class SHAPVisualizer(BaseVisualizer):
    """Visualization helper for SHAP explanations."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize SHAP visualizer."""
        super().__init__(**kwargs)
    
    def plot_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        plot_type: str = "dot",
        max_display: int = 20,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Create SHAP summary plot."""
        
        if shap_values is None:
            raise ExplainabilityError("No SHAP values provided for plotting")
        
        logger.info(f"Creating SHAP summary plot (type: {plot_type})")
        
        try:
            plt.figure(figsize=self.figsize)
            
            if plot_type == "dot":
                shap.summary_plot(
                    shap_values,
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False,
                    **kwargs
                )
            elif plot_type == "bar":
                shap.summary_plot(
                    shap_values,
                    feature_names=feature_names,
                    plot_type="bar",
                    max_display=max_display,
                    show=False,
                    **kwargs
                )
            elif plot_type == "violin":
                shap.summary_plot(
                    shap_values,
                    feature_names=feature_names,
                    plot_type="violin",
                    max_display=max_display,
                    show=False,
                    **kwargs
                )
            else:
                raise ConfigurationError(f"Unknown plot type: {plot_type}")
            
            plt.title(f"SHAP Summary Plot ({plot_type.title()})", fontsize=14, pad=20)
            plt.tight_layout()
            
            fig = plt.gcf()
            
            if save_path:
                self.save_plot(fig, save_path)
            
            return fig
            
        except Exception as e:
            plt.close()
            handle_and_reraise(
                e, ExplainabilityError,
                f"Failed to create SHAP summary plot",
                error_code="PLOT_CREATION_FAILED"
            )

    def plot_dependence(
        self,
        shap_values: np.ndarray,
        feature: Union[str, int],
        feature_names: List[str],
        interaction_feature: Optional[Union[str, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Create SHAP dependence plot."""
        
        # Convert feature name to index if needed
        if isinstance(feature, str):
            if feature not in feature_names:
                raise ExplainabilityError(f"Feature '{feature}' not found")
            feature_idx = feature_names.index(feature)
            feature_name = feature
        else:
            feature_idx = feature
            feature_name = feature_names[feature_idx]
        
        logger.info(f"Creating SHAP dependence plot for: {feature_name}")
        
        try:
            plt.figure(figsize=self.figsize)
            
            shap.dependence_plot(
                feature_idx,
                shap_values,
                feature_names=feature_names,
                interaction_index=interaction_feature,
                show=False,
                **kwargs
            )
            
            plt.title(f"SHAP Dependence Plot: {feature_name}", fontsize=14, pad=20)
            plt.tight_layout()
            
            fig = plt.gcf()
            
            if save_path:
                self.save_plot(fig, save_path)
            
            return fig
            
        except Exception as e:
            plt.close()
            handle_and_reraise(
                e, ExplainabilityError,
                f"Failed to create SHAP dependence plot for {feature_name}",
                error_code="PLOT_CREATION_FAILED"
            )

    def plot_force(
        self,
        shap_values: np.ndarray,
        expected_value: float,
        instance_idx: int,
        feature_names: List[str],
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Create SHAP force plot."""
        
        if instance_idx >= len(shap_values):
            raise ExplainabilityError(f"Instance index {instance_idx} out of range")
        
        logger.info(f"Creating SHAP force plot for instance {instance_idx}")
        
        try:
            force_plot = shap.force_plot(
                expected_value,
                shap_values[instance_idx],
                feature_names=feature_names,
                show=False,
                **kwargs
            )
            
            if save_path:
                shap.save_html(str(save_path), force_plot)
                logger.info(f"SHAP force plot saved to {save_path}")
            
            return force_plot
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                f"Failed to create SHAP force plot for instance {instance_idx}",
                error_code="PLOT_CREATION_FAILED"
            )

    def plot(
        self,
        data: Any,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Generic plot method (delegates to summary plot)."""
        if isinstance(data, dict) and 'shap_values' in data:
            return self.plot_summary(
                shap_values=data['shap_values'],
                feature_names=data['feature_names'],
                save_path=save_path,
                **kwargs
            )
        else:
            raise ExplainabilityError("Invalid data format for SHAP plotting")


# Convenience functions
def quick_shap_analysis(
    model: Any,
    X: pd.DataFrame,
    sample_weight: Optional[np.ndarray] = None,
    max_samples: int = 1000,
    save_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Quick SHAP analysis with default settings.
    
    Args:
        model: Trained model
        X: Features to explain
        sample_weight: Optional sample weights
        max_samples: Maximum samples for computation
        save_dir: Optional directory to save plots
        **kwargs: Additional explainer parameters
        
    Returns:
        Dictionary with SHAP analysis results
        
    Example:
        >>> results = quick_shap_analysis(model, X_test, sample_weight=weights)
        >>> print(f"Top feature: {results['feature_importance'].iloc[0]['feature']}")
        >>> print(f"Expected value: {results['expected_value']:.4f}")
    """
    logger.info(f"🚀 Running quick SHAP analysis on {len(X)} samples")
    
    # Initialize explainer
    explainer = SHAPExplainer(model, **kwargs)
    
    # Compute explanations
    results = explainer.explain(
        X=X,
        sample_weight=sample_weight,
        max_samples=max_samples
    )
    
    # Create visualizations if save directory provided
    if save_dir:
        save_path = Path(save_dir)
        ensure_dir(save_path)
        
        # Summary plots
        explainer.plot_summary(
            plot_type="dot",
            save_path=save_path / "shap_summary_dot.png"
        )
        
        explainer.plot_summary(
            plot_type="bar", 
            save_path=save_path / "shap_summary_bar.png"
        )
        
        # Feature importance CSV
        results['feature_importance'].to_csv(
            save_path / "feature_importance.csv", 
            index=False
        )
        
        # Dependence plots for top 3 features
        top_features = results['feature_importance'].head(3)['feature'].tolist()
        for feature in top_features:
            explainer.plot_dependence(
                feature=feature,
                save_path=save_path / f"dependence_{feature}.png"
            )
        
        logger.info(f"SHAP analysis saved to {save_path}")
    
    return results


# Export key classes and functions
__all__ = [
    'SHAPExplainer',
    'SHAPVisualizer', 
    'quick_shap_analysis'
]