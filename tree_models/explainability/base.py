# tree_models/explainability/base.py
"""Abstract base classes for explainability components.

This module defines the core interfaces and abstract base classes for
all explainability functionality in the tree_models package.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils.exceptions import (
    ExplainabilityError,
    ConfigurationError,
    validate_parameter
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseExplainer(ABC):
    """Abstract base class for all model explainers.
    
    Defines the common interface for different explainability methods
    such as SHAP, LIME, permutation importance, etc.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize base explainer.
        
        Args:
            model: Trained model to explain
            feature_names: Names of input features
            **kwargs: Additional explainer-specific parameters
        """
        self.model = model
        self.feature_names = feature_names
        self.config = kwargs
        self.is_fitted = False
        
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def explain(
        self,
        X: pd.DataFrame,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate explanations for the input data.
        
        Args:
            X: Input features to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            Dictionary containing explanation results
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from explanations.
        
        Returns:
            DataFrame with feature importance scores
        """
        pass
    
    def validate_model(self) -> None:
        """Validate that the model has required methods."""
        required_methods = ['predict']
        
        for method in required_methods:
            if not hasattr(self.model, method):
                raise ConfigurationError(f"Model must have {method} method")
        
        logger.debug("Model validation passed")
    
    def validate_features(self, X: pd.DataFrame) -> None:
        """Validate input features.
        
        Args:
            X: Input features to validate
        """
        if X.empty:
            raise ExplainabilityError("Input features cannot be empty")
        
        if self.feature_names and len(X.columns) != len(self.feature_names):
            raise ExplainabilityError(
                f"Feature count mismatch: expected {len(self.feature_names)}, got {len(X.columns)}"
            )
        
        logger.debug(f"Feature validation passed: {X.shape}")


class BaseVisualizer(ABC):
    """Abstract base class for explanation visualizers.
    
    Handles creation of plots and visualizations for explanations.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300,
        style: str = "default",
        **kwargs: Any
    ) -> None:
        """Initialize base visualizer.
        
        Args:
            figsize: Figure size for plots
            dpi: DPI for saved plots
            style: Plotting style
            **kwargs: Additional visualization parameters
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.config = kwargs
        
        # Set plotting defaults
        try:
            import matplotlib.pyplot as plt
            plt.style.use(style)
        except ImportError:
            logger.warning("Matplotlib not available")
        except Exception:
            logger.warning(f"Could not set style: {style}")
    
    @abstractmethod
    def plot(
        self,
        data: Any,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Create visualization from explanation data.
        
        Args:
            data: Explanation data to visualize
            save_path: Optional path to save plot
            **kwargs: Additional plotting parameters
            
        Returns:
            Plot object or figure
        """
        pass
    
    def save_plot(
        self,
        fig: Any,
        save_path: Union[str, Path],
        **kwargs: Any
    ) -> None:
        """Save plot to file with error handling.
        
        Args:
            fig: Figure or plot object
            save_path: Path to save the plot
            **kwargs: Additional save parameters
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle different plot types
            if hasattr(fig, 'savefig'):  # matplotlib
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', **kwargs)
            elif hasattr(fig, 'write_image'):  # plotly
                fig.write_image(save_path, **kwargs)
            elif hasattr(fig, 'save'):  # other libraries
                fig.save(save_path, **kwargs)
            else:
                logger.warning(f"Unknown plot type, cannot save: {type(fig)}")
                
            logger.info(f"Plot saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save plot to {save_path}: {e}")
            raise ExplainabilityError(f"Plot saving failed: {e}")


class BaseConverter(ABC):
    """Abstract base class for explanation converters.
    
    Converts model predictions or explanations into business-friendly formats.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize base converter.
        
        Args:
            **kwargs: Converter-specific parameters
        """
        self.config = kwargs
        self.is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        data: Any,
        **kwargs: Any
    ) -> 'BaseConverter':
        """Fit the converter on reference data.
        
        Args:
            data: Reference data for fitting
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(
        self,
        data: Any,
        **kwargs: Any
    ) -> Any:
        """Transform data using fitted converter.
        
        Args:
            data: Data to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(
        self,
        data: Any,
        **kwargs: Any
    ) -> Any:
        """Fit converter and transform data in one step.
        
        Args:
            data: Data to fit and transform
            **kwargs: Additional parameters
            
        Returns:
            Transformed data
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    @abstractmethod
    def interpret(
        self,
        transformed_data: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Interpret transformed data in business terms.
        
        Args:
            transformed_data: Data to interpret
            **kwargs: Additional interpretation parameters
            
        Returns:
            Dictionary with interpretation results
        """
        pass


class BaseReasonCodeGenerator(ABC):
    """Abstract base class for reason code generators.
    
    Generates human-readable explanations for individual predictions.
    """
    
    def __init__(
        self,
        max_reasons: int = 5,
        min_impact: float = 0.01,
        **kwargs: Any
    ) -> None:
        """Initialize base reason code generator.
        
        Args:
            max_reasons: Maximum number of reasons per prediction
            min_impact: Minimum impact threshold for inclusion
            **kwargs: Additional generator parameters
        """
        validate_parameter("max_reasons", max_reasons, min_value=1, max_value=20)
        validate_parameter("min_impact", min_impact, min_value=0.0, max_value=1.0)
        
        self.max_reasons = max_reasons
        self.min_impact = min_impact
        self.config = kwargs
        
        logger.debug(f"Initialized reason code generator: max_reasons={max_reasons}")
    
    @abstractmethod
    def generate_reasons(
        self,
        explanation_data: Any,
        instance_data: Any,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Generate reason codes for predictions.
        
        Args:
            explanation_data: Explanation values (e.g., SHAP values)
            instance_data: Original instance data
            **kwargs: Additional generation parameters
            
        Returns:
            List of reason code dictionaries
        """
        pass
    
    @abstractmethod
    def format_reason(
        self,
        feature: str,
        impact: float,
        value: Any,
        **kwargs: Any
    ) -> str:
        """Format a single reason into human-readable text.
        
        Args:
            feature: Feature name
            impact: Impact value
            value: Feature value
            **kwargs: Additional formatting parameters
            
        Returns:
            Human-readable reason text
        """
        pass


# Factory functions for creating explainer instances
def create_explainer(
    explainer_type: str,
    model: Any,
    **kwargs: Any
) -> BaseExplainer:
    """Factory function for creating explainers.
    
    Args:
        explainer_type: Type of explainer to create
        model: Model to explain
        **kwargs: Additional parameters
        
    Returns:
        Initialized explainer instance
    """
    # Import here to avoid circular imports
    from .shap_explainer import SHAPExplainer
    
    explainer_map = {
        'shap': SHAPExplainer,
        # Add more explainer types as they're implemented
    }
    
    if explainer_type not in explainer_map:
        raise ConfigurationError(
            f"Unknown explainer type: {explainer_type}. Available: {list(explainer_map.keys())}"
        )
    
    explainer_class = explainer_map[explainer_type]
    return explainer_class(model, **kwargs)


# Plugin registry for custom explainers
class ExplainerRegistry:
    """Registry for custom explainer implementations."""
    
    _explainers: Dict[str, type] = {}
    _visualizers: Dict[str, type] = {}
    _converters: Dict[str, type] = {}
    
    @classmethod
    def register_explainer(cls, name: str, explainer_class: type) -> None:
        """Register a custom explainer.
        
        Args:
            name: Name for the explainer
            explainer_class: Explainer class (must inherit from BaseExplainer)
        """
        if not issubclass(explainer_class, BaseExplainer):
            raise ConfigurationError(
                "Explainer class must inherit from BaseExplainer"
            )
        
        cls._explainers[name] = explainer_class
        logger.info(f"Registered custom explainer: {name}")
    
    @classmethod
    def get_explainer_class(cls, name: str) -> type:
        """Get registered explainer class by name.
        
        Args:
            name: Explainer name
            
        Returns:
            Explainer class
        """
        if name not in cls._explainers:
            raise ConfigurationError(f"Explainer '{name}' not registered")
        
        return cls._explainers[name]
    
    @classmethod
    def get_available_explainers(cls) -> List[str]:
        """Get list of available explainer names."""
        return list(cls._explainers.keys())


# Utility functions
def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_explanation_data(
    data: Any,
    expected_shape: Optional[Tuple[int, ...]] = None
) -> None:
    """Validate explanation data format.
    
    Args:
        data: Explanation data to validate
        expected_shape: Expected data shape
    """
    if data is None:
        raise ExplainabilityError("Explanation data cannot be None")
    
    if isinstance(data, np.ndarray):
        if expected_shape and data.shape != expected_shape:
            raise ExplainabilityError(
                f"Shape mismatch: expected {expected_shape}, got {data.shape}"
            )
    elif isinstance(data, pd.DataFrame):
        if data.empty:
            raise ExplainabilityError("Explanation DataFrame cannot be empty")
    else:
        logger.warning(f"Unknown explanation data type: {type(data)}")


# Export commonly used types for convenience
__all__ = [
    'BaseExplainer',
    'BaseVisualizer', 
    'BaseConverter',
    'BaseReasonCodeGenerator',
    'create_explainer',
    'ExplainerRegistry',
    'ensure_dir',
    'validate_explanation_data'
]