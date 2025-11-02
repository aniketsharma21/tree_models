"""Tree Models - Model Explainability and Interpretability Components.

This module provides comprehensive model explainability tools including
SHAP integration, business scorecards, reason codes, and partial dependence plots.

Key Components:
- SHAPExplainer: Enhanced SHAP integration with performance optimization
- ScorecardConverter: Business scorecard generation with calibration
- ReasonCodeGenerator: Regulatory-compliant explanations
- PartialDependencePlotter: PD and ICE plots with interactions

Example:
    >>> from tree_models.explainability import SHAPExplainer, ScorecardConverter
    >>> explainer = SHAPExplainer(model, explainer_type='auto')
    >>> converter = ScorecardConverter()
"""

# Core explainability components
from .shap_explainer import SHAPExplainer, quick_shap_analysis
from .scorecard import ScorecardConverter, convert_to_scorecard
from .reason_codes import ReasonCodeGenerator
from .partial_dependence import PartialDependencePlotter

# Base classes
from .base import BaseExplainer

__all__ = [
    # Core implementations
    'SHAPExplainer',
    'ScorecardConverter',
    'ReasonCodeGenerator', 
    'PartialDependencePlotter',
    
    # Quick utility functions
    'quick_shap_analysis',
    'convert_to_scorecard',
    
    # Base classes and results
    'BaseExplainer'
]