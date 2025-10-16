# tree_models/utils/exceptions.py
"""Custom exception hierarchy for tree_models package.

This module defines a comprehensive exception hierarchy that provides
specific error types for different failure modes in the ML pipeline.
"""

from typing import Any, Optional, Dict, List


class TreeModelsError(Exception):
    """Base exception for all tree_models package errors.
    
    This is the root exception class that all other package-specific
    exceptions inherit from. It provides common functionality for
    error context and debugging information.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize TreeModelsError.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            context: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return string representation of error."""
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (Context: {context_str})"
        return base_msg


class ConfigurationError(TreeModelsError):
    """Raised when configuration is invalid or incomplete.
    
    This exception is raised for issues with:
    - Invalid parameter values
    - Missing required configuration
    - Conflicting configuration options
    - Schema validation failures
    """
    pass


class DataValidationError(TreeModelsError):
    """Raised when input data fails validation checks.
    
    This exception is raised for issues with:
    - Invalid data types or shapes
    - Missing required columns
    - Data quality issues (NaN, infinite values)
    - Inconsistent train/test data schemas
    """
    pass


class ModelTrainingError(TreeModelsError):
    """Raised when model training fails.
    
    This exception is raised for issues during:
    - Model fitting process
    - Hyperparameter optimization
    - Cross-validation failures
    - Memory or computational constraints
    """
    pass


class ModelEvaluationError(TreeModelsError):
    """Raised when model evaluation fails.
    
    This exception is raised for issues during:
    - Prediction generation
    - Metric computation
    - Evaluation data preprocessing
    - Performance benchmark failures
    """
    pass


class ExplainabilityError(TreeModelsError):
    """Raised when model explanation fails.
    
    This exception is raised for issues with:
    - SHAP value computation
    - Feature importance calculation
    - Partial dependence plots
    - Reason code generation
    """
    pass


class FeatureSelectionError(TreeModelsError):
    """Raised when feature selection fails.
    
    This exception is raised for issues with:
    - Feature selection algorithms
    - Feature importance ranking
    - Feature subset validation
    - Selection criterion computation
    """
    pass


class RobustnessTestError(TreeModelsError):
    """Raised when robustness testing fails.
    
    This exception is raised for issues with:
    - Stability analysis
    - Sensitivity testing
    - Perturbation analysis
    - Multi-seed validation
    """
    pass


class TrackingError(TreeModelsError):
    """Raised when experiment tracking fails.
    
    This exception is raised for issues with:
    - MLflow connection
    - Metric logging
    - Artifact storage
    - Experiment metadata
    """
    pass


class FileOperationError(TreeModelsError):
    """Raised when file I/O operations fail.
    
    This exception is raised for issues with:
    - File reading/writing
    - Directory creation
    - Path validation
    - Serialization/deserialization
    """
    pass


class PerformanceError(TreeModelsError):
    """Raised when performance constraints are violated.
    
    This exception is raised for issues with:
    - Memory usage exceeding limits
    - Execution time exceeding timeouts
    - Resource allocation failures
    - Scalability constraint violations
    """
    pass


# Utility functions for error handling
def handle_and_reraise(
    exception: Exception,
    error_class: type,
    message: str,
    error_code: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Handle an exception and re-raise as a tree_models exception.
    
    This utility function standardizes exception handling throughout
    the package by converting external exceptions into our custom
    exception hierarchy while preserving the original traceback.
    
    Args:
        exception: Original exception that was caught
        error_class: TreeModelsError subclass to raise
        message: Custom error message
        error_code: Optional error code
        context: Optional error context
        
    Raises:
        error_class: The specified tree_models exception
    """
    if context is None:
        context = {}
    
    context["original_error"] = str(exception)
    context["original_error_type"] = type(exception).__name__
    
    raise error_class(message, error_code, context) from exception


def validate_parameter(
    param_name: str,
    param_value: Any,
    valid_values: Optional[List[Any]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    required: bool = False
) -> None:
    """Validate a parameter value and raise ConfigurationError if invalid.
    
    Args:
        param_name: Name of the parameter being validated
        param_value: Value to validate
        valid_values: List of valid values (if applicable)
        min_value: Minimum allowed value (for numeric parameters)
        max_value: Maximum allowed value (for numeric parameters)
        required: Whether the parameter is required (cannot be None)
        
    Raises:
        ConfigurationError: If validation fails
    """
    if required and param_value is None:
        raise ConfigurationError(
            f"Parameter '{param_name}' is required but was not provided",
            error_code="PARAM_REQUIRED",
            context={"parameter": param_name}
        )
    
    if param_value is None:
        return  # Optional parameter not provided
    
    if valid_values is not None and param_value not in valid_values:
        raise ConfigurationError(
            f"Parameter '{param_name}' must be one of {valid_values}, got {param_value}",
            error_code="PARAM_INVALID_VALUE",
            context={"parameter": param_name, "value": param_value, "valid_values": valid_values}
        )
    
    if min_value is not None and param_value < min_value:
        raise ConfigurationError(
            f"Parameter '{param_name}' must be >= {min_value}, got {param_value}",
            error_code="PARAM_TOO_SMALL",
            context={"parameter": param_name, "value": param_value, "min_value": min_value}
        )
    
    if max_value is not None and param_value > max_value:
        raise ConfigurationError(
            f"Parameter '{param_name}' must be <= {max_value}, got {param_value}",
            error_code="PARAM_TOO_LARGE", 
            context={"parameter": param_name, "value": param_value, "max_value": max_value}
        )


def create_error_context(**kwargs: Any) -> Dict[str, Any]:
    """Create an error context dictionary with standardized keys.
    
    Args:
        **kwargs: Key-value pairs to include in context
        
    Returns:
        Dictionary with error context information
    """
    context = {}
    for key, value in kwargs.items():
        # Convert complex objects to strings for serialization safety
        if hasattr(value, '__dict__') or hasattr(value, '__slots__'):
            context[key] = str(value)
        else:
            context[key] = value
    
    return context