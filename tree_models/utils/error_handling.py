"""Standardized error handling with context and recovery strategies.

This module provides consistent error handling patterns across all modules
with structured context information and recovery mechanisms.
"""

from contextlib import contextmanager
from typing import Any, Dict, Optional, Type, Union, Callable
import traceback
import sys
from dataclasses import dataclass, field
from datetime import datetime

from .logger import get_logger
from .exceptions import TreeModelsError, ModelTrainingError, ModelEvaluationError, ConfigurationError

logger = get_logger(__name__)


@dataclass
class ErrorContext:
    """Structured error context information.
    
    Provides detailed context about where and why an error occurred,
    enabling better debugging and error recovery.
    """
    operation: str
    component: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_data: Optional[Dict[str, Any]] = None
    system_info: Optional[Dict[str, Any]] = None
    recovery_suggestions: Optional[list] = None
    
    def __post_init__(self) -> None:
        """Initialize system info if not provided."""
        if self.system_info is None:
            self.system_info = {
                'python_version': sys.version_info[:3],
                'platform': sys.platform
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'operation': self.operation,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'user_data': self.user_data or {},
            'system_info': self.system_info or {},
            'recovery_suggestions': self.recovery_suggestions or []
        }


class ErrorHandler:
    """Centralized error handling with context and recovery.
    
    Provides consistent error handling patterns with structured
    context information and recovery strategies.
    """
    
    def __init__(self, component_name: str) -> None:
        """Initialize error handler for a specific component.
        
        Args:
            component_name: Name of the component using this handler
        """
        self.component_name = component_name
        self.logger = get_logger(f"{__name__}.{component_name}")
    
    @contextmanager
    def operation_context(
        self,
        operation_name: str,
        user_data: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[list] = None,
        reraise_as: Optional[Type[Exception]] = None
    ):
        """Context manager for standardized operation error handling.
        
        Args:
            operation_name: Name of the operation being performed
            user_data: Additional context data
            recovery_suggestions: List of suggested recovery actions
            reraise_as: Exception type to reraise as (if different)
            
        Example:
            >>> handler = ErrorHandler('ModelTrainer')
            >>> with handler.operation_context('train_model', {'model_type': 'xgboost'}):
            ...     # operation that might fail
            ...     model.fit(X, y)
        """
        context = ErrorContext(
            operation=operation_name,
            component=self.component_name,
            user_data=user_data,
            recovery_suggestions=recovery_suggestions
        )
        
        self.logger.info(f"Starting operation: {operation_name}")
        
        try:
            yield context
            self.logger.info(f"Completed operation: {operation_name}")
            
        except Exception as e:
            # Log detailed error information
            self._log_error(e, context)
            
            # Determine appropriate exception type
            exception_class = reraise_as or self._determine_exception_type(e, operation_name)
            
            # Create enhanced exception with context
            enhanced_exception = self._create_enhanced_exception(
                e, exception_class, context
            )
            
            raise enhanced_exception from e
    
    def handle_validation_error(
        self,
        error_message: str,
        validation_target: str,
        user_data: Optional[Dict[str, Any]] = None
    ) -> ConfigurationError:
        """Handle validation errors with context.
        
        Args:
            error_message: Description of validation failure
            validation_target: What was being validated
            user_data: Additional context data
            
        Returns:
            ConfigurationError with enhanced context
        """
        context = ErrorContext(
            operation=f"validate_{validation_target}",
            component=self.component_name,
            user_data=user_data,
            recovery_suggestions=[
                "Check input data format and types",
                "Verify configuration parameters",
                "Review documentation for requirements"
            ]
        )
        
        self.logger.error(f"Validation failed for {validation_target}: {error_message}")
        
        return ConfigurationError(
            f"Validation failed for {validation_target}: {error_message}",
            context=context.to_dict()
        )
    
    def handle_data_error(
        self,
        error_message: str,
        data_info: Optional[Dict[str, Any]] = None
    ) -> ConfigurationError:
        """Handle data-related errors with context.
        
        Args:
            error_message: Description of data error
            data_info: Information about the problematic data
            
        Returns:
            ConfigurationError with data context
        """
        suggestions = [
            "Check for missing values in data",
            "Verify data types are correct",
            "Ensure data shape matches expectations",
            "Check for duplicate or invalid entries"
        ]
        
        if data_info and 'shape' in data_info:
            suggestions.append(f"Current data shape: {data_info['shape']}")
        
        context = ErrorContext(
            operation="data_validation",
            component=self.component_name,
            user_data=data_info,
            recovery_suggestions=suggestions
        )
        
        self.logger.error(f"Data error: {error_message}")
        
        return ConfigurationError(
            f"Data error: {error_message}",
            context=context.to_dict()
        )
    
    def _log_error(
        self, 
        exception: Exception, 
        context: ErrorContext
    ) -> None:
        """Log detailed error information.
        
        Args:
            exception: The caught exception
            context: Error context information
        """
        self.logger.error(
            f"Operation '{context.operation}' failed in {context.component}",
            extra={
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'context': context.to_dict(),
                'traceback': traceback.format_exc()
            }
        )
    
    def _determine_exception_type(
        self, 
        original_exception: Exception, 
        operation_name: str
    ) -> Type[Exception]:
        """Determine appropriate exception type based on operation and original exception.
        
        Args:
            original_exception: The original exception
            operation_name: Name of the operation that failed
            
        Returns:
            Appropriate exception class
        """
        # Map operation patterns to exception types
        if 'train' in operation_name.lower():
            return ModelTrainingError
        elif 'evaluat' in operation_name.lower() or 'predict' in operation_name.lower():
            return ModelEvaluationError
        elif 'validat' in operation_name.lower() or 'config' in operation_name.lower():
            return ConfigurationError
        else:
            # Default to TreeModelsError for unknown operations
            return TreeModelsError
    
    def _create_enhanced_exception(
        self,
        original_exception: Exception,
        exception_class: Type[Exception],
        context: ErrorContext
    ) -> Exception:
        """Create enhanced exception with context information.
        
        Args:
            original_exception: The original exception
            exception_class: Type of exception to create
            context: Error context information
            
        Returns:
            Enhanced exception instance
        """
        message = f"{context.operation} failed in {context.component}: {original_exception}"
        
        # Add recovery suggestions to message if available
        if context.recovery_suggestions:
            message += f"\n\nRecovery suggestions:\n"
            for i, suggestion in enumerate(context.recovery_suggestions, 1):
                message += f"  {i}. {suggestion}\n"
        
        # Create exception with context if supported
        if hasattr(exception_class, '__init__'):
            try:
                # Try to create with context parameter
                return exception_class(message, context=context.to_dict())
            except TypeError:
                # Fallback to basic message
                return exception_class(message)
        
        return exception_class(message)


# Convenience functions for common error handling patterns
def model_operation_context(
    operation_name: str,
    component_name: str = "ModelOperation",
    **kwargs: Any
):
    """Context manager for model operations.
    
    Args:
        operation_name: Name of the model operation
        component_name: Name of the component
        **kwargs: Additional context parameters
    
    Example:
        >>> with model_operation_context('hyperparameter_tuning', model_type='xgboost'):
        ...     # tuning code here
    """
    handler = ErrorHandler(component_name)
    return handler.operation_context(operation_name, **kwargs)


def data_operation_context(
    operation_name: str,
    component_name: str = "DataOperation", 
    **kwargs: Any
):
    """Context manager for data operations.
    
    Args:
        operation_name: Name of the data operation
        component_name: Name of the component
        **kwargs: Additional context parameters
    """
    handler = ErrorHandler(component_name)
    recovery_suggestions = [
        "Check data format and types",
        "Verify data is not empty",
        "Check for missing or invalid values"
    ]
    kwargs.setdefault('recovery_suggestions', recovery_suggestions)
    return handler.operation_context(operation_name, **kwargs)


def config_operation_context(
    operation_name: str,
    component_name: str = "Configuration",
    **kwargs: Any
):
    """Context manager for configuration operations.
    
    Args:
        operation_name: Name of the configuration operation
        component_name: Name of the component
        **kwargs: Additional context parameters
    """
    handler = ErrorHandler(component_name)
    recovery_suggestions = [
        "Check configuration file syntax",
        "Verify all required parameters are provided",
        "Check parameter types and ranges",
        "Review documentation for valid options"
    ]
    kwargs.setdefault('recovery_suggestions', recovery_suggestions)
    kwargs.setdefault('reraise_as', ConfigurationError)
    return handler.operation_context(operation_name, **kwargs)


# Decorator for automatic error handling
def handle_errors(
    operation_name: Optional[str] = None,
    component_name: Optional[str] = None,
    recovery_suggestions: Optional[list] = None
):
    """Decorator for automatic error handling.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        component_name: Name of the component (defaults to class name)
        recovery_suggestions: List of recovery suggestions
        
    Example:
        >>> class ModelTrainer:
        ...     @handle_errors(recovery_suggestions=['Check model parameters'])
        ...     def train_model(self, X, y):
        ...         # training code
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Determine operation name
            op_name = operation_name or func.__name__
            
            # Determine component name
            comp_name = component_name
            if comp_name is None and args and hasattr(args[0], '__class__'):
                comp_name = args[0].__class__.__name__
            comp_name = comp_name or "UnknownComponent"
            
            # Create error handler
            handler = ErrorHandler(comp_name)
            
            # Execute with error handling
            with handler.operation_context(
                op_name,
                recovery_suggestions=recovery_suggestions
            ):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator