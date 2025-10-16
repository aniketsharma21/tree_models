"""Plugin registry for extensibility.

This module provides a thread-safe singleton registry for custom
implementations of framework components.

Example:
    >>> from tree_models.models.registry import plugin_registry
    >>> plugin_registry.register_trainer('custom', MyTrainer)
"""

from .plugin_registry import PluginRegistry, plugin_registry

__all__ = [
    'PluginRegistry',
    'plugin_registry'
]