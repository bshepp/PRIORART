"""
PRIORART - Systematic Prior Art Generation

The single source of truth for feature extraction across all *vector projects.
"""

__version__ = "0.2.0"

# Re-export core components for convenience
from priorart.core import (
    DomainPlugin,
    Pipeline,
    MethodCategory,
    Method,
    method_decorator,
    DomainRegistry,
    get_registry,
    register_domain,
    get_domain,
    list_domains,
    auto_discover_domains,
)

__all__ = [
    '__version__',
    'DomainPlugin',
    'Pipeline',
    'MethodCategory',
    'Method',
    'method_decorator',
    'DomainRegistry',
    'get_registry',
    'register_domain',
    'get_domain',
    'list_domains',
    'auto_discover_domains',
]
