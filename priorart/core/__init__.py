"""
PRIORART Core Library

Domain-agnostic infrastructure for systematic exploration and documentation
of mathematical method combinations, establishing public prior art.
"""

__version__ = "0.2.0"

# Plugin architecture
from .plugin import (
    DomainPlugin,
    Pipeline,
    MethodCategory,
    Method,
    method_decorator,
)

# Domain registry
from .registry import (
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

