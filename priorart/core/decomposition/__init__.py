"""
PRIORART Decomposition Methods

Domain-agnostic signal decomposition for 2D scalar fields.
Each method decomposes input into (trend, residual) components.
"""

from .registry import (
    register_decomposition,
    get_decomposition,
    list_decompositions,
    run_decomposition,
    get_all_methods_info,
    DECOMPOSITION_REGISTRY,
    DecompositionMethod
)

# Import methods to register them
from . import methods
from . import methods_extended

__all__ = [
    'register_decomposition',
    'get_decomposition', 
    'list_decompositions',
    'run_decomposition',
    'get_all_methods_info',
    'DECOMPOSITION_REGISTRY',
    'DecompositionMethod'
]

