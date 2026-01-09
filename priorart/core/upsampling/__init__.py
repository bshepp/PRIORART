"""
PRIORART Upsampling Methods

Domain-agnostic interpolation/upsampling for 2D scalar fields.
Each method takes input data and scale factor, returns upsampled data.
"""

from .registry import (
    register_upsampling,
    get_upsampling,
    list_upsamplings,
    run_upsampling,
    get_all_methods_info,
    UPSAMPLING_REGISTRY,
    UpsamplingMethod
)

# Import methods to register them
from . import methods
from . import methods_extended

__all__ = [
    'register_upsampling',
    'get_upsampling',
    'list_upsamplings',
    'run_upsampling',
    'get_all_methods_info',
    'UPSAMPLING_REGISTRY',
    'UpsamplingMethod'
]

