"""
2D Scalar Field Domain Plugin

Applies to any gridded 2D data: temperature, pressure, density,
elevation, imagery bands, medical imaging, etc.

This is the generic domain that uses all core decomposition and
upsampling methods. Other domains (terrain, astronomy, etc.) may
provide more specialized pipelines.
"""

from typing import List, Dict, Any, Callable
import numpy as np

from priorart.core.plugin import (
    DomainPlugin,
    Pipeline,
    MethodCategory,
    Method,
)
from priorart.core.decomposition import (
    DECOMPOSITION_REGISTRY,
    run_decomposition,
)
from priorart.core.upsampling import (
    UPSAMPLING_REGISTRY,
    run_upsampling,
)

# Import generators
from .generators import (
    generate_gaussian_blobs,
    generate_sine_waves,
    generate_perlin_noise,
    generate_step_functions,
    generate_fractal_terrain,
    generate_gradient_field,
    generate_mixed_features,
)


def _wrap_decomposition(name: str, method_info) -> Method:
    """Wrap a decomposition method as a Method object."""
    def decomp_func(data: np.ndarray, **kwargs) -> tuple:
        return run_decomposition(name, data, kwargs if kwargs else None)
    
    return Method(
        name=name,
        func=decomp_func,
        category='decomposition',
        default_params=dict(method_info.default_params),
        param_ranges=dict(method_info.param_ranges),
        preserves=method_info.preserves,
        destroys=method_info.destroys,
        description=method_info.description,
    )


def _wrap_upsampling(name: str, method_info) -> Method:
    """Wrap an upsampling method as a Method object."""
    def upsamp_func(data: np.ndarray, **kwargs) -> np.ndarray:
        # For decomposition output, we get (trend, residual)
        # Apply upsampling to both
        if isinstance(data, tuple):
            trend, residual = data
            scale = kwargs.get('scale', 2)
            params = {k: v for k, v in kwargs.items() if k != 'scale'}
            trend_up = run_upsampling(name, trend, scale, params if params else None)
            residual_up = run_upsampling(name, residual, scale, params if params else None)
            return (trend_up, residual_up)
        else:
            scale = kwargs.get('scale', 2)
            params = {k: v for k, v in kwargs.items() if k != 'scale'}
            return run_upsampling(name, data, scale, params if params else None)
    
    return Method(
        name=name,
        func=upsamp_func,
        category='upsampling',
        default_params=dict(method_info.default_params),
        param_ranges=dict(method_info.param_ranges),
        preserves=method_info.preserves,
        destroys=getattr(method_info, 'introduces', ''),
        description=method_info.description,
    )


def _build_decomposition_category() -> MethodCategory:
    """Build the decomposition method category from registry."""
    category = MethodCategory(
        name='decomposition',
        description='Signal decomposition methods that separate trend from residual'
    )
    
    for name, method_info in DECOMPOSITION_REGISTRY.items():
        method = _wrap_decomposition(name, method_info)
        category.register(method)
    
    return category


def _build_upsampling_category() -> MethodCategory:
    """Build the upsampling method category from registry."""
    category = MethodCategory(
        name='upsampling',
        description='Interpolation methods for increasing spatial resolution'
    )
    
    for name, method_info in UPSAMPLING_REGISTRY.items():
        method = _wrap_upsampling(name, method_info)
        category.register(method)
    
    return category


class Scalar2DPlugin(DomainPlugin):
    """Generic 2D scalar field domain plugin.
    
    This domain provides decomposition and upsampling pipelines that
    work on any 2D scalar field. It serves as the base for more
    specialized domains like terrain, astronomy, etc.
    
    Use this domain when:
    - Your data is a generic 2D array (not domain-specific)
    - You want to explore all available methods
    - You're prototyping before creating a specialized domain
    
    Pipelines:
    - decomp_upsample: Full decomposition → upsampling pipeline
    - decomposition: Just the decomposition stage
    - upsampling: Just the upsampling stage
    """
    
    @property
    def name(self) -> str:
        return "scalar2d"
    
    @property
    def description(self) -> str:
        return "Generic 2D scalar field processing with decomposition and upsampling"
    
    def get_pipelines(self) -> List[Pipeline]:
        """Return available processing pipelines."""
        decomp_category = _build_decomposition_category()
        upsamp_category = _build_upsampling_category()
        
        # Full pipeline: decomposition → upsampling
        full_pipeline = Pipeline(
            name='decomp_upsample',
            description='Decompose into trend/residual, then upsample both components',
            stages=[decomp_category, upsamp_category]
        )
        
        # Decomposition-only pipeline
        decomp_pipeline = Pipeline(
            name='decomposition',
            description='Decompose field into trend and residual components',
            stages=[decomp_category]
        )
        
        # Upsampling-only pipeline
        upsamp_pipeline = Pipeline(
            name='upsampling',
            description='Upsample/interpolate field to higher resolution',
            stages=[upsamp_category]
        )
        
        return [full_pipeline, decomp_pipeline, upsamp_pipeline]
    
    def get_generators(self) -> Dict[str, Callable]:
        """Return synthetic test data generators."""
        return {
            'gaussian_blobs': generate_gaussian_blobs,
            'sine_waves': generate_sine_waves,
            'perlin_noise': generate_perlin_noise,
            'step_functions': generate_step_functions,
            'fractal_terrain': generate_fractal_terrain,
            'gradient_field': generate_gradient_field,
            'mixed_features': generate_mixed_features,
        }


# Plugin instance for auto-discovery
PLUGIN = Scalar2DPlugin()


def get_plugin() -> DomainPlugin:
    """Return the scalar2d plugin instance."""
    return PLUGIN
