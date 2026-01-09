"""
Terrain Domain Plugin

Specialized for Digital Elevation Models (DEMs) from LiDAR, 
photogrammetry, or satellite sources.
"""

from typing import List, Dict, Any, Callable
import numpy as np

from priorart.core.plugin import DomainPlugin, Pipeline, MethodCategory, Method
from priorart.core.decomposition import DECOMPOSITION_REGISTRY, run_decomposition
from priorart.core.upsampling import UPSAMPLING_REGISTRY, run_upsampling


def _wrap_decomposition(name, method_info):
    def decomp_func(data, **kwargs):
        return run_decomposition(name, data, kwargs if kwargs else None)
    return Method(
        name=name, func=decomp_func, category='decomposition',
        default_params=dict(method_info.default_params),
        param_ranges=dict(method_info.param_ranges),
        preserves=method_info.preserves, destroys=method_info.destroys,
        description=method_info.description,
    )


def _wrap_upsampling(name, method_info):
    def upsamp_func(data, **kwargs):
        if isinstance(data, tuple):
            t, r = data
            s = kwargs.get('scale', 2)
            p = {k: v for k, v in kwargs.items() if k != 'scale'}
            return (run_upsampling(name, t, s, p), run_upsampling(name, r, s, p))
        s = kwargs.get('scale', 2)
        p = {k: v for k, v in kwargs.items() if k != 'scale'}
        return run_upsampling(name, data, s, p)
    return Method(
        name=name, func=upsamp_func, category='upsampling',
        default_params=dict(method_info.default_params),
        param_ranges=dict(method_info.param_ranges),
        preserves=method_info.preserves,
        destroys=getattr(method_info, 'introduces', ''),
        description=method_info.description,
    )


def _build_decomposition_category():
    cat = MethodCategory(name='decomposition', description='Decomposition')
    for n, i in DECOMPOSITION_REGISTRY.items():
        cat.register(_wrap_decomposition(n, i))
    return cat


def _build_upsampling_category():
    cat = MethodCategory(name='upsampling', description='Upsampling')
    for n, i in UPSAMPLING_REGISTRY.items():
        cat.register(_wrap_upsampling(n, i))
    return cat


class TerrainPlugin(DomainPlugin):
    @property
    def name(self): return "terrain"
    @property
    def description(self): return "DEM processing"
    def get_pipelines(self):
        return [Pipeline(name='decomp_upsample', description='Decompose then upsample',
                stages=[_build_decomposition_category(), _build_upsampling_category()])]
    def get_generators(self):
        return {'fractal': lambda s=256: np.random.randn(s, s)}


PLUGIN = TerrainPlugin()

def get_plugin():
    return PLUGIN
