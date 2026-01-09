"""
Satellite Domain Plugin

Specialized for satellite imagery analysis.

Provides spectral indices and texture analysis pipeline.
"""

from typing import List, Dict, Any, Callable
import numpy as np
from scipy import ndimage

from priorart.core.plugin import (
    DomainPlugin,
    Pipeline,
    MethodCategory,
    Method,
)


# =============================================================================
# TEXTURE METHODS
# =============================================================================

def compute_glcm_contrast(image: np.ndarray, distance: int = 1, **kwargs) -> np.ndarray:
    """Compute GLCM contrast (simplified)."""
    # Simple approximation using gradient magnitude
    gy, gx = np.gradient(image)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return np.array([np.mean(grad_mag), np.std(grad_mag)])


def compute_glcm_homogeneity(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute GLCM homogeneity (simplified)."""
    # Use local variance as proxy
    local_var = ndimage.generic_filter(image, np.var, size=3)
    homogeneity = 1 / (1 + np.mean(local_var))
    return np.array([homogeneity])


def compute_glcm_energy(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute GLCM energy/uniformity (simplified)."""
    # Histogram-based approximation
    hist, _ = np.histogram(image.flatten(), bins=32, density=True)
    energy = np.sum(hist**2)
    return np.array([energy])


def compute_glcm_correlation(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute GLCM correlation (simplified)."""
    # Use autocorrelation at lag 1
    flat = image.flatten()
    if len(flat) < 2:
        return np.array([0.0])
    corr = np.corrcoef(flat[:-1], flat[1:])[0, 1]
    return np.array([corr if np.isfinite(corr) else 0.0])


def compute_all_texture(image: np.ndarray, distance: int = 1, **kwargs) -> np.ndarray:
    """Compute all texture features."""
    contrast = compute_glcm_contrast(image, distance)
    homog = compute_glcm_homogeneity(image)
    energy = compute_glcm_energy(image)
    corr = compute_glcm_correlation(image)
    return np.concatenate([contrast, homog, energy, corr])


# =============================================================================
# SPECTRAL INDEX METHODS (for multi-band imagery)
# =============================================================================

def compute_normalized_diff(image: np.ndarray, band1_slice: tuple = None, 
                            band2_slice: tuple = None, **kwargs) -> np.ndarray:
    """Compute normalized difference index.
    
    For single-band images, uses spatial gradients as proxy.
    """
    if image.ndim == 2:
        # Single band - compute local normalized difference
        gy, gx = np.gradient(image)
        grad_mag = np.sqrt(gx**2 + gy**2)
        denom = np.abs(image) + 1e-10
        ndi = grad_mag / denom
        return np.array([np.mean(ndi), np.std(ndi)])
    else:
        # Multi-band - actual NDVI-style calculation
        if band1_slice is None:
            band1_slice = (slice(None), slice(None), 0)
        if band2_slice is None:
            band2_slice = (slice(None), slice(None), 1)
        
        b1 = image[band1_slice].astype(float)
        b2 = image[band2_slice].astype(float)
        
        ndi = (b1 - b2) / (b1 + b2 + 1e-10)
        return np.array([np.mean(ndi), np.std(ndi)])


def compute_spectral_variance(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute spectral variance across bands or local variance."""
    if image.ndim == 2:
        local_var = ndimage.generic_filter(image, np.var, size=5)
        return np.array([np.mean(local_var), np.std(local_var)])
    else:
        band_means = [np.mean(image[:, :, i]) for i in range(image.shape[2])]
        return np.array([np.var(band_means)])


# =============================================================================
# PLUGIN DEFINITION
# =============================================================================

def _build_texture_category() -> MethodCategory:
    """Build texture analysis category."""
    category = MethodCategory(
        name='texture',
        description='Texture feature extraction (GLCM-based)'
    )
    
    category.register(Method(
        name='contrast',
        func=compute_glcm_contrast,
        category='texture',
        default_params={'distance': 1},
        param_ranges={'distance': [1, 2, 3]},
        description='GLCM contrast'
    ))
    
    category.register(Method(
        name='homogeneity',
        func=compute_glcm_homogeneity,
        category='texture',
        description='GLCM homogeneity'
    ))
    
    category.register(Method(
        name='energy',
        func=compute_glcm_energy,
        category='texture',
        description='GLCM energy/uniformity'
    ))
    
    category.register(Method(
        name='correlation',
        func=compute_glcm_correlation,
        category='texture',
        description='GLCM correlation'
    ))
    
    category.register(Method(
        name='all_texture',
        func=compute_all_texture,
        category='texture',
        default_params={'distance': 1},
        param_ranges={'distance': [1, 2, 3]},
        description='All texture features'
    ))
    
    return category


def _build_spectral_category() -> MethodCategory:
    """Build spectral analysis category."""
    category = MethodCategory(
        name='spectral',
        description='Spectral index computation'
    )
    
    category.register(Method(
        name='normalized_diff',
        func=compute_normalized_diff,
        category='spectral',
        description='Normalized difference index'
    ))
    
    category.register(Method(
        name='spectral_variance',
        func=compute_spectral_variance,
        category='spectral',
        description='Spectral/spatial variance'
    ))
    
    return category


class SatellitePlugin(DomainPlugin):
    """Satellite imagery domain plugin."""
    
    @property
    def name(self) -> str:
        return "satellite"
    
    @property
    def description(self) -> str:
        return "Satellite imagery analysis - texture, spectral indices"
    
    def get_pipelines(self) -> List[Pipeline]:
        texture_category = _build_texture_category()
        spectral_category = _build_spectral_category()
        
        texture_pipeline = Pipeline(
            name='texture',
            description='Texture feature extraction',
            stages=[texture_category]
        )
        
        spectral_pipeline = Pipeline(
            name='spectral',
            description='Spectral index computation',
            stages=[spectral_category]
        )
        
        full_pipeline = Pipeline(
            name='full_analysis',
            description='Texture + Spectral',
            stages=[texture_category, spectral_category]
        )
        
        return [texture_pipeline, spectral_pipeline, full_pipeline]
    
    def get_generators(self) -> Dict[str, Callable]:
        return {
            'urban_pattern': generate_urban_pattern,
            'vegetation_pattern': generate_vegetation_pattern,
        }


def generate_urban_pattern(size: int = 256) -> np.ndarray:
    """Generate urban-like grid pattern."""
    pattern = np.zeros((size, size))
    
    # Add grid lines
    for i in range(0, size, 20):
        pattern[i:i+3, :] = 1
        pattern[:, i:i+3] = 1
    
    # Add some blocks
    for _ in range(20):
        x = np.random.randint(0, size - 20)
        y = np.random.randint(0, size - 20)
        w = np.random.randint(5, 15)
        h = np.random.randint(5, 15)
        pattern[y:y+h, x:x+w] = np.random.uniform(0.3, 0.8)
    
    return pattern


def generate_vegetation_pattern(size: int = 256) -> np.ndarray:
    """Generate vegetation-like pattern."""
    from scipy.ndimage import gaussian_filter
    
    # Base perlin-like noise
    noise = np.random.randn(size // 8, size // 8)
    pattern = ndimage.zoom(noise, 8, order=1)[:size, :size]
    
    # Normalize to 0-1
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-10)
    
    return pattern


# Plugin instance
PLUGIN = SatellitePlugin()

def get_plugin() -> DomainPlugin:
    return PLUGIN
