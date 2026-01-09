"""
Astronomy Domain Plugin

Specialized for astronomical image analysis - galaxies, stars, nebulae.

Provides morphological feature extraction pipeline.
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
# MORPHOLOGICAL FEATURE EXTRACTION METHODS
# =============================================================================

def find_center(image: np.ndarray) -> tuple:
    """Find the center of light (centroid)."""
    image = np.nan_to_num(image, nan=0)
    image = np.maximum(image, 0)
    
    total = np.sum(image)
    if total == 0:
        return image.shape[0] / 2, image.shape[1] / 2
    
    y_coords, x_coords = np.mgrid[:image.shape[0], :image.shape[1]]
    y_center = np.sum(y_coords * image) / total
    x_center = np.sum(x_coords * image) / total
    
    return y_center, x_center


def compute_concentration(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute concentration index C = 5 * log10(R80/R20)."""
    y_c, x_c = find_center(image)
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    r = np.sqrt((y - y_c)**2 + (x - x_c)**2)
    
    flat_r = r.flatten()
    flat_img = image.flatten()
    sort_idx = np.argsort(flat_r)
    
    sorted_flux = flat_img[sort_idx]
    sorted_r = flat_r[sort_idx]
    
    cumsum = np.cumsum(sorted_flux)
    total = cumsum[-1] if cumsum[-1] > 0 else 1
    
    r20_idx = np.searchsorted(cumsum, 0.2 * total)
    r80_idx = np.searchsorted(cumsum, 0.8 * total)
    
    r20 = sorted_r[min(r20_idx, len(sorted_r)-1)]
    r80 = sorted_r[min(r80_idx, len(sorted_r)-1)]
    
    if r20 > 0:
        return np.array([5.0 * np.log10(r80 / r20)])
    return np.array([0.0])


def compute_asymmetry(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute rotational asymmetry A."""
    image = np.nan_to_num(image, nan=0)
    rotated = np.rot90(np.rot90(image))
    diff = np.abs(image - rotated)
    total = np.sum(np.abs(image))
    
    if total > 0:
        return np.array([np.sum(diff) / (2 * total)])
    return np.array([0.0])


def compute_smoothness(image: np.ndarray, sigma: float = 3.0, **kwargs) -> np.ndarray:
    """Compute smoothness/clumpiness S."""
    image = np.nan_to_num(image, nan=0)
    smoothed = ndimage.gaussian_filter(image, sigma=sigma)
    residual = image - smoothed
    total = np.sum(np.abs(image))
    
    if total > 0:
        return np.array([np.sum(np.abs(residual)) / total])
    return np.array([0.0])


def compute_gini(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute Gini coefficient of light distribution."""
    flat = np.abs(image.flatten())
    flat = np.sort(flat)
    n = len(flat)
    
    if n == 0 or np.sum(flat) == 0:
        return np.array([0.0])
    
    cumsum = np.cumsum(flat)
    gini = (2 * np.sum(cumsum) - cumsum[-1] * (n + 1)) / (n * cumsum[-1])
    return np.array([gini])


def compute_m20(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute M20 statistic (second-order moment of brightest 20%)."""
    y_c, x_c = find_center(image)
    
    flat_img = image.flatten()
    sort_idx = np.argsort(flat_img)[::-1]  # Brightest first
    
    cumsum = np.cumsum(flat_img[sort_idx])
    total = cumsum[-1] if cumsum[-1] > 0 else 1
    
    bright_idx = np.searchsorted(cumsum, 0.2 * total)
    brightest_pixels = sort_idx[:max(1, bright_idx)]
    
    y_coords, x_coords = np.unravel_index(brightest_pixels, image.shape)
    
    m_tot = np.sum(image * ((np.arange(image.shape[0])[:, None] - y_c)**2 + 
                            (np.arange(image.shape[1])[None, :] - x_c)**2))
    
    m_20 = np.sum(image.flatten()[brightest_pixels] * 
                  ((y_coords - y_c)**2 + (x_coords - x_c)**2))
    
    if m_tot > 0:
        return np.array([np.log10(m_20 / m_tot)])
    return np.array([0.0])


def compute_all_morphology(image: np.ndarray, sigma: float = 3.0, **kwargs) -> np.ndarray:
    """Compute all morphological features as a single vector."""
    c = compute_concentration(image)[0]
    a = compute_asymmetry(image)[0]
    s = compute_smoothness(image, sigma=sigma)[0]
    g = compute_gini(image)[0]
    m = compute_m20(image)[0]
    
    return np.array([c, a, s, g, m])


# =============================================================================
# RADIAL PROFILE METHODS
# =============================================================================

def compute_radial_profile(image: np.ndarray, n_bins: int = 20, **kwargs) -> np.ndarray:
    """Compute radial profile from center of light."""
    y_c, x_c = find_center(image)
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    r = np.sqrt((y - y_c)**2 + (x - x_c)**2)
    
    max_r = min(image.shape) / 2
    bins = np.linspace(0, max_r, n_bins + 1)
    profile = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if np.sum(mask) > 0:
            profile[i] = np.mean(image[mask])
    
    return profile


def compute_sersic_index(image: np.ndarray, **kwargs) -> np.ndarray:
    """Estimate Sersic index from radial profile."""
    profile = compute_radial_profile(image, n_bins=20)
    profile = np.maximum(profile, 1e-10)
    log_profile = np.log(profile)
    
    # Linear regression on log profile
    x = np.arange(len(profile))
    valid = np.isfinite(log_profile)
    
    if np.sum(valid) < 2:
        return np.array([1.0])
    
    slope = np.polyfit(x[valid], log_profile[valid], 1)[0]
    n_est = -1 / (slope + 1e-10)  # Rough Sersic estimate
    
    return np.array([np.clip(n_est, 0.5, 10.0)])


# =============================================================================
# PLUGIN DEFINITION
# =============================================================================

def _build_morphology_category() -> MethodCategory:
    """Build morphology feature extraction category."""
    category = MethodCategory(
        name='morphology',
        description='Morphological feature extraction for astronomical objects'
    )
    
    category.register(Method(
        name='concentration',
        func=compute_concentration,
        category='morphology',
        description='Light concentration C = 5*log10(R80/R20)'
    ))
    
    category.register(Method(
        name='asymmetry',
        func=compute_asymmetry,
        category='morphology',
        description='Rotational asymmetry'
    ))
    
    category.register(Method(
        name='smoothness',
        func=compute_smoothness,
        category='morphology',
        default_params={'sigma': 3.0},
        param_ranges={'sigma': [1.0, 2.0, 3.0, 5.0]},
        description='Smoothness/clumpiness statistic'
    ))
    
    category.register(Method(
        name='gini',
        func=compute_gini,
        category='morphology',
        description='Gini coefficient of light distribution'
    ))
    
    category.register(Method(
        name='m20',
        func=compute_m20,
        category='morphology',
        description='M20 moment statistic'
    ))
    
    category.register(Method(
        name='all_morphology',
        func=compute_all_morphology,
        category='morphology',
        default_params={'sigma': 3.0},
        param_ranges={'sigma': [1.0, 2.0, 3.0, 5.0]},
        description='All CAS + Gini-M20 features as vector'
    ))
    
    return category


def _build_profile_category() -> MethodCategory:
    """Build radial profile category."""
    category = MethodCategory(
        name='profile',
        description='Radial profile analysis'
    )
    
    category.register(Method(
        name='radial_profile',
        func=compute_radial_profile,
        category='profile',
        default_params={'n_bins': 20},
        param_ranges={'n_bins': [10, 20, 50]},
        description='Azimuthally averaged radial profile'
    ))
    
    category.register(Method(
        name='sersic_index',
        func=compute_sersic_index,
        category='profile',
        description='Estimated Sersic index from profile'
    ))
    
    return category


class AstronomyPlugin(DomainPlugin):
    """Astronomy domain plugin for galaxy/object analysis."""
    
    @property
    def name(self) -> str:
        return "astronomy"
    
    @property
    def description(self) -> str:
        return "Astronomical image analysis - morphology, profiles, photometry"
    
    def get_pipelines(self) -> List[Pipeline]:
        morph_category = _build_morphology_category()
        profile_category = _build_profile_category()
        
        # Morphology pipeline
        morph_pipeline = Pipeline(
            name='morphology',
            description='Morphological feature extraction (CAS, Gini-M20)',
            stages=[morph_category]
        )
        
        # Profile pipeline
        profile_pipeline = Pipeline(
            name='profile',
            description='Radial profile analysis',
            stages=[profile_category]
        )
        
        # Combined pipeline
        combined = Pipeline(
            name='full_analysis',
            description='Morphology + Profile analysis',
            stages=[morph_category, profile_category]
        )
        
        return [morph_pipeline, profile_pipeline, combined]
    
    def get_generators(self) -> Dict[str, Callable]:
        return {
            'gaussian_galaxy': generate_gaussian_galaxy,
            'sersic_galaxy': generate_sersic_galaxy,
        }


def generate_gaussian_galaxy(size: int = 64, sigma: float = 10.0) -> np.ndarray:
    """Generate a simple Gaussian galaxy profile."""
    y, x = np.ogrid[:size, :size]
    center = size / 2
    galaxy = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    return galaxy


def generate_sersic_galaxy(size: int = 64, n: float = 2.0, r_eff: float = 10.0) -> np.ndarray:
    """Generate a Sersic profile galaxy."""
    y, x = np.ogrid[:size, :size]
    center = size / 2
    r = np.sqrt((x - center)**2 + (y - center)**2)
    
    b_n = 2 * n - 1/3 + 4/(405*n)  # Approximation
    galaxy = np.exp(-b_n * ((r / r_eff)**(1/n) - 1))
    
    return galaxy


# Plugin instance for auto-discovery
PLUGIN = AstronomyPlugin()

def get_plugin() -> DomainPlugin:
    """Return the astronomy plugin instance."""
    return PLUGIN
