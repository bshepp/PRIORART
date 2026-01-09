"""
CMB Domain Plugin

Specialized for Cosmic Microwave Background analysis.

Provides power spectrum and statistical analysis pipeline.
"""

from typing import List, Dict, Any, Callable
import numpy as np

from priorart.core.plugin import (
    DomainPlugin,
    Pipeline,
    MethodCategory,
    Method,
)


# =============================================================================
# POWER SPECTRUM METHODS
# =============================================================================

def compute_power_spectrum(image: np.ndarray, n_bins: int = 20, **kwargs) -> np.ndarray:
    """Compute angular power spectrum from 2D map."""
    # FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift)**2
    
    # Radial binning
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    r = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    
    max_r = min(image.shape) // 2
    bins = np.linspace(0, max_r, n_bins + 1)
    spectrum = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if np.sum(mask) > 0:
            spectrum[i] = np.mean(power[mask])
    
    return spectrum


def compute_spectral_index(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute spectral index (power law slope)."""
    spectrum = compute_power_spectrum(image, n_bins=20)
    spectrum = np.maximum(spectrum, 1e-10)
    log_spectrum = np.log(spectrum)
    
    x = np.arange(len(spectrum))
    valid = np.isfinite(log_spectrum) & (spectrum > 0)
    
    if np.sum(valid) < 2:
        return np.array([0.0])
    
    slope = np.polyfit(x[valid], log_spectrum[valid], 1)[0]
    return np.array([slope])


# =============================================================================
# STATISTICAL METHODS  
# =============================================================================

def compute_basic_stats(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute basic statistical moments."""
    flat = image.flatten()
    return np.array([
        np.mean(flat),
        np.std(flat),
        np.min(flat),
        np.max(flat),
    ])


def compute_higher_moments(image: np.ndarray, **kwargs) -> np.ndarray:
    """Compute skewness and kurtosis."""
    flat = image.flatten()
    mean = np.mean(flat)
    std = np.std(flat)
    
    if std == 0:
        return np.array([0.0, 0.0])
    
    centered = (flat - mean) / std
    skewness = np.mean(centered**3)
    kurtosis = np.mean(centered**4) - 3  # Excess kurtosis
    
    return np.array([skewness, kurtosis])


def compute_minkowski_functionals(image: np.ndarray, n_thresholds: int = 10, **kwargs) -> np.ndarray:
    """Compute Minkowski functionals at multiple thresholds."""
    thresholds = np.percentile(image, np.linspace(10, 90, n_thresholds))
    areas = np.zeros(n_thresholds)
    
    for i, thresh in enumerate(thresholds):
        areas[i] = np.mean(image > thresh)
    
    return areas


# =============================================================================
# PLUGIN DEFINITION
# =============================================================================

def _build_spectrum_category() -> MethodCategory:
    """Build power spectrum category."""
    category = MethodCategory(
        name='spectrum',
        description='Power spectrum analysis for CMB maps'
    )
    
    category.register(Method(
        name='power_spectrum',
        func=compute_power_spectrum,
        category='spectrum',
        default_params={'n_bins': 20},
        param_ranges={'n_bins': [10, 20, 50]},
        description='Angular power spectrum Cl'
    ))
    
    category.register(Method(
        name='spectral_index',
        func=compute_spectral_index,
        category='spectrum',
        description='Power law spectral index'
    ))
    
    return category


def _build_statistics_category() -> MethodCategory:
    """Build statistics category."""
    category = MethodCategory(
        name='statistics',
        description='Statistical analysis of CMB fluctuations'
    )
    
    category.register(Method(
        name='basic_stats',
        func=compute_basic_stats,
        category='statistics',
        description='Mean, std, min, max'
    ))
    
    category.register(Method(
        name='higher_moments',
        func=compute_higher_moments,
        category='statistics',
        description='Skewness and kurtosis'
    ))
    
    category.register(Method(
        name='minkowski',
        func=compute_minkowski_functionals,
        category='statistics',
        default_params={'n_thresholds': 10},
        param_ranges={'n_thresholds': [5, 10, 20]},
        description='Minkowski functionals'
    ))
    
    return category


class CMBPlugin(DomainPlugin):
    """CMB domain plugin for cosmological analysis."""
    
    @property
    def name(self) -> str:
        return "cmb"
    
    @property
    def description(self) -> str:
        return "Cosmic Microwave Background analysis - power spectrum, statistics"
    
    def get_pipelines(self) -> List[Pipeline]:
        spectrum_category = _build_spectrum_category()
        stats_category = _build_statistics_category()
        
        spectrum_pipeline = Pipeline(
            name='spectrum',
            description='Power spectrum analysis',
            stages=[spectrum_category]
        )
        
        stats_pipeline = Pipeline(
            name='statistics',
            description='Statistical analysis',
            stages=[stats_category]
        )
        
        full_pipeline = Pipeline(
            name='full_analysis',
            description='Spectrum + Statistics',
            stages=[spectrum_category, stats_category]
        )
        
        return [spectrum_pipeline, stats_pipeline, full_pipeline]
    
    def get_generators(self) -> Dict[str, Callable]:
        return {
            'gaussian_cmb': generate_gaussian_cmb,
        }


def generate_gaussian_cmb(size: int = 256, scale: float = 1.0) -> np.ndarray:
    """Generate Gaussian random CMB-like field."""
    # Generate power spectrum: Cl ~ l^-2 approximately
    k = np.fft.fftfreq(size)
    kx, ky = np.meshgrid(k, k)
    k_mag = np.sqrt(kx**2 + ky**2)
    k_mag[0, 0] = 1  # Avoid division by zero
    
    power = 1 / (k_mag**2 + 0.01)
    power[0, 0] = 0  # No DC component
    
    # Random phases
    phases = np.random.uniform(0, 2*np.pi, (size, size))
    fft_field = np.sqrt(power) * np.exp(1j * phases)
    
    field = np.real(np.fft.ifft2(fft_field))
    return field * scale


# Plugin instance
PLUGIN = CMBPlugin()

def get_plugin() -> DomainPlugin:
    return PLUGIN
