"""
Extended Decomposition Methods

Additional methods to maximize prior art coverage.
These complement the core methods in methods.py.

Domain-agnostic: works on any 2D scalar field.
"""

import numpy as np
from scipy.ndimage import (
    gaussian_filter, median_filter, uniform_filter,
    grey_opening, grey_closing, grey_erosion, grey_dilation,
    laplace
)
from skimage.morphology import disk, square, rectangle, diamond, ellipse
from skimage.filters import difference_of_gaussians

from .registry import register_decomposition


# =============================================================================
# Additional Classical Methods
# =============================================================================

@register_decomposition(
    name='gaussian_anisotropic',
    category='classical',
    default_params={'sigma_x': 10, 'sigma_y': 10},
    param_ranges={
        'sigma_x': [2, 5, 10, 20, 50],
        'sigma_y': [2, 5, 10, 20, 50]
    },
    preserves='directional features aligned with low-sigma axis',
    destroys='features perpendicular to low-sigma axis'
)
def decompose_gaussian_anisotropic(
    data: np.ndarray,
    sigma_x: float = 10,
    sigma_y: float = 10
) -> tuple:
    """Anisotropic Gaussian filtering with different X/Y smoothing."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    trend = gaussian_filter(data_filled, sigma=(sigma_y, sigma_x))
    residual = data_filled - trend
    return trend, residual


@register_decomposition(
    name='median',
    category='edge_preserving',
    default_params={'size': 5},
    param_ranges={'size': [3, 5, 7, 11, 15, 21]},
    preserves='sharp edges, step discontinuities',
    destroys='salt-and-pepper noise, thin lines'
)
def decompose_median(data: np.ndarray, size: int = 5) -> tuple:
    """Median filter decomposition - edge-preserving, removes impulse noise."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    trend = median_filter(data_filled, size=size)
    residual = data_filled - trend
    return trend, residual


@register_decomposition(
    name='uniform',
    category='classical',
    default_params={'size': 10},
    param_ranges={'size': [3, 5, 10, 20, 50, 100]},
    preserves='average local value',
    destroys='all local variation equally'
)
def decompose_uniform(data: np.ndarray, size: int = 10) -> tuple:
    """Uniform (box) filter decomposition - simple averaging."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    trend = uniform_filter(data_filled, size=size)
    residual = data_filled - trend
    return trend, residual


# =============================================================================
# Difference of Gaussians (DoG) - Band-pass filtering
# =============================================================================

@register_decomposition(
    name='dog',
    category='multiscale',
    default_params={'sigma_low': 2, 'sigma_high': 10},
    param_ranges={
        'sigma_low': [1, 2, 3, 5],
        'sigma_high': [5, 10, 20, 50, 100]
    },
    preserves='features at intermediate scales',
    destroys='very small and very large features'
)
def decompose_dog(
    data: np.ndarray,
    sigma_low: float = 2,
    sigma_high: float = 10
) -> tuple:
    """Difference of Gaussians band-pass filtering."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    if sigma_high <= sigma_low:
        sigma_high = sigma_low * 2
    
    residual = difference_of_gaussians(data_filled, sigma_low, sigma_high)
    trend = data_filled - residual
    return trend, residual


@register_decomposition(
    name='dog_multiscale',
    category='multiscale',
    default_params={'sigma_ratio': 1.6, 'n_scales': 4, 'base_sigma': 1.0},
    param_ranges={
        'sigma_ratio': [1.4, 1.6, 2.0],
        'n_scales': [3, 4, 5, 6],
        'base_sigma': [0.5, 1.0, 2.0]
    },
    preserves='multi-scale blob-like features',
    destroys='flat regions, monotonic gradients'
)
def decompose_dog_multiscale(
    data: np.ndarray,
    sigma_ratio: float = 1.6,
    n_scales: int = 4,
    base_sigma: float = 1.0
) -> tuple:
    """Multi-scale Difference of Gaussians."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    combined_residual = np.zeros_like(data_filled)
    
    for i in range(n_scales):
        sigma_low = base_sigma * (sigma_ratio ** i)
        sigma_high = base_sigma * (sigma_ratio ** (i + 1))
        dog = difference_of_gaussians(data_filled, sigma_low, sigma_high)
        combined_residual += dog
    
    combined_residual /= n_scales
    trend = data_filled - combined_residual
    
    return trend, combined_residual


# =============================================================================
# Laplacian of Gaussian (LoG)
# =============================================================================

@register_decomposition(
    name='log',
    category='multiscale',
    default_params={'sigma': 5},
    param_ranges={'sigma': [1, 2, 3, 5, 10, 20]},
    preserves='blob-like features at specified scale',
    destroys='linear features, edges, flat regions'
)
def decompose_log(data: np.ndarray, sigma: float = 5) -> tuple:
    """Laplacian of Gaussian decomposition for blob detection."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    smoothed = gaussian_filter(data_filled, sigma=sigma)
    residual = laplace(smoothed) * (sigma ** 2)
    trend = data_filled - residual
    
    return trend, residual


# =============================================================================
# Morphological Methods with Different Structuring Elements
# =============================================================================

@register_decomposition(
    name='morphological_square',
    category='morphological',
    default_params={'operation': 'opening', 'size': 10},
    param_ranges={
        'operation': ['opening', 'closing', 'gradient', 'average'],
        'size': [3, 5, 10, 15, 20, 50]
    },
    preserves='rectangular features aligned with axes',
    destroys='features smaller than element, circular features'
)
def decompose_morphological_square(
    data: np.ndarray,
    operation: str = 'opening',
    size: int = 10
) -> tuple:
    """Morphological filtering with square structuring element."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    selem = square(size)
    
    if operation == 'opening':
        trend = grey_opening(data_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(data_filled, footprint=selem)
    elif operation == 'gradient':
        dilated = grey_dilation(data_filled, footprint=selem)
        eroded = grey_erosion(data_filled, footprint=selem)
        residual = dilated - eroded
        trend = data_filled - residual
        return trend, residual
    else:
        opened = grey_opening(data_filled, footprint=selem)
        closed = grey_closing(data_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = data_filled - trend
    return trend, residual


@register_decomposition(
    name='morphological_rect',
    category='morphological',
    default_params={'operation': 'opening', 'width': 20, 'height': 5},
    param_ranges={
        'operation': ['opening', 'closing', 'average'],
        'width': [5, 10, 20, 50],
        'height': [3, 5, 10, 20]
    },
    preserves='linear features perpendicular to long axis',
    destroys='features parallel to long axis, small features'
)
def decompose_morphological_rect(
    data: np.ndarray,
    operation: str = 'opening',
    width: int = 20,
    height: int = 5
) -> tuple:
    """Morphological filtering with rectangular structuring element."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    selem = rectangle(height, width)
    
    if operation == 'opening':
        trend = grey_opening(data_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(data_filled, footprint=selem)
    else:
        opened = grey_opening(data_filled, footprint=selem)
        closed = grey_closing(data_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = data_filled - trend
    return trend, residual


@register_decomposition(
    name='morphological_diamond',
    category='morphological',
    default_params={'operation': 'opening', 'radius': 10},
    param_ranges={
        'operation': ['opening', 'closing', 'average'],
        'radius': [3, 5, 10, 15, 20]
    },
    preserves='diamond/rhombus shaped features',
    destroys='features not matching diamond geometry'
)
def decompose_morphological_diamond(
    data: np.ndarray,
    operation: str = 'opening',
    radius: int = 10
) -> tuple:
    """Morphological filtering with diamond structuring element."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    selem = diamond(radius)
    
    if operation == 'opening':
        trend = grey_opening(data_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(data_filled, footprint=selem)
    else:
        opened = grey_opening(data_filled, footprint=selem)
        closed = grey_closing(data_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = data_filled - trend
    return trend, residual


@register_decomposition(
    name='morphological_ellipse',
    category='morphological',
    default_params={'operation': 'opening', 'width': 20, 'height': 10},
    param_ranges={
        'operation': ['opening', 'closing', 'average'],
        'width': [5, 10, 20, 30, 50],
        'height': [5, 10, 20, 30, 50]
    },
    preserves='elliptical features with matching orientation',
    destroys='features not matching ellipse geometry'
)
def decompose_morphological_ellipse(
    data: np.ndarray,
    operation: str = 'opening',
    width: int = 20,
    height: int = 10
) -> tuple:
    """Morphological filtering with elliptical structuring element."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    selem = ellipse(height // 2, width // 2)
    
    if operation == 'opening':
        trend = grey_opening(data_filled, footprint=selem)
    elif operation == 'closing':
        trend = grey_closing(data_filled, footprint=selem)
    else:
        opened = grey_opening(data_filled, footprint=selem)
        closed = grey_closing(data_filled, footprint=selem)
        trend = (opened + closed) / 2
    
    residual = data_filled - trend
    return trend, residual


@register_decomposition(
    name='morphological_gradient',
    category='morphological',
    default_params={'size': 5, 'shape': 'disk'},
    param_ranges={
        'size': [3, 5, 7, 10, 15],
        'shape': ['disk', 'square', 'diamond']
    },
    preserves='edges, boundaries, rapid transitions',
    destroys='flat regions, gradual slopes'
)
def decompose_morphological_gradient(
    data: np.ndarray,
    size: int = 5,
    shape: str = 'disk'
) -> tuple:
    """Morphological gradient = dilation - erosion."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    if shape == 'disk':
        selem = disk(size)
    elif shape == 'square':
        selem = square(size)
    else:
        selem = diamond(size)
    
    dilated = grey_dilation(data_filled, footprint=selem)
    eroded = grey_erosion(data_filled, footprint=selem)
    
    residual = dilated - eroded
    trend = data_filled - residual
    
    return trend, residual


@register_decomposition(
    name='tophat_combined',
    category='morphological',
    default_params={'size': 20},
    param_ranges={'size': [5, 10, 20, 50, 100]},
    preserves='both bright and dark small features',
    destroys='large-scale variation, features larger than element'
)
def decompose_tophat_combined(data: np.ndarray, size: int = 20) -> tuple:
    """Combined white + black top-hat transform."""
    from scipy.ndimage import white_tophat, black_tophat
    
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    selem = disk(size)
    
    white_th = white_tophat(data_filled, footprint=selem)
    black_th = black_tophat(data_filled, footprint=selem)
    
    residual = white_th - black_th
    trend = data_filled - residual
    
    return trend, residual


# =============================================================================
# Anisotropic Diffusion (Perona-Malik)
# =============================================================================

@register_decomposition(
    name='anisotropic_diffusion',
    category='edge_preserving',
    default_params={'iterations': 10, 'kappa': 50, 'gamma': 0.1},
    param_ranges={
        'iterations': [5, 10, 20, 50],
        'kappa': [10, 30, 50, 100],
        'gamma': [0.05, 0.1, 0.15, 0.2]
    },
    preserves='edges above gradient threshold (kappa)',
    destroys='noise, texture below threshold'
)
def decompose_anisotropic_diffusion(
    data: np.ndarray,
    iterations: int = 10,
    kappa: float = 50,
    gamma: float = 0.1
) -> tuple:
    """Perona-Malik anisotropic diffusion."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    img = data_filled.copy().astype(np.float64)
    
    for _ in range(iterations):
        nabla_n = np.roll(img, -1, axis=0) - img
        nabla_s = np.roll(img, 1, axis=0) - img
        nabla_e = np.roll(img, -1, axis=1) - img
        nabla_w = np.roll(img, 1, axis=1) - img
        
        c_n = np.exp(-(nabla_n / kappa) ** 2)
        c_s = np.exp(-(nabla_s / kappa) ** 2)
        c_e = np.exp(-(nabla_e / kappa) ** 2)
        c_w = np.exp(-(nabla_w / kappa) ** 2)
        
        img += gamma * (c_n * nabla_n + c_s * nabla_s + 
                       c_e * nabla_e + c_w * nabla_w)
    
    trend = img
    residual = data_filled - trend
    
    return trend, residual


# =============================================================================
# Rolling Ball Background Subtraction
# =============================================================================

@register_decomposition(
    name='rolling_ball',
    category='morphological',
    default_params={'radius': 50},
    param_ranges={'radius': [10, 25, 50, 100, 200]},
    preserves='features smaller than ball radius',
    destroys='background curvature, large-scale variation'
)
def decompose_rolling_ball(data: np.ndarray, radius: int = 50) -> tuple:
    """Rolling ball background subtraction."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    size = 2 * radius + 1
    y, x = np.ogrid[:size, :size]
    center = radius
    
    dist_sq = (x - center) ** 2 + (y - center) ** 2
    ball = np.where(
        dist_sq <= radius ** 2,
        np.sqrt(radius ** 2 - dist_sq),
        0
    )
    ball = ball.max() - ball
    
    eroded = grey_erosion(data_filled, footprint=ball > 0, structure=ball)
    trend = grey_dilation(eroded, footprint=ball > 0, structure=ball)
    
    residual = data_filled - trend
    
    return trend, residual


# =============================================================================
# Local Polynomial
# =============================================================================

@register_decomposition(
    name='local_polynomial',
    category='trend_removal',
    default_params={'window_size': 51, 'degree': 2},
    param_ranges={
        'window_size': [21, 31, 51, 101],
        'degree': [1, 2, 3]
    },
    preserves='local deviations from local polynomial trend',
    destroys='features smoother than local polynomial'
)
def decompose_local_polynomial(
    data: np.ndarray,
    window_size: int = 51,
    degree: int = 2
) -> tuple:
    """Local polynomial (LOESS-like) trend removal."""
    from scipy.ndimage import sobel
    
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    if degree == 1:
        trend = uniform_filter(data_filled, size=window_size)
    elif degree >= 2:
        sigma = window_size / 4
        trend = gaussian_filter(data_filled, sigma=sigma)
    else:
        trend = uniform_filter(data_filled, size=window_size)
    
    residual = data_filled - trend
    
    return trend, residual


# =============================================================================
# Guided Filter
# =============================================================================

@register_decomposition(
    name='guided',
    category='edge_preserving',
    default_params={'radius': 8, 'eps': 0.01},
    param_ranges={
        'radius': [4, 8, 16, 32],
        'eps': [0.001, 0.01, 0.1, 1.0]
    },
    preserves='edges defined by the guide image',
    destroys='texture not aligned with edges'
)
def decompose_guided(
    data: np.ndarray,
    radius: int = 8,
    eps: float = 0.01
) -> tuple:
    """Self-guided filter decomposition."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    data_min, data_max = data_filled.min(), data_filled.max()
    data_range = data_max - data_min
    if data_range == 0:
        data_range = 1
    data_norm = (data_filled - data_min) / data_range
    
    I = data_norm
    p = data_norm
    
    mean_I = uniform_filter(I, size=2*radius+1)
    mean_p = uniform_filter(p, size=2*radius+1)
    corr_Ip = uniform_filter(I * p, size=2*radius+1)
    corr_II = uniform_filter(I * I, size=2*radius+1)
    
    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = uniform_filter(a, size=2*radius+1)
    mean_b = uniform_filter(b, size=2*radius+1)
    
    q = mean_a * I + mean_b
    
    trend = q * data_range + data_min
    residual = data_filled - trend
    
    return trend, residual


# =============================================================================
# Higher-Degree Polynomial
# =============================================================================

@register_decomposition(
    name='polynomial_high',
    category='trend_removal',
    default_params={'degree': 4},
    param_ranges={'degree': [4, 5, 6]},
    preserves='local deviations from high-order regional trend',
    destroys='large-scale shape up to specified degree'
)
def decompose_polynomial_high(data: np.ndarray, degree: int = 4) -> tuple:
    """High-degree polynomial surface fitting."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    rows, cols = data_filled.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = data_filled.flatten()
    
    X_norm = (X_flat - X_flat.mean()) / (X_flat.std() + 1e-10)
    Y_norm = (Y_flat - Y_flat.mean()) / (Y_flat.std() + 1e-10)
    
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((X_norm ** i) * (Y_norm ** j))
    
    A = np.column_stack(terms)
    coeffs, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)
    
    trend_flat = A @ coeffs
    trend = trend_flat.reshape(data_filled.shape)
    residual = data_filled - trend
    
    return trend, residual


# =============================================================================
# Additional Wavelet Methods
# =============================================================================

@register_decomposition(
    name='wavelet_biorthogonal',
    category='wavelet',
    default_params={'wavelet': 'bior3.5', 'level': 3},
    param_ranges={
        'wavelet': ['bior1.3', 'bior2.4', 'bior3.5', 'bior4.4', 'bior5.5'],
        'level': [1, 2, 3, 4, 5]
    },
    preserves='multi-scale structure with linear phase',
    destroys='high-frequency detail (depends on level)'
)
def decompose_wavelet_biorthogonal(
    data: np.ndarray,
    wavelet: str = 'bior3.5',
    level: int = 3
) -> tuple:
    """Biorthogonal wavelet decomposition."""
    import pywt
    
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    coeffs = pywt.wavedec2(data_filled, wavelet, level=level)
    
    trend_coeffs = [coeffs[0]] + [
        tuple(np.zeros_like(d) for d in detail) 
        for detail in coeffs[1:]
    ]
    residual_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    
    trend = pywt.waverec2(trend_coeffs, wavelet)
    residual = pywt.waverec2(residual_coeffs, wavelet)
    
    trend = trend[:data.shape[0], :data.shape[1]]
    residual = residual[:data.shape[0], :data.shape[1]]
    
    return trend, residual


@register_decomposition(
    name='wavelet_reverse_biorthogonal',
    category='wavelet',
    default_params={'wavelet': 'rbio3.5', 'level': 3},
    param_ranges={
        'wavelet': ['rbio1.3', 'rbio2.4', 'rbio3.5', 'rbio4.4', 'rbio5.5'],
        'level': [1, 2, 3, 4, 5]
    },
    preserves='multi-scale structure with reversed decomposition',
    destroys='high-frequency detail (depends on level)'
)
def decompose_wavelet_reverse_biorthogonal(
    data: np.ndarray,
    wavelet: str = 'rbio3.5',
    level: int = 3
) -> tuple:
    """Reverse biorthogonal wavelet decomposition."""
    import pywt
    
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    coeffs = pywt.wavedec2(data_filled, wavelet, level=level)
    
    trend_coeffs = [coeffs[0]] + [
        tuple(np.zeros_like(d) for d in detail) 
        for detail in coeffs[1:]
    ]
    residual_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    
    trend = pywt.waverec2(trend_coeffs, wavelet)
    residual = pywt.waverec2(residual_coeffs, wavelet)
    
    trend = trend[:data.shape[0], :data.shape[1]]
    residual = residual[:data.shape[0], :data.shape[1]]
    
    return trend, residual

