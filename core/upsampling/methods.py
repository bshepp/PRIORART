"""
Core Upsampling Methods

Implements 4 key upsampling methods:
- Bicubic (Standard baseline)
- Lanczos (Sharp edge preservation)
- B-Spline (Smooth interpolation)
- FFT Zero-padding (Frequency-preserving)

Domain-agnostic: works on any 2D scalar field.
"""

import numpy as np
from scipy.ndimage import zoom
import cv2

from .registry import register_upsampling


# =============================================================================
# Classical Interpolation
# =============================================================================

@register_upsampling(
    name='bicubic',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='smooth curves, good for continuous surfaces',
    introduces='slight ringing at sharp edges'
)
def upsample_bicubic(data: np.ndarray, scale: int = 2) -> np.ndarray:
    """Bicubic interpolation using scipy.ndimage.zoom (order=3)."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    return zoom(data_filled, scale, order=3)


@register_upsampling(
    name='lanczos',
    category='interpolation',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='sharp edges with smooth interpolation',
    introduces='controlled ringing (less than sinc)'
)
def upsample_lanczos(data: np.ndarray, scale: int = 2) -> np.ndarray:
    """Lanczos interpolation using OpenCV."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    h, w = data_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    return cv2.resize(
        data_filled.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )


# =============================================================================
# Spline Methods
# =============================================================================

@register_upsampling(
    name='bspline',
    category='spline',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='smooth surface, less ringing than cubic',
    introduces='slightly more smoothing than cubic'
)
def upsample_bspline(data: np.ndarray, scale: int = 2) -> np.ndarray:
    """Quadratic B-spline interpolation (order=2)."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    return zoom(data_filled, scale, order=2)


# =============================================================================
# Frequency Domain
# =============================================================================

@register_upsampling(
    name='fft_zeropad',
    category='frequency',
    default_params={'scale': 2},
    param_ranges={'scale': [2, 4, 8]},
    preserves='frequency content exactly (band-limited)',
    introduces='Gibbs ringing at discontinuities'
)
def upsample_fft_zeropad(data: np.ndarray, scale: int = 2) -> np.ndarray:
    """FFT upsampling via zero-padding in frequency domain."""
    data_filled = np.nan_to_num(data, nan=np.nanmean(data))
    
    h, w = data_filled.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    fft = np.fft.fft2(data_filled)
    padded = np.zeros((new_h, new_w), dtype=complex)
    
    h_half = (h + 1) // 2
    w_half = (w + 1) // 2
    
    padded[:h_half, :w_half] = fft[:h_half, :w_half]
    
    if w > 1:
        padded[:h_half, -(w - w_half):] = fft[:h_half, w_half:]
    
    if h > 1:
        padded[-(h - h_half):, :w_half] = fft[h_half:, :w_half]
    
    if h > 1 and w > 1:
        padded[-(h - h_half):, -(w - w_half):] = fft[h_half:, w_half:]
    
    result = np.real(np.fft.ifft2(padded)) * (scale ** 2)
    
    return result

