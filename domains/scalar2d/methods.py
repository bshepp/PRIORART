"""
Scalar2D Domain-Specific Methods

Optional extensions to core methods for 2D scalar field processing.
These build on the core decomposition/upsampling methods.
"""

# Currently, all methods from core work directly on 2D scalar fields.
# This file is a placeholder for domain-specific extensions.
#
# Potential additions:
# - Fourier-based methods optimized for periodic fields
# - Gradient-based edge detection specific to scalar gradients
# - Threshold-based segmentation methods
# - Local contrast normalization

from core.decomposition import list_decompositions
from core.upsampling import list_upsamplings


def get_available_methods():
    """List all methods available for 2D scalar fields."""
    return {
        'decomposition': list_decompositions(),
        'upsampling': list_upsamplings()
    }

