"""
Synthetic 2D Scalar Field Generators

Creates reproducible test data for exhaustive prior art exploration.
Each generator produces fields with known characteristics.
"""

import numpy as np
from typing import Tuple, Optional


def generate_gaussian_blobs(
    shape: Tuple[int, int] = (256, 256),
    n_blobs: int = 10,
    min_sigma: float = 10,
    max_sigma: float = 50,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate field with random Gaussian blobs.
    
    Useful for testing blob detection methods.
    """
    rng = np.random.default_rng(seed)
    
    field = np.zeros(shape, dtype=np.float64)
    y, x = np.ogrid[:shape[0], :shape[1]]
    
    for _ in range(n_blobs):
        cy = rng.uniform(0, shape[0])
        cx = rng.uniform(0, shape[1])
        sigma = rng.uniform(min_sigma, max_sigma)
        amplitude = rng.uniform(0.5, 1.5)
        
        blob = amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        field += blob
    
    return field


def generate_sine_waves(
    shape: Tuple[int, int] = (256, 256),
    frequencies: Tuple[float, ...] = (0.02, 0.05, 0.1),
    amplitudes: Tuple[float, ...] = (1.0, 0.5, 0.25),
    angles: Tuple[float, ...] = (0, 45, 90),
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate field with superimposed sine waves at different frequencies/angles.
    
    Useful for testing frequency-domain methods.
    """
    field = np.zeros(shape, dtype=np.float64)
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    
    for freq, amp, angle in zip(frequencies, amplitudes, angles):
        angle_rad = np.deg2rad(angle)
        phase = x * np.cos(angle_rad) + y * np.sin(angle_rad)
        field += amp * np.sin(2 * np.pi * freq * phase)
    
    return field


def generate_perlin_noise(
    shape: Tuple[int, int] = (256, 256),
    scale: float = 50,
    octaves: int = 4,
    persistence: float = 0.5,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate Perlin-like noise via spectral synthesis.
    
    Useful for natural-looking terrain/texture testing.
    """
    rng = np.random.default_rng(seed)
    
    field = np.zeros(shape, dtype=np.float64)
    
    for octave in range(octaves):
        freq = 2 ** octave
        amp = persistence ** octave
        
        # Generate random phase for this octave
        noise_shape = (int(shape[0] / scale * freq) + 2, 
                       int(shape[1] / scale * freq) + 2)
        noise = rng.random(noise_shape)
        
        # Upsample to full resolution
        from scipy.ndimage import zoom
        upsampled = zoom(noise, (shape[0] / noise.shape[0], shape[1] / noise.shape[1]), order=3)
        upsampled = upsampled[:shape[0], :shape[1]]
        
        field += amp * upsampled
    
    # Normalize to 0-1
    field = (field - field.min()) / (field.max() - field.min() + 1e-10)
    
    return field


def generate_step_functions(
    shape: Tuple[int, int] = (256, 256),
    n_steps: int = 5,
    orientation: str = 'horizontal',
    noise_level: float = 0.05,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate field with step discontinuities.
    
    Useful for testing edge-preserving methods.
    """
    rng = np.random.default_rng(seed)
    
    if orientation == 'horizontal':
        step_positions = np.sort(rng.choice(shape[0], n_steps, replace=False))
        values = rng.uniform(0, 1, n_steps + 1)
        
        field = np.zeros(shape, dtype=np.float64)
        prev_pos = 0
        for i, pos in enumerate(step_positions):
            field[prev_pos:pos, :] = values[i]
            prev_pos = pos
        field[prev_pos:, :] = values[-1]
    
    elif orientation == 'vertical':
        step_positions = np.sort(rng.choice(shape[1], n_steps, replace=False))
        values = rng.uniform(0, 1, n_steps + 1)
        
        field = np.zeros(shape, dtype=np.float64)
        prev_pos = 0
        for i, pos in enumerate(step_positions):
            field[:, prev_pos:pos] = values[i]
            prev_pos = pos
        field[:, prev_pos:] = values[-1]
    
    else:  # radial
        y, x = np.ogrid[:shape[0], :shape[1]]
        cy, cx = shape[0] // 2, shape[1] // 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        max_r = np.sqrt(cx**2 + cy**2)
        r_normalized = r / max_r
        
        field = np.zeros(shape, dtype=np.float64)
        for i in range(n_steps):
            threshold = (i + 1) / (n_steps + 1)
            field[r_normalized >= threshold] = rng.uniform(0, 1)
    
    # Add noise
    if noise_level > 0:
        field += rng.normal(0, noise_level, shape)
    
    return field


def generate_fractal_terrain(
    shape: Tuple[int, int] = (256, 256),
    roughness: float = 0.5,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate fractal terrain using diamond-square algorithm.
    
    Useful for terrain-like test data.
    """
    rng = np.random.default_rng(seed)
    
    # Find next power of 2
    size = max(shape)
    power = int(np.ceil(np.log2(size)))
    n = 2 ** power + 1
    
    terrain = np.zeros((n, n), dtype=np.float64)
    
    # Initialize corners
    terrain[0, 0] = rng.random()
    terrain[0, n-1] = rng.random()
    terrain[n-1, 0] = rng.random()
    terrain[n-1, n-1] = rng.random()
    
    step = n - 1
    scale = 1.0
    
    while step > 1:
        half = step // 2
        
        # Diamond step
        for y in range(half, n - 1, step):
            for x in range(half, n - 1, step):
                avg = (terrain[y - half, x - half] +
                       terrain[y - half, x + half] +
                       terrain[y + half, x - half] +
                       terrain[y + half, x + half]) / 4
                terrain[y, x] = avg + rng.uniform(-scale, scale)
        
        # Square step
        for y in range(0, n, half):
            for x in range((y + half) % step, n, step):
                count = 0
                total = 0
                if y >= half:
                    total += terrain[y - half, x]
                    count += 1
                if y + half < n:
                    total += terrain[y + half, x]
                    count += 1
                if x >= half:
                    total += terrain[y, x - half]
                    count += 1
                if x + half < n:
                    total += terrain[y, x + half]
                    count += 1
                terrain[y, x] = total / count + rng.uniform(-scale, scale)
        
        step = half
        scale *= roughness
    
    # Crop to requested shape
    terrain = terrain[:shape[0], :shape[1]]
    
    # Normalize
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-10)
    
    return terrain


def generate_gradient_field(
    shape: Tuple[int, int] = (256, 256),
    direction: str = 'diagonal',
    curvature: float = 0.0
) -> np.ndarray:
    """
    Generate linear or curved gradient field.
    
    Useful for testing trend removal methods.
    """
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    
    # Normalize to 0-1
    x_norm = x / (shape[1] - 1)
    y_norm = y / (shape[0] - 1)
    
    if direction == 'horizontal':
        field = x_norm
    elif direction == 'vertical':
        field = y_norm
    elif direction == 'diagonal':
        field = (x_norm + y_norm) / 2
    else:  # radial
        cx, cy = 0.5, 0.5
        field = np.sqrt((x_norm - cx)**2 + (y_norm - cy)**2)
    
    # Add curvature
    if curvature != 0:
        field = field + curvature * (x_norm - 0.5)**2 + curvature * (y_norm - 0.5)**2
    
    # Normalize
    field = (field - field.min()) / (field.max() - field.min() + 1e-10)
    
    return field


def generate_mixed_features(
    shape: Tuple[int, int] = (256, 256),
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate field with mixed feature types: blobs, edges, waves, noise.
    
    Comprehensive test field for all method types.
    """
    rng = np.random.default_rng(seed)
    
    # Base gradient (trend)
    gradient = generate_gradient_field(shape, 'diagonal', curvature=0.3)
    
    # Add blobs
    blobs = generate_gaussian_blobs(shape, n_blobs=5, seed=seed)
    
    # Add waves
    waves = generate_sine_waves(shape, frequencies=(0.03,), amplitudes=(0.2,), angles=(30,))
    
    # Add noise
    noise = rng.normal(0, 0.05, shape)
    
    # Add some step edges
    steps = generate_step_functions(shape, n_steps=2, orientation='radial', 
                                     noise_level=0, seed=seed)
    
    # Combine
    field = 0.3 * gradient + 0.3 * blobs + 0.2 * waves + 0.1 * steps + noise
    
    # Normalize
    field = (field - field.min()) / (field.max() - field.min() + 1e-10)
    
    return field


def generate_test_suite(
    shape: Tuple[int, int] = (256, 256),
    output_dir: Optional[str] = None,
    seed: int = 42
) -> dict:
    """
    Generate complete test suite of synthetic fields.
    
    Returns dict of {name: array} or saves to output_dir.
    """
    fields = {
        'gaussian_blobs': generate_gaussian_blobs(shape, seed=seed),
        'sine_waves': generate_sine_waves(shape, seed=seed),
        'perlin_noise': generate_perlin_noise(shape, seed=seed),
        'step_horizontal': generate_step_functions(shape, orientation='horizontal', seed=seed),
        'step_radial': generate_step_functions(shape, orientation='radial', seed=seed),
        'fractal_terrain': generate_fractal_terrain(shape, seed=seed),
        'gradient_diagonal': generate_gradient_field(shape, 'diagonal'),
        'gradient_radial': generate_gradient_field(shape, 'radial', curvature=0.5),
        'mixed_features': generate_mixed_features(shape, seed=seed),
    }
    
    if output_dir:
        from pathlib import Path
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        for name, field in fields.items():
            np.save(out_path / f"{name}.npy", field.astype(np.float32))
        
        print(f"Saved {len(fields)} test fields to {out_path}")
    
    return fields


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic test fields')
    parser.add_argument('--output', '-o', type=str, default='./test_fields',
                        help='Output directory')
    parser.add_argument('--size', type=int, default=256,
                        help='Field size (square)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    generate_test_suite(
        shape=(args.size, args.size),
        output_dir=args.output,
        seed=args.seed
    )

