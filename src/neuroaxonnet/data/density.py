"""
Deterministic Gaussian Density Target Builder.

Implements isotropic 2D Gaussian convolution matching published AxoNet specifications:
- Isotropic Gaussian: sigma = 8.0 pixels
- Filter kernel size: 33 x 33 pixels
- Density scaling factor: m = 1000.0
- Count Invariant: integral(density_map) / m == axon_count
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Sequence


def create_gaussian_kernel(kernel_size: int = 33, sigma: float = 8.0, normalize: bool = True) -> np.ndarray:
    """
    Generate an isotropic 2D Gaussian kernel of specified size and sigma.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {kernel_size}")

    radius = kernel_size // 2
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))

    if normalize:
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel / kernel_sum

    return kernel.astype(np.float32)


def generate_density_map(
    shape: Tuple[int, int],
    points: Sequence[Tuple[float, float] | Sequence[float]],
    sigma: float = 8.0,
    kernel_size: int = 33,
    density_scale: float = 1000.0,
    normalize_kernel: bool = True,
) -> np.ndarray:
    """
    Generate continuous 2D density target map from discrete axon centroid coordinates.

    Args:
        shape: (height, width) of output density map (e.g., 192, 192).
        points: Iterable of (x, y) coordinates where x is horizontal [0, width)
                and y is vertical [0, height).
        sigma: Standard deviation of Gaussian blur (AxoNet default: 8.0).
        kernel_size: Odd integer kernel diameter (AxoNet default: 33).
        density_scale: Multiplier m for count density regression (AxoNet default: 1000.0).
        normalize_kernel: When True, ensures discrete kernel integrates to 1.0 so each
                          interior axon centroid contributes exactly `density_scale` to map integral.

    Returns:
        density_map: np.ndarray of shape (height, width) with float32 values >= 0.0.
    """
    h, w = shape
    density = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return density

    kernel = create_gaussian_kernel(kernel_size=kernel_size, sigma=sigma, normalize=normalize_kernel)
    kernel_scaled = kernel * float(density_scale)
    radius = kernel_size // 2

    for pt in points:
        x, y = float(pt[0]), float(pt[1])
        # Skip points strictly outside image boundaries
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        ix = int(round(x))
        iy = int(round(y))

        # Kernel slice bounds
        k_x_min = max(0, radius - ix)
        k_x_max = min(kernel_size, radius + (w - ix))
        k_y_min = max(0, radius - iy)
        k_y_max = min(kernel_size, radius + (h - iy))

        # Target image slice bounds
        img_x_min = max(0, ix - radius)
        img_x_max = min(w, ix + radius + 1)
        img_y_min = max(0, iy - radius)
        img_y_max = min(h, iy + radius + 1)

        density[img_y_min:img_y_max, img_x_min:img_x_max] += kernel_scaled[k_y_min:k_y_max, k_x_min:k_x_max]

    return density


def integrate_density_to_count(density_map: np.ndarray, density_scale: float = 1000.0, mask: np.ndarray | None = None) -> float:
    """
    Integrate 2D density map to estimate axon count:
        Count = sum(density_map[valid_mask]) / density_scale
    """
    if mask is not None:
        valid_density = density_map[mask > 0]
        total_mass = float(np.sum(valid_density))
    else:
        total_mass = float(np.sum(density_map))

    return max(0.0, total_mass / float(density_scale))
