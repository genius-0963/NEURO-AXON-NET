"""
Phase 02 Density Target Builder and Invariant Validator.

Implements:
- Point indicator map averaging across K annotators
- Isotropic Gaussian filter (sigma = 8.0, truncate = 2.0 -> 33 px filter)
- Density scaling by m = 1000.0
- Assertion of all 5 numeric invariants from Section 7.1
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.ndimage import gaussian_filter

from neuroaxonnet.data.schema import AnnotationPoint, AnnotationRecord


def build_gaussian_density_target(
    point_map: np.ndarray,
    sigma: float = 8.0,
    truncate: float = 2.0,  # 2.0 * 8.0 = 16 px radius -> 33 px kernel diameter
    mode: str = "constant",
    density_scale: float = 1000.0,
) -> np.ndarray:
    """
    Converts 2D point indicator map to continuous count-density target.

    Args:
        point_map: 2D array of shape (H, W) containing averaged point counts.
        sigma: Standard deviation (default 8.0).
        truncate: Truncation multiplier (default 2.0, yielding exactly 33 px filter).
        mode: Boundary mode for convolution ('constant' with cval=0.0 prevents false edge bleeding).
        density_scale: Multiplier m (default 1000.0).

    Returns:
        scaled_target: np.ndarray of shape (H, W), dtype float32, all >= 0.0.
    """
    # Verify non-negativity and finite values
    if not np.all(point_map >= 0.0):
        raise ValueError("Point map contains negative values.")
    if not np.all(np.isfinite(point_map)):
        raise ValueError("Point map contains NaN or Inf.")

    # Apply scipy Gaussian filter
    density = gaussian_filter(
        point_map.astype(np.float64),
        sigma=sigma,
        truncate=truncate,
        mode=mode,
        cval=0.0,
    )

    # Scale by m = 1000.0
    scaled_target = float(density_scale) * density
    return np.maximum(0.0, scaled_target).astype(np.float32)


def verify_numeric_invariants(
    shape: Tuple[int, int] = (192, 192),
    sigma: float = 8.0,
    truncate: float = 2.0,
    density_scale: float = 1000.0,
) -> Dict[str, Any]:
    """
    Executes the 5 mandatory Section 7.1 numerical invariant tests.
    """
    h, w = shape
    results = {}

    # Invariant 1: Single interior point produces non-negative smooth target
    # whose integral before scaling is close to 1.0 (scipy discrete Gaussian captures ~95.4% at 2 sigma)
    pt_map1 = np.zeros(shape, dtype=np.float64)
    pt_map1[h // 2, w // 2] = 1.0
    target1 = build_gaussian_density_target(pt_map1, sigma=sigma, truncate=truncate, density_scale=1.0)
    integral1 = float(np.sum(target1))
    # At truncate=2.0 (2 sigma in both directions), discrete Gaussian sum is approx 0.9545
    results["single_point_non_negative"] = bool(np.all(target1 >= 0.0))
    results["single_point_integral"] = round(integral1, 4)
    results["invariant_1_pass"] = bool(results["single_point_non_negative"] and (0.90 <= integral1 <= 1.05))

    # Invariant 2: K identical annotator marks averaged by K must not multiply target count by K
    k = 3
    pt_map_k = np.zeros(shape, dtype=np.float64)
    for _ in range(k):
        pt_map_k[h // 2, w // 2] += 1.0 / float(k)
    target_k = build_gaussian_density_target(pt_map_k, sigma=sigma, truncate=truncate, density_scale=density_scale)
    results["invariant_2_k_annotator_invariance"] = bool(np.allclose(target_k, target1 * density_scale, atol=1e-5))

    # Invariant 3: Linearity / Superposition: Density(A + B) == Density(A) + Density(B)
    pt_map_a = np.zeros(shape, dtype=np.float64)
    pt_map_b = np.zeros(shape, dtype=np.float64)
    pt_map_a[50, 50] = 1.0
    pt_map_b[140, 140] = 1.0
    target_a = build_gaussian_density_target(pt_map_a, sigma=sigma, truncate=truncate, density_scale=density_scale)
    target_b = build_gaussian_density_target(pt_map_b, sigma=sigma, truncate=truncate, density_scale=density_scale)
    target_ab = build_gaussian_density_target(pt_map_a + pt_map_b, sigma=sigma, truncate=truncate, density_scale=density_scale)
    results["invariant_3_superposition"] = bool(np.allclose(target_ab, target_a + target_b, atol=1e-4))

    # Invariant 4: Border point edge-inclusion behavior
    pt_map_border = np.zeros(shape, dtype=np.float64)
    pt_map_border[0, 0] = 1.0  # Top-left corner
    target_border = build_gaussian_density_target(pt_map_border, sigma=sigma, truncate=truncate, density_scale=1.0)
    integral_border = float(np.sum(target_border))
    # In 'constant' mode (cval=0), roughly 1/4 of Gaussian mass falls inside the quadrant
    results["border_corner_integral"] = round(integral_border, 4)
    results["invariant_4_border_documented"] = bool(0.20 <= integral_border <= 0.30)

    # Invariant 5: Data type, shape, finite checks
    results["dtype"] = str(target1.dtype)
    results["shape"] = list(target1.shape)
    results["finite_check"] = bool(np.all(np.isfinite(target1)))
    results["invariant_5_pass"] = bool(target1.dtype == np.float32 and results["finite_check"])

    return results
