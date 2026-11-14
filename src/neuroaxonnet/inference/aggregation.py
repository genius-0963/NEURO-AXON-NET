"""
Density Count Aggregation and Spatial Integration.

Implements count = sum(valid_density) / density_scale.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def integrate_density(
    density_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    density_scale: float = 1000.0,
) -> float:
    """
    Computes total axon count via spatial numerical integration:
        count = sum(density_map[valid_mask]) / density_scale
    """
    arr = np.asarray(density_map, dtype=np.float64)
    if valid_mask is not None:
        msk = np.asarray(valid_mask, dtype=bool)
        total_density = float(np.sum(arr[msk]))
    else:
        total_density = float(np.sum(arr))

    return max(0.0, total_density / float(density_scale))
