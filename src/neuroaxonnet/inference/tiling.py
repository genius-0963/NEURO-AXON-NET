"""
Sliding-Window Whole-Slide Tiler with Blended Overlap (Phase 03).

Provides smooth reconstruction of entire optic nerve cross-sections
using Hann/linear boundary tapering to prevent grid edge artifacts.
"""

from __future__ import annotations

from typing import Callable, Tuple, List, Dict, Any
import numpy as np


def create_2d_hann_window(shape: Tuple[int, int]) -> np.ndarray:
    """Generate 2D separable Hann tapering window."""
    h, w = shape
    wy = np.hanning(h)
    wx = np.hanning(w)
    window = np.outer(wy, wx)
    # Ensure center is non-zero
    return np.maximum(window, 1e-4).astype(np.float32)


class WholeSlideTiler:
    def __init__(
        self,
        patch_size: Tuple[int, int] = (192, 192),
        stride: Tuple[int, int] = (96, 96),  # 50% overlap default
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.window = create_2d_hann_window(patch_size)

    def process_large_image(
        self,
        large_image: np.ndarray,
        patch_predictor_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Processes large cross-section image in overlapping tiles and reconstructs
        the unified density map via weighted accumulation.
        """
        h, w = large_image.shape[:2]
        ph, pw = self.patch_size
        sy, sx = self.stride

        # Pad image to integer multiples of stride
        pad_y = (sy - (h - ph) % sy) % sy if h > ph else (ph - h)
        pad_x = (sx - (w - pw) % sx) % sx if w > pw else (pw - w)

        padded_img = np.pad(large_image, ((0, pad_y), (0, pad_x)), mode="reflect")
        pad_h, pad_w = padded_img.shape[:2]

        accum_density = np.zeros((pad_h, pad_w), dtype=np.float32)
        accum_weights = np.zeros((pad_h, pad_w), dtype=np.float32)

        for y0 in range(0, pad_h - ph + 1, sy):
            for x0 in range(0, pad_w - pw + 1, sx):
                patch = padded_img[y0 : y0 + ph, x0 : x0 + pw]
                pred_density = patch_predictor_fn(patch)
                # Accumulate with Hann window weight
                accum_density[y0 : y0 + ph, x0 : x0 + pw] += pred_density * self.window
                accum_weights[y0 : y0 + ph, x0 : x0 + pw] += self.window

        # Normalize by accumulated weights
        recon_density = accum_density / np.maximum(accum_weights, 1e-6)
        # Crop back to original unpadded dimensions
        return recon_density[:h, :w]
