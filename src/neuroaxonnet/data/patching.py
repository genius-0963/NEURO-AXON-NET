"""
Deterministic Patch Generator and Coordinate Transformer.

Implements deterministic cropping, physical scale tracking, and coordinate mapping.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image

from neuroaxonnet.data.schema import AnnotationPoint, PatchRecord


def extract_patch(
    source_image: np.ndarray,
    x0: int,
    y0: int,
    patch_w: int = 187,
    patch_h: int = 187,
) -> np.ndarray:
    """
    Extract a spatial sub-image patch deterministically using top-left coordinates.
    """
    h, w = source_image.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + patch_w > w or y0 + patch_h > h:
        raise ValueError(
            f"Patch bounds [{x0}:{x0+patch_w}, {y0}:{y0+patch_h}] exceed image bounds [{w}, {h}]"
        )
    return source_image[y0 : y0 + patch_h, x0 : x0 + patch_w].copy()


def transform_patch_and_points(
    patch: np.ndarray,
    points: List[AnnotationPoint],
    target_shape: Tuple[int, int] = (192, 192),
) -> Tuple[np.ndarray, List[AnnotationPoint]]:
    """
    Bilinearly resizes patch from source size (e.g. 187x187) to target size (192x192),
    scaling all continuous annotation coordinates proportionally.
    """
    orig_h, orig_w = patch.shape[:2]
    target_h, target_w = target_shape

    if (orig_h, orig_w) == (target_h, target_w):
        return patch.copy(), [p.model_copy() for p in points]

    scale_x = float(target_w) / float(orig_w)
    scale_y = float(target_h) / float(orig_h)

    # Resize image array using bilinear interpolation
    pil_img = Image.fromarray(patch.astype(np.float32))
    resized_pil = pil_img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32)

    # Scale coordinates with full float precision
    scaled_points: List[AnnotationPoint] = []
    for pt in points:
        nx = pt.x * scale_x
        ny = pt.y * scale_y
        scaled_points.append(
            AnnotationPoint(x=float(nx), y=float(ny), class_name=pt.class_name)
        )

    return resized_arr, scaled_points


def filter_points_half_open(
    points: List[AnnotationPoint],
    width: int = 192,
    height: int = 192,
) -> List[AnnotationPoint]:
    """
    Enforces the half-open boundary policy: [0, width) and [0, height)
    ensuring border centroids are counted exactly once across adjacent tiles.
    """
    valid = []
    for pt in points:
        if 0.0 <= pt.x < float(width) and 0.0 <= pt.y < float(height):
            valid.append(pt)
    return valid
