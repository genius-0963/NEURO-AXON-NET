"""
Versioned Preprocessing Contract and Augmentation Engine (Phase 02).

Exact reproduction of Section 8 specification:
- Bilinear resize 187x187 -> 192x192
- Normalization: (I - mu) / max(2.0 * sigma, eps) clipped to [-1.0, 1.0]
- Mirror/reflect pad: 192x192 -> 224x224 (pad = 16)
- Explicit valid core mask: [16:208, 16:208]
- Synchronized geometric augmentations for image, point map, density target, and valid mask.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image


def resize_bilinear(image: np.ndarray, target_shape: Tuple[int, int] = (192, 192)) -> np.ndarray:
    """Bilinear resize matching PIL Bilinear convention."""
    if image.shape == target_shape:
        return image.astype(np.float32)
    pil_img = Image.fromarray(image.astype(np.float32))
    # PIL takes (width, height)
    resized = pil_img.resize((target_shape[1], target_shape[0]), resample=Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.float32)


def normalize_robust(image: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Two-sigma standardization clipped to [-1.0, 1.0]:
        I_norm = clip((I - mu) / max(2.0 * sigma, eps), -1.0, 1.0)
    """
    mu = float(np.mean(image))
    sigma = float(np.std(image))
    denom = max(2.0 * sigma, float(epsilon))
    normalized = (image - mu) / denom
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def reflect_pad_image(image: np.ndarray, pad: int = 16) -> np.ndarray:
    """
    Mirror / reflection pad by `pad` pixels on each border.
    Converts (192, 192) to (224, 224).
    """
    return np.pad(image, pad_width=pad, mode="reflect")


def create_valid_core_mask(
    tensor_shape: Tuple[int, int] = (224, 224),
    core_slice: Tuple[slice, slice] = (slice(16, 208), slice(16, 208)),
) -> np.ndarray:
    """
    Generate boolean valid-core mask where only the unpadded region is True.
    """
    mask = np.zeros(tensor_shape, dtype=bool)
    mask[core_slice[0], core_slice[1]] = True
    return mask


def preprocess_baseline(
    raw_image: np.ndarray,
    target_shape: Tuple[int, int] = (192, 192),
    pad: int = 16,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes the canonical Phase 02 preprocessing pipeline:
        raw_image -> resize_bilinear -> normalize_robust -> reflect_pad -> (image_pad, valid_mask)
    """
    img_resized = resize_bilinear(raw_image, target_shape)
    img_norm = normalize_robust(img_resized, epsilon=epsilon)
    img_pad = reflect_pad_image(img_norm, pad=pad)
    valid_mask = create_valid_core_mask(
        tensor_shape=(target_shape[0] + 2 * pad, target_shape[1] + 2 * pad),
        core_slice=(slice(pad, pad + target_shape[0]), slice(pad, pad + target_shape[1])),
    )
    return img_pad.astype(np.float32), valid_mask


class SynchronizedAugmentor:
    """
    Applies identical geometric transforms to image, density target, and valid mask.
    """
    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90: bool = True,
        intensity_scale_range: Tuple[float, float] = (0.85, 1.15),
        seed: Optional[int] = None,
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate90 = rotate90
        self.scale_range = intensity_scale_range
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        image_pad: np.ndarray,
        density_pad: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = image_pad.copy()
        tgt = density_pad.copy()
        msk = valid_mask.copy()

        # Horizontal flip
        if self.rng.random() < self.hflip_prob:
            img = np.fliplr(img)
            tgt = np.fliplr(tgt)
            msk = np.fliplr(msk)

        # Vertical flip
        if self.rng.random() < self.vflip_prob:
            img = np.flipud(img)
            tgt = np.flipud(tgt)
            msk = np.flipud(msk)

        # 90-degree orthogonal rotation
        if self.rotate90:
            k = int(self.rng.integers(0, 4))
            if k > 0:
                img = np.rot90(img, k=k)
                tgt = np.rot90(tgt, k=k)
                msk = np.rot90(msk, k=k)

        # Intensity augmentation (applied strictly to input image, NOT density target)
        scale_val = float(self.rng.uniform(self.scale_range[0], self.scale_range[1]))
        img = np.clip(img * scale_val, -1.0, 1.0)

        return img.astype(np.float32), tgt.astype(np.float32), msk
