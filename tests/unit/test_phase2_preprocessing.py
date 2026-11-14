"""
Unit tests for the Section 8 Preprocessing Contract.
"""

import pytest
import numpy as np

from neuroaxonnet.data.preprocessing import (
    resize_bilinear,
    normalize_robust,
    reflect_pad_image,
    create_valid_core_mask,
    preprocess_baseline,
    SynchronizedAugmentor,
)


def test_resize_bilinear_shape():
    raw = np.random.uniform(0, 255, size=(187, 187)).astype(np.float32)
    resized = resize_bilinear(raw, (192, 192))
    assert resized.shape == (192, 192)
    assert resized.dtype == np.float32


def test_normalize_robust_clipping():
    img = np.random.normal(loc=120.0, scale=30.0, size=(192, 192)).astype(np.float32)
    # Add an extreme outlier
    img[0, 0] = 10000.0
    normed = normalize_robust(img)
    assert normed.shape == (192, 192)
    assert np.all(normed >= -1.0)
    assert np.all(normed <= 1.0)
    assert normed[0, 0] == 1.0  # Clipped at positive boundary


def test_reflect_pad_and_valid_mask():
    core = np.ones((192, 192), dtype=np.float32) * 0.5
    padded = reflect_pad_image(core, pad=16)
    assert padded.shape == (224, 224)

    mask = create_valid_core_mask(tensor_shape=(224, 224), core_slice=(slice(16, 208), slice(16, 208)))
    assert mask.shape == (224, 224)
    assert np.sum(mask) == 192 * 192
    assert np.all(mask[16:208, 16:208])
    assert not np.any(mask[:16, :])
    assert not np.any(mask[208:, :])


def test_preprocess_baseline_pipeline():
    raw = np.random.uniform(50, 200, size=(187, 187)).astype(np.float32)
    pad_img, mask = preprocess_baseline(raw, target_shape=(192, 192), pad=16)
    assert pad_img.shape == (224, 224)
    assert mask.shape == (224, 224)
    assert np.all(pad_img >= -1.0)
    assert np.all(pad_img <= 1.0)


def test_synchronized_augmentations():
    img = np.linspace(-0.5, 0.5, 224 * 224, dtype=np.float32).reshape((224, 224))
    tgt = np.ones((224, 224), dtype=np.float32)
    mask = create_valid_core_mask()

    augmentor = SynchronizedAugmentor(
        hflip_prob=1.0, vflip_prob=0.0, rotate90=False,
        intensity_scale_range=(1.0, 1.0), seed=42
    )
    aug_img, aug_tgt, aug_mask = augmentor(img, tgt, mask)

    # Fliplr check
    assert np.allclose(aug_img, np.fliplr(img))
    assert np.allclose(aug_tgt, np.fliplr(tgt))
    assert np.allclose(aug_mask, np.fliplr(mask))
