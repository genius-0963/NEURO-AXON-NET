"""
Microscopy Patch Dataset and Versioned Preprocessing Contract.

Implements the AxoNet data pipeline:
- Scale preservation: 187 x 187 -> 192 x 192 bilinear resize
- Per-image normalization: (I - mu) / (sigma + eps)
- Mirror / reflection padding
- Augmentations: flips, 90-degree rotations, pixel intensity scaling [0.85, 1.15]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from neuroaxonnet.data.density import generate_density_map, integrate_density_to_count
from neuroaxonnet.data.manifest import DatasetManifest, PatchRecord, SplitType


def load_image_as_grayscale(file_path: Union[str, Path]) -> np.ndarray:
    """Load image from path, handling 8-bit, 16-bit, RGB, or grayscale formats."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Microscopy image file not found: {file_path}")

    with Image.open(p) as img:
        # Convert RGB/RGBA to grayscale L
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img, dtype=np.float32)

    return arr


def resize_patch(image: np.ndarray, target_shape: Tuple[int, int] = (192, 192)) -> np.ndarray:
    """
    Resize image to target shape using high-fidelity bilinear resampling.
    Original AxoNet convention: 187x187 -> 192x192.
    """
    if image.shape == target_shape:
        return image.copy()

    pil_img = Image.fromarray(image.astype(np.float32))
    # PIL expects (width, height)
    resized_pil = pil_img.resize((target_shape[1], target_shape[0]), resample=Image.Resampling.BILINEAR)
    return np.array(resized_pil, dtype=np.float32)


def normalize_per_image(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-image standardization (zero mean, unit variance):
        I_norm = (I - mean) / (std + eps)
    """
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / (std + eps)


def apply_mirror_padding(image: np.ndarray, pad_width: int = 16) -> np.ndarray:
    """
    Apply reflection / mirror padding around image borders.
    """
    return np.pad(image, pad_width=pad_width, mode="reflect")


class MicroscopyTransform:
    """
    Versioned Preprocessing & Augmentation Pipeline.
    """
    def __init__(
        self,
        target_shape: Tuple[int, int] = (192, 192),
        normalize: bool = True,
        is_training: bool = False,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        seed: Optional[int] = None,
    ):
        self.target_shape = target_shape
        self.normalize = normalize
        self.is_training = is_training
        self.scale_range = scale_range
        self.rng = np.random.default_rng(seed)

    def __call__(
        self, image: np.ndarray, points: Optional[List[Tuple[float, float]]] = None
    ) -> Tuple[np.ndarray, Optional[List[Tuple[float, float]]]]:
        # 1. Scale check and resize
        orig_h, orig_w = image.shape
        out_image = resize_patch(image, self.target_shape)

        scale_y = self.target_shape[0] / orig_h
        scale_x = self.target_shape[1] / orig_w

        transformed_points = None
        if points is not None:
            transformed_points = [(p[0] * scale_x, p[1] * scale_y) for p in points]

        # 2. Augmentations (training only)
        if self.is_training:
            # Random horizontal flip
            if self.rng.random() > 0.5:
                out_image = np.fliplr(out_image)
                if transformed_points is not None:
                    w = self.target_shape[1]
                    transformed_points = [(w - 1.0 - p[0], p[1]) for p in transformed_points]

            # Random vertical flip
            if self.rng.random() > 0.5:
                out_image = np.flipud(out_image)
                if transformed_points is not None:
                    h = self.target_shape[0]
                    transformed_points = [(p[0], h - 1.0 - p[1]) for p in transformed_points]

            # Random 90 deg rotation (0, 90, 180, 270)
            k = int(self.rng.integers(0, 4))
            if k > 0:
                out_image = np.rot90(out_image, k=k)
                if transformed_points is not None:
                    h, w = self.target_shape
                    pts = []
                    for px, py in transformed_points:
                        if k == 1:
                            pts.append((py, w - 1.0 - px))
                        elif k == 2:
                            pts.append((w - 1.0 - px, h - 1.0 - py))
                        elif k == 3:
                            pts.append((h - 1.0 - py, px))
                    transformed_points = pts

            # Pixel scaling factor [0.85, 1.15]
            scale_factor = self.rng.uniform(self.scale_range[0], self.scale_range[1])
            out_image = out_image * scale_factor

        # 3. Standardization
        if self.normalize:
            out_image = normalize_per_image(out_image)

        return out_image.astype(np.float32), transformed_points


class AxonPatchDataset:
    """
    Dataset iterating over patches registered in a DatasetManifest.
    """
    def __init__(
        self,
        manifest: DatasetManifest,
        split: SplitType,
        transform: Optional[MicroscopyTransform] = None,
        sigma: float = 8.0,
        kernel_size: int = 33,
        density_scale: float = 1000.0,
    ):
        self.manifest = manifest
        self.split = split
        self.records = manifest.get_split(split)
        self.transform = transform or MicroscopyTransform(is_training=(split == SplitType.TRAIN))
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.density_scale = density_scale

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, dict, float]]:
        record = self.records[idx]
        raw_img = load_image_as_grayscale(record.file_path)
        points = [(pt.x, pt.y) for pt in record.axon_points]

        proc_img, proc_pts = self.transform(raw_img, points)

        density_map = generate_density_map(
            shape=proc_img.shape,
            points=proc_pts if proc_pts is not None else [],
            sigma=self.sigma,
            kernel_size=self.kernel_size,
            density_scale=self.density_scale,
        )

        axon_count = integrate_density_to_count(density_map, density_scale=self.density_scale)

        return {
            "image": proc_img[np.newaxis, ...],  # (1, H, W)
            "density_map": density_map[np.newaxis, ...],  # (1, H, W)
            "count": float(axon_count),
            "ground_truth_count": float(record.ground_truth_count),
            "patch_id": record.patch_id,
            "animal_id": record.animal_id,
            "nerve_id": record.nerve_id,
        }
