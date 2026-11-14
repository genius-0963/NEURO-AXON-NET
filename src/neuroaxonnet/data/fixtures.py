"""
Synthetic Optic Nerve Histology Fixture Generator.

Creates mathematically controlled optic nerve patches with known axon centroids,
simulating PPD/toluidine-blue stained light micrographs:
- Realistic annular myelin sheaths and pale axoplasm
- Controlled blur, contrast, and noise
- Specific QC failure cases: Defocused/blurry, empty void, overstained
- Specimen-grouped DatasetManifest for unit, golden, and leakage testing
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageFilter

from neuroaxonnet.data.manifest import AxonPoint, DatasetManifest, PatchRecord, SplitType


def draw_synthetic_axon(
    canvas: np.ndarray,
    center: Tuple[float, float],
    radius: float = 6.0,
    sheath_thickness: float = 2.0,
    axoplasm_val: float = 210.0,
    sheath_val: float = 40.0,
) -> None:
    """Draw a single realistic cross-sectional axon with myelin sheath and axoplasm."""
    h, w = canvas.shape
    cx, cy = center
    y_min = max(0, int(cy - radius - 2))
    y_max = min(h, int(cy + radius + 3))
    x_min = max(0, int(cx - radius - 2))
    x_max = min(w, int(cx + radius + 3))

    y, x = np.ogrid[y_min:y_max, x_min:x_max]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    inner_r_sq = (radius - sheath_thickness) ** 2
    outer_r_sq = radius ** 2

    # Myelin sheath (dark ring)
    sheath_mask = (dist_sq >= inner_r_sq) & (dist_sq <= outer_r_sq)
    canvas[y_min:y_max, x_min:x_max][sheath_mask] = sheath_val

    # Axoplasm (pale inner circle)
    axoplasm_mask = dist_sq < inner_r_sq
    canvas[y_min:y_max, x_min:x_max][axoplasm_mask] = axoplasm_val


def generate_synthetic_micrograph(
    shape: Tuple[int, int] = (187, 187),
    num_axons: int = 45,
    seed: int = 42,
    defect: str = "none",
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Generate a synthetic light microscopy image of an optic nerve patch.
    Defects: 'none', 'blur', 'blank', 'dark', 'washout'.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    if defect == "blank":
        # Background glass slide void (no tissue)
        img = rng.normal(loc=235.0, scale=4.0, size=shape).clip(0, 255).astype(np.float32)
        return img, []

    # Interstitial tissue base
    base_val = 150.0 if defect != "dark" else 35.0
    img = rng.normal(loc=base_val, scale=12.0, size=shape).clip(0, 255)

    # Place non-overlapping axon centroids
    points: List[Tuple[float, float]] = []
    attempts = 0
    max_attempts = num_axons * 50

    while len(points) < num_axons and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(14, w - 14))
        y = float(rng.uniform(14, h - 14))

        # Enforce minimum distance between axons
        if any((x - px) ** 2 + (y - py) ** 2 < 14.0 ** 2 for px, py in points):
            continue

        radius = float(rng.uniform(4.5, 7.5))
        sheath_th = float(rng.uniform(1.8, 2.6))
        draw_synthetic_axon(img, (x, y), radius=radius, sheath_thickness=sheath_th)
        points.append((x, y))

    # Add realistic texture noise
    noise = rng.normal(loc=0.0, scale=5.0, size=shape)
    img = np.clip(img + noise, 0, 255).astype(np.float32)

    # Apply defect transformations
    if defect == "blur":
        pil_img = Image.fromarray(img.astype(np.uint8))
        # Heavy Gaussian defocus blur
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=6.0))
        img = np.array(pil_img, dtype=np.float32)
    elif defect == "dark":
        img = np.clip(img * 0.3, 0, 255).astype(np.float32)
    elif defect == "washout":
        img = np.clip(img * 1.4 + 50.0, 0, 255).astype(np.float32)

    return img, points


def create_fixture_dataset(output_dir: Path | str = "data/fixtures") -> DatasetManifest:
    """
    Build and persist synthetic fixtures for testing:
    - Train: rat_01, rat_02, rat_03
    - Val: rat_04
    - Test: rat_05
    - QC Edge cases: blurry, blank, dark
    """
    base_path = Path(output_dir)
    images_path = base_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    manifest = DatasetManifest(
        manifest_version="1.0.0",
        dataset_name="NeuroAxonNet-Synthetic-Fixtures",
        description="Controlled synthetic test fixtures with ground truth for validation and unit testing.",
    )

    # 1. Standard biological specimens
    animals = {
        "rat_01": (SplitType.TRAIN, 4),
        "rat_02": (SplitType.TRAIN, 4),
        "rat_03": (SplitType.TRAIN, 4),
        "rat_04": (SplitType.VAL, 4),
        "rat_05": (SplitType.TEST, 4),
    }

    seed_idx = 100
    for animal_id, (split, count) in animals.items():
        for i in range(count):
            seed_idx += 1
            patch_id = f"{animal_id}_patch_{i:02d}"
            img_arr, pts = generate_synthetic_micrograph(
                shape=(187, 187), num_axons=int(30 + (seed_idx % 25)), seed=seed_idx
            )
            file_name = f"{patch_id}.png"
            file_path = images_path / file_name

            Image.fromarray(img_arr.astype(np.uint8)).save(file_path)
            sha256 = DatasetManifest.compute_sha256(file_path)

            record = PatchRecord(
                patch_id=patch_id,
                file_path=str(file_path),
                file_sha256=sha256,
                animal_id=animal_id,
                nerve_id=f"{animal_id}_left",
                session_id="session_01",
                coords=(i * 187, 0),
                scale_um_per_pixel=0.0642,
                width=187,
                height=187,
                split=split,
                axon_points=[AxonPoint(x=p[0], y=p[1]) for p in pts],
                ground_truth_count=len(pts),
            )
            manifest.add_patch(record)

    # 2. Defect edge cases for QC tests
    defects = [
        ("defect_blurry", "blur", 40),
        ("defect_blank", "blank", 0),
        ("defect_dark", "dark", 35),
    ]
    for defect_id, defect_type, n_axons in defects:
        seed_idx += 1
        img_arr, pts = generate_synthetic_micrograph(
            shape=(187, 187), num_axons=n_axons, seed=seed_idx, defect=defect_type
        )
        file_name = f"{defect_id}.png"
        file_path = images_path / file_name
        Image.fromarray(img_arr.astype(np.uint8)).save(file_path)
        sha256 = DatasetManifest.compute_sha256(file_path)

        record = PatchRecord(
            patch_id=defect_id,
            file_path=str(file_path),
            file_sha256=sha256,
            animal_id="qc_control_specimen",
            nerve_id="qc_nerve",
            session_id="qc_session",
            coords=(0, 0),
            scale_um_per_pixel=0.0642,
            width=187,
            height=187,
            split=SplitType.TEST,
            axon_points=[AxonPoint(x=p[0], y=p[1]) for p in pts],
            ground_truth_count=len(pts),
        )
        manifest.add_patch(record)

    manifest_file = base_path / "manifest.json"
    manifest.save(manifest_file)
    return manifest


if __name__ == "__main__":
    m = create_fixture_dataset()
    print(f"Created fixture dataset with {m.total_patches} patches at data/fixtures/manifest.json")
