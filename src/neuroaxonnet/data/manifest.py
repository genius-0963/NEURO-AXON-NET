"""
Dataset Manifest and Biological Grouping Module.

Enforces animal-level split isolation and SHA-256 provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class AxonPoint(BaseModel):
    x: float = Field(..., description="X coordinate in pixels [0, W)")
    y: float = Field(..., description="Y coordinate in pixels [0, H)")


class PatchRecord(BaseModel):
    patch_id: str
    file_path: str
    file_sha256: str
    animal_id: str = Field(..., description="Biological specimen ID (primary split unit)")
    nerve_id: str = Field(..., description="Optic nerve identifier (e.g., rat01_left)")
    session_id: str = "session_01"
    coords: Tuple[int, int] = (0, 0)
    scale_um_per_pixel: float = 0.0625  # 12 um / 192 px
    width: int = 192
    height: int = 192
    split: SplitType
    axon_points: List[AxonPoint] = Field(default_factory=list)
    ground_truth_count: int = 0


class DatasetManifest(BaseModel):
    manifest_version: str = "1.0.0"
    dataset_name: str = "NeuroAxonNet-Rat-Baseline"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = "Optic nerve axon counting dataset with strict biological grouping."
    total_patches: int = 0
    patches: List[PatchRecord] = Field(default_factory=list)

    @classmethod
    def compute_sha256(cls, file_path: Path | str) -> str:
        """Compute SHA-256 digest of an image file."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found for checksum: {file_path}")
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def add_patch(self, patch: PatchRecord) -> None:
        self.patches.append(patch)
        self.total_patches = len(self.patches)

    def get_split(self, split: SplitType) -> List[PatchRecord]:
        return [p for p in self.patches if p.split == split]

    def verify_no_leakage(self) -> dict:
        """
        Verify that no biological specimen (animal_id) appears across multiple splits.
        Raises ValueError if biological data leakage is detected.
        """
        train_animals = {p.animal_id for p in self.get_split(SplitType.TRAIN)}
        val_animals = {p.animal_id for p in self.get_split(SplitType.VAL)}
        test_animals = {p.animal_id for p in self.get_split(SplitType.TEST)}

        train_val_overlap = train_animals.intersection(val_animals)
        train_test_overlap = train_animals.intersection(test_animals)
        val_test_overlap = val_animals.intersection(test_animals)

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError(
                f"BIOLOGICAL LEAKAGE DETECTED! "
                f"Train/Val overlap: {train_val_overlap}, "
                f"Train/Test overlap: {train_test_overlap}, "
                f"Val/Test overlap: {val_test_overlap}"
            )

        return {
            "train_animals": sorted(list(train_animals)),
            "val_animals": sorted(list(val_animals)),
            "test_animals": sorted(list(test_animals)),
            "leakage_free": True,
        }

    def summary(self) -> dict:
        leak_audit = self.verify_no_leakage()
        return {
            "total_patches": self.total_patches,
            "train_count": len(self.get_split(SplitType.TRAIN)),
            "val_count": len(self.get_split(SplitType.VAL)),
            "test_count": len(self.get_split(SplitType.TEST)),
            "animals_by_split": {
                "train": leak_audit["train_animals"],
                "val": leak_audit["val_animals"],
                "test": leak_audit["test_animals"],
            },
        }

    def save(self, output_path: Path | str) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, manifest_path: Path | str) -> DatasetManifest:
        p = Path(manifest_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
