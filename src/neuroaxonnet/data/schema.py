"""
Canonical Relational Data Model for NeuroAxonNet (Phase 02).

Implements the 6 primary data entities and the 22-column patch manifest schema.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class EyeSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UNSPECIFIED = "unspecified"


class Specimen(BaseModel):
    specimen_id: str
    animal_id: str = Field(..., description="Unique biological animal identifier")
    species: str = "Rattus norvegicus"
    strain: str = "Brown Norway"
    cohort: str = "Baseline"
    condition: str = "Normal Control"
    site_id: str = "Ethier_Lab_GT"


class OpticNerve(BaseModel):
    nerve_id: str
    specimen_id: str
    eye_side: EyeSide = EyeSide.LEFT
    section_id: str = "sec_01"
    section_protocol: str = "semi_thin_plastic_1um_ppd"


class ImageRecord(BaseModel):
    image_id: str
    source_uri: str
    sha256: str = Field(..., description="Pre-decode SHA-256 hash of raw file bytes")
    width: int
    height: int
    bit_depth: int = 8
    channels: int = 1
    pixel_size_um: Optional[float] = 0.0642  # ~15.7 px / um
    stain: str = "PPD"
    microscope: str = "100x_oil_brightfield"
    acquisition_id: Optional[str] = "acq_01"


class PatchRecord(BaseModel):
    patch_id: str
    image_id: str
    x0: int
    y0: int
    width_px: int = 187
    height_px: int = 187
    physical_width_um: float = 12.0
    split_group_id: str = Field(..., description="Inherited animal/specimen ID for leakage control")
    origin_type: str = "sampled_dataset_patch"  # or "whole_nerve_crop"


class AnnotationPoint(BaseModel):
    x: float = Field(..., description="Continuous X coordinate (column) in pixels")
    y: float = Field(..., description="Continuous Y coordinate (row) in pixels")
    class_name: str = "normal_appearing_axon"


class AnnotationRecord(BaseModel):
    annotation_id: str
    patch_id: str
    annotator_id: str
    label_version: str = "normal_axon_centres_v1"
    coordinate_frame: str = "source_patch_pixels"
    points: List[AnnotationPoint] = Field(default_factory=list)
    edge_rule: str = "full_or_more_than_half_inside_per_protocol"
    review_state: str = "adjudicated"  # "raw", "consensus", "adjudicated"


class TransformRecord(BaseModel):
    transform_id: str = "transform_v1"
    code_commit: str = "git_hash_pinned"
    config_hash: str = "cfg_sha256"
    resize: Tuple[int, int] = (192, 192)
    normalization: str = "two_sigma_clipped_neg1_to_pos1"
    pad: int = 16  # reflect pad to (224, 224)
    target_parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"sigma": 8.0, "kernel_size": 33, "m": 1000.0}
    )


class CanonicalPatchManifestRow(BaseModel):
    """The 22 mandatory fields required by the Phase 02 specification."""
    patch_id: str
    image_id: str
    specimen_id: str
    animal_id: str
    nerve_id: str
    site_id: str
    source_sha256: str
    x0: int
    y0: int
    source_w: int
    source_h: int
    patch_w: int
    patch_h: int
    pixel_size_um: float
    stain: str
    microscope: str
    condition: str
    label_version: str
    split_group: str
    split: SplitType
    transform_version: str
    target_version: str
    ground_truth_count: float = 0.0

    def to_csv_dict(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "image_id": self.image_id,
            "specimen_id": self.specimen_id,
            "animal_id": self.animal_id,
            "nerve_id": self.nerve_id,
            "site_id": self.site_id,
            "source_sha256": self.source_sha256,
            "x0": self.x0,
            "y0": self.y0,
            "source_w": self.source_w,
            "source_h": self.source_h,
            "patch_w": self.patch_w,
            "patch_h": self.patch_h,
            "pixel_size_um": self.pixel_size_um,
            "stain": self.stain,
            "microscope": self.microscope,
            "condition": self.condition,
            "label_version": self.label_version,
            "split_group": self.split_group,
            "split": self.split.value,
            "transform_version": self.transform_version,
            "target_version": self.target_version,
            "ground_truth_count": self.ground_truth_count,
        }
