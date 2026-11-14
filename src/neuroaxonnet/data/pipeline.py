"""
Master Phase 02 Data Engineering Pipeline.

Converts raw microscopy images and annotations into model-ready patches, targets,
and leakage-safe locked split manifests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from neuroaxonnet.data.schema import (
    AnnotationPoint,
    AnnotationRecord,
    CanonicalPatchManifestRow,
    ImageRecord,
    OpticNerve,
    PatchRecord,
    Specimen,
    SplitType,
    TransformRecord,
)
from neuroaxonnet.data.ingestion import read_microscopy_raw, convert_to_model_channels
from neuroaxonnet.data.patching import extract_patch, transform_patch_and_points, filter_points_half_open
from neuroaxonnet.data.annotation_qa import build_consensus_point_map
from neuroaxonnet.data.preprocessing import preprocess_baseline
from neuroaxonnet.data.density_builder import build_gaussian_density_target, verify_numeric_invariants
from neuroaxonnet.data.split_auditor import BiologicalSplitAuditor


class Phase2DataPipeline:
    def __init__(self, base_data_dir: Path | str = "data"):
        self.base_dir = Path(base_data_dir)
        self.raw_dir = self.base_dir / "raw"
        self.patches_dir = self.base_dir / "patches" / "v1"
        self.targets_dir = self.base_dir / "targets" / "v1"
        self.annotations_dir = self.base_dir / "annotations" / "normal_axon_centres_v1"
        self.splits_dir = self.base_dir / "splits" / "v1"
        self.metadata_dir = self.base_dir / "metadata"
        self.qc_dir = self.base_dir / "qc"

        # Create required directories
        for d in [self.raw_dir, self.patches_dir, self.targets_dir, self.annotations_dir, self.splits_dir, self.metadata_dir, self.qc_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def process_fixtures_to_canonical(self, fixtures_manifest_path: Path | str = "data/fixtures/manifest.json") -> Dict[str, Any]:
        """
        Processes generated fixtures through the canonical Phase 02 pipeline.
        """
        p = Path(fixtures_manifest_path)
        if not p.is_file():
            raise FileNotFoundError(f"Fixtures manifest not found at {fixtures_manifest_path}")

        with open(p, "r", encoding="utf-8") as f:
            fix_data = json.load(f)

        canonical_rows: List[CanonicalPatchManifestRow] = []

        # Run numeric invariant check first
        inv_check = verify_numeric_invariants()
        if not inv_check["invariant_1_pass"] or not inv_check["invariant_2_k_annotator_invariance"]:
            raise RuntimeError(f"Numeric invariants failed: {inv_check}")

        for p_dict in fix_data.get("patches", []):
            patch_id = p_dict["patch_id"]
            img_path = p_dict["file_path"]
            animal_id = p_dict["animal_id"]
            split_str = p_dict["split"]
            split_val = SplitType(split_str)
            raw_pts = [AnnotationPoint(x=pt["x"], y=pt["y"]) for pt in p_dict.get("axon_points", [])]

            # 1. Read raw image
            raw_arr, meta = read_microscopy_raw(img_path)
            mono_arr = convert_to_model_channels(raw_arr)

            # 2. Extract patch (already patch sized in fixtures) & transform (187 -> 192)
            resized_patch, scaled_points = transform_patch_and_points(
                mono_arr, raw_pts, target_shape=(192, 192)
            )
            filtered_points = filter_points_half_open(scaled_points, 192, 192)

            # 3. Save annotation record
            anno_rec = AnnotationRecord(
                annotation_id=f"anno_{patch_id}",
                patch_id=patch_id,
                annotator_id="A01_Expert",
                points=filtered_points,
                review_state="adjudicated",
            )
            anno_file = self.annotations_dir / f"{patch_id}.json"
            with open(anno_file, "w", encoding="utf-8") as af:
                af.write(anno_rec.model_dump_json(indent=2))

            # 4. Build density target (192, 192)
            pt_map = build_consensus_point_map((192, 192), [anno_rec])
            density_target = build_gaussian_density_target(
                pt_map, sigma=8.0, truncate=2.0, density_scale=1000.0
            )
            
            # Save target npy
            target_file = self.targets_dir / f"{patch_id}.npy"
            np.save(target_file, density_target)

            # 5. Preprocess baseline (pad to 224x224, valid mask [16:208, 16:208])
            img_pad, valid_mask = preprocess_baseline(resized_patch, target_shape=(192, 192), pad=16)

            # Save model patch npz (contains image_pad and valid_mask)
            patch_file = self.patches_dir / f"{patch_id}.npz"
            np.savez_compressed(patch_file, image=img_pad, valid_mask=valid_mask)

            # 6. Build canonical row
            row = CanonicalPatchManifestRow(
                patch_id=patch_id,
                image_id=f"img_{patch_id}",
                specimen_id=f"specimen_{animal_id}",
                animal_id=animal_id,
                nerve_id=p_dict.get("nerve_id", f"{animal_id}_nerve"),
                site_id="Ethier_Lab_GT",
                source_sha256=meta["source_sha256"],
                x0=p_dict["coords"][0],
                y0=p_dict["coords"][1],
                source_w=meta["width"],
                source_h=meta["height"],
                patch_w=192,
                patch_h=192,
                pixel_size_um=0.0625,
                stain="PPD",
                microscope="100x_oil_brightfield",
                condition="Normal Control",
                label_version="normal_axon_centres_v1",
                split_group=animal_id,
                split=split_val,
                transform_version="v1_pad224_norm2sigma",
                target_version="v1_sigma8_trunc2_m1000",
                ground_truth_count=float(len(filtered_points)),
            )
            canonical_rows.append(row)

        # 7. Audit leakage and export split CSVs
        auditor = BiologicalSplitAuditor(canonical_rows)
        audit_report = auditor.audit_splits()
        split_hashes = auditor.export_split_csvs(self.splits_dir)

        # 8. Save canonical manifest
        manifest_meta = {
            "pipeline_version": "2.0.0",
            "total_records": len(canonical_rows),
            "split_hashes": split_hashes,
            "leakage_audit": audit_report,
            "numeric_invariants": inv_check,
        }
        with open(self.metadata_dir / "canonical_manifest.json", "w", encoding="utf-8") as mf:
            json.dump(manifest_meta, mf, indent=2)

        return manifest_meta
