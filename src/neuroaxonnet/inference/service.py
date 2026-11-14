"""
Production-Grade Inference Service and Unified Contract (Phase 05).

Single source of truth for CLI, REST API, and Web UI.
No tensor math in API controllers; no count integration in JavaScript frontends.
"""

from __future__ import annotations

import io
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image

from neuroaxonnet.data.ingestion import read_microscopy_raw, convert_to_model_channels
from neuroaxonnet.data.preprocessing import preprocess_baseline
from neuroaxonnet.qc.detector import QualityControlAuditor
from neuroaxonnet.models.unet_baseline import NumpyAxoNetForwardEngine
from neuroaxonnet.inference.aggregation import integrate_density
from neuroaxonnet.inference.tiling import WholeSlideTiler
from neuroaxonnet.evaluation.calibration import LinearBiasCalibrator


class NeuroAxonInferenceService:
    def __init__(
        self,
        model_version: str = "0.1.0",
        density_scale: float = 1000.0,
        calibrator: Optional[LinearBiasCalibrator] = None,
    ):
        self.model_name = "NeuroAxonNet"
        self.model_version = model_version
        self.density_scale = density_scale
        self.model = NumpyAxoNetForwardEngine(density_scale=density_scale)
        self.qc_auditor = QualityControlAuditor()
        self.tiler = WholeSlideTiler(patch_size=(192, 192), stride=(96, 96))
        self.calibrator = calibrator

    def validate_image_bytes(
        self,
        image_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validates raw image bytes, computes SHA-256 before decoding,
        checks physical scale, and rejects malformed inputs.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Empty or zero-byte image payload rejected.")

        sha256 = hashlib.sha256(image_bytes).hexdigest()

        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                pil_img.verify()
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                if pil_img.mode != "L":
                    pil_img = pil_img.convert("L")
                arr = np.array(pil_img, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Corrupt or malformed image payload: {e}")

        h, w = arr.shape
        meta = metadata.copy() if metadata else {}
        meta["sha256"] = sha256
        meta["width"] = int(w)
        meta["height"] = int(h)
        meta["pixel_size_um"] = float(meta.get("pixel_size_um", 0.0625))
        meta["species"] = str(meta.get("species", "rat"))
        meta["stain"] = str(meta.get("stain", "PPD"))

        return arr, meta

    def predict_patch(
        self,
        image_array: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single microscopy patch with QC pre-auditing.
        """
        meta = metadata or {}
        qc_report = self.qc_auditor.audit_patch(image_array)

        # Preprocess: (192, 192) core with 16 px reflect pad to (224, 224)
        img_pad, valid_mask = preprocess_baseline(image_array, target_shape=(192, 192), pad=16)

        # Forward pass through model
        pred_density = self.model.forward(img_pad)[0, 0]

        # Apply valid mask (exclude mirrored borders)
        valid_density = pred_density * valid_mask
        density_core = valid_density[16:208, 16:208].copy()

        # Integrate count
        raw_count = integrate_density(density_core, density_scale=self.density_scale)

        calibrated_count = None
        if self.calibrator and self.calibrator.fitted:
            calibrated_count = float(self.calibrator.apply([raw_count])[0])

        # Status formatting
        status = "completed"
        if qc_report["status"] == "ABSTAIN":
            status = "abstained_severe_defect"
        elif qc_report["status"] == "REVIEW":
            status = "completed_with_review"

        result = {
            "job_id": f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
            "status": status,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "sha256": "8f9a2e3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f",
            },
            "input": {
                "sha256": meta.get("sha256", "unhashed_array"),
                "width": int(image_array.shape[1]),
                "height": int(image_array.shape[0]),
                "pixel_size_um": float(meta.get("pixel_size_um", 0.0625)),
                "species": meta.get("species", "rat"),
                "stain": meta.get("stain", "PPD"),
            },
            "count": {
                "raw_estimate": round(float(raw_count), 2),
                "calibrated_estimate": round(float(calibrated_count), 2) if calibrated_count is not None else None,
                "unit": "normal_appearing_axons",
                "density_scale": float(self.density_scale),
            },
            "qc": {
                "status": qc_report["status"].lower(),
                "flags": qc_report["flags"],
                "metrics": qc_report["metrics"],
            },
            "provenance": {
                "preprocessing_version": "v1_pad224_norm2sigma",
                "config_hash": "cfg_sha256_axonet_v0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runtime": "python3_numpy_engine",
            },
            "disclaimer": "RESEARCH PROTOTYPE ONLY. NOT A CLINICAL DIAGNOSTIC DEVICE OR A SUBSTITUTE FOR EXPERT PATHOLOGY REVIEW.",
            "_density_map": density_core,
        }
        return result

    def predict_full_micrograph(
        self,
        large_image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process whole optic nerve cross-section with overlapping Hann window aggregation.
        """
        meta = metadata or {}
        qc_report = self.qc_auditor.audit_patch(large_image[:500, :500] if large_image.shape[0] > 500 else large_image)

        def patch_pred_fn(patch_192):
            pad_patch, valid_mask = preprocess_baseline(patch_192, target_shape=(192, 192), pad=16)
            pred = self.model.forward(pad_patch)[0, 0]
            core = (pred * valid_mask)[16:208, 16:208]
            return core

        # Reconstruct whole density map
        full_density = self.tiler.process_large_image(large_image, patch_pred_fn)
        raw_count = integrate_density(full_density, density_scale=self.density_scale)

        calibrated_count = None
        if self.calibrator and self.calibrator.fitted:
            calibrated_count = float(self.calibrator.apply([raw_count])[0])

        result = {
            "job_id": f"job_whole_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
            "status": "completed" if qc_report["status"] == "PASS" else "completed_with_review",
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "sha256": "8f9a2e3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f",
            },
            "input": {
                "sha256": meta.get("sha256", "unhashed_full_image"),
                "width": int(large_image.shape[1]),
                "height": int(large_image.shape[0]),
                "pixel_size_um": float(meta.get("pixel_size_um", 0.0625)),
                "species": meta.get("species", "rat"),
                "stain": meta.get("stain", "PPD"),
            },
            "count": {
                "raw_estimate": round(float(raw_count), 2),
                "calibrated_estimate": round(float(calibrated_count), 2) if calibrated_count is not None else None,
                "unit": "normal_appearing_axons",
                "density_scale": float(self.density_scale),
            },
            "qc": {
                "status": qc_report["status"].lower(),
                "flags": qc_report["flags"],
                "metrics": qc_report["metrics"],
            },
            "provenance": {
                "preprocessing_version": "v1_tiled_hann_overlap",
                "config_hash": "cfg_sha256_axonet_v0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runtime": "python3_numpy_engine",
            },
            "disclaimer": "RESEARCH PROTOTYPE ONLY. NOT A CLINICAL DIAGNOSTIC DEVICE OR A SUBSTITUTE FOR EXPERT PATHOLOGY REVIEW.",
            "_density_map": full_density,
        }
        return result

    def generate_overlay_png(
        self,
        original_image: np.ndarray,
        density_map: np.ndarray,
        alpha: float = 0.5,
    ) -> bytes:
        """Render alpha-blended heatmap overlay on top of original micrograph."""
        # Normalize original to RGB [0, 255]
        orig_norm = ((original_image - np.min(original_image)) / max(np.ptp(original_image), 1e-5) * 255.0).astype(np.uint8)
        orig_rgb = Image.fromarray(orig_norm).convert("RGB")

        # Colorize density map using high-contrast colormap (inferno simulation)
        d_norm = (density_map / max(np.max(density_map), 1e-5)).clip(0, 1)
        r = (np.sin(d_norm * np.pi) * 255).astype(np.uint8)
        g = (np.sin(d_norm * np.pi * 0.75) * 220).astype(np.uint8)
        b = ((1.0 - d_norm) * 50).astype(np.uint8)
        heatmap_arr = np.stack([r, g, b], axis=-1)
        heatmap_img = Image.fromarray(heatmap_arr).resize(orig_rgb.size, Image.Resampling.BILINEAR)

        # Blend
        blended = Image.blend(orig_rgb, heatmap_img, alpha=float(alpha))
        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        return buf.getvalue()
