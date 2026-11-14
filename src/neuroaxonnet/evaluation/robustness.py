"""
Multi-Axis Robustness and Domain-Shift Evaluation Suite (Phase 04).

Implements Section 8 specification:
- Stain/intensity multiplicative & additive perturbations
- Optical defocus blur sweep
- Noise injection
- Automated QC triage response across perturbation levels
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Callable
import numpy as np
from PIL import Image, ImageFilter

from neuroaxonnet.qc.detector import QualityControlAuditor
from neuroaxonnet.inference.predictor import PatchPredictor
from neuroaxonnet.evaluation.metrics import compute_mae, compute_bias


class RobustnessStressTester:
    def __init__(self, predictor: PatchPredictor):
        self.predictor = predictor
        self.qc = QualityControlAuditor()

    def run_blur_stress_sweep(
        self,
        clean_patches: List[np.ndarray],
        ground_truth_counts: List[float],
        blur_sigmas: Sequence[float] = (0.0, 1.0, 2.0, 4.0, 6.0),
    ) -> List[Dict[str, Any]]:
        """Evaluate counting error and QC triage behavior under increasing optical blur."""
        results = []

        for sigma in blur_sigmas:
            pred_counts = []
            qc_statuses = {"PASS": 0, "REVIEW": 0, "ABSTAIN": 0}

            for patch in clean_patches:
                if sigma == 0.0:
                    perturbed = patch.copy()
                else:
                    pil_img = Image.fromarray(patch.astype(np.uint8))
                    blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
                    perturbed = np.array(blurred_pil, dtype=np.float32)

                res = self.predictor.predict_patch(perturbed, return_qc=True)
                pred_counts.append(res["estimated_count"])
                st = res["qc_report"]["status"]
                qc_statuses[st] = qc_statuses.get(st, 0) + 1

            p_arr = np.array(pred_counts)
            r_arr = np.array(ground_truth_counts)
            mae = compute_mae(p_arr, r_arr)
            bias = compute_bias(p_arr, r_arr)

            results.append({
                "blur_sigma": sigma,
                "mae": round(mae, 2),
                "bias": round(bias, 2),
                "qc_breakdown": qc_statuses,
                "abstention_rate": round((qc_statuses["ABSTAIN"] + qc_statuses["REVIEW"]) / len(clean_patches), 2),
            })

        return results

    def run_stain_intensity_sweep(
        self,
        clean_patches: List[np.ndarray],
        ground_truth_counts: List[float],
        scales: Sequence[float] = (0.7, 0.85, 1.0, 1.15, 1.3),
    ) -> List[Dict[str, Any]]:
        """Evaluate counting stability under stain intensity fluctuations."""
        results = []
        r_arr = np.array(ground_truth_counts)

        for s in scales:
            pred_counts = []
            qc_statuses = {"PASS": 0, "REVIEW": 0, "ABSTAIN": 0}

            for patch in clean_patches:
                scaled = np.clip(patch * s, 0.0, 255.0)
                res = self.predictor.predict_patch(scaled, return_qc=True)
                pred_counts.append(res["estimated_count"])
                st = res["qc_report"]["status"]
                qc_statuses[st] = qc_statuses.get(st, 0) + 1

            p_arr = np.array(pred_counts)
            mae = compute_mae(p_arr, r_arr)
            bias = compute_bias(p_arr, r_arr)

            results.append({
                "intensity_multiplier": s,
                "mae": round(mae, 2),
                "bias": round(bias, 2),
                "qc_breakdown": qc_statuses,
            })

        return results
