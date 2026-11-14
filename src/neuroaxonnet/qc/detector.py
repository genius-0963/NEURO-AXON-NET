"""
Automated Microscopy Quality Control and Abstention Module.

Audits individual patches and whole micrographs for:
- Variance of Laplacian (focus / optical defocus blur)
- Mean luminance and contrast standard deviation (exposure bounds)
- Tissue foreground mask ratio (background slide void detection)
- Assigns tri-state status: PASS, REVIEW, ABSTAIN
"""

from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
from scipy.ndimage import laplace


class QualityControlAuditor:
    def __init__(
        self,
        blur_review_threshold: float = 120.0,
        blur_abstain_threshold: float = 60.0,
        min_tissue_ratio: float = 0.35,
        min_mean_luminance: float = 40.0,
        max_mean_luminance: float = 220.0,
    ):
        self.blur_review_threshold = blur_review_threshold
        self.blur_abstain_threshold = blur_abstain_threshold
        self.min_tissue_ratio = min_tissue_ratio
        self.min_mean_luminance = min_mean_luminance
        self.max_mean_luminance = max_mean_luminance

    def audit_patch(self, image: np.ndarray) -> Dict[str, Any]:
        """Audit single patch for blur, exposure, and background void."""
        laplacian_var = float(np.var(laplace(image.astype(np.float64))))
        mean_val = float(np.mean(image))
        std_val = float(np.std(image))
        # Tissue mask: non-white and non-pitch-black
        tissue_mask = (image < 225.0) & (image > 20.0)
        tissue_ratio = float(np.mean(tissue_mask))

        flags: List[str] = []
        status = "PASS"

        if laplacian_var < self.blur_abstain_threshold:
            status = "ABSTAIN"
            flags.append("SEVERE_DEFOCUS_BLUR")
        elif laplacian_var < self.blur_review_threshold:
            status = "REVIEW"
            flags.append("BORDERLINE_BLUR")

        if tissue_ratio < self.min_tissue_ratio:
            status = "ABSTAIN"
            flags.append("BACKGROUND_SLIDE_VOID")

        if mean_val < self.min_mean_luminance:
            if status != "ABSTAIN":
                status = "REVIEW"
            flags.append("SEVERE_OVERSTAIN")
        elif mean_val > self.max_mean_luminance:
            if status != "ABSTAIN":
                status = "REVIEW"
            flags.append("WASHOUT_UNDERSTAIN")

        return {
            "status": status,
            "flags": flags,
            "metrics": {
                "laplacian_focus": round(laplacian_var, 2),
                "mean_luminance": round(mean_val, 2),
                "contrast_std": round(std_val, 2),
                "tissue_ratio": round(tissue_ratio, 3),
            },
        }
