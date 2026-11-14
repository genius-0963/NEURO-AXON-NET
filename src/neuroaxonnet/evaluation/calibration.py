"""
Linear Bias Calibration Protocol (Phase 04).

Implements Section 7 specification:
- Linear bias model fitted exclusively on validation set
- Preserves raw outputs and computes calibrated outputs side-by-side
- Strictly prohibits test set label refitting
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Sequence
import numpy as np


class LinearBiasCalibrator:
    def __init__(self):
        self.slope: float = 1.0       # a
        self.intercept: float = 0.0   # b
        self.fitted: bool = False
        self.fit_metrics: Dict[str, float] = {}

    def fit_on_validation(
        self,
        val_references: Sequence[float],
        val_predictions: Sequence[float],
    ) -> LinearBiasCalibrator:
        """
        Fit linear relationship on validation split only:
            predicted = a * reference + b
            corrected = (predicted - b) / a
        """
        ref = np.asarray(val_references, dtype=np.float64)
        pred = np.asarray(val_predictions, dtype=np.float64)

        if len(ref) < 2:
            self.slope = 1.0
            self.intercept = 0.0
            self.fitted = True
            return self

        # Ordinary Least Squares: pred = a * ref + b
        cov_matrix = np.cov(ref, pred)
        var_ref = cov_matrix[0, 0]
        cov_ref_pred = cov_matrix[0, 1]

        if var_ref > 1e-8:
            a = cov_ref_pred / var_ref
            b = float(np.mean(pred) - a * np.mean(ref))
        else:
            a = 1.0
            b = 0.0

        # Guard against zero/negative slope
        if abs(a) < 1e-4:
            a = 1.0
            b = 0.0

        self.slope = float(a)
        self.intercept = float(b)
        self.fitted = True

        raw_mae = float(np.mean(np.abs(pred - ref)))
        calibrated_val = self.apply(pred)
        calibrated_mae = float(np.mean(np.abs(calibrated_val - ref)))

        self.fit_metrics = {
            "validation_slope_a": round(self.slope, 4),
            "validation_intercept_b": round(self.intercept, 4),
            "val_raw_mae": round(raw_mae, 2),
            "val_calibrated_mae": round(calibrated_mae, 2),
        }
        return self

    def apply(self, automated_counts: Sequence[float] | np.ndarray) -> np.ndarray:
        """
        Applies calibration: corrected = max(0, (automated - b) / a)
        """
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted on validation data.")
        arr = np.asarray(automated_counts, dtype=np.float64)
        corrected = (arr - self.intercept) / self.slope
        return np.maximum(0.0, corrected)
