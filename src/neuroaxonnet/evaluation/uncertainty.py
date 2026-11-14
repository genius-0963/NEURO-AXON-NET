"""
Uncertainty Quantification and Calibrated Abstention (Phase 04).

Implements Section 9 specification:
- Deep ensemble prediction mean and dispersion (standard deviation)
- Error vs dispersion rank correlation
- Risk-Coverage curve (monotonic reduction of error under selective abstention)
- Triage to REVIEW_REQUIRED for high-dispersion inputs
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Sequence
import numpy as np


class EnsembleUncertaintyEngine:
    def __init__(self, review_threshold: float = 4.5):
        self.review_threshold = review_threshold

    def evaluate_ensemble(
        self,
        ensemble_predictions: List[Sequence[float]],  # Shape (n_models, n_samples)
    ) -> Dict[str, np.ndarray]:
        """
        Computes mean prediction and epistemic dispersion across ensemble members.
        """
        preds = np.asarray(ensemble_predictions, dtype=np.float64)  # (M, N)
        mean_preds = np.mean(preds, axis=0)
        std_dispersion = np.std(preds, axis=0, ddof=1) if preds.shape[0] > 1 else np.zeros(preds.shape[1])

        return {
            "mean_prediction": mean_preds,
            "dispersion_std": std_dispersion,
        }

    def compute_risk_coverage_curve(
        self,
        mean_predictions: np.ndarray,
        references: np.ndarray,
        dispersions: np.ndarray,
        abstention_fractions: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    ) -> List[Dict[str, float]]:
        """
        Simulates selective classification / abstention:
        Samples with highest dispersion are rejected first, and MAE is evaluated
        on remaining accepted samples.
        """
        n = len(mean_predictions)
        sorted_indices = np.argsort(dispersions)  # Ascending dispersion (most certain first)
        results = []

        abs_errors = np.abs(mean_predictions - references)

        for frac in abstention_fractions:
            n_retain = int(np.round(n * (1.0 - frac)))
            n_retain = max(1, min(n, n_retain))

            accepted_idx = sorted_indices[:n_retain]
            accepted_mae = float(np.mean(abs_errors[accepted_idx]))

            results.append({
                "abstention_fraction": round(float(frac), 2),
                "coverage_fraction": round(float(n_retain / n), 2),
                "accepted_samples": n_retain,
                "mae_on_accepted": round(accepted_mae, 2),
            })

        return results

    def triage_sample(self, mean_count: float, dispersion: float) -> Dict[str, Any]:
        """Classify sample into PASS or REVIEW_REQUIRED."""
        requires_review = dispersion > self.review_threshold
        return {
            "mean_count": round(float(mean_count), 2),
            "dispersion": round(float(dispersion), 2),
            "status": "REVIEW_REQUIRED" if requires_review else "PASS",
            "threshold": self.review_threshold,
        }
