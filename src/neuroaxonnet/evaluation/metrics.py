"""
Primary Quantitative Agreement Metrics and Group-Aware Statistical Suite (Phase 04).

Implements Section 5 & 6 specifications:
- MAE, RMSE, Mean Signed Error (Bias)
- Pearson r and R^2
- Lin's Concordance Correlation Coefficient (CCC)
- Bland-Altman Limits of Agreement (mean diff +/- 1.96 * SD)
- Stratified Count Range Metrics
- Specimen-level (animal_id) Group-Aware Bootstrap 95% Confidence Intervals
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np


def compute_mae(predictions: np.ndarray, references: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - references)))


def compute_rmse(predictions: np.ndarray, references: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - references) ** 2)))


def compute_bias(predictions: np.ndarray, references: np.ndarray) -> float:
    """Mean Signed Error: mean(pred - ref)."""
    return float(np.mean(predictions - references))


def compute_r_squared(predictions: np.ndarray, references: np.ndarray) -> float:
    """Coefficient of Determination R^2."""
    ss_res = np.sum((references - predictions) ** 2)
    ss_tot = np.sum((references - np.mean(references)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_pearson_r(predictions: np.ndarray, references: np.ndarray) -> float:
    if len(predictions) < 2:
        return 1.0
    p_dev = predictions - np.mean(predictions)
    r_dev = references - np.mean(references)
    denom = np.sqrt(np.sum(p_dev ** 2) * np.sum(r_dev ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(p_dev * r_dev) / denom)


def compute_concordance_correlation_coefficient(
    predictions: np.ndarray,
    references: np.ndarray,
) -> float:
    """
    Lin's Concordance Correlation Coefficient (CCC):
        CCC = (2 * s_xy) / (s_x^2 + s_y^2 + (mean_x - mean_y)^2)
    Measures agreement relative to the 45-degree identity line.
    """
    if len(predictions) < 2:
        return 1.0
    x = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(references, dtype=np.float64)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    cov_xy = np.mean((x - mean_x) * (y - mean_y)) * (len(x) / (len(x) - 1.0))

    denom = var_x + var_y + (mean_x - mean_y) ** 2
    if denom == 0:
        return 1.0
    return float((2.0 * cov_xy) / denom)


def compute_bland_altman_limits(
    predictions: np.ndarray,
    references: np.ndarray,
) -> Dict[str, float]:
    """
    Bland-Altman analysis:
        diff = pred - ref
        mean_diff = mean(diff)
        sd_diff = std(diff)
        limits = mean_diff +/- 1.96 * sd_diff
    """
    diffs = np.asarray(predictions, dtype=np.float64) - np.asarray(references, dtype=np.float64)
    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    return {
        "mean_difference": round(mean_d, 2),
        "sd_difference": round(sd_d, 2),
        "upper_limit_95": round(mean_d + 1.96 * sd_d, 2),
        "lower_limit_95": round(mean_d - 1.96 * sd_d, 2),
    }


def compute_full_agreement_suite(
    predictions: Sequence[float],
    references: Sequence[float],
) -> Dict[str, Any]:
    """
    Runs the complete quantitative statistical agreement suite (Section 5).
    """
    p = np.asarray(predictions, dtype=np.float64)
    r = np.asarray(references, dtype=np.float64)

    mae = compute_mae(p, r)
    rmse = compute_rmse(p, r)
    bias = compute_bias(p, r)
    r2 = compute_r_squared(p, r)
    pearson = compute_pearson_r(p, r)
    ccc = compute_concordance_correlation_coefficient(p, r)
    ba = compute_bland_altman_limits(p, r)

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "bias": round(bias, 2),
        "r_squared": round(r2, 4),
        "pearson_r": round(pearson, 4),
        "concordance_correlation_ccc": round(ccc, 4),
        "bland_altman": ba,
    }


def group_aware_bootstrap_ci(
    records: List[Dict[str, Any]],
    pred_key: str = "pred_count",
    ref_key: str = "ground_truth_count",
    group_key: str = "animal_id",
    n_bootstraps: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% Confidence Intervals using biological specimen (animal_id)
    as the primary resampling unit (Section 6).
    """
    rng = np.random.default_rng(seed)
    # Group records by specimen
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        g = rec[group_key]
        groups.setdefault(g, []).append(rec)

    unique_groups = list(groups.keys())
    n_groups = len(unique_groups)

    boot_mae = []
    boot_bias = []
    boot_ccc = []

    for _ in range(n_bootstraps):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        sampled_records = []
        for g in sampled_groups:
            sampled_records.extend(groups[g])

        p = np.array([r[pred_key] for r in sampled_records], dtype=np.float64)
        t = np.array([r[ref_key] for r in sampled_records], dtype=np.float64)

        boot_mae.append(compute_mae(p, t))
        boot_bias.append(compute_bias(p, t))
        boot_ccc.append(compute_concordance_correlation_coefficient(p, t))

    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)

    return {
        "mae_ci95": (round(float(np.percentile(boot_mae, low_pct)), 2), round(float(np.percentile(boot_mae, high_pct)), 2)),
        "bias_ci95": (round(float(np.percentile(boot_bias, low_pct)), 2), round(float(np.percentile(boot_bias, high_pct)), 2)),
        "ccc_ci95": (round(float(np.percentile(boot_ccc, low_pct)), 4), round(float(np.percentile(boot_ccc, high_pct)), 4)),
    }
