"""
Unit and Integration Tests for Phase 04 Statistical Agreement, Calibration & Robustness.
"""

import pytest
import numpy as np

from neuroaxonnet.evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_bias,
    compute_r_squared,
    compute_pearson_r,
    compute_concordance_correlation_coefficient,
    compute_bland_altman_limits,
    compute_full_agreement_suite,
    group_aware_bootstrap_ci,
)
from neuroaxonnet.evaluation.calibration import LinearBiasCalibrator
from neuroaxonnet.evaluation.robustness import RobustnessStressTester
from neuroaxonnet.evaluation.uncertainty import EnsembleUncertaintyEngine
from neuroaxonnet.inference.predictor import PatchPredictor
from neuroaxonnet.models.unet_baseline import NumpyAxoNetForwardEngine


def test_concordance_correlation_penalizes_shift():
    ref = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # Exact shift by +10: Pearson r remains 1.0, but Lin's CCC must drop because of location mismatch!
    shifted = ref + 10.0

    r = compute_pearson_r(shifted, ref)
    ccc = compute_concordance_correlation_coefficient(shifted, ref)

    assert np.isclose(r, 1.0)
    assert ccc < 0.90  # CCC penalizes the +10 bias!


def test_bland_altman_limits_calculation():
    ref = np.array([20.0, 30.0, 40.0, 50.0])
    pred = np.array([22.0, 29.0, 43.0, 48.0])
    ba = compute_bland_altman_limits(pred, ref)

    assert "mean_difference" in ba
    assert "upper_limit_95" in ba
    assert "lower_limit_95" in ba
    assert ba["upper_limit_95"] > ba["lower_limit_95"]


def test_group_aware_bootstrap_ci():
    records = [
        {"animal_id": "rat_01", "pred_count": 40.0, "ground_truth_count": 41.0},
        {"animal_id": "rat_01", "pred_count": 42.0, "ground_truth_count": 43.0},
        {"animal_id": "rat_02", "pred_count": 35.0, "ground_truth_count": 36.0},
        {"animal_id": "rat_02", "pred_count": 37.0, "ground_truth_count": 38.0},
    ]
    ci = group_aware_bootstrap_ci(records, n_bootstraps=50, seed=42)
    assert "mae_ci95" in ci
    low, high = ci["mae_ci95"]
    assert low <= high


def test_linear_bias_calibrator_validation_only():
    calibrator = LinearBiasCalibrator()
    # Synthetic systematic undercount: pred = 0.8 * ref + 2.0
    val_ref = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
    val_pred = 0.8 * val_ref + 2.0

    calibrator.fit_on_validation(val_ref, val_pred)
    assert calibrator.fitted is True
    assert np.isclose(calibrator.slope, 0.8, atol=1e-2)

    # Apply to test set without refitting
    test_pred = np.array([26.0, 34.0, 42.0])
    corrected = calibrator.apply(test_pred)
    # Expected corrected: (test_pred - 2.0) / 0.8
    assert np.allclose(corrected, (test_pred - 2.0) / 0.8, atol=1e-2)


def test_ensemble_uncertainty_and_risk_coverage():
    uncertainty_engine = EnsembleUncertaintyEngine(review_threshold=3.0)
    # 3 ensemble models, 5 samples
    m1 = [40.0, 42.0, 38.0, 50.0, 20.0]
    m2 = [41.0, 41.0, 39.0, 55.0, 21.0]
    m3 = [39.0, 43.0, 37.0, 60.0, 19.0]

    res = uncertainty_engine.evaluate_ensemble([m1, m2, m3])
    dispersions = res["dispersion_std"]
    # Sample index 3 has huge disagreement (50, 55, 60) -> high dispersion
    assert dispersions[3] > dispersions[0]

    # Test triage
    triage_high = uncertainty_engine.triage_sample(mean_count=55.0, dispersion=float(dispersions[3]))
    assert triage_high["status"] == "REVIEW_REQUIRED"

    triage_low = uncertainty_engine.triage_sample(mean_count=40.0, dispersion=float(dispersions[0]))
    assert triage_low["status"] == "PASS"

    # Test risk-coverage curve
    refs = np.array([40.0, 42.0, 38.0, 52.0, 20.0])
    curve = uncertainty_engine.compute_risk_coverage_curve(res["mean_prediction"], refs, dispersions)
    assert len(curve) > 0
    # Retaining lowest-dispersion samples should lower or maintain error
    assert curve[-1]["accepted_samples"] < curve[0]["accepted_samples"]


def test_robustness_stress_sweep():
    engine = NumpyAxoNetForwardEngine()
    predictor = PatchPredictor(model=engine, density_scale=1000.0)
    stress_tester = RobustnessStressTester(predictor)

    clean_patches = [np.ones((187, 187), dtype=np.float32) * 120.0 for _ in range(3)]
    gt_counts = [40.0, 40.0, 40.0]

    blur_res = stress_tester.run_blur_stress_sweep(clean_patches, gt_counts, blur_sigmas=(0.0, 2.0))
    assert len(blur_res) == 2
    assert "mae" in blur_res[0]
    assert "abstention_rate" in blur_res[0]
