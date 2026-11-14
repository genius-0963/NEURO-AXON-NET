"""
Unit and Integration Tests for Phase 03 Model Architecture & Training Engine.
"""

import pytest
import numpy as np

from neuroaxonnet.models.unet_baseline import NumpyAxoNetForwardEngine
from neuroaxonnet.training.losses import masked_mse_loss, count_consistency_loss, AxoNetLoss
from neuroaxonnet.training.engine import TrainingEngine
from neuroaxonnet.training.checkpoint import CheckpointManager
from neuroaxonnet.training.samplers import CountStratifiedSampler
from neuroaxonnet.inference.predictor import PatchPredictor, crop_to_valid_region
from neuroaxonnet.inference.aggregation import integrate_density
from neuroaxonnet.inference.tiling import WholeSlideTiler, create_2d_hann_window

try:
    import torch
    from neuroaxonnet.models.unet_baseline import AxoNetUNet
    from neuroaxonnet.models.residual_unet import ResidualAxoNetUNet
    from neuroaxonnet.models.attention_unet import AttentionAxoNetUNet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_numpy_forward_non_negative():
    engine = NumpyAxoNetForwardEngine()
    x = np.random.normal(0, 1, size=(2, 1, 224, 224)).astype(np.float32)
    out = engine.forward(x)
    assert out.shape == (2, 1, 224, 224)
    assert out.dtype == np.float32
    assert np.all(out >= 0.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for neural net tests")
def test_pytorch_unet_baseline_forward_shape_and_non_negativity():
    model = AxoNetUNet(in_channels=1, base_channels=32)
    model.eval()
    x = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 224, 224)
    assert torch.all(out >= 0.0)  # ReLU head ensures non-negativity


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for variant tests")
def test_candidate_variants_forward():
    x = torch.randn(1, 1, 224, 224)
    m_res = ResidualAxoNetUNet(in_channels=1, base_channels=16)
    m_att = AttentionAxoNetUNet(in_channels=1, base_channels=16)

    with torch.no_grad():
        out_res = m_res(x)
        out_att = m_att(x)

    assert out_res.shape == (1, 1, 224, 224)
    assert out_att.shape == (1, 1, 224, 224)
    assert torch.all(out_res >= 0.0)
    assert torch.all(out_att >= 0.0)


def test_masked_mse_excludes_padding():
    pred = np.ones((224, 224), dtype=np.float32) * 5.0
    target = np.ones((224, 224), dtype=np.float32) * 5.0

    mask = np.zeros((224, 224), dtype=bool)
    mask[16:208, 16:208] = True  # valid core

    # Zero error on valid core
    loss = masked_mse_loss(pred, target, mask)
    assert np.isclose(loss, 0.0)

    # Error outside valid core in padding region
    pred[0, 0] = 999.0
    loss_masked = masked_mse_loss(pred, target, mask)
    assert np.isclose(loss_masked, 0.0)  # Padding is strictly ignored!


def test_count_stratified_sampler():
    counts = [10.0, 10.0, 10.0, 10.0, 50.0]  # Rare class 50.0
    sampler = CountStratifiedSampler(counts, num_bins=5)
    # Rare count (index 4) should have higher sampling probability than common count (indices 0-3)
    assert sampler.weights[4] > sampler.weights[0]


def test_checkpoint_manager_roundtrip(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=tmp_path)
    state = {"weights": np.ones((5, 5))}
    saved_p = mgr.save_checkpoint(
        model_state_dict=state,
        optimizer_state_dict={},
        epoch=5,
        metrics={"val_loss": 0.42},
        config={"lr": 1e-4},
        split_manifest_hash="abc123sha",
        filename="test_ckpt.pt",
    )
    loaded = mgr.load_checkpoint(saved_p)
    assert loaded["metadata"]["split_manifest_hash"] == "abc123sha"
    assert loaded["metadata"]["metrics"]["val_loss"] == 0.42


def test_patch_predictor_end_to_end():
    engine = NumpyAxoNetForwardEngine()
    predictor = PatchPredictor(model=engine, density_scale=1000.0)
    raw_patch = np.random.uniform(50, 200, size=(187, 187)).astype(np.float32)

    res = predictor.predict_patch(raw_patch)
    assert res["density_map"].shape == (192, 192)
    assert res["estimated_count"] >= 0.0
    assert "qc_report" in res
    assert "RESEARCH PROTOTYPE ONLY" in res["disclaimer"]


def test_whole_slide_tiler_hann_reconstruction():
    tiler = WholeSlideTiler(patch_size=(64, 64), stride=(32, 32))
    large_img = np.ones((128, 128), dtype=np.float32) * 100.0

    def mock_patch_pred(patch):
        return np.ones(patch.shape, dtype=np.float32) * 2.0

    recon = tiler.process_large_image(large_img, mock_patch_pred)
    assert recon.shape == (128, 128)
    # With uniform prediction, Hann window reconstruction equals 2.0 everywhere
    assert np.allclose(recon, 2.0, atol=1e-2)
