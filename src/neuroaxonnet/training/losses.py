"""
Objective Functions and Count Consistency Losses for NeuroAxonNet (Phase 03).

Implements:
- Masked Mean Squared Error (MSE) on valid non-padded pixels (Section 5)
- Optional Count Consistency Smooth-L1 regularizer
- Support for both PyTorch and NumPy arrays
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def masked_mse_loss(
    pred_density: Any,
    target_density: Any,
    valid_mask: Any,
    eps: float = 1e-6,
) -> Any:
    """
    Computes pixelwise MSE over valid pixels only:
        pixel_error = (pred_density - target_density) ** 2
        loss_mse = sum(pixel_error * valid_mask) / max(sum(valid_mask), eps)
    """
    if HAS_TORCH and isinstance(pred_density, torch.Tensor):
        mask = valid_mask.float()
        pixel_error = (pred_density - target_density) ** 2
        denom = torch.clamp(mask.sum(), min=eps)
        return (pixel_error * mask).sum() / denom
    else:
        # NumPy computation
        p = np.asarray(pred_density, dtype=np.float32)
        t = np.asarray(target_density, dtype=np.float32)
        m = np.asarray(valid_mask, dtype=np.float32)
        pixel_error = (p - t) ** 2
        denom = max(float(np.sum(m)), eps)
        return float(np.sum(pixel_error * m) / denom)


def count_consistency_loss(
    pred_density: Any,
    target_density: Any,
    valid_mask: Any,
    density_scale: float = 1000.0,
    beta: float = 1.0,
) -> Tuple[Any, float, float]:
    """
    Computes Smooth L1 error between total estimated axon counts:
        pred_count = sum(pred_density * valid_mask) / density_scale
        target_count = sum(target_density * valid_mask) / density_scale
    """
    if HAS_TORCH and isinstance(pred_density, torch.Tensor):
        mask = valid_mask.float()
        pred_count = (pred_density * mask).sum() / float(density_scale)
        target_count = (target_density * mask).sum() / float(density_scale)
        loss = F.smooth_l1_loss(pred_count, target_count, beta=beta)
        return loss, float(pred_count.item()), float(target_count.item())
    else:
        p = np.asarray(pred_density, dtype=np.float32)
        t = np.asarray(target_density, dtype=np.float32)
        m = np.asarray(valid_mask, dtype=np.float32)
        pred_c = float(np.sum(p * m) / float(density_scale))
        target_c = float(np.sum(t * m) / float(density_scale))
        diff = abs(pred_c - target_c)
        if diff < beta:
            loss = 0.5 * (diff ** 2) / beta
        else:
            loss = diff - 0.5 * beta
        return loss, pred_c, target_c


class AxoNetLoss(nn.Module if HAS_TORCH else object):
    """
    Composite objective function module.
    """
    def __init__(
        self,
        density_scale: float = 1000.0,
        lambda_count: float = 0.0,  # 0.0 for pure faithful baseline
    ):
        if HAS_TORCH:
            super().__init__()
        self.density_scale = density_scale
        self.lambda_count = lambda_count

    def forward(
        self,
        pred_density: Any,
        target_density: Any,
        valid_mask: Any,
    ) -> Dict[str, Any]:
        loss_mse = masked_mse_loss(pred_density, target_density, valid_mask)
        loss_count, pred_c, target_c = count_consistency_loss(
            pred_density, target_density, valid_mask, density_scale=self.density_scale
        )

        total_loss = loss_mse
        if self.lambda_count > 0.0:
            total_loss = total_loss + self.lambda_count * loss_count

        return {
            "loss_total": total_loss,
            "loss_mse": loss_mse,
            "loss_count": loss_count,
            "pred_count": pred_c,
            "target_count": target_c,
            "count_error": abs(pred_c - target_c),
        }
