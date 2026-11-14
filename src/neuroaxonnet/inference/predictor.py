"""
Inference Predictor Engine and Interface Contract (Phase 03).

Implements Section 8 & 9 inference contract:
- Preprocess raw patch (resize 192, norm, 224 reflect pad, valid mask)
- Model inference with non-negativity constraint
- Unmasking and cropping to original observed region
- Density count integration: count = sum(valid_density) / density_scale
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, Optional
import numpy as np

from neuroaxonnet.data.preprocessing import preprocess_baseline
from neuroaxonnet.inference.aggregation import integrate_density
from neuroaxonnet.qc.detector import QualityControlAuditor

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def crop_to_valid_region(
    padded_tensor: np.ndarray,
    pad: int = 16,
    target_shape: Tuple[int, int] = (192, 192),
) -> np.ndarray:
    """
    Crops out the 16-pixel context reflection padding to return the core valid region.
    """
    if padded_tensor.ndim == 2:
        return padded_tensor[pad : pad + target_shape[0], pad : pad + target_shape[1]].copy()
    elif padded_tensor.ndim == 3:
        return padded_tensor[:, pad : pad + target_shape[0], pad : pad + target_shape[1]].copy()
    elif padded_tensor.ndim == 4:
        return padded_tensor[:, :, pad : pad + target_shape[0], pad : pad + target_shape[1]].copy()
    else:
        raise ValueError(f"Unexpected dimensionality: {padded_tensor.ndim}")


class PatchPredictor:
    def __init__(
        self,
        model: Any,
        density_scale: float = 1000.0,
        pad: int = 16,
    ):
        self.model = model
        self.density_scale = density_scale
        self.pad = pad
        self.qc = QualityControlAuditor()

    def predict_patch(
        self,
        image: np.ndarray,
        return_qc: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes the exact Phase 03 inference contract (Section 8 & 9).
        """
        qc_report = self.qc.audit_patch(image) if return_qc else None

        # 1. Preprocess: resize to (192, 192), norm, reflect pad to (224, 224), valid mask
        img_pad, valid_mask = preprocess_baseline(image, target_shape=(192, 192), pad=self.pad)

        # 2. Forward pass with non-negative density output
        if HAS_TORCH and isinstance(self.model, torch.nn.Module):
            self.model.eval()
            with torch.no_grad():
                t_in = torch.from_numpy(img_pad).float().unsqueeze(0).unsqueeze(0)
                pred_out = self.model(t_in)[0, 0].cpu().numpy()
        else:
            # Fallback NumPy execution engine
            pred_out = self.model.forward(img_pad)[0, 0]

        # 3. Apply valid region mask (exclude mirrored context padding)
        valid_density = pred_out * valid_mask

        # 4. Crop to observed core valid region (192, 192)
        density_map = crop_to_valid_region(valid_density, pad=self.pad, target_shape=(192, 192))

        # 5. Integrate density to count: sum(density) / density_scale
        count = integrate_density(density_map, density_scale=self.density_scale)

        return {
            "density_map": density_map.astype(np.float32),
            "estimated_count": float(count),
            "density_scale": float(self.density_scale),
            "qc_report": qc_report,
            "disclaimer": "RESEARCH PROTOTYPE ONLY. NOT A CLINICAL DIAGNOSTIC DEVICE.",
        }
