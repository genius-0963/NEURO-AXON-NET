# Model Card — NeuroAxonNet V0 Baseline

- **Model ID**: `neuroaxonnet-v0-baseline`
- **Release Version**: 0.1.0 (Phase 04 Evaluation & Robustness Release)
- **Author**: NeuroAxonNet AI Engineering Team
- **Date**: 2026-11-14

---

## 1. Intended Use and Context
- **Primary Function**: Automated estimation of continuous 2D count-density maps and total normal-appearing axon counts in semi-thin cross-sections of rat optic nerve micrographs.
- **Intended Audience**: Preclinical vision researchers, neuroscientists studying glaucoma models and retinal ganglion cell (RGC) neuroprotection.
- **Regulatory Status**: **RESEARCH PROTOTYPE ONLY. NOT A CLINICAL DIAGNOSTIC DEVICE OR A SUBSTITUTE FOR EXPERT PATHOLOGY REVIEW.**

---

## 2. Inclusions and Strict Exclusions
- **In Scope**:
  - Light microscopy images of rat optic nerve stained with p-phenylenediamine (PPD) or toluidine blue.
  - $12 \times 12\,\mu\text{m}$ sub-images ($187\times 187 \to 192\times 192$ px at $\sim 0.0625\,\mu\text{m}/\text{px}$).
- **Explicit Exclusions**:
  - Human clinical diagnostic use.
  - Automated cross-species extrapolation (e.g. human, NHP, mouse) without separate ground truth validation.
  - Degenerated axon debris quantification (labels cover normal-appearing axons only).
  - Morphology profiling (diameter, g-ratio) without expert segmentation masks.

---

## 3. Data Provenance & Training Pipeline
- **Dataset**: Ethier Laboratory Optic Nerve Axon Dataset (14 Brown Norway rats, 27 optic nerves, 1,514 sub-images).
- **Partitioning**: Specimen-level hierarchical splitting (`animal_id`) guaranteeing 0% biological leakage.
- **Preprocessing Contract**: Bilinear resize to $192\times 192$, robust standardization $(I-\mu)/(2\sigma)$ clipped to $[-1, 1]$, reflection padding to $224\times 224$ with boolean valid-core masking `[16:208, 16:208]`.
- **Target Parameters**: Isotropic Gaussian blur $\sigma = 8.0$, filter size $33\times 33$, density scale multiplier $m = 1000.0$.

---

## 4. Quantitative Performance Summary (Locked Internal Test)
- **Mean Absolute Error (MAE)**: $\le 4.4$ axons per patch.
- **Pearson Correlation ($r$)**: $\ge 0.938$.
- **Lin's Concordance Correlation (CCC)**: $\ge 0.930$.
- **Systematic Bias**: $\le \pm 0.5$ axons after validation-derived linear calibration.
- **Bland-Altman 95% Limits of Agreement**: $[-6.5, +6.2]$ axons.

---

## 5. Automated Quality Control & Abstention Triage
- **Focus Detection**: Patches with variance of Laplacian $\sigma_{\Delta}^2 < 60$ automatically trigger `ABSTAIN` (Severe Blur).
- **Background Voids**: Patches with tissue coverage $< 35\%$ trigger `ABSTAIN` (Slide Void).
- **Stain Extremes**: Luminance $< 40$ or $> 220$ triggers `REVIEW_REQUIRED`.
