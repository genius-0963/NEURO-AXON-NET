# NAX-ADR-005: Quality Control (QC) and Calibrated Abstention Framework

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: Neurohistology Domain Expert + ML Lead
- **Phase**: Phase 01 (Discovery & Baseline Specification)

## Context
Independent validation studies (e.g. Chuter et al., 2026) revealed that machine learning models for optic nerve analysis degrade significantly under domain shift (microscope lens blur, overstaining, tissue folds, illumination gradients). Rather than outputting false counts with silent overconfidence, a research-grade tool must diagnose image degradation and flag unreliable inputs.

## Decision
1. **Automated QC Pre-Check**: Before inference, each patch or tiled region is audited by `neuroaxonnet.qc.detector`:
   - **Focus / Sharpness Metric**: Variance of Laplacian $\sigma_{\Delta}^2$. Patches with $\sigma_{\Delta}^2 < 120.0$ trigger `BLUR_DETECTED`.
   - **Luminance & Contrast Bounds**: Checks whether mean pixel intensity $\mu \in [40, 220]$ and standard deviation $\sigma > 15$. Overexposed or blank areas trigger `POOR_CONTRAST` or `BACKGROUND_VOID`.
   - **Tissue Coverage Ratio**: Validates the area fraction containing tissue structures vs slide background. Tissue ratio $< 0.35$ triggers `LOW_TISSUE_COVERAGE`.
2. **Three-Tier QC Status**:
   - `PASS`: Clean sample within standard operating range.
   - `REVIEW`: Marginal quality or minor boundary artifacts; count reported alongside advisory alert.
   - `ABSTAIN`: Severe out-of-domain degradation (extreme blur, empty field, fold); count withheld or marked uncalibrated with mandatory human histology review.

## Consequences
- Prevents silent failure modes and builds transparent trust with pathology researchers.
- Provides actionable diagnostic metrics in JSON/CSV exports and UI dashboards.
