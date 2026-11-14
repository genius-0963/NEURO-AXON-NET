# NAX-ADR-002: Primary Task Definition - Density Regression and Count Integration

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: Domain Scientist + ML Lead
- **Phase**: Phase 01 (Discovery & Baseline Specification)

## Context
Axon counting in optic nerve micrographs can be approached via:
1. Object detection (bounding boxes around axons)
2. Semantic / instance segmentation (binary masks separating axoplasm and myelin)
3. Count-density regression (continuous 2D density map whose spatial integral equals the count)

AxoNet 2.0 (2023) explored full segmentation, but independent validation (Chuter et al., 2026) demonstrated severe segmentation degradation under external domain shift (Dice 0.29–0.40; recall 0.18–0.27), while density-based count regression retained higher rank-order fidelity and robustness.

## Decision
1. **Primary Output**: The core output of NeuroAxonNet version 0.1 is an **axon-density map** $D(x, y)$, where pixel values represent local count density scaled by factor $m = 1000$.
2. **Count Estimation via Integration**:
   $$\text{Estimated Count} = \frac{1}{m} \iint_{\Omega} D(x, y) \, dx \, dy = \frac{1}{1000} \sum_{(x,y) \in \text{valid}} D(x, y)$$
3. **Non-negativity Invariant**: The model's terminal activation must be non-negative (ReLU), ensuring density values cannot be negative.
4. **Scope Constraint**: NeuroAxonNet 0.1 explicitly does **not** claim full morphological axon segmentation, axoplasm/myelin ratio profiling, or clinical glaucoma staging.

## Consequences
- **Positive**: Direct scientific replication of published AxoNet results; avoids brittle pixelwise segmentation failures on out-of-domain micrographs.
- **Positive**: Enables smooth sliding-window tile stitching across entire whole-nerve cross-sections via linear blend of overlapping density fields.
- **Limitation**: Individual axon diameters or g-ratios are not extracted in this release (slated for Phase 02+ with dedicated expert annotation masks).
