# NAX-ADR-010: Baseline Model Architecture, Output Contract and Variant Guardrails

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: ML Lead + Tech Lead
- **Phase**: Phase 03 (Model Architecture & Core Implementation)

## Context
AxoNet adapts a U-Net encoder-decoder to regress continuous 2D count-density maps rather than binary segmentation masks. Architectural claims must be disciplined: the initial goal is faithful reproduction of the baseline objective before introducing candidate variants.

## Decision
1. **V0 Baseline Architecture**:
   - One grayscale channel $[B, 1, 224, 224]$ ($192 \times 192$ valid core + 16 px reflection padding).
   - Reduced-width channel progression: 32 -> 64 -> 128 -> 256 with 512 bottleneck.
   - Density head: $1\times 1$ conv to 1 channel with ReLU activation. Sigmoid and softmax are strictly prohibited.
2. **Loss Invariant**:
   - Primary loss is pixelwise Masked MSE over the valid core mask `valid_mask[16:208, 16:208]`. Mirrored context pixels are excluded from the loss.
   - Count consistency regularizer ($\lambda_{\text{count}}$) is quarantined behind an experimental ablation flag and kept at 0.0 for baseline reproduction.
3. **Novelty Guardrails**:
   - V1 (Residual U-Net) and V2 (Attention U-Net) are candidate engineering variants, not defensible standalone novelty claims. They remain isolated behind configuration flags.

## Consequences
- Guarantees that performance numbers can be directly compared against published historical baselines.
- Eliminates confounding factors between architecture tweaks and preprocessing contracts.
