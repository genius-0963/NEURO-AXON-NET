# Phase 3 Exit Gate Sign-Off Specification

- **Project ID**: NAX-ENG-P03-001
- **Target Milestone**: Phase 3 Exit Gate (Model Architecture & Core Implementation)
- **Status**: Complete / Signed by Engineering Team

## Exit Gate Evaluation Matrix

| Exit Criteria | Verification Evidence | Status | Lead Owner |
| :--- | :--- | :--- | :--- |
| **V0 Model Shape & Dtype** | Forward pass $[B, 1, 224, 224] \to [B, 1, 224, 224]$ non-negative float32 verified. | **VERIFIED** | ML Lead |
| **Non-Negativity Constraint** | Terminal ReLU activation asserts all output density values $\ge 0.0$. | **VERIFIED** | ML Lead |
| **One-Batch Overfit Test** | Gradient backprop and loss convergence verified without NaN or exploding gradients. | **VERIFIED** | ML Lead |
| **Count Function & Masking** | Density integral excludes mirrored 16 px border; count equals $\sum(D) / 1000$. | **VERIFIED** | QA Lead |
| **Checkpoint Reproducibility** | Checkpoints store split manifest hash, model config, and reload cleanly in new sessions. | **VERIFIED** | Tech Lead |
| **Variant Isolation (V1–V4)** | Residual and Attention variants quarantined behind model configuration flags. | **VERIFIED** | Chief Architect |

**VERDICT: APPROVED TO ADVANCE TO PHASE 04 (ROBUSTNESS & TILING).**
