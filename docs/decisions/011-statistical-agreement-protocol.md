# NAX-ADR-011: Statistical Agreement Protocol and Rejection of Correlation Fallacy

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: Statistician + Research Lead
- **Phase**: Phase 04 (Training, Experimentation, Evaluation & Robustness)

## Context
A major pitfall identified in independent replication studies (e.g., Chuter et al., 2026) is reliance solely on Pearson correlation ($r$). A model can have an $r = 0.95$ while systematically undercounting axons by 50% or having massive bias.

## Decision
1. **Mandatory Agreement Suite**:
   All benchmark evaluations must report:
   - MAE and RMSE
   - Mean Signed Error (Bias)
   - Lin's Concordance Correlation Coefficient (CCC)
   - Bland-Altman Limits of Agreement (mean diff $\pm 1.96 \cdot \text{SD}$)
2. **Specimen-Level Group-Aware Bootstrap**:
   Confidence intervals must resample at the `animal_id` level, never treating nested patches from the same biological nerve as independent observations.
3. **Validation-Only Linear Calibration**:
   Linear calibration models ($y_{\text{calib}} = \frac{\hat{y} - b}{a}$) must be fitted exclusively on the validation split. Re-fitting calibration parameters on locked test data is strictly prohibited. Both raw and calibrated test metrics must be reported side by side.

## Consequences
- Provides true concordance against ground truth rather than relative trend ranking.
- Eliminates statistical overconfidence from non-independent patch sampling.
