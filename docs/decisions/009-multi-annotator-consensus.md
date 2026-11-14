# NAX-ADR-009: Multi-Annotator Consensus and Point Matching Formulation

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: Annotation Lead + Neurohistology Expert
- **Phase**: Phase 02 (Data Engineering & Preprocessing)

## Context
Axon annotations exhibit inter-observer variability: different experts may place centroids slightly offset by 1–3 pixels or disagree on ambiguous degenerated structures near borders.

## Decision
1. **Indicator Map Averaging**:
   When $K$ annotators mark patch $P$, the discrete point mass at $(x, y)$ is:
   $$\text{point\_map}[\text{round}(y), \text{round}(x)] += \frac{1.0}{K}$$
   Applying Gaussian filter $G_{\sigma=8}$ to this normalized map guarantees that $K$ identical annotator marks do not inflate the total axon count.
2. **Inter-Annotator Point Localization & Matching**:
   - For QA evaluation, points between annotator pairs are matched using bipartite minimum-cost matching within an expert tolerance $\delta_{\max} = 5.0$ pixels ($\approx 0.31\,\mu\text{m}$).
   - Metrics computed: Precision, Recall, F1-score, and Mean Localization Error ($\text{px}$).
3. **Adjudication Routing**:
   Disagreements with absolute count difference $> 5$ axons or F1 $< 0.85$ are flagged for domain expert adjudication.

## Consequences
- Provides mathematically sound multi-expert ground truth.
- Supplies quantitative inter-rater reliability metrics required for peer review.
