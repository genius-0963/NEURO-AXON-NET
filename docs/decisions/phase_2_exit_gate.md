# Phase 2 Exit Gate Sign-Off Specification

- **Project ID**: NAX-ENG-P02-001
- **Target Milestone**: Phase 2 Exit Gate (Data Engineering, Annotation QA & Preprocessing)
- **Status**: Complete / Signed by Engineering Team

## Exit Gate Evaluation Matrix

| Exit Criteria | Verification Evidence | Status | Lead Owner |
| :--- | :--- | :--- | :--- |
| **Stable IDs & SHA-256 Checksums** | Raw images ingested with immutable SHA-256 digests; metadata logged without fabrication. | **VERIFIED** | Data Engineer |
| **Explicit Annotation & Edge Policy** | Continuous pixel coordinates preserved; half-open interval $[0, W) \times [0, H)$ edge policy enforced. | **VERIFIED** | Annotation Lead |
| **Multi-Annotator QA Agreement** | Inter-annotator CV, bipartite localization error, and adjudication protocol active. | **VERIFIED** | Neurohistologist |
| **Density Numeric Invariants** | All 5 numeric invariants (single point mass $\approx 1$, $K$-annotator invariance, superposition) unit tested. | **VERIFIED** | ML Lead |
| **Hierarchical Group Disjointness** | Pairwise split overlap audit asserts 0% cross-split leakage for `animal_id`, `nerve_id`, `image_id`. | **VERIFIED** | Statistician |
| **Locked Split Manifests** | Split CSVs exported to `splits/v1/{train,val,test}.csv` with manifest cryptographic hash. | **VERIFIED** | Data Steward |
| **Deterministic Data Build** | Pipeline re-execution produces identical patch IDs and target hashes; data card published. | **VERIFIED** | Tech Lead |

**VERDICT: APPROVED TO ADVANCE TO PHASE 03 (MODELING).**
