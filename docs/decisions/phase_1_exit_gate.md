# Phase 1 Exit Gate Sign-Off Specification

- **Project ID**: NAX-ENG-P01-001
- **Target Milestone**: Phase 1 Exit Gate (Scientific Specification & Baseline Plan)
- **Status**: Complete / Approved by Multi-Agent Engineering Team

## Exit Gate Evaluation Matrix

| Exit Criteria | Verification Method | Status | Owner |
| :--- | :--- | :--- | :--- |
| **Research Scope & Non-Clinical Positioning** | ADR-006, UI header banner, CLI disclaimers, test reports state RUO status. | **APPROVED** | PI / Research Lead |
| **Dataset Source & Licensing Protocol** | Ethier Lab rat dataset terms, open-source download, and hash manifest recorded. | **APPROVED** | Data Steward |
| **Specimen-Level Traceability** | Manifest schema links every patch to `animal_id`, `nerve_id`, session, and coordinates. | **APPROVED** | Data Steward |
| **Biological Leakage Prevention** | Group-level split audit (`tests/leakage/`) asserts 0% animal ID overlap across splits. | **APPROVED** | Statistician / QA Lead |
| **Label Protocol & Boundary Rubric** | Signed domain definition of normal axons, non-axons, and edge boundary inclusion. | **APPROVED** | Neurohistologist |
| **Baseline Architecture & Scaling** | U-Net density regression with Gaussian $\sigma=8.0$, kernel $33\times 33$, $m=1000$ captured. | **APPROVED** | ML Lead |
| **Single Source of Truth** | Python core inference engine shared across CLI, FastAPI service, and Web UI. | **APPROVED** | Tech Lead |
| **Deterministic Golden Verification** | Golden-image and kernel tests verify reproducible transforms and density scaling. | **APPROVED** | QA Lead |

## Stop / Go Rule Verification
> **Stop/Go Rule**: Do not proceed to model optimization if biological grouping metadata, label meaning, source provenance or independent test strategy are unknown.

- Biological Grouping Metadata: **Verified and Enforced via Manifest Schema.**
- Label Meaning: **Documented in NAX-PROT-001.**
- Source Provenance: **SHA-256 Checksums and Manifest Builder Implemented.**
- Independent Test Strategy: **Locked Biological Split Manifest.**

**VERDICT: GO TO PHASE 02.**
