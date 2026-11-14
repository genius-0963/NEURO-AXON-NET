# NAX-ADR-006: Research-Use-Only Governance and Clinical Positioning

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: PI / Research Lead & Tech Lead
- **Phase**: Phase 01 (Discovery & Baseline Specification)

## Context
Optic nerve axon counting is vital for glaucoma research, optic neuropathy models, and neuroprotection assays in preclinical animal models. Under no circumstances should this research tool be construed as, or deployed as, a clinical diagnostic medical device for human disease management.

## Decision
1. **Mandatory Disclaimer Policy (NFR-03)**:
   - Every user interface view, CLI execution log, CSV/JSON report, and generated figure must explicitly state:
     > *"RESEARCH PROTOTYPE ONLY. NOT A CLINICAL DIAGNOSTIC DEVICE OR A SUBSTITUTE FOR EXPERT PATHOLOGY REVIEW."*
2. **Species & Target Boundary**:
   - Version 0.1 explicitly estimates **normal-appearing retinal ganglion cell axons in rat light microscopy optic nerve cross-sections**.
   - Prohibits automated species extrapolation (mouse, non-human primate, human) without separately validated ground truth benchmarks.

## Consequences
- Complies with ethical research standards and regulatory boundaries.
- Establishes a transparent scope for academic publication and peer review.
