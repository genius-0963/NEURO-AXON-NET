# NAX-ADR-003: Biological Specimen Group Splitting and Leakage Prohibition

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: Data Lead + Statistician
- **Phase**: Phase 01 (Discovery & Baseline Specification)

## Context
A major pitfall in biological image analysis is patch-level random splitting. Because adjacent $12 \times 12\,\mu\text{m}$ sub-images from the same optic nerve cross-section share identical tissue fixation, staining chemistry, illumination gradients, and animal physiology, randomly partitioning patches produces massive data leakage and artificially inflated validation/test performance.

## Decision
1. **Split Unit**: The strict partitioning unit is the **biological specimen / individual animal ID** (e.g., `rat_01`, `rat_02`).
2. **Nerve Integrity**: All patches derived from an animal's left or right optic nerve must strictly reside in the *same* split partition.
3. **Split Ratio**: Target 60% Train, 20% Validation, 20% Held-Out Test, grouped strictly by specimen.
4. **Mandatory Automated Audit**:
   - Automated test suite (`tests/leakage/test_split_leakage.py`) must assert:
     $$\text{Animals}(\text{Train}) \cap \text{Animals}(\text{Val}) = \emptyset$$
     $$\text{Animals}(\text{Train}) \cap \text{Animals}(\text{Test}) = \emptyset$$
     $$\text{Animals}(\text{Val}) \cap \text{Animals}(\text{Test}) = \emptyset$$
   - Any commit violating this invariant will fail CI and trigger build abort.

## Consequences
- **Positive**: Test metrics reflect true biological out-of-specimen generalization, not memorization of stain artifacts or slice preparation idiosyncrasies.
- **Requirement**: Manifest schema must explicitly store `animal_id`, `nerve_id`, `slice_id`, and `patch_coordinates`.
