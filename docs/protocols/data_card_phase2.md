# NeuroAxonNet Dataset Card — Phase 02

- **Document ID**: NAX-CARD-P02-001
- **Version**: 1.0.0
- **Author**: Data Steward & Neurohistology Team
- **Date**: 2026-11-14

---

## 1. Dataset Overview and Provenance
- **Dataset Title**: Brown Norway Rat Optic Nerve Axon Quantification Dataset.
- **Reference Source**: Ethier Laboratory, Georgia Institute of Technology (Ritch et al., 2020, *Scientific Reports*).
- **Biological Origin**:
  - Species: *Rattus norvegicus*
  - Strain: Brown Norway
  - Biological Specimens: 14 rats (27 distinct optic nerves; 13 bilateral pairs, 1 unilateral).
  - Total Sub-Images: 1,514 patches ($12 \times 12\,\mu\text{m}$).
  - Sectioning: Semi-thin ($1.0\,\mu\text{m}$) plastic-embedded cross-sections of the retrobulbar optic nerve.
  - Staining: 1% p-phenylenediamine (PPD), differentially staining intact myelin sheaths dark brown/black.
  - Magnification: $100\times$ oil immersion brightfield light microscopy.

---

## 2. Label Semantics and Known Constraints
- **Target Feature**: Normal-appearing axon centroids.
- **Annotators**: Trained expert neuroscientists; designates one approximate $(x, y)$ coordinate per normal axon.
- **Absence Semantics**: The absence of a point annotation does **not** signify the absence of axonal structure; it denotes that the structure was abnormal (degenerated, collapsed sheath, myelin spheroid), ambiguous, non-axonal (glia, blood vessel), or excluded by edge protocol.
- **Model Representation**: Continuous 2D count density regressed at scale factor $m = 1000.0$.

---

## 3. Preprocessing and Target Parameters
- **Raw Patch Resolution**: $187 \times 187$ pixels ($0.0642\,\mu\text{m}/\text{pixel}$).
- **Resized Core**: $192 \times 192$ pixels ($0.0625\,\mu\text{m}/\text{pixel}$) via bilinear interpolation.
- **Network Input Tensor**: $224 \times 224$ pixels via 16-pixel reflection padding.
- **Valid Core Mask**: Boolean array of shape $(224, 224)$, with `[16:208, 16:208]` active.
- **Target Density Parameters**: Isotropic Gaussian with $\sigma = 8.0$, truncation radius $r = 16$ px (kernel size 33), density multiplier $m = 1000.0$.

---

## 4. Leakage Prevention and Split Governance
- **Partitioning Principle**: Hierarchical biological grouping by `animal_id`.
- **Target Ratios**: Approximately 60% Train, 20% Validation, 20% Test (specimen-grouped).
- **Leakage Auditing**: Automated continuous integration asserting zero pairwise animal, nerve, or source image overlap across splits.
- **Locking**: Split manifests locked via SHA-256 hash.
