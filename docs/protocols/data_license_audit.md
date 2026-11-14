# Dataset Acquisition, Provenance and License Audit

- **Document ID**: NAX-DATA-P01-001
- **Data Steward**: Data Steward & Compliance Officer
- **Date**: 2026-11-14

---

## 1. Source Dataset Specifications

- **Reference Dataset**: Ethier Laboratory Optic Nerve Axon Dataset (Georgia Institute of Technology)
- **Publication Citation**: Ritch, M. D., Hannon, B. G., Read, A. T., Feola, A. J., Cull, G. A., Reynaud, J., Morrison, J. C., Burgoyne, C. F., & Ethier, C. R. (2020). *Scientific Reports*, 10(1), 8034.
- **Access URL**: Open-access scientific repository repository (Ethier Lab AxoNet repository)
- **Biological Origin**:
  - Species: *Rattus norvegicus* (Brown Norway & Sprague-Dawley rats)
  - Tissue: Optic nerve cross-sections, retrobulbar region
  - Fixation & Stain: Glutaraldehyde/paraformaldehyde fixation, p-phenylenediamine (PPD) stained, plastic-embedded, semi-thin sectioning ($1.0\,\mu\text{m}$).
  - Microscopy: Brightfield light microscopy, $100\times$ oil immersion objective.
- **Dataset Scale**:
  - 1,514 sub-images ($12 \times 12\,\mu\text{m}$ fields).
  - Original digitization size: $187 \times 187$ pixels.
  - Standardized preprocessing size: $192 \times 192$ pixels.
  - Multi-expert manual ground-truth point annotations marking normal axon centers.

---

## 2. Licensing and Permitted Use Terms

1. **Permitted Use**: Academic research, reproducibility validation, benchmarking, and development of open-source algorithmic methods.
2. **Attribution**: Attribution to the original authors (Ritch et al., 2020) is mandatory in all resulting publications and documentation.
3. **Data Redistribution Policy**: Raw proprietary histological scans must not be redistributed directly inside source control repositories. Access is governed via immutable manifest files referencing checksum-verified remote storage endpoints.
4. **Non-Clinical Restriction**: Data was generated for preclinical animal research; no patient health information (PHI) is present.

---

## 3. Data Integrity and Checksum Verification

Every image ingested into NeuroAxonNet must be verified using SHA-256 hash digests:
- Image bytes are read without transcode modification to compute initial checksum.
- Metadata is attached with immutable fields: `image_sha256`, `animal_id`, `nerve_id`, `session_id`, `patch_coords`, and `label_points_count`.
