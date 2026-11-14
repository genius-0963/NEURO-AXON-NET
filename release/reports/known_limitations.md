# Known Scientific Limitations and Boundary Conditions

- **Document ID**: NAX-DOC-P05-001
- **Release Version**: 0.1.0

---

## 1. Domain Generalization Boundaries
- The model was trained and baseline-validated exclusively on *Rattus norvegicus* optic nerves stained with p-phenylenediamine (PPD) and imaged at $100\times$ oil immersion magnification ($0.0625\,\mu\text{m}/\text{px}$).
- When applied to out-of-domain micrographs (mouse, non-human primate, alternative stains, lower numerical aperture), the system may underestimate counts or produce elevated variance. In such cases, the system emits `OUT_OF_DOMAIN` or `REVIEW_REQUIRED` alerts.

## 2. Pathological and Degenerated Axons
- The baseline training objective regresses Gaussian point masses centered strictly on **normal-appearing axons**. Degenerated axons, collapsed myelin sheaths, and myelin spheroids are not annotated. The count cannot be interpreted as viable axon yield in advanced neuropathy models without histological calibration.

## 3. Optical Artifacts
- Severe lens defocus blur ($\sigma_{\Delta}^2 < 60$) and background glass slide voids trigger automated **abstention** (`ABSTAIN`). The system intentionally withholds automatic counts to prevent silent failure.
