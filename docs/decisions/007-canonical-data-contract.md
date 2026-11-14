# NAX-ADR-007: Canonical Relational Data Model and Manifest Contract

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: Data Lead + Tech Lead
- **Phase**: Phase 02 (Data Engineering & Preprocessing)

## Context
In deep learning for computational pathology, pipelines frequently fail because metadata (biological specimen, stain, microscope, pixel spacing) is decoupled from the image tensors, or split membership is inferred from arbitrary directory structures.

## Decision
1. **Relational Data Model**:
   - `Specimen`: Biological grouping (`specimen_id`, `animal_id`, `species`, `strain`, `cohort`, `condition`, `site_id`).
   - `OpticNerve`: Specimen lateral division (`nerve_id`, `specimen_id`, `eye_side`, `section_id`, `section_protocol`).
   - `Image`: Immutable source file (`image_id`, `source_uri`, `sha256`, `width`, `height`, `bit_depth`, `channels`, `pixel_size_um`, `stain`, `microscope`, `acquisition_id`).
   - `Patch`: Deterministic spatial crop (`patch_id`, `image_id`, `x0`, `y0`, `width_px`, `height_px`, `physical_width_um`, `split_group_id`).
   - `Annotation`: Point coordinate record (`annotation_id`, `patch_id`, `annotator_id`, `label_version`, `points[]`, `edge_rule`, `review_state`).
   - `Transform`: Preprocessing versioning (`transform_id`, `code_commit`, `config_hash`, `resize`, `normalization`, `pad`, `target_parameters`).

2. **Patch Manifest as Immutable Source of Truth**:
   The manifest must contain all 22 required columns:
   `patch_id, image_id, specimen_id, animal_id, nerve_id, site_id, source_sha256, x0, y0, source_w, source_h, patch_w, patch_h, pixel_size_um, stain, microscope, condition, label_version, split_group, split, transform_version, target_version`.
   Splits are never inferred from folder names; every exported checkpoint must bind the SHA-256 hash of the split manifest.

## Consequences
- Guarantees complete biological traceability from model weights back to physical tissue section.
- Prevents silent schema drift or untracked changes in normalization.
