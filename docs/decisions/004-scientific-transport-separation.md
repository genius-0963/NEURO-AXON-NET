# NAX-ADR-004: Separation of Scientific Core from Transport and Presentation

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: Tech Lead
- **Phase**: Phase 01 (Discovery & Baseline Specification)

## Context
Bioinformatics and computational pathology tools frequently suffer from logic drift when image normalization, tile overlap handling, or count integration are re-implemented across command-line batch processors, web server backends, and JavaScript frontend clients.

## Decision
1. **Single Source of Truth**: All scientific logic (image decoding, checksum calculation, versioned normalization, tiled sliding-window inference, density integration, calibration, and QC checks) is implemented exclusively in the Python library `neuroaxonnet.inference.engine`.
2. **Thin Transport Layers**:
   - **CLI (`neuroaxonnet.cli`)**: Thin wrapper invoking `InferenceEngine.predict_patch()` or `predict_full()`.
   - **REST API (`neuroaxonnet.api.server`)**: Thin FastAPI routes translating HTTP multipart uploads into `InferenceEngine` calls.
   - **Web UI (`neuroaxonnet.api.static`)**: Renders visual heatmaps and metrics produced directly by `InferenceEngine`. Frontends are prohibited from calculating or re-scaling axon density integrals.

## Consequences
- Guarantees 100% numerical parity between CLI evaluations, automated benchmarks, and interactive user interface queries.
- Simplifies auditing, regression testing, and scientific validation.
