# NAX-ADR-012: Unified Runtime and Server-Side Job Lifecycle

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: Tech Lead + Backend Engineer
- **Phase**: Phase 05 (Inference Platform, Deployment, QA & Release)

## Context
Deploying deep learning in scientific workflows often introduces logic divergence between CLI tools, background workers, and web user interfaces. Furthermore, web frontends that calculate density integrals or apply unverified post-processing compromise research auditability.

## Decision
1. **Single Source of Truth**:
   The Python class `neuroaxonnet.inference.service.NeuroAxonInferenceService` owns all image validation, bit-depth handling, SHA-256 pre-hashes, padding, model forward execution, masking, count integration, and QC auditing.
2. **Stateless Web API Layer**:
   FastAPI routes in `neuroaxonnet.api.server` serve strictly as thin transport wrappers. The browser client is strictly prohibited from computing integrals or adjusting density thresholds.
3. **Idempotent Job Lifecycle**:
   Every submission is assigned a unique server-side `job_id`. Artifacts (`overlay.png`, `summary.csv`, `result.json`) are generated deterministically and cached for reproducible review.

## Consequences
- Guaranteed 100% numerical parity between CLI batch evaluation and web UI queries.
- Clean separation of concerns ensuring regulatory research compliance.
