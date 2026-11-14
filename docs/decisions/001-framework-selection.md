# NAX-ADR-001: Framework Selection and Inference Architecture

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: ML Lead & Tech Lead
- **Phase**: Phase 01 (Discovery & Baseline Specification)

## Context
NeuroAxonNet reproduces and hardens the AxoNet deep learning model for estimating normal-appearing retinal ganglion cell (RGC) axon counts in optic nerve light microscopy cross-sections.
The original AxoNet codebase (Ritch et al., 2019/2020) was implemented in legacy frameworks (Keras/TensorFlow 1.x). The team evaluated framework options for modern maintenance, reproducibility, memory footprint, and research extensibility.

## Decision
1. **Primary Deep Learning Engine**: Standardize on **PyTorch (>=2.0)** for model training, gradient propagation, and GPU/MPS accelerated experimentation.
2. **Reference Parity Adapter**: Maintain an exact numerical forward engine in **pure NumPy/SciPy** within `neuroaxonnet.models.reference_engine`. This adapter runs zero-dependency inference, guarantees mathematical parity against serialized weight tensors, and operates seamlessly in lightweight environments and CI runners without gigabyte-scale deep learning binaries.
3. **Format Support**: Support model export to standard PyTorch state dictionaries (`.pt`), TorchScript, and open array weight tensors (`.npz`) for cross-platform portability.

## Consequences
- **Positive**: Clean modular code, broad community support, MPS acceleration on macOS, CUDA on Linux clusters.
- **Positive**: Parity tests and lightweight deployments can execute fast automated CI testing without heavy GPU container dependencies.
- **Trade-off**: Requires maintaining parity validation tests between PyTorch forward layers and reference NumPy convolution/activation routines.
