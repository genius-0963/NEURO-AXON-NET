# NAX-ADR-008: Preprocessing Normalization and Context Reflection Padding Contract

- **Status**: Approved
- **Date**: 2026-11-14
- **Decision Owner**: ML Lead + Neurohistology Expert
- **Phase**: Phase 02 (Data Engineering & Preprocessing)

## Context
AxoNet relies on pixelwise density regression where convolutions near borders lose context without padding. However, naive zero-padding introduces severe dark-edge gradients that falsely elevate density near borders, while unmasked reflection padding risks counting mirrored axons twice.

## Decision
1. **Bilinear Resize**: Resize from raw $187 \times 187$ to $192 \times 192$ using bilinear interpolation with recorded align-corners convention.
2. **Robust Standardization**:
   $$\mu = \text{mean}(I), \quad \sigma = \text{std}(I)$$
   $$I_{\text{norm}} = \text{clip}\left(\frac{I - \mu}{\max(2.0 \cdot \sigma, \epsilon)}, -1.0, 1.0\right)$$
   Using $2\sigma$ scaling bounds dynamic range into $[-1, 1]$ while mitigating extreme outlier pixel noise.
3. **Reflection Padding & Valid Core**:
   - Mirror pad by 16 pixels on each boundary to expand tensor from $192 \times 192$ to $224 \times 224$.
   - Maintain an explicit boolean mask:
     $$\text{valid\_mask}[16:208, 16:208] = \text{True}, \quad \text{else False}$$
   - Mirrored pixels provide spatial receptive field context for convolutions, but are **strictly excluded** from loss computation and count integration:
     $$\text{Count} = \frac{1}{m} \sum_{(x, y) \in \text{valid\_mask}} D(x, y)$$

## Consequences
- Eliminates edge discontinuity artifacts.
- Guarantees count integral corresponds exactly to the $12 \times 12\,\mu\text{m}$ observed field.
