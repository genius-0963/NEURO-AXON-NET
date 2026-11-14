# Neurohistology Domain Label Protocol and Boundary Specification

- **Protocol ID**: NAX-PROT-001
- **Domain Expert / Lead Histologist**: Domain Scientist & Neurohistology Expert
- **Status**: Signed and Approved
- **Target Biological Model**: Light microscopy cross-sections of rat optic nerve (PPD / toluidine blue stain)

---

## 1. Morphological Annotation Criteria

In accordance with the foundational AxoNet protocol (Ritch et al., 2019/2020), point annotations mark the **center of each normal-appearing axon**.

### 1.1 In-Class: Normal-Appearing Axons (Positive Labels)
A structure is marked as a **normal-appearing axon** if and only if it satisfies all of the following:
1. **Intact Myelin Sheath**: Displays a distinct, relatively uniform, darkly stained ring or oval encircling the axoplasm.
2. **Clear Central Axoplasm**: Contains a translucent, pale interior free of dense pyknotic debris or hyper-dense inclusions.
3. **Regular Geometry**: Demonstrates roughly circular, elliptical, or moderately compressed cross-sectional morphology characteristic of healthy uncompressed nerve bundles.

### 1.2 Out-of-Class: Degenerated / Abnormal Axons (Excluded)
The following pathological structures are **strictly excluded** from the baseline target count:
1. **Myelin Ovoids / Whorls**: Dark multi-layered collapsed or vacuolated myelin figures lacking normal axoplasm.
2. **Axonal Swelling / Spheroids**: Giant pale or granular degenerate structures indicative of Wallerian degeneration.
3. **Empty / Collapsed Sheaths**: Residual myelin rings without intact axonal contents.

### 1.3 Out-of-Class: Non-Axonal Histological Features (Excluded)
1. **Glial Cell Nuclei**: Astrocytes and oligodendrocytes display large, punctate, granular chromatin structures lacking a surrounding dark myelin ring.
2. **Capillary Endothelium & Lumens**: Blood vessel lumens and elongated endothelial cell nuclei are excluded.
3. **Pial Septa & Meningeal Stroma**: Connective tissue trabeculae dividing fascicles are non-axonal.

---

## 2. Spatial Scale and Geometric Convention

- **Physical Sub-Image Window**: Exactly $12\,\mu\text{m} \times 12\,\mu\text{m}$.
- **Acquisition Sampling**: Original $187 \times 187$ pixels ($0.0642\,\mu\text{m}/\text{pixel}$).
- **Resized Convention**: Resized to $192 \times 192$ pixels ($0.0625\,\mu\text{m}/\text{pixel}$) using bilinear interpolation.
- **Rule on Scale Changes**: Arbitrary downsampling or padding without recording the physical scale ($\mu\text{m}/\text{pixel}$) is prohibited.

---

## 3. Boundary & Tiling Policy

To prevent double-counting or loss of edge axons when processing whole optic nerve cross-sections in overlapping tiles:
1. **Centroid-Based Attribution**: Each axon is modeled as a single spatial point $(x, y)$.
2. **Half-Open Interval Invariant**: An axon belongs to tile $[x_0, x_0 + W) \times [y_0, y_0 + H)$ if:
   $$x_0 \le x < x_0 + W \quad \text{and} \quad y_0 \le y < y_0 + H$$
3. **Tiled Density Integration**: When reconstructing full nerve cross-sections, density maps are combined using linear window tapering (Hann / Bartlett window), preventing boundary step artifacts.
4. **Valid Tissue Masking**: Areas outside the optic nerve sheath or holes caused by tissue tearing must have density zeroed out before integration.

---

## 4. Quality Control (QC) Histological Rubric

| Defect Class | Detection Metric | Acceptable Range | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Optical Defocus / Blur** | Variance of Laplacian ($\sigma_{\Delta}^2$) | $\ge 120.0$ | Flag `REVIEW` if $60 \le \sigma_{\Delta}^2 < 120$; `ABSTAIN` if $< 60$ |
| **Underexposure (Overstain)** | Mean intensity ($\mu_{pixel}$) | $\ge 40$ | Flag `REVIEW` if $< 40$ |
| **Overexposure (Washed Out)** | Mean intensity ($\mu_{pixel}$) | $\le 220$ | Flag `REVIEW` if $> 220$ |
| **Tissue Fold / Detachment** | High-contrast line edge fraction | $< 0.15$ | Flag `REVIEW` |
| **Empty Background Void** | Tissue foreground mask ratio | $\ge 0.35$ | Flag `ABSTAIN` if $< 0.35$ (non-tissue region) |
