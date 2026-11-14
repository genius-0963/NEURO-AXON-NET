#!/usr/bin/env python3
"""
CLI script to generate and print the Phase 02 Data Quality Dashboard.
"""

from neuroaxonnet.qc.dashboard import Phase2DataQualityDashboard


def main():
    dashboard = Phase2DataQualityDashboard()
    report = dashboard.generate_full_report()
    out_file = dashboard.save_report()

    print("\n" + "=" * 65)
    print("      NEUROAXONNET PHASE 02: DATA QUALITY & INTEGRITY GATES     ")
    print("=" * 65)
    print(f"Overall Gate Decision:    [{report['gate_decision']}]")
    print(f"Saved JSON Report:        {out_file}")

    views = report["views"]

    # 1. Image Quality
    iq = views["image_quality"]
    print("\n1. IMAGE QUALITY AUDIT:")
    print(f"   - Status Distribution: {iq['status_breakdown']}")
    print(f"   - Flagged Samples:     {iq['flagged_patches_count']}")
    for s in iq.get("flagged_samples", []):
        print(f"     * {s['patch_id']}: [{s['status']}] {s['flags']} (Laplacian: {s['metrics']['laplacian_focus']})")

    # 2. Biological Distribution
    bd = views["biological_distribution"]
    print("\n2. BIOLOGICAL SPECIMEN DISTRIBUTION:")
    print(f"   - Total Patches:       {bd['total_patches']}")
    print(f"   - Zero Leakage Status: [{'PASS' if bd['zero_leakage_verified'] else 'FAIL'}]")
    for split_name, animals in bd["animals_per_split"].items():
        print(f"   - {split_name.upper()} Animals:        {animals}")

    # 3. Label Distribution
    ld = views["label_distribution"]
    print("\n3. LABEL DISTRIBUTION & ANNOTATION STATS:")
    print(f"   - Mean Axon Count:     {ld['mean_count']} axons/patch")
    print(f"   - Median Count:        {ld['median_count']} axons/patch")
    print(f"   - Count Range:         [{ld['min_count']}, {ld['max_count']}]")
    print(f"   - Zero Count Ratio:    {ld['zero_count_fraction'] * 100:.1f}%")
    print(f"   - Percentiles:         p25={ld['count_percentiles']['p25']}, p50={ld['count_percentiles']['p50']}, p75={ld['count_percentiles']['p75']}")

    # 4. Acquisition Domain
    ad = views["acquisition_domain"]
    print("\n4. ACQUISITION DOMAIN & METADATA:")
    print(f"   - Stains:              {ad['stains']}")
    print(f"   - Microscopes:         {ad['microscopes']}")
    print(f"   - Pixel Spacing:       {ad['pixel_spacing_um']} um/pixel")
    print(f"   - Metadata Complete:   {ad['metadata_completeness'] * 100:.0f}%")

    # 5. Target Numeric Invariants
    ti = views["target_integrity"]
    print("\n5. TARGET INTEGRITY & DENSITY INVARIANTS:")
    print(f"   - All Invariants Pass: [{'PASS' if ti['numeric_invariants_pass'] else 'FAIL'}]")
    print(f"   - Single Point Mass:   {ti['details']['single_point_integral']} (expected ~1.0)")
    print(f"   - K-Annotator Invar:   {ti['details']['invariant_2_k_annotator_invariance']}")
    print(f"   - Superposition:       {ti['details']['invariant_3_superposition']}")
    print(f"   - Target Dtype:        {ti['details']['dtype']} {ti['details']['shape']}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
