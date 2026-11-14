"""
Biological Leakage Audit Test for Phase 02.

Asserts 0% pairwise overlap across all split CSVs for:
- animal_id
- specimen_id
- nerve_id
- image_id
- source_sha256
"""

import pytest
import csv
from pathlib import Path

from neuroaxonnet.data.schema import CanonicalPatchManifestRow, SplitType
from neuroaxonnet.data.split_auditor import BiologicalSplitAuditor, LeakageAuditError


def test_split_csvs_zero_leakage():
    splits_dir = Path("data/splits/v1")
    assert splits_dir.is_dir(), "Splits directory data/splits/v1 does not exist."

    rows = []
    for s in [SplitType.TRAIN, SplitType.VAL, SplitType.TEST]:
        csv_file = splits_dir / f"{s.value}.csv"
        assert csv_file.is_file(), f"Split CSV {csv_file} missing."
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for d in reader:
                rows.append(CanonicalPatchManifestRow(**d))

    assert len(rows) > 0, "No rows found across split CSVs."
    auditor = BiologicalSplitAuditor(rows)
    report = auditor.audit_splits()
    assert report["passed"] is True


def test_leakage_detector_catches_overlap():
    row_train = CanonicalPatchManifestRow(
        patch_id="p1", image_id="img1", specimen_id="spec1", animal_id="rat_01",
        nerve_id="n1", site_id="s1", source_sha256="sha1", x0=0, y0=0, source_w=187, source_h=187,
        patch_w=192, patch_h=192, pixel_size_um=0.0625, stain="PPD", microscope="100x",
        condition="Control", label_version="v1", split_group="rat_01", split=SplitType.TRAIN,
        transform_version="v1", target_version="v1", ground_truth_count=40.0
    )
    # Deliberate biological leakage: rat_01 also in test split
    row_test = row_train.model_copy(update={"patch_id": "p2", "split": SplitType.TEST})

    auditor = BiologicalSplitAuditor([row_train, row_test])
    with pytest.raises(LeakageAuditError, match="CRITICAL LEAKAGE DETECTED"):
        auditor.audit_splits()
