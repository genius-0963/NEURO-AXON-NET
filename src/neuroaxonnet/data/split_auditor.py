"""
Hierarchical Biological Group Partitioner and Leakage Auditor (Phase 02).

Enforces:
- Specimen-level partitioning before patch sampling
- Pairwise disjointness on: animal_id, specimen_id, nerve_id, image_id, source_sha256
- Split manifest locking and cryptographic hashing
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import numpy as np

from neuroaxonnet.data.schema import CanonicalPatchManifestRow, SplitType


class LeakageAuditError(Exception):
    """Raised when cross-split biological or image leakage is detected."""
    pass


class BiologicalSplitAuditor:
    """
    Audits and enforces zero leakage across dataset partitions.
    """
    AUDIT_KEYS = ("animal_id", "specimen_id", "nerve_id", "image_id", "source_sha256")

    def __init__(self, records: List[CanonicalPatchManifestRow]):
        self.records = records

    def audit_splits(self) -> Dict[str, Any]:
        """
        Verify that no key in AUDIT_KEYS overlaps between any pair of splits.
        """
        splits: Dict[SplitType, List[CanonicalPatchManifestRow]] = {
            SplitType.TRAIN: [r for r in self.records if r.split == SplitType.TRAIN],
            SplitType.VAL: [r for r in self.records if r.split == SplitType.VAL],
            SplitType.TEST: [r for r in self.records if r.split == SplitType.TEST],
        }

        report: Dict[str, Any] = {
            "counts": {s.value: len(rows) for s, rows in splits.items()},
            "overlaps": {},
            "passed": True,
        }

        pairs = [
            (SplitType.TRAIN, SplitType.VAL),
            (SplitType.TRAIN, SplitType.TEST),
            (SplitType.VAL, SplitType.TEST),
        ]

        for s1, s2 in pairs:
            pair_key = f"{s1.value}_vs_{s2.value}"
            report["overlaps"][pair_key] = {}
            for key in self.AUDIT_KEYS:
                vals_1 = {getattr(r, key) for r in splits[s1]}
                vals_2 = {getattr(r, key) for r in splits[s2]}
                overlap = vals_1.intersection(vals_2)
                if overlap:
                    report["overlaps"][pair_key][key] = sorted(list(overlap))
                    report["passed"] = False
                    raise LeakageAuditError(
                        f"CRITICAL LEAKAGE DETECTED between {s1.value} and {s2.value} on key '{key}': {overlap}"
                    )

        return report

    def export_split_csvs(self, output_dir: Path | str = "data/splits/v1") -> Dict[str, str]:
        """
        Exports train.csv, val.csv, test.csv and computes their SHA-256 hashes.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        # First run full audit
        self.audit_splits()

        fieldnames = list(self.records[0].to_csv_dict().keys()) if self.records else []
        file_hashes = {}

        for split_type in [SplitType.TRAIN, SplitType.VAL, SplitType.TEST]:
            split_records = [r for r in self.records if r.split == split_type]
            csv_path = out_p / f"{split_type.value}.csv"

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in split_records:
                    writer.writerow(r.to_csv_dict())

            # Compute SHA-256 of the exported CSV
            hasher = hashlib.sha256()
            with open(csv_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            file_hashes[split_type.value] = hasher.hexdigest()

        return file_hashes
