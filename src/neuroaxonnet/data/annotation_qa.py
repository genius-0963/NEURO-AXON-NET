"""
Annotation Quality Assurance and Multi-Annotator Consensus Module.

Implements:
- Multi-annotator count agreement (absolute error, CV)
- Bipartite point localization matching within distance tolerance
- Precision, recall, F1, and mean spatial localization error
- K-annotator indicator map builder with 1.0 / K averaging
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from neuroaxonnet.data.schema import AnnotationPoint, AnnotationRecord


def calculate_count_agreement(annotator_counts: List[int | float]) -> Dict[str, float]:
    """
    Compute inter-annotator count consistency metrics.
    """
    if len(annotator_counts) < 2:
        return {"count_discrepancy": 0.0, "cv": 0.0, "mean_count": float(annotator_counts[0]) if annotator_counts else 0.0}

    counts = np.array(annotator_counts, dtype=np.float64)
    mean_val = float(np.mean(counts))
    std_val = float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0
    cv = (std_val / mean_val) if mean_val > 0 else 0.0
    max_diff = float(np.max(counts) - np.min(counts))

    return {
        "mean_count": round(mean_val, 2),
        "count_discrepancy": round(max_diff, 2),
        "cv": round(cv, 4),
        "requires_adjudication": max_diff > 5.0 or cv > 0.15,
    }


def match_points_one_to_one(
    points_a: List[AnnotationPoint],
    points_b: List[AnnotationPoint],
    tolerance_px: float = 5.0,
) -> Dict[str, Any]:
    """
    Match points between two annotators using greedy nearest-neighbor one-to-one assignment
    within an expert distance tolerance (default 5.0 pixels ~ 0.31 um).

    Returns:
        precision, recall, f1, mean_localization_error, matched_pairs, unmatched_a, unmatched_b
    """
    if not points_a and not points_b:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mean_loc_error": 0.0, "matches": 0}
    if not points_a or not points_b:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_loc_error": 0.0, "matches": 0}

    coords_a = np.array([[p.x, p.y] for p in points_a], dtype=np.float64)
    coords_b = np.array([[p.x, p.y] for p in points_b], dtype=np.float64)

    # Compute Euclidean distance matrix (N_a, N_b)
    diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    # Greedy bipartite matching by ascending distance
    matched_a = set()
    matched_b = set()
    matched_distances = []

    # Flatten and sort candidate pairs
    flat_indices = np.argsort(dist_matrix, axis=None)
    for idx in flat_indices:
        i, j = np.unravel_index(idx, dist_matrix.shape)
        d = dist_matrix[i, j]
        if d > tolerance_px:
            break
        if i not in matched_a and j not in matched_b:
            matched_a.add(i)
            matched_b.add(j)
            matched_distances.append(float(d))

    n_matches = len(matched_distances)
    precision = n_matches / len(points_a) if points_a else 0.0
    recall = n_matches / len(points_b) if points_b else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_err = float(np.mean(matched_distances)) if matched_distances else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_loc_error": round(mean_err, 2),
        "matches": n_matches,
        "unmatched_annotator_a": len(points_a) - n_matches,
        "unmatched_annotator_b": len(points_b) - n_matches,
        "requires_adjudication": f1 < 0.85,
    }


def build_consensus_point_map(
    shape: Tuple[int, int],
    annotator_records: List[AnnotationRecord],
) -> np.ndarray:
    """
    Build discrete point indicator map averaged across K annotators:
        point_map[round(y), round(x)] += 1.0 / K
    """
    h, w = shape
    point_map = np.zeros((h, w), dtype=np.float64)
    k = len(annotator_records)
    if k == 0:
        return point_map

    weight = 1.0 / float(k)
    for rec in annotator_records:
        for pt in rec.points:
            ix = int(round(pt.x))
            iy = int(round(pt.y))
            if 0 <= ix < w and 0 <= iy < h:
                point_map[iy, ix] += weight

    return point_map.astype(np.float32)
