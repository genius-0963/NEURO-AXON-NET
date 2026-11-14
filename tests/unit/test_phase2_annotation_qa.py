"""
Unit tests for Annotation QA, localization matching, and consensus builder (Section 6).
"""

import pytest
import numpy as np

from neuroaxonnet.data.schema import AnnotationPoint, AnnotationRecord
from neuroaxonnet.data.annotation_qa import (
    calculate_count_agreement,
    match_points_one_to_one,
    build_consensus_point_map,
)


def test_count_agreement_metrics():
    # Close agreement: counts 40 and 42
    res = calculate_count_agreement([40, 42])
    assert res["mean_count"] == 41.0
    assert res["count_discrepancy"] == 2.0
    assert not res["requires_adjudication"]

    # Severe disagreement: counts 30 and 48
    disagree = calculate_count_agreement([30, 48])
    assert disagree["count_discrepancy"] == 18.0
    assert disagree["requires_adjudication"]


def test_point_localization_bipartite_matching():
    pts_a = [
        AnnotationPoint(x=10.0, y=10.0),
        AnnotationPoint(x=50.0, y=50.0),
        AnnotationPoint(x=100.0, y=100.0),
    ]
    # Annotator B: slightly jittered (+1.0 px) and 1 extra false positive point
    pts_b = [
        AnnotationPoint(x=11.0, y=10.0),
        AnnotationPoint(x=50.0, y=51.0),
        AnnotationPoint(x=99.5, y=100.0),
        AnnotationPoint(x=180.0, y=180.0),  # FP
    ]

    metrics = match_points_one_to_one(pts_a, pts_b, tolerance_px=5.0)
    assert metrics["matches"] == 3
    assert metrics["precision"] == 1.0  # 3 matched / 3 in A
    assert metrics["recall"] == 0.75    # 3 matched / 4 in B
    assert metrics["mean_loc_error"] < 1.5
    assert not metrics["requires_adjudication"]


def test_consensus_point_map_averaging():
    shape = (192, 192)
    # Two annotators marking same centroid at (50, 50)
    rec1 = AnnotationRecord(
        annotation_id="a1", patch_id="p1", annotator_id="A1",
        points=[AnnotationPoint(x=50.0, y=50.0)]
    )
    rec2 = AnnotationRecord(
        annotation_id="a2", patch_id="p1", annotator_id="A2",
        points=[AnnotationPoint(x=50.0, y=50.0)]
    )

    pmap = build_consensus_point_map(shape, [rec1, rec2])
    # Each adds 0.5, total mass at (50, 50) is exactly 1.0
    assert np.isclose(pmap[50, 50], 1.0)
    assert np.isclose(np.sum(pmap), 1.0)
