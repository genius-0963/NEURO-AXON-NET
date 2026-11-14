"""
Unit tests for the 5 Density Numeric Invariants (Section 7.1).
"""

from neuroaxonnet.data.density_builder import verify_numeric_invariants


def test_all_five_density_numeric_invariants():
    res = verify_numeric_invariants()

    # 1. Single interior point produces non-negative smooth target with integral ~1.0
    assert res["single_point_non_negative"] is True
    assert res["invariant_1_pass"] is True
    assert 0.90 <= res["single_point_integral"] <= 1.05

    # 2. K identical annotator marks averaged by K do not multiply count
    assert res["invariant_2_k_annotator_invariance"] is True

    # 3. Linearity / Superposition: Density(A + B) == Density(A) + Density(B)
    assert res["invariant_3_superposition"] is True

    # 4. Border point edge-inclusion behavior documented
    assert res["invariant_4_border_documented"] is True

    # 5. Dtype is float32, shape is [192, 192], finite checks pass
    assert res["invariant_5_pass"] is True
    assert res["dtype"] == "float32"
    assert res["shape"] == [192, 192]
    assert res["finite_check"] is True
