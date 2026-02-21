"""
Basic tests for largecrimsoncanine Python bindings.

Run with: pytest tests/
"""
import pytest


def test_import():
    import largecrimsoncanine as lcc
    assert lcc is not None


def test_multivector_creation():
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([1.0, 0.0, 0.0, 0.0])
    assert mv is not None


def test_scalar_extraction():
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([3.0, 0.0, 0.0, 0.0])
    assert mv.scalar() == 3.0


def test_geometric_product_basis_vectors_anticommute():
    """e1 * e2 should equal -( e2 * e1 )"""
    import largecrimsoncanine as lcc
    # e1: coeffs [0]=scalar [1]=e1 [2]=e2 [3]=e12
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e2 = lcc.Multivector([0.0, 0.0, 1.0, 0.0])
    e1e2 = e1 * e2
    e2e1 = e2 * e1
    result_1 = e1e2.to_list()
    result_2 = e2e1.to_list()
    assert result_1[3] == 1.0   # e12 coefficient
    assert result_2[3] == -1.0  # e12 coefficient, opposite sign


def test_basis_vector_squares_to_scalar():
    """e1 * e1 should equal 1 (Euclidean metric)"""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    result = (e1 * e1).to_list()
    assert result[0] == 1.0  # scalar part
    assert result[1] == 0.0  # no e1 part


def test_grade_projection():
    import largecrimsoncanine as lcc
    # Mixed multivector: scalar + e1 + e12
    mv = lcc.Multivector([1.0, 2.0, 0.0, 3.0])
    grade0 = mv.grade(0).to_list()
    grade1 = mv.grade(1).to_list()
    grade2 = mv.grade(2).to_list()
    assert grade0[0] == 1.0
    assert grade0[1] == 0.0
    assert grade1[1] == 2.0
    assert grade1[0] == 0.0
    assert grade2[3] == 3.0
    assert grade2[0] == 0.0


def test_invalid_coeffs_length():
    import largecrimsoncanine as lcc
    with pytest.raises(Exception):
        lcc.Multivector([1.0, 2.0, 3.0])  # length 3 is not a power of 2

