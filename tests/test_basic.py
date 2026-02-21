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


def test_addition():
    """Test multivector addition."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    b = lcc.Multivector([0.5, 1.5, 2.5, 3.5])
    c = a + b
    result = c.to_list()
    assert result == [1.5, 3.5, 5.5, 7.5]


def test_subtraction():
    """Test multivector subtraction."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    b = lcc.Multivector([0.5, 1.0, 1.5, 2.0])
    c = a - b
    result = c.to_list()
    assert result == [0.5, 1.0, 1.5, 2.0]


def test_negation():
    """Test multivector negation."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, -2.0, 3.0, -4.0])
    b = -a
    result = b.to_list()
    assert result == [-1.0, 2.0, -3.0, 4.0]


def test_scale():
    """Test scalar multiplication."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    b = a.scale(2.0)
    result = b.to_list()
    assert result == [2.0, 4.0, 6.0, 8.0]


def test_scale_negative():
    """Test scalar multiplication with negative scalar."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    b = a.scale(-0.5)
    result = b.to_list()
    assert result == [-0.5, -1.0, -1.5, -2.0]


def test_equality():
    """Test multivector equality."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    b = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    c = lcc.Multivector([1.0, 2.0, 3.0, 5.0])
    assert a == b
    assert not (a == c)


def test_approx_eq():
    """Test approximate equality."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    b = lcc.Multivector([1.0 + 1e-11, 2.0, 3.0, 4.0])
    c = lcc.Multivector([1.0 + 1e-9, 2.0, 3.0, 4.0])
    assert a.approx_eq(b)  # within default tolerance
    assert not a.approx_eq(c)  # outside default tolerance
    assert a.approx_eq(c, tol=1e-8)  # within custom tolerance


def test_dimension_mismatch_add():
    """Adding multivectors of different dimensions should raise."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])  # 2D
    b = lcc.Multivector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])  # 3D
    with pytest.raises(Exception):
        a + b


def test_reverse_scalar():
    """Reverse of a scalar is itself."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector([5.0, 0.0, 0.0, 0.0])
    assert s.reverse().to_list() == [5.0, 0.0, 0.0, 0.0]


def test_reverse_vector():
    """Reverse of a vector is itself (grade 1 unchanged)."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector([0.0, 3.0, 4.0, 0.0])
    assert v.reverse().to_list() == [0.0, 3.0, 4.0, 0.0]


def test_reverse_bivector():
    """Reverse of a bivector flips sign (grade 2)."""
    import largecrimsoncanine as lcc
    b = lcc.Multivector([0.0, 0.0, 0.0, 7.0])
    assert b.reverse().to_list() == [0.0, 0.0, 0.0, -7.0]


def test_reverse_mixed():
    """Reverse of mixed multivector: scalar+vector unchanged, bivector flipped."""
    import largecrimsoncanine as lcc
    # scalar=1, e1=2, e2=3, e12=4
    mv = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    rev = mv.reverse().to_list()
    assert rev[0] == 1.0   # scalar unchanged
    assert rev[1] == 2.0   # e1 unchanged
    assert rev[2] == 3.0   # e2 unchanged
    assert rev[3] == -4.0  # e12 flipped


def test_tilde_alias():
    """tilde() should be an alias for reverse()."""
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    assert mv.reverse().to_list() == mv.tilde().to_list()


def test_left_contraction_vectors():
    """Left contraction of two vectors should give their scalar (dot) product."""
    import largecrimsoncanine as lcc
    # e1 ⌋ e1 = 1 (Euclidean metric)
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    result = e1.left_contraction(e1).to_list()
    assert result[0] == 1.0  # scalar
    assert result[1] == 0.0
    assert result[2] == 0.0
    assert result[3] == 0.0


def test_left_contraction_orthogonal_vectors():
    """Left contraction of orthogonal vectors is zero."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e2 = lcc.Multivector([0.0, 0.0, 1.0, 0.0])
    result = e1.left_contraction(e2).to_list()
    assert result == [0.0, 0.0, 0.0, 0.0]


def test_left_contraction_vector_bivector():
    """Left contraction of vector with bivector gives vector."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e12 = lcc.Multivector([0.0, 0.0, 0.0, 1.0])
    # e1 ⌋ e12 = e1 ⌋ (e1^e2) = (e1·e1)e2 - (e1·e2)e1 = e2
    result = e1.left_contraction(e12).to_list()
    assert result[0] == 0.0  # no scalar
    assert result[1] == 0.0  # no e1
    assert result[2] == 1.0  # e2
    assert result[3] == 0.0  # no e12


def test_inner_alias():
    """inner() should be an alias for left_contraction()."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e12 = lcc.Multivector([0.0, 0.0, 0.0, 1.0])
    assert e1.left_contraction(e12).to_list() == e1.inner(e12).to_list()


def test_norm_squared_vector():
    """Norm squared of a vector is sum of squared components."""
    import largecrimsoncanine as lcc
    # v = 3*e1 + 4*e2, |v|² = 9 + 16 = 25
    v = lcc.Multivector([0.0, 3.0, 4.0, 0.0])
    assert v.norm_squared() == 25.0


def test_norm_vector():
    """Norm of a vector is sqrt of sum of squared components."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector([0.0, 3.0, 4.0, 0.0])
    assert v.norm() == 5.0


def test_norm_scalar():
    """Norm of a scalar is its absolute value."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector([5.0, 0.0, 0.0, 0.0])
    assert s.norm() == 5.0
    s_neg = lcc.Multivector([-5.0, 0.0, 0.0, 0.0])
    assert s_neg.norm() == 5.0


def test_normalize_vector():
    """Normalizing a vector gives a unit vector."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector([0.0, 3.0, 4.0, 0.0])
    v_norm = v.normalized()
    assert abs(v_norm.norm() - 1.0) < 1e-10
    # Direction should be preserved: (3/5, 4/5)
    coeffs = v_norm.to_list()
    assert abs(coeffs[1] - 0.6) < 1e-10
    assert abs(coeffs[2] - 0.8) < 1e-10


def test_normalize_zero_raises():
    """Normalizing a zero multivector should raise."""
    import largecrimsoncanine as lcc
    zero = lcc.Multivector([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(Exception):
        zero.normalized()


# =============================================================================
# OUTER PRODUCT (WEDGE) TESTS
# =============================================================================

def test_outer_product_orthogonal_vectors():
    """e1 ^ e2 should equal e12."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e2 = lcc.Multivector([0.0, 0.0, 1.0, 0.0])
    result = e1.outer_product(e2).to_list()
    assert result[3] == 1.0  # e12 coefficient
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == 0.0


def test_outer_product_anticommutes():
    """e2 ^ e1 should equal -e12."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e2 = lcc.Multivector([0.0, 0.0, 1.0, 0.0])
    result = e2.outer_product(e1).to_list()
    assert result[3] == -1.0  # e12 coefficient, opposite sign


def test_outer_product_same_vector_zero():
    """e1 ^ e1 should equal 0 (key property of wedge product)."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    result = e1.outer_product(e1).to_list()
    assert result == [0.0, 0.0, 0.0, 0.0]


def test_wedge_operator():
    """Test ^ operator for outer product."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e2 = lcc.Multivector([0.0, 0.0, 1.0, 0.0])
    result = (e1 ^ e2).to_list()
    assert result[3] == 1.0


def test_wedge_alias():
    """wedge() should be an alias for outer_product()."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e2 = lcc.Multivector([0.0, 0.0, 1.0, 0.0])
    assert e1.outer_product(e2).to_list() == e1.wedge(e2).to_list()


# =============================================================================
# LEFT CONTRACTION SUBSET REQUIREMENT
# =============================================================================

def test_left_contraction_non_subset_zero():
    """e2 ⌋ e1 should be 0 (e2 not subset of e1)."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e2 = lcc.Multivector([0.0, 0.0, 1.0, 0.0])
    result = e2.left_contraction(e1).to_list()
    assert result == [0.0, 0.0, 0.0, 0.0]


def test_left_contraction_operator():
    """Test | operator for left contraction."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e12 = lcc.Multivector([0.0, 0.0, 0.0, 1.0])
    result = (e1 | e12).to_list()
    assert result[2] == 1.0  # e2


def test_lc_alias():
    """lc() should be an alias for left_contraction()."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e12 = lcc.Multivector([0.0, 0.0, 0.0, 1.0])
    assert e1.left_contraction(e12).to_list() == e1.lc(e12).to_list()


# =============================================================================
# SCALAR MULTIPLICATION AND DIVISION
# =============================================================================

def test_scalar_multiply_right():
    """mv * 2.0 should scale all coefficients."""
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    result = (mv * 2.0).to_list()
    assert result == [2.0, 4.0, 6.0, 8.0]


def test_scalar_multiply_left():
    """2.0 * mv should scale all coefficients."""
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    result = (2.0 * mv).to_list()
    assert result == [2.0, 4.0, 6.0, 8.0]


def test_scalar_division():
    """mv / 2.0 should halve all coefficients."""
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([2.0, 4.0, 6.0, 8.0])
    result = (mv / 2.0).to_list()
    assert result == [1.0, 2.0, 3.0, 4.0]


def test_division_by_zero_raises():
    """Division by zero should raise."""
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ZeroDivisionError):
        mv / 0.0


# =============================================================================
# METHOD ALIASES
# =============================================================================

def test_gp_alias():
    """gp() should be an alias for geometric_product()."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0])
    e2 = lcc.Multivector([0.0, 0.0, 1.0, 0.0])
    assert e1.geometric_product(e2).to_list() == e1.gp(e2).to_list()


# =============================================================================
# DUNDER METHODS
# =============================================================================

def test_len():
    """len(mv) should return coefficient count."""
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    assert len(mv) == 4
    mv8 = lcc.Multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert len(mv8) == 8


def test_getitem():
    """mv[i] should return coefficient at index i."""
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    assert mv[0] == 1.0
    assert mv[1] == 2.0
    assert mv[2] == 3.0
    assert mv[3] == 4.0


def test_getitem_negative_index():
    """mv[-1] should return last coefficient."""
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    assert mv[-1] == 4.0
    assert mv[-2] == 3.0


def test_getitem_out_of_range():
    """Out of range index should raise IndexError."""
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(IndexError):
        mv[10]


def test_str():
    """str(mv) should give human-readable representation."""
    import largecrimsoncanine as lcc
    # Pure scalar
    s = lcc.Multivector([5.0, 0.0, 0.0, 0.0])
    assert str(s) == "5"
    # Zero
    zero = lcc.Multivector([0.0, 0.0, 0.0, 0.0])
    assert str(zero) == "0"
    # Vector
    v = lcc.Multivector([0.0, 3.0, 0.0, 0.0])
    assert "e1" in str(v)


def test_repr_shows_dims():
    """repr(mv) should show dimensions."""
    import largecrimsoncanine as lcc
    mv = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    r = repr(mv)
    assert "dims=2" in r


# =============================================================================
# DIMENSION MISMATCH TESTS
# =============================================================================

def test_dimension_mismatch_geometric_product():
    """Geometric product of different dimensions should raise."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])  # 2D
    b = lcc.Multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 3D
    with pytest.raises(Exception):
        a * b


def test_dimension_mismatch_outer_product():
    """Outer product of different dimensions should raise."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    b = lcc.Multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(Exception):
        a ^ b


def test_dimension_mismatch_left_contraction():
    """Left contraction of different dimensions should raise."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    b = lcc.Multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(Exception):
        a | b


def test_dimension_mismatch_subtraction():
    """Subtraction of different dimensions should raise."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    b = lcc.Multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(Exception):
        a - b


# =============================================================================
# ZERO MULTIVECTOR TESTS
# =============================================================================

def test_zero_addition():
    """0 + A = A."""
    import largecrimsoncanine as lcc
    zero = lcc.Multivector([0.0, 0.0, 0.0, 0.0])
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    result = (zero + a).to_list()
    assert result == [1.0, 2.0, 3.0, 4.0]


def test_zero_multiplication():
    """0 * A = 0."""
    import largecrimsoncanine as lcc
    zero = lcc.Multivector([0.0, 0.0, 0.0, 0.0])
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    result = (zero * a).to_list()
    assert result == [0.0, 0.0, 0.0, 0.0]


def test_zero_outer_product():
    """0 ^ A = 0."""
    import largecrimsoncanine as lcc
    zero = lcc.Multivector([0.0, 0.0, 0.0, 0.0])
    a = lcc.Multivector([1.0, 2.0, 3.0, 4.0])
    result = (zero ^ a).to_list()
    assert result == [0.0, 0.0, 0.0, 0.0]


def test_zero_norm():
    """|0| = 0."""
    import largecrimsoncanine as lcc
    zero = lcc.Multivector([0.0, 0.0, 0.0, 0.0])
    assert zero.norm() == 0.0
    assert zero.norm_squared() == 0.0


# =============================================================================
# 3D ALGEBRA TESTS
# =============================================================================

def test_3d_basis_vectors():
    """Test 3D algebra (Cl(3) with 8 coefficients)."""
    import largecrimsoncanine as lcc
    # e1, e2, e3 in 3D
    e1 = lcc.Multivector([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e2 = lcc.Multivector([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e3 = lcc.Multivector([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    # e1 * e1 = 1
    assert (e1 * e1).to_list()[0] == 1.0

    # e1 ^ e2 = e12 (index 3)
    e12 = (e1 ^ e2).to_list()
    assert e12[3] == 1.0

    # e1 ^ e2 ^ e3 = e123 (index 7)
    e123 = (e1 ^ e2 ^ e3).to_list()
    assert e123[7] == 1.0


def test_3d_trivector_reverse():
    """Reverse of e123 flips sign (grade 3)."""
    import largecrimsoncanine as lcc
    e123 = lcc.Multivector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    rev = e123.reverse().to_list()
    assert rev[7] == -1.0  # grade 3 flips


# =============================================================================
# GEOMETRIC PRODUCT PROPERTIES
# =============================================================================

def test_geometric_product_associativity():
    """(A * B) * C = A * (B * C)."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 0.0, 0.0])
    b = lcc.Multivector([0.0, 1.0, 1.0, 0.0])
    c = lcc.Multivector([0.5, 0.0, 0.0, 1.0])

    left = ((a * b) * c).to_list()
    right = (a * (b * c)).to_list()

    for i in range(4):
        assert abs(left[i] - right[i]) < 1e-10


def test_geometric_product_distributivity():
    """A * (B + C) = A*B + A*C."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector([1.0, 2.0, 0.0, 0.0])
    b = lcc.Multivector([0.0, 1.0, 1.0, 0.0])
    c = lcc.Multivector([0.5, 0.0, 0.0, 1.0])

    left = (a * (b + c)).to_list()
    right = ((a * b) + (a * c)).to_list()

    for i in range(4):
        assert abs(left[i] - right[i]) < 1e-10


def test_norm_bivector():
    """Norm of a bivector."""
    import largecrimsoncanine as lcc
    # Pure bivector e12
    e12 = lcc.Multivector([0.0, 0.0, 0.0, 1.0])
    # |e12|² = e12 * ~e12 = e12 * (-e12) = -e12² = -(-1) = 1
    # Actually in Euclidean: e12² = e1*e2*e1*e2 = -e1*e1*e2*e2 = -1
    # So e12 * ~e12 = e12 * (-e12) = -(e12)² = -(-1) = 1
    assert e12.norm_squared() == 1.0
    assert e12.norm() == 1.0


# =============================================================================
# CONSTRUCTOR TESTS
# =============================================================================

def test_zero_constructor():
    """Multivector.zero(dims) creates a zero multivector."""
    import largecrimsoncanine as lcc
    z = lcc.Multivector.zero(3)
    assert len(z) == 8  # 2^3 = 8 coefficients
    assert all(c == 0.0 for c in z.to_list())


def test_zero_constructor_dimension_error():
    """Multivector.zero(0) should raise."""
    import largecrimsoncanine as lcc
    with pytest.raises(ValueError):
        lcc.Multivector.zero(0)


def test_from_scalar_constructor():
    """Multivector.from_scalar creates a pure scalar."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(5.0, dims=3)
    assert s.scalar() == 5.0
    assert s[1] == 0.0  # no e1
    assert s[2] == 0.0  # no e2
    assert len(s) == 8


def test_from_vector_constructor():
    """Multivector.from_vector creates a vector from coordinates."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    # Should be 1*e1 + 2*e2 + 3*e3 in Cl(3)
    assert len(v) == 8  # 2^3
    assert v[0] == 0.0  # no scalar
    assert v[1] == 1.0  # e1 coefficient
    assert v[2] == 2.0  # e2 coefficient
    assert v[4] == 3.0  # e3 coefficient (index 4 = 0b100)


def test_from_vector_2d():
    """from_vector with 2 coordinates creates 2D vector."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([3.0, 4.0])
    assert len(v) == 4  # 2^2
    assert v[1] == 3.0  # e1
    assert v[2] == 4.0  # e2
    assert v.norm() == 5.0  # 3-4-5 triangle


def test_from_vector_empty_error():
    """from_vector with empty list should raise."""
    import largecrimsoncanine as lcc
    with pytest.raises(ValueError):
        lcc.Multivector.from_vector([])


def test_basis_constructor():
    """Multivector.basis creates a single basis vector."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.basis(1, dims=3)
    e2 = lcc.Multivector.basis(2, dims=3)
    e3 = lcc.Multivector.basis(3, dims=3)

    assert e1[1] == 1.0 and e1[2] == 0.0 and e1[4] == 0.0
    assert e2[1] == 0.0 and e2[2] == 1.0 and e2[4] == 0.0
    assert e3[1] == 0.0 and e3[2] == 0.0 and e3[4] == 1.0


def test_basis_index_error():
    """Multivector.basis with invalid index should raise."""
    import largecrimsoncanine as lcc
    with pytest.raises(ValueError):
        lcc.Multivector.basis(0, dims=3)  # 0 is invalid
    with pytest.raises(ValueError):
        lcc.Multivector.basis(4, dims=3)  # 4 > dims


def test_basis_squares_to_one():
    """Basis vectors should square to 1 (Euclidean)."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.basis(1, dims=3)
    e2 = lcc.Multivector.basis(2, dims=3)

    assert (e1 * e1).scalar() == 1.0
    assert (e2 * e2).scalar() == 1.0


def test_basis_anticommute():
    """Different basis vectors should anticommute."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.basis(1, dims=3)
    e2 = lcc.Multivector.basis(2, dims=3)

    e1e2 = e1 * e2
    e2e1 = e2 * e1

    # e1*e2 + e2*e1 should be zero
    diff = e1e2 + e2e1
    assert all(abs(c) < 1e-10 for c in diff.to_list())


def test_pseudoscalar_constructor():
    """Multivector.pseudoscalar creates the unit pseudoscalar."""
    import largecrimsoncanine as lcc
    I2 = lcc.Multivector.pseudoscalar(2)
    I3 = lcc.Multivector.pseudoscalar(3)

    # In 2D, pseudoscalar is e12 at index 3
    assert I2[3] == 1.0
    assert I2[0] == 0.0 and I2[1] == 0.0 and I2[2] == 0.0

    # In 3D, pseudoscalar is e123 at index 7
    assert I3[7] == 1.0


def test_pseudoscalar_squares():
    """Pseudoscalar squared depends on dimension."""
    import largecrimsoncanine as lcc
    # In 2D: (e12)² = e1*e2*e1*e2 = -e1*e1*e2*e2 = -1
    I2 = lcc.Multivector.pseudoscalar(2)
    I2_sq = I2 * I2
    assert I2_sq.scalar() == -1.0

    # In 3D: (e123)² = e1*e2*e3*e1*e2*e3 = -1
    I3 = lcc.Multivector.pseudoscalar(3)
    I3_sq = I3 * I3
    assert I3_sq.scalar() == -1.0


def test_from_bivector_constructor():
    """Multivector.from_bivector creates a bivector from components."""
    import largecrimsoncanine as lcc
    # In 3D, bivector components are [e12, e13, e23]
    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)

    # e12 is at index 3 (0b011)
    assert B[3] == 1.0
    # e13 is at index 5 (0b101)
    assert B[5] == 0.0
    # e23 is at index 6 (0b110)
    assert B[6] == 0.0


def test_from_bivector_all_components():
    """from_bivector with all non-zero components."""
    import largecrimsoncanine as lcc
    B = lcc.Multivector.from_bivector([1.0, 2.0, 3.0], dims=3)

    assert B[3] == 1.0  # e12
    assert B[5] == 2.0  # e13
    assert B[6] == 3.0  # e23


def test_from_bivector_2d():
    """from_bivector in 2D has only one component."""
    import largecrimsoncanine as lcc
    B = lcc.Multivector.from_bivector([5.0], dims=2)

    assert B[3] == 5.0  # e12
    assert len(B) == 4


def test_from_bivector_wrong_components_error():
    """from_bivector with wrong number of components should raise."""
    import largecrimsoncanine as lcc
    with pytest.raises(ValueError):
        lcc.Multivector.from_bivector([1.0, 2.0], dims=3)  # needs 3


def test_constructors_work_with_operations():
    """Verify constructors produce multivectors that work with operations."""
    import largecrimsoncanine as lcc

    # Build a vector using from_vector
    v = lcc.Multivector.from_vector([3.0, 4.0])

    # Build the same vector using basis
    e1 = lcc.Multivector.basis(1, dims=2)
    e2 = lcc.Multivector.basis(2, dims=2)
    v2 = e1 * 3.0 + e2 * 4.0

    # Should be equal
    assert v.approx_eq(v2)

    # Norm should work
    assert abs(v.norm() - 5.0) < 1e-10


def test_constructor_with_wedge():
    """Basis vectors wedged should equal pseudoscalar."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.basis(1, dims=2)
    e2 = lcc.Multivector.basis(2, dims=2)
    I = lcc.Multivector.pseudoscalar(2)

    # e1 ^ e2 should equal the pseudoscalar
    e1_wedge_e2 = e1 ^ e2
    assert e1_wedge_e2.approx_eq(I)


# =============================================================================
# CONSTRUCTOR EDGE CASE TESTS
# =============================================================================

def test_from_scalar_zero():
    """from_scalar(0.0) creates true zero multivector."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(0.0, dims=2)
    assert s.norm() == 0.0
    assert all(c == 0.0 for c in s.to_list())


def test_from_scalar_negative():
    """from_scalar with negative value."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(-5.0, dims=2)
    assert s.scalar() == -5.0


def test_from_scalar_dims_zero_error():
    """from_scalar with dims=0 should raise."""
    import largecrimsoncanine as lcc
    with pytest.raises(ValueError):
        lcc.Multivector.from_scalar(1.0, dims=0)


def test_pseudoscalar_dims_zero_error():
    """pseudoscalar with dims=0 should raise."""
    import largecrimsoncanine as lcc
    with pytest.raises(ValueError):
        lcc.Multivector.pseudoscalar(0)


def test_pseudoscalar_norm():
    """Pseudoscalar should have unit norm."""
    import largecrimsoncanine as lcc
    I2 = lcc.Multivector.pseudoscalar(2)
    I3 = lcc.Multivector.pseudoscalar(3)
    assert I2.norm() == 1.0
    assert I3.norm() == 1.0


def test_from_bivector_all_zeros():
    """from_bivector with all zeros creates zero bivector."""
    import largecrimsoncanine as lcc
    B = lcc.Multivector.from_bivector([0.0, 0.0, 0.0], dims=3)
    assert B.norm() == 0.0
    assert all(c == 0.0 for c in B.to_list())


def test_from_bivector_dims_one_error():
    """from_bivector with dims=1 should raise (no bivectors in 1D)."""
    import largecrimsoncanine as lcc
    with pytest.raises(ValueError):
        lcc.Multivector.from_bivector([1.0], dims=1)


def test_4d_zero_constructor():
    """Test 4D algebra (Cl(4) with 16 coefficients)."""
    import largecrimsoncanine as lcc
    z = lcc.Multivector.zero(4)
    assert len(z) == 16  # 2^4 = 16
    assert all(c == 0.0 for c in z.to_list())


def test_4d_basis_vectors():
    """Test 4D basis vectors."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.basis(1, dims=4)
    e2 = lcc.Multivector.basis(2, dims=4)
    e3 = lcc.Multivector.basis(3, dims=4)
    e4 = lcc.Multivector.basis(4, dims=4)

    # All should have 16 coefficients
    assert len(e1) == 16
    assert len(e4) == 16

    # e4 should be at index 8 (2^3)
    assert e4[8] == 1.0

    # All should square to 1
    assert (e1 * e1).scalar() == 1.0
    assert (e4 * e4).scalar() == 1.0


def test_4d_pseudoscalar():
    """Test 4D pseudoscalar (e1234)."""
    import largecrimsoncanine as lcc
    I4 = lcc.Multivector.pseudoscalar(4)

    # Should be at index 15 (2^4 - 1 = 0b1111)
    assert I4[15] == 1.0
    assert I4.norm() == 1.0

    # In 4D: (e1234)² = +1 (even number of swaps)
    I4_sq = I4 * I4
    assert I4_sq.scalar() == 1.0


def test_4d_bivector():
    """Test 4D bivector (6 components)."""
    import largecrimsoncanine as lcc
    # 4D has C(4,2) = 6 bivector components: e12, e13, e14, e23, e24, e34
    B = lcc.Multivector.from_bivector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dims=4)

    # Check indices: e12=3, e13=5, e14=9, e23=6, e24=10, e34=12
    assert B[3] == 1.0   # e12
    assert B[5] == 2.0   # e13
    assert B[9] == 3.0   # e14
    assert B[6] == 4.0   # e23
    assert B[10] == 5.0  # e24
    assert B[12] == 6.0  # e34


def test_scalar_plus_vector():
    """Scalar + vector creates mixed multivector."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(5.0, dims=2)
    v = lcc.Multivector.from_vector([3.0, 4.0])

    mv = s + v
    assert mv[0] == 5.0  # scalar part
    assert mv[1] == 3.0  # e1
    assert mv[2] == 4.0  # e2


def test_scalar_times_vector():
    """Scalar * vector = scaled vector."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(2.0, dims=2)
    v = lcc.Multivector.from_vector([3.0, 4.0])

    result = s * v
    assert result[0] == 0.0  # no scalar
    assert result[1] == 6.0  # 2 * 3
    assert result[2] == 8.0  # 2 * 4


def test_from_vector_single_component():
    """from_vector with single component creates 1D vector."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([5.0])
    assert len(v) == 2  # 2^1
    assert v[0] == 0.0  # no scalar
    assert v[1] == 5.0  # e1


def test_basis_dims_zero_error():
    """basis with dims=0 should raise."""
    import largecrimsoncanine as lcc
    with pytest.raises(ValueError):
        lcc.Multivector.basis(1, dims=0)


# =============================================================================
# INVERSE AND DIVISION TESTS
# =============================================================================

def test_scalar_inverse():
    """Inverse of a scalar."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(2.0, dims=2)
    s_inv = s.inverse()

    # 2 * (1/2) = 1
    result = s * s_inv
    assert abs(result.scalar() - 1.0) < 1e-10
    assert abs(result[1]) < 1e-10  # no e1
    assert abs(result[2]) < 1e-10  # no e2


def test_vector_inverse():
    """Inverse of a vector."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([3.0, 4.0])  # |v|² = 25

    v_inv = v.inverse()

    # v * v⁻¹ should equal 1
    result = v * v_inv
    assert abs(result.scalar() - 1.0) < 1e-10
    # All other components should be zero
    for i in range(1, len(result)):
        assert abs(result[i]) < 1e-10


def test_basis_vector_inverse():
    """Inverse of basis vector."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.basis(1, dims=3)

    e1_inv = e1.inverse()

    # e1 * e1⁻¹ = 1
    result = e1 * e1_inv
    assert abs(result.scalar() - 1.0) < 1e-10


def test_bivector_inverse():
    """Inverse of a bivector."""
    import largecrimsoncanine as lcc
    # e12 has norm_squared = 1 (in Euclidean)
    e12 = lcc.Multivector.pseudoscalar(2)

    e12_inv = e12.inverse()

    # e12 * e12⁻¹ = 1
    result = e12 * e12_inv
    assert abs(result.scalar() - 1.0) < 1e-10


def test_rotor_inverse():
    """Inverse of a rotor (scalar + bivector)."""
    import largecrimsoncanine as lcc
    import math

    # Create a unit rotor: cos(θ/2) + sin(θ/2)*e12
    theta = math.pi / 4  # 45 degrees
    s = lcc.Multivector.from_scalar(math.cos(theta / 2), dims=2)
    e12 = lcc.Multivector.pseudoscalar(2)
    R = s + e12 * math.sin(theta / 2)

    R_inv = R.inverse()

    # R * R⁻¹ = 1
    result = R * R_inv
    assert abs(result.scalar() - 1.0) < 1e-10
    assert abs(result[3]) < 1e-10  # no e12


def test_inverse_zero_raises():
    """Inverse of zero multivector should raise."""
    import largecrimsoncanine as lcc
    z = lcc.Multivector.zero(3)
    with pytest.raises(ValueError):
        z.inverse()


def test_division_by_scalar():
    """A / scalar works."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([6.0, 8.0])
    result = v / 2.0
    assert result[1] == 3.0
    assert result[2] == 4.0


def test_division_by_multivector():
    """A / B = A * B⁻¹ works."""
    import largecrimsoncanine as lcc
    # v / v should give 1 (scalar)
    v = lcc.Multivector.from_vector([3.0, 4.0])
    result = v / v
    assert abs(result.scalar() - 1.0) < 1e-10


def test_division_solves_equation():
    """Test that A / B solves the equation X * B = A."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.basis(1, dims=2)
    e2 = lcc.Multivector.basis(2, dims=2)

    # A = e1 + 2*e2
    A = e1 + e2 * 2.0

    # B = e1
    B = e1

    # X = A / B should satisfy X * B = A
    X = A / B
    result = X * B

    assert result.approx_eq(A)


def test_division_by_zero_scalar_raises():
    """Division by zero scalar should raise."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0])
    with pytest.raises(ZeroDivisionError):
        v / 0.0


def test_division_by_zero_multivector_raises():
    """Division by zero multivector should raise."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0])
    z = lcc.Multivector.zero(2)
    with pytest.raises(ValueError):
        v / z


def test_division_dimension_mismatch():
    """Division with dimension mismatch should raise."""
    import largecrimsoncanine as lcc
    v2 = lcc.Multivector.from_vector([1.0, 2.0])  # Cl(2)
    v3 = lcc.Multivector.from_vector([1.0, 2.0, 3.0])  # Cl(3)
    with pytest.raises(ValueError):
        v2 / v3


def test_inverse_inverse_identity():
    """(A⁻¹)⁻¹ = A"""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([3.0, 4.0])
    v_inv_inv = v.inverse().inverse()
    assert v.approx_eq(v_inv_inv)


def test_product_inverse():
    """(A * B)⁻¹ = B⁻¹ * A⁻¹ for invertible A, B."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.basis(1, dims=2)
    e2 = lcc.Multivector.basis(2, dims=2)

    AB = e1 * e2
    AB_inv = AB.inverse()

    # B⁻¹ * A⁻¹
    B_inv_A_inv = e2.inverse() * e1.inverse()

    assert AB_inv.approx_eq(B_inv_A_inv)

