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

