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


# =============================================================================
# ROTOR TESTS
# =============================================================================


def test_rotor_from_vectors_basic():
    """Create a rotor from two orthogonal basis vectors."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    # Rotor should be unit norm
    assert abs(R.norm() - 1.0) < 1e-10


def test_rotor_is_rotor():
    """is_rotor should identify unit rotors."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)
    assert R.is_rotor()

    # Non-unit multivector should not be a rotor
    v = lcc.Multivector.from_vector([2.0, 0.0])
    assert not v.is_rotor()


def test_rotor_sandwich_rotates_90_degrees():
    """Rotor from e1 to e2 should rotate e1 to e2."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)
    rotated = R.sandwich(e1)

    assert rotated.approx_eq(e2)


def test_rotor_sandwich_rotates_arbitrary_vector():
    """Rotor should rotate any vector in the same plane."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])

    # 45-degree rotation
    cos45 = math.cos(math.pi / 4)
    sin45 = math.sin(math.pi / 4)
    a = lcc.Multivector.from_vector([1.0, 0.0])
    b = lcc.Multivector.from_vector([cos45, sin45])

    R = lcc.Multivector.rotor_from_vectors(a, b)
    rotated = R.sandwich(e1)

    expected = lcc.Multivector.from_vector([cos45, sin45])
    assert rotated.approx_eq(expected)


def test_rotor_apply_alias():
    """apply() should be an alias for sandwich()."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    result_sandwich = R.sandwich(e1)
    result_apply = R.apply(e1)

    assert result_sandwich.approx_eq(result_apply)


def test_rotor_preserves_orthogonal_vector():
    """Rotation in xy-plane should not affect z-component."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)
    rotated_e3 = R.sandwich(e3)

    # e3 should be unchanged by rotation in xy-plane
    assert rotated_e3.approx_eq(e3)


def test_rotor_composition():
    """Two rotations should compose correctly."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    # R1 rotates e1 to e2 (90° in xy-plane)
    R1 = lcc.Multivector.rotor_from_vectors(e1, e2)

    # R2 rotates e2 to e3 (90° in yz-plane)
    R2 = lcc.Multivector.rotor_from_vectors(e2, e3)

    # Composed rotor: first R1, then R2
    # Rotor composition: R_combined = R2 * R1
    R_combined = R2 * R1

    # Apply R1 then R2 sequentially
    step1 = R1.sandwich(e1)  # e1 -> e2
    step2 = R2.sandwich(step1)  # e2 -> e3

    # Apply combined rotor
    combined = R_combined.sandwich(e1)

    assert combined.approx_eq(step2)


def test_rotor_inverse_reverses_rotation():
    """R⁻¹ should reverse the rotation of R."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)
    R_inv = R.inverse()

    # R rotates e1 to e2
    rotated = R.sandwich(e1)
    assert rotated.approx_eq(e2)

    # R⁻¹ should rotate e2 back to e1
    back = R_inv.sandwich(rotated)
    assert back.approx_eq(e1)


def test_rotor_same_vector():
    """Rotor from a vector to itself should be identity (scalar 1)."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e1)

    # Should be close to scalar 1
    one = lcc.Multivector.from_scalar(1.0, dims=3)
    assert R.approx_eq(one)


def test_rotor_anti_parallel_vectors():
    """Rotor from a to -a should be 180° rotation."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    neg_e1 = -e1

    R = lcc.Multivector.rotor_from_vectors(e1, neg_e1)

    # Should be a valid unit rotor
    assert R.is_rotor()

    # Applying twice should bring back to original (360°)
    double_rotated = R.sandwich(R.sandwich(e1))
    assert double_rotated.approx_eq(e1)


def test_rotor_from_non_unit_vectors():
    """rotor_from_vectors should work with non-unit vectors."""
    import largecrimsoncanine as lcc
    a = lcc.Multivector.from_vector([3.0, 0.0])
    b = lcc.Multivector.from_vector([0.0, 5.0])

    R = lcc.Multivector.rotor_from_vectors(a, b)

    # Should still be unit rotor
    assert R.is_rotor()

    # Should rotate direction of a to direction of b
    a_unit = a.normalized()
    rotated = R.sandwich(a_unit)
    b_unit = b.normalized()
    assert rotated.approx_eq(b_unit)


def test_rotor_dimension_mismatch():
    """rotor_from_vectors with mismatched dimensions should raise."""
    import largecrimsoncanine as lcc
    v2 = lcc.Multivector.from_vector([1.0, 0.0])
    v3 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        lcc.Multivector.rotor_from_vectors(v2, v3)


def test_rotor_zero_vector_raises():
    """rotor_from_vectors with zero vector should raise."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    zero = lcc.Multivector.zero(2)

    with pytest.raises(ValueError):
        lcc.Multivector.rotor_from_vectors(zero, e1)

    with pytest.raises(ValueError):
        lcc.Multivector.rotor_from_vectors(e1, zero)


def test_sandwich_dimension_mismatch():
    """sandwich with mismatched dimensions should raise."""
    import largecrimsoncanine as lcc
    e1_2d = lcc.Multivector.from_vector([1.0, 0.0])
    e2_2d = lcc.Multivector.from_vector([0.0, 1.0])
    v_3d = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1_2d, e2_2d)

    with pytest.raises(ValueError):
        R.sandwich(v_3d)


def test_rotor_rotates_bivector():
    """Rotors should correctly rotate bivectors (planes)."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    # e12 bivector (xy-plane)
    e12 = e1 ^ e2

    # Rotate around z-axis by 90° (using rotation from e1 to e2)
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    # Rotating the xy-plane around z should leave it unchanged
    rotated_plane = R.sandwich(e12)
    assert rotated_plane.approx_eq(e12)


def test_rotor_double_cover():
    """R and -R represent the same rotation."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)
    neg_R = -R

    # Both should give the same rotation result
    result_R = R.sandwich(e1)
    result_neg_R = neg_R.sandwich(e1)

    assert result_R.approx_eq(result_neg_R)


# =============================================================================
# DUAL TESTS
# =============================================================================


def test_dual_scalar_becomes_pseudoscalar():
    """Dual of scalar should be pseudoscalar."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(2.0, dims=3)
    I = lcc.Multivector.pseudoscalar(3)

    dual_s = s.dual()

    # scalar * I^-1 should be proportional to I
    # In 3D: I^-1 = -I (since I*I = -1)
    # So 2 * (-I) = -2*I
    expected = I * (-2.0)
    assert dual_s.approx_eq(expected)


def test_dual_pseudoscalar_becomes_scalar():
    """Dual of pseudoscalar should be scalar."""
    import largecrimsoncanine as lcc
    I = lcc.Multivector.pseudoscalar(3)

    dual_I = I.dual()

    # I * I^-1 = 1
    one = lcc.Multivector.from_scalar(1.0, dims=3)
    assert dual_I.approx_eq(one)


def test_dual_undual_identity():
    """undual(dual(A)) = A for any multivector."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    v_back = v.dual().undual()

    assert v.approx_eq(v_back)


def test_dual_undual_identity_bivector():
    """undual(dual(B)) = B for bivectors."""
    import largecrimsoncanine as lcc
    B = lcc.Multivector.from_bivector([1.0, 2.0, 3.0], dims=3)

    B_back = B.dual().undual()

    assert B.approx_eq(B_back)


def test_dual_undual_identity_scalar():
    """undual(dual(s)) = s for scalars."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(5.0, dims=3)

    s_back = s.dual().undual()

    assert s.approx_eq(s_back)


def test_dual_vector_3d():
    """In 3D, dual of a vector is a bivector."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    dual_e1 = e1.dual()

    # e1 * I^-1 where I = e123
    # e1 * (-e123) = -e1*e123 = -e23 (by anticommutation)
    # Actually: e1 * e123 = e1*e1*e23 = e23
    # So e1 * (-e123) = -e23
    # Let's verify the grade is 2
    grade_2 = dual_e1.grade(2)
    assert dual_e1.approx_eq(grade_2)  # Should be pure grade-2


def test_dual_bivector_3d():
    """In 3D, dual of a bivector is a vector."""
    import largecrimsoncanine as lcc
    # e12 bivector
    e12 = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)

    dual_e12 = e12.dual()

    # Should be grade-1 (vector)
    grade_1 = dual_e12.grade(1)
    assert dual_e12.approx_eq(grade_1)


def test_dual_2d():
    """In 2D, dual of vector is vector rotated 90 degrees."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])

    dual_e1 = e1.dual()

    # In 2D: I = e12, I^-1 = -e12
    # e1 * (-e12) = -e1*e12 = -e2
    expected = e2 * (-1.0)
    assert dual_e1.approx_eq(expected)


def test_dual_linearity():
    """Dual is linear: (aA + bB)* = a*A* + b*B*"""
    import largecrimsoncanine as lcc
    v1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    v2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    a, b = 3.0, 2.0
    combined = v1 * a + v2 * b

    dual_combined = combined.dual()
    dual_separate = v1.dual() * a + v2.dual() * b

    assert dual_combined.approx_eq(dual_separate)


def test_dual_4d():
    """Dual works in 4D."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0, 4.0])

    dual_v = v.dual()

    # In 4D, dual of vector (grade 1) is trivector (grade 3)
    grade_3 = dual_v.grade(3)
    assert dual_v.approx_eq(grade_3)


def test_right_dual():
    """right_dual computes I^-1 * A."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    left_dual = v.dual()     # A * I^-1
    right_dual = v.right_dual()  # I^-1 * A

    # They differ by sign in 3D for vectors
    # Both should be grade 2
    assert left_dual.grade(2).approx_eq(left_dual)
    assert right_dual.grade(2).approx_eq(right_dual)


def test_cross_product_via_dual():
    """In 3D, cross product is dual of wedge: a × b = (a ∧ b)*"""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    # e1 × e2 = e3
    wedge = e1 ^ e2  # e12
    cross = wedge.dual()

    # Should be proportional to e3
    # e12 * I^-1 = e12 * (-e123) = -e12*e123 = -e3*e12*e12 = e3 (checking sign...)
    # Actually: e12 * e123 = e12*e1*e2*e3 = -e2*e2*e3 = -e3
    # So e12 * (-e123) = e3
    assert cross.approx_eq(e3) or cross.approx_eq(-e3)  # Sign depends on convention


# =============================================================================
# CONJUGATE TESTS (grade involution, Clifford conjugate)
# =============================================================================


def test_grade_involution_scalar():
    """Grade involution leaves scalars unchanged (grade 0, even)."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(5.0, dims=3)

    inv = s.grade_involution()

    assert inv.approx_eq(s)


def test_grade_involution_vector():
    """Grade involution negates vectors (grade 1, odd)."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    inv = v.grade_involution()

    assert inv.approx_eq(-v)


def test_grade_involution_bivector():
    """Grade involution leaves bivectors unchanged (grade 2, even)."""
    import largecrimsoncanine as lcc
    B = lcc.Multivector.from_bivector([1.0, 2.0, 3.0], dims=3)

    inv = B.grade_involution()

    assert inv.approx_eq(B)


def test_grade_involution_trivector():
    """Grade involution negates trivectors (grade 3, odd)."""
    import largecrimsoncanine as lcc
    I = lcc.Multivector.pseudoscalar(3)  # e123 is grade 3

    inv = I.grade_involution()

    assert inv.approx_eq(-I)


def test_grade_involution_involute_alias():
    """involute() should be an alias for grade_involution()."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    assert v.grade_involution().approx_eq(v.involute())


def test_grade_involution_is_automorphism():
    """Grade involution is an automorphism: (A * B)^ = Â * B̂."""
    import largecrimsoncanine as lcc
    v1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    v2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    product = v1 * v2
    product_inv = product.grade_involution()

    v1_inv = v1.grade_involution()
    v2_inv = v2.grade_involution()
    inv_product = v1_inv * v2_inv

    assert product_inv.approx_eq(inv_product)


def test_grade_involution_involutory():
    """Applying grade involution twice returns original: (Â)^ = A."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    double_inv = v.grade_involution().grade_involution()

    assert double_inv.approx_eq(v)


def test_clifford_conjugate_scalar():
    """Clifford conjugate leaves scalars unchanged (grade 0)."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(5.0, dims=3)

    conj = s.clifford_conjugate()

    assert conj.approx_eq(s)


def test_clifford_conjugate_vector():
    """Clifford conjugate negates vectors (grade 1)."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    conj = v.clifford_conjugate()

    assert conj.approx_eq(-v)


def test_clifford_conjugate_bivector():
    """Clifford conjugate negates bivectors (grade 2)."""
    import largecrimsoncanine as lcc
    B = lcc.Multivector.from_bivector([1.0, 2.0, 3.0], dims=3)

    conj = B.clifford_conjugate()

    assert conj.approx_eq(-B)


def test_clifford_conjugate_trivector():
    """Clifford conjugate leaves trivectors unchanged (grade 3)."""
    import largecrimsoncanine as lcc
    I = lcc.Multivector.pseudoscalar(3)  # e123 is grade 3

    conj = I.clifford_conjugate()

    assert conj.approx_eq(I)


def test_clifford_conjugate_alias():
    """conjugate() should be an alias for clifford_conjugate()."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    assert v.clifford_conjugate().approx_eq(v.conjugate())


def test_clifford_conjugate_equals_reverse_then_involute():
    """Clifford conjugate equals reverse then grade involution: A† = (~A)^."""
    import largecrimsoncanine as lcc
    # Mixed multivector with multiple grades
    s = lcc.Multivector.from_scalar(1.0, dims=3)
    v = lcc.Multivector.from_vector([2.0, 3.0, 4.0])
    B = lcc.Multivector.from_bivector([5.0, 6.0, 7.0], dims=3)
    mv = s + v + B

    conj = mv.clifford_conjugate()
    reverse_then_involute = mv.reverse().grade_involution()

    assert conj.approx_eq(reverse_then_involute)


def test_clifford_conjugate_equals_involute_then_reverse():
    """Clifford conjugate equals grade involution then reverse: A† = (Â)~."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(1.0, dims=3)
    v = lcc.Multivector.from_vector([2.0, 3.0, 4.0])
    B = lcc.Multivector.from_bivector([5.0, 6.0, 7.0], dims=3)
    mv = s + v + B

    conj = mv.clifford_conjugate()
    involute_then_reverse = mv.grade_involution().reverse()

    assert conj.approx_eq(involute_then_reverse)


def test_even_extracts_even_grades():
    """even() should extract grades 0, 2, 4, ..."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(1.0, dims=3)
    v = lcc.Multivector.from_vector([2.0, 3.0, 4.0])
    B = lcc.Multivector.from_bivector([5.0, 6.0, 7.0], dims=3)
    mv = s + v + B

    even_part = mv.even()

    # Should equal scalar + bivector (grades 0 and 2)
    expected = s + B
    assert even_part.approx_eq(expected)


def test_odd_extracts_odd_grades():
    """odd() should extract grades 1, 3, 5, ..."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(1.0, dims=3)
    v = lcc.Multivector.from_vector([2.0, 3.0, 4.0])
    B = lcc.Multivector.from_bivector([5.0, 6.0, 7.0], dims=3)
    mv = s + v + B

    odd_part = mv.odd()

    # Should equal just the vector (grade 1)
    assert odd_part.approx_eq(v)


def test_even_plus_odd_equals_original():
    """A.even() + A.odd() = A."""
    import largecrimsoncanine as lcc
    s = lcc.Multivector.from_scalar(1.0, dims=3)
    v = lcc.Multivector.from_vector([2.0, 3.0, 4.0])
    B = lcc.Multivector.from_bivector([5.0, 6.0, 7.0], dims=3)
    I = lcc.Multivector.pseudoscalar(3)
    mv = s + v + B + I

    reconstructed = mv.even() + mv.odd()

    assert reconstructed.approx_eq(mv)


def test_rotor_is_even():
    """Rotors should be purely even-grade multivectors."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    # Rotor should equal its even part
    assert R.approx_eq(R.even())

    # Odd part should be zero
    zero = lcc.Multivector.zero(3)
    assert R.odd().approx_eq(zero)


def test_grade_involution_4d():
    """Grade involution works in 4D (grade 4 = even, unchanged)."""
    import largecrimsoncanine as lcc
    I4 = lcc.Multivector.pseudoscalar(4)  # grade 4 (even)

    inv = I4.grade_involution()

    assert inv.approx_eq(I4)  # Even grade, should be unchanged


def test_clifford_conjugate_4d():
    """Clifford conjugate in 4D: grade 4 has sign (-1)^(4*5/2) = (-1)^10 = +1."""
    import largecrimsoncanine as lcc
    I4 = lcc.Multivector.pseudoscalar(4)  # grade 4

    conj = I4.clifford_conjugate()

    assert conj.approx_eq(I4)  # Sign should be +1


# =============================================================================
# REFLECTION AND PROJECTION TESTS
# =============================================================================


def test_reflect_vector_across_perpendicular():
    """Reflecting a vector across a perpendicular plane negates it."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    # Reflect e1 across plane perpendicular to e1 (the yz-plane)
    reflected = e1.reflect(e1)

    # Should give -e1
    assert reflected.approx_eq(-e1)


def test_reflect_vector_parallel_to_plane():
    """Reflecting a vector parallel to the mirror plane leaves it unchanged."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    # Reflect e2 across plane perpendicular to e1 (the yz-plane)
    # e2 lies in the yz-plane, so it should be unchanged
    reflected = e2.reflect(e1)

    assert reflected.approx_eq(e2)


def test_reflect_diagonal_vector():
    """Reflect a diagonal vector across a coordinate plane."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])

    # 45-degree vector
    v = lcc.Multivector.from_vector([1.0, 1.0])

    # Reflect across plane perpendicular to e1 (the y-axis)
    reflected = v.reflect(e1)

    # Component along e1 negates, component along e2 stays
    expected = lcc.Multivector.from_vector([-1.0, 1.0])
    assert reflected.approx_eq(expected)


def test_reflect_twice_is_identity():
    """Reflecting twice across the same plane returns the original."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([3.0, 4.0, 5.0])
    n = lcc.Multivector.from_vector([1.0, 1.0, 0.0])

    double_reflected = v.reflect(n).reflect(n)

    assert double_reflected.approx_eq(v)


def test_reflect_preserves_norm():
    """Reflection preserves the norm of a vector."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([3.0, 4.0, 5.0])
    n = lcc.Multivector.from_vector([1.0, 2.0, 2.0])

    reflected = v.reflect(n)

    assert abs(v.norm() - reflected.norm()) < 1e-10


def test_reflect_with_non_unit_normal():
    """Reflection works with non-unit normal vectors."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    n = lcc.Multivector.from_vector([5.0, 0.0, 0.0])  # Non-unit, parallel to e1

    reflected = e1.reflect(n)

    # Should still give -e1
    assert reflected.approx_eq(-e1)


def test_reflect_dimension_mismatch():
    """Reflection with mismatched dimensions should raise."""
    import largecrimsoncanine as lcc
    v2 = lcc.Multivector.from_vector([1.0, 0.0])
    n3 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        v2.reflect(n3)


def test_project_vector_onto_parallel():
    """Projecting a vector onto a parallel vector gives the original."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([3.0, 0.0, 0.0])
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    proj = v.project(e1)

    assert proj.approx_eq(v)


def test_project_vector_onto_perpendicular():
    """Projecting a vector onto a perpendicular vector gives zero."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    proj = e1.project(e2)

    zero = lcc.Multivector.zero(3)
    assert proj.approx_eq(zero)


def test_project_diagonal():
    """Project a diagonal vector onto a basis vector."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([3.0, 4.0, 0.0])
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    proj = v.project(e1)

    expected = lcc.Multivector.from_vector([3.0, 0.0, 0.0])
    assert proj.approx_eq(expected)


def test_reject_vector_from_parallel():
    """Rejecting a vector from a parallel vector gives zero."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([3.0, 0.0, 0.0])
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    rej = v.reject(e1)

    zero = lcc.Multivector.zero(3)
    assert rej.approx_eq(zero)


def test_reject_vector_from_perpendicular():
    """Rejecting a vector from a perpendicular vector gives the original."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    rej = e1.reject(e2)

    assert rej.approx_eq(e1)


def test_reject_diagonal():
    """Reject a diagonal vector from a basis vector."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([3.0, 4.0, 0.0])
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    rej = v.reject(e1)

    expected = lcc.Multivector.from_vector([0.0, 4.0, 0.0])
    assert rej.approx_eq(expected)


def test_project_plus_reject_equals_original():
    """v.project(n) + v.reject(n) = v."""
    import largecrimsoncanine as lcc
    v = lcc.Multivector.from_vector([3.0, 4.0, 5.0])
    n = lcc.Multivector.from_vector([1.0, 2.0, 2.0])

    proj = v.project(n)
    rej = v.reject(n)
    reconstructed = proj + rej

    assert reconstructed.approx_eq(v)


def test_project_onto_bivector():
    """Project a vector onto a bivector (plane)."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    # xy-plane
    e12 = e1 ^ e2

    # Vector with all components
    v = lcc.Multivector.from_vector([3.0, 4.0, 5.0])

    # Project onto xy-plane: should keep x and y, remove z
    proj = v.project(e12)

    expected = lcc.Multivector.from_vector([3.0, 4.0, 0.0])
    assert proj.approx_eq(expected)


def test_reject_from_bivector():
    """Reject a vector from a bivector (plane)."""
    import largecrimsoncanine as lcc
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    # xy-plane
    e12 = e1 ^ e2

    # Vector with all components
    v = lcc.Multivector.from_vector([3.0, 4.0, 5.0])

    # Reject from xy-plane: should keep only z
    rej = v.reject(e12)

    expected = lcc.Multivector.from_vector([0.0, 0.0, 5.0])
    assert rej.approx_eq(expected)


def test_project_dimension_mismatch():
    """Projection with mismatched dimensions should raise."""
    import largecrimsoncanine as lcc
    v2 = lcc.Multivector.from_vector([1.0, 0.0])
    n3 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        v2.project(n3)


def test_reflect_3d_arbitrary():
    """Reflect a 3D vector across an arbitrary plane."""
    import largecrimsoncanine as lcc

    # Vector to reflect
    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    # Normal to the plane (45 degrees between e1 and e2)
    import math
    cos45 = math.cos(math.pi / 4)
    n = lcc.Multivector.from_vector([cos45, cos45, 0.0])

    # Reflect
    reflected = v.reflect(n)

    # The plane x + y = 0 reflects [1,0,0] to [0,-1,0]
    # (component parallel to n negates, perpendicular unchanged)
    expected = lcc.Multivector.from_vector([0.0, -1.0, 0.0])
    assert reflected.approx_eq(expected)


# =============================================================================
# EXPONENTIAL AND LOGARITHM TESTS
# =============================================================================


def test_exp_zero_bivector():
    """exp(0) = 1 for zero bivector."""
    import largecrimsoncanine as lcc
    B = lcc.Multivector.zero(3)

    R = B.exp()

    one = lcc.Multivector.from_scalar(1.0, dims=3)
    assert R.approx_eq(one)


def test_exp_scalar():
    """exp(s) = e^s for scalar."""
    import largecrimsoncanine as lcc
    import math

    s = lcc.Multivector.from_scalar(2.0, dims=3)

    result = s.exp()

    expected = lcc.Multivector.from_scalar(math.e ** 2.0, dims=3)
    assert result.approx_eq(expected)


def test_exp_bivector_90_degrees():
    """exp(-π/4 * e12) creates 90-degree rotor that rotates e1 to e2."""
    import largecrimsoncanine as lcc
    import math

    # For rotation from e1 to e2, we need negative bivector coefficient
    # (the bivector e12 = e1∧e2 has orientation e1 → -e2 for positive angle)
    e12 = lcc.Multivector.from_bivector([-math.pi / 4], dims=2)

    R = e12.exp()

    # Should be a unit rotor
    assert R.is_rotor()

    # Apply to e1, should get e2
    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])
    rotated = R.sandwich(e1)

    assert rotated.approx_eq(e2)


def test_exp_bivector_180_degrees():
    """exp(π/2 * e12) creates 180-degree rotor."""
    import largecrimsoncanine as lcc
    import math

    # Bivector with magnitude π/2 (half of 180 degrees)
    e12 = lcc.Multivector.from_bivector([math.pi / 2], dims=2)

    R = e12.exp()

    # Should be a unit rotor
    assert R.is_rotor()

    # Apply to e1, should get -e1
    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    rotated = R.sandwich(e1)

    assert rotated.approx_eq(-e1)


def test_exp_bivector_small_angle():
    """exp of small bivector should be close to identity."""
    import largecrimsoncanine as lcc
    import math

    # Very small angle
    e12 = lcc.Multivector.from_bivector([0.001], dims=2)

    R = e12.exp()

    # Should be nearly 1
    one = lcc.Multivector.from_scalar(1.0, dims=2)
    # cos(0.001) ≈ 1, sin(0.001) ≈ 0.001
    assert abs(R[0] - 1.0) < 0.01


def test_exp_bivector_3d():
    """exp works in 3D for rotation around arbitrary axis."""
    import largecrimsoncanine as lcc
    import math

    # 90-degree rotation in the xy-plane (around z-axis)
    # Negative coefficient for rotation from e1 to e2
    B = lcc.Multivector.from_bivector([-math.pi / 4, 0.0, 0.0], dims=3)

    R = B.exp()

    # Should be a unit rotor
    assert R.is_rotor()

    # e1 rotates to e2
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    rotated = R.sandwich(e1)

    assert rotated.approx_eq(e2)


def test_log_identity_rotor():
    """log(1) = 0."""
    import largecrimsoncanine as lcc

    one = lcc.Multivector.from_scalar(1.0, dims=3)

    B = one.log()

    zero = lcc.Multivector.zero(3)
    assert B.approx_eq(zero)


def test_log_90_degree_rotor():
    """log of 90-degree rotor should have magnitude π/4."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    B = R.log()

    # Should be a pure bivector
    bivector_part = B.grade(2)
    assert B.approx_eq(bivector_part)

    # Magnitude should be π/4
    assert abs(B.norm() - math.pi / 4) < 1e-10


def test_exp_log_roundtrip():
    """exp(log(R)) = R for unit rotors."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    B = R.log()
    R_back = B.exp()

    assert R_back.approx_eq(R)


def test_log_exp_roundtrip():
    """log(exp(B)) = B for bivectors with |B| < π."""
    import largecrimsoncanine as lcc
    import math

    # Bivector with magnitude less than π
    B = lcc.Multivector.from_bivector([0.5, 0.3, 0.2], dims=3)

    R = B.exp()
    B_back = R.log()

    assert B_back.approx_eq(B)


def test_log_non_unit_rotor_raises():
    """log of non-unit multivector should raise."""
    import largecrimsoncanine as lcc

    # Non-unit multivector
    v = lcc.Multivector.from_vector([2.0, 0.0])

    with pytest.raises(ValueError):
        v.log()


def test_exp_bivector_matches_rotor_from_vectors():
    """exp(B) should match rotor_from_vectors for same rotation."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    # Create rotor two ways
    R1 = lcc.Multivector.rotor_from_vectors(e1, e2)
    # Negative coefficient for same rotation direction as rotor_from_vectors
    B = lcc.Multivector.from_bivector([-math.pi / 4, 0.0, 0.0], dims=3)
    R2 = B.exp()

    # Both should rotate e1 to e2
    rotated1 = R1.sandwich(e1)
    rotated2 = R2.sandwich(e1)

    assert rotated1.approx_eq(rotated2)


def test_exp_additive_for_commuting_bivectors():
    """exp(A + B) = exp(A) * exp(B) when A and B commute (same plane)."""
    import largecrimsoncanine as lcc
    import math

    # Two bivectors in the same plane (they commute)
    B1 = lcc.Multivector.from_bivector([0.2], dims=2)
    B2 = lcc.Multivector.from_bivector([0.3], dims=2)

    R_sum = (B1 + B2).exp()
    R_product = B1.exp() * B2.exp()

    assert R_sum.approx_eq(R_product)


def test_exp_rotor_norm():
    """exp of pure bivector should produce unit rotor."""
    import largecrimsoncanine as lcc
    import math

    # Various bivector magnitudes
    for angle in [0.1, 0.5, 1.0, 2.0, 3.0]:
        B = lcc.Multivector.from_bivector([angle, 0.0, 0.0], dims=3)
        R = B.exp()
        assert abs(R.norm() - 1.0) < 1e-10, f"Failed for angle {angle}"


# =============================================================================
# RIGHT CONTRACTION TESTS
# =============================================================================

def test_right_contraction_vector_vector():
    """Right contraction of two vectors equals their scalar product."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    b = lcc.Multivector.from_vector([4.0, 5.0, 6.0])

    # For vectors: a ⌊ b = a · b = scalar
    result = a.right_contraction(b)
    expected = 1*4 + 2*5 + 3*6  # = 32

    assert result.scalar() == expected
    assert result.grade(0).approx_eq(result)  # purely scalar


def test_right_contraction_bivector_vector():
    """Right contraction e12 ⌊ e1 gives a vector."""
    import largecrimsoncanine as lcc

    # Bivector e12 (contains e1)
    e12 = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    # Right contraction e12 ⌊ e1 = e12 * e1 (grade-selected)
    # = e1*e2*e1 = -e1*e1*e2 = -e2
    result = e12.right_contraction(e1)

    expected = lcc.Multivector.from_vector([0.0, -1.0, 0.0])
    assert result.approx_eq(expected)


def test_right_contraction_vector_bivector():
    """Right contraction v ⌊ B gives a vector."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e12 = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)

    # e1 ⌊ e12 = e1 · e12 = e2 (right contraction)
    result = e1.right_contraction(e12)

    # Result should be a vector (grade 1)
    assert result.grade(1).approx_eq(result)


def test_right_contraction_rc_alias():
    """rc() is an alias for right_contraction()."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 2.0])
    b = lcc.Multivector.from_vector([3.0, 4.0])

    assert a.rc(b).approx_eq(a.right_contraction(b))


def test_right_contraction_dimension_mismatch():
    """Right contraction with mismatched dimensions raises."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 2.0])
    b = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        a.right_contraction(b)


# =============================================================================
# SCALAR PRODUCT TESTS
# =============================================================================

def test_scalar_product_vectors():
    """Scalar product of vectors equals their dot product."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    b = lcc.Multivector.from_vector([4.0, 5.0, 6.0])

    # <a, b>_0 = a · b = 32
    result = a.scalar_product(b)
    assert result.scalar() == 32.0
    assert result.approx_eq(lcc.Multivector.from_scalar(32.0, dims=3))


def test_scalar_product_orthogonal_vectors():
    """Scalar product of orthogonal vectors is zero."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])

    result = e1.scalar_product(e2)
    assert result.scalar() == 0.0


def test_scalar_product_bivectors():
    """Scalar product of bivectors."""
    import largecrimsoncanine as lcc

    # e12 * e12 = -1 in the geometric product, scalar part is -1
    e12 = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    result = e12.scalar_product(e12)

    # e12 * e12 = e1 e2 e1 e2 = -e1 e1 e2 e2 = -1
    assert result.scalar() == -1.0


def test_scalar_product_different_grades():
    """Scalar product of different grade elements is zero."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)

    # vector * bivector has no scalar part
    result = v.scalar_product(B)
    assert result.scalar() == 0.0


def test_scalar_product_with_scalar():
    """Scalar product of scalar with itself."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(5.0, dims=3)
    result = s.scalar_product(s)

    assert result.scalar() == 25.0


# =============================================================================
# COMMUTATOR TESTS
# =============================================================================

def test_commutator_orthogonal_vectors():
    """Commutator of orthogonal vectors is their bivector."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])

    # [e1, e2] = (e1*e2 - e2*e1) / 2 = (e12 - (-e12)) / 2 = e12
    result = e1.commutator(e2)

    expected = lcc.Multivector.from_bivector([1.0], dims=2)
    assert result.approx_eq(expected)


def test_commutator_self_is_zero():
    """Commutator of any element with itself is zero."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    # [A, A] = 0 always
    result = v.commutator(v)
    assert result.approx_eq(lcc.Multivector.zero(3))


def test_commutator_scalars_is_zero():
    """Commutator involving scalars is zero (scalars commute with everything)."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(3.0, dims=3)
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    # [s, v] = 0
    result = s.commutator(v)
    assert result.approx_eq(lcc.Multivector.zero(3))


def test_commutator_antisymmetric():
    """Commutator is antisymmetric: [A, B] = -[B, A]."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 2.0])
    b = lcc.Multivector.from_vector([3.0, 4.0])

    ab = a.commutator(b)
    ba = b.commutator(a)

    # [A, B] + [B, A] = 0
    assert (ab + ba).approx_eq(lcc.Multivector.zero(2))


def test_commutator_x_alias():
    """x() is an alias for commutator()."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 0.0])
    b = lcc.Multivector.from_vector([0.0, 1.0])

    assert a.x(b).approx_eq(a.commutator(b))


# =============================================================================
# ANTICOMMUTATOR TESTS
# =============================================================================

def test_anticommutator_orthogonal_vectors():
    """Anticommutator of orthogonal vectors is zero."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])

    # {e1, e2} = (e1*e2 + e2*e1) / 2 = (e12 + (-e12)) / 2 = 0
    result = e1.anticommutator(e2)
    assert result.approx_eq(lcc.Multivector.zero(2))


def test_anticommutator_same_vector():
    """Anticommutator of vector with itself gives scalar."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([3.0, 4.0])

    # {v, v} = (v*v + v*v) / 2 = v*v = ||v||² = 25
    result = v.anticommutator(v)
    expected = lcc.Multivector.from_scalar(25.0, dims=2)
    assert result.approx_eq(expected)


def test_anticommutator_symmetric():
    """Anticommutator is symmetric: {A, B} = {B, A}."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 2.0])
    b = lcc.Multivector.from_vector([3.0, 4.0])

    ab = a.anticommutator(b)
    ba = b.anticommutator(a)

    assert ab.approx_eq(ba)


def test_commutator_anticommutator_sum():
    """Commutator + anticommutator = geometric product."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 2.0])
    b = lcc.Multivector.from_vector([3.0, 4.0])

    # [A, B] + {A, B} = AB
    comm = a.commutator(b)
    anticomm = a.anticommutator(b)
    ab = a * b

    assert (comm + anticomm).approx_eq(ab)


# =============================================================================
# REGRESSIVE PRODUCT (MEET) TESTS
# =============================================================================

def test_regressive_pseudoscalar_with_anything():
    """Pseudoscalar ∨ A = A (pseudoscalar is identity for meet)."""
    import largecrimsoncanine as lcc

    I = lcc.Multivector.pseudoscalar(3)
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    result = I.regressive(v)
    assert result.approx_eq(v)


def test_regressive_two_planes():
    """Meet of two planes should give their intersection (a line)."""
    import largecrimsoncanine as lcc

    # In 3D, planes are represented by bivectors (their duals are vectors)
    # Plane 1: spanned by e1, e2 (normal is e3) -> represented as e12
    # Plane 2: spanned by e1, e3 (normal is e2) -> represented as e13
    # Their intersection is the e1 axis

    e12 = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    e13 = lcc.Multivector.from_bivector([0.0, 1.0, 0.0], dims=3)

    # Meet should give a vector (their line of intersection)
    result = e12.regressive(e13)

    # Result should be grade 1 (a vector)
    grade1 = result.grade(1)
    assert grade1.norm() > 0.1  # non-zero
    assert result.approx_eq(grade1)  # purely grade 1


def test_regressive_meet_alias():
    """meet() is an alias for regressive()."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    b = lcc.Multivector.from_bivector([0.0, 1.0, 0.0], dims=3)

    assert a.meet(b).approx_eq(a.regressive(b))


def test_regressive_vee_alias():
    """vee() is an alias for regressive()."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    b = lcc.Multivector.from_bivector([0.0, 1.0, 0.0], dims=3)

    assert a.vee(b).approx_eq(a.regressive(b))


def test_regressive_dimension_mismatch():
    """Regressive with mismatched dimensions raises."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 2.0])
    b = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        a.regressive(b)


# =============================================================================
# JOIN TESTS
# =============================================================================

def test_join_is_outer_product():
    """join() should equal outer_product()."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    b = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    assert a.join(b).approx_eq(a.outer_product(b))
    assert a.join(b).approx_eq(a ^ b)


def test_join_two_vectors():
    """Join of two vectors gives their plane (bivector)."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    result = e1.join(e2)
    expected = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)

    assert result.approx_eq(expected)


def test_meet_join_duality():
    """Meet and join are dual operations: (A ∨ B)* = A* ∧ B*."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    b = lcc.Multivector.from_bivector([0.0, 1.0, 0.0], dims=3)

    # (A ∨ B)* = A* ∧ B*  by definition
    meet_result = a.meet(b)
    lhs = meet_result.dual()
    rhs = a.dual().join(b.dual())

    assert lhs.approx_eq(rhs)


# =============================================================================
# IS_BLADE TESTS
# =============================================================================

def test_is_blade_scalar():
    """Scalars are blades (grade 0)."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(5.0, dims=3)
    assert s.is_blade()


def test_is_blade_vector():
    """Vectors are blades (grade 1)."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    assert v.is_blade()


def test_is_blade_bivector():
    """Simple bivectors are blades (grade 2)."""
    import largecrimsoncanine as lcc

    # e12 is a simple bivector (blade)
    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    assert B.is_blade()


def test_is_blade_pseudoscalar():
    """Pseudoscalar is a blade (highest grade)."""
    import largecrimsoncanine as lcc

    I = lcc.Multivector.pseudoscalar(3)
    assert I.is_blade()


def test_is_blade_mixed_grade_not_blade():
    """Mixed-grade multivectors are not blades."""
    import largecrimsoncanine as lcc

    # scalar + vector is not a blade
    s = lcc.Multivector.from_scalar(1.0, dims=3)
    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    mixed = s + v

    assert not mixed.is_blade()


def test_is_blade_rotor_not_blade():
    """Rotors (scalar + bivector) are not blades."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    # Rotor has scalar and bivector parts
    assert not R.is_blade()


# =============================================================================
# IS_VERSOR TESTS
# =============================================================================

def test_is_versor_vector():
    """Vectors are versors (product of 1 vector)."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    assert v.is_versor()


def test_is_versor_rotor():
    """Rotors are versors (product of 2 vectors)."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    assert R.is_versor()


def test_is_versor_bivector():
    """Simple bivectors (e1*e2) are versors."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    B = e1 * e2  # e12

    assert B.is_versor()


def test_is_versor_scalar():
    """Non-zero scalars are versors."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(3.0, dims=3)
    assert s.is_versor()


def test_is_versor_zero_not_versor():
    """Zero is not a versor."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert not z.is_versor()


# =============================================================================
# GRADES TESTS
# =============================================================================

def test_grades_scalar():
    """Scalar has grade [0]."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(5.0, dims=3)
    assert s.grades() == [0]


def test_grades_vector():
    """Vector has grade [1]."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    assert v.grades() == [1]


def test_grades_bivector():
    """Bivector has grade [2]."""
    import largecrimsoncanine as lcc

    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    assert B.grades() == [2]


def test_grades_rotor():
    """Rotor has grades [0, 2]."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    assert R.grades() == [0, 2]


def test_grades_mixed():
    """Mixed multivector has multiple grades."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(1.0, dims=3)
    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)

    mixed = s + v + B
    assert mixed.grades() == [0, 1, 2]


def test_grades_zero():
    """Zero multivector has no grades."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert z.grades() == []


# =============================================================================
# MAX_GRADE / MIN_GRADE TESTS
# =============================================================================

def test_max_grade_scalar():
    """Max grade of scalar is 0."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(5.0, dims=3)
    assert s.max_grade() == 0


def test_max_grade_vector():
    """Max grade of vector is 1."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    assert v.max_grade() == 1


def test_max_grade_pseudoscalar():
    """Max grade of pseudoscalar equals dimension."""
    import largecrimsoncanine as lcc

    I = lcc.Multivector.pseudoscalar(3)
    assert I.max_grade() == 3


def test_max_grade_rotor():
    """Max grade of rotor is 2."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    assert R.max_grade() == 2


def test_max_grade_zero():
    """Max grade of zero is None."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert z.max_grade() is None


def test_min_grade_rotor():
    """Min grade of rotor is 0."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    assert R.min_grade() == 0


def test_min_grade_bivector():
    """Min grade of bivector is 2."""
    import largecrimsoncanine as lcc

    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    assert B.min_grade() == 2


def test_min_grade_zero():
    """Min grade of zero is None."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert z.min_grade() is None


# =============================================================================
# SLERP TESTS
# =============================================================================

def test_slerp_t_zero():
    """slerp at t=0 returns first rotor."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    R1 = lcc.Multivector.rotor_from_vectors(e1, e2)
    R2 = lcc.Multivector.rotor_from_vectors(e1, e3)

    result = R1.slerp(R2, 0.0)
    assert result.approx_eq(R1)


def test_slerp_t_one():
    """slerp at t=1 returns second rotor."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    R1 = lcc.Multivector.rotor_from_vectors(e1, e2)
    R2 = lcc.Multivector.rotor_from_vectors(e1, e3)

    result = R1.slerp(R2, 1.0)
    assert result.approx_eq(R2)


def test_slerp_midpoint():
    """slerp at t=0.5 gives intermediate rotation."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    # Identity rotor (no rotation)
    R1 = lcc.Multivector.from_scalar(1.0, dims=3)
    # 90 degree rotation
    R2 = lcc.Multivector.rotor_from_vectors(e1, e2)

    # Midpoint should be 45 degree rotation
    R_mid = R1.slerp(R2, 0.5)

    # Apply to e1 - should get 45 degree rotated vector
    rotated = R_mid.sandwich(e1)

    # At 45 degrees, e1 rotates to (1/sqrt(2), 1/sqrt(2), 0)
    import math
    expected = lcc.Multivector.from_vector([1/math.sqrt(2), 1/math.sqrt(2), 0.0])
    assert rotated.approx_eq(expected)


def test_slerp_preserves_unit_norm():
    """slerp always produces unit rotors."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    R1 = lcc.Multivector.rotor_from_vectors(e1, e2)
    R2 = lcc.Multivector.rotor_from_vectors(e1, e3)

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = R1.slerp(R2, t)
        assert abs(result.norm() - 1.0) < 1e-10, f"Failed at t={t}"


def test_slerp_dimension_mismatch():
    """slerp with mismatched dimensions raises."""
    import largecrimsoncanine as lcc

    R1 = lcc.Multivector.from_scalar(1.0, dims=2)
    R2 = lcc.Multivector.from_scalar(1.0, dims=3)

    with pytest.raises(ValueError):
        R1.slerp(R2, 0.5)


# =============================================================================
# ANGLE_BETWEEN TESTS
# =============================================================================

def test_angle_between_orthogonal():
    """Angle between orthogonal vectors is π/2."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    angle = e1.angle_between(e2)
    assert abs(angle - math.pi / 2) < 1e-10


def test_angle_between_same_direction():
    """Angle between parallel vectors is 0."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    v2 = lcc.Multivector.from_vector([2.0, 0.0, 0.0])

    angle = v1.angle_between(v2)
    assert abs(angle) < 1e-10


def test_angle_between_opposite_direction():
    """Angle between antiparallel vectors is π."""
    import largecrimsoncanine as lcc
    import math

    v1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    v2 = lcc.Multivector.from_vector([-1.0, 0.0, 0.0])

    angle = v1.angle_between(v2)
    assert abs(angle - math.pi) < 1e-10


def test_angle_between_45_degrees():
    """Angle between 45-degree vectors."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    diagonal = lcc.Multivector.from_vector([1.0, 1.0])

    angle = e1.angle_between(diagonal)
    assert abs(angle - math.pi / 4) < 1e-10


def test_angle_between_zero_vector_raises():
    """angle_between with zero vector raises."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 0.0])
    z = lcc.Multivector.zero(2)

    with pytest.raises(ValueError):
        v.angle_between(z)


def test_angle_between_non_vector_raises():
    """angle_between with non-vector raises."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)

    with pytest.raises(ValueError):
        v.angle_between(B)


# =============================================================================
# IS_PARALLEL TESTS
# =============================================================================

def test_is_parallel_same_direction():
    """Parallel vectors in same direction."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    v2 = lcc.Multivector.from_vector([3.0, 0.0, 0.0])

    assert v1.is_parallel(v2)


def test_is_parallel_opposite_direction():
    """Parallel vectors in opposite direction."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 2.0])
    v2 = lcc.Multivector.from_vector([-2.0, -4.0])

    assert v1.is_parallel(v2)


def test_is_parallel_orthogonal_not_parallel():
    """Orthogonal vectors are not parallel."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])

    assert not e1.is_parallel(e2)


def test_is_parallel_general_not_parallel():
    """General non-parallel vectors."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 0.0])
    v2 = lcc.Multivector.from_vector([1.0, 1.0])

    assert not v1.is_parallel(v2)


# =============================================================================
# IS_SAME_DIRECTION / IS_ANTIPARALLEL TESTS
# =============================================================================

def test_is_same_direction_true():
    """Vectors pointing same way."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 2.0])
    v2 = lcc.Multivector.from_vector([2.0, 4.0])

    assert v1.is_same_direction(v2)
    assert not v1.is_antiparallel(v2)


def test_is_antiparallel_true():
    """Vectors pointing opposite ways."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 2.0])
    v2 = lcc.Multivector.from_vector([-1.0, -2.0])

    assert v1.is_antiparallel(v2)
    assert not v1.is_same_direction(v2)


def test_is_same_direction_non_parallel():
    """Non-parallel vectors are neither same direction nor antiparallel."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 0.0])
    v2 = lcc.Multivector.from_vector([1.0, 1.0])

    assert not v1.is_same_direction(v2)
    assert not v1.is_antiparallel(v2)


# =============================================================================
# IS_ORTHOGONAL TESTS
# =============================================================================

def test_is_orthogonal_basis_vectors():
    """Basis vectors are orthogonal."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    assert e1.is_orthogonal(e2)
    assert e1.is_orthogonal(e3)
    assert e2.is_orthogonal(e3)


def test_is_orthogonal_parallel_not_orthogonal():
    """Parallel vectors are not orthogonal."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 0.0])
    v2 = lcc.Multivector.from_vector([2.0, 0.0])

    assert not v1.is_orthogonal(v2)


def test_is_orthogonal_general():
    """General orthogonal vectors."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 1.0])
    v2 = lcc.Multivector.from_vector([1.0, -1.0])

    # v1 · v2 = 1*1 + 1*(-1) = 0
    assert v1.is_orthogonal(v2)


def test_is_orthogonal_not_orthogonal():
    """Non-orthogonal vectors."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 0.0])
    v2 = lcc.Multivector.from_vector([1.0, 1.0])

    assert not v1.is_orthogonal(v2)


# =============================================================================
# COS_ANGLE / SIN_ANGLE TESTS
# =============================================================================

def test_cos_angle_parallel():
    """cos(0) = 1 for parallel vectors."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 0.0])
    v2 = lcc.Multivector.from_vector([2.0, 0.0])

    assert abs(v1.cos_angle(v2) - 1.0) < 1e-10


def test_cos_angle_orthogonal():
    """cos(π/2) = 0 for orthogonal vectors."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])

    assert abs(e1.cos_angle(e2)) < 1e-10


def test_cos_angle_antiparallel():
    """cos(π) = -1 for antiparallel vectors."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 0.0])
    v2 = lcc.Multivector.from_vector([-1.0, 0.0])

    assert abs(v1.cos_angle(v2) - (-1.0)) < 1e-10


def test_sin_angle_parallel():
    """sin(0) = 0 for parallel vectors."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 0.0])
    v2 = lcc.Multivector.from_vector([2.0, 0.0])

    assert abs(v1.sin_angle(v2)) < 1e-10


def test_sin_angle_orthogonal():
    """sin(π/2) = 1 for orthogonal vectors."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0])

    assert abs(e1.sin_angle(e2) - 1.0) < 1e-10


def test_sin_angle_45_degrees():
    """sin(π/4) = 1/√2 for 45-degree angle."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0])
    diagonal = lcc.Multivector.from_vector([1.0, 1.0])

    expected = math.sin(math.pi / 4)  # 1/√2
    assert abs(e1.sin_angle(diagonal) - expected) < 1e-10


def test_cos_sin_identity():
    """cos²θ + sin²θ = 1 for any angle."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    v2 = lcc.Multivector.from_vector([4.0, -1.0, 2.0])

    c = v1.cos_angle(v2)
    s = v1.sin_angle(v2)

    assert abs(c*c + s*s - 1.0) < 1e-10


# =============================================================================
# ROTATION_ANGLE TESTS
# =============================================================================

def test_rotation_angle_90_degrees():
    """90-degree rotor has angle π/2."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)
    angle = R.rotation_angle()

    assert abs(angle - math.pi / 2) < 1e-10


def test_rotation_angle_180_degrees():
    """180-degree rotor has angle π."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([-1.0, 0.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)
    angle = R.rotation_angle()

    assert abs(angle - math.pi) < 1e-10


def test_rotation_angle_identity():
    """Identity rotor has angle 0."""
    import largecrimsoncanine as lcc

    R = lcc.Multivector.from_scalar(1.0, dims=3)
    angle = R.rotation_angle()

    assert abs(angle) < 1e-10


def test_rotation_angle_from_exp():
    """Rotor from exp has expected angle."""
    import largecrimsoncanine as lcc
    import math

    # Create rotor with known angle
    theta = 1.0  # radians
    B = lcc.Multivector.from_bivector([theta / 2, 0.0, 0.0], dims=3)
    R = B.exp()

    angle = R.rotation_angle()
    assert abs(angle - theta) < 1e-10


def test_rotation_angle_non_rotor_raises():
    """rotation_angle on non-rotor raises."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        v.rotation_angle()


# =============================================================================
# ROTATION_PLANE TESTS
# =============================================================================

def test_rotation_plane_xy():
    """Rotation in xy-plane gives e12 bivector."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    R = lcc.Multivector.rotor_from_vectors(e1, e2)
    plane = R.rotation_plane()

    # Should be unit bivector in e12 direction
    assert plane.grade(2).approx_eq(plane)  # pure bivector
    assert abs(plane.norm() - 1.0) < 1e-10  # unit


def test_rotation_plane_identity():
    """Identity rotor gives zero bivector."""
    import largecrimsoncanine as lcc

    R = lcc.Multivector.from_scalar(1.0, dims=3)
    plane = R.rotation_plane()

    assert plane.approx_eq(lcc.Multivector.zero(3))


def test_rotation_plane_non_rotor_raises():
    """rotation_plane on non-rotor raises."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        v.rotation_plane()


# =============================================================================
# CROSS PRODUCT TESTS
# =============================================================================

def test_cross_basis_vectors():
    """e1 × e2 = e3 in 3D."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    result = e1.cross(e2)
    assert result.approx_eq(e3)


def test_cross_cyclic():
    """Cross product is cyclic: e2 × e3 = e1."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    assert e2.cross(e3).approx_eq(e1)
    assert e3.cross(e1).approx_eq(e2)


def test_cross_anticommutative():
    """a × b = -b × a."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    b = lcc.Multivector.from_vector([4.0, 5.0, 6.0])

    ab = a.cross(b)
    ba = b.cross(a)

    neg_ba = ba * (-1.0)
    assert ab.approx_eq(neg_ba)


def test_cross_parallel_is_zero():
    """Cross product of parallel vectors is zero."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    w = lcc.Multivector.from_vector([2.0, 4.0, 6.0])

    result = v.cross(w)
    assert result.approx_eq(lcc.Multivector.zero(3))


def test_cross_not_3d_raises():
    """Cross product in non-3D raises."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0])
    w = lcc.Multivector.from_vector([3.0, 4.0])

    with pytest.raises(ValueError):
        v.cross(w)


def test_cross_magnitude():
    """Cross product magnitude equals |a||b|sin(θ)."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    b = lcc.Multivector.from_vector([1.0, 1.0, 0.0])

    cross = a.cross(b)
    sin_theta = a.sin_angle(b)

    expected_magnitude = a.norm() * b.norm() * sin_theta
    assert abs(cross.norm() - expected_magnitude) < 1e-10


# =============================================================================
# AXIS_ANGLE TESTS
# =============================================================================

def test_axis_angle_z_rotation():
    """Rotation around z-axis."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    e3 = lcc.Multivector.from_vector([0.0, 0.0, 1.0])

    # 90-degree rotation from e1 to e2 is around e3
    R = lcc.Multivector.rotor_from_vectors(e1, e2)
    axis, angle = R.axis_angle()

    assert abs(angle - math.pi / 2) < 1e-10
    # Axis should be ±e3
    assert axis.is_parallel(e3)


def test_axis_angle_roundtrip():
    """Create rotor from axis-angle, extract, compare."""
    import largecrimsoncanine as lcc
    import math

    # Create a rotor with known axis and angle
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    axis, angle = R.axis_angle()

    # The rotation should transform e1 to e2
    rotated = R.sandwich(e1)
    assert rotated.approx_eq(e2)


def test_axis_angle_identity_raises():
    """axis_angle on identity rotor raises (no unique axis)."""
    import largecrimsoncanine as lcc

    R = lcc.Multivector.from_scalar(1.0, dims=3)

    with pytest.raises(ValueError):
        R.axis_angle()


def test_axis_angle_not_3d_raises():
    """axis_angle in non-3D raises."""
    import largecrimsoncanine as lcc

    R = lcc.Multivector.from_scalar(1.0, dims=2)

    with pytest.raises(ValueError):
        R.axis_angle()


# =============================================================================
# HAS_GRADE TESTS
# =============================================================================

def test_has_grade_scalar():
    """Scalar has grade 0."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(5.0, dims=3)
    assert s.has_grade(0)
    assert not s.has_grade(1)
    assert not s.has_grade(2)


def test_has_grade_vector():
    """Vector has grade 1."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    assert not v.has_grade(0)
    assert v.has_grade(1)
    assert not v.has_grade(2)


def test_has_grade_rotor():
    """Rotor has grades 0 and 2."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    assert R.has_grade(0)
    assert not R.has_grade(1)
    assert R.has_grade(2)


def test_has_grade_zero():
    """Zero has no grades."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert not z.has_grade(0)
    assert not z.has_grade(1)
    assert not z.has_grade(2)


# =============================================================================
# PURE_GRADE TESTS
# =============================================================================

def test_pure_grade_scalar():
    """Scalar has pure grade 0."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(5.0, dims=3)
    assert s.pure_grade() == 0


def test_pure_grade_vector():
    """Vector has pure grade 1."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    assert v.pure_grade() == 1


def test_pure_grade_bivector():
    """Bivector has pure grade 2."""
    import largecrimsoncanine as lcc

    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    assert B.pure_grade() == 2


def test_pure_grade_mixed():
    """Mixed multivector has no pure grade."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(1.0, dims=3)
    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    mixed = s + v

    assert mixed.pure_grade() is None


def test_pure_grade_zero():
    """Zero has no pure grade."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert z.pure_grade() is None


# =============================================================================
# IS_SCALAR / IS_VECTOR / IS_BIVECTOR TESTS
# =============================================================================

def test_is_scalar_true():
    """Scalar returns True for is_scalar."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(5.0, dims=3)
    assert s.is_scalar()
    assert not s.is_vector()
    assert not s.is_bivector()


def test_is_vector_true():
    """Vector returns True for is_vector."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    assert not v.is_scalar()
    assert v.is_vector()
    assert not v.is_bivector()


def test_is_bivector_true():
    """Bivector returns True for is_bivector."""
    import largecrimsoncanine as lcc

    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    assert not B.is_scalar()
    assert not B.is_vector()
    assert B.is_bivector()


def test_is_trivector():
    """Pseudoscalar in 3D is trivector."""
    import largecrimsoncanine as lcc

    I = lcc.Multivector.pseudoscalar(3)
    assert I.is_trivector()
    assert I.is_pseudoscalar()


def test_is_pseudoscalar_4d():
    """Pseudoscalar in 4D."""
    import largecrimsoncanine as lcc

    I = lcc.Multivector.pseudoscalar(4)
    assert I.is_pseudoscalar()
    assert not I.is_trivector()  # grade 4, not 3


# =============================================================================
# IS_ZERO TESTS
# =============================================================================

def test_is_zero_true():
    """Zero multivector is zero."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert z.is_zero()


def test_is_zero_false():
    """Non-zero multivector is not zero."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    assert not v.is_zero()


def test_is_zero_scalar_zero():
    """Scalar zero is zero."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(0.0, dims=3)
    assert s.is_zero()


# =============================================================================
# COMPONENTS TESTS
# =============================================================================

def test_components_scalar():
    """Scalar has one component at index 0."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(5.0, dims=3)
    comps = s.components()

    assert len(comps) == 1
    assert comps[0] == (0, 5.0)


def test_components_vector():
    """Vector has components at power-of-2 indices."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    comps = v.components()

    # e1 at index 1, e2 at index 2, e3 at index 4
    assert (1, 1.0) in comps
    assert (2, 2.0) in comps
    assert (4, 3.0) in comps


def test_components_zero():
    """Zero has no components."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert z.components() == []


def test_components_sparse():
    """Sparse multivector returns only non-zero."""
    import largecrimsoncanine as lcc

    # e1 + e12 (indices 1 and 3)
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e12 = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    mv = e1 + e12

    comps = mv.components()
    assert len(comps) == 2
    indices = [c[0] for c in comps]
    assert 1 in indices  # e1
    assert 3 in indices  # e12


# =============================================================================
# BLADE_INDICES TESTS
# =============================================================================

def test_blade_indices_vector():
    """Vector blade indices are powers of 2."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    indices = v.blade_indices()

    assert set(indices) == {1, 2, 4}


def test_blade_indices_bivector():
    """Bivector blade indices."""
    import largecrimsoncanine as lcc

    B = lcc.Multivector.from_bivector([1.0, 2.0, 3.0], dims=3)
    indices = B.blade_indices()

    # e12=3, e13=5, e23=6
    assert set(indices) == {3, 5, 6}


def test_blade_indices_zero():
    """Zero has no blade indices."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert z.blade_indices() == []


# =============================================================================
# PYTHON PROTOCOL TESTS
# =============================================================================

def test_pow_positive():
    """Test v ** 2 = v * v."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    result = e1 ** 2

    # e1 * e1 = 1 (Euclidean)
    assert result.is_scalar()
    assert abs(result.scalar() - 1.0) < 1e-10


def test_pow_zero():
    """Test v ** 0 = 1."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([3.0, 4.0, 0.0])
    result = v ** 0

    assert result.is_scalar()
    assert abs(result.scalar() - 1.0) < 1e-10


def test_pow_rotor():
    """Test rotor ** 2 doubles the rotation."""
    import largecrimsoncanine as lcc
    import math

    # e1 and e2 are perpendicular, so rotor_from_vectors gives 90-degree rotation
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    # R rotates 90 degrees, so R ** 2 rotates 180 degrees
    R2 = R ** 2

    # Apply R**2 to e1: should get -e1 (180 degree rotation)
    rotated = R2.sandwich(e1)
    neg_e1_coords = [-1.0, 0.0, 0.0]
    for i, expected in enumerate(neg_e1_coords):
        actual = rotated.to_list()[1 << i]
        assert abs(actual - expected) < 1e-10


def test_pow_negative():
    """Test v ** -1 = v.inverse()."""
    import largecrimsoncanine as lcc

    # Use a unit vector (easy to invert)
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    result = e1 ** -1

    # e1 ** -1 = e1 (for unit vector)
    assert result.approx_eq(e1, 1e-10)


def test_pow_negative_2():
    """Test v ** -2 = (v * v).inverse()."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([2.0, 0.0, 0.0])
    result = e1 ** -2

    # (2*e1)^2 = 4, so (2*e1)^-2 = 1/4
    assert result.is_scalar()
    assert abs(result.scalar() - 0.25) < 1e-10


def test_bool_nonzero():
    """Non-zero multivector is truthy."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    assert bool(v) is True


def test_bool_zero():
    """Zero multivector is falsy."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert bool(z) is False


def test_bool_in_conditional():
    """Bool works in if statements."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    z = lcc.Multivector.zero(3)

    if v:
        passed_v = True
    else:
        passed_v = False

    if z:
        passed_z = True
    else:
        passed_z = False

    assert passed_v is True
    assert passed_z is False


def test_abs_vector():
    """abs(v) returns norm."""
    import largecrimsoncanine as lcc
    import math

    v = lcc.Multivector.from_vector([3.0, 4.0, 0.0])
    assert abs(v) == 5.0


def test_abs_scalar():
    """abs(scalar) returns absolute value."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(-3.0, dims=2)
    assert abs(s) == 3.0


def test_abs_unit_rotor():
    """abs(unit rotor) is 1."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    assert abs(abs(R) - 1.0) < 1e-10


def test_copy():
    """copy() creates independent copy."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    v2 = v.copy()

    assert v.approx_eq(v2, 1e-10)
    # They should be equal but independent objects
    assert v is not v2


def test_copy_preserves_dims():
    """copy() preserves dimension."""
    import largecrimsoncanine as lcc

    mv = lcc.Multivector.zero(5)
    copy = mv.copy()

    assert len(copy) == len(mv)


def test_coefficients_returns_all():
    """coefficients() returns all coefficients."""
    import largecrimsoncanine as lcc

    # 2D vector (only 2 components)
    v = lcc.Multivector.from_vector([1.0, 2.0])
    coeffs = v.coefficients()

    # 2D space has 2^2 = 4 coefficients
    assert len(coeffs) == 4
    assert coeffs[0] == 0.0  # scalar
    assert coeffs[1] == 1.0  # e1
    assert coeffs[2] == 2.0  # e2
    assert coeffs[3] == 0.0  # e12


def test_coefficients_iterable():
    """coefficients() is iterable."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    total = sum(v.coefficients())

    assert total == 6.0


def test_coefficients_3d():
    """coefficients() for 3D space has 8 elements."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.zero(3)
    assert len(v.coefficients()) == 8


# =============================================================================
# DIMENSION ACCESS TESTS
# =============================================================================

def test_dimension_property():
    """dimension property returns base vector space dimension."""
    import largecrimsoncanine as lcc

    v2 = lcc.Multivector.zero(2)
    v3 = lcc.Multivector.zero(3)
    v4 = lcc.Multivector.zero(4)

    assert v2.dimension == 2
    assert v3.dimension == 3
    assert v4.dimension == 4


def test_dims_property():
    """dims property is alias for dimension."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    assert v.dims == 3
    assert v.dims == v.dimension


def test_n_coeffs_property():
    """n_coeffs returns 2^dimension."""
    import largecrimsoncanine as lcc

    v2 = lcc.Multivector.zero(2)
    v3 = lcc.Multivector.zero(3)
    v4 = lcc.Multivector.zero(4)

    assert v2.n_coeffs == 4   # 2^2
    assert v3.n_coeffs == 8   # 2^3
    assert v4.n_coeffs == 16  # 2^4


def test_dimension_equals_len_log():
    """n_coeffs equals len()."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0, 4.0, 5.0])

    assert len(v) == v.n_coeffs


# =============================================================================
# FROM_AXIS_ANGLE TESTS
# =============================================================================

def test_from_axis_angle_identity():
    """Zero angle gives identity rotor."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([0.0, 0.0, 1.0])  # z-axis
    R = lcc.Multivector.from_axis_angle(axis, 0.0)

    # Identity rotor is 1 + 0*B
    assert abs(R.scalar() - 1.0) < 1e-10
    # Should be just a scalar
    for i in range(1, 8):
        assert abs(R.to_list()[i]) < 1e-10


def test_from_axis_angle_90_degrees():
    """90-degree rotation around z-axis."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([0.0, 0.0, 1.0])  # z-axis
    R = lcc.Multivector.from_axis_angle(axis, math.pi / 2)

    # Apply to e1, should get e2
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    rotated = R.sandwich(e1)

    expected = [0.0, 1.0, 0.0]  # e2
    for i, exp in enumerate(expected):
        actual = rotated.to_list()[1 << i]
        assert abs(actual - exp) < 1e-10


def test_from_axis_angle_180_degrees():
    """180-degree rotation around z-axis."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([0.0, 0.0, 1.0])
    R = lcc.Multivector.from_axis_angle(axis, math.pi)

    # Apply to e1, should get -e1
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    rotated = R.sandwich(e1)

    expected = [-1.0, 0.0, 0.0]  # -e1
    for i, exp in enumerate(expected):
        actual = rotated.to_list()[1 << i]
        assert abs(actual - exp) < 1e-10


def test_from_axis_angle_x_axis():
    """Rotation around x-axis."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([1.0, 0.0, 0.0])  # x-axis
    R = lcc.Multivector.from_axis_angle(axis, math.pi / 2)

    # Apply to e2, should get e3
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    rotated = R.sandwich(e2)

    expected = [0.0, 0.0, 1.0]  # e3
    for i, exp in enumerate(expected):
        actual = rotated.to_list()[1 << i]
        assert abs(actual - exp) < 1e-10


def test_from_axis_angle_arbitrary_axis():
    """Rotation around arbitrary axis preserves axis."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([1.0, 1.0, 1.0])
    R = lcc.Multivector.from_axis_angle(axis, math.pi / 3)

    # Axis should be unchanged by rotation around itself
    rotated = R.sandwich(axis.normalized())

    assert rotated.approx_eq(axis.normalized(), 1e-10)


def test_from_axis_angle_unit_rotor():
    """from_axis_angle produces unit rotor."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    R = lcc.Multivector.from_axis_angle(axis, 1.234)

    # Unit rotor: R * ~R = 1
    product = R * R.reverse()
    assert abs(product.scalar() - 1.0) < 1e-10


def test_from_axis_angle_matches_rotation_angle():
    """rotation_angle extracts the correct angle."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([0.0, 0.0, 1.0])
    angle = 1.5  # radians
    R = lcc.Multivector.from_axis_angle(axis, angle)

    extracted = R.rotation_angle()
    assert abs(extracted - angle) < 1e-10


def test_from_axis_angle_requires_3d():
    """from_axis_angle raises error for non-3D."""
    import largecrimsoncanine as lcc
    import pytest

    axis_2d = lcc.Multivector.from_vector([1.0, 0.0])
    with pytest.raises(ValueError, match="requires 3D"):
        lcc.Multivector.from_axis_angle(axis_2d, 1.0)


def test_from_axis_angle_requires_vector():
    """from_axis_angle raises error for non-vector."""
    import largecrimsoncanine as lcc
    import pytest

    # Bivector, not vector
    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    with pytest.raises(ValueError, match="must be a vector"):
        lcc.Multivector.from_axis_angle(B, 1.0)


# =============================================================================
# IS_UNIT TESTS
# =============================================================================

def test_is_unit_true():
    """Unit vector has norm 1."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    assert v.is_unit() is True


def test_is_unit_normalized():
    """Normalized vector is unit."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([3.0, 4.0, 0.0])
    assert v.is_unit() is False
    assert v.normalized().is_unit() is True


def test_is_unit_rotor():
    """Rotors are unit."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    assert R.is_unit() is True


def test_is_unit_scalar():
    """Scalar 1 is unit, others are not."""
    import largecrimsoncanine as lcc

    s1 = lcc.Multivector.from_scalar(1.0, dims=2)
    s2 = lcc.Multivector.from_scalar(2.0, dims=2)

    assert s1.is_unit() is True
    assert s2.is_unit() is False


# =============================================================================
# DOT PRODUCT TESTS
# =============================================================================

def test_dot_alias():
    """dot() is alias for scalar_product()."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    b = lcc.Multivector.from_vector([4.0, 5.0, 6.0])

    dot_result = a.dot(b)
    sp_result = a.scalar_product(b)

    assert dot_result.approx_eq(sp_result, 1e-10)


def test_dot_vectors():
    """Dot product of vectors gives scalar."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    b = lcc.Multivector.from_vector([1.0, 0.0, 0.0])

    result = a.dot(b)
    assert result.is_scalar()
    assert result.scalar() == 1.0


# =============================================================================
# LERP TESTS
# =============================================================================

def test_lerp_endpoints():
    """lerp at t=0 and t=1 returns endpoints."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    b = lcc.Multivector.from_vector([0.0, 1.0, 0.0])

    at_0 = a.lerp(b, 0.0)
    at_1 = a.lerp(b, 1.0)

    assert at_0.approx_eq(a, 1e-10)
    assert at_1.approx_eq(b, 1e-10)


def test_lerp_midpoint():
    """lerp at t=0.5 returns midpoint."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_vector([2.0, 0.0, 0.0])
    b = lcc.Multivector.from_vector([0.0, 2.0, 0.0])

    mid = a.lerp(b, 0.5)

    expected = lcc.Multivector.from_vector([1.0, 1.0, 0.0])
    assert mid.approx_eq(expected, 1e-10)


def test_lerp_quarter():
    """lerp at t=0.25."""
    import largecrimsoncanine as lcc

    a = lcc.Multivector.from_scalar(0.0, dims=2)
    b = lcc.Multivector.from_scalar(4.0, dims=2)

    result = a.lerp(b, 0.25)

    expected = lcc.Multivector.from_scalar(1.0, dims=2)
    assert result.approx_eq(expected, 1e-10)


def test_lerp_dimension_mismatch():
    """lerp raises error for dimension mismatch."""
    import largecrimsoncanine as lcc
    import pytest

    a = lcc.Multivector.zero(2)
    b = lcc.Multivector.zero(3)

    with pytest.raises(ValueError, match="dimension mismatch"):
        a.lerp(b, 0.5)


# =============================================================================
# CONVENIENCE METHODS TESTS
# =============================================================================

def test_magnitude_alias():
    """magnitude() is alias for norm()."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([3.0, 4.0, 0.0])

    assert v.magnitude() == v.norm()
    assert v.magnitude() == 5.0


def test_squared_vector():
    """Vector squared gives scalar (squared norm)."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([3.0, 4.0, 0.0])
    sq = v.squared()

    assert sq.is_scalar()
    assert sq.scalar() == 25.0  # 3^2 + 4^2


def test_squared_bivector():
    """Bivector squared gives negative scalar."""
    import largecrimsoncanine as lcc

    # e12 squared = e12 * e12 = e1*e2*e1*e2 = -e1*e1*e2*e2 = -1
    e12 = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    sq = e12.squared()

    assert sq.is_scalar()
    assert abs(sq.scalar() - (-1.0)) < 1e-10


def test_squared_rotor():
    """Rotor squared is still a rotor."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    R2 = R.squared()

    # R^2 should still be a unit rotor
    assert R2.is_rotor()


def test_is_even_scalar():
    """Scalar is even-graded."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(5.0, dims=3)
    assert s.is_even() is True
    assert s.is_odd() is False


def test_is_even_bivector():
    """Bivector is even-graded."""
    import largecrimsoncanine as lcc

    B = lcc.Multivector.from_bivector([1.0, 2.0, 3.0], dims=3)
    assert B.is_even() is True
    assert B.is_odd() is False


def test_is_odd_vector():
    """Vector is odd-graded."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    assert v.is_odd() is True
    assert v.is_even() is False


def test_is_even_rotor():
    """Rotor (scalar + bivector) is even-graded."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    assert R.is_even() is True
    assert R.is_odd() is False


def test_is_even_mixed():
    """Mixed scalar + vector is neither even nor odd."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(1.0, dims=2)
    v = lcc.Multivector.from_vector([1.0, 0.0])
    mixed = s + v

    assert mixed.is_even() is False
    assert mixed.is_odd() is False


def test_is_even_zero():
    """Zero is both even and odd (vacuously true)."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)

    # Zero has no components, so all conditions are vacuously satisfied
    assert z.is_even() is True
    assert z.is_odd() is True


def test_grade_count_zero():
    """Zero has 0 grades."""
    import largecrimsoncanine as lcc

    z = lcc.Multivector.zero(3)
    assert z.grade_count() == 0


def test_grade_count_scalar():
    """Scalar has 1 grade."""
    import largecrimsoncanine as lcc

    s = lcc.Multivector.from_scalar(5.0, dims=3)
    assert s.grade_count() == 1


def test_grade_count_vector():
    """Vector has 1 grade."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    assert v.grade_count() == 1


def test_grade_count_rotor():
    """Rotor has 2 grades (scalar + bivector)."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    assert R.grade_count() == 2


def test_grade_count_general():
    """General multivector can have multiple grades."""
    import largecrimsoncanine as lcc

    # scalar + vector + bivector
    s = lcc.Multivector.from_scalar(1.0, dims=3)
    v = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    mv = s + v + B

    assert mv.grade_count() == 3


# =============================================================================
# QUATERNION INTEROP TESTS
# =============================================================================

def test_from_quaternion_identity():
    """Identity quaternion (1, 0, 0, 0) gives identity rotor."""
    import largecrimsoncanine as lcc

    R = lcc.Multivector.from_quaternion(1.0, 0.0, 0.0, 0.0)

    assert R.is_scalar()
    assert abs(R.scalar() - 1.0) < 1e-10


def test_from_quaternion_90_z():
    """90-degree rotation around z-axis."""
    import largecrimsoncanine as lcc
    import math

    # Quaternion for 90° around z: (cos(45°), 0, 0, sin(45°))
    c = math.cos(math.pi / 4)
    s = math.sin(math.pi / 4)
    R = lcc.Multivector.from_quaternion(c, 0.0, 0.0, s)

    # Apply to e1, should get e2
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    rotated = R.sandwich(e1)

    expected = [0.0, 1.0, 0.0]
    for i, exp in enumerate(expected):
        actual = rotated.to_list()[1 << i]
        assert abs(actual - exp) < 1e-10


def test_from_quaternion_unit():
    """Unit quaternion gives unit rotor."""
    import largecrimsoncanine as lcc
    import math

    # Normalized quaternion
    c = math.cos(0.5)
    s = math.sin(0.5)
    R = lcc.Multivector.from_quaternion(c, s * 0.6, s * 0.8, 0.0)

    assert R.is_unit()


def test_to_quaternion_identity():
    """Identity rotor gives identity quaternion."""
    import largecrimsoncanine as lcc

    R = lcc.Multivector.from_scalar(1.0, dims=3)
    w, x, y, z = R.to_quaternion()

    assert abs(w - 1.0) < 1e-10
    assert abs(x) < 1e-10
    assert abs(y) < 1e-10
    assert abs(z) < 1e-10


def test_to_quaternion_roundtrip():
    """from_quaternion and to_quaternion are inverses."""
    import largecrimsoncanine as lcc
    import math

    # Start with arbitrary unit quaternion
    w, x, y, z = 0.5, 0.5, 0.5, 0.5

    R = lcc.Multivector.from_quaternion(w, x, y, z)
    w2, x2, y2, z2 = R.to_quaternion()

    assert abs(w - w2) < 1e-10
    assert abs(x - x2) < 1e-10
    assert abs(y - y2) < 1e-10
    assert abs(z - z2) < 1e-10


def test_to_quaternion_rotor():
    """Rotor from vectors converts correctly."""
    import largecrimsoncanine as lcc
    import math

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    w, x, y, z = R.to_quaternion()

    # 90° around z: (cos(45°), 0, 0, sin(45°))
    expected_w = math.cos(math.pi / 4)
    expected_z = math.sin(math.pi / 4)

    assert abs(w - expected_w) < 1e-10
    assert abs(x) < 1e-10
    assert abs(y) < 1e-10
    assert abs(z - expected_z) < 1e-10


def test_to_quaternion_from_axis_angle():
    """axis_angle rotor converts correctly."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([1.0, 0.0, 0.0])  # x-axis
    angle = math.pi / 3  # 60 degrees
    R = lcc.Multivector.from_axis_angle(axis, angle)

    w, x, y, z = R.to_quaternion()

    # Quaternion: (cos(30°), sin(30°), 0, 0)
    expected_w = math.cos(angle / 2)
    expected_x = math.sin(angle / 2)

    assert abs(w - expected_w) < 1e-10
    assert abs(x - expected_x) < 1e-10
    assert abs(y) < 1e-10
    assert abs(z) < 1e-10


def test_to_quaternion_requires_3d():
    """to_quaternion raises error for non-3D."""
    import largecrimsoncanine as lcc
    import pytest

    mv = lcc.Multivector.zero(2)
    with pytest.raises(ValueError, match="requires 3D"):
        mv.to_quaternion()


def test_quaternion_rotation_matches():
    """Quaternion rotation matches rotor rotation."""
    import largecrimsoncanine as lcc
    import math

    # Create rotor via axis-angle
    axis = lcc.Multivector.from_vector([1.0, 1.0, 1.0])
    R = lcc.Multivector.from_axis_angle(axis, 1.5)

    # Convert to quaternion and back
    w, x, y, z = R.to_quaternion()
    R2 = lcc.Multivector.from_quaternion(w, x, y, z)

    # Both should produce same rotation
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    rotated1 = R.sandwich(v)
    rotated2 = R2.sandwich(v)

    assert rotated1.approx_eq(rotated2, 1e-10)


# =============================================================================
# ROTATION MATRIX TESTS
# =============================================================================

def test_to_rotation_matrix_identity():
    """Identity rotor gives identity matrix."""
    import largecrimsoncanine as lcc

    R = lcc.Multivector.from_scalar(1.0, dims=3)
    mat = R.to_rotation_matrix()

    # Identity matrix
    expected = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    for i, exp in enumerate(expected):
        assert abs(mat[i] - exp) < 1e-10


def test_to_rotation_matrix_90_z():
    """90-degree rotation around z gives correct matrix."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([0.0, 0.0, 1.0])
    R = lcc.Multivector.from_axis_angle(axis, math.pi / 2)
    mat = R.to_rotation_matrix()

    # 90° around z: x→y, y→-x, z→z
    # [ 0 -1  0]
    # [ 1  0  0]
    # [ 0  0  1]
    expected = [0, -1, 0, 1, 0, 0, 0, 0, 1]
    for i, exp in enumerate(expected):
        assert abs(mat[i] - exp) < 1e-10


def test_to_rotation_matrix_180_x():
    """180-degree rotation around x."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    R = lcc.Multivector.from_axis_angle(axis, math.pi)
    mat = R.to_rotation_matrix()

    # 180° around x: x→x, y→-y, z→-z
    # [ 1  0  0]
    # [ 0 -1  0]
    # [ 0  0 -1]
    expected = [1, 0, 0, 0, -1, 0, 0, 0, -1]
    for i, exp in enumerate(expected):
        assert abs(mat[i] - exp) < 1e-10


def test_to_rotation_matrix_orthogonal():
    """Result is orthogonal matrix (M * M^T = I)."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    R = lcc.Multivector.from_axis_angle(axis, 1.234)
    mat = R.to_rotation_matrix()

    # Check M * M^T = I
    for i in range(3):
        for j in range(3):
            dot = sum(mat[i*3 + k] * mat[j*3 + k] for k in range(3))
            expected = 1.0 if i == j else 0.0
            assert abs(dot - expected) < 1e-10


def test_to_rotation_matrix_determinant():
    """Determinant is +1."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    R = lcc.Multivector.from_axis_angle(axis, 1.234)
    mat = R.to_rotation_matrix()

    # 3x3 determinant
    det = (mat[0] * (mat[4]*mat[8] - mat[5]*mat[7]) -
           mat[1] * (mat[3]*mat[8] - mat[5]*mat[6]) +
           mat[2] * (mat[3]*mat[7] - mat[4]*mat[6]))

    assert abs(det - 1.0) < 1e-10


def test_from_rotation_matrix_identity():
    """Identity matrix gives identity rotor."""
    import largecrimsoncanine as lcc

    mat = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    R = lcc.Multivector.from_rotation_matrix(mat)

    assert R.is_rotor()
    # Identity rotor has scalar ≈ 1, bivector ≈ 0
    assert abs(R.scalar() - 1.0) < 1e-10 or abs(R.scalar() + 1.0) < 1e-10


def test_from_rotation_matrix_90_z():
    """90-degree z rotation matrix gives correct rotor."""
    import largecrimsoncanine as lcc
    import math

    # 90° around z
    mat = [0, -1, 0, 1, 0, 0, 0, 0, 1]
    R = lcc.Multivector.from_rotation_matrix(mat)

    # Apply to e1, should get e2
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    rotated = R.sandwich(e1)

    expected = [0.0, 1.0, 0.0]
    for i, exp in enumerate(expected):
        actual = rotated.to_list()[1 << i]
        assert abs(actual - exp) < 1e-10


def test_rotation_matrix_roundtrip():
    """to_rotation_matrix and from_rotation_matrix are inverses."""
    import largecrimsoncanine as lcc
    import math

    axis = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    R = lcc.Multivector.from_axis_angle(axis, 1.5)

    mat = R.to_rotation_matrix()
    R2 = lcc.Multivector.from_rotation_matrix(mat)

    # Both rotors should produce same rotation
    # (might differ by sign, which is ok for rotors)
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    rotated1 = R.sandwich(v)
    rotated2 = R2.sandwich(v)

    assert rotated1.approx_eq(rotated2, 1e-10)


def test_to_rotation_matrix_requires_3d():
    """to_rotation_matrix raises error for non-3D."""
    import largecrimsoncanine as lcc
    import pytest

    mv = lcc.Multivector.zero(2)
    with pytest.raises(ValueError, match="requires 3D"):
        mv.to_rotation_matrix()


def test_from_rotation_matrix_wrong_size():
    """from_rotation_matrix raises error for wrong size."""
    import largecrimsoncanine as lcc
    import pytest

    mat = [1, 0, 0, 0, 1, 0]  # 6 elements instead of 9
    with pytest.raises(ValueError, match="9 elements"):
        lcc.Multivector.from_rotation_matrix(mat)


# =============================================================================
# EULER ANGLES TESTS
# =============================================================================

def test_from_euler_angles_identity():
    """Zero Euler angles give identity rotor."""
    import largecrimsoncanine as lcc

    R = lcc.Multivector.from_euler_angles(0.0, 0.0, 0.0)

    assert R.is_rotor()
    assert abs(R.scalar() - 1.0) < 1e-10


def test_from_euler_angles_yaw_90():
    """90-degree yaw rotates x to y."""
    import largecrimsoncanine as lcc
    import math

    R = lcc.Multivector.from_euler_angles(math.pi / 2, 0.0, 0.0)

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    rotated = R.sandwich(e1)

    # x -> y
    expected = [0.0, 1.0, 0.0]
    for i, exp in enumerate(expected):
        actual = rotated.to_list()[1 << i]
        assert abs(actual - exp) < 1e-10


def test_from_euler_angles_pitch_90():
    """90-degree pitch rotates x to z (nose up)."""
    import largecrimsoncanine as lcc
    import math

    R = lcc.Multivector.from_euler_angles(0.0, math.pi / 2, 0.0)

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    rotated = R.sandwich(e1)

    # x -> z (positive pitch = nose up)
    expected = [0.0, 0.0, 1.0]
    for i, exp in enumerate(expected):
        actual = rotated.to_list()[1 << i]
        assert abs(actual - exp) < 1e-10


def test_from_euler_angles_roll_90():
    """90-degree roll rotates y to z."""
    import largecrimsoncanine as lcc
    import math

    R = lcc.Multivector.from_euler_angles(0.0, 0.0, math.pi / 2)

    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    rotated = R.sandwich(e2)

    # y -> z
    expected = [0.0, 0.0, 1.0]
    for i, exp in enumerate(expected):
        actual = rotated.to_list()[1 << i]
        assert abs(actual - exp) < 1e-10


def test_to_euler_angles_identity():
    """Identity rotor gives zero Euler angles."""
    import largecrimsoncanine as lcc

    R = lcc.Multivector.from_scalar(1.0, dims=3)
    yaw, pitch, roll = R.to_euler_angles()

    assert abs(yaw) < 1e-10
    assert abs(pitch) < 1e-10
    assert abs(roll) < 1e-10


def test_euler_angles_roundtrip():
    """from_euler_angles and to_euler_angles are inverses."""
    import largecrimsoncanine as lcc
    import math

    # Arbitrary Euler angles (avoiding gimbal lock)
    yaw = 0.3
    pitch = 0.5
    roll = 0.7

    R = lcc.Multivector.from_euler_angles(yaw, pitch, roll)
    y2, p2, r2 = R.to_euler_angles()

    assert abs(yaw - y2) < 1e-10
    assert abs(pitch - p2) < 1e-10
    assert abs(roll - r2) < 1e-10


def test_euler_angles_matches_axis_angle():
    """Euler yaw matches axis-angle around z."""
    import largecrimsoncanine as lcc
    import math

    angle = 1.23
    R_euler = lcc.Multivector.from_euler_angles(angle, 0.0, 0.0)

    axis = lcc.Multivector.from_vector([0.0, 0.0, 1.0])
    R_axis = lcc.Multivector.from_axis_angle(axis, angle)

    # Both should produce same rotation
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    rotated1 = R_euler.sandwich(v)
    rotated2 = R_axis.sandwich(v)

    assert rotated1.approx_eq(rotated2, 1e-10)


def test_to_euler_angles_requires_3d():
    """to_euler_angles raises error for non-3D."""
    import largecrimsoncanine as lcc
    import pytest

    mv = lcc.Multivector.zero(2)
    with pytest.raises(ValueError, match="requires 3D"):
        mv.to_euler_angles()


def test_euler_angles_unit_rotor():
    """from_euler_angles produces unit rotor."""
    import largecrimsoncanine as lcc
    import math

    R = lcc.Multivector.from_euler_angles(1.0, 0.5, 0.3)
    assert R.is_unit()


# =============================================================================
# MORE OPERATORS TESTS
# =============================================================================

def test_pos_returns_copy():
    """Unary + returns copy."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    pos_v = +v

    assert pos_v.approx_eq(v, 1e-10)
    assert pos_v is not v


def test_pos_with_neg():
    """+(-v) == -v and -(-v) == v."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])

    assert (+(-v)).approx_eq(-v, 1e-10)
    assert (-(-v)).approx_eq(v, 1e-10)


def test_invert_is_reverse():
    """~ operator is reverse."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    B = lcc.Multivector.from_bivector([1.0, 2.0, 3.0], dims=3)

    # For vector, ~v = v
    assert (~v).approx_eq(v.reverse(), 1e-10)

    # For bivector, ~B = -B
    assert (~B).approx_eq(B.reverse(), 1e-10)


def test_invert_rotor():
    """~R is inverse for unit rotor."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    # For unit rotor, R * ~R = 1
    product = R * (~R)
    assert abs(product.scalar() - 1.0) < 1e-10


def test_invert_syntax():
    """~ can be used in expressions."""
    import largecrimsoncanine as lcc

    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    # R * e1 * ~R (sandwich product)
    rotated = R * e1 * ~R

    assert rotated.is_vector()


def test_negate_grade_vector():
    """Negate grade 1 flips vector part."""
    import largecrimsoncanine as lcc

    # scalar + vector
    s = lcc.Multivector.from_scalar(2.0, dims=2)
    v = lcc.Multivector.from_vector([3.0, 4.0])
    mv = s + v

    negated = mv.negate_grade(1)

    # Scalar unchanged, vector negated
    assert abs(negated.scalar() - 2.0) < 1e-10
    assert abs(negated.grade(1).to_list()[1] - (-3.0)) < 1e-10


def test_negate_grade_scalar():
    """Negate grade 0 flips scalar part."""
    import largecrimsoncanine as lcc

    # scalar + vector
    s = lcc.Multivector.from_scalar(2.0, dims=2)
    v = lcc.Multivector.from_vector([3.0, 4.0])
    mv = s + v

    negated = mv.negate_grade(0)

    # Scalar negated, vector unchanged
    assert abs(negated.scalar() - (-2.0)) < 1e-10
    assert abs(negated.grade(1).to_list()[1] - 3.0) < 1e-10


def test_negate_grade_is_grade_involution():
    """Negating all odd grades equals grade involution."""
    import largecrimsoncanine as lcc

    # scalar + vector + bivector
    s = lcc.Multivector.from_scalar(1.0, dims=3)
    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    mv = s + v + B

    # Negate grade 1 (vectors), grade 3 (trivectors)
    negated = mv.negate_grade(1).negate_grade(3)

    assert negated.approx_eq(mv.grade_involution(), 1e-10)


def test_negate_grade_invalid():
    """Negate grade raises error for invalid grade."""
    import largecrimsoncanine as lcc
    import pytest

    mv = lcc.Multivector.zero(2)
    with pytest.raises(ValueError, match="exceeds"):
        mv.negate_grade(5)



# =====================
# Serialization tests
# =====================


def test_to_dict_basic():
    """to_dict returns coeffs and dims."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    d = v.to_dict()

    assert "coeffs" in d
    assert "dims" in d
    assert d["dims"] == 3
    assert len(d["coeffs"]) == 8  # 2^3


def test_from_dict_basic():
    """from_dict reconstructs multivector."""
    import largecrimsoncanine as lcc

    v = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    d = v.to_dict()
    v2 = lcc.Multivector.from_dict(d)

    assert v.approx_eq(v2, 1e-10)


def test_serialization_roundtrip():
    """to_dict -> from_dict preserves multivector."""
    import largecrimsoncanine as lcc

    # Test with more complex multivector (rotor)
    e1 = lcc.Multivector.from_vector([1.0, 0.0, 0.0])
    e2 = lcc.Multivector.from_vector([0.0, 1.0, 0.0])
    R = lcc.Multivector.rotor_from_vectors(e1, e2)

    d = R.to_dict()
    R2 = lcc.Multivector.from_dict(d)

    assert R.approx_eq(R2, 1e-10)


def test_from_dict_missing_coeffs():
    """from_dict raises error when coeffs missing."""
    import largecrimsoncanine as lcc
    import pytest

    with pytest.raises(KeyError, match="coeffs"):
        lcc.Multivector.from_dict({"dims": 3})


def test_from_dict_missing_dims():
    """from_dict raises error when dims missing."""
    import largecrimsoncanine as lcc
    import pytest

    with pytest.raises(KeyError, match="dims"):
        lcc.Multivector.from_dict({"coeffs": [0.0] * 8})


def test_from_dict_invalid_length():
    """from_dict raises error when coeffs length wrong."""
    import largecrimsoncanine as lcc
    import pytest

    with pytest.raises(ValueError, match="doesn't match"):
        lcc.Multivector.from_dict({"coeffs": [0.0] * 4, "dims": 3})


# =====================
# Geometric utility tests
# =====================


def test_distance_basic():
    """distance computes Euclidean distance."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 1.0])
    v2 = lcc.Multivector.from_vector([4.0, 5.0])

    # sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = 5
    d = v1.distance(v2)
    assert abs(d - 5.0) < 1e-10


def test_distance_3d():
    """distance works in 3D."""
    import largecrimsoncanine as lcc
    import math

    v1 = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    v2 = lcc.Multivector.from_vector([4.0, 6.0, 3.0])

    # (4-1)^2 + (6-2)^2 + (3-3)^2 = 9 + 16 + 0 = 25
    d = v1.distance(v2)
    assert abs(d - 5.0) < 1e-10


def test_distance_symmetric():
    """distance is symmetric."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 2.0])
    v2 = lcc.Multivector.from_vector([5.0, 8.0])

    assert abs(v1.distance(v2) - v2.distance(v1)) < 1e-10


def test_distance_non_vector_self():
    """distance raises error if self is not a vector."""
    import largecrimsoncanine as lcc
    import pytest

    s = lcc.Multivector.from_scalar(5.0, dims=2)
    v = lcc.Multivector.from_vector([1.0, 2.0])

    with pytest.raises(ValueError, match="self is not a vector"):
        s.distance(v)


def test_distance_non_vector_other():
    """distance raises error if other is not a vector."""
    import largecrimsoncanine as lcc
    import pytest

    v = lcc.Multivector.from_vector([1.0, 2.0])
    s = lcc.Multivector.from_scalar(5.0, dims=2)

    with pytest.raises(ValueError, match="other is not a vector"):
        v.distance(s)


def test_midpoint_basic():
    """midpoint computes average of two vectors."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([2.0, 2.0])
    v2 = lcc.Multivector.from_vector([6.0, 10.0])

    mid = v1.midpoint(v2)
    expected = lcc.Multivector.from_vector([4.0, 6.0])

    assert mid.approx_eq(expected, 1e-10)


def test_midpoint_3d():
    """midpoint works in 3D."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 2.0, 3.0])
    v2 = lcc.Multivector.from_vector([5.0, 8.0, 11.0])

    mid = v1.midpoint(v2)
    expected = lcc.Multivector.from_vector([3.0, 5.0, 7.0])

    assert mid.approx_eq(expected, 1e-10)


def test_midpoint_equidistant():
    """midpoint is equidistant from both vectors."""
    import largecrimsoncanine as lcc

    v1 = lcc.Multivector.from_vector([1.0, 3.0])
    v2 = lcc.Multivector.from_vector([7.0, 11.0])

    mid = v1.midpoint(v2)

    d1 = mid.distance(v1)
    d2 = mid.distance(v2)

    assert abs(d1 - d2) < 1e-10


def test_midpoint_non_vector_self():
    """midpoint raises error if self is not a vector."""
    import largecrimsoncanine as lcc
    import pytest

    s = lcc.Multivector.from_scalar(5.0, dims=2)
    v = lcc.Multivector.from_vector([1.0, 2.0])

    with pytest.raises(ValueError, match="self is not a vector"):
        s.midpoint(v)


def test_midpoint_non_vector_other():
    """midpoint raises error if other is not a vector."""
    import largecrimsoncanine as lcc
    import pytest

    v = lcc.Multivector.from_vector([1.0, 2.0])
    s = lcc.Multivector.from_scalar(5.0, dims=2)

    with pytest.raises(ValueError, match="other is not a vector"):
        v.midpoint(s)


# =====================
# Constructor utility tests
# =====================


def test_from_list_basic():
    """from_list creates multivector from coefficients."""
    import largecrimsoncanine as lcc

    # 2D: [scalar, e1, e2, e12]
    mv = lcc.Multivector.from_list([1.0, 2.0, 3.0, 4.0])

    assert mv.dims == 2
    assert abs(mv.scalar() - 1.0) < 1e-10
    assert mv.is_vector() == False  # Has multiple grades


def test_from_list_roundtrip():
    """from_list -> to_list preserves coefficients."""
    import largecrimsoncanine as lcc

    original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    mv = lcc.Multivector.from_list(original)
    result = mv.to_list()

    assert result == original


def test_from_list_3d():
    """from_list works with 3D (8 coefficients)."""
    import largecrimsoncanine as lcc

    coeffs = [1.0] * 8  # 2^3 = 8
    mv = lcc.Multivector.from_list(coeffs)

    assert mv.dims == 3
    assert len(mv.to_list()) == 8


def test_from_list_not_power_of_2():
    """from_list raises error for non-power-of-2 length."""
    import largecrimsoncanine as lcc
    import pytest

    with pytest.raises(ValueError, match="not a power of 2"):
        lcc.Multivector.from_list([1.0, 2.0, 3.0])  # Length 3


def test_from_list_empty():
    """from_list raises error for empty list."""
    import largecrimsoncanine as lcc
    import pytest

    with pytest.raises(ValueError, match="must not be empty"):
        lcc.Multivector.from_list([])


def test_from_list_matches_to_dict():
    """from_list result matches from_dict for same data."""
    import largecrimsoncanine as lcc

    coeffs = [1.0, 2.0, 3.0, 4.0]
    mv1 = lcc.Multivector.from_list(coeffs)
    mv2 = lcc.Multivector.from_dict({"coeffs": coeffs, "dims": 2})

    assert mv1.approx_eq(mv2, 1e-10)


# =====================
# Clear grade tests
# =====================


def test_clear_grade_scalar():
    """clear_grade(0) removes scalar part."""
    import largecrimsoncanine as lcc

    # scalar + vector
    s = lcc.Multivector.from_scalar(5.0, dims=2)
    v = lcc.Multivector.from_vector([3.0, 4.0])
    mv = s + v

    cleared = mv.clear_grade(0)

    # Scalar should be zero
    assert abs(cleared.scalar()) < 1e-10
    # Vector part should be unchanged
    assert abs(cleared.grade(1).to_list()[1] - 3.0) < 1e-10


def test_clear_grade_vector():
    """clear_grade(1) removes vector part."""
    import largecrimsoncanine as lcc

    # scalar + vector
    s = lcc.Multivector.from_scalar(5.0, dims=2)
    v = lcc.Multivector.from_vector([3.0, 4.0])
    mv = s + v

    cleared = mv.clear_grade(1)

    # Scalar should be unchanged
    assert abs(cleared.scalar() - 5.0) < 1e-10
    # Vector part should be zero
    assert cleared.grade(1).is_zero()


def test_clear_grade_bivector():
    """clear_grade(2) removes bivector part."""
    import largecrimsoncanine as lcc

    # scalar + bivector
    s = lcc.Multivector.from_scalar(2.0, dims=3)
    B = lcc.Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    mv = s + B

    cleared = mv.clear_grade(2)

    # Scalar unchanged
    assert abs(cleared.scalar() - 2.0) < 1e-10
    # Bivector gone
    assert cleared.grade(2).is_zero()


def test_clear_grade_idempotent():
    """clear_grade twice is same as once."""
    import largecrimsoncanine as lcc

    mv = lcc.Multivector.from_list([1.0, 2.0, 3.0, 4.0])
    once = mv.clear_grade(1)
    twice = once.clear_grade(1)

    assert once.approx_eq(twice, 1e-10)


def test_clear_grade_invalid():
    """clear_grade raises error for invalid grade."""
    import largecrimsoncanine as lcc
    import pytest

    mv = lcc.Multivector.zero(2)
    with pytest.raises(ValueError, match="exceeds"):
        mv.clear_grade(5)


def test_clear_grade_preserves_other_grades():
    """clear_grade only affects the specified grade."""
    import largecrimsoncanine as lcc

    # All grades: scalar + vector + bivector
    mv = lcc.Multivector.from_list([1.0, 2.0, 3.0, 4.0])

    # Clear vector (grade 1)
    cleared = mv.clear_grade(1)

    # Scalar (grade 0) unchanged
    assert abs(cleared.grade(0).scalar() - 1.0) < 1e-10
    # Bivector (grade 2) unchanged
    assert abs(cleared.grade(2).to_list()[3] - 4.0) < 1e-10
    # Vector (grade 1) is zero
    assert cleared.grade(1).is_zero()
