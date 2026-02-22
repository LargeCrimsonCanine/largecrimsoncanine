"""Tests for batched multivector operations.

These tests verify the NumPy-backed batch operations that enable
efficient processing of thousands of multivectors at once.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from largecrimsoncanine import Algebra, Multivector, MultivectorBatch


# =============================================================================
# Batch Creation Tests
# =============================================================================

class TestBatchCreation:
    """Test batch construction methods."""

    def test_from_numpy(self):
        """Test creating a batch from raw NumPy array."""
        R3 = Algebra.euclidean(3)
        coeffs = np.random.randn(100, 8)
        batch = MultivectorBatch.from_numpy(R3, coeffs)
        assert batch.len() == 100
        assert batch.num_blades() == 8

    def test_from_numpy_wrong_columns(self):
        """Test that wrong number of columns raises error."""
        R3 = Algebra.euclidean(3)
        coeffs = np.random.randn(100, 4)  # Wrong: should be 8
        with pytest.raises(ValueError, match="must have 8 columns"):
            MultivectorBatch.from_numpy(R3, coeffs)

    def test_from_vectors(self):
        """Test creating a batch of vectors from coordinates."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(100, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)
        assert batch.len() == 100

        # Check that we can extract the coordinates back
        extracted = batch.to_vector_coords()
        assert_allclose(extracted, coords)

    def test_from_vectors_wrong_columns(self):
        """Test that wrong coordinate dimension raises error."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(100, 2)  # Wrong: should be 3
        with pytest.raises(ValueError, match="must have 3 columns"):
            MultivectorBatch.from_vectors(R3, coords)

    def test_from_scalars(self):
        """Test creating a batch of scalars."""
        R3 = Algebra.euclidean(3)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        batch = MultivectorBatch.from_scalars(R3, values)
        assert batch.len() == 5

        # Extract scalars
        scalars = batch.scalar()
        assert_allclose(scalars, values)

    def test_from_bivectors(self):
        """Test creating a batch of bivectors."""
        R3 = Algebra.euclidean(3)
        # 3 bivector components in 3D: e12, e13, e23
        biv_coeffs = np.random.randn(50, 3)
        batch = MultivectorBatch.from_bivectors(R3, biv_coeffs)
        assert batch.len() == 50

    def test_to_numpy(self):
        """Test converting batch to NumPy array."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(20, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)
        arr = batch.to_numpy()
        assert arr.shape == (20, 8)

    def test_empty_batch(self):
        """Test behavior with empty batch."""
        R3 = Algebra.euclidean(3)
        coeffs = np.zeros((0, 8))
        batch = MultivectorBatch.from_numpy(R3, coeffs)
        assert batch.len() == 0
        assert batch.is_empty()


# =============================================================================
# Element Access Tests
# =============================================================================

class TestElementAccess:
    """Test accessing individual elements from batches."""

    def test_get_element(self):
        """Test getting a single multivector by index."""
        R3 = Algebra.euclidean(3)
        coords = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        batch = MultivectorBatch.from_vectors(R3, coords)

        # Get first element (e1)
        mv0 = batch.get(0)
        assert_allclose(mv0.to_vector_coords(), [1.0, 0.0, 0.0])

        # Get second element (e2)
        mv1 = batch.get(1)
        assert_allclose(mv1.to_vector_coords(), [0.0, 1.0, 0.0])

    def test_getitem(self):
        """Test Python indexing."""
        R3 = Algebra.euclidean(3)
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        batch = MultivectorBatch.from_vectors(R3, coords)

        # Positive index
        mv = batch[0]
        assert_allclose(mv.to_vector_coords(), [1.0, 2.0, 3.0])

        # Negative index
        mv = batch[-1]
        assert_allclose(mv.to_vector_coords(), [4.0, 5.0, 6.0])

    def test_index_out_of_bounds(self):
        """Test that out-of-bounds access raises IndexError."""
        R3 = Algebra.euclidean(3)
        batch = MultivectorBatch.from_vectors(R3, np.random.randn(5, 3))

        with pytest.raises(IndexError):
            batch.get(10)

        with pytest.raises(IndexError):
            batch[100]

    def test_set_element(self):
        """Test setting a single multivector by index."""
        R3 = Algebra.euclidean(3)
        batch = MultivectorBatch.from_vectors(R3, np.zeros((3, 3)))

        # Set middle element
        new_mv = Multivector.from_vector([1.0, 2.0, 3.0])
        batch.set(1, new_mv)

        # Verify
        extracted = batch.get(1)
        assert_allclose(extracted.to_vector_coords(), [1.0, 2.0, 3.0])

    def test_len(self):
        """Test __len__ method."""
        R3 = Algebra.euclidean(3)
        batch = MultivectorBatch.from_vectors(R3, np.random.randn(42, 3))
        assert len(batch) == 42


# =============================================================================
# Sandwich Product Tests
# =============================================================================

class TestSandwichProduct:
    """Test the sandwich (rotor application) operation."""

    def test_sandwich_identity(self):
        """Test that identity rotor leaves vectors unchanged."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(100, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        # Identity rotor: just the scalar 1
        identity = Multivector.from_scalar(1.0, 3)
        rotated = batch.sandwich(identity)

        extracted = rotated.to_vector_coords()
        assert_allclose(extracted, coords, rtol=1e-10)

    def test_sandwich_preserves_norm(self):
        """Test that sandwich product preserves vector norms."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(1000, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        # Create a rotor for 45 degree rotation around Z axis
        axis = Multivector.from_vector([0.0, 0.0, 1.0])
        rotor = Multivector.from_axis_angle(axis, np.pi / 4)

        rotated = batch.sandwich(rotor)

        # Compare norms
        original_norms = batch.norm()
        rotated_norms = rotated.norm()
        assert_allclose(rotated_norms, original_norms, rtol=1e-10)

    def test_sandwich_rotation_z(self):
        """Test 90 degree rotation around Z axis."""
        R3 = Algebra.euclidean(3)

        # e1 should map to e2 under 90 degree rotation around e3
        coords = np.array([[1.0, 0.0, 0.0]])
        batch = MultivectorBatch.from_vectors(R3, coords)

        axis = Multivector.from_vector([0.0, 0.0, 1.0])
        rotor = Multivector.from_axis_angle(axis, np.pi / 2)

        rotated = batch.sandwich(rotor)
        result = rotated.to_vector_coords()

        expected = np.array([[0.0, 1.0, 0.0]])
        assert_allclose(result, expected, atol=1e-10)

    def test_sandwich_batch_vs_loop(self):
        """Verify batch sandwich gives same results as looping."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(50, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        # Create rotor
        axis = Multivector.from_vector([1.0, 1.0, 1.0])
        rotor = Multivector.from_axis_angle(axis, 0.7)

        # Batch operation
        batch_result = batch.sandwich(rotor)
        batch_coords = batch_result.to_vector_coords()

        # Loop operation
        loop_coords = np.zeros_like(coords)
        for i in range(len(coords)):
            mv = Multivector.from_vector(coords[i].tolist())
            rotated = rotor.sandwich(mv)
            loop_coords[i] = rotated.to_vector_coords()

        assert_allclose(batch_coords, loop_coords, rtol=1e-10)


# =============================================================================
# Geometric Product Tests
# =============================================================================

class TestGeometricProduct:
    """Test batch geometric product."""

    def test_gp_same_size(self):
        """Test element-wise geometric product."""
        R3 = Algebra.euclidean(3)
        a_coords = np.random.randn(10, 3)
        b_coords = np.random.randn(10, 3)

        batch_a = MultivectorBatch.from_vectors(R3, a_coords)
        batch_b = MultivectorBatch.from_vectors(R3, b_coords)

        result = batch_a.geometric_product(batch_b)
        assert result.len() == 10

        # Verify against loop
        for i in range(10):
            mv_a = Multivector.from_vector(a_coords[i].tolist())
            mv_b = Multivector.from_vector(b_coords[i].tolist())
            expected = mv_a.geometric_product(mv_b)

            actual = result.get(i)
            for j in range(8):
                assert abs(actual.coefficients()[j] - expected.coefficients()[j]) < 1e-10

    def test_gp_broadcast_single(self):
        """Test broadcasting a single multivector to batch."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(20, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        # Single scalar
        scalar = MultivectorBatch.from_scalars(R3, np.array([2.0]))

        # Should scale all vectors by 2
        result = batch.geometric_product(scalar)
        assert result.len() == 20

        result_coords = result.to_vector_coords()
        assert_allclose(result_coords, coords * 2.0, rtol=1e-10)

    def test_gp_size_mismatch_error(self):
        """Test that mismatched sizes raise error."""
        R3 = Algebra.euclidean(3)
        batch_a = MultivectorBatch.from_vectors(R3, np.random.randn(10, 3))
        batch_b = MultivectorBatch.from_vectors(R3, np.random.randn(5, 3))

        with pytest.raises(ValueError, match="batch sizes must match"):
            batch_a.geometric_product(batch_b)


# =============================================================================
# Outer Product Tests
# =============================================================================

class TestOuterProduct:
    """Test batch outer (wedge) product."""

    def test_wedge_vectors(self):
        """Test wedging two vectors gives bivector."""
        R3 = Algebra.euclidean(3)

        # e1 wedge e2 = e12
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])

        batch_a = MultivectorBatch.from_vectors(R3, a)
        batch_b = MultivectorBatch.from_vectors(R3, b)

        result = batch_a.outer_product(batch_b)

        # Get the bivector part
        grade2 = result.grade(2)
        mv = grade2.get(0)

        # e12 is at index 3
        assert abs(mv.coefficients()[3] - 1.0) < 1e-10

    def test_wedge_anticommutative(self):
        """Test that a ^ b = -b ^ a for vectors."""
        R3 = Algebra.euclidean(3)
        a = np.random.randn(10, 3)
        b = np.random.randn(10, 3)

        batch_a = MultivectorBatch.from_vectors(R3, a)
        batch_b = MultivectorBatch.from_vectors(R3, b)

        ab = batch_a.wedge(batch_b)
        ba = batch_b.wedge(batch_a)

        # Check that ab = -ba
        ab_arr = ab.to_numpy()
        ba_arr = ba.to_numpy()
        assert_allclose(ab_arr, -ba_arr, rtol=1e-10)


# =============================================================================
# Left Contraction Tests
# =============================================================================

class TestLeftContraction:
    """Test batch left contraction (inner product)."""

    def test_lc_vectors_scalar(self):
        """Test that v . w gives scalar (dot product)."""
        R3 = Algebra.euclidean(3)
        a = np.random.randn(20, 3)
        b = np.random.randn(20, 3)

        batch_a = MultivectorBatch.from_vectors(R3, a)
        batch_b = MultivectorBatch.from_vectors(R3, b)

        result = batch_a.left_contraction(batch_b)

        # Extract scalar parts
        scalars = result.scalar()

        # Should match numpy dot products
        expected = np.sum(a * b, axis=1)
        assert_allclose(scalars, expected, rtol=1e-10)


# =============================================================================
# Norm and Normalization Tests
# =============================================================================

class TestNormalization:
    """Test norm and normalization operations."""

    def test_norm(self):
        """Test computing batch norms."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(100, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        norms = batch.norm()

        # Should match numpy norms
        expected = np.linalg.norm(coords, axis=1)
        assert_allclose(norms, expected, rtol=1e-10)

    def test_norm_squared(self):
        """Test computing squared norms."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(50, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        norm_sq = batch.norm_squared()

        expected = np.sum(coords ** 2, axis=1)
        assert_allclose(norm_sq, expected, rtol=1e-10)

    def test_normalized(self):
        """Test normalizing batch of vectors."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(100, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        normalized = batch.normalized()
        norms = normalized.norm()

        # All norms should be 1
        assert_allclose(norms, np.ones(100), rtol=1e-10)


# =============================================================================
# Reverse and Involutions Tests
# =============================================================================

class TestInvolutions:
    """Test reverse and grade involution."""

    def test_reverse_vectors(self):
        """Test that vectors are unchanged by reverse."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(20, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        reversed_batch = batch.reverse()
        extracted = reversed_batch.to_vector_coords()

        # Vectors (grade 1) unchanged by reverse
        assert_allclose(extracted, coords, rtol=1e-10)

    def test_reverse_bivectors(self):
        """Test that bivectors are negated by reverse."""
        R3 = Algebra.euclidean(3)
        biv_coeffs = np.random.randn(20, 3)
        batch = MultivectorBatch.from_bivectors(R3, biv_coeffs)

        reversed_batch = batch.reverse()

        # Bivectors should be negated
        orig_arr = batch.to_numpy()
        rev_arr = reversed_batch.to_numpy()

        # Compare non-zero columns (bivector indices: 3, 5, 6)
        assert_allclose(rev_arr[:, 3], -orig_arr[:, 3], rtol=1e-10)
        assert_allclose(rev_arr[:, 5], -orig_arr[:, 5], rtol=1e-10)
        assert_allclose(rev_arr[:, 6], -orig_arr[:, 6], rtol=1e-10)

    def test_grade_involution_vectors(self):
        """Test that vectors are negated by grade involution."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(20, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        inverted = batch.grade_involution()
        extracted = inverted.to_vector_coords()

        # Vectors (odd grade) negated
        assert_allclose(extracted, -coords, rtol=1e-10)


# =============================================================================
# Arithmetic Operations Tests
# =============================================================================

class TestArithmetic:
    """Test basic arithmetic operations."""

    def test_add(self):
        """Test batch addition."""
        R3 = Algebra.euclidean(3)
        a = np.random.randn(20, 3)
        b = np.random.randn(20, 3)

        batch_a = MultivectorBatch.from_vectors(R3, a)
        batch_b = MultivectorBatch.from_vectors(R3, b)

        result = batch_a + batch_b
        extracted = result.to_vector_coords()

        assert_allclose(extracted, a + b, rtol=1e-10)

    def test_sub(self):
        """Test batch subtraction."""
        R3 = Algebra.euclidean(3)
        a = np.random.randn(20, 3)
        b = np.random.randn(20, 3)

        batch_a = MultivectorBatch.from_vectors(R3, a)
        batch_b = MultivectorBatch.from_vectors(R3, b)

        result = batch_a - batch_b
        extracted = result.to_vector_coords()

        assert_allclose(extracted, a - b, rtol=1e-10)

    def test_scale(self):
        """Test scalar multiplication."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(20, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        # Both orders should work
        result1 = batch * 2.5
        result2 = batch.scale(2.5)

        expected = coords * 2.5
        assert_allclose(result1.to_vector_coords(), expected, rtol=1e-10)
        assert_allclose(result2.to_vector_coords(), expected, rtol=1e-10)

    def test_neg(self):
        """Test negation."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(20, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        negated = -batch
        extracted = negated.to_vector_coords()

        assert_allclose(extracted, -coords, rtol=1e-10)

    def test_div(self):
        """Test scalar division."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(20, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        result = batch / 2.0
        expected = coords / 2.0

        assert_allclose(result.to_vector_coords(), expected, rtol=1e-10)

    def test_div_by_zero(self):
        """Test division by zero raises error."""
        R3 = Algebra.euclidean(3)
        batch = MultivectorBatch.from_vectors(R3, np.random.randn(5, 3))

        with pytest.raises(ZeroDivisionError):
            batch / 0.0


# =============================================================================
# Slicing and Concatenation Tests
# =============================================================================

class TestSlicingConcat:
    """Test slicing and concatenation."""

    def test_slice(self):
        """Test slicing a batch."""
        R3 = Algebra.euclidean(3)
        coords = np.random.randn(100, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        sliced = batch.slice(10, 20)
        assert sliced.len() == 10

        extracted = sliced.to_vector_coords()
        assert_allclose(extracted, coords[10:20], rtol=1e-10)

    def test_concat(self):
        """Test concatenating batches."""
        R3 = Algebra.euclidean(3)
        coords_a = np.random.randn(30, 3)
        coords_b = np.random.randn(20, 3)

        batch_a = MultivectorBatch.from_vectors(R3, coords_a)
        batch_b = MultivectorBatch.from_vectors(R3, coords_b)

        combined = batch_a.concat(batch_b)
        assert combined.len() == 50

        extracted = combined.to_vector_coords()
        expected = np.vstack([coords_a, coords_b])
        assert_allclose(extracted, expected, rtol=1e-10)


# =============================================================================
# Grade Extraction Tests
# =============================================================================

class TestGradeExtraction:
    """Test extracting specific grades from batch."""

    def test_grade_extraction(self):
        """Test extracting grade-k parts."""
        R3 = Algebra.euclidean(3)

        # Create mixed multivectors: scalar + vector + bivector
        coeffs = np.random.randn(10, 8)
        batch = MultivectorBatch.from_numpy(R3, coeffs)

        # Extract vectors only
        vectors = batch.grade(1)
        vec_arr = vectors.to_numpy()

        # Should only have grade-1 components (indices 1, 2, 4)
        for i in range(10):
            assert vec_arr[i, 0] == 0  # scalar
            assert vec_arr[i, 3] == 0  # bivector
            assert vec_arr[i, 5] == 0  # bivector
            assert vec_arr[i, 6] == 0  # bivector
            assert vec_arr[i, 7] == 0  # trivector


# =============================================================================
# String Representation Tests
# =============================================================================

class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ format."""
        R3 = Algebra.euclidean(3)
        batch = MultivectorBatch.from_vectors(R3, np.random.randn(42, 3))
        s = repr(batch)
        assert "42" in s
        assert "Cl(3,0,0)" in s


# =============================================================================
# Performance Benchmark Tests
# =============================================================================

class TestPerformance:
    """Basic performance sanity checks (not full benchmarks)."""

    def test_large_batch(self):
        """Test that large batches work correctly."""
        R3 = Algebra.euclidean(3)
        n = 10000
        coords = np.random.randn(n, 3)
        batch = MultivectorBatch.from_vectors(R3, coords)

        # Apply rotation
        axis = Multivector.from_vector([0.0, 0.0, 1.0])
        rotor = Multivector.from_axis_angle(axis, np.pi / 6)
        rotated = batch.sandwich(rotor)

        # Check norms are preserved
        orig_norms = batch.norm()
        rot_norms = rotated.norm()
        assert_allclose(rot_norms, orig_norms, rtol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
