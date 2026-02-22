"""Tests for the Algebra class and mixed-signature algebras (PGA, CGA, STA)."""

import pytest
import math
import largecrimsoncanine as lcc
from largecrimsoncanine import Algebra, Multivector


class TestAlgebraConstruction:
    """Test algebra construction and properties."""

    def test_euclidean_algebra(self):
        """Test Euclidean algebra creation."""
        R3 = Algebra.euclidean(3)
        assert R3.dimension == 3
        assert R3.num_blades == 8
        assert R3.p == 3
        assert R3.q == 0
        assert R3.r == 0
        assert R3.is_euclidean()
        assert not R3.is_pga()

    def test_pga_algebra(self):
        """Test PGA algebra creation."""
        PGA3D = Algebra.pga(3)
        assert PGA3D.dimension == 4  # 3 Euclidean + 1 degenerate
        assert PGA3D.num_blades == 16
        assert PGA3D.p == 3
        assert PGA3D.q == 0
        assert PGA3D.r == 1
        assert not PGA3D.is_euclidean()
        assert PGA3D.is_pga()

    def test_cga_algebra(self):
        """Test CGA algebra creation."""
        CGA3D = Algebra.cga(3)
        assert CGA3D.dimension == 5  # 3 Euclidean + e+ + e-
        assert CGA3D.num_blades == 32
        assert CGA3D.p == 4
        assert CGA3D.q == 1
        assert CGA3D.r == 0

    def test_sta_algebra(self):
        """Test Spacetime Algebra creation."""
        STA = Algebra.sta()
        assert STA.dimension == 4
        assert STA.num_blades == 16
        assert STA.p == 1  # timelike
        assert STA.q == 3  # spacelike
        assert STA.r == 0

    def test_custom_signature(self):
        """Test custom signature construction."""
        # Minkowski space Cl(1,3)
        minkowski = Algebra(1, 3, 0)
        assert minkowski.p == 1
        assert minkowski.q == 3
        assert minkowski.r == 0

        # Custom Cl(2,1,1)
        custom = Algebra(2, 1, 1)
        assert custom.dimension == 4
        assert custom.p == 2
        assert custom.q == 1
        assert custom.r == 1


class TestAlgebraBladenames:
    """Test blade naming."""

    def test_euclidean_blade_names(self):
        """Test blade names in Euclidean algebra."""
        R3 = Algebra.euclidean(3)
        assert R3.blade_name(0) == "1"
        assert R3.blade_name(1) == "e1"
        assert R3.blade_name(2) == "e2"
        assert R3.blade_name(3) == "e12"
        assert R3.blade_name(4) == "e3"
        assert R3.blade_name(7) == "e123"

    def test_pga_blade_names(self):
        """Test blade names in PGA."""
        PGA2D = Algebra.pga(2)
        assert PGA2D.blade_name(0) == "1"
        # Degenerate vector is e3 (at index 4 = 2^2)
        assert PGA2D.blade_name(4) == "e3"


class TestAlgebraEquality:
    """Test algebra equality and hashing."""

    def test_algebra_equality(self):
        """Test that algebras with same signature are equal."""
        R3a = Algebra.euclidean(3)
        R3b = Algebra.euclidean(3)
        R4 = Algebra.euclidean(4)

        assert R3a == R3b
        assert R3a != R4

    def test_algebra_hash(self):
        """Test algebra hashing for use in sets/dicts."""
        R3 = Algebra.euclidean(3)
        PGA = Algebra.pga(3)

        algebra_set = {R3, PGA}
        assert len(algebra_set) == 2
        assert R3 in algebra_set


class TestAlgebraRepr:
    """Test algebra string representations."""

    def test_repr_euclidean(self):
        """Test repr for Euclidean algebra."""
        R3 = Algebra.euclidean(3)
        assert str(R3) == "Cl(3,0,0)"

    def test_repr_pga(self):
        """Test repr for PGA."""
        PGA3D = Algebra.pga(3)
        assert str(PGA3D) == "Cl(3,0,1)"

    def test_repr_sta(self):
        """Test repr for STA."""
        STA = Algebra.sta()
        assert str(STA) == "Cl(1,3,0)"


class TestMultivectorWithAlgebra:
    """Test multivector construction with explicit algebra."""

    def test_zero_in_algebra(self):
        """Test creating zero multivector in algebra."""
        PGA = Algebra.pga(3)
        zero = Multivector.zero_in(PGA)
        assert len(zero.coefficients()) == 16  # 2^4 blades
        assert all(c == 0.0 for c in zero.coefficients())

    def test_scalar_in_algebra(self):
        """Test creating scalar in algebra."""
        PGA = Algebra.pga(3)
        s = Multivector.scalar_in(5.0, PGA)
        assert s.scalar() == 5.0

    def test_vector_in_algebra(self):
        """Test creating vector in algebra."""
        PGA = Algebra.pga(3)
        v = Multivector.vector_in([1.0, 2.0, 3.0, 0.0], PGA)
        coords = v.to_vector_coords()
        assert coords == [1.0, 2.0, 3.0, 0.0]

    def test_basis_in_algebra(self):
        """Test creating basis vector in algebra."""
        PGA = Algebra.pga(3)
        e1 = Multivector.basis_in(1, PGA)
        e4 = Multivector.basis_in(4, PGA)  # The degenerate direction

        # Check they're unit vectors
        assert e1.norm() == pytest.approx(1.0)
        # e4 is null, so norm is 0
        # (Actually in PGA the null direction has norm 0)

    def test_pseudoscalar_in_algebra(self):
        """Test creating pseudoscalar in algebra."""
        PGA = Algebra.pga(3)
        I = Multivector.pseudoscalar_in(PGA)
        assert I.max_grade() == 4


class TestPGAMetric:
    """Test PGA-specific metric properties."""

    def test_null_vector_squares_to_zero(self):
        """Test that the degenerate basis vector squares to 0."""
        PGA = Algebra.pga(3)
        e0 = Multivector.basis_in(4, PGA)  # e0 is the 4th basis vector in PGA3D

        # e0 * e0 should be 0 (null/degenerate)
        e0_squared = e0 * e0
        assert e0_squared.scalar() == pytest.approx(0.0)

    def test_euclidean_vectors_square_positive(self):
        """Test Euclidean vectors in PGA square to +1."""
        PGA = Algebra.pga(3)
        e1 = Multivector.basis_in(1, PGA)
        e2 = Multivector.basis_in(2, PGA)
        e3 = Multivector.basis_in(3, PGA)

        # Euclidean vectors square to +1
        assert (e1 * e1).scalar() == pytest.approx(1.0)
        assert (e2 * e2).scalar() == pytest.approx(1.0)
        assert (e3 * e3).scalar() == pytest.approx(1.0)


class TestSTAMetric:
    """Test Spacetime Algebra metric properties."""

    def test_timelike_squares_positive(self):
        """Test timelike basis vector squares to +1."""
        STA = Algebra.sta()
        gamma0 = Multivector.basis_in(1, STA)  # First basis is timelike

        # gamma0^2 = +1
        assert (gamma0 * gamma0).scalar() == pytest.approx(1.0)

    def test_spacelike_squares_negative(self):
        """Test spacelike basis vectors square to -1."""
        STA = Algebra.sta()
        gamma1 = Multivector.basis_in(2, STA)
        gamma2 = Multivector.basis_in(3, STA)
        gamma3 = Multivector.basis_in(4, STA)

        # gamma_i^2 = -1 for i = 1,2,3
        assert (gamma1 * gamma1).scalar() == pytest.approx(-1.0)
        assert (gamma2 * gamma2).scalar() == pytest.approx(-1.0)
        assert (gamma3 * gamma3).scalar() == pytest.approx(-1.0)

    def test_different_basis_anticommute(self):
        """Test that different basis vectors anticommute."""
        STA = Algebra.sta()
        gamma0 = Multivector.basis_in(1, STA)
        gamma1 = Multivector.basis_in(2, STA)

        # gamma0 * gamma1 + gamma1 * gamma0 = 0
        anticommutator = gamma0 * gamma1 + gamma1 * gamma0
        assert all(abs(c) < 1e-10 for c in anticommutator.coefficients())


class TestCGAMetric:
    """Test Conformal Geometric Algebra metric properties."""

    def test_positive_directions(self):
        """Test positive signature directions."""
        CGA = Algebra.cga(3)
        # First 4 directions are positive (e1, e2, e3, e+)
        for i in range(1, 5):
            ei = Multivector.basis_in(i, CGA)
            assert (ei * ei).scalar() == pytest.approx(1.0), f"e{i} should square to +1"

    def test_negative_direction(self):
        """Test negative signature direction."""
        CGA = Algebra.cga(3)
        # 5th direction (e-) is negative
        e_minus = Multivector.basis_in(5, CGA)
        assert (e_minus * e_minus).scalar() == pytest.approx(-1.0)


class TestBackwardCompatibility:
    """Test that old API still works."""

    def test_from_vector_still_works(self):
        """Test that from_vector without algebra still works."""
        v = Multivector.from_vector([1, 2, 3])
        assert v.norm() == pytest.approx(math.sqrt(14))

    def test_dims_still_works(self):
        """Test that dims property still works."""
        v = Multivector.from_vector([1, 2, 3])
        assert v.dims == 3

    def test_products_still_work(self):
        """Test products without explicit algebra."""
        a = Multivector.from_vector([1, 0, 0])
        b = Multivector.from_vector([0, 1, 0])

        # Wedge product
        ab = a ^ b
        assert ab.bivector_part().to_bivector_coords()[0] == pytest.approx(1.0)
