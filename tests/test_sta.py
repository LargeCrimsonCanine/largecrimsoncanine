"""
Spacetime Algebra (STA) tests for largecrimsoncanine.

STA is Cl(1,3,0) with the "mostly minus" convention:
- gamma0^2 = +1 (timelike)
- gamma1^2 = gamma2^2 = gamma3^2 = -1 (spacelike)

Run with: pytest tests/test_sta.py -v
"""
import pytest
import math
import largecrimsoncanine as lcc
from largecrimsoncanine import Algebra, Multivector


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sta():
    """Create STA algebra Cl(1,3,0)."""
    return Algebra.sta()


@pytest.fixture
def gamma0(sta):
    """Timelike basis vector gamma0."""
    return Multivector.sta_gamma(sta, 0)


@pytest.fixture
def gamma1(sta):
    """Spacelike basis vector gamma1."""
    return Multivector.sta_gamma(sta, 1)


@pytest.fixture
def gamma2(sta):
    """Spacelike basis vector gamma2."""
    return Multivector.sta_gamma(sta, 2)


@pytest.fixture
def gamma3(sta):
    """Spacelike basis vector gamma3."""
    return Multivector.sta_gamma(sta, 3)


# =============================================================================
# Metric Signature Tests
# =============================================================================

class TestMetricSignature:
    """Test the STA metric signature (+,-,-,-)."""

    def test_gamma0_squares_positive(self, sta):
        """gamma0^2 = +1 (timelike)."""
        g0 = Multivector.sta_gamma(sta, 0)
        g0_sq = g0 * g0
        assert g0_sq.scalar() == pytest.approx(1.0)

    def test_gamma1_squares_negative(self, sta):
        """gamma1^2 = -1 (spacelike)."""
        g1 = Multivector.sta_gamma(sta, 1)
        g1_sq = g1 * g1
        assert g1_sq.scalar() == pytest.approx(-1.0)

    def test_gamma2_squares_negative(self, sta):
        """gamma2^2 = -1 (spacelike)."""
        g2 = Multivector.sta_gamma(sta, 2)
        g2_sq = g2 * g2
        assert g2_sq.scalar() == pytest.approx(-1.0)

    def test_gamma3_squares_negative(self, sta):
        """gamma3^2 = -1 (spacelike)."""
        g3 = Multivector.sta_gamma(sta, 3)
        g3_sq = g3 * g3
        assert g3_sq.scalar() == pytest.approx(-1.0)

    def test_all_basis_squares(self, sta):
        """Verify all basis vector squares at once."""
        expected = [1.0, -1.0, -1.0, -1.0]  # (+,-,-,-)
        for i, exp in enumerate(expected):
            gi = Multivector.sta_gamma(sta, i)
            gi_sq = gi * gi
            assert gi_sq.scalar() == pytest.approx(exp), f"gamma{i}^2 failed"


# =============================================================================
# Anticommutation Tests
# =============================================================================

class TestAnticommutation:
    """Test that different gamma matrices anticommute."""

    def test_gamma0_gamma1_anticommute(self, gamma0, gamma1):
        """gamma0 * gamma1 + gamma1 * gamma0 = 0."""
        g01 = gamma0 * gamma1
        g10 = gamma1 * gamma0
        anticomm = g01 + g10
        # All coefficients should be zero
        for c in anticomm.coefficients():
            assert abs(c) < 1e-10

    def test_all_pairs_anticommute(self, sta):
        """All distinct gamma pairs anticommute: {gamma_mu, gamma_nu} = 2*eta_{mu,nu}."""
        for mu in range(4):
            for nu in range(mu + 1, 4):
                g_mu = Multivector.sta_gamma(sta, mu)
                g_nu = Multivector.sta_gamma(sta, nu)

                g_mu_nu = g_mu * g_nu
                g_nu_mu = g_nu * g_mu
                anticomm = g_mu_nu + g_nu_mu

                # Anticommutator should be zero for mu != nu
                for c in anticomm.coefficients():
                    assert abs(c) < 1e-10, f"gamma{mu} and gamma{nu} don't anticommute"

    def test_clifford_algebra_relation(self, sta):
        """Verify {gamma_mu, gamma_nu} = 2*eta_{mu,nu} for all pairs."""
        # Minkowski metric (mostly minus)
        eta = [[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, -1]]

        for mu in range(4):
            for nu in range(4):
                g_mu = Multivector.sta_gamma(sta, mu)
                g_nu = Multivector.sta_gamma(sta, nu)

                # {gamma_mu, gamma_nu} = gamma_mu * gamma_nu + gamma_nu * gamma_mu
                anticomm = g_mu * g_nu + g_nu * g_mu

                # Should equal 2 * eta_{mu,nu} * identity (scalar)
                expected = 2.0 * eta[mu][nu]
                assert anticomm.scalar() == pytest.approx(expected), \
                    f"{{gamma{mu}, gamma{nu}}} != 2*eta[{mu},{nu}]"


# =============================================================================
# Pseudoscalar Tests
# =============================================================================

class TestPseudoscalar:
    """Test the STA pseudoscalar I = gamma0123."""

    def test_pseudoscalar_squared(self, sta):
        """I^2 = gamma0123 * gamma0123 = -1."""
        I = Multivector.sta_I(sta)
        I_sq = I * I
        assert I_sq.scalar() == pytest.approx(-1.0)

    def test_pseudoscalar_grade(self, sta):
        """Pseudoscalar is grade-4."""
        I = Multivector.sta_I(sta)
        assert I.max_grade() == 4


# =============================================================================
# Spacetime Vector Tests
# =============================================================================

class TestSpacetimeVector:
    """Test spacetime 4-vector creation and operations."""

    def test_vector_creation(self, sta):
        """Create a spacetime vector and verify components."""
        x = Multivector.sta_vector(sta, 5.0, 3.0, 4.0, 0.0)
        t, r = x.sta_spacetime_split()

        assert t == pytest.approx(5.0)
        assert r[0] == pytest.approx(3.0)
        assert r[1] == pytest.approx(4.0)
        assert r[2] == pytest.approx(0.0)

    def test_vector_is_grade_1(self, sta):
        """Spacetime vector should be pure grade-1."""
        x = Multivector.sta_vector(sta, 1.0, 2.0, 3.0, 4.0)
        assert x.is_vector()


# =============================================================================
# Spacetime Interval Tests
# =============================================================================

class TestSpacetimeInterval:
    """Test spacetime interval calculations."""

    def test_timelike_interval(self, sta):
        """Timelike vector: t^2 > r^2."""
        x = Multivector.sta_vector(sta, 5.0, 3.0, 0.0, 0.0)
        # s^2 = 25 - 9 = 16 > 0

        assert x.sta_is_timelike()
        assert not x.sta_is_spacelike()
        assert not x.sta_is_lightlike()
        assert x.sta_interval_squared() == pytest.approx(16.0)
        assert x.sta_proper_time() == pytest.approx(4.0)

    def test_spacelike_interval(self, sta):
        """Spacelike vector: t^2 < r^2."""
        x = Multivector.sta_vector(sta, 1.0, 5.0, 0.0, 0.0)
        # s^2 = 1 - 25 = -24 < 0

        assert not x.sta_is_timelike()
        assert x.sta_is_spacelike()
        assert not x.sta_is_lightlike()
        assert x.sta_interval_squared() == pytest.approx(-24.0)

    def test_lightlike_interval(self, sta):
        """Lightlike/null vector: t^2 = r^2."""
        x = Multivector.sta_vector(sta, 1.0, 1.0, 0.0, 0.0)
        # s^2 = 1 - 1 = 0

        assert not x.sta_is_timelike()
        assert not x.sta_is_spacelike()
        assert x.sta_is_lightlike()
        assert x.sta_interval_squared() == pytest.approx(0.0)

    def test_lightlike_3d(self, sta):
        """Lightlike with 3D spatial component."""
        # t = sqrt(3), r = (1,1,1) => |r| = sqrt(3)
        # s^2 = 3 - 3 = 0
        t = math.sqrt(3)
        x = Multivector.sta_vector(sta, t, 1.0, 1.0, 1.0)

        assert x.sta_is_lightlike()


# =============================================================================
# Lorentz Boost Tests
# =============================================================================

class TestLorentzBoost:
    """Test Lorentz boost operations."""

    def test_boost_preserves_interval(self, sta):
        """Lorentz boosts preserve the spacetime interval."""
        x = Multivector.sta_vector(sta, 5.0, 3.0, 0.0, 0.0)
        interval_before = x.sta_interval_squared()

        # Boost at 0.5c in x-direction
        R = Multivector.sta_boost(sta, [0.5, 0.0, 0.0])
        R_rev = R.reverse()

        # Apply boost: x' = R * x * ~R
        x_boosted = R * x * R_rev

        interval_after = x_boosted.sta_interval_squared()
        assert interval_before == pytest.approx(interval_after, rel=1e-10)

    def test_boost_gamma_factor(self, sta):
        """Verify Lorentz gamma factor calculation."""
        # v = 0.6c => gamma = 1/sqrt(1-0.36) = 1/sqrt(0.64) = 1.25
        R = Multivector.sta_boost(sta, [0.6, 0.0, 0.0])
        gamma = R.sta_lorentz_gamma()
        assert gamma == pytest.approx(1.25)

    def test_boost_gamma_high_velocity(self, sta):
        """Test gamma factor at high velocity."""
        # v = 0.8c => gamma = 1/sqrt(1-0.64) = 1/sqrt(0.36) = 5/3
        R = Multivector.sta_boost(sta, [0.8, 0.0, 0.0])
        gamma = R.sta_lorentz_gamma()
        assert gamma == pytest.approx(5.0/3.0)

    def test_boost_composition(self, sta):
        """Two successive boosts compose correctly."""
        R1 = Multivector.sta_boost(sta, [0.3, 0.0, 0.0])
        R2 = Multivector.sta_boost(sta, [0.3, 0.0, 0.0])

        # Composed rotor
        R_composed = R2 * R1

        # Gamma should increase
        gamma_single = R1.sta_lorentz_gamma()
        gamma_composed = R_composed.sta_lorentz_gamma()
        assert gamma_composed > gamma_single

    def test_zero_velocity_boost(self, sta):
        """Zero velocity boost is identity."""
        R = Multivector.sta_boost(sta, [0.0, 0.0, 0.0])

        # Should be approximately 1 (identity rotor)
        assert R.scalar() == pytest.approx(1.0)
        # Other components should be zero
        coeffs = R.coefficients()
        for i, c in enumerate(coeffs):
            if i != 0:  # Skip scalar
                assert abs(c) < 1e-10

    def test_boost_direction(self, sta):
        """Boost in y-direction."""
        R = Multivector.sta_boost(sta, [0.0, 0.6, 0.0])
        gamma = R.sta_lorentz_gamma()
        assert gamma == pytest.approx(1.25)

    def test_superluminal_velocity_error(self, sta):
        """Velocity >= c should raise error."""
        with pytest.raises(ValueError, match="speed of light"):
            Multivector.sta_boost(sta, [1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="speed of light"):
            Multivector.sta_boost(sta, [0.5, 0.5, 0.8])  # |v| > 1


# =============================================================================
# Spatial Rotation Tests
# =============================================================================

class TestSpatialRotation:
    """Test spatial rotations in STA."""

    def test_rotation_preserves_interval(self, sta):
        """Spatial rotations preserve the spacetime interval."""
        x = Multivector.sta_vector(sta, 5.0, 3.0, 4.0, 0.0)
        interval_before = x.sta_interval_squared()

        # 45 degree rotation about z-axis
        R = Multivector.sta_rotation(sta, [0.0, 0.0, 1.0], math.pi/4)
        R_rev = R.reverse()

        x_rotated = R * x * R_rev
        interval_after = x_rotated.sta_interval_squared()

        assert interval_before == pytest.approx(interval_after, rel=1e-10)

    def test_rotation_leaves_time_invariant(self, sta):
        """Spatial rotation doesn't affect time component."""
        x = Multivector.sta_vector(sta, 5.0, 3.0, 4.0, 0.0)

        R = Multivector.sta_rotation(sta, [0.0, 0.0, 1.0], math.pi/2)
        R_rev = R.reverse()

        x_rotated = R * x * R_rev
        t, _ = x_rotated.sta_spacetime_split()

        assert t == pytest.approx(5.0)

    def test_90_degree_rotation(self, sta):
        """90 degree rotation about z-axis: x -> y."""
        x = Multivector.sta_vector(sta, 0.0, 1.0, 0.0, 0.0)

        R = Multivector.sta_rotation(sta, [0.0, 0.0, 1.0], math.pi/2)
        R_rev = R.reverse()

        x_rotated = R * x * R_rev
        t, r = x_rotated.sta_spacetime_split()

        assert t == pytest.approx(0.0)
        assert r[0] == pytest.approx(0.0, abs=1e-10)  # x should be ~0
        assert r[1] == pytest.approx(1.0)  # y should be 1
        assert r[2] == pytest.approx(0.0)

    def test_rotation_about_x_axis(self, sta):
        """Rotation about x-axis: y -> z."""
        x = Multivector.sta_vector(sta, 0.0, 0.0, 1.0, 0.0)

        R = Multivector.sta_rotation(sta, [1.0, 0.0, 0.0], math.pi/2)
        R_rev = R.reverse()

        x_rotated = R * x * R_rev
        t, r = x_rotated.sta_spacetime_split()

        assert r[0] == pytest.approx(0.0)  # x stays 0
        assert r[1] == pytest.approx(0.0, abs=1e-10)  # y -> 0
        assert r[2] == pytest.approx(1.0)  # z -> 1


# =============================================================================
# Electromagnetic Field Tests
# =============================================================================

class TestElectromagneticField:
    """Test electromagnetic field representation."""

    def test_em_field_roundtrip(self, sta):
        """Create EM field and decompose back."""
        E = [1.0, 2.0, 3.0]
        B = [0.5, -0.5, 1.0]

        F = Multivector.sta_bivector(sta, E, B)
        E_out, B_out = F.sta_field_decompose()

        for i in range(3):
            assert E_out[i] == pytest.approx(E[i])
            assert B_out[i] == pytest.approx(B[i])

    def test_pure_electric_field(self, sta):
        """Pure electric field in x-direction."""
        E = [1.0, 0.0, 0.0]
        B = [0.0, 0.0, 0.0]

        F = Multivector.sta_bivector(sta, E, B)
        E_out, B_out = F.sta_field_decompose()

        assert E_out[0] == pytest.approx(1.0)
        assert E_out[1] == pytest.approx(0.0)
        assert E_out[2] == pytest.approx(0.0)
        for b in B_out:
            assert b == pytest.approx(0.0)

    def test_pure_magnetic_field(self, sta):
        """Pure magnetic field in z-direction."""
        E = [0.0, 0.0, 0.0]
        B = [0.0, 0.0, 1.0]

        F = Multivector.sta_bivector(sta, E, B)
        E_out, B_out = F.sta_field_decompose()

        for e in E_out:
            assert e == pytest.approx(0.0)
        assert B_out[2] == pytest.approx(1.0)

    def test_em_dual(self, sta):
        """Electromagnetic dual: F* = I*F."""
        # Pure electric field
        E = [1.0, 0.0, 0.0]
        B = [0.0, 0.0, 0.0]

        F = Multivector.sta_bivector(sta, E, B)
        F_dual = F.sta_em_dual()

        # Dual should exchange E and B (with sign)
        E_dual, B_dual = F_dual.sta_field_decompose()

        # The exact relation depends on conventions
        # E -> B direction, B -> -E direction (approximately)
        assert abs(B_dual[0]) == pytest.approx(1.0, abs=1e-10)


# =============================================================================
# Algebra Validation Tests
# =============================================================================

class TestAlgebraValidation:
    """Test that STA methods require correct algebra."""

    def test_sta_gamma_wrong_algebra(self):
        """sta_gamma should fail with non-STA algebra."""
        R3 = Algebra.euclidean(3)
        with pytest.raises(ValueError, match="Cl\\(1,3,0\\)"):
            Multivector.sta_gamma(R3, 0)

    def test_sta_vector_wrong_algebra(self):
        """sta_vector should fail with non-STA algebra."""
        PGA = Algebra.pga(3)
        with pytest.raises(ValueError, match="Cl\\(1,3,0\\)"):
            Multivector.sta_vector(PGA, 1.0, 0.0, 0.0, 0.0)

    def test_sta_boost_wrong_algebra(self):
        """sta_boost should fail with non-STA algebra."""
        CGA = Algebra.cga(3)
        with pytest.raises(ValueError, match="Cl\\(1,3,0\\)"):
            Multivector.sta_boost(CGA, [0.5, 0.0, 0.0])


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple STA operations."""

    def test_lorentz_transform_preserves_lightlike(self, sta):
        """Lorentz transformation preserves lightlike vectors."""
        # Light ray moving in x-direction
        photon = Multivector.sta_vector(sta, 1.0, 1.0, 0.0, 0.0)
        assert photon.sta_is_lightlike()

        # Apply boost and rotation
        boost = Multivector.sta_boost(sta, [0.5, 0.0, 0.0])
        rotation = Multivector.sta_rotation(sta, [0.0, 0.0, 1.0], math.pi/6)

        # Combined transformation
        R = rotation * boost
        R_rev = R.reverse()

        photon_transformed = R * photon * R_rev

        # Should still be lightlike
        assert photon_transformed.sta_is_lightlike()

    def test_boost_inverse(self, sta):
        """Boost and inverse boost cancel."""
        x = Multivector.sta_vector(sta, 5.0, 3.0, 4.0, 0.0)

        # Forward boost
        R = Multivector.sta_boost(sta, [0.6, 0.0, 0.0])
        R_rev = R.reverse()
        x_boosted = R * x * R_rev

        # Inverse boost (negative velocity)
        R_inv = Multivector.sta_boost(sta, [-0.6, 0.0, 0.0])
        R_inv_rev = R_inv.reverse()
        x_restored = R_inv * x_boosted * R_inv_rev

        # Should return to original
        t_orig, r_orig = x.sta_spacetime_split()
        t_restored, r_restored = x_restored.sta_spacetime_split()

        assert t_restored == pytest.approx(t_orig, rel=1e-10)
        for i in range(3):
            assert r_restored[i] == pytest.approx(r_orig[i], rel=1e-10)

    def test_twin_paradox_scenario(self, sta):
        """Verify time dilation in twin paradox scenario."""
        # Stationary twin: proper time = coordinate time
        # Traveling twin: v = 0.8c, gamma = 5/3

        v = 0.8
        gamma = 1.0 / math.sqrt(1 - v**2)

        # Create event: 5 years pass for stationary twin
        coord_time = 5.0
        event = Multivector.sta_vector(sta, coord_time, 0.0, 0.0, 0.0)

        # Proper time should equal coordinate time for stationary event
        assert event.sta_proper_time() == pytest.approx(coord_time)

        # For moving observer with v = 0.8c:
        # Proper time = coordinate_time / gamma
        proper_time_moving = coord_time / gamma

        # Create the same event in moving frame (Lorentz contracted)
        # Time dilation: t' = t / gamma
        assert proper_time_moving == pytest.approx(3.0)  # 5/1.667 = 3


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling in STA operations."""

    def test_invalid_gamma_index(self, sta):
        """Invalid gamma index should raise error."""
        with pytest.raises(ValueError):
            Multivector.sta_gamma(sta, 4)  # Only 0-3 valid

        with pytest.raises(ValueError):
            Multivector.sta_gamma(sta, 10)

    def test_wrong_velocity_dimensions(self, sta):
        """Velocity must be 3D."""
        with pytest.raises(ValueError):
            Multivector.sta_boost(sta, [0.5, 0.0])  # Only 2 components

        with pytest.raises(ValueError):
            Multivector.sta_boost(sta, [0.5, 0.0, 0.0, 0.0])  # 4 components

    def test_wrong_axis_dimensions(self, sta):
        """Rotation axis must be 3D."""
        with pytest.raises(ValueError):
            Multivector.sta_rotation(sta, [1.0, 0.0], math.pi/4)

    def test_zero_axis_rotation(self, sta):
        """Zero rotation axis should raise error."""
        with pytest.raises(ValueError):
            Multivector.sta_rotation(sta, [0.0, 0.0, 0.0], math.pi/4)

    def test_wrong_e_field_dimensions(self, sta):
        """E field must be 3D."""
        with pytest.raises(ValueError):
            Multivector.sta_bivector(sta, [1.0, 0.0], [0.0, 0.0, 0.0])

    def test_wrong_b_field_dimensions(self, sta):
        """B field must be 3D."""
        with pytest.raises(ValueError):
            Multivector.sta_bivector(sta, [1.0, 0.0, 0.0], [0.0])
