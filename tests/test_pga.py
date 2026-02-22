"""Tests for Projective Geometric Algebra (PGA) convenience methods.

PGA3D is Cl(3,0,1) - 3 Euclidean basis vectors + 1 degenerate basis vector.
Used for robotics, graphics, and kinematics.

References:
- Geometric Algebra for Computer Science (Dorst, Fontijne, Mann)
- ganja.js by Steven De Keninck
- bivector.net PGA tutorials
"""

import pytest
import math
import largecrimsoncanine as lcc
from largecrimsoncanine import Algebra, Multivector


class TestPGAPointCreation:
    """Test PGA point creation and normalization."""

    @pytest.fixture
    def pga3d(self):
        """Create PGA3D algebra."""
        return Algebra.pga(3)

    def test_origin_point(self, pga3d):
        """Test creating point at origin."""
        origin = Multivector.pga_point(pga3d, 0.0, 0.0, 0.0)
        x, y, z = origin.pga_point_coords()
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(0.0)

    def test_point_coordinates(self, pga3d):
        """Test point with various coordinates."""
        p = Multivector.pga_point(pga3d, 1.0, 2.0, 3.0)
        x, y, z = p.pga_point_coords()
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(2.0)
        assert z == pytest.approx(3.0)

    def test_negative_coordinates(self, pga3d):
        """Test point with negative coordinates."""
        p = Multivector.pga_point(pga3d, -5.0, -10.0, -15.0)
        x, y, z = p.pga_point_coords()
        assert x == pytest.approx(-5.0)
        assert y == pytest.approx(-10.0)
        assert z == pytest.approx(-15.0)

    def test_point_normalization(self, pga3d):
        """Test normalizing a scaled point."""
        p = Multivector.pga_point(pga3d, 1.0, 2.0, 3.0)
        p_scaled = p * 5.0  # Scale the point
        p_norm = p_scaled.pga_normalize()

        # Coordinates should be unchanged after normalization
        x, y, z = p_norm.pga_point_coords()
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(2.0)
        assert z == pytest.approx(3.0)

    def test_point_is_trivector(self, pga3d):
        """Test that points are grade-3 elements."""
        p = Multivector.pga_point(pga3d, 1.0, 2.0, 3.0)
        grades = p.grades()
        assert 3 in grades

    def test_invalid_algebra_raises(self):
        """Test that non-PGA algebra raises error."""
        R3 = Algebra.euclidean(3)
        with pytest.raises(ValueError, match="PGA3D"):
            Multivector.pga_point(R3, 0.0, 0.0, 0.0)


class TestPGAPlaneCreation:
    """Test PGA plane creation."""

    @pytest.fixture
    def pga3d(self):
        return Algebra.pga(3)

    def test_xy_plane(self, pga3d):
        """Test creating the XY plane (z = 0)."""
        xy = Multivector.pga_plane(pga3d, 0.0, 0.0, 1.0, 0.0)
        # Should be a vector (grade 1)
        assert xy.pure_grade() == 1

    def test_offset_plane(self, pga3d):
        """Test plane with offset."""
        # Plane z = 5, i.e., z - 5 = 0
        plane = Multivector.pga_plane(pga3d, 0.0, 0.0, 1.0, -5.0)
        # Check the e0 coefficient represents the offset
        coords = plane.to_vector_coords()
        assert coords[3] == pytest.approx(-5.0)  # e0 coefficient

    def test_arbitrary_plane(self, pga3d):
        """Test arbitrary plane equation."""
        # 2x + 3y - z + 4 = 0
        plane = Multivector.pga_plane(pga3d, 2.0, 3.0, -1.0, 4.0)
        coords = plane.to_vector_coords()
        assert coords[0] == pytest.approx(2.0)   # e1
        assert coords[1] == pytest.approx(3.0)   # e2
        assert coords[2] == pytest.approx(-1.0)  # e3
        assert coords[3] == pytest.approx(4.0)   # e0


class TestPGALineFromPoints:
    """Test creating lines from two points."""

    @pytest.fixture
    def pga3d(self):
        return Algebra.pga(3)

    def test_x_axis_line(self, pga3d):
        """Test line along X-axis through origin."""
        p1 = Multivector.pga_point(pga3d, 0.0, 0.0, 0.0)
        p2 = Multivector.pga_point(pga3d, 1.0, 0.0, 0.0)
        line = Multivector.pga_line_from_points(p1, p2)

        # Line should be a bivector (grade 2)
        assert line.pure_grade() == 2

    def test_diagonal_line(self, pga3d):
        """Test line from origin to (1,1,1)."""
        p1 = Multivector.pga_point(pga3d, 0.0, 0.0, 0.0)
        p2 = Multivector.pga_point(pga3d, 1.0, 1.0, 1.0)
        line = Multivector.pga_line_from_points(p1, p2)

        # Line should be a bivector
        assert line.pure_grade() == 2

    def test_offset_line(self, pga3d):
        """Test line not passing through origin."""
        p1 = Multivector.pga_point(pga3d, 1.0, 0.0, 0.0)
        p2 = Multivector.pga_point(pga3d, 1.0, 1.0, 0.0)
        line = Multivector.pga_line_from_points(p1, p2)

        # Should still be a bivector
        assert line.pure_grade() == 2


class TestPGALineFromPlanes:
    """Test creating lines from plane intersections."""

    @pytest.fixture
    def pga3d(self):
        return Algebra.pga(3)

    def test_x_axis_from_planes(self, pga3d):
        """Test X-axis as intersection of XY and XZ planes."""
        xy = Multivector.pga_plane(pga3d, 0.0, 0.0, 1.0, 0.0)  # z = 0
        xz = Multivector.pga_plane(pga3d, 0.0, 1.0, 0.0, 0.0)  # y = 0
        line = Multivector.pga_line_from_planes(xy, xz)

        # Should be a bivector
        assert line.pure_grade() == 2

    def test_arbitrary_intersection(self, pga3d):
        """Test intersection of two arbitrary planes."""
        p1 = Multivector.pga_plane(pga3d, 1.0, 0.0, 0.0, 0.0)  # x = 0 (YZ plane)
        p2 = Multivector.pga_plane(pga3d, 0.0, 1.0, 0.0, 0.0)  # y = 0 (XZ plane)
        line = Multivector.pga_line_from_planes(p1, p2)

        # This is the Z-axis
        assert line.pure_grade() == 2


class TestPGAPointPlaneDistance:
    """Test point-plane distance calculations."""

    @pytest.fixture
    def pga3d(self):
        return Algebra.pga(3)

    def test_point_on_plane(self, pga3d):
        """Test distance is zero for point on plane."""
        p = Multivector.pga_point(pga3d, 1.0, 2.0, 0.0)
        xy = Multivector.pga_plane(pga3d, 0.0, 0.0, 1.0, 0.0)  # z = 0

        dist = p.pga_point_plane_distance(xy)
        assert dist == pytest.approx(0.0)

    def test_point_above_plane(self, pga3d):
        """Test distance for point above plane."""
        p = Multivector.pga_point(pga3d, 0.0, 0.0, 5.0)
        xy = Multivector.pga_plane(pga3d, 0.0, 0.0, 1.0, 0.0)  # z = 0

        dist = p.pga_point_plane_distance(xy)
        assert dist == pytest.approx(5.0)

    def test_point_below_plane(self, pga3d):
        """Test negative distance for point below plane."""
        p = Multivector.pga_point(pga3d, 0.0, 0.0, -3.0)
        xy = Multivector.pga_plane(pga3d, 0.0, 0.0, 1.0, 0.0)  # z = 0

        dist = p.pga_point_plane_distance(xy)
        assert dist == pytest.approx(-3.0)

    def test_offset_plane_distance(self, pga3d):
        """Test distance to offset plane."""
        p = Multivector.pga_point(pga3d, 0.0, 0.0, 10.0)
        # Plane z = 7
        plane = Multivector.pga_plane(pga3d, 0.0, 0.0, 1.0, -7.0)

        dist = p.pga_point_plane_distance(plane)
        assert dist == pytest.approx(3.0)

    def test_arbitrary_plane_distance(self, pga3d):
        """Test distance to non-axis-aligned plane."""
        # Plane x + y + z = 0 (normalized: divide by sqrt(3))
        p = Multivector.pga_point(pga3d, 1.0, 1.0, 1.0)
        plane = Multivector.pga_plane(pga3d, 1.0, 1.0, 1.0, 0.0)

        # Distance = (1+1+1)/sqrt(3) = 3/sqrt(3) = sqrt(3)
        dist = p.pga_point_plane_distance(plane)
        assert dist == pytest.approx(math.sqrt(3))


class TestPGATranslator:
    """Test PGA translation motors."""

    @pytest.fixture
    def pga3d(self):
        return Algebra.pga(3)

    def test_translator_is_motor(self, pga3d):
        """Test translator is a valid motor."""
        T = Multivector.pga_translator(pga3d, 1.0, 2.0, 3.0)
        assert T.is_pga_motor()

    def test_translate_point(self, pga3d):
        """Test translating a point."""
        T = Multivector.pga_translator(pga3d, 1.0, 0.0, 0.0)
        p = Multivector.pga_point(pga3d, 0.0, 0.0, 0.0)

        # Apply translation: T * p * ~T
        p_moved = T.sandwich(p)

        # Extract coordinates
        x, y, z = p_moved.pga_point_coords()
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(0.0)

    def test_translate_arbitrary_point(self, pga3d):
        """Test translating point from (1,2,3) by (4,5,6)."""
        T = Multivector.pga_translator(pga3d, 4.0, 5.0, 6.0)
        p = Multivector.pga_point(pga3d, 1.0, 2.0, 3.0)

        p_moved = T.sandwich(p)
        x, y, z = p_moved.pga_point_coords()

        assert x == pytest.approx(5.0)  # 1 + 4
        assert y == pytest.approx(7.0)  # 2 + 5
        assert z == pytest.approx(9.0)  # 3 + 6

    def test_identity_translation(self, pga3d):
        """Test zero translation leaves point unchanged."""
        T = Multivector.pga_translator(pga3d, 0.0, 0.0, 0.0)
        p = Multivector.pga_point(pga3d, 7.0, 8.0, 9.0)

        p_moved = T.sandwich(p)
        x, y, z = p_moved.pga_point_coords()

        assert x == pytest.approx(7.0)
        assert y == pytest.approx(8.0)
        assert z == pytest.approx(9.0)


class TestPGAMotorFromAxisAngle:
    """Test PGA rotation motors."""

    @pytest.fixture
    def pga3d(self):
        return Algebra.pga(3)

    def test_motor_is_valid(self, pga3d):
        """Test motor from axis-angle is valid."""
        motor = Multivector.pga_motor_from_axis_angle(
            pga3d,
            [0.0, 0.0, 1.0],  # Z-axis
            [0.0, 0.0, 0.0],  # Through origin
            math.pi / 2      # 90 degrees
        )
        assert motor.is_pga_motor()

    def test_rotation_90_degrees_z(self, pga3d):
        """Test 90 degree rotation about Z-axis."""
        motor = Multivector.pga_motor_from_axis_angle(
            pga3d,
            [0.0, 0.0, 1.0],  # Z-axis
            [0.0, 0.0, 0.0],  # Through origin
            math.pi / 2      # 90 degrees
        )

        # Rotate point (1, 0, 0) should give (0, 1, 0)
        p = Multivector.pga_point(pga3d, 1.0, 0.0, 0.0)
        p_rot = motor.sandwich(p)

        x, y, z = p_rot.pga_point_coords()
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(1.0)
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_rotation_180_degrees(self, pga3d):
        """Test 180 degree rotation."""
        motor = Multivector.pga_motor_from_axis_angle(
            pga3d,
            [0.0, 0.0, 1.0],  # Z-axis
            [0.0, 0.0, 0.0],  # Through origin
            math.pi          # 180 degrees
        )

        # Rotate point (1, 0, 0) should give (-1, 0, 0)
        p = Multivector.pga_point(pga3d, 1.0, 0.0, 0.0)
        p_rot = motor.sandwich(p)

        x, y, z = p_rot.pga_point_coords()
        assert x == pytest.approx(-1.0)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_rotation_about_x_axis(self, pga3d):
        """Test rotation about X-axis."""
        motor = Multivector.pga_motor_from_axis_angle(
            pga3d,
            [1.0, 0.0, 0.0],  # X-axis
            [0.0, 0.0, 0.0],  # Through origin
            math.pi / 2      # 90 degrees
        )

        # Rotate (0, 1, 0) should give (0, 0, 1)
        p = Multivector.pga_point(pga3d, 0.0, 1.0, 0.0)
        p_rot = motor.sandwich(p)

        x, y, z = p_rot.pga_point_coords()
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert z == pytest.approx(1.0)

    def test_identity_rotation(self, pga3d):
        """Test zero rotation leaves point unchanged."""
        motor = Multivector.pga_motor_from_axis_angle(
            pga3d,
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.0  # No rotation
        )

        p = Multivector.pga_point(pga3d, 5.0, 6.0, 7.0)
        p_rot = motor.sandwich(p)

        x, y, z = p_rot.pga_point_coords()
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(6.0)
        assert z == pytest.approx(7.0)


class TestPGAMotorComposition:
    """Test composing PGA motors."""

    @pytest.fixture
    def pga3d(self):
        return Algebra.pga(3)

    def test_compose_translations(self, pga3d):
        """Test composing two translations."""
        T1 = Multivector.pga_translator(pga3d, 1.0, 0.0, 0.0)
        T2 = Multivector.pga_translator(pga3d, 0.0, 2.0, 0.0)

        # Compose: T2 * T1 (applied right to left)
        T_composed = T2 * T1

        # Apply to origin
        p = Multivector.pga_point(pga3d, 0.0, 0.0, 0.0)
        p_moved = T_composed.sandwich(p)

        x, y, z = p_moved.pga_point_coords()
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(2.0)
        assert z == pytest.approx(0.0)

    def test_compose_rotations(self, pga3d):
        """Test composing two rotations."""
        # Two 90-degree rotations about Z-axis = 180 degrees
        R1 = Multivector.pga_motor_from_axis_angle(
            pga3d, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], math.pi / 2
        )
        R2 = Multivector.pga_motor_from_axis_angle(
            pga3d, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], math.pi / 2
        )

        R_composed = R2 * R1

        # Apply to (1, 0, 0) should give (-1, 0, 0)
        p = Multivector.pga_point(pga3d, 1.0, 0.0, 0.0)
        p_rot = R_composed.sandwich(p)

        x, y, z = p_rot.pga_point_coords()
        assert x == pytest.approx(-1.0)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_rotation_then_translation(self, pga3d):
        """Test rotation followed by translation."""
        # 90 degrees about Z through origin
        R = Multivector.pga_motor_from_axis_angle(
            pga3d, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], math.pi / 2
        )
        # Translate by (1, 0, 0)
        T = Multivector.pga_translator(pga3d, 1.0, 0.0, 0.0)

        # T * R: first rotate, then translate
        M = T * R

        # Start at (1, 0, 0):
        # After rotation: (0, 1, 0)
        # After translation: (1, 1, 0)
        p = Multivector.pga_point(pga3d, 1.0, 0.0, 0.0)
        p_moved = M.sandwich(p)

        x, y, z = p_moved.pga_point_coords()
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(1.0)
        assert z == pytest.approx(0.0, abs=1e-10)


class TestPGAIdealAndEuclideanParts:
    """Test extracting ideal and Euclidean parts."""

    @pytest.fixture
    def pga3d(self):
        return Algebra.pga(3)

    def test_plane_ideal_part(self, pga3d):
        """Test extracting ideal part of a plane."""
        plane = Multivector.pga_plane(pga3d, 1.0, 2.0, 3.0, 5.0)
        ideal = plane.pga_ideal_part()

        # Only e0 component should remain
        coords = ideal.to_vector_coords()
        assert coords[0] == pytest.approx(0.0)  # e1
        assert coords[1] == pytest.approx(0.0)  # e2
        assert coords[2] == pytest.approx(0.0)  # e3
        assert coords[3] == pytest.approx(5.0)  # e0

    def test_plane_euclidean_part(self, pga3d):
        """Test extracting Euclidean part of a plane."""
        plane = Multivector.pga_plane(pga3d, 1.0, 2.0, 3.0, 5.0)
        eucl = plane.pga_euclidean_part()

        # Only e1, e2, e3 components should remain
        coords = eucl.to_vector_coords()
        assert coords[0] == pytest.approx(1.0)  # e1
        assert coords[1] == pytest.approx(2.0)  # e2
        assert coords[2] == pytest.approx(3.0)  # e3
        assert coords[3] == pytest.approx(0.0)  # e0

    def test_translator_parts(self, pga3d):
        """Test parts of a translator."""
        T = Multivector.pga_translator(pga3d, 2.0, 4.0, 6.0)

        # Euclidean part should be scalar 1
        eucl = T.pga_euclidean_part()
        assert eucl.scalar() == pytest.approx(1.0)

        # Ideal part should have the translation bivectors
        ideal = T.pga_ideal_part()
        # Should have e01, e02, e03 components
        blades = ideal.blades()
        # Check grade 2 components exist
        assert any(g == 2 for _, _, g in blades)


class TestPGAMotorValidation:
    """Test motor validation."""

    @pytest.fixture
    def pga3d(self):
        return Algebra.pga(3)

    def test_translator_is_motor(self, pga3d):
        """Test translator passes motor check."""
        T = Multivector.pga_translator(pga3d, 1.0, 2.0, 3.0)
        assert T.is_pga_motor()

    def test_rotor_is_motor(self, pga3d):
        """Test rotation motor passes motor check."""
        R = Multivector.pga_motor_from_axis_angle(
            pga3d, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], math.pi / 4
        )
        assert R.is_pga_motor()

    def test_point_is_not_motor(self, pga3d):
        """Test point is not a motor."""
        p = Multivector.pga_point(pga3d, 1.0, 2.0, 3.0)
        assert not p.is_pga_motor()

    def test_plane_is_not_motor(self, pga3d):
        """Test plane is not a motor."""
        plane = Multivector.pga_plane(pga3d, 1.0, 0.0, 0.0, 0.0)
        assert not plane.is_pga_motor()

    def test_zero_is_not_motor(self, pga3d):
        """Test zero is not a motor."""
        zero = Multivector.zero_in(pga3d)
        assert not zero.is_pga_motor()


class TestPGARegression:
    """Regression tests for PGA operations."""

    @pytest.fixture
    def pga3d(self):
        return Algebra.pga(3)

    def test_point_roundtrip(self, pga3d):
        """Test creating and extracting point coordinates."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 2.5, -3.7),
                  (100.0, -200.0, 300.0), (0.001, 0.002, 0.003)]

        for x_in, y_in, z_in in coords:
            p = Multivector.pga_point(pga3d, x_in, y_in, z_in)
            x_out, y_out, z_out = p.pga_point_coords()

            assert x_out == pytest.approx(x_in)
            assert y_out == pytest.approx(y_in)
            assert z_out == pytest.approx(z_in)

    def test_dual_construction(self, pga3d):
        """Test that dual constructions are consistent."""
        # Line from two points
        p1 = Multivector.pga_point(pga3d, 0.0, 0.0, 0.0)
        p2 = Multivector.pga_point(pga3d, 1.0, 0.0, 0.0)
        line1 = Multivector.pga_line_from_points(p1, p2)

        # Line from two planes (Y=0 and Z=0 both contain X-axis)
        xz = Multivector.pga_plane(pga3d, 0.0, 1.0, 0.0, 0.0)  # y = 0
        xy = Multivector.pga_plane(pga3d, 0.0, 0.0, 1.0, 0.0)  # z = 0
        line2 = Multivector.pga_line_from_planes(xz, xy)

        # Both should be bivectors representing the X-axis
        assert line1.pure_grade() == 2
        assert line2.pure_grade() == 2

        # They should be proportional (same line, possibly different scale)
        # Normalize and compare
        n1 = line1.norm()
        n2 = line2.norm()
        if n1 > 1e-10 and n2 > 1e-10:
            line1_norm = line1 * (1.0 / n1)
            line2_norm = line2 * (1.0 / n2)
            # Either equal or opposite sign
            diff1 = (line1_norm - line2_norm).norm()
            diff2 = (line1_norm + line2_norm).norm()
            assert min(diff1, diff2) < 0.1  # Allow some tolerance

    def test_reflection_preservation(self, pga3d):
        """Test that double reflection returns to original."""
        # Reflect point through XY plane twice
        p = Multivector.pga_point(pga3d, 1.0, 2.0, 3.0)
        xy = Multivector.pga_plane(pga3d, 0.0, 0.0, 1.0, 0.0)

        # Reflection: p' = -xy * p * xy (plane acts as versor)
        p_ref1 = -xy * p * xy
        p_ref2 = -xy * p_ref1 * xy

        # Should be back at original (approximately)
        x1, y1, z1 = p.pga_point_coords()
        x2, y2, z2 = p_ref2.pga_normalize().pga_point_coords()

        assert x2 == pytest.approx(x1, abs=1e-10)
        assert y2 == pytest.approx(y1, abs=1e-10)
        assert z2 == pytest.approx(z1, abs=1e-10)
