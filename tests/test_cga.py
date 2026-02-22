"""Tests for Conformal Geometric Algebra (CGA) convenience methods.

CGA embeds Euclidean space conformally using the null vectors eo (origin) and
einf (infinity). This allows spheres, circles, and planes to be represented
as simple blades.

Reference: Dorst "Geometric Algebra for Computer Science" Chapter 13.
"""

import pytest
import math
import largecrimsoncanine as lcc
from largecrimsoncanine import Algebra, Multivector


class TestCGANullVectors:
    """Test the CGA null vector constructors."""

    def test_cga_eo_creation(self):
        """Test creating the origin null vector eo."""
        CGA3D = Algebra.cga(3)
        eo = Multivector.cga_eo(CGA3D)

        # eo should be a vector (grade 1)
        assert eo.max_grade() == 1

    def test_cga_einf_creation(self):
        """Test creating the infinity null vector einf."""
        CGA3D = Algebra.cga(3)
        einf = Multivector.cga_einf(CGA3D)

        # einf should be a vector (grade 1)
        assert einf.max_grade() == 1

    def test_eo_is_null(self):
        """Test that eo squares to zero (null vector)."""
        CGA3D = Algebra.cga(3)
        eo = Multivector.cga_eo(CGA3D)

        # eo * eo should be 0 (scalar)
        eo_sq = eo * eo
        assert abs(eo_sq.scalar()) < 1e-10

    def test_einf_is_null(self):
        """Test that einf squares to zero (null vector)."""
        CGA3D = Algebra.cga(3)
        einf = Multivector.cga_einf(CGA3D)

        # einf * einf should be 0 (scalar)
        einf_sq = einf * einf
        assert abs(einf_sq.scalar()) < 1e-10

    def test_eo_einf_inner_product(self):
        """Test that eo . einf = -1."""
        CGA3D = Algebra.cga(3)
        eo = Multivector.cga_eo(CGA3D)
        einf = Multivector.cga_einf(CGA3D)

        # eo . einf should be -1
        eo_dot_einf = eo.inner(einf)
        assert abs(eo_dot_einf.scalar() + 1.0) < 1e-10

    def test_cga_algebra_required(self):
        """Test that CGA methods require a CGA algebra."""
        R3 = Algebra.euclidean(3)
        with pytest.raises(ValueError, match="CGA algebra"):
            Multivector.cga_eo(R3)

        with pytest.raises(ValueError, match="CGA algebra"):
            Multivector.cga_einf(R3)


class TestCGAPoints:
    """Test CGA point embedding."""

    def test_origin_point(self):
        """Test that origin point equals eo."""
        CGA3D = Algebra.cga(3)
        origin = Multivector.cga_point(CGA3D, 0.0, 0.0, 0.0)
        eo = Multivector.cga_eo(CGA3D)

        # Origin point should equal eo
        diff = origin - eo
        assert diff.norm() < 1e-10

    def test_point_embedding_structure(self):
        """Test that point has correct structure: x + 0.5*|x|^2*einf + eo."""
        CGA3D = Algebra.cga(3)
        P = Multivector.cga_point(CGA3D, 1.0, 0.0, 0.0)

        # Should have e1 coefficient = 1.0
        coeffs = P.coefficients()
        assert abs(coeffs[1] - 1.0) < 1e-10  # e1 = 1
        assert abs(coeffs[2]) < 1e-10  # e2 = 0
        assert abs(coeffs[4]) < 1e-10  # e3 = 0

    def test_point_from_coords(self):
        """Test point creation from coordinate list."""
        CGA2D = Algebra.cga(2)
        P = Multivector.cga_point_from_coords(CGA2D, [3.0, 4.0])

        # Should have correct Euclidean components
        coeffs = P.coefficients()
        assert abs(coeffs[1] - 3.0) < 1e-10  # e1
        assert abs(coeffs[2] - 4.0) < 1e-10  # e2

    def test_point_from_coords_wrong_dimension(self):
        """Test error when coords don't match dimension."""
        CGA3D = Algebra.cga(3)
        with pytest.raises(ValueError, match="coords length"):
            Multivector.cga_point_from_coords(CGA3D, [1.0, 2.0])  # Need 3 coords


class TestCGASpheres:
    """Test CGA sphere representation."""

    def test_sphere_creation(self):
        """Test basic sphere creation."""
        CGA3D = Algebra.cga(3)
        S = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)

        # Sphere should be a vector (grade 1)
        assert S.max_grade() == 1

    def test_sphere_center_extraction(self):
        """Test extracting center from sphere."""
        CGA3D = Algebra.cga(3)
        S = Multivector.cga_sphere(CGA3D, 1.0, 2.0, 3.0, 5.0)

        center = S.cga_extract_center()
        assert abs(center[0] - 1.0) < 1e-10
        assert abs(center[1] - 2.0) < 1e-10
        assert abs(center[2] - 3.0) < 1e-10

    def test_sphere_radius_extraction(self):
        """Test extracting radius from sphere."""
        CGA3D = Algebra.cga(3)
        S = Multivector.cga_sphere(CGA3D, 1.0, 2.0, 3.0, 5.0)

        radius = S.cga_extract_radius()
        assert abs(radius - 5.0) < 1e-10

    def test_unit_sphere_at_origin(self):
        """Test unit sphere centered at origin."""
        CGA3D = Algebra.cga(3)
        S = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)

        center = S.cga_extract_center()
        radius = S.cga_extract_radius()

        assert abs(center[0]) < 1e-10
        assert abs(center[1]) < 1e-10
        assert abs(center[2]) < 1e-10
        assert abs(radius - 1.0) < 1e-10

    def test_point_as_zero_radius_sphere(self):
        """Test that a point can be seen as a zero-radius sphere."""
        CGA3D = Algebra.cga(3)
        P = Multivector.cga_point(CGA3D, 1.0, 2.0, 3.0)
        S = Multivector.cga_sphere(CGA3D, 1.0, 2.0, 3.0, 0.0)

        # Point and zero-radius sphere should be the same
        diff = P - S
        assert diff.norm() < 1e-10


class TestCGAPlanes:
    """Test CGA plane representation."""

    def test_plane_creation(self):
        """Test basic plane creation."""
        CGA3D = Algebra.cga(3)
        xy_plane = Multivector.cga_plane(CGA3D, 0.0, 0.0, 1.0, 0.0)

        # Plane should be a vector (grade 1)
        assert xy_plane.max_grade() == 1

    def test_plane_normal_components(self):
        """Test that plane has correct normal components."""
        CGA3D = Algebra.cga(3)
        plane = Multivector.cga_plane(CGA3D, 1.0, 2.0, 3.0, 4.0)

        coeffs = plane.coefficients()
        assert abs(coeffs[1] - 1.0) < 1e-10  # e1 (a)
        assert abs(coeffs[2] - 2.0) < 1e-10  # e2 (b)
        assert abs(coeffs[4] - 3.0) < 1e-10  # e3 (c)

    def test_plane_is_flat(self):
        """Test that plane is identified as flat."""
        CGA3D = Algebra.cga(3)
        plane = Multivector.cga_plane(CGA3D, 0.0, 0.0, 1.0, 0.0)

        assert plane.cga_is_flat()


class TestCGACircles:
    """Test CGA circle representation."""

    def test_circle_from_sphere_intersection(self):
        """Test creating a circle from two intersecting spheres."""
        CGA3D = Algebra.cga(3)

        # Two unit spheres, one at origin, one at (1, 0, 0)
        S1 = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)
        S2 = Multivector.cga_sphere(CGA3D, 1.0, 0.0, 0.0, 1.0)

        # Circle is the wedge product of two spheres
        C = Multivector.cga_circle(S1, S2)

        # Circle should be a bivector (grade 2)
        assert C.max_grade() == 2


class TestCGAPointPairs:
    """Test CGA point pair representation."""

    def test_point_pair_creation(self):
        """Test creating a point pair."""
        CGA3D = Algebra.cga(3)
        P1 = Multivector.cga_point(CGA3D, 0.0, 0.0, 0.0)
        P2 = Multivector.cga_point(CGA3D, 1.0, 0.0, 0.0)

        PP = Multivector.cga_point_pair(P1, P2)

        # Point pair should be a bivector (grade 2)
        assert PP.max_grade() == 2


class TestCGALines:
    """Test CGA line representation."""

    def test_line_from_plane_intersection(self):
        """Test creating a line from two intersecting planes."""
        CGA3D = Algebra.cga(3)

        # XY plane (z = 0) and XZ plane (y = 0) intersect at X-axis
        xy_plane = Multivector.cga_plane(CGA3D, 0.0, 0.0, 1.0, 0.0)
        xz_plane = Multivector.cga_plane(CGA3D, 0.0, 1.0, 0.0, 0.0)

        x_axis = Multivector.cga_line(xy_plane, xz_plane)

        # Line should be a bivector (grade 2)
        assert x_axis.max_grade() == 2


class TestCGAPointOnSphere:
    """Test point-on-sphere checking."""

    def test_point_on_unit_sphere(self):
        """Test that points on unit sphere are detected."""
        CGA3D = Algebra.cga(3)
        S = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)

        # Point at (1, 0, 0) should be on the unit sphere
        P_on = Multivector.cga_point(CGA3D, 1.0, 0.0, 0.0)
        assert P_on.cga_point_on_sphere(S)

        # Point at (0, 1, 0) should also be on the unit sphere
        P_on2 = Multivector.cga_point(CGA3D, 0.0, 1.0, 0.0)
        assert P_on2.cga_point_on_sphere(S)

    def test_point_inside_sphere(self):
        """Test that points inside sphere are not detected as on sphere."""
        CGA3D = Algebra.cga(3)
        S = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)

        # Point at (0.5, 0, 0) should be inside the unit sphere
        P_inside = Multivector.cga_point(CGA3D, 0.5, 0.0, 0.0)
        assert not P_inside.cga_point_on_sphere(S)

    def test_point_outside_sphere(self):
        """Test that points outside sphere are not detected as on sphere."""
        CGA3D = Algebra.cga(3)
        S = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)

        # Point at (2, 0, 0) should be outside the unit sphere
        P_outside = Multivector.cga_point(CGA3D, 2.0, 0.0, 0.0)
        assert not P_outside.cga_point_on_sphere(S)


class TestCGADistance:
    """Test CGA distance calculations."""

    def test_distance_origin_to_point(self):
        """Test distance from origin to a point."""
        CGA3D = Algebra.cga(3)
        P1 = Multivector.cga_point(CGA3D, 0.0, 0.0, 0.0)
        P2 = Multivector.cga_point(CGA3D, 3.0, 4.0, 0.0)

        d = P1.cga_distance(P2)
        assert abs(d - 5.0) < 1e-10  # 3-4-5 triangle

    def test_distance_between_points(self):
        """Test distance between two arbitrary points."""
        CGA3D = Algebra.cga(3)
        P1 = Multivector.cga_point(CGA3D, 1.0, 2.0, 3.0)
        P2 = Multivector.cga_point(CGA3D, 4.0, 6.0, 3.0)

        d = P1.cga_distance(P2)
        expected = math.sqrt(9 + 16)  # sqrt((4-1)^2 + (6-2)^2 + (3-3)^2)
        assert abs(d - expected) < 1e-10

    def test_distance_is_symmetric(self):
        """Test that distance is symmetric: d(P1, P2) = d(P2, P1)."""
        CGA3D = Algebra.cga(3)
        P1 = Multivector.cga_point(CGA3D, 1.0, 2.0, 3.0)
        P2 = Multivector.cga_point(CGA3D, 4.0, 5.0, 6.0)

        d12 = P1.cga_distance(P2)
        d21 = P2.cga_distance(P1)
        assert abs(d12 - d21) < 1e-10

    def test_distance_to_self_is_zero(self):
        """Test that distance from a point to itself is zero."""
        CGA3D = Algebra.cga(3)
        P = Multivector.cga_point(CGA3D, 1.0, 2.0, 3.0)

        d = P.cga_distance(P)
        assert abs(d) < 1e-10


class TestCGARoundFlat:
    """Test classification of CGA elements as round or flat."""

    def test_sphere_is_round(self):
        """Test that sphere is classified as round."""
        CGA3D = Algebra.cga(3)
        S = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)

        assert S.cga_is_round()
        assert not S.cga_is_flat()

    def test_plane_is_flat(self):
        """Test that plane is classified as flat."""
        CGA3D = Algebra.cga(3)
        plane = Multivector.cga_plane(CGA3D, 0.0, 0.0, 1.0, 0.0)

        assert plane.cga_is_flat()


class TestCGADimensions:
    """Test CGA with different Euclidean dimensions."""

    def test_cga2d(self):
        """Test CGA with 2D Euclidean space."""
        CGA2D = Algebra.cga(2)
        assert CGA2D.dimension == 4  # 2 Euclidean + e+ + e-
        assert CGA2D.p == 3
        assert CGA2D.q == 1

        # Create point in 2D
        P = Multivector.cga_point_from_coords(CGA2D, [1.0, 2.0])
        assert P.max_grade() == 1

    def test_cga4d(self):
        """Test CGA with 4D Euclidean space."""
        CGA4D = Algebra.cga(4)
        assert CGA4D.dimension == 6  # 4 Euclidean + e+ + e-
        assert CGA4D.p == 5
        assert CGA4D.q == 1

        # Create point in 4D
        P = Multivector.cga_point_from_coords(CGA4D, [1.0, 2.0, 3.0, 4.0])
        assert P.max_grade() == 1


class TestCGAIntegration:
    """Integration tests combining multiple CGA operations."""

    def test_sphere_through_four_points(self):
        """Test that a sphere passes through four non-coplanar points."""
        CGA3D = Algebra.cga(3)

        # Four points on a unit sphere
        P1 = Multivector.cga_point(CGA3D, 1.0, 0.0, 0.0)
        P2 = Multivector.cga_point(CGA3D, 0.0, 1.0, 0.0)
        P3 = Multivector.cga_point(CGA3D, 0.0, 0.0, 1.0)
        P4 = Multivector.cga_point(CGA3D, -1.0, 0.0, 0.0)

        # The sphere through these points is their outer product
        S = P1 ^ P2 ^ P3 ^ P4

        # This should be a 4-vector (quadvector)
        assert S.max_grade() == 4

    def test_reflection_in_plane(self):
        """Test reflecting a point in a plane."""
        CGA3D = Algebra.cga(3)

        # Create xy-plane (z = 0)
        plane = Multivector.cga_plane(CGA3D, 0.0, 0.0, 1.0, 0.0)

        # Create a point above the plane
        P = Multivector.cga_point(CGA3D, 1.0, 2.0, 3.0)

        # Reflection: P' = plane * P * plane.reverse()
        plane_rev = plane.reverse()
        P_reflected = plane.sandwich(P)

        # Extract coordinates of reflected point
        center = P_reflected.cga_extract_center()

        # z should be negated
        assert abs(center[0] - 1.0) < 1e-10
        assert abs(center[1] - 2.0) < 1e-10
        assert abs(center[2] + 3.0) < 1e-10  # z = -3

    def test_translation_via_point_difference(self):
        """Test computing the translation vector between two points.

        This test verifies that we can use CGA to compute the displacement
        between two points.
        """
        CGA3D = Algebra.cga(3)

        # Create two points
        P1 = Multivector.cga_point(CGA3D, 0.0, 0.0, 0.0)
        P2 = Multivector.cga_point(CGA3D, 1.0, 2.0, 3.0)

        # The distance between them should be sqrt(1 + 4 + 9) = sqrt(14)
        d = P1.cga_distance(P2)
        expected = math.sqrt(14)
        assert abs(d - expected) < 1e-9, f"distance should be {expected}, got {d}"

        # Extract centers
        c1 = P1.cga_extract_center()
        c2 = P2.cga_extract_center()

        assert abs(c1[0]) < 1e-10
        assert abs(c1[1]) < 1e-10
        assert abs(c1[2]) < 1e-10

        assert abs(c2[0] - 1.0) < 1e-10
        assert abs(c2[1] - 2.0) < 1e-10
        assert abs(c2[2] - 3.0) < 1e-10
