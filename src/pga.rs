//! Projective Geometric Algebra (PGA) convenience methods.
//!
//! PGA uses the degenerate basis vector e0 (squares to 0) to represent
//! homogeneous coordinates. In PGA3D (Cl(3,0,1)):
//! - Points are grade-3 (trivectors): e123 + x*e032 + y*e013 + z*e021
//! - Planes are grade-1 (vectors): a*e1 + b*e2 + c*e3 + d*e0
//! - Lines are grade-2 (bivectors)
//! - Motors are even-grade (rotations + translations)
//!
//! # Basis Convention
//!
//! In PGA3D (Cl(3,0,1)), we use:
//! - e1, e2, e3: Euclidean basis vectors (square to +1)
//! - e0: Degenerate basis vector (squares to 0)
//!
//! The degenerate vector e0 is stored as the LAST basis vector internally
//! (index 3, blade index 8), following the convention that r vectors
//! come after the p + q vectors in Cl(p,q,r).
//!
//! # Blade Indices in PGA3D (dimension 4, 16 blades)
//!
//! Scalars (grade 0):
//!   0: 1
//!
//! Vectors (grade 1):
//!   1: e1, 2: e2, 4: e3, 8: e0
//!
//! Bivectors (grade 2):
//!   3: e12, 5: e13, 6: e23, 9: e01, 10: e02, 12: e03
//!
//! Trivectors (grade 3):
//!   7: e123, 11: e012, 13: e013, 14: e023
//!
//! Pseudoscalar (grade 4):
//!   15: e0123
//!
//! # Reference
//!
//! - "Geometric Algebra for Computer Science" by Dorst, Fontijne, Mann
//! - ganja.js by Steven De Keninck
//! - bivector.net PGA tutorials

use crate::algebra;
use crate::pyalgebra::PyAlgebra;
use crate::Multivector;
use pyo3::prelude::*;

/// PGA-specific blade indices for PGA3D (Cl(3,0,1))
#[allow(dead_code)]
pub mod pga3d_blades {
    // Scalars
    pub const SCALAR: usize = 0;

    // Vectors (grade 1)
    pub const E1: usize = 1;
    pub const E2: usize = 2;
    pub const E3: usize = 4;
    pub const E0: usize = 8; // Degenerate/ideal basis vector

    // Bivectors (grade 2) - Euclidean
    pub const E12: usize = 3;
    pub const E13: usize = 5;
    pub const E23: usize = 6;

    // Bivectors (grade 2) - Ideal (contain e0)
    pub const E01: usize = 9;
    pub const E02: usize = 10;
    pub const E03: usize = 12;

    // Trivectors (grade 3)
    pub const E123: usize = 7;  // Euclidean pseudoscalar / ideal point weight
    pub const E012: usize = 11; // e0 ^ e1 ^ e2
    pub const E013: usize = 13; // e0 ^ e1 ^ e3
    pub const E023: usize = 14; // e0 ^ e2 ^ e3

    // Pseudoscalar (grade 4)
    pub const E0123: usize = 15;
}

/// PGA convenience methods for Multivector.
///
/// These methods provide PGA-specific operations like creating points,
/// planes, lines, and motors (rigid transformations).
#[pymethods]
impl Multivector {
    // =========================================================================
    // PGA ELEMENT CONSTRUCTORS
    // =========================================================================

    /// Create a PGA point at (x, y, z).
    ///
    /// In PGA3D, points are represented as trivectors:
    /// P = e123 + x*e032 + y*e013 + z*e021
    ///
    /// The e123 component is the "weight" (typically 1 for normalized points).
    /// The other components encode the position using the ideal (e0-containing) parts.
    ///
    /// # Arguments
    /// * `algebra` - A PGA algebra (should be Cl(3,0,1))
    /// * `x`, `y`, `z` - The Cartesian coordinates of the point
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// origin = Multivector.pga_point(PGA3D, 0.0, 0.0, 0.0)
    /// p = Multivector.pga_point(PGA3D, 1.0, 2.0, 3.0)
    /// ```
    #[staticmethod]
    pub fn pga_point(algebra: &PyAlgebra, x: f64, y: f64, z: f64) -> PyResult<Self> {
        if !algebra.inner.signature.is_pga() || algebra.inner.dimension() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pga_point requires a PGA3D algebra Cl(3,0,1)"
            ));
        }

        let size = algebra.inner.num_blades();
        let mut coeffs = vec![0.0; size];

        // Point in PGA3D: e123 + x*e032 + y*e013 + z*e021
        // With our blade ordering:
        // e123 = 7, e032 = e023 = 14, e013 = 13, e021 = e012 = 11
        // But we need to be careful about signs from reordering!
        //
        // Standard PGA convention (ganja.js, bivector.net):
        // Point P at (x,y,z) = e123 + x*e032 + y*e013 + z*e021
        //
        // e032 = e0 ^ e3 ^ e2 = -e023 (index 14)
        // e013 = e0 ^ e1 ^ e3 = e013 (index 13)
        // e021 = e0 ^ e2 ^ e1 = -e012 (index 11)

        coeffs[pga3d_blades::E123] = 1.0;  // Weight = 1
        coeffs[pga3d_blades::E023] = -x;   // e032 = -e023
        coeffs[pga3d_blades::E013] = y;    // e013
        coeffs[pga3d_blades::E012] = -z;   // e021 = -e012

        Ok(Multivector {
            coeffs,
            dims: 4,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Create a PGA plane with equation ax + by + cz + d = 0.
    ///
    /// In PGA3D, planes are represented as vectors:
    /// Pi = a*e1 + b*e2 + c*e3 + d*e0
    ///
    /// The (a, b, c) coefficients give the plane normal.
    /// The d coefficient is the signed distance from the origin (when normalized).
    ///
    /// # Arguments
    /// * `algebra` - A PGA algebra (should be Cl(3,0,1))
    /// * `a`, `b`, `c` - The plane normal components
    /// * `d` - The plane offset
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// # XY plane (z = 0)
    /// xy_plane = Multivector.pga_plane(PGA3D, 0.0, 0.0, 1.0, 0.0)
    /// # Plane x + y + z = 1
    /// p = Multivector.pga_plane(PGA3D, 1.0, 1.0, 1.0, -1.0)
    /// ```
    #[staticmethod]
    pub fn pga_plane(algebra: &PyAlgebra, a: f64, b: f64, c: f64, d: f64) -> PyResult<Self> {
        if !algebra.inner.signature.is_pga() || algebra.inner.dimension() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pga_plane requires a PGA3D algebra Cl(3,0,1)"
            ));
        }

        let size = algebra.inner.num_blades();
        let mut coeffs = vec![0.0; size];

        // Plane: a*e1 + b*e2 + c*e3 + d*e0
        coeffs[pga3d_blades::E1] = a;
        coeffs[pga3d_blades::E2] = b;
        coeffs[pga3d_blades::E3] = c;
        coeffs[pga3d_blades::E0] = d;

        Ok(Multivector {
            coeffs,
            dims: 4,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Create a line through two points using the regressive product.
    ///
    /// The line L = P1 & P2 (regressive/vee product) represents the line
    /// passing through both points.
    ///
    /// # Arguments
    /// * `p1`, `p2` - Two PGA points (trivectors)
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// p1 = Multivector.pga_point(PGA3D, 0.0, 0.0, 0.0)
    /// p2 = Multivector.pga_point(PGA3D, 1.0, 0.0, 0.0)
    /// line = Multivector.pga_line_from_points(p1, p2)  # X-axis
    /// ```
    #[staticmethod]
    pub fn pga_line_from_points(
        p1: PyRef<'_, Multivector>,
        p2: PyRef<'_, Multivector>,
    ) -> PyResult<Self> {
        // In PGA, the regressive product (vee/meet) is:
        // A ∨ B = dual(dual(A) ∧ dual(B))
        // where dual uses the complement map
        let p1_dual = p1.pga_dual()?;
        let p2_dual = p2.pga_dual()?;
        let wedge = p1_dual.outer_product(&p2_dual)?;
        wedge.pga_dual()
    }

    /// Create a line from the intersection of two planes using the outer product.
    ///
    /// The line L = Pi1 ^ Pi2 (wedge/outer product) represents the
    /// intersection line of both planes.
    ///
    /// # Arguments
    /// * `pi1`, `pi2` - Two PGA planes (vectors)
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// # XY plane and XZ plane intersect along the X-axis
    /// xy = Multivector.pga_plane(PGA3D, 0.0, 0.0, 1.0, 0.0)
    /// xz = Multivector.pga_plane(PGA3D, 0.0, 1.0, 0.0, 0.0)
    /// x_axis = Multivector.pga_line_from_planes(xy, xz)
    /// ```
    #[staticmethod]
    pub fn pga_line_from_planes(
        pi1: PyRef<'_, Multivector>,
        pi2: PyRef<'_, Multivector>,
    ) -> PyResult<Self> {
        // Line = Pi1 ^ Pi2 (outer product)
        pi1.outer_product(&pi2)
    }

    /// Create a motor (rigid transformation) from axis, point, and angle.
    ///
    /// A motor represents a screw motion: rotation about an axis combined
    /// with translation along that axis.
    ///
    /// For pure rotation (no translation), pass a point on the axis.
    ///
    /// # Arguments
    /// * `algebra` - A PGA algebra (should be Cl(3,0,1))
    /// * `axis` - Direction of the rotation axis (normalized internally)
    /// * `point` - A point on the rotation axis
    /// * `angle` - Rotation angle in radians
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// # Rotation of 90 degrees about the Z-axis through the origin
    /// axis = [0.0, 0.0, 1.0]
    /// motor = Multivector.pga_motor_from_axis_angle(PGA3D, axis, [0.0, 0.0, 0.0], math.pi/2)
    /// ```
    #[staticmethod]
    pub fn pga_motor_from_axis_angle(
        algebra: &PyAlgebra,
        axis: Vec<f64>,
        point: Vec<f64>,
        angle: f64,
    ) -> PyResult<Self> {
        if !algebra.inner.signature.is_pga() || algebra.inner.dimension() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pga_motor_from_axis_angle requires a PGA3D algebra Cl(3,0,1)"
            ));
        }

        if axis.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "axis must have 3 components"
            ));
        }

        if point.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "point must have 3 components"
            ));
        }

        // Normalize axis
        let axis_len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if axis_len < 1e-10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "axis has zero length"
            ));
        }
        let ax = axis[0] / axis_len;
        let ay = axis[1] / axis_len;
        let az = axis[2] / axis_len;
        let px = point[0];
        let py = point[1];
        let pz = point[2];

        let half_angle = angle / 2.0;
        let c = half_angle.cos();
        let s = half_angle.sin();

        let size = algebra.inner.num_blades();
        let mut coeffs = vec![0.0; size];

        // Motor = exp(-angle/2 * L) where L is the line (axis through point)
        //
        // For a line through point P with direction d:
        // L = P & (P + d*e123)  [dual of the line]
        //
        // The exponential gives:
        // Motor = cos(angle/2) + sin(angle/2) * L_normalized
        //
        // For a line through origin with direction (ax, ay, az):
        // The bivector part is ax*e23 + ay*e31 + az*e12
        //
        // For offset by point (px, py, pz), we add ideal components.

        // Scalar part
        coeffs[pga3d_blades::SCALAR] = c;

        // Bivector part (Euclidean rotation)
        // For right-hand rule rotation about axis (ax, ay, az):
        // The rotation bivector is the Hodge dual of the axis in 3D Euclidean space
        // Rotation about Z (0,0,1) uses e12 with positive coefficient for CCW rotation
        // Rotation about X (1,0,0) uses e23 with positive coefficient for CCW rotation
        // Rotation about Y (0,1,0) uses e31 = -e13 with positive coefficient for CCW rotation
        //
        // Negating to get the correct rotation direction (convention matching ganja.js)
        coeffs[pga3d_blades::E23] = -s * ax;
        coeffs[pga3d_blades::E13] = s * ay;  // e31 = -e13, so -(-s*ay) = s*ay for e13
        coeffs[pga3d_blades::E12] = -s * az;

        // Ideal bivector part (translation from rotating about offset axis)
        // moment = point × axis (cross product gives the translation direction)
        // These go into e01, e02, e03
        let mx = py * az - pz * ay;
        let my = pz * ax - px * az;
        let mz = px * ay - py * ax;

        coeffs[pga3d_blades::E01] = s * mx;
        coeffs[pga3d_blades::E02] = s * my;
        coeffs[pga3d_blades::E03] = s * mz;

        Ok(Multivector {
            coeffs,
            dims: 4,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Create a pure translation motor.
    ///
    /// A translator moves points by the vector (dx, dy, dz).
    ///
    /// # Arguments
    /// * `algebra` - A PGA algebra (should be Cl(3,0,1))
    /// * `dx`, `dy`, `dz` - Translation vector components
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// T = Multivector.pga_translator(PGA3D, 1.0, 0.0, 0.0)  # Translate by (1, 0, 0)
    /// p = Multivector.pga_point(PGA3D, 0.0, 0.0, 0.0)
    /// p_moved = T.sandwich(p)
    /// # Extract coordinates - should be (1, 0, 0)
    /// ```
    #[staticmethod]
    pub fn pga_translator(algebra: &PyAlgebra, dx: f64, dy: f64, dz: f64) -> PyResult<Self> {
        if !algebra.inner.signature.is_pga() || algebra.inner.dimension() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pga_translator requires a PGA3D algebra Cl(3,0,1)"
            ));
        }

        let size = algebra.inner.num_blades();
        let mut coeffs = vec![0.0; size];

        // Translator T = 1 + (d/2) where d = dx*e01 + dy*e02 + dz*e03
        // This is the exponential of the ideal line: exp(-d*e0∞/2)
        //
        // In PGA the translation is: T = 1 + (1/2)(dx*e01 + dy*e02 + dz*e03)

        coeffs[pga3d_blades::SCALAR] = 1.0;
        coeffs[pga3d_blades::E01] = dx / 2.0;
        coeffs[pga3d_blades::E02] = dy / 2.0;
        coeffs[pga3d_blades::E03] = dz / 2.0;

        Ok(Multivector {
            coeffs,
            dims: 4,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Normalize a PGA point (make the e123 coefficient equal to 1).
    ///
    /// In PGA, points are projective: P and k*P represent the same point.
    /// This normalizes so the weight (e123 coefficient) is 1.
    ///
    /// # Returns
    /// A normalized point, or error if the weight is zero (ideal point).
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// p = Multivector.pga_point(PGA3D, 1.0, 2.0, 3.0)
    /// p_scaled = p * 5.0  # Scale the point
    /// p_norm = p_scaled.pga_normalize()  # Back to weight 1
    /// ```
    pub fn pga_normalize(&self) -> PyResult<Self> {
        // Check this is a PGA element
        let alg = self.get_algebra();
        if !alg.signature.is_pga() || self.dims != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pga_normalize requires a PGA3D element"
            ));
        }

        // Get the e123 coefficient (weight)
        let weight = self.coeffs[pga3d_blades::E123];

        if weight.abs() < 1e-10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot normalize ideal point (e123 coefficient is zero)"
            ));
        }

        // Scale all coefficients by 1/weight
        let coeffs: Vec<f64> = self.coeffs.iter().map(|&c| c / weight).collect();

        Ok(Multivector {
            coeffs,
            dims: self.dims,
            algebra_opt: self.algebra_opt.clone(),
        })
    }

    /// Extract the ideal (e0-containing) part of a PGA element.
    ///
    /// The ideal part consists of all blades that contain the degenerate
    /// basis vector e0. These represent "at infinity" or translation components.
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// plane = Multivector.pga_plane(PGA3D, 1.0, 0.0, 0.0, 2.0)
    /// ideal = plane.pga_ideal_part()  # Just the e0 component
    /// ```
    pub fn pga_ideal_part(&self) -> PyResult<Self> {
        let alg = self.get_algebra();
        if !alg.signature.is_pga() || self.dims != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pga_ideal_part requires a PGA3D element"
            ));
        }

        let mut coeffs = vec![0.0; self.coeffs.len()];

        // e0 is at bit position 3 (index 8)
        // Blades containing e0 have bit 3 set
        for (i, &c) in self.coeffs.iter().enumerate() {
            if (i & pga3d_blades::E0) != 0 {
                coeffs[i] = c;
            }
        }

        Ok(Multivector {
            coeffs,
            dims: self.dims,
            algebra_opt: self.algebra_opt.clone(),
        })
    }

    /// Extract the Euclidean (non-e0) part of a PGA element.
    ///
    /// The Euclidean part consists of all blades that do NOT contain
    /// the degenerate basis vector e0. These represent the "finite"
    /// or rotational components.
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// plane = Multivector.pga_plane(PGA3D, 1.0, 0.0, 0.0, 2.0)
    /// eucl = plane.pga_euclidean_part()  # Just the e1 component
    /// ```
    pub fn pga_euclidean_part(&self) -> PyResult<Self> {
        let alg = self.get_algebra();
        if !alg.signature.is_pga() || self.dims != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pga_euclidean_part requires a PGA3D element"
            ));
        }

        let mut coeffs = vec![0.0; self.coeffs.len()];

        // e0 is at bit position 3 (index 8)
        // Euclidean blades do NOT have bit 3 set
        for (i, &c) in self.coeffs.iter().enumerate() {
            if (i & pga3d_blades::E0) == 0 {
                coeffs[i] = c;
            }
        }

        Ok(Multivector {
            coeffs,
            dims: self.dims,
            algebra_opt: self.algebra_opt.clone(),
        })
    }

    /// Compute the PGA dual using a grade-based complement.
    ///
    /// In PGA3D, the dual maps each blade to its complement blade.
    /// Unlike the standard dual (which uses the pseudoscalar inverse),
    /// this implementation works with the degenerate pseudoscalar.
    ///
    /// The complement is computed by XORing each blade index with 15 (e0123).
    pub fn pga_dual(&self) -> PyResult<Self> {
        let alg = self.get_algebra();
        if !alg.signature.is_pga() || self.dims != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pga_dual requires a PGA3D element"
            ));
        }

        let mut coeffs = vec![0.0; self.coeffs.len()];

        // Complement each blade: blade i -> blade (15 XOR i)
        // This swaps e.g. 1 <-> e0123, e1 <-> e023, e2 <-> e013, etc.
        // We also need to account for the sign from reordering
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c != 0.0 {
                let dual_index = 15 ^ i;  // XOR with e0123 index

                // Compute the sign: grade(i) * (4 - grade(i)) swaps
                // plus metric factors from e0
                let grade_i = algebra::blade_grade(i);

                // In PGA, the dual sign depends on convention
                // Using the formula: (-1)^(grade * (grade-1) / 2)
                // and accounting for the e0 factor
                let sign = if (grade_i * (grade_i - 1) / 2) % 2 == 0 { 1.0 } else { -1.0 };

                coeffs[dual_index] += sign * c;
            }
        }

        Ok(Multivector {
            coeffs,
            dims: self.dims,
            algebra_opt: self.algebra_opt.clone(),
        })
    }

    /// Extract Cartesian coordinates (x, y, z) from a PGA point.
    ///
    /// The point must be normalized (e123 = 1) or will be normalized first.
    ///
    /// # Returns
    /// A tuple (x, y, z) of the point's Cartesian coordinates.
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// p = Multivector.pga_point(PGA3D, 1.0, 2.0, 3.0)
    /// x, y, z = p.pga_point_coords()  # (1.0, 2.0, 3.0)
    /// ```
    pub fn pga_point_coords(&self) -> PyResult<(f64, f64, f64)> {
        let alg = self.get_algebra();
        if !alg.signature.is_pga() || self.dims != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pga_point_coords requires a PGA3D element"
            ));
        }

        let weight = self.coeffs[pga3d_blades::E123];
        if weight.abs() < 1e-10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot extract coordinates from ideal point (e123 = 0)"
            ));
        }

        // Point = e123 + x*e032 + y*e013 + z*e021
        // e032 = -e023 (index 14)
        // e013 (index 13)
        // e021 = -e012 (index 11)
        //
        // So: x = -coeffs[E023] / weight
        //     y = coeffs[E013] / weight
        //     z = -coeffs[E012] / weight

        let x = -self.coeffs[pga3d_blades::E023] / weight;
        let y = self.coeffs[pga3d_blades::E013] / weight;
        let z = -self.coeffs[pga3d_blades::E012] / weight;

        Ok((x, y, z))
    }

    /// Compute the point-plane distance in PGA.
    ///
    /// Returns the signed distance from a point to a plane.
    /// Positive distance means the point is on the side of the plane
    /// that the normal points to.
    ///
    /// # Arguments
    /// * `plane` - A PGA plane (vector)
    ///
    /// # Returns
    /// The signed distance (requires both elements to be normalized).
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// p = Multivector.pga_point(PGA3D, 0.0, 0.0, 1.0)
    /// xy_plane = Multivector.pga_plane(PGA3D, 0.0, 0.0, 1.0, 0.0)
    /// d = p.pga_point_plane_distance(xy_plane)  # Should be 1.0
    /// ```
    pub fn pga_point_plane_distance(&self, plane: &Multivector) -> PyResult<f64> {
        let alg = self.get_algebra();
        if !alg.signature.is_pga() || self.dims != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pga_point_plane_distance requires PGA3D elements"
            ));
        }

        // Normalize point
        let point_weight = self.coeffs[pga3d_blades::E123];
        if point_weight.abs() < 1e-10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot compute distance from ideal point"
            ));
        }

        // Normalize plane normal
        let a = plane.coeffs[pga3d_blades::E1];
        let b = plane.coeffs[pga3d_blades::E2];
        let c = plane.coeffs[pga3d_blades::E3];
        let d = plane.coeffs[pga3d_blades::E0];
        let normal_len = (a * a + b * b + c * c).sqrt();

        if normal_len < 1e-10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "plane has zero normal"
            ));
        }

        // Get point coordinates
        let (x, y, z) = self.pga_point_coords()?;

        // Distance = (ax + by + cz + d) / |normal|
        let dist = (a * x + b * y + c * z + d) / normal_len;

        Ok(dist)
    }

    /// Check if this is a valid PGA motor (normalized even-grade element).
    ///
    /// A motor M satisfies M * ~M = 1 (within tolerance) and has only
    /// even-grade components (scalar, bivector, quadvector).
    ///
    /// # Example
    /// ```python
    /// PGA3D = Algebra.pga(3)
    /// T = Multivector.pga_translator(PGA3D, 1.0, 0.0, 0.0)
    /// assert T.is_pga_motor()
    /// ```
    #[pyo3(signature = (tol=1e-10))]
    pub fn is_pga_motor(&self, tol: f64) -> bool {
        let alg = self.get_algebra();
        if !alg.signature.is_pga() || self.dims != 4 {
            return false;
        }

        // Check only even grades (0, 2, 4) have non-zero coefficients
        for (i, &c) in self.coeffs.iter().enumerate() {
            let grade = algebra::blade_grade(i);
            if grade % 2 == 1 && c.abs() > tol {
                return false;
            }
        }

        // Check M * ~M = 1
        if let Ok(m_rev) = Ok::<_, PyErr>(self.reverse()) {
            if let Ok(product) = self.geometric_product(&m_rev) {
                // Should be approximately scalar 1
                if (product.coeffs[0] - 1.0).abs() > tol {
                    return false;
                }
                // All other components should be near zero
                for &c in product.coeffs.iter().skip(1) {
                    if c.abs() > tol {
                        return false;
                    }
                }
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Algebra, Signature};
    use approx::assert_relative_eq;
    use std::sync::Arc;

    fn pga3d() -> Arc<Algebra> {
        Algebra::pga(3)
    }

    #[test]
    fn test_pga_blade_indices() {
        // Verify our blade index constants are correct
        assert_eq!(pga3d_blades::SCALAR, 0);
        assert_eq!(pga3d_blades::E1, 1);
        assert_eq!(pga3d_blades::E2, 2);
        assert_eq!(pga3d_blades::E12, 3);
        assert_eq!(pga3d_blades::E3, 4);
        assert_eq!(pga3d_blades::E13, 5);
        assert_eq!(pga3d_blades::E23, 6);
        assert_eq!(pga3d_blades::E123, 7);
        assert_eq!(pga3d_blades::E0, 8);
        assert_eq!(pga3d_blades::E01, 9);
        assert_eq!(pga3d_blades::E02, 10);
        assert_eq!(pga3d_blades::E012, 11);
        assert_eq!(pga3d_blades::E03, 12);
        assert_eq!(pga3d_blades::E013, 13);
        assert_eq!(pga3d_blades::E023, 14);
        assert_eq!(pga3d_blades::E0123, 15);
    }

    #[test]
    fn test_pga_degenerate_squares_to_zero() {
        let alg = pga3d();
        // e0 * e0 should be 0
        let (blade, sign) = alg.product(pga3d_blades::E0, pga3d_blades::E0);
        assert_eq!(blade, 0);
        assert_relative_eq!(sign, 0.0);
    }

    #[test]
    fn test_pga_euclidean_squares_to_one() {
        let alg = pga3d();
        // e1, e2, e3 should square to +1
        for &e in &[pga3d_blades::E1, pga3d_blades::E2, pga3d_blades::E3] {
            let (blade, sign) = alg.product(e, e);
            assert_eq!(blade, 0);
            assert_relative_eq!(sign, 1.0);
        }
    }
}
