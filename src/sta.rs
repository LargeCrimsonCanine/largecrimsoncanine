//! Spacetime Algebra (STA) convenience methods.
//!
//! STA uses the Minkowski metric (+,-,-,-) for relativistic physics.
//! This is Cl(1,3,0) with the "mostly minus" convention.
//!
//! Basis vectors:
//! - gamma0 (index 1): Timelike, squares to +1
//! - gamma1 (index 2): Spacelike, squares to -1
//! - gamma2 (index 4): Spacelike, squares to -1
//! - gamma3 (index 8): Spacelike, squares to -1
//!
//! Key objects:
//! - Spacetime vectors (4-vectors): events, momenta
//! - Spacetime bivectors: electromagnetic field tensor, angular momentum
//! - Rotors: Lorentz transformations (boosts and rotations)
//!
//! References:
//! - Hestenes, "Space-Time Algebra" (1966, 2015)
//! - Doran & Lasenby, "Geometric Algebra for Physicists" (2003)

use pyo3::prelude::*;
use std::sync::Arc;

use crate::algebra::Algebra;
use crate::multivector::Multivector;
use crate::pyalgebra::PyAlgebra;

/// Basis vector indices in STA.
/// In Cl(1,3,0), we use binary encoding:
/// - gamma0 = e1 = index 1 (bit 0 set)
/// - gamma1 = e2 = index 2 (bit 1 set)
/// - gamma2 = e3 = index 4 (bit 2 set)
/// - gamma3 = e4 = index 8 (bit 3 set)
const GAMMA0_IDX: usize = 1;
const GAMMA1_IDX: usize = 2;
const GAMMA2_IDX: usize = 4;
const GAMMA3_IDX: usize = 8;

/// Bivector indices in STA (6 bivectors total).
/// Timelike bivectors (gamma0 wedge gamma_i):
const GAMMA01_IDX: usize = GAMMA0_IDX | GAMMA1_IDX; // 3
const GAMMA02_IDX: usize = GAMMA0_IDX | GAMMA2_IDX; // 5
const GAMMA03_IDX: usize = GAMMA0_IDX | GAMMA3_IDX; // 9

/// Spacelike bivectors (gamma_i wedge gamma_j):
const GAMMA12_IDX: usize = GAMMA1_IDX | GAMMA2_IDX; // 6
const GAMMA13_IDX: usize = GAMMA1_IDX | GAMMA3_IDX; // 10
const GAMMA23_IDX: usize = GAMMA2_IDX | GAMMA3_IDX; // 12

/// Pseudoscalar index (gamma0123)
const PSEUDOSCALAR_IDX: usize = GAMMA0_IDX | GAMMA1_IDX | GAMMA2_IDX | GAMMA3_IDX; // 15

/// Tolerance for lightlike detection
const LIGHTLIKE_TOL: f64 = 1e-10;

/// Check if the given algebra is STA (Cl(1,3,0)).
fn is_sta(algebra: &Arc<Algebra>) -> bool {
    algebra.signature.p == 1 && algebra.signature.q == 3 && algebra.signature.r == 0
}

/// Validate that the algebra is STA, returning an error if not.
fn require_sta(algebra: &Arc<Algebra>) -> PyResult<()> {
    if !is_sta(algebra) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "STA methods require Cl(1,3,0) algebra, got Cl({},{},{})",
            algebra.signature.p, algebra.signature.q, algebra.signature.r
        )));
    }
    Ok(())
}

/// STA-specific methods for Multivector.
///
/// These methods are added to the Multivector class and provide
/// convenient access to spacetime algebra operations.
#[pymethods]
impl Multivector {
    // =========================================================================
    // CONSTRUCTORS (static methods)
    // =========================================================================

    /// Create a spacetime 4-vector in STA.
    ///
    /// Creates x = t*gamma0 + x*gamma1 + y*gamma2 + z*gamma3
    ///
    /// The components are in natural units where c = 1.
    ///
    /// # Arguments
    /// * `algebra` - The STA algebra (Cl(1,3,0))
    /// * `t` - Time component
    /// * `x` - Spatial x component
    /// * `y` - Spatial y component
    /// * `z` - Spatial z component
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// event = Multivector.sta_vector(STA, 1.0, 0.5, 0.0, 0.0)
    /// # Creates spacetime vector: gamma0 + 0.5*gamma1
    /// ```
    #[staticmethod]
    pub fn sta_vector(algebra: &PyAlgebra, t: f64, x: f64, y: f64, z: f64) -> PyResult<Self> {
        require_sta(&algebra.inner)?;

        let mut coeffs = vec![0.0; 16];
        coeffs[GAMMA0_IDX] = t;
        coeffs[GAMMA1_IDX] = x;
        coeffs[GAMMA2_IDX] = y;
        coeffs[GAMMA3_IDX] = z;

        Ok(Multivector {
            coeffs,
            dims: 4,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Get a gamma matrix (basis vector) in STA.
    ///
    /// Returns the i-th gamma matrix:
    /// - i=0: gamma0 (timelike, squares to +1)
    /// - i=1: gamma1 (spacelike, squares to -1)
    /// - i=2: gamma2 (spacelike, squares to -1)
    /// - i=3: gamma3 (spacelike, squares to -1)
    ///
    /// # Arguments
    /// * `algebra` - The STA algebra (Cl(1,3,0))
    /// * `i` - Index (0-3)
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// gamma0 = Multivector.sta_gamma(STA, 0)
    /// gamma1 = Multivector.sta_gamma(STA, 1)
    /// # gamma0 * gamma0 = +1 (timelike)
    /// # gamma1 * gamma1 = -1 (spacelike)
    /// ```
    #[staticmethod]
    pub fn sta_gamma(algebra: &PyAlgebra, i: usize) -> PyResult<Self> {
        require_sta(&algebra.inner)?;

        if i > 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "gamma index must be 0-3, got {}",
                i
            )));
        }

        let mut coeffs = vec![0.0; 16];
        let idx = 1 << i; // 1, 2, 4, 8 for i = 0, 1, 2, 3
        coeffs[idx] = 1.0;

        Ok(Multivector {
            coeffs,
            dims: 4,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Create an electromagnetic field tensor in STA.
    ///
    /// The electromagnetic field is represented as a bivector:
    /// F = E + I*B
    ///
    /// where:
    /// - E = Ex*gamma01 + Ey*gamma02 + Ez*gamma03 (electric field)
    /// - I*B = Bx*gamma23 + By*gamma31 + Bz*gamma12 (magnetic field dual)
    ///
    /// Note: gamma31 = -gamma13 due to antisymmetry.
    ///
    /// # Arguments
    /// * `algebra` - The STA algebra (Cl(1,3,0))
    /// * `e` - Electric field components [Ex, Ey, Ez]
    /// * `b` - Magnetic field components [Bx, By, Bz]
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// # Uniform electric field in x-direction
    /// F = Multivector.sta_bivector(STA, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    /// ```
    #[staticmethod]
    pub fn sta_bivector(algebra: &PyAlgebra, e: Vec<f64>, b: Vec<f64>) -> PyResult<Self> {
        require_sta(&algebra.inner)?;

        if e.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "E field must have 3 components, got {}",
                e.len()
            )));
        }
        if b.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "B field must have 3 components, got {}",
                b.len()
            )));
        }

        let mut coeffs = vec![0.0; 16];

        // Electric field: E_i * gamma0i
        coeffs[GAMMA01_IDX] = e[0]; // Ex * gamma01
        coeffs[GAMMA02_IDX] = e[1]; // Ey * gamma02
        coeffs[GAMMA03_IDX] = e[2]; // Ez * gamma03

        // Magnetic field: I*B maps Bx -> gamma23, By -> -gamma13, Bz -> gamma12
        // In STA with I = gamma0123:
        // I * (Bx*sigma1 + By*sigma2 + Bz*sigma3) where sigma_i = gamma_i * gamma_0
        // This gives: Bx*gamma23 - By*gamma13 + Bz*gamma12
        coeffs[GAMMA23_IDX] = b[0];       // Bx * gamma23
        coeffs[GAMMA13_IDX] = -b[1];      // -By * gamma13 (note: GAMMA13_IDX = 10)
        coeffs[GAMMA12_IDX] = b[2];       // Bz * gamma12

        Ok(Multivector {
            coeffs,
            dims: 4,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Create a Lorentz boost rotor along a direction.
    ///
    /// Creates a rotor that performs a Lorentz boost with rapidity phi
    /// in the direction of the given velocity vector.
    ///
    /// The rotor is: R = exp(-phi/2 * gamma0 * v_hat)
    ///
    /// where v_hat is the normalized direction and phi = arctanh(|v|/c).
    /// For |v| << c, phi ~ |v|/c.
    ///
    /// # Arguments
    /// * `algebra` - The STA algebra (Cl(1,3,0))
    /// * `velocity` - 3D velocity vector [vx, vy, vz] (in units where c=1)
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// # Boost at 0.5c in x-direction
    /// R = Multivector.sta_boost(STA, [0.5, 0.0, 0.0])
    /// # Apply boost: x_boosted = R * x * ~R
    /// ```
    #[staticmethod]
    pub fn sta_boost(algebra: &PyAlgebra, velocity: Vec<f64>) -> PyResult<Self> {
        require_sta(&algebra.inner)?;

        if velocity.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "velocity must have 3 components, got {}",
                velocity.len()
            )));
        }

        let vx = velocity[0];
        let vy = velocity[1];
        let vz = velocity[2];
        let v_mag = (vx * vx + vy * vy + vz * vz).sqrt();

        if v_mag < 1e-14 {
            // No boost, return identity
            let mut coeffs = vec![0.0; 16];
            coeffs[0] = 1.0; // scalar = 1
            return Ok(Multivector {
                coeffs,
                dims: 4,
                algebra_opt: Some(algebra.inner.clone()),
            });
        }

        if v_mag >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "velocity magnitude {} >= c (speed of light = 1 in natural units)",
                v_mag
            )));
        }

        // Rapidity: phi = arctanh(v/c)
        let phi = v_mag.atanh();

        // Direction unit vector
        let nx = vx / v_mag;
        let ny = vy / v_mag;
        let nz = vz / v_mag;

        // Boost bivector: B = gamma0 * (nx*gamma1 + ny*gamma2 + nz*gamma3)
        //                   = nx*gamma01 + ny*gamma02 + nz*gamma03
        // Rotor: R = exp(-phi/2 * B) = cosh(phi/2) - sinh(phi/2) * B
        // Note: gamma0i bivectors square to +1 (timelike bivector)
        let half_phi = phi / 2.0;
        let cosh_half = half_phi.cosh();
        let sinh_half = half_phi.sinh();

        let mut coeffs = vec![0.0; 16];
        coeffs[0] = cosh_half; // scalar
        coeffs[GAMMA01_IDX] = -sinh_half * nx; // gamma01
        coeffs[GAMMA02_IDX] = -sinh_half * ny; // gamma02
        coeffs[GAMMA03_IDX] = -sinh_half * nz; // gamma03

        Ok(Multivector {
            coeffs,
            dims: 4,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Create a spatial rotation rotor in STA.
    ///
    /// Creates a rotor that performs rotation by angle theta about
    /// the given axis (a 3D vector).
    ///
    /// The rotation is in the spatial subspace (gamma1, gamma2, gamma3)
    /// and leaves gamma0 invariant.
    ///
    /// Rotor: R = exp(-theta/2 * B)
    /// where B is the bivector dual to the axis.
    ///
    /// # Arguments
    /// * `algebra` - The STA algebra (Cl(1,3,0))
    /// * `axis` - 3D rotation axis [ax, ay, az]
    /// * `angle` - Rotation angle in radians
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// import math
    /// # 90 degree rotation about z-axis
    /// R = Multivector.sta_rotation(STA, [0.0, 0.0, 1.0], math.pi/2)
    /// ```
    #[staticmethod]
    pub fn sta_rotation(algebra: &PyAlgebra, axis: Vec<f64>, angle: f64) -> PyResult<Self> {
        require_sta(&algebra.inner)?;

        if axis.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "axis must have 3 components, got {}",
                axis.len()
            )));
        }

        let ax = axis[0];
        let ay = axis[1];
        let az = axis[2];
        let axis_mag = (ax * ax + ay * ay + az * az).sqrt();

        if axis_mag < 1e-14 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "axis has zero magnitude",
            ));
        }

        // Normalize axis
        let nx = ax / axis_mag;
        let ny = ay / axis_mag;
        let nz = az / axis_mag;

        // Rotation bivector: B = n_x * gamma23 + n_y * gamma31 + n_z * gamma12
        // Note: gamma31 = -gamma13, so we use ny for gamma13 (negated)
        // These are spacelike bivectors that square to -1
        //
        // Rotor: R = exp(theta/2 * B) = cos(theta/2) + sin(theta/2) * B
        // This gives right-hand rule rotation: positive angle about z takes x -> y
        let half_angle = angle / 2.0;
        let cos_half = half_angle.cos();
        let sin_half = half_angle.sin();

        let mut coeffs = vec![0.0; 16];
        coeffs[0] = cos_half; // scalar
        coeffs[GAMMA23_IDX] = sin_half * nx;  // gamma23 (rotation about x)
        coeffs[GAMMA13_IDX] = -sin_half * ny; // gamma13 (rotation about y, note sign: gamma31=-gamma13)
        coeffs[GAMMA12_IDX] = sin_half * nz;  // gamma12 (rotation about z)

        Ok(Multivector {
            coeffs,
            dims: 4,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Get the STA pseudoscalar I = gamma0123.
    ///
    /// The pseudoscalar satisfies I^2 = -1 and commutes with all
    /// even-grade elements.
    ///
    /// # Arguments
    /// * `algebra` - The STA algebra (Cl(1,3,0))
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// I = Multivector.sta_I(STA)
    /// print((I * I).scalar())  # -1.0
    /// ```
    #[staticmethod]
    #[pyo3(name = "sta_I")]
    pub fn sta_pseudoscalar(algebra: &PyAlgebra) -> PyResult<Self> {
        require_sta(&algebra.inner)?;

        let mut coeffs = vec![0.0; 16];
        coeffs[PSEUDOSCALAR_IDX] = 1.0;

        Ok(Multivector {
            coeffs,
            dims: 4,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    // =========================================================================
    // INSTANCE METHODS
    // =========================================================================

    /// Split a spacetime vector into temporal and spatial parts.
    ///
    /// For a vector x = t*gamma0 + r (where r is purely spatial),
    /// returns (t, [rx, ry, rz]).
    ///
    /// This is the spacetime split relative to the gamma0 frame.
    ///
    /// # Returns
    /// Tuple of (time_component, [spatial_x, spatial_y, spatial_z])
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// x = Multivector.sta_vector(STA, 5.0, 3.0, 4.0, 0.0)
    /// t, r = x.sta_spacetime_split()
    /// # t = 5.0, r = [3.0, 4.0, 0.0]
    /// ```
    pub fn sta_spacetime_split(&self) -> PyResult<(f64, Vec<f64>)> {
        let algebra = self.get_algebra();
        require_sta(&algebra)?;

        // Extract vector components
        let t = self.coeffs[GAMMA0_IDX];
        let x = self.coeffs[GAMMA1_IDX];
        let y = self.coeffs[GAMMA2_IDX];
        let z = self.coeffs[GAMMA3_IDX];

        Ok((t, vec![x, y, z]))
    }

    /// Compute the proper time interval (spacetime interval).
    ///
    /// For a spacetime vector x, returns sqrt(x * ~x) = sqrt(t^2 - r^2).
    ///
    /// For timelike vectors (t^2 > r^2), this gives the proper time.
    /// For spacelike vectors (t^2 < r^2), returns sqrt(|x * ~x|) * sign.
    ///
    /// # Returns
    /// The proper time (positive for timelike, negative for spacelike)
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// # Timelike vector: proper time = sqrt(5^2 - 3^2) = 4
    /// x = Multivector.sta_vector(STA, 5.0, 3.0, 0.0, 0.0)
    /// tau = x.sta_proper_time()  # 4.0
    /// ```
    pub fn sta_proper_time(&self) -> PyResult<f64> {
        let algebra = self.get_algebra();
        require_sta(&algebra)?;

        // Compute x * ~x using the metric
        // For a pure vector x = t*gamma0 + x*gamma1 + y*gamma2 + z*gamma3
        // x * ~x = x * x (vectors are self-reverse)
        //       = t^2 * gamma0^2 + x^2 * gamma1^2 + y^2 * gamma2^2 + z^2 * gamma3^2
        //       = t^2 * (+1) + x^2 * (-1) + y^2 * (-1) + z^2 * (-1)
        //       = t^2 - x^2 - y^2 - z^2
        let t = self.coeffs[GAMMA0_IDX];
        let x = self.coeffs[GAMMA1_IDX];
        let y = self.coeffs[GAMMA2_IDX];
        let z = self.coeffs[GAMMA3_IDX];

        let interval_sq = t * t - x * x - y * y - z * z;

        if interval_sq >= 0.0 {
            Ok(interval_sq.sqrt())
        } else {
            // Spacelike: return negative sqrt of absolute value
            Ok(-(-interval_sq).sqrt())
        }
    }

    /// Compute the spacetime interval squared (x * x in STA metric).
    ///
    /// For a spacetime vector, returns t^2 - x^2 - y^2 - z^2.
    ///
    /// - Positive: timelike
    /// - Zero: lightlike (null)
    /// - Negative: spacelike
    ///
    /// # Returns
    /// The squared interval
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// x = Multivector.sta_vector(STA, 5.0, 3.0, 0.0, 0.0)
    /// s2 = x.sta_interval_squared()  # 16.0 (timelike)
    /// ```
    pub fn sta_interval_squared(&self) -> PyResult<f64> {
        let algebra = self.get_algebra();
        require_sta(&algebra)?;

        let t = self.coeffs[GAMMA0_IDX];
        let x = self.coeffs[GAMMA1_IDX];
        let y = self.coeffs[GAMMA2_IDX];
        let z = self.coeffs[GAMMA3_IDX];

        Ok(t * t - x * x - y * y - z * z)
    }

    /// Check if this vector is timelike (x^2 > 0).
    ///
    /// A timelike vector has positive squared interval (t^2 > r^2).
    /// Massive particles have timelike worldlines.
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// x = Multivector.sta_vector(STA, 5.0, 3.0, 0.0, 0.0)
    /// print(x.sta_is_timelike())  # True
    /// ```
    #[pyo3(signature = (tol=None))]
    pub fn sta_is_timelike(&self, tol: Option<f64>) -> PyResult<bool> {
        let interval_sq = self.sta_interval_squared()?;
        let tolerance = tol.unwrap_or(LIGHTLIKE_TOL);
        Ok(interval_sq > tolerance)
    }

    /// Check if this vector is spacelike (x^2 < 0).
    ///
    /// A spacelike vector has negative squared interval (t^2 < r^2).
    /// Spacelike vectors connect events outside each other's light cones.
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// x = Multivector.sta_vector(STA, 1.0, 5.0, 0.0, 0.0)
    /// print(x.sta_is_spacelike())  # True
    /// ```
    #[pyo3(signature = (tol=None))]
    pub fn sta_is_spacelike(&self, tol: Option<f64>) -> PyResult<bool> {
        let interval_sq = self.sta_interval_squared()?;
        let tolerance = tol.unwrap_or(LIGHTLIKE_TOL);
        Ok(interval_sq < -tolerance)
    }

    /// Check if this vector is lightlike/null (x^2 = 0).
    ///
    /// A lightlike (null) vector has zero squared interval (t^2 = r^2).
    /// Photons travel along lightlike worldlines.
    ///
    /// # Arguments
    /// * `tol` - Tolerance for comparison (default 1e-10)
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// # Light ray: t = r
    /// x = Multivector.sta_vector(STA, 1.0, 1.0, 0.0, 0.0)
    /// print(x.sta_is_lightlike())  # True
    /// ```
    #[pyo3(signature = (tol=None))]
    pub fn sta_is_lightlike(&self, tol: Option<f64>) -> PyResult<bool> {
        let interval_sq = self.sta_interval_squared()?;
        let tolerance = tol.unwrap_or(LIGHTLIKE_TOL);
        Ok(interval_sq.abs() <= tolerance)
    }

    /// Extract the electric and magnetic field components from a bivector.
    ///
    /// For an electromagnetic field tensor F = E + I*B, returns (E, B).
    ///
    /// # Returns
    /// Tuple of ([Ex, Ey, Ez], [Bx, By, Bz])
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// F = Multivector.sta_bivector(STA, [1.0, 0.0, 0.0], [0.0, 0.5, 0.0])
    /// E, B = F.sta_field_decompose()
    /// # E = [1.0, 0.0, 0.0], B = [0.0, 0.5, 0.0]
    /// ```
    pub fn sta_field_decompose(&self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let algebra = self.get_algebra();
        require_sta(&algebra)?;

        // Extract electric field from timelike bivectors
        let ex = self.coeffs[GAMMA01_IDX];
        let ey = self.coeffs[GAMMA02_IDX];
        let ez = self.coeffs[GAMMA03_IDX];

        // Extract magnetic field from spacelike bivectors
        // Bx from gamma23, By from -gamma13, Bz from gamma12
        let bx = self.coeffs[GAMMA23_IDX];
        let by = -self.coeffs[GAMMA13_IDX]; // Note the sign
        let bz = self.coeffs[GAMMA12_IDX];

        Ok((vec![ex, ey, ez], vec![bx, by, bz]))
    }

    /// Compute the electromagnetic dual F* = I*F.
    ///
    /// The dual swaps electric and magnetic fields:
    /// F* = I*F => E' = B, B' = -E (in appropriate units)
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// F = Multivector.sta_bivector(STA, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    /// F_dual = F.sta_em_dual()
    /// ```
    pub fn sta_em_dual(&self) -> PyResult<Self> {
        let algebra = self.get_algebra();
        require_sta(&algebra)?;

        // Compute I * F where I = gamma0123
        let pseudoscalar = Multivector::sta_pseudoscalar(&PyAlgebra {
            inner: algebra.clone(),
        })?;

        pseudoscalar.geometric_product(self)
    }

    /// Get the Lorentz gamma factor for a velocity.
    ///
    /// For a boost rotor R = cosh(phi/2) + sinh(phi/2) * B,
    /// returns gamma = cosh(phi) = 1/sqrt(1 - v^2/c^2).
    ///
    /// This only makes sense for boost rotors (not spatial rotations).
    ///
    /// # Returns
    /// The Lorentz factor gamma
    ///
    /// # Example
    /// ```python
    /// STA = Algebra.sta()
    /// R = Multivector.sta_boost(STA, [0.6, 0.0, 0.0])
    /// gamma = R.sta_lorentz_gamma()  # 1.25
    /// ```
    pub fn sta_lorentz_gamma(&self) -> PyResult<f64> {
        let algebra = self.get_algebra();
        require_sta(&algebra)?;

        // For a boost rotor R = cosh(phi/2) + sinh(phi/2) * B
        // gamma = cosh(phi) = 2*cosh^2(phi/2) - 1 = 2*s^2 - 1
        // where s = scalar part of R
        let s = self.coeffs[0];

        // Also need to verify this is mostly a boost rotor
        // (timelike bivector part dominates over spacelike)
        Ok(2.0 * s * s - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sta_algebra() -> Arc<Algebra> {
        Algebra::sta()
    }

    fn sta_py_algebra() -> PyAlgebra {
        PyAlgebra {
            inner: sta_algebra(),
        }
    }

    // =========================================================================
    // Metric tests
    // =========================================================================

    #[test]
    fn test_gamma_squares() {
        let sta = sta_py_algebra();

        // gamma0^2 = +1 (timelike)
        let g0 = Multivector::sta_gamma(&sta, 0).unwrap();
        let g0_sq = g0.geometric_product(&g0).unwrap();
        assert_relative_eq!(g0_sq.coeffs[0], 1.0, epsilon = 1e-10);

        // gamma1^2 = gamma2^2 = gamma3^2 = -1 (spacelike)
        for i in 1..=3 {
            let gi = Multivector::sta_gamma(&sta, i).unwrap();
            let gi_sq = gi.geometric_product(&gi).unwrap();
            assert_relative_eq!(gi_sq.coeffs[0], -1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_anticommutation() {
        let sta = sta_py_algebra();

        for mu in 0..4 {
            for nu in (mu + 1)..4 {
                let g_mu = Multivector::sta_gamma(&sta, mu).unwrap();
                let g_nu = Multivector::sta_gamma(&sta, nu).unwrap();

                // gamma_mu * gamma_nu
                let mu_nu = g_mu.geometric_product(&g_nu).unwrap();
                // gamma_nu * gamma_mu
                let nu_mu = g_nu.geometric_product(&g_mu).unwrap();

                // Anticommutator: {gamma_mu, gamma_nu} = gamma_mu*gamma_nu + gamma_nu*gamma_mu = 0
                let anticomm = mu_nu.__add__(&nu_mu).unwrap();

                for c in &anticomm.coeffs {
                    assert_relative_eq!(*c, 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    // =========================================================================
    // Pseudoscalar tests
    // =========================================================================

    #[test]
    fn test_pseudoscalar_squared() {
        let sta = sta_py_algebra();

        let i = Multivector::sta_pseudoscalar(&sta).unwrap();
        let i_sq = i.geometric_product(&i).unwrap();

        // I^2 = gamma0123 * gamma0123 = -1
        assert_relative_eq!(i_sq.coeffs[0], -1.0, epsilon = 1e-10);
    }

    // =========================================================================
    // Vector and interval tests
    // =========================================================================

    #[test]
    fn test_spacetime_vector() {
        let sta = sta_py_algebra();

        let x = Multivector::sta_vector(&sta, 5.0, 3.0, 4.0, 0.0).unwrap();

        let (t, r) = x.sta_spacetime_split().unwrap();
        assert_relative_eq!(t, 5.0, epsilon = 1e-10);
        assert_relative_eq!(r[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(r[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(r[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interval_timelike() {
        let sta = sta_py_algebra();

        // Timelike: t=5, r=3 => s^2 = 25 - 9 = 16 > 0
        let x = Multivector::sta_vector(&sta, 5.0, 3.0, 0.0, 0.0).unwrap();

        assert!(x.sta_is_timelike(None).unwrap());
        assert!(!x.sta_is_spacelike(None).unwrap());
        assert!(!x.sta_is_lightlike(None).unwrap());

        assert_relative_eq!(x.sta_interval_squared().unwrap(), 16.0, epsilon = 1e-10);
        assert_relative_eq!(x.sta_proper_time().unwrap(), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interval_spacelike() {
        let sta = sta_py_algebra();

        // Spacelike: t=1, r=5 => s^2 = 1 - 25 = -24 < 0
        let x = Multivector::sta_vector(&sta, 1.0, 5.0, 0.0, 0.0).unwrap();

        assert!(!x.sta_is_timelike(None).unwrap());
        assert!(x.sta_is_spacelike(None).unwrap());
        assert!(!x.sta_is_lightlike(None).unwrap());

        assert_relative_eq!(x.sta_interval_squared().unwrap(), -24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interval_lightlike() {
        let sta = sta_py_algebra();

        // Lightlike: t = |r| => s^2 = 0
        let x = Multivector::sta_vector(&sta, 1.0, 1.0, 0.0, 0.0).unwrap();

        assert!(!x.sta_is_timelike(None).unwrap());
        assert!(!x.sta_is_spacelike(None).unwrap());
        assert!(x.sta_is_lightlike(None).unwrap());

        assert_relative_eq!(x.sta_interval_squared().unwrap(), 0.0, epsilon = 1e-10);
    }

    // =========================================================================
    // Boost tests
    // =========================================================================

    #[test]
    fn test_boost_preserves_interval() {
        let sta = sta_py_algebra();

        // Create a timelike vector
        let x = Multivector::sta_vector(&sta, 5.0, 3.0, 0.0, 0.0).unwrap();
        let interval_before = x.sta_interval_squared().unwrap();

        // Apply a boost
        let r = Multivector::sta_boost(&sta, vec![0.5, 0.0, 0.0]).unwrap();
        let r_rev = r.reverse();

        // x' = R * x * ~R
        let x_boosted = r.geometric_product(&x).unwrap();
        let x_boosted = x_boosted.geometric_product(&r_rev).unwrap();

        let interval_after = x_boosted.sta_interval_squared().unwrap();

        // Interval should be preserved
        assert_relative_eq!(interval_before, interval_after, epsilon = 1e-10);
    }

    #[test]
    fn test_boost_gamma_factor() {
        let sta = sta_py_algebra();

        // v = 0.6c => gamma = 1 / sqrt(1 - 0.36) = 1 / sqrt(0.64) = 1.25
        let r = Multivector::sta_boost(&sta, vec![0.6, 0.0, 0.0]).unwrap();
        let gamma = r.sta_lorentz_gamma().unwrap();

        assert_relative_eq!(gamma, 1.25, epsilon = 1e-10);
    }

    #[test]
    fn test_boost_composition() {
        let sta = sta_py_algebra();

        // Two successive boosts in the same direction should compose
        let r1 = Multivector::sta_boost(&sta, vec![0.3, 0.0, 0.0]).unwrap();
        let r2 = Multivector::sta_boost(&sta, vec![0.3, 0.0, 0.0]).unwrap();

        // R_combined = R2 * R1
        let r_combined = r2.geometric_product(&r1).unwrap();

        // The combined velocity is given by relativistic velocity addition:
        // v = (v1 + v2) / (1 + v1*v2) = 0.6 / 1.09 ~ 0.5505
        // But for our test, just verify the gamma increases
        let gamma_single = r1.sta_lorentz_gamma().unwrap();
        let gamma_combined = r_combined.sta_lorentz_gamma().unwrap();

        assert!(gamma_combined > gamma_single);
    }

    // =========================================================================
    // Rotation tests
    // =========================================================================

    #[test]
    fn test_rotation_preserves_interval() {
        let sta = sta_py_algebra();

        // Create a timelike vector
        let x = Multivector::sta_vector(&sta, 5.0, 3.0, 4.0, 0.0).unwrap();
        let interval_before = x.sta_interval_squared().unwrap();

        // Apply a spatial rotation
        let r = Multivector::sta_rotation(&sta, vec![0.0, 0.0, 1.0], std::f64::consts::PI / 4.0)
            .unwrap();
        let r_rev = r.reverse();

        // x' = R * x * ~R
        let x_rotated = r.geometric_product(&x).unwrap();
        let x_rotated = x_rotated.geometric_product(&r_rev).unwrap();

        let interval_after = x_rotated.sta_interval_squared().unwrap();

        // Interval should be preserved
        assert_relative_eq!(interval_before, interval_after, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_leaves_time_invariant() {
        let sta = sta_py_algebra();

        // Create a timelike vector
        let x = Multivector::sta_vector(&sta, 5.0, 3.0, 4.0, 0.0).unwrap();

        // Apply a spatial rotation about z-axis
        let r = Multivector::sta_rotation(&sta, vec![0.0, 0.0, 1.0], std::f64::consts::PI / 2.0)
            .unwrap();
        let r_rev = r.reverse();

        let x_rotated = r.geometric_product(&x).unwrap();
        let x_rotated = x_rotated.geometric_product(&r_rev).unwrap();

        // Time component should be unchanged
        assert_relative_eq!(x_rotated.coeffs[GAMMA0_IDX], 5.0, epsilon = 1e-10);

        // Spatial components should be rotated (x -> y, y -> -x)
        assert_relative_eq!(x_rotated.coeffs[GAMMA1_IDX], -4.0, epsilon = 1e-10);
        assert_relative_eq!(x_rotated.coeffs[GAMMA2_IDX], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x_rotated.coeffs[GAMMA3_IDX], 0.0, epsilon = 1e-10);
    }

    // =========================================================================
    // Electromagnetic field tests
    // =========================================================================

    #[test]
    fn test_em_field_roundtrip() {
        let sta = sta_py_algebra();

        let e_in = vec![1.0, 2.0, 3.0];
        let b_in = vec![0.5, -0.5, 1.0];

        let f = Multivector::sta_bivector(&sta, e_in.clone(), b_in.clone()).unwrap();
        let (e_out, b_out) = f.sta_field_decompose().unwrap();

        for i in 0..3 {
            assert_relative_eq!(e_out[i], e_in[i], epsilon = 1e-10);
            assert_relative_eq!(b_out[i], b_in[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_em_dual() {
        let sta = sta_py_algebra();

        // Pure electric field
        let f = Multivector::sta_bivector(&sta, vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]).unwrap();

        let f_dual = f.sta_em_dual().unwrap();
        let (e_dual, b_dual) = f_dual.sta_field_decompose().unwrap();

        // Hodge dual maps E -> B.  In Cl(1,3) with I = gamma0123,
        // I * gamma01 = gamma0123 * gamma01 = +gamma23 = +B_x.
        // Sign follows from I*F (left-multiply by pseudoscalar).
        assert_relative_eq!(b_dual[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(e_dual[0], 0.0, epsilon = 1e-10);
    }
}
