//! Conformal Geometric Algebra (CGA) convenience methods.
//!
//! CGA uses two extra basis vectors (e+, e-) to conformally embed
//! Euclidean space. This allows spheres, circles, and planes to be
//! represented as simple blades.
//!
//! # Conventions (Dorst "Geometric Algebra for Computer Science" Chapter 13)
//!
//! CGA3D is Cl(4,1,0) - 4 positive + 1 negative basis vectors.
//! Standard basis: e1, e2, e3, e+ (or e4), e- (or e5).
//!
//! The null vectors are defined as:
//! - e_o = (e- - e+)/2  (origin, represents the point at origin)
//! - e_∞ = e- + e+      (infinity, represents the point at infinity)
//!
//! These satisfy:
//! - e_o · e_o = 0  (null)
//! - e_∞ · e_∞ = 0  (null)
//! - e_o · e_∞ = -1
//!
//! # Point Embedding
//!
//! A Euclidean point x = (x, y, z) is embedded as:
//! P = x·e1 + y·e2 + z·e3 + 0.5·|x|²·e_∞ + e_o
//!
//! This is called the "standard form" or "homogeneous form".
//!
//! # Geometric Objects
//!
//! - **Point**: P = x + 0.5*|x|²*e∞ + eo
//! - **Sphere**: S = P - 0.5*r²*e∞ (where P is center point)
//! - **Plane**: π = n + d*e∞ (where n is unit normal, d is signed distance from origin)
//! - **Circle**: C = S1 ∧ S2 (intersection of two spheres)
//! - **Line**: L = π1 ∧ π2 (intersection of two planes)
//! - **Point pair**: PP = P1 ∧ P2

use crate::multivector::Multivector;
use crate::pyalgebra::PyAlgebra;
use pyo3::prelude::*;

/// CGA helper functions (internal)
impl Multivector {
    /// Internal: Get the index of e+ (positive extra basis) in CGA.
    /// In Cl(n+1,1,0), e+ is at index 2^n (the (n+1)th basis vector).
    fn cga_eplus_index(n: usize) -> usize {
        1 << n // e.g., for n=3: index 8 (e4)
    }

    /// Internal: Get the index of e- (negative extra basis) in CGA.
    /// In Cl(n+1,1,0), e- is at index 2^(n+1) (the (n+2)th basis vector).
    fn cga_eminus_index(n: usize) -> usize {
        1 << (n + 1) // e.g., for n=3: index 16 (e5)
    }
}

/// CGA-specific constructors and methods
#[pymethods]
impl Multivector {
    // ========================================================================
    // CGA NULL VECTOR CONSTRUCTORS
    // ========================================================================

    /// Create the origin null vector e_o in CGA.
    ///
    /// The origin is defined as: e_o = (e- - e+) / 2
    ///
    /// In CGA, the origin represents the point at (0,0,0).
    ///
    /// # Arguments
    /// * `algebra` - A CGA algebra created with `Algebra.cga(n)`
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// eo = Multivector.cga_eo(CGA3D)
    /// # eo represents the origin point
    /// ```
    #[staticmethod]
    pub fn cga_eo(algebra: &PyAlgebra) -> PyResult<Self> {
        // Verify this is a CGA algebra (q=1, r=0)
        if algebra.inner.signature.q != 1 || algebra.inner.signature.r != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_eo requires a CGA algebra (q=1, r=0); use Algebra.cga(n)"
            ));
        }

        let dim = algebra.inner.dimension();
        let n = dim - 2; // Euclidean dimension
        let size = algebra.inner.num_blades();
        let mut coeffs = vec![0.0; size];

        // e_o = (e- - e+) / 2
        let eplus_idx = Self::cga_eplus_index(n);
        let eminus_idx = Self::cga_eminus_index(n);

        coeffs[eplus_idx] = -0.5;  // -e+ / 2
        coeffs[eminus_idx] = 0.5;   // e- / 2

        Ok(Multivector {
            coeffs,
            dims: dim,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Create the infinity null vector e_∞ in CGA.
    ///
    /// Infinity is defined as: e_∞ = e- + e+
    ///
    /// In CGA, infinity represents the point at infinity.
    ///
    /// # Arguments
    /// * `algebra` - A CGA algebra created with `Algebra.cga(n)`
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// einf = Multivector.cga_einf(CGA3D)
    /// # einf represents the point at infinity
    /// ```
    #[staticmethod]
    pub fn cga_einf(algebra: &PyAlgebra) -> PyResult<Self> {
        // Verify this is a CGA algebra (q=1, r=0)
        if algebra.inner.signature.q != 1 || algebra.inner.signature.r != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_einf requires a CGA algebra (q=1, r=0); use Algebra.cga(n)"
            ));
        }

        let dim = algebra.inner.dimension();
        let n = dim - 2; // Euclidean dimension
        let size = algebra.inner.num_blades();
        let mut coeffs = vec![0.0; size];

        // e_∞ = e- + e+
        let eplus_idx = Self::cga_eplus_index(n);
        let eminus_idx = Self::cga_eminus_index(n);

        coeffs[eplus_idx] = 1.0;   // e+
        coeffs[eminus_idx] = 1.0;  // e-

        Ok(Multivector {
            coeffs,
            dims: dim,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    // ========================================================================
    // CGA GEOMETRIC OBJECT CONSTRUCTORS
    // ========================================================================

    /// Create a conformal point from Euclidean coordinates.
    ///
    /// The conformal embedding of a point x is:
    /// P = x + 0.5*|x|²*e∞ + eo
    ///
    /// This is the standard "up" mapping from Euclidean to conformal space.
    ///
    /// # Arguments
    /// * `algebra` - A CGA algebra created with `Algebra.cga(n)`
    /// * `x`, `y`, `z` - Euclidean coordinates (for 3D CGA)
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// P = Multivector.cga_point(CGA3D, 1.0, 2.0, 3.0)
    /// # P represents the Euclidean point (1, 2, 3)
    /// ```
    #[staticmethod]
    pub fn cga_point(algebra: &PyAlgebra, x: f64, y: f64, z: f64) -> PyResult<Self> {
        // Verify this is a 3D CGA algebra
        if algebra.inner.signature.q != 1 || algebra.inner.signature.r != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_point requires a CGA algebra; use Algebra.cga(3)"
            ));
        }
        let dim = algebra.inner.dimension();
        if dim != 5 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_point(x,y,z) requires CGA3D (dimension 5); use Algebra.cga(3)"
            ));
        }

        let size = algebra.inner.num_blades();
        let mut coeffs = vec![0.0; size];
        let n = 3; // Euclidean dimension

        // Euclidean part: x*e1 + y*e2 + z*e3
        coeffs[1] = x;  // e1
        coeffs[2] = y;  // e2
        coeffs[4] = z;  // e3

        // |x|² = x² + y² + z²
        let norm_sq = x * x + y * y + z * z;

        // e_o = (e- - e+) / 2
        // e_∞ = e- + e+
        // P = x + 0.5*|x|²*e_∞ + e_o
        //   = x + 0.5*|x|²*(e- + e+) + (e- - e+)/2
        //   = x + (0.5*|x|² + 0.5)*e- + (0.5*|x|² - 0.5)*e+
        let eplus_idx = Self::cga_eplus_index(n);   // index 8
        let eminus_idx = Self::cga_eminus_index(n); // index 16

        coeffs[eplus_idx] = 0.5 * norm_sq - 0.5;   // e+ coefficient
        coeffs[eminus_idx] = 0.5 * norm_sq + 0.5;  // e- coefficient

        Ok(Multivector {
            coeffs,
            dims: dim,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Create a conformal point from a vector of Euclidean coordinates.
    ///
    /// This is a more general version of cga_point that works for any dimension.
    ///
    /// # Arguments
    /// * `algebra` - A CGA algebra created with `Algebra.cga(n)`
    /// * `coords` - List of Euclidean coordinates [x, y, ...] of length n
    ///
    /// # Example
    /// ```python
    /// CGA2D = Algebra.cga(2)
    /// P = Multivector.cga_point_from_coords(CGA2D, [1.0, 2.0])
    /// # P represents the 2D point (1, 2)
    /// ```
    #[staticmethod]
    pub fn cga_point_from_coords(algebra: &PyAlgebra, coords: Vec<f64>) -> PyResult<Self> {
        // Verify this is a CGA algebra
        if algebra.inner.signature.q != 1 || algebra.inner.signature.r != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_point_from_coords requires a CGA algebra; use Algebra.cga(n)"
            ));
        }

        let dim = algebra.inner.dimension();
        let n = dim - 2; // Euclidean dimension

        if coords.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coords length {} doesn't match Euclidean dimension {}; \
                for CGA{} use {} coordinates",
                coords.len(), n, n, n
            )));
        }

        let size = algebra.inner.num_blades();
        let mut coeffs = vec![0.0; size];

        // Euclidean part
        let mut norm_sq = 0.0;
        for (i, &c) in coords.iter().enumerate() {
            coeffs[1 << i] = c;
            norm_sq += c * c;
        }

        // CGA part
        let eplus_idx = Self::cga_eplus_index(n);
        let eminus_idx = Self::cga_eminus_index(n);

        coeffs[eplus_idx] = 0.5 * norm_sq - 0.5;
        coeffs[eminus_idx] = 0.5 * norm_sq + 0.5;

        Ok(Multivector {
            coeffs,
            dims: dim,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Create a sphere in CGA from center coordinates and radius.
    ///
    /// A sphere is represented as: S = P - 0.5*r²*e∞
    /// where P is the conformal point at the center.
    ///
    /// When r² > 0, this is a real sphere.
    /// When r² < 0, this is an imaginary sphere (useful for inversions).
    /// When r² = 0, this reduces to a point.
    ///
    /// # Arguments
    /// * `algebra` - A CGA algebra created with `Algebra.cga(3)`
    /// * `cx`, `cy`, `cz` - Center coordinates
    /// * `r` - Radius of the sphere
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// S = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)
    /// # S is a unit sphere centered at origin
    /// ```
    #[staticmethod]
    pub fn cga_sphere(algebra: &PyAlgebra, cx: f64, cy: f64, cz: f64, r: f64) -> PyResult<Self> {
        // Start with the center point
        let mut sphere = Self::cga_point(algebra, cx, cy, cz)?;

        // S = P - 0.5*r²*e∞
        // e∞ = e+ + e-
        let n = 3;
        let eplus_idx = Self::cga_eplus_index(n);
        let eminus_idx = Self::cga_eminus_index(n);

        let r_sq_half = 0.5 * r * r;
        sphere.coeffs[eplus_idx] -= r_sq_half;
        sphere.coeffs[eminus_idx] -= r_sq_half;

        Ok(sphere)
    }

    /// Create a plane in CGA from the equation ax + by + cz + d = 0.
    ///
    /// A plane is represented as: π = n + d*e∞
    /// where n = a*e1 + b*e2 + c*e3 is the (not necessarily unit) normal.
    ///
    /// For a normalized plane (|n| = 1), d is the signed distance from origin.
    ///
    /// # Arguments
    /// * `algebra` - A CGA algebra created with `Algebra.cga(3)`
    /// * `a`, `b`, `c` - Normal vector components
    /// * `d` - Constant term (distance from origin when normal is unit)
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// # xy-plane (z = 0)
    /// xy_plane = Multivector.cga_plane(CGA3D, 0.0, 0.0, 1.0, 0.0)
    /// # Plane z = 5
    /// z5_plane = Multivector.cga_plane(CGA3D, 0.0, 0.0, 1.0, -5.0)
    /// ```
    #[staticmethod]
    pub fn cga_plane(algebra: &PyAlgebra, a: f64, b: f64, c: f64, d: f64) -> PyResult<Self> {
        // Verify this is a 3D CGA algebra
        if algebra.inner.signature.q != 1 || algebra.inner.signature.r != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_plane requires a CGA algebra; use Algebra.cga(3)"
            ));
        }
        let dim = algebra.inner.dimension();
        if dim != 5 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_plane requires CGA3D (dimension 5); use Algebra.cga(3)"
            ));
        }

        let size = algebra.inner.num_blades();
        let mut coeffs = vec![0.0; size];
        let n = 3;

        // Normal vector part: a*e1 + b*e2 + c*e3
        coeffs[1] = a;  // e1
        coeffs[2] = b;  // e2
        coeffs[4] = c;  // e3

        // Add d*e∞ = d*(e+ + e-)
        let eplus_idx = Self::cga_eplus_index(n);
        let eminus_idx = Self::cga_eminus_index(n);

        coeffs[eplus_idx] = d;
        coeffs[eminus_idx] = d;

        Ok(Multivector {
            coeffs,
            dims: dim,
            algebra_opt: Some(algebra.inner.clone()),
        })
    }

    /// Create a circle as the intersection of two spheres.
    ///
    /// In CGA, a circle is the wedge product of two spheres: C = S1 ∧ S2.
    /// The resulting circle lies on both spheres.
    ///
    /// # Arguments
    /// * `s1` - First sphere
    /// * `s2` - Second sphere
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// S1 = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)
    /// S2 = Multivector.cga_sphere(CGA3D, 1.0, 0.0, 0.0, 1.0)
    /// C = Multivector.cga_circle(S1, S2)
    /// # C is the circle where the two unit spheres intersect
    /// ```
    #[staticmethod]
    pub fn cga_circle(s1: &Multivector, s2: &Multivector) -> PyResult<Self> {
        s1.outer_product(s2)
    }

    /// Create a point pair from two conformal points.
    ///
    /// A point pair is the wedge product of two points: PP = P1 ∧ P2.
    /// It represents the zero-dimensional object consisting of two points.
    ///
    /// # Arguments
    /// * `p1` - First conformal point
    /// * `p2` - Second conformal point
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// P1 = Multivector.cga_point(CGA3D, 0.0, 0.0, 0.0)
    /// P2 = Multivector.cga_point(CGA3D, 1.0, 0.0, 0.0)
    /// PP = Multivector.cga_point_pair(P1, P2)
    /// ```
    #[staticmethod]
    pub fn cga_point_pair(p1: &Multivector, p2: &Multivector) -> PyResult<Self> {
        p1.outer_product(p2)
    }

    /// Create a line as the intersection of two planes.
    ///
    /// In CGA, a line can be constructed as the wedge of two planes: L = π1 ∧ π2.
    ///
    /// # Arguments
    /// * `plane1` - First plane
    /// * `plane2` - Second plane
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// xz_plane = Multivector.cga_plane(CGA3D, 0.0, 1.0, 0.0, 0.0)  # y = 0
    /// xy_plane = Multivector.cga_plane(CGA3D, 0.0, 0.0, 1.0, 0.0)  # z = 0
    /// x_axis = Multivector.cga_line(xz_plane, xy_plane)  # x-axis
    /// ```
    #[staticmethod]
    pub fn cga_line(plane1: &Multivector, plane2: &Multivector) -> PyResult<Self> {
        plane1.outer_product(plane2)
    }

    // ========================================================================
    // CGA EXTRACTION METHODS
    // ========================================================================

    /// Extract the center of a sphere or circle in CGA.
    ///
    /// For a sphere S = P - 0.5*r²*e∞, this returns the Euclidean coordinates
    /// of the center point P.
    ///
    /// # Returns
    /// A tuple (x, y, z) representing the center coordinates.
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// S = Multivector.cga_sphere(CGA3D, 1.0, 2.0, 3.0, 5.0)
    /// center = S.cga_extract_center()
    /// # center == (1.0, 2.0, 3.0)
    /// ```
    pub fn cga_extract_center(&self) -> PyResult<(f64, f64, f64)> {
        // Verify this is a CGA multivector
        let algebra = self.algebra_opt.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "cga_extract_center requires a CGA multivector with explicit algebra"
            )
        })?;

        if algebra.signature.q != 1 || algebra.signature.r != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_extract_center requires a CGA multivector"
            ));
        }

        let dim = algebra.dimension();
        if dim != 5 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_extract_center currently only supports CGA3D (dimension 5)"
            ));
        }

        let n = 3; // Euclidean dimension
        let eplus_idx = Self::cga_eplus_index(n);
        let eminus_idx = Self::cga_eminus_index(n);

        // For a point/sphere, we need to normalize by the e_∞·X component
        // e_∞ = e+ + e-
        // The coefficient of e_∞ in X is related to -(X · e_o)
        // For a normalized point P = x + 0.5|x|²e_∞ + e_o,
        // -e_∞ · P = -e_∞ · e_o = 1

        // Actually, for sphere S, the e- and e+ components encode both center and radius
        // S = x + (0.5|x|² - 0.5r²)e_∞ + e_o
        //   = x + (0.5|x|² - 0.5r² + 0.5)e- + (0.5|x|² - 0.5r² - 0.5)e+

        // The weight (normalization factor) for a CGA vector is -einf · X
        // For our representation, einf = e+ + e-, so
        // weight = -(e+·X + e-·X) = -(X[eplus] * 1 + X[eminus] * (-1))
        // Since e+² = 1 and e-² = -1
        // weight = -X[eplus] + X[eminus]

        let weight = self.coeffs[eminus_idx] - self.coeffs[eplus_idx];

        if weight.abs() < 1e-12 {
            // This might be a plane (weight = 0 at infinity)
            // For a plane, return the normal vector direction scaled by d
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot extract center from a flat object (plane/line); \
                use cga_is_flat() to check first"
            ));
        }

        // Extract Euclidean coordinates (divided by weight)
        let x = self.coeffs[1] / weight;
        let y = self.coeffs[2] / weight;
        let z = self.coeffs[4] / weight;

        Ok((x, y, z))
    }

    /// Extract the radius of a sphere in CGA.
    ///
    /// For a sphere S = P - 0.5*r²*e∞, this returns the radius r.
    /// Returns the radius if real (r² > 0), or the imaginary radius magnitude
    /// with a negative sign if imaginary (r² < 0).
    ///
    /// # Returns
    /// The radius (positive for real spheres, negative for imaginary).
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// S = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 5.0)
    /// r = S.cga_extract_radius()
    /// # r == 5.0
    /// ```
    pub fn cga_extract_radius(&self) -> PyResult<f64> {
        // Verify this is a CGA multivector
        let algebra = self.algebra_opt.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "cga_extract_radius requires a CGA multivector with explicit algebra"
            )
        })?;

        if algebra.signature.q != 1 || algebra.signature.r != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_extract_radius requires a CGA multivector"
            ));
        }

        let dim = algebra.dimension();
        if dim != 5 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_extract_radius currently only supports CGA3D (dimension 5)"
            ));
        }

        // For a sphere, r² = X · X / (e∞ · X)²
        // We compute the squared norm using the metric
        let norm_sq = self.squared_norm_inner()?;

        let n = 3;
        let eplus_idx = Self::cga_eplus_index(n);
        let eminus_idx = Self::cga_eminus_index(n);

        let weight = self.coeffs[eminus_idx] - self.coeffs[eplus_idx];
        let weight_sq = weight * weight;

        if weight_sq.abs() < 1e-24 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot extract radius from a flat object (plane/line)"
            ));
        }

        // r² = |X|² / weight²
        // For conformal objects, |X|² = X · X (using the CGA metric)
        let r_sq = norm_sq / weight_sq;

        if r_sq >= 0.0 {
            Ok(r_sq.sqrt())
        } else {
            // Imaginary sphere - return negative to indicate
            Ok(-(-r_sq).sqrt())
        }
    }

    /// Check if a CGA element represents a round object (sphere, circle, point pair).
    ///
    /// Round objects have positive squared radius (X · X > 0).
    /// Points are a degenerate case with X · X ≈ 0.
    ///
    /// # Returns
    /// `true` if the element is round, `false` if flat.
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// S = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)
    /// assert S.cga_is_round()  # True - sphere is round
    /// ```
    pub fn cga_is_round(&self) -> PyResult<bool> {
        // Check if X · X > 0 (round) vs X · X < 0 (imaginary) vs X · X ≈ 0 (point/flat)
        let norm_sq = self.squared_norm_inner()?;
        Ok(norm_sq > 1e-10)
    }

    /// Check if a CGA element represents a flat object (plane, line, flat point).
    ///
    /// Flat objects have no e_o component in their normalized form,
    /// which means the weight (e∞ · X) is zero or very small.
    ///
    /// # Returns
    /// `true` if the element is flat, `false` if round.
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// plane = Multivector.cga_plane(CGA3D, 0.0, 0.0, 1.0, 0.0)
    /// assert plane.cga_is_flat()  # True - plane is flat
    /// ```
    pub fn cga_is_flat(&self) -> PyResult<bool> {
        // A flat object has zero weight (e∞ · X = 0)
        let algebra = self.algebra_opt.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "cga_is_flat requires a CGA multivector with explicit algebra"
            )
        })?;

        if algebra.signature.q != 1 || algebra.signature.r != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cga_is_flat requires a CGA multivector"
            ));
        }

        let dim = algebra.dimension();
        let n = dim - 2;

        // For vectors (grade-1), check e+ - e- component (weight)
        // Actually for proper check, we need to compute einf · X
        // which for a vector X is the scalar part of einf * X

        // Simple heuristic: flat objects have small weight
        let eplus_idx = Self::cga_eplus_index(n);
        let eminus_idx = Self::cga_eminus_index(n);

        // Weight = eminus - eplus (for grade-1 vectors)
        // For higher grades, this is more complex
        let weight = (self.coeffs[eminus_idx] - self.coeffs[eplus_idx]).abs();

        // Also check the squared norm
        let norm_sq = self.squared_norm_inner()?;

        // Flat if weight is small OR norm_sq is negative (imaginary radius)
        Ok(weight < 1e-10 || norm_sq < -1e-10)
    }

    /// Helper: compute squared norm using the inner product with reverse
    /// This gives X · X = ⟨X X̃⟩_0
    fn squared_norm_inner(&self) -> PyResult<f64> {
        let rev = self.reverse();
        let product = self.geometric_product(&rev)?;
        Ok(product.scalar())
    }

    /// Check if a conformal point lies on a sphere.
    ///
    /// A point P lies on sphere S if and only if P · S = 0.
    ///
    /// # Arguments
    /// * `sphere` - The sphere to test against
    ///
    /// # Returns
    /// `true` if this point lies on the sphere, `false` otherwise.
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// S = Multivector.cga_sphere(CGA3D, 0.0, 0.0, 0.0, 1.0)  # unit sphere
    /// P1 = Multivector.cga_point(CGA3D, 1.0, 0.0, 0.0)  # on sphere
    /// P2 = Multivector.cga_point(CGA3D, 0.5, 0.0, 0.0)  # inside sphere
    /// assert P1.cga_point_on_sphere(S)  # True
    /// assert not P2.cga_point_on_sphere(S)  # False
    /// ```
    pub fn cga_point_on_sphere(&self, sphere: &Multivector) -> PyResult<bool> {
        // P on S iff P · S = 0 (inner product)
        let inner = self.inner(sphere)?;

        // Check if the result is approximately zero
        let is_zero = inner.coeffs.iter().all(|&c| c.abs() < 1e-10);
        Ok(is_zero)
    }

    /// Compute the distance between two conformal points.
    ///
    /// The distance is given by: d(P1, P2) = sqrt(-2 * P1 · P2)
    ///
    /// # Arguments
    /// * `other` - Another conformal point
    ///
    /// # Returns
    /// The Euclidean distance between the two points.
    ///
    /// # Example
    /// ```python
    /// CGA3D = Algebra.cga(3)
    /// P1 = Multivector.cga_point(CGA3D, 0.0, 0.0, 0.0)
    /// P2 = Multivector.cga_point(CGA3D, 3.0, 4.0, 0.0)
    /// d = P1.cga_distance(P2)
    /// # d == 5.0
    /// ```
    pub fn cga_distance(&self, other: &Multivector) -> PyResult<f64> {
        // d² = -2 * (P1 · P2)
        let inner = self.inner(other)?;
        let dot = inner.scalar();
        let d_sq = -2.0 * dot;

        if d_sq < 0.0 {
            // Numerical error - points should give non-negative distance
            if d_sq > -1e-10 {
                Ok(0.0)
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid distance calculation: d² = {} (should be >= 0)",
                    d_sq
                )))
            }
        } else {
            Ok(d_sq.sqrt())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Algebra;

    fn cga3d() -> PyAlgebra {
        PyAlgebra {
            inner: Algebra::cga(3),
        }
    }

    #[test]
    fn test_cga_eo_einf_properties() {
        let alg = cga3d();

        let eo = Multivector::cga_eo(&alg).unwrap();
        let einf = Multivector::cga_einf(&alg).unwrap();

        // eo · eo = 0 (null vector)
        let eo_sq = eo.geometric_product(&eo).unwrap();
        assert!((eo_sq.scalar()).abs() < 1e-10, "eo should be null");

        // einf · einf = 0 (null vector)
        let einf_sq = einf.geometric_product(&einf).unwrap();
        assert!((einf_sq.scalar()).abs() < 1e-10, "einf should be null");

        // eo · einf = -1
        let eo_einf = eo.inner(&einf).unwrap();
        assert!((eo_einf.scalar() + 1.0).abs() < 1e-10, "eo · einf should be -1");
    }

    #[test]
    fn test_cga_point_at_origin() {
        let alg = cga3d();

        let origin = Multivector::cga_point(&alg, 0.0, 0.0, 0.0).unwrap();
        let eo = Multivector::cga_eo(&alg).unwrap();

        // Point at origin should equal eo
        for i in 0..origin.coeffs.len() {
            assert!(
                (origin.coeffs[i] - eo.coeffs[i]).abs() < 1e-10,
                "origin point should equal eo at index {}: {} vs {}",
                i, origin.coeffs[i], eo.coeffs[i]
            );
        }
    }

    #[test]
    fn test_cga_point_embedding() {
        let alg = cga3d();

        // Point at (1, 0, 0)
        let p = Multivector::cga_point(&alg, 1.0, 0.0, 0.0).unwrap();

        // Should have e1 = 1
        assert!((p.coeffs[1] - 1.0).abs() < 1e-10);

        // Should have e2 = 0, e3 = 0
        assert!(p.coeffs[2].abs() < 1e-10);
        assert!(p.coeffs[4].abs() < 1e-10);
    }

    #[test]
    fn test_cga_sphere_center_extraction() {
        let alg = cga3d();

        let center = (1.0, 2.0, 3.0);
        let radius = 5.0;

        let sphere = Multivector::cga_sphere(&alg, center.0, center.1, center.2, radius).unwrap();
        let extracted = sphere.cga_extract_center().unwrap();

        assert!((extracted.0 - center.0).abs() < 1e-10);
        assert!((extracted.1 - center.1).abs() < 1e-10);
        assert!((extracted.2 - center.2).abs() < 1e-10);
    }

    #[test]
    fn test_cga_sphere_radius_extraction() {
        let alg = cga3d();

        let sphere = Multivector::cga_sphere(&alg, 1.0, 2.0, 3.0, 5.0).unwrap();
        let r = sphere.cga_extract_radius().unwrap();

        assert!((r - 5.0).abs() < 1e-10, "extracted radius {} should be 5.0", r);
    }

    #[test]
    fn test_cga_point_distance() {
        let alg = cga3d();

        let p1 = Multivector::cga_point(&alg, 0.0, 0.0, 0.0).unwrap();
        let p2 = Multivector::cga_point(&alg, 3.0, 4.0, 0.0).unwrap();

        let d = p1.cga_distance(&p2).unwrap();
        assert!((d - 5.0).abs() < 1e-10, "distance should be 5.0, got {}", d);
    }

    #[test]
    fn test_cga_point_on_sphere() {
        let alg = cga3d();

        // Unit sphere at origin
        let sphere = Multivector::cga_sphere(&alg, 0.0, 0.0, 0.0, 1.0).unwrap();

        // Point on sphere
        let p_on = Multivector::cga_point(&alg, 1.0, 0.0, 0.0).unwrap();
        assert!(p_on.cga_point_on_sphere(&sphere).unwrap());

        // Point inside sphere
        let p_inside = Multivector::cga_point(&alg, 0.5, 0.0, 0.0).unwrap();
        assert!(!p_inside.cga_point_on_sphere(&sphere).unwrap());
    }

    #[test]
    fn test_cga_plane() {
        let alg = cga3d();

        // xy-plane (z = 0)
        let plane = Multivector::cga_plane(&alg, 0.0, 0.0, 1.0, 0.0).unwrap();

        // Should have e3 = 1
        assert!((plane.coeffs[4] - 1.0).abs() < 1e-10);

        // Should have no e1, e2
        assert!(plane.coeffs[1].abs() < 1e-10);
        assert!(plane.coeffs[2].abs() < 1e-10);
    }
}
