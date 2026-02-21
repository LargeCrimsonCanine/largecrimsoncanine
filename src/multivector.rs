use pyo3::prelude::*;
use crate::algebra;

/// A multivector in the Clifford algebra Cl(n).
///
/// A multivector is the fundamental object of geometric algebra.
/// It is a sum of components of different grades: scalars (grade 0),
/// vectors (grade 1), bivectors (grade 2), and so on.
///
/// Internally stored as a flat coefficient array indexed by basis blade.
/// Basis blade ordering follows the canonical binary representation:
/// index 0 = scalar, index 1 = e1, index 2 = e2, index 3 = e12, etc.
///
/// Current limitation: assumes Euclidean metric (all basis vectors square to +1).
/// Mixed-signature algebra Cl(p,q,r) support is planned. See ARCHITECTURE.md.
///
/// Reference: Dorst et al. ch.2 [VERIFY]
#[pyclass]
#[derive(Debug, Clone)]
pub struct Multivector {
    /// Coefficients for each basis blade.
    /// Length is always 2^n where n is the dimension of the base vector space.
    pub coeffs: Vec<f64>,

    /// Dimension of the base vector space.
    pub dims: usize,
}

#[pymethods]
impl Multivector {
    /// Create a new multivector from a coefficient list.
    ///
    /// The length of coeffs must be a power of 2.
    /// coeffs[0] is the scalar part, coeffs[1] is e1, coeffs[2] is e2,
    /// coeffs[3] is e12, and so on.
    ///
    /// # Errors
    /// Returns an error if coeffs is empty or its length is not a power of 2.
    #[new]
    pub fn new(coeffs: Vec<f64>) -> PyResult<Self> {
        let len = coeffs.len();
        if len == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "coeffs must not be empty",
            ));
        }
        if len & (len - 1) != 0 {
            let lower = 1usize << (usize::BITS - len.leading_zeros() - 1);
            let upper = lower << 1;
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coeffs length must be a power of 2 (got {}, expected {} or {})",
                len, lower, upper
            )));
        }
        let dims = (len as f64).log2() as usize;
        Ok(Multivector { coeffs, dims })
    }

    /// Return the scalar (grade-0) part of this multivector.
    pub fn scalar(&self) -> f64 {
        self.coeffs[0]
    }

    /// Project onto a specific grade.
    ///
    /// Returns a new multivector containing only the grade-k components.
    /// Components not of grade k are zeroed.
    ///
    /// # Errors
    /// Returns an error if k exceeds the dimension of the algebra.
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn grade(&self, k: usize) -> PyResult<Self> {
        if k > self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "grade {} exceeds algebra dimension {} (max grade is {})",
                k, self.dims, self.dims
            )));
        }
        let mut result = vec![0.0f64; self.coeffs.len()];
        for (i, &c) in self.coeffs.iter().enumerate() {
            if algebra::blade_grade(i) == k {
                result[i] = c;
            }
        }
        Ok(Multivector { coeffs: result, dims: self.dims })
    }

    /// Compute the geometric product of two multivectors.
    ///
    /// The geometric product is the fundamental operation of Clifford algebra.
    /// For orthonormal basis vectors e_i: e_i * e_i = 1, e_i * e_j = -e_j * e_i for i != j.
    ///
    /// Both multivectors must have the same dimension.
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn geometric_product(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left is Cl({}), right is Cl({})",
                self.dims, other.dims
            )));
        }
        let size = self.coeffs.len();
        let mut result = vec![0.0f64; size];

        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0.0 { continue; }
            for (j, &b) in other.coeffs.iter().enumerate() {
                if b == 0.0 { continue; }
                let (blade, sign) = algebra::blade_product(i, j);
                result[blade] += sign * a * b;
            }
        }

        Ok(Multivector { coeffs: result, dims: self.dims })
    }

    /// Compute the outer (wedge) product of two multivectors.
    ///
    /// The outer product increases grade: the outer product of a grade-r
    /// and grade-s multivector is a grade-(r+s) multivector.
    /// It is zero if any basis vector appears more than once.
    ///
    /// Both multivectors must have the same dimension.
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn outer_product(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left is Cl({}), right is Cl({})",
                self.dims, other.dims
            )));
        }
        let size = self.coeffs.len();
        let mut result = vec![0.0f64; size];

        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0.0 { continue; }
            for (j, &b) in other.coeffs.iter().enumerate() {
                if b == 0.0 { continue; }
                // Outer product is zero when blades share a basis vector.
                if i & j != 0 { continue; }
                let (blade, sign) = algebra::blade_product(i, j);
                result[blade] += sign * a * b;
            }
        }

        Ok(Multivector { coeffs: result, dims: self.dims })
    }

    /// Compute the left contraction (inner product) of two multivectors.
    ///
    /// The left contraction A ⌋ B measures how much of B is "along" A.
    /// For blades of grade r and s: A_r ⌋ B_s = ⟨A_r B_s⟩_{s-r} if s >= r, else 0.
    ///
    /// This is the grade-lowering part of the geometric product.
    /// The left contraction of a vector with another vector gives their scalar product.
    ///
    /// Both multivectors must have the same dimension.
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn left_contraction(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left is Cl({}), right is Cl({})",
                self.dims, other.dims
            )));
        }
        let size = self.coeffs.len();
        let mut result = vec![0.0f64; size];

        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0.0 { continue; }
            let grade_a = algebra::blade_grade(i);
            for (j, &b) in other.coeffs.iter().enumerate() {
                if b == 0.0 { continue; }
                let grade_b = algebra::blade_grade(j);
                // Left contraction is zero when grade(A) > grade(B)
                if grade_a > grade_b { continue; }
                let (blade, sign) = algebra::blade_product(i, j);
                let result_grade = algebra::blade_grade(blade);
                // Only keep the grade (s-r) component
                if result_grade == grade_b - grade_a {
                    result[blade] += sign * a * b;
                }
            }
        }

        Ok(Multivector { coeffs: result, dims: self.dims })
    }

    /// Alias for left_contraction.
    pub fn inner(&self, other: &Multivector) -> PyResult<Self> {
        self.left_contraction(other)
    }

    /// Return the coefficient array as a Python list.
    pub fn to_list(&self) -> Vec<f64> {
        self.coeffs.clone()
    }

    pub fn __repr__(&self) -> String {
        format!("Multivector({:?})", self.coeffs)
    }

    pub fn __mul__(&self, other: &Multivector) -> PyResult<Self> {
        self.geometric_product(other)
    }

    pub fn __xor__(&self, other: &Multivector) -> PyResult<Self> {
        self.outer_product(other)
    }

    /// Add two multivectors component-wise.
    ///
    /// Both multivectors must have the same dimension.
    pub fn __add__(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left is Cl({}), right is Cl({})",
                self.dims, other.dims
            )));
        }
        let coeffs: Vec<f64> = self.coeffs.iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Multivector { coeffs, dims: self.dims })
    }

    /// Subtract two multivectors component-wise.
    ///
    /// Both multivectors must have the same dimension.
    pub fn __sub__(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left is Cl({}), right is Cl({})",
                self.dims, other.dims
            )));
        }
        let coeffs: Vec<f64> = self.coeffs.iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Multivector { coeffs, dims: self.dims })
    }

    /// Negate a multivector (flip sign of all coefficients).
    pub fn __neg__(&self) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|c| -c).collect();
        Multivector { coeffs, dims: self.dims }
    }

    /// Multiply all coefficients by a scalar.
    pub fn scale(&self, scalar: f64) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|c| c * scalar).collect();
        Multivector { coeffs, dims: self.dims }
    }

    /// Check equality of two multivectors.
    ///
    /// Uses exact floating-point comparison. For approximate comparison,
    /// use `approx_eq` with a tolerance.
    pub fn __eq__(&self, other: &Multivector) -> bool {
        self.dims == other.dims && self.coeffs == other.coeffs
    }

    /// Check approximate equality within a tolerance.
    ///
    /// Returns true if all coefficients differ by at most `tol`.
    #[pyo3(signature = (other, tol=1e-10))]
    pub fn approx_eq(&self, other: &Multivector, tol: f64) -> bool {
        if self.dims != other.dims {
            return false;
        }
        self.coeffs.iter()
            .zip(other.coeffs.iter())
            .all(|(a, b)| (a - b).abs() <= tol)
    }

    /// Compute the reverse of this multivector.
    ///
    /// The reverse operation flips the order of basis vectors in each blade.
    /// For a blade of grade k, this introduces a sign factor of (-1)^(k(k-1)/2).
    ///
    /// The reverse is useful for computing norms and inverses:
    /// - For a vector v: v * ~v = |v|^2 (a scalar)
    /// - For a rotor R: R * ~R = 1
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn reverse(&self) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter()
            .enumerate()
            .map(|(i, &c)| {
                let grade = algebra::blade_grade(i);
                c * algebra::reverse_sign(grade)
            })
            .collect();
        Multivector { coeffs, dims: self.dims }
    }

    /// Alias for reverse, using the tilde notation common in GA literature.
    pub fn tilde(&self) -> Self {
        self.reverse()
    }

    /// Compute the squared norm of this multivector.
    ///
    /// For a multivector A: |A|² = ⟨A ~A⟩₀ (scalar part of A times its reverse).
    ///
    /// This is always a scalar. For Euclidean signature, it is non-negative.
    /// For mixed-signature algebras, it may be negative.
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn norm_squared(&self) -> f64 {
        // Compute A * ~A and extract scalar part
        let rev = self.reverse();
        let mut scalar = 0.0f64;

        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0.0 { continue; }
            for (j, &b) in rev.coeffs.iter().enumerate() {
                if b == 0.0 { continue; }
                let (blade, sign) = algebra::blade_product(i, j);
                // Only accumulate scalar part (blade index 0)
                if blade == 0 {
                    scalar += sign * a * b;
                }
            }
        }

        scalar
    }

    /// Compute the norm (magnitude) of this multivector.
    ///
    /// Returns sqrt(|norm_squared|). The absolute value handles mixed-signature
    /// algebras where norm_squared may be negative.
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn norm(&self) -> f64 {
        self.norm_squared().abs().sqrt()
    }

    /// Return a normalized copy of this multivector (unit norm).
    ///
    /// Returns None if the norm is zero (cannot normalize).
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn normalize(&self) -> Option<Self> {
        let n = self.norm();
        if n == 0.0 {
            None
        } else {
            Some(self.scale(1.0 / n))
        }
    }

    /// Return a normalized copy, or raise an error if norm is zero.
    ///
    /// This is a Python-friendly version that raises ValueError instead of returning None.
    pub fn normalized(&self) -> PyResult<Self> {
        self.normalize().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("cannot normalize zero multivector")
        })
    }
}

