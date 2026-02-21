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
    #[new]
    pub fn new(coeffs: Vec<f64>) -> PyResult<Self> {
        let len = coeffs.len();
        if len == 0 || (len & (len - 1)) != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "coeffs length must be a power of 2",
            ));
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
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn grade(&self, k: usize) -> PyResult<Self> {
        if k > self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "grade exceeds dimension",
            ));
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
    /// This implementation assumes a Euclidean metric (all basis vectors square to +1).
    /// Support for mixed-signature algebras is planned.
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn geometric_product(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "multivectors must have the same dimension",
            ));
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
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn outer_product(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "multivectors must have the same dimension",
            ));
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
}

