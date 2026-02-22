//! Python bindings for the Algebra type.

use pyo3::prelude::*;
use std::sync::Arc;

use crate::algebra::{Algebra, Signature};

/// A geometric algebra Cl(p,q,r) with pre-computed product tables.
///
/// This is the main entry point for working with non-Euclidean algebras
/// like PGA (projective), CGA (conformal), and STA (spacetime).
///
/// # Examples
/// ```python
/// # Create a Euclidean 3D algebra
/// R3 = Algebra.euclidean(3)
///
/// # Create PGA3D (Projective Geometric Algebra)
/// PGA3D = Algebra.pga(3)
///
/// # Create CGA3D (Conformal Geometric Algebra)
/// CGA3D = Algebra.cga(3)
///
/// # Create Spacetime Algebra
/// STA = Algebra.sta()
///
/// # Create arbitrary signature Cl(p,q,r)
/// custom = Algebra(2, 1, 1)  # 2 positive, 1 negative, 1 degenerate
/// ```
#[pyclass(name = "Algebra", frozen)]
#[derive(Clone)]
pub struct PyAlgebra {
    pub inner: Arc<Algebra>,
}

#[pymethods]
impl PyAlgebra {
    /// Create an algebra with signature Cl(p, q, r).
    ///
    /// Args:
    ///     p: Number of basis vectors that square to +1
    ///     q: Number of basis vectors that square to -1
    ///     r: Number of basis vectors that square to 0 (degenerate)
    ///
    /// Example:
    /// ```python
    /// # Minkowski space Cl(1,3)
    /// minkowski = Algebra(1, 3, 0)
    ///
    /// # PGA3D as Cl(3,0,1)
    /// pga3d = Algebra(3, 0, 1)
    /// ```
    #[new]
    #[pyo3(signature = (p, q=0, r=0))]
    pub fn new(p: usize, q: usize, r: usize) -> Self {
        let sig = Signature::new(p, q, r);
        PyAlgebra {
            inner: Algebra::new(sig),
        }
    }

    /// Create a Euclidean algebra Cl(n,0,0).
    ///
    /// All basis vectors square to +1.
    ///
    /// Example:
    /// ```python
    /// R3 = Algebra.euclidean(3)
    /// print(R3)  # Cl(3,0,0)
    /// ```
    #[staticmethod]
    pub fn euclidean(n: usize) -> Self {
        PyAlgebra {
            inner: Algebra::euclidean(n),
        }
    }

    /// Create a Projective Geometric Algebra Cl(n,0,1).
    ///
    /// PGA adds one degenerate (null) basis vector for representing
    /// points, lines, and planes in projective space.
    ///
    /// Example:
    /// ```python
    /// PGA3D = Algebra.pga(3)  # Cl(3,0,1), dimension 4
    /// print(PGA3D.dimension)  # 4
    /// ```
    #[staticmethod]
    pub fn pga(n: usize) -> Self {
        PyAlgebra {
            inner: Algebra::pga(n),
        }
    }

    /// Create a Conformal Geometric Algebra Cl(n+1,1,0).
    ///
    /// CGA adds two basis vectors (one positive, one negative) for
    /// representing spheres, circles, and conformal transformations.
    ///
    /// Example:
    /// ```python
    /// CGA3D = Algebra.cga(3)  # Cl(4,1,0), dimension 5
    /// print(CGA3D.dimension)  # 5
    /// ```
    #[staticmethod]
    pub fn cga(n: usize) -> Self {
        PyAlgebra {
            inner: Algebra::cga(n),
        }
    }

    /// Create the Spacetime Algebra Cl(1,3,0).
    ///
    /// Uses the "mostly minus" convention: one timelike (+1) and
    /// three spacelike (-1) basis vectors.
    ///
    /// Example:
    /// ```python
    /// STA = Algebra.sta()
    /// print(STA)  # Cl(1,3,0)
    /// ```
    #[staticmethod]
    pub fn sta() -> Self {
        PyAlgebra {
            inner: Algebra::sta(),
        }
    }

    /// The dimension of the underlying vector space.
    #[getter]
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// The total number of basis blades (2^dimension).
    #[getter]
    pub fn num_blades(&self) -> usize {
        self.inner.num_blades()
    }

    /// Number of basis vectors that square to +1.
    #[getter]
    pub fn p(&self) -> usize {
        self.inner.signature.p
    }

    /// Number of basis vectors that square to -1.
    #[getter]
    pub fn q(&self) -> usize {
        self.inner.signature.q
    }

    /// Number of degenerate basis vectors (square to 0).
    #[getter]
    pub fn r(&self) -> usize {
        self.inner.signature.r
    }

    /// Whether this is a Euclidean algebra (q=0, r=0).
    pub fn is_euclidean(&self) -> bool {
        self.inner.signature.is_euclidean()
    }

    /// Whether this is a PGA algebra (r=1, q=0).
    pub fn is_pga(&self) -> bool {
        self.inner.signature.is_pga()
    }

    /// Get the name of a basis blade by index.
    ///
    /// Example:
    /// ```python
    /// R3 = Algebra.euclidean(3)
    /// print(R3.blade_name(0))  # "1" (scalar)
    /// print(R3.blade_name(1))  # "e1"
    /// print(R3.blade_name(7))  # "e123" (pseudoscalar)
    /// ```
    pub fn blade_name(&self, index: usize) -> String {
        self.inner.blade_name(index).to_string()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner.signature)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner.signature)
    }

    fn __eq__(&self, other: &PyAlgebra) -> bool {
        self.inner.signature == other.inner.signature
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.signature.hash(&mut hasher);
        hasher.finish()
    }
}
