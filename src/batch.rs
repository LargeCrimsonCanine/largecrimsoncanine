//! Batched operations on arrays of multivectors.
//!
//! Enables operations like rotating 1000 vectors at once:
//! ```python
//! points = MultivectorBatch.from_vectors(algebra, np.random.randn(1000, 3))
//! rotated = rotor.sandwich_batch(points)
//! ```
//!
//! This module provides NumPy-backed batched operations for high-performance
//! geometric algebra computations on large datasets.

use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::sync::Arc;

use crate::algebra::{self, Algebra};
use crate::multivector::Multivector;
use crate::pyalgebra::PyAlgebra;

/// A batch of multivectors sharing the same algebra.
///
/// This allows efficient vectorized operations on thousands of multivectors
/// simultaneously, with data stored as a contiguous NumPy-compatible array.
///
/// # Examples
/// ```python
/// import numpy as np
/// from largecrimsoncanine import Algebra, MultivectorBatch
///
/// R3 = Algebra.euclidean(3)
/// coords = np.random.randn(1000, 3)  # 1000 random 3D vectors
/// batch = MultivectorBatch.from_vectors(R3, coords)
///
/// # Apply a rotation to all vectors at once
/// rotor = Multivector.from_axis_angle([0, 0, 1], 0.5)
/// rotated = batch.sandwich(rotor)
/// ```
#[pyclass(name = "MultivectorBatch")]
#[derive(Debug, Clone)]
pub struct PyMultivectorBatch {
    /// Coefficients for each multivector in the batch.
    /// Shape: (num_instances, num_blades)
    pub coeffs: Array2<f64>,
    /// The algebra shared by all multivectors in this batch.
    pub algebra: Arc<Algebra>,
}

impl PyMultivectorBatch {
    /// Create a new batch from raw coefficient array and algebra.
    pub fn new(coeffs: Array2<f64>, algebra: Arc<Algebra>) -> Self {
        Self { coeffs, algebra }
    }
}

#[pymethods]
impl PyMultivectorBatch {
    /// Create a batch from a NumPy array of coefficients.
    ///
    /// Args:
    ///     algebra: The algebra for all multivectors in the batch.
    ///     coeffs: NumPy array of shape (N, num_blades) where N is the batch size
    ///             and num_blades = 2^dimension.
    ///
    /// Returns:
    ///     A MultivectorBatch containing N multivectors.
    ///
    /// Raises:
    ///     ValueError: If coeffs has wrong number of columns for the algebra.
    ///
    /// Example:
    /// ```python
    /// R3 = Algebra.euclidean(3)
    /// # Create 100 random multivectors in 3D (8 blades each)
    /// coeffs = np.random.randn(100, 8)
    /// batch = MultivectorBatch.from_numpy(R3, coeffs)
    /// ```
    #[staticmethod]
    pub fn from_numpy(algebra: &PyAlgebra, coeffs: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let arr = coeffs.as_array();
        let num_blades = algebra.inner.num_blades();

        if arr.ncols() != num_blades {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coeffs must have {} columns for {} but got {}",
                num_blades, algebra.inner.signature, arr.ncols()
            )));
        }

        Ok(Self {
            coeffs: arr.to_owned(),
            algebra: algebra.inner.clone(),
        })
    }

    /// Create a batch of vectors from coordinate array.
    ///
    /// Args:
    ///     algebra: The algebra for the vectors.
    ///     coords: NumPy array of shape (N, dim) where dim is the algebra dimension.
    ///
    /// Returns:
    ///     A MultivectorBatch containing N vectors (grade-1 multivectors).
    ///
    /// Example:
    /// ```python
    /// R3 = Algebra.euclidean(3)
    /// points = np.random.randn(1000, 3)  # 1000 points in 3D
    /// batch = MultivectorBatch.from_vectors(R3, points)
    /// ```
    #[staticmethod]
    pub fn from_vectors(algebra: &PyAlgebra, coords: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let arr = coords.as_array();
        let dim = algebra.inner.dimension();
        let num_blades = algebra.inner.num_blades();
        let n = arr.nrows();

        if arr.ncols() != dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coords must have {} columns for {} but got {}",
                dim, algebra.inner.signature, arr.ncols()
            )));
        }

        // Build coefficient array: vectors have coefficients at indices 1, 2, 4, 8, ...
        let mut coeffs = Array2::<f64>::zeros((n, num_blades));
        for row in 0..n {
            for i in 0..dim {
                let blade_idx = 1 << i; // e1=1, e2=2, e3=4, etc.
                coeffs[[row, blade_idx]] = arr[[row, i]];
            }
        }

        Ok(Self {
            coeffs,
            algebra: algebra.inner.clone(),
        })
    }

    /// Create a batch of scalars from a 1D array.
    ///
    /// Args:
    ///     algebra: The algebra for the scalars.
    ///     values: NumPy array of shape (N,) containing scalar values.
    ///
    /// Returns:
    ///     A MultivectorBatch containing N scalars (grade-0 multivectors).
    ///
    /// Example:
    /// ```python
    /// R3 = Algebra.euclidean(3)
    /// scalars = np.array([1.0, 2.0, 3.0, 4.0])
    /// batch = MultivectorBatch.from_scalars(R3, scalars)
    /// ```
    #[staticmethod]
    pub fn from_scalars(algebra: &PyAlgebra, values: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let arr = values.as_array();
        let n = arr.len();
        let num_blades = algebra.inner.num_blades();

        // Scalars go in coefficient 0
        let mut coeffs = Array2::<f64>::zeros((n, num_blades));
        for (i, &val) in arr.iter().enumerate() {
            coeffs[[i, 0]] = val;
        }

        Ok(Self {
            coeffs,
            algebra: algebra.inner.clone(),
        })
    }

    /// Create a batch of bivectors from coefficient array.
    ///
    /// Args:
    ///     algebra: The algebra for the bivectors.
    ///     bivector_coeffs: NumPy array of shape (N, num_bivectors) where
    ///         num_bivectors = dim*(dim-1)/2.
    ///
    /// Returns:
    ///     A MultivectorBatch containing N bivectors (grade-2 multivectors).
    ///
    /// Example:
    /// ```python
    /// R3 = Algebra.euclidean(3)
    /// # 3 bivector components in 3D: e12, e13, e23
    /// biv_coeffs = np.random.randn(100, 3)
    /// batch = MultivectorBatch.from_bivectors(R3, biv_coeffs)
    /// ```
    #[staticmethod]
    pub fn from_bivectors(algebra: &PyAlgebra, bivector_coeffs: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let arr = bivector_coeffs.as_array();
        let dim = algebra.inner.dimension();
        let num_blades = algebra.inner.num_blades();
        let n = arr.nrows();

        // Count expected bivector components
        let num_bivectors = dim * (dim - 1) / 2;
        if arr.ncols() != num_bivectors {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bivector_coeffs must have {} columns for {} but got {}",
                num_bivectors, algebra.inner.signature, arr.ncols()
            )));
        }

        // Get bivector blade indices
        let bivector_indices: Vec<usize> = algebra.inner.blades_of_grade(2);

        // Build coefficient array
        let mut coeffs = Array2::<f64>::zeros((n, num_blades));
        for row in 0..n {
            for (col, &blade_idx) in bivector_indices.iter().enumerate() {
                coeffs[[row, blade_idx]] = arr[[row, col]];
            }
        }

        Ok(Self {
            coeffs,
            algebra: algebra.inner.clone(),
        })
    }

    /// Convert the batch to a NumPy array.
    ///
    /// Returns:
    ///     NumPy array of shape (N, num_blades) containing all coefficients.
    ///
    /// Example:
    /// ```python
    /// batch = MultivectorBatch.from_vectors(R3, coords)
    /// arr = batch.to_numpy()  # Shape: (N, 8) for 3D
    /// ```
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.coeffs.clone().into_pyarray(py)
    }

    /// Extract vector coordinates from the batch.
    ///
    /// Returns:
    ///     NumPy array of shape (N, dim) containing vector components.
    ///
    /// Example:
    /// ```python
    /// batch = MultivectorBatch.from_vectors(R3, coords)
    /// rotated = batch.sandwich(rotor)
    /// rotated_coords = rotated.to_vector_coords()  # Shape: (N, 3)
    /// ```
    pub fn to_vector_coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.coeffs.nrows();
        let dim = self.algebra.dimension();
        let mut result = Array2::<f64>::zeros((n, dim));

        for row in 0..n {
            for i in 0..dim {
                let blade_idx = 1 << i;
                result[[row, i]] = self.coeffs[[row, blade_idx]];
            }
        }

        result.into_pyarray(py)
    }

    /// Number of multivectors in the batch.
    pub fn len(&self) -> usize {
        self.coeffs.nrows()
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.coeffs.nrows() == 0
    }

    /// Number of blades per multivector (2^dimension).
    pub fn num_blades(&self) -> usize {
        self.algebra.num_blades()
    }

    /// Get a single multivector by index.
    ///
    /// Args:
    ///     index: Index of the multivector to retrieve (0-indexed).
    ///
    /// Returns:
    ///     The Multivector at the given index.
    ///
    /// Raises:
    ///     IndexError: If index is out of bounds.
    pub fn get(&self, index: usize) -> PyResult<Multivector> {
        if index >= self.coeffs.nrows() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of bounds for batch of size {}",
                index,
                self.coeffs.nrows()
            )));
        }

        let coeffs: Vec<f64> = self.coeffs.row(index).to_vec();
        Ok(Multivector {
            coeffs,
            dims: self.algebra.dimension(),
            algebra_opt: Some(self.algebra.clone()),
        })
    }

    /// Set a single multivector by index.
    ///
    /// Args:
    ///     index: Index of the multivector to set (0-indexed).
    ///     mv: The Multivector to store at the given index.
    ///
    /// Raises:
    ///     IndexError: If index is out of bounds.
    ///     ValueError: If multivector has wrong dimension.
    pub fn set(&mut self, index: usize, mv: &Multivector) -> PyResult<()> {
        if index >= self.coeffs.nrows() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of bounds for batch of size {}",
                index,
                self.coeffs.nrows()
            )));
        }

        if mv.coeffs.len() != self.algebra.num_blades() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "multivector has {} blades but batch expects {}",
                mv.coeffs.len(),
                self.algebra.num_blades()
            )));
        }

        for (i, &c) in mv.coeffs.iter().enumerate() {
            self.coeffs[[index, i]] = c;
        }
        Ok(())
    }

    /// Batch geometric product: self * other.
    ///
    /// If other is a single Multivector, it broadcasts across all elements.
    /// If other is a MultivectorBatch, element-wise product is computed.
    ///
    /// Args:
    ///     other: Either a Multivector or MultivectorBatch.
    ///
    /// Returns:
    ///     MultivectorBatch containing the products.
    ///
    /// Example:
    /// ```python
    /// batch = MultivectorBatch.from_vectors(R3, coords)
    /// product = batch.geometric_product(scalar_mv)  # Broadcasts
    /// ```
    pub fn geometric_product(&self, other: &PyMultivectorBatch) -> PyResult<Self> {
        if self.algebra.signature != other.algebra.signature {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "algebra mismatch: left is {} but right is {}",
                self.algebra.signature, other.algebra.signature
            )));
        }

        let n_self = self.coeffs.nrows();
        let n_other = other.coeffs.nrows();

        // Handle broadcasting
        let n = if n_self == n_other {
            n_self
        } else if n_other == 1 {
            n_self
        } else if n_self == 1 {
            n_other
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch sizes must match or be 1 for broadcasting: {} vs {}",
                n_self, n_other
            )));
        };

        let num_blades = self.algebra.num_blades();
        let mut result = Array2::<f64>::zeros((n, num_blades));

        let signs = self.algebra.cayley_signs();
        let blades = self.algebra.cayley_blades();

        for row in 0..n {
            let self_row = if n_self == 1 { 0 } else { row };
            let other_row = if n_other == 1 { 0 } else { row };

            for i in 0..num_blades {
                let a = self.coeffs[[self_row, i]];
                if a == 0.0 {
                    continue;
                }
                for j in 0..num_blades {
                    let b = other.coeffs[[other_row, j]];
                    if b == 0.0 {
                        continue;
                    }
                    let idx = i * num_blades + j;
                    let sign = signs[idx];
                    let blade = blades[idx];
                    result[[row, blade]] += sign * a * b;
                }
            }
        }

        Ok(Self {
            coeffs: result,
            algebra: self.algebra.clone(),
        })
    }

    /// Alias for geometric_product.
    pub fn gp(&self, other: &PyMultivectorBatch) -> PyResult<Self> {
        self.geometric_product(other)
    }

    /// Apply a rotor to all multivectors: R * self * ~R (sandwich product).
    ///
    /// This is the standard way to apply rotations/transformations in GA.
    /// The rotor is applied to each multivector in the batch.
    ///
    /// Args:
    ///     rotor: A Multivector representing a rotor (should be normalized).
    ///
    /// Returns:
    ///     MultivectorBatch with transformed multivectors.
    ///
    /// Example:
    /// ```python
    /// # Rotate 1000 points by 45 degrees around Z axis
    /// rotor = Multivector.from_axis_angle([0, 0, 1], np.pi/4)
    /// rotated = batch.sandwich(rotor)
    /// ```
    pub fn sandwich(&self, rotor: &Multivector) -> PyResult<Self> {
        // Check algebra compatibility
        let rotor_alg = rotor.get_algebra();
        if self.algebra.signature != rotor_alg.signature {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "algebra mismatch: batch is {} but rotor is {}",
                self.algebra.signature, rotor_alg.signature
            )));
        }

        let n = self.coeffs.nrows();
        let num_blades = self.algebra.num_blades();
        let mut result = Array2::<f64>::zeros((n, num_blades));

        let signs = self.algebra.cayley_signs();
        let blades = self.algebra.cayley_blades();

        // Compute rotor reverse
        let rotor_rev: Vec<f64> = rotor
            .coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * algebra::reverse_sign(algebra::blade_grade(i)))
            .collect();

        // For each element in batch: R * x * ~R
        for row in 0..n {
            // First compute R * x
            let mut rx = vec![0.0f64; num_blades];
            for i in 0..num_blades {
                let a = rotor.coeffs[i];
                if a == 0.0 {
                    continue;
                }
                for j in 0..num_blades {
                    let b = self.coeffs[[row, j]];
                    if b == 0.0 {
                        continue;
                    }
                    let idx = i * num_blades + j;
                    let sign = signs[idx];
                    let blade = blades[idx];
                    rx[blade] += sign * a * b;
                }
            }

            // Then compute (R * x) * ~R
            for i in 0..num_blades {
                let a = rx[i];
                if a == 0.0 {
                    continue;
                }
                for j in 0..num_blades {
                    let b = rotor_rev[j];
                    if b == 0.0 {
                        continue;
                    }
                    let idx = i * num_blades + j;
                    let sign = signs[idx];
                    let blade = blades[idx];
                    result[[row, blade]] += sign * a * b;
                }
            }
        }

        Ok(Self {
            coeffs: result,
            algebra: self.algebra.clone(),
        })
    }

    /// Alias for sandwich (common in robotics literature).
    pub fn apply(&self, rotor: &Multivector) -> PyResult<Self> {
        self.sandwich(rotor)
    }

    /// Batch outer (wedge) product.
    ///
    /// Computes the outer product element-wise for matching batches,
    /// or broadcasts if one batch has size 1.
    pub fn outer_product(&self, other: &PyMultivectorBatch) -> PyResult<Self> {
        if self.algebra.signature != other.algebra.signature {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "algebra mismatch: left is {} but right is {}",
                self.algebra.signature, other.algebra.signature
            )));
        }

        let n_self = self.coeffs.nrows();
        let n_other = other.coeffs.nrows();

        let n = if n_self == n_other {
            n_self
        } else if n_other == 1 {
            n_self
        } else if n_self == 1 {
            n_other
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch sizes must match or be 1 for broadcasting: {} vs {}",
                n_self, n_other
            )));
        };

        let num_blades = self.algebra.num_blades();
        let mut result = Array2::<f64>::zeros((n, num_blades));

        let signs = self.algebra.cayley_signs();
        let blades = self.algebra.cayley_blades();

        for row in 0..n {
            let self_row = if n_self == 1 { 0 } else { row };
            let other_row = if n_other == 1 { 0 } else { row };

            for i in 0..num_blades {
                let a = self.coeffs[[self_row, i]];
                if a == 0.0 {
                    continue;
                }
                for j in 0..num_blades {
                    // Outer product is zero when blades share a basis vector
                    if i & j != 0 {
                        continue;
                    }
                    let b = other.coeffs[[other_row, j]];
                    if b == 0.0 {
                        continue;
                    }
                    let idx = i * num_blades + j;
                    let sign = signs[idx];
                    let blade = blades[idx];
                    result[[row, blade]] += sign * a * b;
                }
            }
        }

        Ok(Self {
            coeffs: result,
            algebra: self.algebra.clone(),
        })
    }

    /// Alias for outer_product.
    pub fn wedge(&self, other: &PyMultivectorBatch) -> PyResult<Self> {
        self.outer_product(other)
    }

    /// Batch left contraction (inner product).
    ///
    /// Computes the left contraction element-wise for matching batches,
    /// or broadcasts if one batch has size 1.
    pub fn left_contraction(&self, other: &PyMultivectorBatch) -> PyResult<Self> {
        if self.algebra.signature != other.algebra.signature {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "algebra mismatch: left is {} but right is {}",
                self.algebra.signature, other.algebra.signature
            )));
        }

        let n_self = self.coeffs.nrows();
        let n_other = other.coeffs.nrows();

        let n = if n_self == n_other {
            n_self
        } else if n_other == 1 {
            n_self
        } else if n_self == 1 {
            n_other
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch sizes must match or be 1 for broadcasting: {} vs {}",
                n_self, n_other
            )));
        };

        let num_blades = self.algebra.num_blades();
        let mut result = Array2::<f64>::zeros((n, num_blades));

        let signs = self.algebra.cayley_signs();
        let blades = self.algebra.cayley_blades();

        for row in 0..n {
            let self_row = if n_self == 1 { 0 } else { row };
            let other_row = if n_other == 1 { 0 } else { row };

            for i in 0..num_blades {
                let a = self.coeffs[[self_row, i]];
                if a == 0.0 {
                    continue;
                }
                let grade_a = algebra::blade_grade(i);
                for j in 0..num_blades {
                    let b = other.coeffs[[other_row, j]];
                    if b == 0.0 {
                        continue;
                    }
                    let grade_b = algebra::blade_grade(j);
                    // Left contraction is zero when grade(A) > grade(B)
                    if grade_a > grade_b {
                        continue;
                    }
                    // Left contraction requires A subset of B
                    if (i & j) != i {
                        continue;
                    }
                    let idx = i * num_blades + j;
                    let sign = signs[idx];
                    let blade = blades[idx];
                    let result_grade = algebra::blade_grade(blade);
                    // Only keep the grade (s-r) component
                    if result_grade == grade_b - grade_a {
                        result[[row, blade]] += sign * a * b;
                    }
                }
            }
        }

        Ok(Self {
            coeffs: result,
            algebra: self.algebra.clone(),
        })
    }

    /// Alias for left_contraction.
    pub fn inner(&self, other: &PyMultivectorBatch) -> PyResult<Self> {
        self.left_contraction(other)
    }

    /// Alias for left_contraction.
    pub fn lc(&self, other: &PyMultivectorBatch) -> PyResult<Self> {
        self.left_contraction(other)
    }

    /// Compute the norm of each multivector in the batch.
    ///
    /// Returns:
    ///     NumPy array of shape (N,) containing the norms.
    ///
    /// Example:
    /// ```python
    /// batch = MultivectorBatch.from_vectors(R3, coords)
    /// norms = batch.norm()  # Shape: (N,)
    /// ```
    pub fn norm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let n = self.coeffs.nrows();
        let num_blades = self.algebra.num_blades();
        let signs = self.algebra.cayley_signs();
        let blades = self.algebra.cayley_blades();

        let mut result = Array1::<f64>::zeros(n);

        for row in 0..n {
            // Compute reverse
            let mut rev = vec![0.0f64; num_blades];
            for i in 0..num_blades {
                rev[i] = self.coeffs[[row, i]] * algebra::reverse_sign(algebra::blade_grade(i));
            }

            // Compute self * reverse, get scalar part
            let mut norm_sq = 0.0;
            for i in 0..num_blades {
                let a = self.coeffs[[row, i]];
                if a == 0.0 {
                    continue;
                }
                for j in 0..num_blades {
                    let b = rev[j];
                    if b == 0.0 {
                        continue;
                    }
                    let idx = i * num_blades + j;
                    let blade = blades[idx];
                    if blade == 0 {
                        // Scalar component
                        norm_sq += signs[idx] * a * b;
                    }
                }
            }

            result[row] = norm_sq.abs().sqrt();
        }

        result.into_pyarray(py)
    }

    /// Compute the squared norm of each multivector in the batch.
    ///
    /// Returns:
    ///     NumPy array of shape (N,) containing the squared norms.
    pub fn norm_squared<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let n = self.coeffs.nrows();
        let num_blades = self.algebra.num_blades();
        let signs = self.algebra.cayley_signs();
        let blades = self.algebra.cayley_blades();

        let mut result = Array1::<f64>::zeros(n);

        for row in 0..n {
            // Compute reverse
            let mut rev = vec![0.0f64; num_blades];
            for i in 0..num_blades {
                rev[i] = self.coeffs[[row, i]] * algebra::reverse_sign(algebra::blade_grade(i));
            }

            // Compute self * reverse, get scalar part
            let mut norm_sq = 0.0;
            for i in 0..num_blades {
                let a = self.coeffs[[row, i]];
                if a == 0.0 {
                    continue;
                }
                for j in 0..num_blades {
                    let b = rev[j];
                    if b == 0.0 {
                        continue;
                    }
                    let idx = i * num_blades + j;
                    let blade = blades[idx];
                    if blade == 0 {
                        norm_sq += signs[idx] * a * b;
                    }
                }
            }

            result[row] = norm_sq;
        }

        result.into_pyarray(py)
    }

    /// Normalize each multivector in the batch.
    ///
    /// Multivectors with zero norm are left unchanged.
    ///
    /// Returns:
    ///     MultivectorBatch with normalized multivectors.
    pub fn normalized(&self) -> Self {
        let n = self.coeffs.nrows();
        let num_blades = self.algebra.num_blades();
        let signs = self.algebra.cayley_signs();
        let blades = self.algebra.cayley_blades();

        let mut result = self.coeffs.clone();

        for row in 0..n {
            // Compute reverse
            let mut rev = vec![0.0f64; num_blades];
            for i in 0..num_blades {
                rev[i] = self.coeffs[[row, i]] * algebra::reverse_sign(algebra::blade_grade(i));
            }

            // Compute norm
            let mut norm_sq = 0.0;
            for i in 0..num_blades {
                let a = self.coeffs[[row, i]];
                if a == 0.0 {
                    continue;
                }
                for j in 0..num_blades {
                    let b = rev[j];
                    if b == 0.0 {
                        continue;
                    }
                    let idx = i * num_blades + j;
                    let blade = blades[idx];
                    if blade == 0 {
                        norm_sq += signs[idx] * a * b;
                    }
                }
            }

            let norm = norm_sq.abs().sqrt();
            if norm > 1e-14 {
                for i in 0..num_blades {
                    result[[row, i]] /= norm;
                }
            }
        }

        Self {
            coeffs: result,
            algebra: self.algebra.clone(),
        }
    }

    /// Compute the reverse of each multivector in the batch.
    ///
    /// Returns:
    ///     MultivectorBatch with reversed multivectors.
    pub fn reverse(&self) -> Self {
        let n = self.coeffs.nrows();
        let num_blades = self.algebra.num_blades();
        let mut result = Array2::<f64>::zeros((n, num_blades));

        for row in 0..n {
            for i in 0..num_blades {
                result[[row, i]] =
                    self.coeffs[[row, i]] * algebra::reverse_sign(algebra::blade_grade(i));
            }
        }

        Self {
            coeffs: result,
            algebra: self.algebra.clone(),
        }
    }

    /// Compute the grade involution of each multivector in the batch.
    ///
    /// Grade involution negates all odd-grade components.
    pub fn grade_involution(&self) -> Self {
        let n = self.coeffs.nrows();
        let num_blades = self.algebra.num_blades();
        let mut result = Array2::<f64>::zeros((n, num_blades));

        for row in 0..n {
            for i in 0..num_blades {
                result[[row, i]] =
                    self.coeffs[[row, i]] * algebra::grade_involution_sign(algebra::blade_grade(i));
            }
        }

        Self {
            coeffs: result,
            algebra: self.algebra.clone(),
        }
    }

    /// Add two batches element-wise.
    pub fn __add__(&self, other: &PyMultivectorBatch) -> PyResult<Self> {
        if self.algebra.signature != other.algebra.signature {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "algebra mismatch: left is {} but right is {}",
                self.algebra.signature, other.algebra.signature
            )));
        }

        let n_self = self.coeffs.nrows();
        let n_other = other.coeffs.nrows();

        if n_self == n_other {
            Ok(Self {
                coeffs: &self.coeffs + &other.coeffs,
                algebra: self.algebra.clone(),
            })
        } else if n_other == 1 {
            let mut result = self.coeffs.clone();
            for row in 0..n_self {
                for col in 0..self.algebra.num_blades() {
                    result[[row, col]] += other.coeffs[[0, col]];
                }
            }
            Ok(Self {
                coeffs: result,
                algebra: self.algebra.clone(),
            })
        } else if n_self == 1 {
            let mut result = other.coeffs.clone();
            for row in 0..n_other {
                for col in 0..self.algebra.num_blades() {
                    result[[row, col]] += self.coeffs[[0, col]];
                }
            }
            Ok(Self {
                coeffs: result,
                algebra: self.algebra.clone(),
            })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch sizes must match or be 1 for broadcasting: {} vs {}",
                n_self, n_other
            )))
        }
    }

    /// Subtract two batches element-wise.
    pub fn __sub__(&self, other: &PyMultivectorBatch) -> PyResult<Self> {
        if self.algebra.signature != other.algebra.signature {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "algebra mismatch: left is {} but right is {}",
                self.algebra.signature, other.algebra.signature
            )));
        }

        let n_self = self.coeffs.nrows();
        let n_other = other.coeffs.nrows();

        if n_self == n_other {
            Ok(Self {
                coeffs: &self.coeffs - &other.coeffs,
                algebra: self.algebra.clone(),
            })
        } else if n_other == 1 {
            let mut result = self.coeffs.clone();
            for row in 0..n_self {
                for col in 0..self.algebra.num_blades() {
                    result[[row, col]] -= other.coeffs[[0, col]];
                }
            }
            Ok(Self {
                coeffs: result,
                algebra: self.algebra.clone(),
            })
        } else if n_self == 1 {
            let mut result = Array2::<f64>::zeros((n_other, self.algebra.num_blades()));
            for row in 0..n_other {
                for col in 0..self.algebra.num_blades() {
                    result[[row, col]] = self.coeffs[[0, col]] - other.coeffs[[row, col]];
                }
            }
            Ok(Self {
                coeffs: result,
                algebra: self.algebra.clone(),
            })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch sizes must match or be 1 for broadcasting: {} vs {}",
                n_self, n_other
            )))
        }
    }

    /// Scale all multivectors by a scalar.
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            coeffs: &self.coeffs * factor,
            algebra: self.algebra.clone(),
        }
    }

    /// Negate all multivectors.
    pub fn __neg__(&self) -> Self {
        self.scale(-1.0)
    }

    /// Scalar multiplication.
    pub fn __mul__(&self, scalar: f64) -> Self {
        self.scale(scalar)
    }

    /// Right scalar multiplication.
    pub fn __rmul__(&self, scalar: f64) -> Self {
        self.scale(scalar)
    }

    /// Scalar division.
    pub fn __truediv__(&self, scalar: f64) -> PyResult<Self> {
        if scalar == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "division by zero",
            ));
        }
        Ok(self.scale(1.0 / scalar))
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "MultivectorBatch(size={}, algebra={})",
            self.coeffs.nrows(),
            self.algebra.signature
        )
    }

    /// Length (batch size).
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Get item by index.
    pub fn __getitem__(&self, index: isize) -> PyResult<Multivector> {
        let n = self.coeffs.nrows() as isize;
        let idx = if index < 0 { n + index } else { index };
        if idx < 0 || idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of bounds for batch of size {}",
                index, n
            )));
        }
        self.get(idx as usize)
    }

    /// Extract scalar (grade-0) parts as array.
    pub fn scalar<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let n = self.coeffs.nrows();
        let mut result = Array1::<f64>::zeros(n);
        for row in 0..n {
            result[row] = self.coeffs[[row, 0]];
        }
        result.into_pyarray(py)
    }

    /// Extract grade-k parts as a new batch.
    pub fn grade(&self, k: usize) -> PyResult<Self> {
        let dim = self.algebra.dimension();
        if k > dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "grade {} exceeds dimension {} of algebra",
                k, dim
            )));
        }

        let n = self.coeffs.nrows();
        let num_blades = self.algebra.num_blades();
        let mut result = Array2::<f64>::zeros((n, num_blades));

        for row in 0..n {
            for i in 0..num_blades {
                if algebra::blade_grade(i) == k {
                    result[[row, i]] = self.coeffs[[row, i]];
                }
            }
        }

        Ok(Self {
            coeffs: result,
            algebra: self.algebra.clone(),
        })
    }

    /// Concatenate two batches.
    pub fn concat(&self, other: &PyMultivectorBatch) -> PyResult<Self> {
        if self.algebra.signature != other.algebra.signature {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "algebra mismatch: left is {} but right is {}",
                self.algebra.signature, other.algebra.signature
            )));
        }

        let combined = ndarray::concatenate(
            Axis(0),
            &[self.coeffs.view(), other.coeffs.view()],
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("concat failed: {}", e)))?;

        Ok(Self {
            coeffs: combined,
            algebra: self.algebra.clone(),
        })
    }

    /// Slice the batch.
    ///
    /// Args:
    ///     start: Start index (inclusive).
    ///     end: End index (exclusive).
    ///
    /// Returns:
    ///     A new MultivectorBatch containing elements [start:end].
    pub fn slice(&self, start: usize, end: usize) -> PyResult<Self> {
        let n = self.coeffs.nrows();
        if start > n || end > n || start > end {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "invalid slice [{}:{}] for batch of size {}",
                start, end, n
            )));
        }

        let sliced = self.coeffs.slice(ndarray::s![start..end, ..]).to_owned();
        Ok(Self {
            coeffs: sliced,
            algebra: self.algebra.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_creation() {
        let alg = Algebra::euclidean(3);
        let coeffs = Array2::<f64>::zeros((10, 8));
        let batch = PyMultivectorBatch::new(coeffs, alg);
        assert_eq!(batch.len(), 10);
        assert_eq!(batch.num_blades(), 8);
    }

    #[test]
    fn test_batch_vectors() {
        let alg = Algebra::euclidean(3);

        // Create some test coordinates
        let coords = Array2::from_shape_vec((3, 3), vec![
            1.0, 0.0, 0.0,  // e1
            0.0, 1.0, 0.0,  // e2
            0.0, 0.0, 1.0,  // e3
        ]).unwrap();

        // Build batch manually (from_vectors requires Python)
        let mut coeffs = Array2::<f64>::zeros((3, 8));
        coeffs[[0, 1]] = 1.0; // e1
        coeffs[[1, 2]] = 1.0; // e2
        coeffs[[2, 4]] = 1.0; // e3

        let batch = PyMultivectorBatch::new(coeffs, alg.clone());

        // Verify
        let mv0 = batch.get(0).unwrap();
        assert_eq!(mv0.coeffs[1], 1.0); // e1 coefficient
        assert_eq!(mv0.coeffs[2], 0.0); // e2 coefficient
    }

    #[test]
    fn test_batch_reverse() {
        let alg = Algebra::euclidean(2);

        // Create a batch with e1 + e12
        let mut coeffs = Array2::<f64>::zeros((1, 4));
        coeffs[[0, 1]] = 1.0; // e1
        coeffs[[0, 3]] = 1.0; // e12

        let batch = PyMultivectorBatch::new(coeffs, alg);
        let reversed = batch.reverse();

        // e1 stays e1, e12 becomes -e12
        assert_eq!(reversed.coeffs[[0, 1]], 1.0);
        assert_eq!(reversed.coeffs[[0, 3]], -1.0);
    }
}
