use crate::algebra;
use pyo3::prelude::*;

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
#[pyclass(eq, hash, frozen)]
#[derive(Debug, Clone)]
pub struct Multivector {
    /// Coefficients for each basis blade.
    /// Length is always 2^n where n is the dimension of the base vector space.
    pub coeffs: Vec<f64>,

    /// Dimension of the base vector space.
    pub dims: usize,
}

impl PartialEq for Multivector {
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims && self.coeffs == other.coeffs
    }
}

impl Eq for Multivector {}

impl std::hash::Hash for Multivector {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dims.hash(state);
        for &c in &self.coeffs {
            c.to_bits().hash(state);
        }
    }
}

/// Iterator over multivector coefficients.
#[pyclass]
pub struct CoeffIterator {
    coeffs: Vec<f64>,
    index: usize,
}

#[pymethods]
impl CoeffIterator {
    fn __iter__(slf: pyo3::PyRef<'_, Self>) -> pyo3::PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: pyo3::PyRefMut<'_, Self>) -> Option<f64> {
        if slf.index < slf.coeffs.len() {
            let val = slf.coeffs[slf.index];
            slf.index += 1;
            Some(val)
        } else {
            None
        }
    }
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
                "coeffs must not be empty; provide a list of coefficients \
                (e.g., [1.0] for a scalar, [0.0, 1.0, 0.0, 0.0] for e1 in 2D)",
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
        // len is guaranteed to be a power of 2, so trailing_zeros gives log2
        let dims = len.trailing_zeros() as usize;
        Ok(Multivector { coeffs, dims })
    }

    /// Return the scalar (grade-0) part of this multivector.
    pub fn scalar(&self) -> f64 {
        self.coeffs[0]
    }

    /// Return the vector (grade-1) part of this multivector.
    ///
    /// Example:
    /// ```python
    /// mv = some_multivector
    /// v = mv.vector_part()  # grade-1 projection
    /// ```
    pub fn vector_part(&self) -> Self {
        self.grade(1).unwrap_or_else(|_| self.clone())
    }

    /// Return the bivector (grade-2) part of this multivector.
    ///
    /// Example:
    /// ```python
    /// mv = some_multivector
    /// B = mv.bivector_part()  # grade-2 projection
    /// ```
    pub fn bivector_part(&self) -> Self {
        self.grade(2).unwrap_or_else(|_| self.clone())
    }

    /// Return the trivector (grade-3) part of this multivector.
    ///
    /// Example:
    /// ```python
    /// mv = some_multivector
    /// T = mv.trivector_part()  # grade-3 projection
    /// ```
    pub fn trivector_part(&self) -> Self {
        self.grade(3).unwrap_or_else(|_| self.clone())
    }

    /// Extract vector (grade-1) coefficients as a list.
    ///
    /// Returns a list of length `dims` containing the vector components
    /// [x, y, z, ...] in order. Useful for interoperating with numpy or
    /// other libraries that expect coordinate arrays.
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([3.0, 4.0, 5.0])
    /// coords = v.to_vector_coords()  # [3.0, 4.0, 5.0]
    /// import numpy as np
    /// arr = np.array(coords)  # Easy numpy conversion
    /// ```
    pub fn to_vector_coords(&self) -> Vec<f64> {
        // Vector coefficients are at indices 1, 2, 4, 8, ... (powers of 2)
        (0..self.dims).map(|i| self.coeffs[1 << i]).collect()
    }

    /// Extract bivector (grade-2) coefficients as a list.
    ///
    /// Returns a list of bivector components in canonical order.
    /// For 3D: [e12, e13, e23] (indices 3, 5, 6)
    /// For 4D: [e12, e13, e14, e23, e24, e34] (6 components)
    ///
    /// The number of components is dims*(dims-1)/2.
    ///
    /// Useful for extracting rotation planes or angular velocities.
    ///
    /// Example:
    /// ```python
    /// B = Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    /// comps = B.to_bivector_coords()  # [1.0, 0.0, 0.0]
    /// ```
    pub fn to_bivector_coords(&self) -> Vec<f64> {
        // Bivector indices have exactly 2 bits set
        self.coeffs
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| if i.count_ones() == 2 { Some(c) } else { None })
            .collect()
    }

    /// Extract trivector (grade-3) coefficients as a list.
    ///
    /// Returns a list of trivector components in canonical order.
    /// The number of components is dims*(dims-1)*(dims-2)/6.
    ///
    /// Example:
    /// ```python
    /// T = some_multivector.trivector_part()
    /// comps = T.to_trivector_coords()
    /// ```
    pub fn to_trivector_coords(&self) -> Vec<f64> {
        // Trivector indices have exactly 3 bits set
        self.coeffs
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| if i.count_ones() == 3 { Some(c) } else { None })
            .collect()
    }

    /// Decompose this multivector into individual blade components.
    ///
    /// Returns a list of (index, coefficient, grade) tuples for each
    /// non-zero blade.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, 2.0, 3.0, 4.0])
    /// blades = mv.blades()
    /// # [(0, 1.0, 0), (1, 2.0, 1), (2, 3.0, 1), (3, 4.0, 2)]
    /// ```
    pub fn blades(&self) -> Vec<(usize, f64, usize)> {
        self.coeffs
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| {
                if c != 0.0 {
                    Some((i, c, i.count_ones() as usize))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return a dictionary mapping grade -> multivector for each non-zero grade part.
    ///
    /// Example:
    /// ```python
    /// mv = e1 + e2 + e12
    /// parts = mv.grade_parts()
    /// # {1: <vector part>, 2: <bivector part>}
    /// ```
    pub fn grade_parts(&self) -> std::collections::HashMap<usize, Multivector> {
        let mut parts = std::collections::HashMap::new();
        for g in self.grades() {
            if let Ok(part) = self.grade(g) {
                parts.insert(g, part);
            }
        }
        parts
    }

    // =========================================================================
    // CONSTRUCTORS (static methods)
    // =========================================================================

    /// Create a zero multivector in the given dimension.
    ///
    /// Example:
    /// ```python
    /// zero = Multivector.zero(3)  # Zero in Cl(3)
    /// ```
    #[staticmethod]
    pub fn zero(dims: usize) -> PyResult<Self> {
        if dims == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension must be at least 1",
            ));
        }
        let size = 1usize << dims;
        Ok(Multivector {
            coeffs: vec![0.0; size],
            dims,
        })
    }

    /// Create a scalar (grade-0) multivector.
    ///
    /// Example:
    /// ```python
    /// five = Multivector.from_scalar(5.0, dims=3)  # 5.0 in Cl(3)
    /// ```
    #[staticmethod]
    pub fn from_scalar(value: f64, dims: usize) -> PyResult<Self> {
        if dims == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension must be at least 1",
            ));
        }
        let size = 1usize << dims;
        let mut coeffs = vec![0.0; size];
        coeffs[0] = value;
        Ok(Multivector { coeffs, dims })
    }

    /// Create a multivector from a list of coefficients.
    ///
    /// The length of the coefficient list must be a power of 2 (2^dims).
    /// The dimension is inferred from the length.
    ///
    /// Coefficients are ordered by blade index:
    /// - Index 0: scalar (1)
    /// - Index 1: e1, Index 2: e2, Index 4: e3, ...
    /// - Index 3: e12, Index 5: e13, Index 6: e23, ...
    /// - etc.
    ///
    /// Example:
    /// ```python
    /// # 2D: [scalar, e1, e2, e12]
    /// mv = Multivector.from_list([1.0, 2.0, 3.0, 4.0])
    /// # Creates: 1 + 2*e1 + 3*e2 + 4*e12
    /// ```
    ///
    /// # Errors
    /// Returns an error if the length is not a power of 2.
    #[staticmethod]
    pub fn from_list(coeffs: Vec<f64>) -> PyResult<Self> {
        let len = coeffs.len();
        if len == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "coefficient list must not be empty",
            ));
        }
        // Check if len is a power of 2
        if (len & (len - 1)) != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coefficient list length {} is not a power of 2; \
                expected 2^dims (e.g., 2, 4, 8, 16, ...)",
                len
            )));
        }
        // Compute dims: len = 2^dims, so dims = log2(len)
        let dims = len.trailing_zeros() as usize;
        Ok(Multivector { coeffs, dims })
    }

    /// Create a vector (grade-1) multivector from coordinates.
    ///
    /// The length of coords determines the dimension.
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([1.0, 2.0, 3.0])  # e1 + 2*e2 + 3*e3 in Cl(3)
    /// ```
    #[staticmethod]
    pub fn from_vector(coords: Vec<f64>) -> PyResult<Self> {
        let dims = coords.len();
        if dims == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "coords must not be empty; provide at least one coordinate",
            ));
        }
        let size = 1usize << dims;
        let mut coeffs = vec![0.0; size];
        // Basis vectors have indices 1, 2, 4, 8, ... (powers of 2)
        for (i, &c) in coords.iter().enumerate() {
            coeffs[1 << i] = c;
        }
        Ok(Multivector { coeffs, dims })
    }

    /// Create a single basis vector e_i (1-indexed).
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.basis(1, dims=3)  # e1 in Cl(3)
    /// e2 = Multivector.basis(2, dims=3)  # e2 in Cl(3)
    /// ```
    #[staticmethod]
    pub fn basis(index: usize, dims: usize) -> PyResult<Self> {
        if dims == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension must be at least 1",
            ));
        }
        if index == 0 || index > dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "basis index must be between 1 and {} (got {}); \
                use index=1 for e1, index=2 for e2, etc.",
                dims, index
            )));
        }
        let size = 1usize << dims;
        let mut coeffs = vec![0.0; size];
        // e_i has coefficient index 2^(i-1) (e1=1, e2=2, e3=4, ...)
        coeffs[1 << (index - 1)] = 1.0;
        Ok(Multivector { coeffs, dims })
    }

    /// Create the unit pseudoscalar (highest grade element) for the given dimension.
    ///
    /// The pseudoscalar is e1 ∧ e2 ∧ ... ∧ en, which has index 2^n - 1.
    ///
    /// Example:
    /// ```python
    /// I = Multivector.pseudoscalar(3)  # e123 in Cl(3)
    /// ```
    #[staticmethod]
    pub fn pseudoscalar(dims: usize) -> PyResult<Self> {
        if dims == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension must be at least 1",
            ));
        }
        let size = 1usize << dims;
        let mut coeffs = vec![0.0; size];
        // Pseudoscalar has all bits set: index = 2^n - 1
        coeffs[size - 1] = 1.0;
        Ok(Multivector { coeffs, dims })
    }

    /// Create the unit vector e1 (first basis vector).
    ///
    /// Shorthand for `Multivector.basis(1, dims)`.
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.e1(3)  # e1 in Cl(3)
    /// ```
    #[staticmethod]
    pub fn e1(dims: usize) -> PyResult<Self> {
        Self::basis(1, dims)
    }

    /// Create the unit vector e2 (second basis vector).
    ///
    /// Shorthand for `Multivector.basis(2, dims)`.
    /// Requires dims >= 2.
    ///
    /// Example:
    /// ```python
    /// e2 = Multivector.e2(3)  # e2 in Cl(3)
    /// ```
    #[staticmethod]
    pub fn e2(dims: usize) -> PyResult<Self> {
        if dims < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "e2 requires dimension >= 2",
            ));
        }
        Self::basis(2, dims)
    }

    /// Create the unit vector e3 (third basis vector).
    ///
    /// Shorthand for `Multivector.basis(3, dims)`.
    /// Requires dims >= 3.
    ///
    /// Example:
    /// ```python
    /// e3 = Multivector.e3(3)  # e3 in Cl(3)
    /// ```
    #[staticmethod]
    pub fn e3(dims: usize) -> PyResult<Self> {
        if dims < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "e3 requires dimension >= 3",
            ));
        }
        Self::basis(3, dims)
    }

    /// Create the unit basis vector e4 in Cl(n).
    ///
    /// Shorthand for `basis(4, dims)`.
    ///
    /// Raises ValueError if dimension < 4.
    ///
    /// Example:
    /// ```python
    /// e4 = Multivector.e4(4)  # e4 in Cl(4)
    /// ```
    #[staticmethod]
    pub fn e4(dims: usize) -> PyResult<Self> {
        if dims < 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "e4 requires dimension >= 4",
            ));
        }
        Self::basis(4, dims)
    }

    /// Create the unit basis bivector e12 = e1 ∧ e2 in Cl(n).
    ///
    /// This is the unit bivector spanning the plane of e1 and e2.
    ///
    /// Raises ValueError if dimension < 2.
    ///
    /// Example:
    /// ```python
    /// e12 = Multivector.e12(3)  # e12 bivector in Cl(3)
    /// # e12 squares to -1: e12 * e12 = -1
    /// ```
    #[staticmethod]
    pub fn e12(dims: usize) -> PyResult<Self> {
        if dims < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "e12 requires dimension >= 2",
            ));
        }
        // e12 = e1 ∧ e2, index 3 (binary 011)
        let size = 1usize << dims;
        let mut coeffs = vec![0.0; size];
        coeffs[3] = 1.0;
        Ok(Multivector { coeffs, dims })
    }

    /// Create the unit basis bivector e23 = e2 ∧ e3 in Cl(n).
    ///
    /// This is the unit bivector spanning the plane of e2 and e3.
    ///
    /// Raises ValueError if dimension < 3.
    ///
    /// Example:
    /// ```python
    /// e23 = Multivector.e23(3)  # e23 bivector in Cl(3)
    /// ```
    #[staticmethod]
    pub fn e23(dims: usize) -> PyResult<Self> {
        if dims < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "e23 requires dimension >= 3",
            ));
        }
        // e23 = e2 ∧ e3, index 6 (binary 110)
        let size = 1usize << dims;
        let mut coeffs = vec![0.0; size];
        coeffs[6] = 1.0;
        Ok(Multivector { coeffs, dims })
    }

    /// Create the unit basis bivector e31 = e3 ∧ e1 in Cl(n).
    ///
    /// This is the unit bivector spanning the plane of e3 and e1.
    /// Note: e31 = -e13 due to antisymmetry of the wedge product.
    ///
    /// Raises ValueError if dimension < 3.
    ///
    /// Example:
    /// ```python
    /// e31 = Multivector.e31(3)  # e31 bivector in Cl(3)
    /// ```
    #[staticmethod]
    pub fn e31(dims: usize) -> PyResult<Self> {
        if dims < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "e31 requires dimension >= 3",
            ));
        }
        // e31 = e3 ∧ e1 = -e1 ∧ e3 = -e13
        // e13 has index 5 (binary 101), so e31 = -e13
        let size = 1usize << dims;
        let mut coeffs = vec![0.0; size];
        coeffs[5] = -1.0;
        Ok(Multivector { coeffs, dims })
    }

    /// Create the unit pseudoscalar e123 = e1 ∧ e2 ∧ e3 in 3D.
    ///
    /// This is the unit trivector in Cl(3), equivalent to `pseudoscalar(3)`.
    ///
    /// The pseudoscalar is used for:
    /// - Computing duals
    /// - Cross products: a × b = (a ∧ b) * e123⁻¹
    /// - Volume calculations
    ///
    /// Raises ValueError if dimension != 3.
    ///
    /// Example:
    /// ```python
    /// I = Multivector.e123()  # Unit pseudoscalar in Cl(3)
    /// ```
    #[staticmethod]
    pub fn e123() -> PyResult<Self> {
        // e123 has index 7 (binary 111)
        let mut coeffs = vec![0.0; 8];
        coeffs[7] = 1.0;
        Ok(Multivector { coeffs, dims: 3 })
    }

    /// Create a random multivector with coefficients in [0, 1).
    ///
    /// Useful for testing and experimentation.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.random(3)  # random multivector in Cl(3)
    /// ```
    #[staticmethod]
    pub fn random(dims: usize) -> PyResult<Self> {
        if dims == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension must be at least 1",
            ));
        }
        use std::time::{SystemTime, UNIX_EPOCH};
        let size = 1usize << dims;
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        // Simple LCG for reproducible but varied random numbers
        let mut state = seed;
        let coeffs: Vec<f64> = (0..size)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state as f64) / (u64::MAX as f64)
            })
            .collect();
        Ok(Multivector { coeffs, dims })
    }

    /// Create a random unit vector.
    ///
    /// Returns a normalized vector with random direction.
    ///
    /// Example:
    /// ```python
    /// v = Multivector.random_vector(3)  # random unit vector in R^3
    /// assert abs(v.norm() - 1.0) < 1e-10
    /// ```
    #[staticmethod]
    pub fn random_vector(dims: usize) -> PyResult<Self> {
        if dims == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension must be at least 1",
            ));
        }
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        // Simple LCG
        let mut state = seed;
        let coords: Vec<f64> = (0..dims)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Map to [-1, 1] for better distribution
                2.0 * (state as f64) / (u64::MAX as f64) - 1.0
            })
            .collect();
        let v = Self::from_vector(coords)?;
        v.normalized()
    }

    /// Create a random rotor (rotation).
    ///
    /// Returns a normalized rotor representing a random rotation.
    /// Uses exponential map: generates a random bivector and exponentiates it.
    ///
    /// Example:
    /// ```python
    /// R = Multivector.random_rotor(3)  # random 3D rotation
    /// assert R.is_rotor()
    /// ```
    #[staticmethod]
    pub fn random_rotor(dims: usize) -> PyResult<Self> {
        if dims < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rotors require dimension >= 2",
            ));
        }
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Number of grade-2 components (bivector dimension)
        let n_bivector = dims * (dims - 1) / 2;
        let size = 1usize << dims;
        let mut coeffs = vec![0.0; size];

        // Simple LCG for random bivector components
        let mut state = seed;
        for (i, coeff) in coeffs.iter_mut().enumerate() {
            let grade = i.count_ones() as usize;
            if grade == 2 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Random angle component, scaled to [-pi, pi]
                let angle = std::f64::consts::PI * (2.0 * (state as f64) / (u64::MAX as f64) - 1.0);
                // Scale by 1/sqrt(n_bivector) for reasonable total angle
                *coeff = angle / (n_bivector as f64).sqrt();
            }
        }

        let bivector = Multivector { coeffs, dims };
        bivector.exp()
    }

    /// Create a rotor that rotates vector `a` to vector `b`.
    ///
    /// The rotor R satisfies: R * a * ~R ∝ b (parallel to b).
    /// If both vectors are unit vectors, R * a * ~R = b exactly.
    ///
    /// The rotation is through the plane containing a and b,
    /// by the angle between them.
    ///
    /// Both vectors must have the same dimension.
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// e2 = Multivector.from_vector([0.0, 1.0, 0.0])
    /// R = Multivector.rotor_from_vectors(e1, e2)  # 90° rotation in xy-plane
    /// rotated = R.sandwich(e1)  # should equal e2
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    #[staticmethod]
    pub fn rotor_from_vectors(
        a: PyRef<'_, Multivector>,
        b: PyRef<'_, Multivector>,
    ) -> PyResult<Self> {
        if a.dims != b.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: first vector is Cl({}) but second vector is Cl({}); \
                both must have the same dimension",
                a.dims, b.dims
            )));
        }

        // Normalize both vectors
        let a_norm = a.normalize().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "first vector has zero norm; cannot create rotor from zero vector",
            )
        })?;
        let b_norm = b.normalize().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "second vector has zero norm; cannot create rotor from zero vector",
            )
        })?;

        // Rotor R = (1 + b̂*â) / |1 + b̂*â|
        // This gives a rotor that rotates â to b̂
        let one = Multivector::from_scalar(1.0, a.dims)?;
        let ba = b_norm.geometric_product(&a_norm)?;
        let r_unnorm = one.__add__(&ba)?;

        // Handle anti-parallel case (a and b point in opposite directions)
        // In this case 1 + b̂â ≈ 0, need to find perpendicular vector
        let r_norm_sq = r_unnorm.norm_squared();
        if r_norm_sq.abs() < 1e-10 {
            // Vectors are anti-parallel, find any perpendicular vector
            // Use e_i where a has smallest component
            let mut min_idx = 0;
            let mut min_val = f64::MAX;
            for i in 0..a.dims {
                let coeff = a.coeffs[1 << i].abs();
                if coeff < min_val {
                    min_val = coeff;
                    min_idx = i;
                }
            }
            let perp = Multivector::basis(min_idx + 1, a.dims)?;
            // Rotor for 180° rotation: R = perp * â (any perpendicular vector works)
            let perp_norm = perp.normalize().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "failed to create perpendicular vector for anti-parallel case",
                )
            })?;
            perp_norm.geometric_product(&a_norm)
        } else {
            r_unnorm.normalized()
        }
    }

    /// Create a 3D rotor from an axis vector and rotation angle.
    ///
    /// The axis is a 3D vector that defines the rotation axis.
    /// The angle is in radians, with positive values rotating counter-clockwise
    /// when looking along the axis (right-hand rule).
    ///
    /// Formula: R = cos(θ/2) + sin(θ/2) * (axis normalized as bivector)
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    #[staticmethod]
    pub fn from_axis_angle(axis: PyRef<'_, Multivector>, angle: f64) -> PyResult<Self> {
        if axis.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "from_axis_angle requires 3D vectors, got Cl({})",
                axis.dims
            )));
        }

        // Check axis is a vector
        let axis_grade = axis.pure_grade();
        if axis_grade != Some(1) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "axis must be a vector (grade 1)",
            ));
        }

        // Normalize the axis
        let axis_unit = axis.normalize().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("axis has zero norm; cannot create rotor")
        })?;

        // Get axis components
        let ax = axis_unit.coeffs[1]; // e1 component
        let ay = axis_unit.coeffs[2]; // e2 component
        let az = axis_unit.coeffs[4]; // e3 component

        // The bivector representing the rotation plane is the negative dual of the axis.
        // This follows the right-hand rule: positive angle rotates counterclockwise
        // when looking along the axis direction.
        //
        // For unit axis (ax, ay, az), the bivector is:
        // B = -ax*e23 + ay*e13 - az*e12
        let half_angle = angle / 2.0;
        let cos_half = half_angle.cos();
        let sin_half = half_angle.sin();

        // R = cos(θ/2) + sin(θ/2) * B
        // Coefficients: [scalar, e1, e2, e12, e3, e13, e23, e123]
        let mut coeffs = vec![0.0; 8];
        coeffs[0] = cos_half; // scalar
        coeffs[3] = -sin_half * az; // e12
        coeffs[5] = sin_half * ay; // e13
        coeffs[6] = -sin_half * ax; // e23

        Ok(Multivector { coeffs, dims: 3 })
    }

    /// Create a 3D rotor from quaternion components (w, x, y, z).
    ///
    /// Quaternion convention: q = w + x*i + y*j + z*k
    ///
    /// The mapping to geometric algebra is:
    /// - w → scalar part
    /// - (x, y, z) → bivector part (mapped to -e23, -e13, -e12)
    ///
    /// This uses the common convention where quaternion multiplication
    /// matches rotor composition.
    #[staticmethod]
    pub fn from_quaternion(w: f64, x: f64, y: f64, z: f64) -> Self {
        // Quaternion: q = w + x*i + y*j + z*k
        // GA rotor mapping: i ↔ -e23, j ↔ -e13, k ↔ -e12
        // So: R = w - x*e23 - y*e13 - z*e12
        let mut coeffs = vec![0.0; 8];
        coeffs[0] = w; // scalar
        coeffs[3] = -z; // e12 (from -z*k)
        coeffs[5] = -y; // e13 (from -y*j)
        coeffs[6] = -x; // e23 (from -x*i)

        Multivector { coeffs, dims: 3 }
    }

    /// Convert this 3D rotor to quaternion components (w, x, y, z).
    ///
    /// Returns (w, x, y, z) where q = w + x*i + y*j + z*k.
    ///
    /// Raises ValueError if not a 3D multivector.
    ///
    /// Note: Works best for unit rotors. Non-unit multivectors
    /// will be converted but may not represent valid rotations.
    pub fn to_quaternion(&self) -> PyResult<(f64, f64, f64, f64)> {
        if self.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "to_quaternion requires 3D multivector, got Cl({})",
                self.dims
            )));
        }

        // Inverse of the mapping in from_quaternion:
        // R = w - x*e23 - y*e13 - z*e12
        // So: w = coeffs[0], x = -coeffs[6], y = -coeffs[5], z = -coeffs[3]
        let w = self.coeffs[0];
        let x = -self.coeffs[6]; // e23
        let y = -self.coeffs[5]; // e13
        let z = -self.coeffs[3]; // e12

        Ok((w, x, y, z))
    }

    /// Convert this 3D rotor to a 3x3 rotation matrix.
    ///
    /// Returns the matrix as a flat list of 9 elements in row-major order:
    /// [m00, m01, m02, m10, m11, m12, m20, m21, m22]
    ///
    /// The matrix is orthogonal with determinant +1 (proper rotation).
    ///
    /// Raises ValueError if not a 3D multivector.
    pub fn to_rotation_matrix(&self) -> PyResult<Vec<f64>> {
        if self.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "to_rotation_matrix requires 3D multivector, got Cl({})",
                self.dims
            )));
        }

        // Convert to quaternion first, then to rotation matrix
        let (w, x, y, z) = self.to_quaternion()?;

        // Standard quaternion to rotation matrix formula
        // Assumes unit quaternion (w² + x² + y² + z² = 1)
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        Ok(vec![
            1.0 - 2.0 * (yy + zz), // m00
            2.0 * (xy - wz),       // m01
            2.0 * (xz + wy),       // m02
            2.0 * (xy + wz),       // m10
            1.0 - 2.0 * (xx + zz), // m11
            2.0 * (yz - wx),       // m12
            2.0 * (xz - wy),       // m20
            2.0 * (yz + wx),       // m21
            1.0 - 2.0 * (xx + yy), // m22
        ])
    }

    /// Create a 3D rotor from a 3x3 rotation matrix.
    ///
    /// The matrix should be provided as a flat list of 9 elements in row-major order:
    /// [m00, m01, m02, m10, m11, m12, m20, m21, m22]
    ///
    /// The matrix should be orthogonal with determinant +1. Non-orthogonal
    /// matrices will produce undefined results.
    #[staticmethod]
    pub fn from_rotation_matrix(matrix: Vec<f64>) -> PyResult<Self> {
        if matrix.len() != 9 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "rotation matrix must have 9 elements, got {}",
                matrix.len()
            )));
        }

        let m00 = matrix[0];
        let m01 = matrix[1];
        let m02 = matrix[2];
        let m10 = matrix[3];
        let m11 = matrix[4];
        let m12 = matrix[5];
        let m20 = matrix[6];
        let m21 = matrix[7];
        let m22 = matrix[8];

        // Shepperd's method for extracting quaternion from rotation matrix
        // More numerically stable than direct formula
        let trace = m00 + m11 + m22;

        let (w, x, y, z) = if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            (0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s)
        } else if m00 > m11 && m00 > m22 {
            let s = 2.0 * (1.0 + m00 - m11 - m22).sqrt();
            ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
        } else if m11 > m22 {
            let s = 2.0 * (1.0 + m11 - m00 - m22).sqrt();
            ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
        } else {
            let s = 2.0 * (1.0 + m22 - m00 - m11).sqrt();
            ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
        };

        Ok(Self::from_quaternion(w, x, y, z))
    }

    /// Create a 3D rotor from Euler angles (intrinsic ZYX convention).
    ///
    /// This is the common "yaw-pitch-roll" convention:
    /// - yaw: rotation around Z axis (in radians)
    /// - pitch: rotation around Y axis (in radians)
    /// - roll: rotation around X axis (in radians)
    ///
    /// The rotations are applied in order: first roll (X), then pitch (Y), then yaw (Z).
    /// This matches the convention used in aerospace and robotics.
    #[staticmethod]
    pub fn from_euler_angles(yaw: f64, pitch: f64, roll: f64) -> Self {
        // Half angles
        let cy = (yaw / 2.0).cos();
        let sy = (yaw / 2.0).sin();
        let cp = (pitch / 2.0).cos();
        let sp = (pitch / 2.0).sin();
        let cr = (roll / 2.0).cos();
        let sr = (roll / 2.0).sin();

        // Quaternion from Euler angles (ZYX intrinsic = XYZ extrinsic)
        let w = cr * cp * cy + sr * sp * sy;
        let x = sr * cp * cy - cr * sp * sy;
        let y = cr * sp * cy + sr * cp * sy;
        let z = cr * cp * sy - sr * sp * cy;

        Self::from_quaternion(w, x, y, z)
    }

    /// Convert this 3D rotor to Euler angles (intrinsic ZYX convention).
    ///
    /// Returns (yaw, pitch, roll) in radians, where:
    /// - yaw: rotation around Z axis
    /// - pitch: rotation around Y axis
    /// - roll: rotation around X axis
    ///
    /// Warning: Gimbal lock occurs when pitch = ±π/2. In this case,
    /// yaw and roll become coupled and the decomposition is not unique.
    ///
    /// Raises ValueError if not a 3D multivector.
    pub fn to_euler_angles(&self) -> PyResult<(f64, f64, f64)> {
        if self.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "to_euler_angles requires 3D multivector, got Cl({})",
                self.dims
            )));
        }

        let (w, x, y, z) = self.to_quaternion()?;

        // Roll (x-axis rotation)
        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (y-axis rotation)
        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            // Gimbal lock
            std::f64::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        // Yaw (z-axis rotation)
        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        Ok((yaw, pitch, roll))
    }

    /// Create a bivector (grade-2) from components.
    ///
    /// For 2D: provide [e12_coeff]
    /// For 3D: provide [e12_coeff, e13_coeff, e23_coeff]
    /// For 4D: provide [e12, e13, e14, e23, e24, e34]
    ///
    /// Components are in lexicographic order of basis vector indices.
    ///
    /// Example:
    /// ```python
    /// B = Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)  # e12 in Cl(3)
    /// ```
    #[staticmethod]
    pub fn from_bivector(components: Vec<f64>, dims: usize) -> PyResult<Self> {
        if dims < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension must be at least 2 for bivectors",
            ));
        }
        // Number of grade-2 blades is C(n, 2) = n*(n-1)/2
        let expected_len = dims * (dims - 1) / 2;
        if components.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} bivector components for Cl({}) (got {}); \
                components are [e12, e13, e23, ...] in lexicographic order",
                expected_len,
                dims,
                components.len()
            )));
        }
        let size = 1usize << dims;
        let mut coeffs = vec![0.0; size];
        // Iterate through grade-2 blades in lexicographic order
        let mut comp_idx = 0;
        for i in 0..dims {
            for j in (i + 1)..dims {
                // Blade index for e_i ∧ e_j where i < j
                let blade_idx = (1 << i) | (1 << j);
                coeffs[blade_idx] = components[comp_idx];
                comp_idx += 1;
            }
        }
        Ok(Multivector { coeffs, dims })
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
        Ok(Multivector {
            coeffs: result,
            dims: self.dims,
        })
    }

    /// Negate the coefficients of a specific grade.
    ///
    /// Returns a new multivector with grade k coefficients negated.
    /// All other grades are unchanged.
    pub fn negate_grade(&self, k: usize) -> PyResult<Self> {
        if k > self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "grade {} exceeds algebra dimension {} (max grade is {})",
                k, self.dims, self.dims
            )));
        }
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| if algebra::blade_grade(i) == k { -c } else { c })
            .collect();
        Ok(Multivector {
            coeffs,
            dims: self.dims,
        })
    }

    /// Clear (zero out) a specific grade of this multivector.
    ///
    /// Returns a new multivector with grade k coefficients set to zero.
    /// All other grades are unchanged.
    ///
    /// Example:
    /// ```python
    /// # Remove the scalar part of a multivector
    /// mv_no_scalar = mv.clear_grade(0)
    ///
    /// # Remove the vector part
    /// mv_no_vector = mv.clear_grade(1)
    /// ```
    ///
    /// # Errors
    /// Returns an error if k exceeds the algebra dimension.
    pub fn clear_grade(&self, k: usize) -> PyResult<Self> {
        if k > self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "grade {} exceeds algebra dimension {} (max grade is {})",
                k, self.dims, self.dims
            )));
        }
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| if algebra::blade_grade(i) == k { 0.0 } else { c })
            .collect();
        Ok(Multivector {
            coeffs,
            dims: self.dims,
        })
    }

    /// Scale a specific grade by a factor.
    ///
    /// Returns a new multivector with grade k coefficients multiplied by factor.
    /// All other grades are unchanged.
    ///
    /// Example:
    /// ```python
    /// # Double the vector part
    /// mv2 = mv.scale_grade(1, 2.0)
    ///
    /// # Halve the bivector part
    /// mv2 = mv.scale_grade(2, 0.5)
    /// ```
    ///
    /// # Errors
    /// Returns an error if k exceeds the algebra dimension.
    pub fn scale_grade(&self, k: usize, factor: f64) -> PyResult<Self> {
        if k > self.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "grade {} exceeds algebra dimension {} (max grade is {})",
                k, self.dims, self.dims
            )));
        }
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                if algebra::blade_grade(i) == k {
                    c * factor
                } else {
                    c
                }
            })
            .collect();
        Ok(Multivector {
            coeffs,
            dims: self.dims,
        })
    }

    /// Add a scalar value to this multivector.
    ///
    /// Returns a new multivector with the scalar part increased by value.
    /// Equivalent to `mv + Multivector.from_scalar(value, mv.dims)`.
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([1.0, 2.0])
    /// mv = v.add_scalar(5.0)  # 5 + e1 + 2*e2
    /// mv.scalar()  # 5.0
    /// ```
    pub fn add_scalar(&self, value: f64) -> Self {
        let mut coeffs = self.coeffs.clone();
        coeffs[0] += value;
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Return a copy with the scalar part set to a specific value.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_vector([1.0, 2.0])
    /// mv2 = mv.with_scalar(10.0)  # 10 + e1 + 2*e2
    /// mv2.scalar()  # 10.0
    /// ```
    pub fn with_scalar(&self, value: f64) -> Self {
        let mut coeffs = self.coeffs.clone();
        coeffs[0] = value;
        Multivector {
            coeffs,
            dims: self.dims,
        }
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
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }
        let size = self.coeffs.len();
        let mut result = vec![0.0f64; size];

        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            for (j, &b) in other.coeffs.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let (blade, sign) = algebra::blade_product(i, j);
                result[blade] += sign * a * b;
            }
        }

        Ok(Multivector {
            coeffs: result,
            dims: self.dims,
        })
    }

    /// Alias for geometric_product.
    pub fn gp(&self, other: &Multivector) -> PyResult<Self> {
        self.geometric_product(other)
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
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }
        let size = self.coeffs.len();
        let mut result = vec![0.0f64; size];

        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            for (j, &b) in other.coeffs.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                // Outer product is zero when blades share a basis vector.
                if i & j != 0 {
                    continue;
                }
                let (blade, sign) = algebra::blade_product(i, j);
                result[blade] += sign * a * b;
            }
        }

        Ok(Multivector {
            coeffs: result,
            dims: self.dims,
        })
    }

    /// Alias for outer_product (wedge product).
    pub fn wedge(&self, other: &Multivector) -> PyResult<Self> {
        self.outer_product(other)
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
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }
        let size = self.coeffs.len();
        let mut result = vec![0.0f64; size];

        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let grade_a = algebra::blade_grade(i);
            for (j, &b) in other.coeffs.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let grade_b = algebra::blade_grade(j);
                // Left contraction is zero when grade(A) > grade(B)
                if grade_a > grade_b {
                    continue;
                }
                // Left contraction requires A ⊆ B (all basis vectors in A appear in B)
                if (i & j) != i {
                    continue;
                }
                let (blade, sign) = algebra::blade_product(i, j);
                let result_grade = algebra::blade_grade(blade);
                // Only keep the grade (s-r) component
                if result_grade == grade_b - grade_a {
                    result[blade] += sign * a * b;
                }
            }
        }

        Ok(Multivector {
            coeffs: result,
            dims: self.dims,
        })
    }

    /// Alias for left_contraction.
    pub fn inner(&self, other: &Multivector) -> PyResult<Self> {
        self.left_contraction(other)
    }

    /// Short alias for left_contraction.
    pub fn lc(&self, other: &Multivector) -> PyResult<Self> {
        self.left_contraction(other)
    }

    /// Compute the right contraction of two multivectors.
    ///
    /// The right contraction A ⌊ B is the "dual" of left contraction.
    /// For blades of grade r and s: A_r ⌊ B_s = ⟨A_r B_s⟩_{r-s} if r >= s, else 0.
    ///
    /// While left contraction (A ⌋ B) asks "how much of B is along A",
    /// right contraction (A ⌊ B) asks "how much of A is along B".
    ///
    /// Both multivectors must have the same dimension.
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn right_contraction(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }
        let size = self.coeffs.len();
        let mut result = vec![0.0f64; size];

        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let grade_a = algebra::blade_grade(i);
            for (j, &b) in other.coeffs.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let grade_b = algebra::blade_grade(j);
                // Right contraction is zero when grade(A) < grade(B)
                if grade_a < grade_b {
                    continue;
                }
                // Right contraction requires B ⊆ A (all basis vectors in B appear in A)
                if (i & j) != j {
                    continue;
                }
                let (blade, sign) = algebra::blade_product(i, j);
                let result_grade = algebra::blade_grade(blade);
                // Only keep the grade (r-s) component
                if result_grade == grade_a - grade_b {
                    result[blade] += sign * a * b;
                }
            }
        }

        Ok(Multivector {
            coeffs: result,
            dims: self.dims,
        })
    }

    /// Short alias for right_contraction.
    pub fn rc(&self, other: &Multivector) -> PyResult<Self> {
        self.right_contraction(other)
    }

    /// Compute the scalar product of two multivectors.
    ///
    /// The scalar product extracts only the scalar (grade-0) part of the
    /// geometric product: A ∗ B = ⟨A B⟩₀.
    ///
    /// This is a symmetric bilinear form. For vectors, it equals the dot product.
    /// For general multivectors, it measures the "overlap" between them.
    ///
    /// Returns a scalar multivector (only grade-0 component non-zero).
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn scalar_product(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }

        let mut scalar = 0.0f64;

        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            for (j, &b) in other.coeffs.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let (blade, sign) = algebra::blade_product(i, j);
                // Only accumulate scalar part (blade index 0)
                if blade == 0 {
                    scalar += sign * a * b;
                }
            }
        }

        Multivector::from_scalar(scalar, self.dims)
    }

    /// Alias for scalar_product (dot product notation).
    pub fn dot(&self, other: &Multivector) -> PyResult<Self> {
        self.scalar_product(other)
    }

    /// Compute the commutator product of two multivectors.
    ///
    /// The commutator (also called the antisymmetric product) is defined as:
    /// [A, B] = (A * B - B * A) / 2
    ///
    /// It measures the "non-commutativity" of A and B. For commuting elements,
    /// the commutator is zero.
    ///
    /// The commutator of two vectors gives their wedge product.
    /// The commutator of two bivectors gives another bivector.
    ///
    /// Reference: Dorst et al. ch.6 [VERIFY]
    pub fn commutator(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }
        let ab = self.geometric_product(other)?;
        let ba = other.geometric_product(self)?;
        let diff = ab.__sub__(&ba)?;
        Ok(diff.scale(0.5))
    }

    /// Compute the anticommutator product of two multivectors.
    ///
    /// The anticommutator (also called the symmetric product) is defined as:
    /// {A, B} = (A * B + B * A) / 2
    ///
    /// It extracts the symmetric part of the geometric product.
    ///
    /// Reference: Dorst et al. ch.6 [VERIFY]
    pub fn anticommutator(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }
        let ab = self.geometric_product(other)?;
        let ba = other.geometric_product(self)?;
        let sum = ab.__add__(&ba)?;
        Ok(sum.scale(0.5))
    }

    /// Short alias for commutator (cross product notation).
    pub fn x(&self, other: &Multivector) -> PyResult<Self> {
        self.commutator(other)
    }

    /// Compute the regressive product (meet) of two multivectors.
    ///
    /// The regressive product A ∨ B is the dual of the outer product:
    /// A ∨ B = (A* ∧ B*)*
    ///
    /// It computes the "meet" (intersection) of the subspaces represented
    /// by A and B. For example, the meet of two planes gives their line
    /// of intersection.
    ///
    /// Reference: Dorst et al. ch.5 [VERIFY]
    pub fn regressive(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }
        // A ∨ B = (A* ∧ B*)*
        let a_dual = self.dual()?;
        let b_dual = other.dual()?;
        let wedge = a_dual.outer_product(&b_dual)?;
        wedge.undual()
    }

    /// Alias for regressive product.
    pub fn meet(&self, other: &Multivector) -> PyResult<Self> {
        self.regressive(other)
    }

    /// Short alias for regressive (wedge symbol notation: ∨).
    pub fn vee(&self, other: &Multivector) -> PyResult<Self> {
        self.regressive(other)
    }

    /// Alias for outer product, for symmetry with meet.
    pub fn join(&self, other: &Multivector) -> PyResult<Self> {
        self.outer_product(other)
    }

    /// Return the coefficient array as a Python list.
    pub fn to_list(&self) -> Vec<f64> {
        self.coeffs.clone()
    }

    pub fn __repr__(&self) -> String {
        format!("Multivector({:?}, dims={})", self.coeffs, self.dims)
    }

    /// Human-readable string showing non-zero basis blade components.
    pub fn __str__(&self) -> String {
        let mut parts = Vec::new();
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let blade_name = Self::blade_name(i);
            if blade_name == "1" {
                parts.push(format!("{}", c));
            } else {
                parts.push(format!("{}*{}", c, blade_name));
            }
        }
        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" + ")
        }
    }

    /// Format the multivector with a custom format spec.
    ///
    /// Supports standard float format specs applied to each coefficient:
    /// - `.2f` - 2 decimal places
    /// - `.3e` - 3 decimal places in scientific notation
    /// - `.4g` - 4 significant figures
    /// - `+.2f` - always show sign, 2 decimal places
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_vector([1.234, 2.567])
    /// f"{mv:.2f}"  # "1.23*e1 + 2.57*e2"
    /// format(mv, ".1e")  # "1.2e+00*e1 + 2.6e+00*e2"
    /// ```
    pub fn __format__(&self, spec: &str) -> PyResult<String> {
        if spec.is_empty() {
            return Ok(self.__str__());
        }

        let mut parts = Vec::new();
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let blade_name = Self::blade_name(i);
            // Parse the format spec and apply to the coefficient
            let formatted = Self::format_float(c, spec)?;
            if blade_name == "1" {
                parts.push(formatted);
            } else {
                parts.push(format!("{}*{}", formatted, blade_name));
            }
        }
        if parts.is_empty() {
            Ok("0".to_string())
        } else {
            Ok(parts.join(" + "))
        }
    }

    /// Compare norms: returns true if self.norm() < other.norm().
    ///
    /// Useful for sorting multivectors by magnitude.
    pub fn norm_lt(&self, other: &Multivector) -> bool {
        self.norm() < other.norm()
    }

    /// Compare norms: returns true if self.norm() > other.norm().
    pub fn norm_gt(&self, other: &Multivector) -> bool {
        self.norm() > other.norm()
    }

    /// Return the grade with the largest total coefficient magnitude.
    ///
    /// For a mixed multivector, this identifies which grade dominates.
    /// Returns None for the zero multivector.
    ///
    /// Example:
    /// ```python
    /// mv = 0.1 + 5*e1 + 2*e2 + 0.5*e12
    /// mv.dominant_grade()  # 1 (vector part dominates)
    /// ```
    pub fn dominant_grade(&self) -> Option<usize> {
        if self.is_zero() {
            return None;
        }
        let mut grade_magnitudes: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::new();
        for (i, &c) in self.coeffs.iter().enumerate() {
            let grade = i.count_ones() as usize;
            *grade_magnitudes.entry(grade).or_insert(0.0) += c.abs();
        }
        grade_magnitudes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(g, _)| g)
    }

    /// Return the blade index with the largest absolute coefficient.
    ///
    /// Returns None for the zero multivector.
    ///
    /// Example:
    /// ```python
    /// mv = 1 + 5*e1 + 2*e2
    /// mv.dominant_blade()  # 1 (index of e1)
    /// ```
    pub fn dominant_blade(&self) -> Option<usize> {
        if self.is_zero() {
            return None;
        }
        self.coeffs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
    }

    /// Return blades sorted by absolute coefficient magnitude (descending).
    ///
    /// Returns list of (index, coefficient, grade) tuples.
    ///
    /// Example:
    /// ```python
    /// mv = 1 + 5*e1 + 2*e2
    /// mv.sorted_blades()  # [(1, 5.0, 1), (2, 2.0, 1), (0, 1.0, 0)]
    /// ```
    pub fn sorted_blades(&self) -> Vec<(usize, f64, usize)> {
        let mut blades: Vec<(usize, f64, usize)> = self
            .coeffs
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| {
                if c != 0.0 {
                    Some((i, c, i.count_ones() as usize))
                } else {
                    None
                }
            })
            .collect();
        blades.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        blades
    }

    /// Return the number of coefficients.
    pub fn __len__(&self) -> usize {
        self.coeffs.len()
    }

    /// Iterate over coefficients.
    ///
    /// Returns an iterator over all blade coefficients in index order.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_vector([1.0, 2.0])
    /// list(mv)  # [0.0, 1.0, 2.0, 0.0]
    /// for coeff in mv:
    ///     print(coeff)
    /// ```
    pub fn __iter__(slf: pyo3::PyRef<'_, Self>) -> CoeffIterator {
        CoeffIterator {
            coeffs: slf.coeffs.clone(),
            index: 0,
        }
    }

    /// Return the dimension of the base vector space.
    ///
    /// For Cl(n), this returns n. The multivector has 2^n coefficients.
    #[getter]
    pub fn dimension(&self) -> usize {
        self.dims
    }

    /// Alias for dimension.
    #[getter]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Return the number of coefficients (2^dimension).
    #[getter]
    pub fn n_coeffs(&self) -> usize {
        self.coeffs.len()
    }

    /// Get coefficient by index.
    pub fn __getitem__(&self, index: isize) -> PyResult<f64> {
        let len = self.coeffs.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        if idx < 0 || idx >= len {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of range for {} coefficients \
                (valid range: 0 to {}, or -{} to -1)",
                index,
                len,
                len - 1,
                len
            )))
        } else {
            Ok(self.coeffs[idx as usize])
        }
    }

    /// Multiplication: geometric product with Multivector, or scalar multiplication with float.
    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(mv) = other.extract::<PyRef<Multivector>>() {
            self.geometric_product(&mv)
        } else if let Ok(scalar) = other.extract::<f64>() {
            Ok(self.scale(scalar))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported operand type for *: expected Multivector or float",
            ))
        }
    }

    /// Right multiplication for scalar * Multivector.
    pub fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(scalar) = other.extract::<f64>() {
            Ok(self.scale(scalar))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported operand type for *: expected float",
            ))
        }
    }

    /// Division: scalar division or geometric division (A * B⁻¹).
    pub fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(mv) = other.extract::<PyRef<Multivector>>() {
            self.div(&mv)
        } else if let Ok(scalar) = other.extract::<f64>() {
            if scalar == 0.0 {
                Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "cannot divide multivector by zero",
                ))
            } else {
                Ok(self.scale(1.0 / scalar))
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported operand type for /: expected Multivector or float",
            ))
        }
    }

    pub fn __xor__(&self, other: &Multivector) -> PyResult<Self> {
        self.outer_product(other)
    }

    /// Left contraction operator (|).
    pub fn __or__(&self, other: &Multivector) -> PyResult<Self> {
        self.left_contraction(other)
    }

    /// Add two multivectors component-wise.
    ///
    /// Both multivectors must have the same dimension.
    pub fn __add__(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Multivector {
            coeffs,
            dims: self.dims,
        })
    }

    /// Subtract two multivectors component-wise.
    ///
    /// Both multivectors must have the same dimension.
    pub fn __sub__(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Multivector {
            coeffs,
            dims: self.dims,
        })
    }

    /// Negate a multivector (flip sign of all coefficients).
    pub fn __neg__(&self) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|c| -c).collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Unary plus (returns self unchanged).
    pub fn __pos__(&self) -> Self {
        self.clone()
    }

    /// Inversion operator (~) returns the reverse.
    ///
    /// In GA literature, ~A denotes the reverse of A.
    /// This flips the order of basis vectors in each blade.
    pub fn __invert__(&self) -> Self {
        self.reverse()
    }

    /// Multiply all coefficients by a scalar.
    pub fn scale(&self, scalar: f64) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|c| c * scalar).collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Check approximate equality within a tolerance.
    ///
    /// Returns true if all coefficients differ by at most `tol`.
    #[pyo3(signature = (other, tol=1e-10))]
    pub fn approx_eq(&self, other: &Multivector, tol: f64) -> bool {
        if self.dims != other.dims {
            return false;
        }
        self.coeffs
            .iter()
            .zip(other.coeffs.iter())
            .all(|(a, b)| (a - b).abs() <= tol)
    }

    /// Exponentiation operator (**).
    ///
    /// For integer exponents:
    /// - n > 0: repeated geometric product
    /// - n = 0: returns 1 (scalar)
    /// - n < 0: repeated inverse (requires invertible multivector)
    ///
    /// Useful for computing powers of rotors: R**2 rotates twice as much.
    pub fn __pow__(&self, exp: i32, _modulo: Option<i32>) -> PyResult<Self> {
        if exp == 0 {
            return Self::from_scalar(1.0, self.dims);
        }

        let (base, n) = if exp < 0 {
            (self.inverse()?, (-exp) as u32)
        } else {
            (self.clone(), exp as u32)
        };

        // Binary exponentiation
        let mut result = Self::from_scalar(1.0, self.dims)?;
        let mut current = base;
        let mut remaining = n;

        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result.geometric_product(&current)?;
            }
            remaining >>= 1;
            if remaining > 0 {
                current = current.geometric_product(&current)?;
            }
        }

        Ok(result)
    }

    /// Check if multivector is non-zero (truthiness).
    ///
    /// Returns True if any coefficient is non-zero.
    pub fn __bool__(&self) -> bool {
        self.coeffs.iter().any(|&c| c != 0.0)
    }

    /// Get coefficient by blade index (named method alternative to indexing).
    ///
    /// Equivalent to `mv[index]` but more explicit.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_vector([1.0, 2.0, 3.0])
    /// mv.coefficient(1)  # e1 coefficient = 1.0
    /// mv.coefficient(3)  # e12 coefficient = 0.0
    /// ```
    ///
    /// # Errors
    /// Returns ValueError if index is out of bounds.
    pub fn coefficient(&self, index: usize) -> PyResult<f64> {
        self.coeffs.get(index).copied().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "blade index {} out of bounds (max index is {})",
                index,
                self.coeffs.len() - 1
            ))
        })
    }

    /// Return a new multivector with a coefficient changed.
    ///
    /// Creates a copy with the coefficient at `index` set to `value`.
    /// The original multivector is unchanged.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.zero(2)
    /// mv2 = mv.set_coefficient(1, 5.0)  # Set e1 coefficient to 5
    /// mv2.coefficient(1)  # 5.0
    /// mv.coefficient(1)   # 0.0 (original unchanged)
    /// ```
    ///
    /// # Errors
    /// Returns ValueError if index is out of bounds.
    pub fn set_coefficient(&self, index: usize, value: f64) -> PyResult<Self> {
        if index >= self.coeffs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "blade index {} out of bounds (max index is {})",
                index,
                self.coeffs.len() - 1
            )));
        }
        let mut coeffs = self.coeffs.clone();
        coeffs[index] = value;
        Ok(Multivector {
            coeffs,
            dims: self.dims,
        })
    }

    /// Round all coefficients to a specified number of decimal places.
    ///
    /// Returns a new multivector with rounded coefficients.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_vector([1.234567, 2.345678])
    /// rounded = mv.round_coefficients(2)
    /// rounded.to_list()  # [0.0, 1.23, 2.35, 0.0]
    /// ```
    pub fn round_coefficients(&self, ndigits: i32) -> Self {
        let factor = 10.0_f64.powi(ndigits);
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .map(|&c| (c * factor).round() / factor)
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Clean near-zero coefficients by setting them to exactly zero.
    ///
    /// Any coefficient with absolute value less than epsilon is set to 0.0.
    /// Useful for removing floating-point noise after computations.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_vector([1.0, 1e-15, 0.5])
    /// cleaned = mv.clean(1e-10)
    /// cleaned.to_list()  # [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    /// ```
    #[pyo3(signature = (epsilon = 1e-10))]
    pub fn clean(&self, epsilon: f64) -> Self {
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .map(|&c| if c.abs() < epsilon { 0.0 } else { c })
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Return a multivector with absolute values of all coefficients.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_vector([-1.0, 2.0, -3.0])
    /// abs_mv = mv.abs_coefficients()
    /// abs_mv.to_list()  # [0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    /// ```
    pub fn abs_coefficients(&self) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|&c| c.abs()).collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Clamp all coefficients to a range [min_val, max_val].
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_vector([0.5, 2.0, -1.0])
    /// clamped = mv.clamp_coefficients(0.0, 1.0)
    /// clamped.to_list()  # [0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    /// ```
    pub fn clamp_coefficients(&self, min_val: f64, max_val: f64) -> Self {
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .map(|&c| c.clamp(min_val, max_val))
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Keep only coefficients from specified grades.
    ///
    /// Returns a new multivector with only the specified grades retained.
    /// Other grades are set to zero.
    ///
    /// Example:
    /// ```python
    /// mv = 1 + e1 + e2 + e12  # mixed grades 0, 1, 2
    /// mv.filter_grades([1])  # keeps only e1 + e2
    /// mv.filter_grades([0, 2])  # keeps 1 + e12
    /// ```
    pub fn filter_grades(&self, grades: Vec<usize>) -> Self {
        let grade_set: std::collections::HashSet<usize> = grades.into_iter().collect();
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let grade = i.count_ones() as usize;
                if grade_set.contains(&grade) {
                    c
                } else {
                    0.0
                }
            })
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Apply a threshold: zero out coefficients with absolute value below threshold.
    ///
    /// Similar to `clean()` but returns a new multivector.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([0.001, 1.0, 0.0001, 2.0])
    /// mv.threshold(0.01)  # zeroes the small values
    /// ```
    #[pyo3(signature = (min_abs = 1e-10))]
    pub fn threshold(&self, min_abs: f64) -> Self {
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .map(|&c| if c.abs() < min_abs { 0.0 } else { c })
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Return the sign of each coefficient (-1, 0, or 1).
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([-2.0, 0.0, 3.0, -0.5])
    /// signs = mv.sign()
    /// signs.to_list()  # [-1.0, 0.0, 1.0, -1.0]
    /// ```
    pub fn sign(&self) -> Self {
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .map(|&c| {
                if c > 0.0 {
                    1.0
                } else if c < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Return multivector with only positive coefficients (negative set to zero).
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([-1.0, 2.0, -3.0, 4.0])
    /// mv.positive_part().to_list()  # [0.0, 2.0, 0.0, 4.0]
    /// ```
    pub fn positive_part(&self) -> Self {
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .map(|&c| if c > 0.0 { c } else { 0.0 })
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Return multivector with only negative coefficients (positive set to zero).
    ///
    /// Note: the coefficients remain negative in the result.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([-1.0, 2.0, -3.0, 4.0])
    /// mv.negative_part().to_list()  # [-1.0, 0.0, -3.0, 0.0]
    /// ```
    pub fn negative_part(&self) -> Self {
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .map(|&c| if c < 0.0 { c } else { 0.0 })
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Return the norm (magnitude) via abs().
    pub fn __abs__(&self) -> f64 {
        self.norm()
    }

    /// Round all coefficients to ndigits decimal places.
    ///
    /// Implements Python's round() builtin.
    #[pyo3(signature = (ndigits=None))]
    pub fn __round__(&self, ndigits: Option<i32>) -> Self {
        match ndigits {
            Some(n) => self.round_coefficients(n),
            None => {
                // Round to nearest integer
                let coeffs: Vec<f64> = self.coeffs.iter().map(|&c| c.round()).collect();
                Multivector {
                    coeffs,
                    dims: self.dims,
                }
            }
        }
    }

    /// Floor all coefficients (round toward negative infinity).
    ///
    /// Implements math.floor() for multivectors.
    pub fn __floor__(&self) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|&c| c.floor()).collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Ceil all coefficients (round toward positive infinity).
    ///
    /// Implements math.ceil() for multivectors.
    pub fn __ceil__(&self) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|&c| c.ceil()).collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Truncate all coefficients (round toward zero).
    ///
    /// Implements math.trunc() for multivectors.
    pub fn __trunc__(&self) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|&c| c.trunc()).collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Convert to float if this is a scalar multivector.
    ///
    /// Raises ValueError if the multivector has non-scalar components.
    pub fn __float__(&self) -> PyResult<f64> {
        if self.is_scalar() {
            Ok(self.coeffs[0])
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "cannot convert non-scalar multivector to float",
            ))
        }
    }

    /// Convert to int if this is a scalar multivector.
    ///
    /// Raises ValueError if the multivector has non-scalar components.
    pub fn __int__(&self) -> PyResult<i64> {
        if self.is_scalar() {
            Ok(self.coeffs[0] as i64)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "cannot convert non-scalar multivector to int",
            ))
        }
    }

    /// Create a copy of this multivector.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Support for Python's copy.copy().
    pub fn __copy__(&self) -> Self {
        self.clone()
    }

    /// Support for Python's copy.deepcopy().
    pub fn __deepcopy__(&self, _memo: pyo3::PyObject) -> Self {
        self.clone()
    }

    /// Pickle support: get state as (coeffs, dims) tuple.
    pub fn __getstate__(&self) -> (Vec<f64>, usize) {
        (self.coeffs.clone(), self.dims)
    }

    /// Pickle support: reduce to constructor call.
    ///
    /// Returns (cls, args) where cls(*args) reconstructs the multivector.
    pub fn __reduce__(&self) -> PyResult<(pyo3::PyObject, (Vec<f64>,))> {
        pyo3::Python::with_gil(|py| {
            let cls = py.get_type::<Multivector>();
            Ok((cls.into_any().unbind(), (self.coeffs.clone(),)))
        })
    }

    /// Serialize this multivector to a dictionary.
    ///
    /// Returns a dict with 'coeffs' (list of floats) and 'dims' (int).
    /// Useful for JSON serialization.
    pub fn to_dict(&self) -> std::collections::HashMap<String, pyo3::PyObject> {
        use pyo3::IntoPyObjectExt;
        pyo3::Python::with_gil(|py| {
            let mut dict = std::collections::HashMap::new();
            dict.insert(
                "coeffs".to_string(),
                self.coeffs.clone().into_py_any(py).unwrap(),
            );
            dict.insert("dims".to_string(), self.dims.into_py_any(py).unwrap());
            dict
        })
    }

    /// Create a multivector from a dictionary.
    ///
    /// The dict must have 'coeffs' (list of floats) and 'dims' (int).
    #[staticmethod]
    pub fn from_dict(
        dict: std::collections::HashMap<String, pyo3::Py<pyo3::PyAny>>,
    ) -> PyResult<Self> {
        pyo3::Python::with_gil(|py| {
            let coeffs: Vec<f64> = dict
                .get("coeffs")
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'coeffs' key"))?
                .extract(py)?;

            let dims: usize = dict
                .get("dims")
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'dims' key"))?
                .extract(py)?;

            // Validate
            let expected_len = 1usize << dims;
            if coeffs.len() != expected_len {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "coeffs length {} doesn't match dims {} (expected {})",
                    coeffs.len(),
                    dims,
                    expected_len
                )));
            }

            Ok(Multivector { coeffs, dims })
        })
    }

    /// Compute the Euclidean distance between two vectors.
    ///
    /// This is the standard Euclidean distance: ||a - b||.
    ///
    /// Both multivectors must be pure vectors (grade 1 only).
    ///
    /// # Errors
    /// Returns an error if either multivector is not a pure vector.
    pub fn distance(&self, other: &Self) -> PyResult<f64> {
        if !self.is_vector() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "distance() requires both arguments to be vectors (self is not a vector)",
            ));
        }
        if !other.is_vector() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "distance() requires both arguments to be vectors (other is not a vector)",
            ));
        }
        let diff = self.__sub__(other)?;
        Ok(diff.norm())
    }

    /// Compute the midpoint between two vectors.
    ///
    /// Returns (a + b) / 2, the point equidistant from both vectors.
    ///
    /// Both multivectors must be pure vectors (grade 1 only).
    ///
    /// # Errors
    /// Returns an error if either multivector is not a pure vector.
    pub fn midpoint(&self, other: &Self) -> PyResult<Self> {
        if !self.is_vector() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "midpoint() requires both arguments to be vectors (self is not a vector)",
            ));
        }
        if !other.is_vector() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "midpoint() requires both arguments to be vectors (other is not a vector)",
            ));
        }
        let sum = self.__add__(other)?;
        Ok(sum.scale(0.5))
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
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let grade = algebra::blade_grade(i);
                c * algebra::reverse_sign(grade)
            })
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Alias for reverse, using the tilde notation common in GA literature.
    pub fn tilde(&self) -> Self {
        self.reverse()
    }

    /// Compute the grade involution (main involution) of this multivector.
    ///
    /// Grade involution negates odd-grade components while leaving even grades unchanged.
    /// For a blade of grade k: Â = (-1)^k * A.
    ///
    /// Properties:
    /// - Scalars and bivectors are unchanged
    /// - Vectors and trivectors are negated
    /// - (A * B)^ = Â * B̂ (automorphism)
    ///
    /// Useful for:
    /// - Separating even and odd parts: A_even = (A + Â)/2, A_odd = (A - Â)/2
    /// - Defining Clifford conjugate: A† = (Â)~ = (~A)^
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn grade_involution(&self) -> Self {
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let grade = algebra::blade_grade(i);
                c * algebra::grade_involution_sign(grade)
            })
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Alias for grade_involution (hat notation).
    pub fn involute(&self) -> Self {
        self.grade_involution()
    }

    /// Compute the Clifford conjugate of this multivector.
    ///
    /// Clifford conjugation combines reverse and grade involution:
    /// A† = (Â)~ = (~A)^ (both compositions give the same result).
    ///
    /// For a blade of grade k: A† = (-1)^(k(k+1)/2) * A.
    ///
    /// Sign pattern by grade:
    /// - Grade 0 (scalars): +1 (unchanged)
    /// - Grade 1 (vectors): -1 (negated)
    /// - Grade 2 (bivectors): -1 (negated)
    /// - Grade 3 (trivectors): +1 (unchanged)
    /// - Pattern repeats with period 4
    ///
    /// Useful for:
    /// - Computing norms in non-Euclidean algebras
    /// - Defining the "bar" conjugate in physics applications
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn clifford_conjugate(&self) -> Self {
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let grade = algebra::blade_grade(i);
                c * algebra::clifford_conjugate_sign(grade)
            })
            .collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
    }

    /// Alias for clifford_conjugate (dagger notation).
    pub fn conjugate(&self) -> Self {
        self.clifford_conjugate()
    }

    /// Extract the even-grade part of this multivector.
    ///
    /// Returns a new multivector containing only grades 0, 2, 4, ...
    /// Computed as (A + Â) / 2 where Â is the grade involution.
    ///
    /// Rotors are always even-grade multivectors.
    pub fn even(&self) -> Self {
        let inv = self.grade_involution();
        let sum = self
            .coeffs
            .iter()
            .zip(inv.coeffs.iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect();
        Multivector {
            coeffs: sum,
            dims: self.dims,
        }
    }

    /// Extract the odd-grade part of this multivector.
    ///
    /// Returns a new multivector containing only grades 1, 3, 5, ...
    /// Computed as (A - Â) / 2 where Â is the grade involution.
    pub fn odd(&self) -> Self {
        let inv = self.grade_involution();
        let sum = self
            .coeffs
            .iter()
            .zip(inv.coeffs.iter())
            .map(|(a, b)| (a - b) / 2.0)
            .collect();
        Multivector {
            coeffs: sum,
            dims: self.dims,
        }
    }

    /// Check if this multivector contains only even-grade components.
    ///
    /// Returns true if only grades 0, 2, 4, ... are non-zero.
    /// Scalars, bivectors, and rotors are typically even.
    #[pyo3(signature = (tol=1e-10))]
    pub fn is_even(&self, tol: f64) -> bool {
        self.coeffs
            .iter()
            .enumerate()
            .all(|(i, &c)| algebra::blade_grade(i).is_multiple_of(2) || c.abs() <= tol)
    }

    /// Check if this multivector contains only odd-grade components.
    ///
    /// Returns true if only grades 1, 3, 5, ... are non-zero.
    /// Vectors and trivectors are typically odd.
    #[pyo3(signature = (tol=1e-10))]
    pub fn is_odd(&self, tol: f64) -> bool {
        self.coeffs
            .iter()
            .enumerate()
            .all(|(i, &c)| !algebra::blade_grade(i).is_multiple_of(2) || c.abs() <= tol)
    }

    /// Return the number of distinct grades present.
    ///
    /// Zero multivector returns 0. A pure k-vector returns 1.
    pub fn grade_count(&self) -> usize {
        self.grades().len()
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
            if a == 0.0 {
                continue;
            }
            for (j, &b) in rev.coeffs.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
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

    /// Alias for norm (magnitude notation).
    pub fn magnitude(&self) -> f64 {
        self.norm()
    }

    /// Compute the norm of just the grade-k part of this multivector.
    ///
    /// Returns the norm of the projection onto grade k.
    /// Useful for analyzing the contribution of each grade.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    /// scalar_norm = mv.grade_norm(0)  # norm of scalar part
    /// vector_norm = mv.grade_norm(1)  # norm of vector part
    /// bivector_norm = mv.grade_norm(2)  # norm of bivector part
    /// ```
    pub fn grade_norm(&self, k: usize) -> PyResult<f64> {
        let grade_part = self.grade(k)?;
        Ok(grade_part.norm())
    }

    /// Compute the geometric product of this multivector with itself.
    ///
    /// Returns A * A. For vectors, this equals the squared norm.
    pub fn squared(&self) -> PyResult<Self> {
        self.geometric_product(self)
    }

    /// Return a normalized copy of this multivector (unit norm).
    ///
    /// Raises ValueError if the multivector has zero norm.
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn normalized(&self) -> PyResult<Self> {
        let n = self.norm();
        if n == 0.0 {
            Err(pyo3::exceptions::PyValueError::new_err(
                "cannot normalize zero multivector; normalization requires non-zero norm \
                (check that your multivector has at least one non-zero coefficient)",
            ))
        } else {
            Ok(self.scale(1.0 / n))
        }
    }

    /// Compute the inverse of this multivector.
    ///
    /// For a blade or versor A: A⁻¹ = Ã / (A * Ã) where Ã is the reverse.
    /// This satisfies A * A⁻¹ = 1.
    ///
    /// Works correctly for:
    /// - Scalars: s⁻¹ = 1/s
    /// - Vectors: v⁻¹ = v / |v|²
    /// - Blades: B⁻¹ = B̃ / (B * B̃)
    /// - Versors (products of non-null vectors)
    ///
    /// Raises ValueError if the multivector has zero norm (not invertible).
    ///
    /// Note: For general multivectors that are not blades or versors,
    /// this formula may not produce a true inverse. Use with caution
    /// for mixed-grade multivectors.
    ///
    /// Reference: Dorst et al. ch.4 [VERIFY]
    pub fn inverse(&self) -> PyResult<Self> {
        let norm_sq = self.norm_squared();
        if norm_sq == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot invert multivector with zero norm; \
                only non-null blades and versors are invertible",
            ));
        }
        // A⁻¹ = Ã / (A * Ã) = Ã / norm_squared
        Ok(self.reverse().scale(1.0 / norm_sq))
    }

    /// Compute A / B as A * B⁻¹ (right division).
    ///
    /// Raises ValueError if B is not invertible.
    pub fn div(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }
        let other_inv = other.inverse()?;
        self.geometric_product(&other_inv)
    }

    // =========================================================================
    // ROTOR OPERATIONS
    // =========================================================================

    /// Apply this rotor to a multivector via the sandwich product: R * x * ~R.
    ///
    /// This is the standard way to apply a rotation (or other orthogonal transformation)
    /// represented by a rotor. The sandwich product preserves the grade structure of x.
    ///
    /// For a unit rotor R and a vector v: R.sandwich(v) rotates v.
    /// For a unit rotor R and a bivector B: R.sandwich(B) rotates the plane B.
    ///
    /// The rotor should be normalized (R * ~R = 1) for this to be a pure rotation.
    /// Non-unit rotors will also scale the result.
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// e2 = Multivector.from_vector([0.0, 1.0, 0.0])
    /// R = Multivector.rotor_from_vectors(e1, e2)  # 90° rotation
    /// rotated = R.sandwich(e1)  # should equal e2
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    pub fn sandwich(&self, x: &Multivector) -> PyResult<Self> {
        if self.dims != x.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: rotor is Cl({}) but operand is Cl({}); \
                both must have the same dimension",
                self.dims, x.dims
            )));
        }
        // R * x * ~R
        let rx = self.geometric_product(x)?;
        let r_rev = self.reverse();
        rx.geometric_product(&r_rev)
    }

    /// Alias for sandwich product. Common in robotics literature.
    pub fn apply(&self, x: &Multivector) -> PyResult<Self> {
        self.sandwich(x)
    }

    /// Check if this multivector is a valid rotor (unit versor).
    ///
    /// A rotor R satisfies R * ~R = 1 (within tolerance).
    /// Returns true if this multivector is approximately a unit rotor.
    #[pyo3(signature = (tol=1e-10))]
    pub fn is_rotor(&self, tol: f64) -> bool {
        let norm_sq = self.norm_squared();
        (norm_sq - 1.0).abs() <= tol
    }

    /// Check if this multivector has unit norm.
    ///
    /// Returns true if |A| ≈ 1 within tolerance.
    #[pyo3(signature = (tol=1e-10))]
    pub fn is_unit(&self, tol: f64) -> bool {
        (self.norm() - 1.0).abs() <= tol
    }

    /// Check if this multivector is a blade (simple k-vector).
    ///
    /// A blade is a multivector that can be written as the outer product
    /// of k vectors. For a blade B of grade k > 0: B ∧ B = 0.
    /// Scalars (grade 0) are blades by definition.
    ///
    /// Returns true if this multivector is approximately a blade.
    ///
    /// Reference: Dorst et al. ch.4 [VERIFY]
    #[pyo3(signature = (tol=1e-10))]
    pub fn is_blade(&self, tol: f64) -> bool {
        // Check if single grade
        let grades = self.grades();
        if grades.len() != 1 {
            return false;
        }

        // Scalars (grade 0) are blades by definition
        if grades[0] == 0 {
            return true;
        }

        // For a blade of grade k > 0: B ∧ B = 0
        if let Ok(bb) = self.outer_product(self) {
            bb.norm() <= tol
        } else {
            false
        }
    }

    /// Check if this multivector is a versor.
    ///
    /// A versor is a product of non-null vectors. Versors are invertible
    /// and their sandwich product preserves grades.
    ///
    /// A multivector V is a versor if V * ~V is a non-zero scalar.
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    #[pyo3(signature = (tol=1e-10))]
    pub fn is_versor(&self, tol: f64) -> bool {
        // V * ~V should be a scalar (grade 0 only)
        let rev = self.reverse();
        if let Ok(vvr) = self.geometric_product(&rev) {
            // Check that only grade 0 is non-zero
            for (i, &c) in vvr.coeffs.iter().enumerate() {
                if i == 0 {
                    // Scalar part should be non-zero
                    if c.abs() <= tol {
                        return false;
                    }
                } else {
                    // All other grades should be zero
                    if c.abs() > tol {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Return a list of grades that have non-zero components.
    ///
    /// Example:
    /// ```text
    /// v = Multivector.from_vector([1.0, 2.0, 3.0])
    /// v.grades()  # returns [1]
    ///
    /// rotor = e1 * e2  # a bivector
    /// rotor.grades()  # returns [2]
    ///
    /// mixed = scalar + vector
    /// mixed.grades()  # returns [0, 1]
    /// ```
    pub fn grades(&self) -> Vec<usize> {
        let mut result = Vec::new();
        let mut seen = vec![false; self.dims + 1];

        for (i, &c) in self.coeffs.iter().enumerate() {
            if c != 0.0 {
                let g = algebra::blade_grade(i);
                if !seen[g] {
                    seen[g] = true;
                    result.push(g);
                }
            }
        }
        result.sort();
        result
    }

    /// Return the highest grade with a non-zero component.
    ///
    /// Returns None (Python: None) if the multivector is zero.
    pub fn max_grade(&self) -> Option<usize> {
        let mut max_g: Option<usize> = None;

        for (i, &c) in self.coeffs.iter().enumerate() {
            if c != 0.0 {
                let g = algebra::blade_grade(i);
                max_g = Some(max_g.map_or(g, |m| m.max(g)));
            }
        }
        max_g
    }

    /// Return the lowest grade with a non-zero component.
    ///
    /// Returns None (Python: None) if the multivector is zero.
    pub fn min_grade(&self) -> Option<usize> {
        let mut min_g: Option<usize> = None;

        for (i, &c) in self.coeffs.iter().enumerate() {
            if c != 0.0 {
                let g = algebra::blade_grade(i);
                min_g = Some(min_g.map_or(g, |m| m.min(g)));
            }
        }
        min_g
    }

    /// Check if a specific grade has non-zero components.
    ///
    /// Returns true if there is at least one non-zero coefficient at grade k.
    pub fn has_grade(&self, k: usize) -> bool {
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c != 0.0 && algebra::blade_grade(i) == k {
                return true;
            }
        }
        false
    }

    /// Return the grade if this is a pure k-vector, None otherwise.
    ///
    /// A pure k-vector has all its non-zero components at a single grade.
    /// Returns None for zero multivectors or mixed-grade multivectors.
    pub fn pure_grade(&self) -> Option<usize> {
        let grades = self.grades();
        if grades.len() == 1 {
            Some(grades[0])
        } else {
            None
        }
    }

    /// Check if this multivector is a scalar (grade 0 only).
    pub fn is_scalar(&self) -> bool {
        self.pure_grade() == Some(0)
    }

    /// Check if this multivector is a vector (grade 1 only).
    pub fn is_vector(&self) -> bool {
        self.pure_grade() == Some(1)
    }

    /// Check if this multivector is a bivector (grade 2 only).
    pub fn is_bivector(&self) -> bool {
        self.pure_grade() == Some(2)
    }

    /// Check if this multivector is a trivector (grade 3 only).
    pub fn is_trivector(&self) -> bool {
        self.pure_grade() == Some(3)
    }

    /// Check if this multivector is the pseudoscalar (highest grade only).
    pub fn is_pseudoscalar(&self) -> bool {
        self.pure_grade() == Some(self.dims)
    }

    /// Check if this multivector is zero (all coefficients are zero).
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0.0)
    }

    /// Return the non-zero components as a list of (index, coefficient) pairs.
    ///
    /// The index is the basis blade index (binary encoding).
    /// Use this for sparse iteration over multivector components.
    pub fn components(&self) -> Vec<(usize, f64)> {
        self.coeffs
            .iter()
            .enumerate()
            .filter(|(_, &c)| c != 0.0)
            .map(|(i, &c)| (i, c))
            .collect()
    }

    /// Return all coefficients as a list.
    ///
    /// The list has length 2^n where n is the dimension.
    /// Index i corresponds to the coefficient of the basis blade
    /// with binary index i.
    pub fn coefficients(&self) -> Vec<f64> {
        self.coeffs.clone()
    }

    /// Return the indices of non-zero basis blades.
    pub fn blade_indices(&self) -> Vec<usize> {
        self.coeffs
            .iter()
            .enumerate()
            .filter(|(_, &c)| c != 0.0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Sum of all coefficients.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, 2.0, 3.0, 4.0])
    /// mv.sum_coefficients()  # 10.0
    /// ```
    pub fn sum_coefficients(&self) -> f64 {
        self.coeffs.iter().sum()
    }

    /// Maximum coefficient value.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, -5.0, 3.0, 2.0])
    /// mv.max_coefficient()  # 3.0
    /// ```
    pub fn max_coefficient(&self) -> f64 {
        self.coeffs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum coefficient value.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, -5.0, 3.0, 2.0])
    /// mv.min_coefficient()  # -5.0
    /// ```
    pub fn min_coefficient(&self) -> f64 {
        self.coeffs.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Count of non-zero coefficients.
    ///
    /// Useful for checking sparsity of a multivector.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_vector([1.0, 0.0, 3.0])
    /// mv.nonzero_count()  # 2
    /// ```
    pub fn nonzero_count(&self) -> usize {
        self.coeffs.iter().filter(|&&c| c != 0.0).count()
    }

    /// Compute the sparsity of this multivector.
    ///
    /// Returns the fraction of coefficients that are zero, from 0.0 to 1.0.
    /// A sparsity of 1.0 means all coefficients are zero.
    /// A sparsity of 0.0 means all coefficients are non-zero.
    ///
    /// Useful for debugging and understanding multivector structure.
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([1.0, 2.0, 3.0])  # 3 non-zero of 8 coeffs
    /// v.sparsity()  # 0.625 (5 zeros out of 8)
    /// ```
    pub fn sparsity(&self) -> f64 {
        if self.coeffs.is_empty() {
            return 1.0;
        }
        let zero_count = self.coeffs.iter().filter(|&&c| c == 0.0).count();
        zero_count as f64 / self.coeffs.len() as f64
    }

    /// Compute the density of this multivector.
    ///
    /// Returns the fraction of coefficients that are non-zero, from 0.0 to 1.0.
    /// This is 1.0 - sparsity().
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([1.0, 2.0, 3.0])  # 3 non-zero of 8 coeffs
    /// v.density()  # 0.375 (3 non-zeros out of 8)
    /// ```
    pub fn density(&self) -> f64 {
        1.0 - self.sparsity()
    }

    /// Mean (average) of all coefficients.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, 2.0, 3.0, 4.0])
    /// mv.mean_coefficient()  # 2.5
    /// ```
    pub fn mean_coefficient(&self) -> f64 {
        if self.coeffs.is_empty() {
            return 0.0;
        }
        self.sum_coefficients() / self.coeffs.len() as f64
    }

    /// Variance of all coefficients.
    ///
    /// Uses population variance (divides by n, not n-1).
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, 2.0, 3.0, 4.0])
    /// mv.variance_coefficient()  # 1.25
    /// ```
    pub fn variance_coefficient(&self) -> f64 {
        if self.coeffs.is_empty() {
            return 0.0;
        }
        let mean = self.mean_coefficient();
        let sum_sq: f64 = self.coeffs.iter().map(|&c| (c - mean).powi(2)).sum();
        sum_sq / self.coeffs.len() as f64
    }

    /// Standard deviation of all coefficients.
    ///
    /// Square root of the variance.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, 2.0, 3.0, 4.0])
    /// mv.std_coefficient()  # ~1.118
    /// ```
    pub fn std_coefficient(&self) -> f64 {
        self.variance_coefficient().sqrt()
    }

    /// Median of all coefficients.
    ///
    /// For even number of coefficients, returns average of two middle values.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, 2.0, 3.0, 4.0])
    /// mv.median_coefficient()  # 2.5
    /// ```
    pub fn median_coefficient(&self) -> f64 {
        if self.coeffs.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.coeffs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Range of coefficients (max - min).
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, -5.0, 3.0, 2.0])
    /// mv.range_coefficient()  # 8.0 (3 - (-5))
    /// ```
    pub fn range_coefficient(&self) -> f64 {
        self.max_coefficient() - self.min_coefficient()
    }

    /// L1 norm (sum of absolute values of coefficients).
    ///
    /// Different from the geometric norm, this is the taxicab/Manhattan norm
    /// of the coefficient vector.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, -2.0, 3.0, -4.0])
    /// mv.l1_norm()  # 10.0
    /// ```
    pub fn l1_norm(&self) -> f64 {
        self.coeffs.iter().map(|&c| c.abs()).sum()
    }

    /// L-infinity norm (maximum absolute coefficient).
    ///
    /// Also known as the supremum norm or Chebyshev norm.
    ///
    /// Example:
    /// ```python
    /// mv = Multivector.from_list([1.0, -5.0, 3.0, 2.0])
    /// mv.linf_norm()  # 5.0
    /// ```
    pub fn linf_norm(&self) -> f64 {
        self.coeffs.iter().map(|&c| c.abs()).fold(0.0_f64, f64::max)
    }

    /// Spherical linear interpolation between two unit rotors.
    ///
    /// Interpolates from self (at t=0) to other (at t=1) along the
    /// shortest geodesic on the rotor manifold.
    ///
    /// Both rotors should be unit rotors (R * ~R = 1).
    ///
    /// Example:
    /// ```text
    /// R1 = Multivector.rotor_from_vectors(e1, e2)  # 90° rotation
    /// R2 = Multivector.rotor_from_vectors(e1, e3)  # different rotation
    /// R_mid = R1.slerp(R2, 0.5)  # halfway between
    /// ```
    ///
    /// Reference: Dorst et al. ch.10 [VERIFY]
    pub fn slerp(&self, other: &Multivector, t: f64) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left rotor is Cl({}) but right rotor is Cl({}); \
                both rotors must have the same dimension",
                self.dims, other.dims
            )));
        }

        // R_t = R1 * (R1^-1 * R2)^t
        // Using log/exp: R_t = R1 * exp(t * log(R1^-1 * R2))

        // First compute R1^-1 * R2
        let r1_inv = self.inverse()?;
        let r1_inv_r2 = r1_inv.geometric_product(other)?;

        // Take the log to get the bivector
        let log_ratio = r1_inv_r2.log()?;

        // Scale by t
        let scaled = log_ratio.scale(t);

        // Exponentiate back to rotor
        let exp_scaled = scaled.exp()?;

        // Multiply by R1
        self.geometric_product(&exp_scaled)
    }

    /// Linear interpolation between two multivectors.
    ///
    /// Returns (1 - t) * self + t * other.
    ///
    /// For t=0 returns self, for t=1 returns other.
    /// Unlike slerp, this doesn't preserve unit norm for rotors.
    pub fn lerp(&self, other: &Multivector, t: f64) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }

        let a = self.scale(1.0 - t);
        let b = other.scale(t);
        a.__add__(&b)
    }

    /// Normalized linear interpolation between two multivectors.
    ///
    /// Returns normalize((1 - t) * self + t * other).
    ///
    /// This is lerp followed by normalization. It's faster than slerp but
    /// doesn't have constant angular velocity - the interpolation speeds up
    /// near the endpoints when vectors are far apart.
    ///
    /// Commonly used for:
    /// - Fast vector interpolation where constant speed isn't critical
    /// - Quick approximation of slerp when t is near 0 or 1
    /// - Performance-critical applications where many interpolations are needed
    ///
    /// For t=0 returns normalized self, for t=1 returns normalized other.
    ///
    /// Raises ValueError if the interpolated result has zero norm.
    ///
    /// Example:
    /// ```python
    /// v1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// v2 = Multivector.from_vector([0.0, 1.0, 0.0])
    /// mid = v1.nlerp(v2, 0.5)  # Normalized midpoint
    /// assert abs(mid.norm() - 1.0) < 1e-10
    /// ```
    pub fn nlerp(&self, other: &Multivector, t: f64) -> PyResult<Self> {
        let lerped = self.lerp(other, t)?;
        lerped.normalized()
    }

    /// Extract the rotation angle from a rotor.
    ///
    /// For a rotor R = cos(θ/2) + sin(θ/2) * B̂ where B̂ is a unit bivector,
    /// this returns θ (the full rotation angle, not the half-angle).
    ///
    /// Returns a value in [0, 2π].
    ///
    /// Raises ValueError if the multivector is not a valid rotor.
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    #[pyo3(signature = (tol=1e-10))]
    pub fn rotation_angle(&self, tol: f64) -> PyResult<f64> {
        if !self.is_rotor(tol) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "multivector is not a valid rotor (must satisfy R * ~R = 1)",
            ));
        }

        // R = cos(θ/2) + sin(θ/2) * B̂
        // The scalar part is cos(θ/2)
        let cos_half = self.scalar();

        // The bivector part has magnitude sin(θ/2)
        let bivector = self.grade(2)?;
        let sin_half = bivector.norm();

        // θ = 2 * atan2(sin(θ/2), cos(θ/2))
        let half_angle = sin_half.atan2(cos_half);
        Ok(2.0 * half_angle.abs())
    }

    /// Extract the rotation plane (bivector) from a rotor.
    ///
    /// For a rotor R = cos(θ/2) + sin(θ/2) * B̂, this returns the unit
    /// bivector B̂ representing the plane of rotation.
    ///
    /// For the identity rotor (θ = 0), returns the zero bivector.
    ///
    /// Raises ValueError if the multivector is not a valid rotor.
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    #[pyo3(signature = (tol=1e-10))]
    pub fn rotation_plane(&self, tol: f64) -> PyResult<Self> {
        if !self.is_rotor(tol) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "multivector is not a valid rotor (must satisfy R * ~R = 1)",
            ));
        }

        let bivector = self.grade(2)?;
        let norm = bivector.norm();

        if norm <= tol {
            // Identity rotor - return zero bivector
            Ok(bivector)
        } else {
            // Normalize to get unit bivector
            Ok(bivector.scale(1.0 / norm))
        }
    }

    /// Compute the 3D cross product of two vectors.
    ///
    /// The cross product a × b is computed as the dual of the wedge product:
    /// a × b = (a ∧ b)*
    ///
    /// This is only defined for vectors in 3D (Cl(3)).
    ///
    /// Raises ValueError if dimensions are not 3 or inputs are not vectors.
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn cross(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cross product is only defined in 3D; this multivector is in Cl({})",
                self.dims
            )));
        }
        if other.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cross product is only defined in 3D; other multivector is in Cl({})",
                other.dims
            )));
        }

        // Cross product = dual of wedge product
        let wedge = self.outer_product(other)?;
        wedge.dual()
    }

    /// Decompose a rotor into axis-angle representation.
    ///
    /// Returns (axis, angle) where axis is a unit vector perpendicular to
    /// the rotation plane, and angle is the rotation angle in radians.
    ///
    /// Only defined for 3D rotors. In 3D, every rotation has a unique axis.
    ///
    /// Raises ValueError if not a valid 3D rotor or if the rotor is identity.
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    #[pyo3(signature = (tol=1e-10))]
    pub fn axis_angle(&self, tol: f64) -> PyResult<(Self, f64)> {
        if self.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "axis_angle is only defined in 3D; this multivector is in Cl({})",
                self.dims
            )));
        }
        if !self.is_rotor(tol) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "multivector is not a valid rotor (must satisfy R * ~R = 1)",
            ));
        }

        let angle = self.rotation_angle(tol)?;

        if angle.abs() <= tol {
            // Identity rotor - no unique axis
            return Err(pyo3::exceptions::PyValueError::new_err(
                "identity rotor has no unique rotation axis; angle is zero",
            ));
        }

        // The rotation plane bivector's dual is the axis
        let plane = self.rotation_plane(tol)?;
        let axis = plane.dual()?;

        // Normalize the axis (should already be unit, but be safe)
        let axis_normalized = axis.normalized()?;

        Ok((axis_normalized, angle))
    }

    // =========================================================================
    // GEOMETRIC PREDICATES
    // =========================================================================

    /// Compute the angle between two vectors.
    ///
    /// Returns the angle in radians between this vector and other.
    /// Both must be grade-1 multivectors (vectors).
    ///
    /// The angle is computed as: θ = arccos(a · b / (|a| |b|))
    ///
    /// Raises ValueError if either multivector is not a vector or has zero norm.
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn angle_between(&self, other: &Multivector) -> PyResult<f64> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }

        // Check both are vectors (grade 1)
        let self_grades = self.grades();
        let other_grades = other.grades();

        if self_grades != vec![1] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "first operand must be a vector (grade 1); \
                use grade(1) to extract the vector part if needed",
            ));
        }
        if other_grades != vec![1] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "second operand must be a vector (grade 1); \
                use grade(1) to extract the vector part if needed",
            ));
        }

        let norm_a = self.norm();
        let norm_b = other.norm();

        if norm_a == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "first vector has zero norm; angle is undefined for zero vectors",
            ));
        }
        if norm_b == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "second vector has zero norm; angle is undefined for zero vectors",
            ));
        }

        // Compute dot product (scalar part of geometric product for vectors)
        let gp = self.geometric_product(other)?;
        let dot = gp.scalar();

        // cos(θ) = a · b / (|a| |b|)
        let cos_theta = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
        Ok(cos_theta.acos())
    }

    /// Compute the signed angle between two vectors given a reference normal.
    ///
    /// Returns the angle in radians from self to other, with sign determined
    /// by the normal vector. Positive if the rotation from self to other
    /// follows the right-hand rule around the normal.
    ///
    /// This is useful for:
    /// - Measuring rotation direction in a plane
    /// - Computing winding angles
    /// - Determining clockwise vs counterclockwise
    ///
    /// All three vectors must be in the same dimension space.
    /// Only works in 3D (requires cross product for sign determination).
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// e2 = Multivector.from_vector([0.0, 1.0, 0.0])
    /// e3 = Multivector.from_vector([0.0, 0.0, 1.0])
    /// angle = e1.signed_angle(e2, e3)  # +π/2
    /// angle = e2.signed_angle(e1, e3)  # -π/2
    /// ```
    ///
    /// Reference: Standard signed angle computation
    pub fn signed_angle(&self, other: &Multivector, normal: &Multivector) -> PyResult<f64> {
        if self.dims != 3 || other.dims != 3 || normal.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "signed_angle requires all vectors to be in 3D",
            ));
        }

        // Get unsigned angle
        let unsigned = self.angle_between(other)?;

        // Compute cross product to determine sign
        let cross = self.cross(other)?;

        // Sign is determined by whether cross is same direction as normal
        let dot_mv = cross.scalar_product(normal)?;
        let dot = dot_mv.scalar();

        if dot >= 0.0 {
            Ok(unsigned)
        } else {
            Ok(-unsigned)
        }
    }

    /// Check if two vectors are parallel (same or opposite direction).
    ///
    /// Two vectors are parallel if their wedge product is zero.
    ///
    /// Returns true if the vectors are parallel within tolerance.
    #[pyo3(signature = (other, tol=1e-10))]
    pub fn is_parallel(&self, other: &Multivector, tol: f64) -> PyResult<bool> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }

        // a ∧ b = 0 means parallel
        let wedge = self.outer_product(other)?;
        Ok(wedge.norm() <= tol)
    }

    /// Check if two vectors point in the same direction.
    ///
    /// Returns true if the vectors are parallel AND their dot product is positive.
    #[pyo3(signature = (other, tol=1e-10))]
    pub fn is_same_direction(&self, other: &Multivector, tol: f64) -> PyResult<bool> {
        if !self.is_parallel(other, tol)? {
            return Ok(false);
        }

        // Check dot product is positive
        let gp = self.geometric_product(other)?;
        Ok(gp.scalar() > 0.0)
    }

    /// Check if two vectors point in opposite directions.
    ///
    /// Returns true if the vectors are parallel AND their dot product is negative.
    #[pyo3(signature = (other, tol=1e-10))]
    pub fn is_antiparallel(&self, other: &Multivector, tol: f64) -> PyResult<bool> {
        if !self.is_parallel(other, tol)? {
            return Ok(false);
        }

        // Check dot product is negative
        let gp = self.geometric_product(other)?;
        Ok(gp.scalar() < 0.0)
    }

    /// Check if two vectors are orthogonal (perpendicular).
    ///
    /// Two vectors are orthogonal if their dot product (scalar product) is zero.
    ///
    /// Returns true if the vectors are orthogonal within tolerance.
    #[pyo3(signature = (other, tol=1e-10))]
    pub fn is_orthogonal(&self, other: &Multivector, tol: f64) -> PyResult<bool> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }

        // a · b = 0 means orthogonal (for vectors, this is the scalar part of gp)
        let gp = self.geometric_product(other)?;
        Ok(gp.scalar().abs() <= tol)
    }

    /// Compute the cosine of the angle between two vectors.
    ///
    /// More efficient than angle_between() when you only need the cosine.
    /// Returns a value in [-1, 1].
    ///
    /// Raises ValueError if either vector has zero norm.
    pub fn cos_angle(&self, other: &Multivector) -> PyResult<f64> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }

        let norm_a = self.norm();
        let norm_b = other.norm();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot compute angle for zero vector",
            ));
        }

        let gp = self.geometric_product(other)?;
        let dot = gp.scalar();

        Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
    }

    /// Compute the sine of the angle between two vectors.
    ///
    /// Uses the magnitude of the wedge product: |a ∧ b| = |a| |b| sin(θ)
    /// Returns a value in [0, 1] (always non-negative).
    ///
    /// Raises ValueError if either vector has zero norm.
    pub fn sin_angle(&self, other: &Multivector) -> PyResult<f64> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: left operand is Cl({}) but right operand is Cl({}); \
                both multivectors must have the same dimension",
                self.dims, other.dims
            )));
        }

        let norm_a = self.norm();
        let norm_b = other.norm();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot compute angle for zero vector",
            ));
        }

        let wedge = self.outer_product(other)?;
        let wedge_norm = wedge.norm();

        Ok((wedge_norm / (norm_a * norm_b)).clamp(0.0, 1.0))
    }

    // =========================================================================
    // REFLECTION AND PROJECTION
    // =========================================================================

    /// Reflect this multivector across the hyperplane perpendicular to n.
    ///
    /// For a vector v and unit vector n, the reflection is: -n * v * n⁻¹.
    /// This reflects v through the hyperplane orthogonal to n.
    ///
    /// If n is not a unit vector, the result is still a valid reflection
    /// (the formula handles normalization via n⁻¹).
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// e2 = Multivector.from_vector([0.0, 1.0, 0.0])
    /// reflected = e1.reflect(e2)  # Reflects e1 across the yz-plane
    /// # Result: e1 (unchanged, since e1 is parallel to the yz-plane normal)
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    pub fn reflect(&self, n: &Multivector) -> PyResult<Self> {
        if self.dims != n.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: operand is Cl({}) but mirror normal is Cl({}); \
                both must have the same dimension",
                self.dims, n.dims
            )));
        }
        // Reflection formula: -n * x * n⁻¹
        let n_inv = n.inverse()?;
        let nx = n.geometric_product(self)?;
        let nxn_inv = nx.geometric_product(&n_inv)?;
        Ok(nxn_inv.__neg__())
    }

    /// Project this multivector onto a blade B.
    ///
    /// The projection of A onto B gives the component of A that lies
    /// entirely within the subspace represented by B.
    ///
    /// Formula: (A ⌋ B) * B⁻¹ where ⌋ is left contraction.
    ///
    /// For vectors: v.project(n) gives the component of v parallel to n.
    /// For bivectors: projects onto a plane.
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([3.0, 4.0, 0.0])
    /// e1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// proj = v.project(e1)  # Component along e1
    /// # Result: 3.0*e1
    /// ```
    ///
    /// Reference: Dorst et al. ch.4 [VERIFY]
    pub fn project(&self, blade: &Multivector) -> PyResult<Self> {
        if self.dims != blade.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: operand is Cl({}) but blade is Cl({}); \
                both must have the same dimension",
                self.dims, blade.dims
            )));
        }
        // Projection formula: (A ⌋ B) * B⁻¹
        let contraction = self.left_contraction(blade)?;
        let blade_inv = blade.inverse()?;
        contraction.geometric_product(&blade_inv)
    }

    /// Compute the rejection of this multivector from a blade B.
    ///
    /// The rejection is the component perpendicular to the subspace
    /// represented by B. It satisfies: A = A.project(B) + A.reject(B).
    ///
    /// Formula: A - (A ⌋ B) * B⁻¹
    ///
    /// For vectors: v.reject(n) gives the component of v perpendicular to n.
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([3.0, 4.0, 0.0])
    /// e1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// rej = v.reject(e1)  # Component perpendicular to e1
    /// # Result: 4.0*e2
    /// ```
    ///
    /// Reference: Dorst et al. ch.4 [VERIFY]
    pub fn reject(&self, blade: &Multivector) -> PyResult<Self> {
        let proj = self.project(blade)?;
        self.__sub__(&proj)
    }

    // =========================================================================
    // VECTOR UTILITIES
    // =========================================================================

    /// Get the component of this vector parallel to another vector.
    ///
    /// This is an alias for `project(v)` with clearer naming for vector operations.
    /// For a vector a and direction v, returns the component of a along v.
    ///
    /// Formula: (a · v / |v|²) * v = (a ⌋ v) * v⁻¹
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([3.0, 4.0, 0.0])
    /// e1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// parallel = v.parallel_component(e1)  # 3.0*e1
    /// ```
    ///
    /// Reference: Standard vector projection
    pub fn parallel_component(&self, direction: &Multivector) -> PyResult<Self> {
        self.project(direction)
    }

    /// Get the component of this vector perpendicular to another vector.
    ///
    /// This is an alias for `reject(v)` with clearer naming for vector operations.
    /// For a vector a and direction v, returns the component of a perpendicular to v.
    ///
    /// Satisfies: a = a.parallel_component(v) + a.perpendicular_component(v)
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([3.0, 4.0, 0.0])
    /// e1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// perp = v.perpendicular_component(e1)  # 4.0*e2
    /// ```
    ///
    /// Reference: Standard vector rejection
    pub fn perpendicular_component(&self, direction: &Multivector) -> PyResult<Self> {
        self.reject(direction)
    }

    /// Rotate this multivector by a rotor.
    ///
    /// Computes R * self * ~R where R is the rotor.
    /// This is the inverse calling convention of `rotor.sandwich(x)` -
    /// instead of calling `rotor.apply(vector)`, you can call `vector.rotate_by(rotor)`.
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// e2 = Multivector.from_vector([0.0, 1.0, 0.0])
    /// R = Multivector.rotor_from_vectors(e1, e2)  # 90° rotation
    /// rotated = e1.rotate_by(R)  # should equal e2
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    pub fn rotate_by(&self, rotor: &Multivector) -> PyResult<Self> {
        rotor.sandwich(self)
    }

    /// Rotate this vector by a given angle in a specified plane.
    ///
    /// Creates a rotor from the plane bivector and angle, then applies it.
    /// This is a convenience method combining exp() and sandwich().
    ///
    /// The plane should be a unit bivector or will be normalized.
    /// Positive angles follow the right-hand rule around the plane's normal.
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.e1(3)
    /// e12 = Multivector.e12(3)  # xy-plane
    /// rotated = e1.rotate_in_plane(math.pi / 2, e12)  # rotate 90° in xy-plane
    /// # Result: e2
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    pub fn rotate_in_plane(&self, angle: f64, plane: &Multivector) -> PyResult<Self> {
        if self.dims != plane.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: vector is Cl({}) but plane is Cl({}); \
                both must have the same dimension",
                self.dims, plane.dims
            )));
        }

        // Normalize the plane bivector
        let plane_norm = plane.norm();
        if plane_norm == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "plane bivector has zero norm; cannot rotate in zero plane",
            ));
        }
        let unit_plane = plane.scale(1.0 / plane_norm);

        // Create rotor: R = exp(-angle/2 * B) = cos(angle/2) - sin(angle/2) * B
        let half_angle = angle / 2.0;
        let cos_half = half_angle.cos();
        let sin_half = half_angle.sin();

        let scalar_part = Multivector::from_scalar(cos_half, self.dims)?;
        let bivector_part = unit_plane.scale(-sin_half);
        let rotor = scalar_part.__add__(&bivector_part)?;

        // Apply rotation
        rotor.sandwich(self)
    }

    /// Project this vector onto a plane (bivector).
    ///
    /// Returns the component of this vector that lies within the plane
    /// represented by the bivector. The result is perpendicular to the
    /// plane's normal (dual of the bivector in 3D).
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([1.0, 2.0, 3.0])
    /// e12 = Multivector.e12(3)  # xy-plane
    /// proj = v.project_onto_plane(e12)  # [1.0, 2.0, 0.0]
    /// ```
    ///
    /// Reference: Dorst et al. ch.4 [VERIFY]
    pub fn project_onto_plane(&self, plane: &Multivector) -> PyResult<Self> {
        self.project(plane)
    }

    /// Compute the scalar triple product a · (b × c).
    ///
    /// For three vectors a, b, c in 3D, computes the scalar triple product
    /// which gives the signed volume of the parallelepiped spanned by the vectors.
    ///
    /// In geometric algebra: a · (b × c) = a ⌋ (b ∧ c) = a ∧ b ∧ c (grade-3 part)
    ///
    /// Properties:
    /// - |a · (b × c)| = volume of parallelepiped
    /// - Sign indicates orientation (right-hand rule)
    /// - Zero if vectors are coplanar
    ///
    /// Only defined for vectors in 3D (Cl(3)).
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.from_vector([1.0, 0.0, 0.0])
    /// e2 = Multivector.from_vector([0.0, 1.0, 0.0])
    /// e3 = Multivector.from_vector([0.0, 0.0, 1.0])
    /// vol = e1.triple_product(e2, e3)  # 1.0
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn triple_product(&self, b: &Multivector, c: &Multivector) -> PyResult<f64> {
        if self.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "triple product is only defined in 3D; this multivector is in Cl({})",
                self.dims
            )));
        }
        if b.dims != 3 || c.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all three vectors must be in 3D for triple product",
            ));
        }

        // a · (b × c) = a ∧ b ∧ c (the trivector coefficient)
        let wedge_bc = b.outer_product(c)?;
        let trivector = self.outer_product(&wedge_bc)?;

        // Extract the pseudoscalar (trivector) coefficient
        // In 3D, the pseudoscalar is at index 7 (e123)
        Ok(trivector.coeffs[7])
    }

    /// Return a unit (normalized) copy of this multivector.
    ///
    /// This is an alias for `normalized()` - a common shorthand in vector math.
    /// Returns a multivector with the same direction but unit norm.
    ///
    /// Raises ValueError if the norm is zero.
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([3.0, 4.0, 0.0])
    /// u = v.unit()  # [0.6, 0.8, 0.0]
    /// assert abs(u.norm() - 1.0) < 1e-10
    /// ```
    ///
    /// Reference: Standard normalization
    pub fn unit(&self) -> PyResult<Self> {
        self.normalized()
    }

    // =========================================================================
    // DUAL OPERATIONS
    // =========================================================================

    /// Compute the dual of this multivector: A* = A * I⁻¹
    ///
    /// The dual operation maps grades to their complements:
    /// - A scalar (grade 0) becomes a pseudoscalar (grade n)
    /// - A vector (grade 1) becomes a pseudovector (grade n-1)
    /// - etc.
    ///
    /// This is a fundamental operation in geometric algebra, used for:
    /// - Converting between vectors and hyperplanes
    /// - Implementing cross products in 3D (v₁ × v₂ = (v₁ ∧ v₂)*)
    /// - Meet and join operations in projective geometry
    ///
    /// Example:
    /// ```python
    /// # In 3D, dual of a bivector is a vector
    /// e12 = Multivector.from_bivector([1.0, 0.0, 0.0], dims=3)
    /// e3 = e12.dual()  # Proportional to e3
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn dual(&self) -> PyResult<Self> {
        // I = pseudoscalar, I⁻¹ = ~I / (I * ~I)
        let pseudoscalar = Multivector::pseudoscalar(self.dims)?;
        let ps_inv = pseudoscalar.inverse()?;
        self.geometric_product(&ps_inv)
    }

    /// Compute the undual of this multivector: A = A* * I
    ///
    /// This reverses the dual operation: undual(dual(A)) = A.
    ///
    /// Example:
    /// ```python
    /// v = Multivector.from_vector([1.0, 2.0, 3.0])
    /// v_dual = v.dual()
    /// v_back = v_dual.undual()
    /// assert v.approx_eq(v_back)
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn undual(&self) -> PyResult<Self> {
        let pseudoscalar = Multivector::pseudoscalar(self.dims)?;
        self.geometric_product(&pseudoscalar)
    }

    /// Compute the Hodge dual (right dual): *A = I⁻¹ * A
    ///
    /// An alternative dual convention where the pseudoscalar inverse
    /// is multiplied on the left instead of the right.
    ///
    /// In many cases `dual()` and `right_dual()` differ only by sign,
    /// but the distinction matters for oriented quantities.
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn right_dual(&self) -> PyResult<Self> {
        let pseudoscalar = Multivector::pseudoscalar(self.dims)?;
        let ps_inv = pseudoscalar.inverse()?;
        ps_inv.geometric_product(self)
    }

    // =========================================================================
    // EXPONENTIAL AND LOGARITHM
    // =========================================================================

    /// Compute the exponential of this multivector.
    ///
    /// For a pure bivector B (representing a rotation plane), the exponential
    /// produces a rotor:
    ///
    /// `exp(B) = cos(|B|) + sin(|B|) * B/|B|`
    ///
    /// where |B| is the norm of the bivector. If B has norm θ/2, then exp(B)
    /// is a rotor that performs rotation by angle θ.
    ///
    /// For scalars: exp(s) = e^s (the standard exponential).
    ///
    /// For general multivectors, this uses a Taylor series approximation.
    /// The series converges for all multivectors, but may be slow for
    /// large norms.
    ///
    /// Example:
    /// ```text
    /// import math
    /// # Create a bivector for 90-degree rotation in xy-plane
    /// e12 = Multivector.from_bivector([math.pi/4], dims=2)  # θ/2 = π/4
    /// R = e12.exp()  # Rotor for 90-degree rotation
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    pub fn exp(&self) -> PyResult<Self> {
        // Check if pure scalar
        let scalar_part = self.grade(0)?;
        let non_scalar = self.__sub__(&scalar_part)?;
        let non_scalar_norm = non_scalar.norm();

        if non_scalar_norm < 1e-14 {
            // Pure scalar: exp(s) = e^s
            let s = self.coeffs[0];
            return Multivector::from_scalar(s.exp(), self.dims);
        }

        // Check if pure bivector (most common case for rotations)
        let bivector_part = self.grade(2)?;
        let remaining = non_scalar.__sub__(&bivector_part)?;
        let remaining_norm = remaining.norm();

        if remaining_norm < 1e-14 && scalar_part.norm() < 1e-14 {
            // Pure bivector: exp(B) = cos(|B|) + sin(|B|) * B/|B|
            let b_norm = bivector_part.norm();
            if b_norm < 1e-14 {
                // Zero bivector: exp(0) = 1
                return Multivector::from_scalar(1.0, self.dims);
            }
            let cos_b = b_norm.cos();
            let sin_b = b_norm.sin();
            let b_unit = bivector_part.scale(1.0 / b_norm);
            let scaled_b = b_unit.scale(sin_b);

            let one = Multivector::from_scalar(cos_b, self.dims)?;
            return one.__add__(&scaled_b);
        }

        // General case: Taylor series
        // exp(A) = 1 + A + A²/2! + A³/3! + ...
        let mut result = Multivector::from_scalar(1.0, self.dims)?;
        let mut term = Multivector::from_scalar(1.0, self.dims)?;

        for n in 1..50 {
            term = term.geometric_product(self)?;
            term = term.scale(1.0 / n as f64);
            result = result.__add__(&term)?;

            // Check convergence
            if term.norm() < 1e-15 {
                break;
            }
        }

        Ok(result)
    }

    /// Compute the logarithm of this rotor.
    ///
    /// For a rotor R, returns a bivector B such that exp(B) = R.
    ///
    /// The rotor must be a unit rotor (R * ~R = 1). The result is a
    /// bivector with norm equal to half the rotation angle.
    ///
    /// This is useful for interpolating rotations (SLERP) and
    /// computing rotation velocities.
    ///
    /// Example:
    /// ```text
    /// # Create a rotor, then recover the bivector
    /// e1 = Multivector.from_vector([1.0, 0.0])
    /// e2 = Multivector.from_vector([0.0, 1.0])
    /// R = Multivector.rotor_from_vectors(e1, e2)  # 90-degree rotation
    /// B = R.log()  # Should have |B| = π/4
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    pub fn log(&self) -> PyResult<Self> {
        // Check if approximately unit rotor
        let norm_sq = self.norm_squared();
        if (norm_sq - 1.0).abs() > 1e-8 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "log requires a unit rotor (norm² = 1), but got norm² = {}; \
                use normalized() first if needed",
                norm_sq
            )));
        }

        // Extract scalar and bivector parts
        let scalar_part = self.grade(0)?;
        let bivector_part = self.grade(2)?;

        let s = scalar_part.coeffs[0];
        let b_norm = bivector_part.norm();

        if b_norm < 1e-14 {
            // Nearly pure scalar rotor
            if s > 0.0 {
                // R ≈ 1, log(1) = 0
                return Multivector::zero(self.dims);
            } else {
                // R ≈ -1, which is a 360° rotation
                // The bivector direction is undefined, return zero
                return Multivector::zero(self.dims);
            }
        }

        // log(R) = log(cos(θ) + sin(θ)*B̂) = θ*B̂
        // where θ = atan2(|B|, s)
        let theta = b_norm.atan2(s);
        let b_unit = bivector_part.scale(1.0 / b_norm);

        Ok(b_unit.scale(theta))
    }

    /// Compute the square root of this rotor.
    ///
    /// For a rotor R representing rotation by angle θ, returns the rotor
    /// representing rotation by θ/2.
    ///
    /// This is computed as: sqrt(R) = exp(log(R) / 2)
    ///
    /// The result satisfies: sqrt(R) * sqrt(R) = R (within numerical precision).
    ///
    /// Only valid for unit rotors.
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.from_vector([1.0, 0.0])
    /// e2 = Multivector.from_vector([0.0, 1.0])
    /// R = Multivector.rotor_from_vectors(e1, e2)  # 90° rotation
    /// half_R = R.sqrt()  # 45° rotation
    /// # half_R * half_R ≈ R
    /// ```
    ///
    /// Reference: Dorst et al. ch.10 [VERIFY]
    pub fn sqrt(&self) -> PyResult<Self> {
        let log_r = self.log()?;
        let half_log = log_r.scale(0.5);
        half_log.exp()
    }

    /// Raise this rotor to a floating-point power.
    ///
    /// For a rotor R representing rotation by angle θ, R^t represents
    /// rotation by angle t*θ.
    ///
    /// This is computed as: R^t = exp(t * log(R))
    ///
    /// Useful for:
    /// - Fractional rotations (R^0.5 = half rotation)
    /// - Smooth interpolation (combined with slerp)
    /// - Animation curves
    ///
    /// Only valid for unit rotors.
    ///
    /// Example:
    /// ```python
    /// e1 = Multivector.from_vector([1.0, 0.0])
    /// e2 = Multivector.from_vector([0.0, 1.0])
    /// R = Multivector.rotor_from_vectors(e1, e2)  # 90° rotation
    /// R_quarter = R.powf(0.25)  # 22.5° rotation
    /// R_double = R.powf(2.0)  # 180° rotation
    /// ```
    ///
    /// Reference: Dorst et al. ch.10 [VERIFY]
    pub fn powf(&self, t: f64) -> PyResult<Self> {
        let log_r = self.log()?;
        let scaled = log_r.scale(t);
        scaled.exp()
    }

    /// Reflect this multivector through a plane defined by a bivector.
    ///
    /// The plane is specified by a bivector B representing the plane's orientation.
    /// For a plane with normal n, the bivector B = I·n (dual of normal) defines
    /// the reflection plane.
    ///
    /// Formula: x' = B * x * B⁻¹ (for bivector B with norm² = -1)
    ///
    /// # Arguments
    /// * `plane` - A bivector (grade-2 element) defining the reflection plane
    ///
    /// # Returns
    /// The reflected multivector
    ///
    /// # Example
    /// ```python
    /// v = Multivector.from_vector([1, 1, 0])
    /// xy_plane = Multivector.e12(3)  # The xy-plane
    /// reflected = v.reflect_in_plane(xy_plane)
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    pub fn reflect_in_plane(&self, plane: &Multivector) -> PyResult<Self> {
        if self.dims != plane.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: operand is Cl({}) but plane is Cl({}); \
                both must have the same dimension",
                self.dims, plane.dims
            )));
        }
        // Reflection through a plane defined by bivector B:
        // For a unit bivector B, the formula is: x' = B * x * B (since B^{-1} = -B and we need extra sign)
        // Vectors in the plane are unchanged, perpendicular vectors are negated
        let plane_norm = plane.norm();
        if plane_norm == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot reflect in zero plane",
            ));
        }
        let unit_plane = plane.scale(1.0 / plane_norm);
        let bx = unit_plane.geometric_product(self)?;
        bx.geometric_product(&unit_plane)
    }

    /// Apply two successive reflections across hyperplanes perpendicular to n1 and n2.
    ///
    /// Two reflections produce a rotation in the plane spanned by n1 and n2,
    /// with angle equal to twice the angle between n1 and n2.
    ///
    /// This is mathematically equivalent to applying the rotor R = n2 * n1 to self.
    ///
    /// # Arguments
    /// * `n1` - First reflection normal (vector)
    /// * `n2` - Second reflection normal (vector)
    ///
    /// # Returns
    /// The doubly-reflected multivector (equivalent to a rotation)
    ///
    /// # Example
    /// ```python
    /// v = Multivector.from_vector([1, 0, 0])
    /// n1 = Multivector.e1(3)  # First reflection across yz-plane
    /// n2 = Multivector.e2(3)  # Second reflection across xz-plane
    /// # Result is a 180° rotation in the xy-plane
    /// result = v.double_reflection(n1, n2)
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    pub fn double_reflection(&self, n1: &Multivector, n2: &Multivector) -> PyResult<Self> {
        if self.dims != n1.dims || self.dims != n2.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: operand is Cl({}) but normals are Cl({}) and Cl({}); \
                all must have the same dimension",
                self.dims, n1.dims, n2.dims
            )));
        }
        // Two reflections: first by n1, then by n2
        let first_reflection = self.reflect(n1)?;
        first_reflection.reflect(n2)
    }

    /// Check if this multivector represents a reflection (unit versor of odd grade).
    ///
    /// A reflection versor is a product of an odd number of unit vectors,
    /// where the simplest case is a single unit vector.
    ///
    /// # Returns
    /// True if this is a unit versor with odd grade structure
    ///
    /// # Example
    /// ```python
    /// n = Multivector.e1(3)  # Unit vector is a reflection
    /// assert n.is_reflection()
    ///
    /// R = Multivector.rotor_from_vectors(e1, e2)  # Rotor is NOT a reflection
    /// assert not R.is_reflection()
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    pub fn is_reflection(&self) -> bool {
        // A reflection versor has unit norm and is an odd versor
        let norm = self.norm();
        if (norm - 1.0).abs() > 1e-10 {
            return false;
        }
        // Check if it's odd: should be a product of odd number of vectors
        // An odd versor has non-zero odd-grade components and zero even-grade components
        self.is_odd(1e-10)
    }

    /// Create a rotor from two reflection vectors.
    ///
    /// The product of two unit vectors n2 * n1 creates a rotor that performs
    /// the same transformation as two successive reflections.
    /// The rotation angle is twice the angle between n1 and n2.
    ///
    /// # Arguments
    /// * `n1` - First reflection normal (vector)
    /// * `n2` - Second reflection normal (vector)
    ///
    /// # Returns
    /// A rotor R = n2 * n1
    ///
    /// # Example
    /// ```python
    /// n1 = Multivector.e1(3)
    /// n2 = Multivector.e2(3)
    /// R = Multivector.rotor_from_reflections(n1, n2)
    /// # R represents a 180° rotation in the xy-plane
    /// ```
    ///
    /// Reference: Dorst et al. ch.7 [VERIFY]
    #[staticmethod]
    pub fn rotor_from_reflections(n1: &Multivector, n2: &Multivector) -> PyResult<Self> {
        if n1.dims != n2.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: n1 is Cl({}) but n2 is Cl({}); \
                both must have the same dimension",
                n1.dims, n2.dims
            )));
        }
        // Rotor from two reflections: R = n2 * n1
        // Normalize both vectors first
        let n1_unit = n1.normalized()?;
        let n2_unit = n2.normalized()?;
        n2_unit.geometric_product(&n1_unit)
    }

    /// Calculate the signed area of the parallelogram spanned by two vectors.
    ///
    /// This is the norm of the wedge product (outer product) of the two vectors.
    /// For 2D vectors, this gives the signed area directly.
    /// For 3D vectors, this gives the area of the parallelogram in the plane they span.
    ///
    /// # Arguments
    /// * `other` - The second vector
    ///
    /// # Returns
    /// The area of the parallelogram (always non-negative)
    ///
    /// # Example
    /// ```python
    /// a = Multivector.from_vector([1, 0, 0])
    /// b = Multivector.from_vector([0, 1, 0])
    /// area = a.area(b)  # Returns 1.0 (unit square)
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn area(&self, other: &Multivector) -> PyResult<f64> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: self is Cl({}) but other is Cl({})",
                self.dims, other.dims
            )));
        }
        let wedge = self.wedge(other)?;
        Ok(wedge.norm())
    }

    /// Calculate the signed volume of the parallelepiped spanned by three vectors.
    ///
    /// This is the norm of the triple wedge product a ∧ b ∧ c.
    /// In 3D, this equals |a · (b × c)|, the absolute scalar triple product.
    ///
    /// # Arguments
    /// * `b` - The second vector
    /// * `c` - The third vector
    ///
    /// # Returns
    /// The volume of the parallelepiped (always non-negative)
    ///
    /// # Example
    /// ```python
    /// e1 = Multivector.e1(3)
    /// e2 = Multivector.e2(3)
    /// e3 = Multivector.e3(3)
    /// vol = e1.volume(e2, e3)  # Returns 1.0 (unit cube)
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn volume(&self, b: &Multivector, c: &Multivector) -> PyResult<f64> {
        if self.dims != b.dims || self.dims != c.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: vectors have dimensions {}, {}, {}",
                self.dims, b.dims, c.dims
            )));
        }
        let ab = self.wedge(b)?;
        let abc = ab.wedge(c)?;
        Ok(abc.norm())
    }

    /// Calculate the angle between this vector and a plane (bivector).
    ///
    /// The angle to a plane is the complement of the angle to the plane's normal.
    /// A vector in the plane has angle 0, perpendicular has angle π/2.
    ///
    /// # Arguments
    /// * `plane` - A bivector representing the plane
    ///
    /// # Returns
    /// The angle in radians (0 to π/2)
    ///
    /// # Example
    /// ```python
    /// v = Multivector.from_vector([1, 1, 1])  # Diagonal vector
    /// xy_plane = Multivector.e12(3)
    /// angle = v.angle_to_plane(xy_plane)  # Angle from xy-plane
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn angle_to_plane(&self, plane: &Multivector) -> PyResult<f64> {
        if self.dims != plane.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: vector is Cl({}) but plane is Cl({})",
                self.dims, plane.dims
            )));
        }
        // The angle to the plane is π/2 minus the angle to the normal
        // We can compute this using the dual of the plane
        let plane_dual = plane.dual()?;
        let angle_to_normal = self.angle_between(&plane_dual)?;
        // Angle to plane is complement of angle to normal
        Ok((std::f64::consts::FRAC_PI_2 - angle_to_normal).abs())
    }

    /// Calculate the perpendicular distance from this point to a plane.
    ///
    /// The plane is defined by a bivector (its orientation) and a point on the plane.
    ///
    /// # Arguments
    /// * `plane` - A bivector representing the plane's orientation
    /// * `point_on_plane` - Any point (vector) that lies on the plane
    ///
    /// # Returns
    /// The perpendicular distance (always non-negative)
    ///
    /// # Example
    /// ```python
    /// point = Multivector.from_vector([0, 0, 5])
    /// xy_plane = Multivector.e12(3)
    /// origin = Multivector.from_vector([0, 0, 0])
    /// dist = point.distance_to_plane(xy_plane, origin)  # Returns 5.0
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn distance_to_plane(&self, plane: &Multivector, point_on_plane: &Multivector) -> PyResult<f64> {
        if self.dims != plane.dims || self.dims != point_on_plane.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: point is Cl({}), plane is Cl({}), point_on_plane is Cl({})",
                self.dims, plane.dims, point_on_plane.dims
            )));
        }
        // Get the normal to the plane (dual of bivector)
        let normal = plane.dual()?.normalized()?;
        // Vector from point_on_plane to this point
        let displacement = self.__sub__(point_on_plane)?;
        // Distance is absolute value of dot product with normal
        let dist = displacement.scalar_product(&normal)?;
        Ok(dist.scalar().abs())
    }

    /// Check if this vector lies in a plane (bivector).
    ///
    /// A vector lies in a plane if it is perpendicular to the plane's normal,
    /// or equivalently if the wedge product with the plane is zero.
    ///
    /// # Arguments
    /// * `plane` - A bivector representing the plane
    /// * `tol` - Tolerance for the check (default 1e-10)
    ///
    /// # Returns
    /// True if the vector lies in the plane
    ///
    /// # Example
    /// ```python
    /// v = Multivector.from_vector([3, 4, 0])
    /// xy_plane = Multivector.e12(3)
    /// assert v.lies_in_plane(xy_plane)  # True, v is in xy-plane
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    #[pyo3(signature = (plane, tol=1e-10))]
    pub fn lies_in_plane(&self, plane: &Multivector, tol: f64) -> PyResult<bool> {
        if self.dims != plane.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: vector is Cl({}) but plane is Cl({})",
                self.dims, plane.dims
            )));
        }
        // A vector lies in a plane if their wedge product is zero
        let wedge = self.wedge(plane)?;
        Ok(wedge.norm() < tol)
    }

    /// Check if this vector is perpendicular to a plane (parallel to its normal).
    ///
    /// # Arguments
    /// * `plane` - A bivector representing the plane
    /// * `tol` - Tolerance for the check (default 1e-10)
    ///
    /// # Returns
    /// True if the vector is perpendicular to the plane
    ///
    /// # Example
    /// ```python
    /// e3 = Multivector.e3(3)
    /// xy_plane = Multivector.e12(3)
    /// assert e3.is_perpendicular_to_plane(xy_plane)  # True
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    #[pyo3(signature = (plane, tol=1e-10))]
    pub fn is_perpendicular_to_plane(&self, plane: &Multivector, tol: f64) -> PyResult<bool> {
        if self.dims != plane.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: vector is Cl({}) but plane is Cl({})",
                self.dims, plane.dims
            )));
        }
        // A vector is perpendicular to a plane if its projection onto the plane is zero
        let proj = self.project(plane)?;
        Ok(proj.norm() < tol)
    }

    /// Extract the highest-grade component of the multivector.
    ///
    /// Returns the multivector containing only the coefficients of the highest
    /// non-zero grade.
    ///
    /// # Returns
    /// The highest-grade part of the multivector, or zero if all coefficients are zero
    ///
    /// # Example
    /// ```python
    /// mv = Multivector.from_scalar(1.0, 3) + Multivector.e1(3) + Multivector.e12(3)
    /// highest = mv.max_grade_part()  # Returns the bivector part (e12)
    /// ```
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn max_grade_part(&self) -> PyResult<Self> {
        match self.max_grade() {
            Some(k) => self.grade(k),
            None => Ok(Self::zero(self.dims)?),
        }
    }

    /// Extract the lowest-grade component of the multivector.
    ///
    /// Returns the multivector containing only the coefficients of the lowest
    /// non-zero grade.
    ///
    /// # Returns
    /// The lowest-grade part of the multivector, or zero if all coefficients are zero
    ///
    /// # Example
    /// ```python
    /// mv = Multivector.from_scalar(1.0, 3) + Multivector.e1(3) + Multivector.e12(3)
    /// lowest = mv.min_grade_part()  # Returns the scalar part
    /// ```
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn min_grade_part(&self) -> PyResult<Self> {
        match self.min_grade() {
            Some(k) => self.grade(k),
            None => Ok(Self::zero(self.dims)?),
        }
    }

    /// Split the multivector into its even and odd parts.
    ///
    /// Returns a tuple of (even_part, odd_part).
    ///
    /// # Returns
    /// A tuple (even, odd) where even contains grades 0, 2, 4, ... and odd contains grades 1, 3, 5, ...
    ///
    /// # Example
    /// ```python
    /// rotor = Multivector.from_axis_angle(e3, 0.5)
    /// even, odd = rotor.split_even_odd()  # rotor is even, odd should be zero
    /// ```
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn split_even_odd(&self) -> (Self, Self) {
        (self.even(), self.odd())
    }

    /// Compute the square of this multivector.
    ///
    /// For blades in Euclidean space, this always returns a scalar.
    /// For general multivectors, this returns a full multivector.
    ///
    /// # Returns
    /// The geometric product of this multivector with itself
    ///
    /// # Example
    /// ```python
    /// e1 = Multivector.e1(3)
    /// assert e1.blade_square() == 1.0  # Unit vector squares to 1
    ///
    /// e12 = Multivector.e12(3)
    /// assert e12.blade_square() == -1.0  # Bivector squares to -1
    /// ```
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    pub fn blade_square(&self) -> PyResult<f64> {
        let sq = self.geometric_product(self)?;
        // For a blade, the result should be a scalar
        Ok(sq.scalar())
    }

    /// Check if this blade is null (squares to zero).
    ///
    /// In Euclidean space, no non-zero blade is null. This is useful for
    /// checking degenerate cases or for mixed-signature algebras.
    ///
    /// # Arguments
    /// * `tol` - Tolerance for the zero check (default 1e-10)
    ///
    /// # Returns
    /// True if the blade squares to approximately zero
    ///
    /// # Example
    /// ```python
    /// v = Multivector.from_vector([1, 0, 0])
    /// assert not v.is_null()  # Non-zero vectors are not null in Euclidean space
    /// ```
    ///
    /// Reference: Dorst et al. ch.2 [VERIFY]
    #[pyo3(signature = (tol=1e-10))]
    pub fn is_null(&self, tol: f64) -> PyResult<bool> {
        let sq = self.blade_square()?;
        Ok(sq.abs() < tol)
    }

    /// Compute the complement of this multivector.
    ///
    /// The complement is the dual multiplied by the pseudoscalar, effectively
    /// mapping grade-k elements to grade-(n-k) elements.
    ///
    /// For a multivector A in Cl(n), complement(A) = A * I where I is the pseudoscalar.
    ///
    /// # Returns
    /// The complement of the multivector
    ///
    /// # Example
    /// ```python
    /// e1 = Multivector.e1(3)
    /// comp = e1.complement()  # Returns e2 ∧ e3 (a bivector)
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn complement(&self) -> PyResult<Self> {
        let pseudoscalar = Self::pseudoscalar(self.dims)?;
        self.geometric_product(&pseudoscalar)
    }

    /// Compute the reverse complement (undual operation).
    ///
    /// The reverse complement maps a multivector by multiplying by the inverse pseudoscalar.
    ///
    /// # Returns
    /// The reverse complement of the multivector
    ///
    /// # Example
    /// ```python
    /// e23 = Multivector.e23(3)
    /// rev_comp = e23.reverse_complement()  # Returns e1
    /// ```
    ///
    /// Reference: Dorst et al. ch.3 [VERIFY]
    pub fn reverse_complement(&self) -> PyResult<Self> {
        let pseudoscalar = Self::pseudoscalar(self.dims)?;
        let pseudoscalar_inv = pseudoscalar.inverse()?;
        self.geometric_product(&pseudoscalar_inv)
    }

    /// Get the number of distinct non-zero grades in this multivector.
    ///
    /// # Returns
    /// The count of grades with non-zero coefficients
    ///
    /// # Example
    /// ```python
    /// mv = Multivector.from_scalar(1.0, 3) + Multivector.e1(3)
    /// assert mv.grade_count() == 2  # Has scalar and vector parts
    /// ```
    pub fn num_grades(&self) -> usize {
        self.grades().len()
    }

    /// Check if this multivector has exactly one non-zero grade.
    ///
    /// # Returns
    /// True if the multivector is a pure k-vector (blade or sum of k-blades)
    ///
    /// # Example
    /// ```python
    /// v = Multivector.from_vector([1, 2, 3])
    /// assert v.is_homogeneous()  # True, only grade 1
    ///
    /// rotor = Multivector.rotor_from_vectors(e1, e2)
    /// assert not rotor.is_homogeneous()  # False, has scalar and bivector parts
    /// ```
    pub fn is_homogeneous(&self) -> bool {
        self.grades().len() <= 1
    }

    /// Get the grade of this multivector if it has exactly one non-zero grade.
    ///
    /// # Returns
    /// Some(k) if the multivector has only grade-k components, None otherwise
    ///
    /// # Example
    /// ```python
    /// v = Multivector.from_vector([1, 2, 3])
    /// assert v.homogeneous_grade() == 1
    ///
    /// rotor = Multivector.rotor_from_vectors(e1, e2)
    /// assert rotor.homogeneous_grade() is None
    /// ```
    pub fn homogeneous_grade(&self) -> Option<usize> {
        let grades = self.grades();
        if grades.len() == 1 {
            Some(grades[0])
        } else {
            None
        }
    }

    /// Compose this rotor with another rotor.
    ///
    /// Rotor composition applies rotations in sequence: first self, then other.
    /// The result R_combined = other * self applies self's rotation first.
    ///
    /// # Arguments
    /// * `other` - The rotor to compose with (applied second)
    ///
    /// # Returns
    /// The composed rotor (normalized)
    ///
    /// # Example
    /// ```python
    /// R1 = Multivector.from_axis_angle(e3, pi/4)  # 45° around z
    /// R2 = Multivector.from_axis_angle(e3, pi/4)  # Another 45°
    /// R_combined = R1.compose_with(R2)  # 90° around z
    /// ```
    ///
    /// Reference: Dorst et al. ch.10 [VERIFY]
    pub fn compose_with(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: self is Cl({}) but other is Cl({})",
                self.dims, other.dims
            )));
        }
        // Compose: R_combined = other * self (apply self first, then other)
        let composed = other.geometric_product(self)?;
        composed.normalized()
    }

    /// Compute the inverse of this rotor.
    ///
    /// For a unit rotor, the inverse is the reverse. This method normalizes
    /// the result to handle non-unit rotors.
    ///
    /// # Returns
    /// The inverse rotor R⁻¹ such that R * R⁻¹ = 1
    ///
    /// # Example
    /// ```python
    /// R = Multivector.from_axis_angle(e3, pi/3)
    /// R_inv = R.inverse_rotor()
    /// identity = R.compose_with(R_inv)  # Should be 1
    /// ```
    ///
    /// Reference: Dorst et al. ch.10 [VERIFY]
    pub fn inverse_rotor(&self) -> PyResult<Self> {
        // For a unit rotor, inverse = reverse
        // For non-unit, we need to divide by norm squared
        let rev = self.reverse();
        let norm_sq = self.norm_squared();
        if norm_sq == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot invert zero rotor",
            ));
        }
        Ok(rev.scale(1.0 / norm_sq))
    }

    /// Compute the rotor that transforms this rotor into another.
    ///
    /// Returns R such that other = R * self, i.e., R = other * self⁻¹.
    ///
    /// # Arguments
    /// * `other` - The target rotor
    ///
    /// # Returns
    /// The difference rotor
    ///
    /// # Example
    /// ```python
    /// R1 = Multivector.from_axis_angle(e3, pi/4)
    /// R2 = Multivector.from_axis_angle(e3, 3*pi/4)
    /// R_diff = R1.rotor_difference(R2)  # 90° (pi/2) rotation
    /// ```
    ///
    /// Reference: Dorst et al. ch.10 [VERIFY]
    pub fn rotor_difference(&self, other: &Multivector) -> PyResult<Self> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: self is Cl({}) but other is Cl({})",
                self.dims, other.dims
            )));
        }
        let self_inv = self.inverse_rotor()?;
        let diff = other.geometric_product(&self_inv)?;
        diff.normalized()
    }

    /// Create a rotor that rotates one plane (bivector) to another.
    ///
    /// Given two bivectors representing planes, compute the rotor that
    /// transforms the first plane into the second.
    ///
    /// # Arguments
    /// * `plane1` - The source plane (bivector)
    /// * `plane2` - The target plane (bivector)
    ///
    /// # Returns
    /// The rotor that transforms plane1 to plane2
    ///
    /// # Example
    /// ```python
    /// xy_plane = Multivector.e12(3)
    /// xz_plane = Multivector.e31(3)  # Note: e31 = -e13
    /// R = Multivector.rotor_between_planes(xy_plane, xz_plane)
    /// ```
    ///
    /// Reference: Dorst et al. ch.10 [VERIFY]
    #[staticmethod]
    pub fn rotor_between_planes(plane1: &Multivector, plane2: &Multivector) -> PyResult<Self> {
        if plane1.dims != plane2.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: plane1 is Cl({}) but plane2 is Cl({})",
                plane1.dims, plane2.dims
            )));
        }
        // Get the duals (normals) of the planes
        let n1 = plane1.dual()?.normalized()?;
        let n2 = plane2.dual()?.normalized()?;
        // Create rotor from the two normals: R = (1 + n2*n1) / |1 + n2*n1|
        let one = Multivector::from_scalar(1.0, plane1.dims)?;
        let n2n1 = n2.geometric_product(&n1)?;
        let r_unnorm = one.__add__(&n2n1)?;
        r_unnorm.normalized()
    }

    /// Normalize this rotor to unit norm.
    ///
    /// Rotors should have unit norm for correct rotation behavior.
    /// This method returns a normalized copy.
    ///
    /// # Returns
    /// A unit rotor
    ///
    /// # Example
    /// ```python
    /// R = some_rotor
    /// R_unit = R.normalize_rotor()
    /// assert abs(R_unit.norm() - 1.0) < 1e-10
    /// ```
    pub fn normalize_rotor(&self) -> PyResult<Self> {
        self.normalized()
    }

    /// Check if two rotors represent the same rotation.
    ///
    /// Rotors R and -R represent the same rotation, so this method
    /// checks if self ≈ ±other.
    ///
    /// # Arguments
    /// * `other` - The rotor to compare with
    /// * `tol` - Tolerance for comparison (default 1e-10)
    ///
    /// # Returns
    /// True if the rotors represent the same rotation
    ///
    /// # Example
    /// ```python
    /// R1 = Multivector.from_axis_angle(e3, pi/2)
    /// R2 = R1.__neg__()  # -R1
    /// assert R1.same_rotation(R2)  # True, same rotation
    /// ```
    #[pyo3(signature = (other, tol=1e-10))]
    pub fn same_rotation(&self, other: &Multivector, tol: f64) -> PyResult<bool> {
        if self.dims != other.dims {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimension mismatch: self is Cl({}) but other is Cl({})",
                self.dims, other.dims
            )));
        }
        // Check if self ≈ other or self ≈ -other
        let diff1 = self.__sub__(other)?;
        let diff2 = self.__add__(other)?;
        Ok(diff1.norm() < tol || diff2.norm() < tol)
    }

    /// Decompose a 3D rotor into axis and angle.
    ///
    /// Returns the rotation axis (unit vector) and angle in radians.
    /// This is the inverse of from_axis_angle().
    ///
    /// # Returns
    /// A tuple (axis, angle) where axis is a unit vector and angle is in radians
    ///
    /// # Example
    /// ```python
    /// R = Multivector.from_axis_angle(e3, pi/3)
    /// axis, angle = R.decompose_rotor()
    /// # axis ≈ e3, angle ≈ pi/3
    /// ```
    ///
    /// Reference: Dorst et al. ch.10 [VERIFY]
    pub fn decompose_rotor(&self) -> PyResult<(Self, f64)> {
        if self.dims != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "decompose_rotor is only implemented for 3D rotors",
            ));
        }
        let angle = self.rotation_angle(1e-10)?;
        let plane = self.rotation_plane(1e-10)?;
        // The axis is the dual of the rotation plane
        let axis = plane.dual()?.normalized()?;
        Ok((axis, angle))
    }

    /// Get the rotation angle in degrees.
    ///
    /// Convenience method that converts the rotation angle to degrees.
    ///
    /// # Returns
    /// The rotation angle in degrees
    ///
    /// # Example
    /// ```python
    /// R = Multivector.from_axis_angle(e3, pi/2)
    /// assert abs(R.rotation_angle_degrees() - 90.0) < 1e-10
    /// ```
    pub fn rotation_angle_degrees(&self) -> PyResult<f64> {
        let radians = self.rotation_angle(1e-10)?;
        Ok(radians.to_degrees())
    }
}

// Rust-only methods (not exposed to Python)
impl Multivector {
    /// Return a normalized copy of this multivector (unit norm).
    ///
    /// Returns None if the norm is zero (cannot normalize).
    /// For Python users, use `normalized()` which raises an exception instead.
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

    /// Convert a blade index to a human-readable name.
    ///
    /// Index 0 = "1" (scalar), Index 1 = "e1", Index 2 = "e2",
    /// Index 3 = "e12", etc.
    fn blade_name(index: usize) -> String {
        if index == 0 {
            return "1".to_string();
        }
        let mut name = String::from("e");
        let mut idx = index;
        let mut basis = 1;
        while idx > 0 {
            if idx & 1 == 1 {
                name.push_str(&basis.to_string());
            }
            idx >>= 1;
            basis += 1;
        }
        name
    }

    /// Format a float according to a Python-style format spec.
    ///
    /// Supports: .Nf, .Ne, .Ng, +.Nf, etc.
    fn format_float(value: f64, spec: &str) -> PyResult<String> {
        // Parse basic format specs: [+].[precision][type]
        let mut chars = spec.chars().peekable();
        let mut show_sign = false;
        let mut precision: Option<usize> = None;
        let mut fmt_type = 'g'; // default

        // Check for + sign
        if chars.peek() == Some(&'+') {
            show_sign = true;
            chars.next();
        }

        // Check for .precision
        if chars.peek() == Some(&'.') {
            chars.next();
            let mut prec_str = String::new();
            while chars.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                prec_str.push(chars.next().unwrap());
            }
            if !prec_str.is_empty() {
                precision = Some(prec_str.parse().unwrap());
            }
        }

        // Check for type
        if let Some(c) = chars.next() {
            fmt_type = c;
        }

        // Validate no remaining characters
        if chars.next().is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid format spec: {}",
                spec
            )));
        }

        let prec = precision.unwrap_or(6);
        let formatted = match fmt_type {
            'f' | 'F' => format!("{:.prec$}", value, prec = prec),
            'e' => format!("{:.prec$e}", value, prec = prec),
            'E' => format!("{:.prec$E}", value, prec = prec),
            'g' | 'G' => {
                // General format: use shortest of f or e
                let f_fmt = format!("{:.prec$}", value, prec = prec);
                let e_fmt = format!("{:.prec$e}", value, prec = prec);
                if f_fmt.len() <= e_fmt.len() {
                    f_fmt
                } else {
                    e_fmt
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown format type: {}",
                    fmt_type
                )));
            }
        };

        if show_sign && value >= 0.0 {
            Ok(format!("+{}", formatted))
        } else {
            Ok(formatted)
        }
    }
}
