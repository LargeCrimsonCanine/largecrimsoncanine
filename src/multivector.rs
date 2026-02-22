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

    /// Return the number of coefficients.
    pub fn __len__(&self) -> usize {
        self.coeffs.len()
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

    /// Multiply all coefficients by a scalar.
    pub fn scale(&self, scalar: f64) -> Self {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|c| c * scalar).collect();
        Multivector {
            coeffs,
            dims: self.dims,
        }
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

    /// Return the norm (magnitude) via abs().
    pub fn __abs__(&self) -> f64 {
        self.norm()
    }

    /// Create a copy of this multivector.
    pub fn copy(&self) -> Self {
        self.clone()
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
}
