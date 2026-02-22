# Changelog

All notable changes to LargeCrimsonCanine will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Core**
- Core `Multivector` type with coefficient array storage
- Geometric product (`*` operator, `gp()`)
- Outer (wedge) product (`^` operator, `wedge()`, `join()`)
- Left contraction / inner product (`|` operator, `lc()`, `inner()`)
- Right contraction (`rc()`, `right_contraction()`)
- Scalar product (`scalar_product()`)
- Commutator (`commutator()`, `x()`)
- Anticommutator (`anticommutator()`)
- Regressive product / meet (`regressive()`, `meet()`, `vee()`)
- Grade projection (`grade(k)`)
- Scalar extraction (`scalar()`)
- Blade grade and product primitives in `algebra.rs`

**Constructors**
- `Multivector.zero(dims)` — zero multivector
- `Multivector.from_scalar(value, dims)` — scalar multivector
- `Multivector.from_list(coeffs)` — multivector from coefficient list
- `Multivector.from_vector(coords)` — vector from coordinates
- `Multivector.from_bivector(components, dims)` — bivector from components
- `Multivector.basis(index, dims)` — single basis vector
- `Multivector.e1(dims)` — unit vector e1 (shorthand for basis(1, dims))
- `Multivector.e2(dims)` — unit vector e2 (shorthand for basis(2, dims))
- `Multivector.e3(dims)` — unit vector e3 (shorthand for basis(3, dims))
- `Multivector.e4(dims)` — unit vector e4 (shorthand for basis(4, dims))
- `Multivector.e12(dims)` — unit bivector e1∧e2
- `Multivector.e23(dims)` — unit bivector e2∧e3
- `Multivector.e31(dims)` — unit bivector e3∧e1
- `Multivector.e123()` — unit pseudoscalar in 3D (e1∧e2∧e3)
- `Multivector.pseudoscalar(dims)` — unit pseudoscalar
- `Multivector.random(dims)` — random multivector with coefficients in [0, 1)
- `Multivector.random_vector(dims)` — random unit vector
- `Multivector.random_rotor(dims)` — random rotor (rotation)
- `Multivector.from_axis_angle(axis, angle)` — 3D rotor from axis and angle
- `Multivector.from_quaternion(w, x, y, z)` — 3D rotor from quaternion
- `to_quaternion()` — convert 3D rotor to quaternion (w, x, y, z)
- `Multivector.from_rotation_matrix(matrix)` — 3D rotor from 3x3 rotation matrix
- `to_rotation_matrix()` — convert 3D rotor to 3x3 rotation matrix
- `Multivector.from_euler_angles(yaw, pitch, roll)` — 3D rotor from Euler angles (ZYX)
- `to_euler_angles()` — convert 3D rotor to Euler angles (yaw, pitch, roll)

**Arithmetic**
- Addition, subtraction, negation (`+`, `-`, unary `-`)
- Scalar multiplication and division (`*`, `/`)
- Equality and approximate equality (`==`, `approx_eq()`)

**Norms and Inverses**
- `reverse()`, `tilde()` — reverse operation
- `norm()`, `norm_squared()` — magnitude
- `grade_norm(k)` — norm of grade-k part only
- `normalized()` — unit multivector
- `inverse()` — multiplicative inverse for blades/versors
- `div()` — multivector division (A * B⁻¹)

**Conjugations**
- `grade_involution()`, `involute()` — negate odd grades
- `clifford_conjugate()`, `conjugate()` — reverse + grade involution
- `even()`, `odd()` — extract even/odd grade parts

**Rotors**
- `Multivector.rotor_from_vectors(a, b)` — rotor rotating a to b
- `sandwich(x)`, `apply(x)` — rotor application (R * x * ~R)
- `is_rotor()` — check if unit rotor
- `exp()` — exponential (bivector → rotor)
- `log()` — logarithm (rotor → bivector)
- `sqrt()` — square root of rotor (half the rotation angle)
- `powf(t)` — raise rotor to float power (fractional rotations)
- `slerp(other, t)` — spherical linear interpolation between rotors
- `rotation_angle()` — extract rotation angle from rotor
- `rotation_plane()` — extract rotation plane (bivector) from rotor
- `axis_angle()` — decompose 3D rotor into axis and angle
- `cross(other)` — 3D cross product (a × b = (a ∧ b)*)

**Duality**
- `dual()` — left dual (A * I⁻¹)
- `undual()` — reverse dual (A * I)
- `right_dual()` — right dual (I⁻¹ * A)

**Geometric Operations**
- `reflect(n)` — reflection across hyperplane perpendicular to n
- `project(B)` — projection onto blade B
- `reject(B)` — rejection from blade B
- `parallel_component(v)` — component parallel to vector (alias for project)
- `perpendicular_component(v)` — component perpendicular to vector (alias for reject)
- `rotate_by(rotor)` — apply rotor rotation to this multivector
- `rotate_in_plane(angle, plane)` — rotate by angle in a given bivector plane
- `project_onto_plane(plane)` — project onto a plane (bivector)
- `triple_product(b, c)` — scalar triple product a·(b×c) in 3D
- `unit()` — return unit (normalized) multivector (alias for normalized)

**Utilities**
- `is_blade()` — check if multivector is a simple k-vector
- `is_versor()` — check if multivector is a product of vectors
- `grades()` — list of grades with non-zero components
- `max_grade()`, `min_grade()` — highest/lowest non-zero grade
- `has_grade(k)` — check if grade k has non-zero components
- `pure_grade()` — return grade if single-grade, None otherwise
- `is_scalar()`, `is_vector()`, `is_bivector()`, `is_trivector()` — grade checks
- `vector_part()` — extract grade-1 component
- `bivector_part()` — extract grade-2 component
- `trivector_part()` — extract grade-3 component
- `to_vector_coords()` — extract vector coefficients as list [x, y, z, ...]
- `to_bivector_coords()` — extract bivector coefficients as list
- `to_trivector_coords()` — extract trivector coefficients as list
- `blades()` — decompose into (index, coefficient, grade) tuples
- `grade_parts()` — dict mapping grade to multivector for each grade
- `sorted_blades()` — blades sorted by coefficient magnitude (descending)
- `dominant_grade()` — grade with largest total coefficient magnitude
- `dominant_blade()` — blade index with largest absolute coefficient
- `norm_lt(other)`, `norm_gt(other)` — compare multivectors by norm
- `is_unit()` — check if norm is 1
- `dot()` — alias for scalar_product
- `lerp(other, t)` — linear interpolation
- `nlerp(other, t)` — normalized linear interpolation (lerp + normalize)
- `magnitude()` — alias for norm
- `squared()` — geometric product with self (A * A)
- `is_even()`, `is_odd()` — check if only even/odd grades
- `grade_count()` — number of distinct grades
- `negate_grade(k)` — negate coefficients of grade k
- `clear_grade(k)` — zero out coefficients of grade k
- `scale_grade(k, factor)` — multiply grade k by factor
- `add_scalar(value)` — add value to scalar part
- `with_scalar(value)` — set scalar part to value
- `is_pseudoscalar()` — check if highest grade only
- `is_zero()` — check if all coefficients are zero
- `components()` — list of (index, coefficient) pairs
- `blade_indices()` — list of non-zero blade indices
- `coefficient(index)` — get coefficient by blade index
- `sum_coefficients()` — sum of all coefficients
- `max_coefficient()` — maximum coefficient value
- `min_coefficient()` — minimum coefficient value
- `mean_coefficient()` — mean (average) of all coefficients
- `variance_coefficient()` — variance of all coefficients
- `std_coefficient()` — standard deviation of all coefficients
- `median_coefficient()` — median of all coefficients
- `range_coefficient()` — max - min of coefficients
- `l1_norm()` — sum of absolute values (taxicab norm)
- `linf_norm()` — maximum absolute value (Chebyshev norm)
- `nonzero_count()` — count of non-zero coefficients
- `sparsity()` — fraction of zero coefficients (0.0 to 1.0)
- `density()` — fraction of non-zero coefficients (1.0 - sparsity)
- `distance(other)` — Euclidean distance between two vectors
- `midpoint(other)` — midpoint between two vectors
- `round_coefficients(ndigits)` — round coefficients to n decimal places
- `clean(epsilon)` — set near-zero coefficients to zero
- `abs_coefficients()` — absolute value of all coefficients
- `clamp_coefficients(min, max)` — clamp coefficients to range
- `filter_grades(grades)` — keep only specified grades
- `threshold(min_abs)` — zero out coefficients below threshold
- `sign()` — return sign of each coefficient (-1, 0, or 1)
- `positive_part()` — keep only positive coefficients
- `negative_part()` — keep only negative coefficients

**Serialization**
- `to_dict()` — serialize to dictionary (coeffs, dims)
- `Multivector.from_dict(d)` — deserialize from dictionary

**Geometric Predicates**
- `angle_between(other)` — angle between vectors in radians
- `signed_angle(other, normal)` — signed angle with direction from normal
- `cos_angle(other)`, `sin_angle(other)` — trig functions of angle
- `is_parallel(other)` — check if vectors are parallel
- `is_same_direction(other)` — check if vectors point same way
- `is_antiparallel(other)` — check if vectors point opposite ways
- `is_orthogonal(other)` — check if vectors are perpendicular

**Python Protocols**
- `**` operator (`__pow__`) — integer exponentiation with binary algorithm
- `abs()` function (`__abs__`) — returns norm
- `bool()` function (`__bool__`) — non-zero check
- `+` unary operator (`__pos__`) — returns copy
- `~` operator (`__invert__`) — returns reverse
- `iter()` function (`__iter__`) — iterate over coefficients
- `copy()` — explicit copy method
- `copy.copy()` support (`__copy__`) — shallow copy via copy module
- `copy.deepcopy()` support (`__deepcopy__`) — deep copy via copy module
- `hash()` function (`__hash__`) — hashable, usable in sets and dict keys
- `pickle` support (`__reduce__`, `__getstate__`) — serialization
- `==` operator via `PartialEq` trait — exact equality (frozen pyclass)
- `round()` function (`__round__`) — round coefficients to n decimal places
- `math.floor()` (`__floor__`) — floor all coefficients
- `math.ceil()` (`__ceil__`) — ceil all coefficients
- `math.trunc()` (`__trunc__`) — truncate all coefficients
- `float()` function (`__float__`) — convert scalar multivector to float
- `int()` function (`__int__`) — convert scalar multivector to int
- `format()` function (`__format__`) — custom formatting with specs (.2f, .3e, +.4g)
- `coefficients()` — return all coefficients as list
- `set_coefficient(index, value)` — return copy with coefficient changed
- `dimension` / `dims` — base vector space dimension (property)
- `n_coeffs` — number of coefficients, equals 2^dimension (property)

**Infrastructure**
- Python test suite (665 tests)
- Rust test suite (7 tests)
- PyO3 bindings
- GitHub Actions CI/CD with path filtering and caching
- PyPI release workflow with trusted publishing
- Security scanning (cargo audit, dependency review)
- ARCHITECTURE.md documenting design decisions
- CONTRIBUTING.md with comment conventions, citation policy, and accessibility guidelines

### Known Limitations
- Euclidean metric only (Cl(p,q,r) support planned for v0.2)
- Runtime dimension checking (compile-time const generics planned for v1.0)

