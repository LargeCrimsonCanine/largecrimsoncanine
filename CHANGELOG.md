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
- `fat_dot(other)` — Hestenes inner product (grade-lowering)
- `symmetric_product(other)` — symmetrized geometric product (a*b + b*a)/2
- `commutes_with(other)` — check if elements commute under geometric product
- `anticommutes_with(other)` — check if elements anticommute
- `hat()` — alias for grade_involution
- `dagger()` — alias for reverse
- `bar()` — alias for clifford_conjugate
- `is_spinor()` — check if unit even versor (rotor)
- `square()` — geometric square A*A
- `exponential()` — alias for exp (bivector → rotor)
- `logarithm()` — alias for log (rotor → bivector)
- Regressive product / meet (`regressive()`, `meet()`, `vee()`)
- Grade projection (`grade(k)`)
- Scalar extraction (`scalar()`)
- Blade grade and product primitives in `algebra.rs`

**Algebra System (Cl(p,q,r) Support)**
- `Algebra` class for working with arbitrary Clifford algebras
- `Signature` struct for metric signatures (p positive, q negative, r degenerate)
- Pre-computed Cayley tables for O(1) product lookup
- `Algebra.euclidean(n)` — Euclidean algebra Cl(n,0,0)
- `Algebra.pga(n)` — Projective Geometric Algebra Cl(n,0,1)
- `Algebra.cga(n)` — Conformal Geometric Algebra Cl(n+1,1,0)
- `Algebra.sta()` — Spacetime Algebra Cl(1,3,0)
- `Algebra(p, q, r)` — arbitrary signature Cl(p,q,r)
- `algebra.dimension` — vector space dimension
- `algebra.num_blades` — total basis blades (2^dim)
- `algebra.p`, `algebra.q`, `algebra.r` — signature components
- `algebra.is_euclidean()`, `algebra.is_pga()` — signature checks
- `algebra.blade_name(index)` — human-readable blade names
- Algebra equality and hashing for use in collections
- Thread-safe algebra caching via `Arc<Algebra>`

**Algebra-Aware Constructors**
- `Multivector.zero_in(algebra)` — zero multivector in algebra
- `Multivector.scalar_in(value, algebra)` — scalar in algebra
- `Multivector.vector_in(coords, algebra)` — vector in algebra
- `Multivector.basis_in(index, algebra)` — basis vector in algebra
- `Multivector.pseudoscalar_in(algebra)` — pseudoscalar in algebra
- `multivector.algebra()` — get algebra if explicitly set

**PGA Convenience Methods (Projective Geometric Algebra)**
- `pga_point(coords)` — create point from Euclidean coordinates
- `pga_plane(normal, distance)` — create plane from normal and distance
- `pga_line_from_points(p1, p2)` — create line through two points
- `pga_motor_from_axis_angle(axis, angle)` — create motor (rotation) from axis and angle
- `pga_translator(direction, distance)` — create translator along direction
- `pga_normalize()` — normalize PGA element (ideal norm = 1)
- `pga_ideal_part()` — extract ideal (at infinity) part
- `pga_euclidean_part()` — extract Euclidean (finite) part

**CGA Convenience Methods (Conformal Geometric Algebra)**
- `cga_eo()` — origin point (e_o = (e_- - e_+)/2)
- `cga_einf()` — point at infinity (e_∞ = e_- + e_+)
- `cga_point(coords)` — embed Euclidean point in CGA
- `cga_sphere(center, radius)` — create sphere from center and radius
- `cga_plane(normal, distance)` — create plane from normal and distance
- `cga_circle(center, radius, normal)` — create circle
- `cga_extract_center()` — extract center from round object
- `cga_extract_radius()` — extract radius from round object
- `cga_is_round()` — check if object is round (sphere, circle)
- `cga_is_flat()` — check if object is flat (plane, line)

**STA Convenience Methods (Spacetime Algebra)**
- `sta_vector(t, x, y, z)` — create 4-vector (event)
- `sta_gamma(i)` — get gamma matrix (γ_0, γ_1, γ_2, γ_3)
- `sta_bivector(components)` — create spacetime bivector
- `sta_boost(rapidity, direction)` — create Lorentz boost
- `sta_rotation(angle, plane)` — create spatial rotation
- `sta_spacetime_split(observer)` — split relative to observer (time, space)
- `sta_proper_time()` — compute proper time of worldline
- `sta_is_timelike()` — check if interval is timelike (τ² > 0)
- `sta_is_spacelike()` — check if interval is spacelike (τ² < 0)
- `sta_is_lightlike()` — check if interval is lightlike (τ² ≈ 0)

**SIMD Acceleration**
- Automatic SIMD for algebras with 16+ blades
- Uses `wide::f64x4` for portable 4-wide operations
- Falls back to scalar for smaller algebras
- Accelerates geometric product, additions, scaling

**Batch Operations (NumPy Integration)**
- `MultivectorBatch` class for vectorized operations
- `MultivectorBatch.from_numpy(coeffs, algebra)` — create from NumPy array
- `MultivectorBatch.from_vectors(coords, algebra)` — create from vector coords
- `MultivectorBatch.from_points_pga(coords)` — create PGA points
- `to_numpy()` — export to NumPy array
- Batch geometric product, outer product, inner product
- Batch sandwich product for transformations
- Batch norms, normalization, reversal
- Broadcasting support for scalar operations

**Visualization (ganja.js Integration)**
- `lcc.viz.show(elements, algebra)` — display in Jupyter notebook
- `lcc.viz.Graph` class for building visualizations
- Support for PGA, CGA, and Euclidean algebras
- Color and label customization
- IPython HTML rendering

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
- `allclose(other, rtol, atol)` — numpy-style tolerance comparison
- `almost_zero(tol)` — check if all coefficients are near zero
- `max_abs_diff(other)` — maximum absolute coefficient difference
- `relative_error(other)` — relative error norm (||a-b|| / ||a||)
- `is_normalized(tol)` — check if norm is close to 1
- `snap_to_zero(tol)` — set near-zero coefficients to exactly zero
- `coefficient_distance(other)` — Euclidean distance in coefficient space
- `is_simple_blade(tol)` — check if approximately a simple blade
- `dominant_coefficient()` — get coefficient with largest absolute value
- `normalize_dominant()` — scale so dominant coefficient equals 1

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
- `compose_with(other)` — compose two rotors (apply self first, then other)
- `inverse_rotor()` — compute inverse of a rotor
- `rotor_difference(other)` — rotor that transforms self to other
- `Multivector.rotor_between_planes(plane1, plane2)` — rotor rotating plane1 to plane2
- `same_rotation(other)` — check if two rotors represent the same rotation (R ≈ ±other)
- `decompose_rotor()` — decompose 3D rotor into (axis, angle) tuple
- `normalize_rotor()` — normalize rotor to unit norm
- `rotation_angle_degrees()` — rotation angle in degrees
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
- `reflect_in_plane(plane)` — reflection through a plane (bivector)
- `double_reflection(n1, n2)` — two successive reflections (produces rotation)
- `is_reflection()` — check if this is a reflection versor (unit odd versor)
- `Multivector.rotor_from_reflections(n1, n2)` — create rotor from two reflection vectors
- `project(B)` — projection onto blade B
- `reject(B)` — rejection from blade B
- `parallel_component(v)` — component parallel to vector (alias for project)
- `perpendicular_component(v)` — component perpendicular to vector (alias for reject)
- `rotate_by(rotor)` — apply rotor rotation to this multivector
- `rotate_in_plane(angle, plane)` — rotate by angle in a given bivector plane
- `project_onto_plane(plane)` — project onto a plane (bivector)
- `area(other)` — signed area of parallelogram spanned by two vectors
- `volume(b, c)` — signed volume of parallelepiped spanned by three vectors
- `angle_to_plane(plane)` — angle between vector and plane
- `distance_to_plane(plane, point_on_plane)` — perpendicular distance to plane
- `lies_in_plane(plane)` — check if vector lies in a plane
- `is_perpendicular_to_plane(plane)` — check if vector is perpendicular to plane
- `max_grade_part()` — extract highest-grade component
- `min_grade_part()` — extract lowest-grade component
- `split_even_odd()` — return (even_part, odd_part) tuple
- `blade_square()` — compute scalar square of a blade
- `is_null()` — check if blade squares to zero
- `complement()` — multiply by pseudoscalar (grade complement)
- `reverse_complement()` — multiply by inverse pseudoscalar
- `num_grades()` — count distinct non-zero grades
- `is_homogeneous()` — check if single-grade
- `homogeneous_grade()` — return grade if single-grade, None otherwise
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
- Python test suite (1014 tests)
- Rust test suite (7 tests)
- Benchmark suite (Criterion for Rust, vs_competition.py for Python comparisons)
- PyO3 bindings
- GitHub Actions CI/CD with path filtering and caching
- PyPI release workflow with trusted publishing
- Security scanning (cargo audit, dependency review)
- ARCHITECTURE.md documenting design decisions
- CONTRIBUTING.md with comment conventions, citation policy, and accessibility guidelines

### Known Limitations
- Runtime dimension checking (compile-time const generics planned for v1.0)
- ML framework integration (PyTorch/JAX) planned for future release
- Symbolic computation and code generation planned for future release

