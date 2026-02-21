# Changelog

All notable changes to LargeCrimsonCanine will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Core**
- Core `Multivector` type with coefficient array storage
- Geometric product (`*` operator, `gp()`)
- Outer (wedge) product (`^` operator, `wedge()`)
- Left contraction / inner product (`|` operator, `lc()`, `inner()`)
- Grade projection (`grade(k)`)
- Scalar extraction (`scalar()`)
- Blade grade and product primitives in `algebra.rs`

**Constructors**
- `Multivector.zero(dims)` — zero multivector
- `Multivector.from_scalar(value, dims)` — scalar multivector
- `Multivector.from_vector(coords)` — vector from coordinates
- `Multivector.from_bivector(components, dims)` — bivector from components
- `Multivector.basis(index, dims)` — single basis vector
- `Multivector.pseudoscalar(dims)` — unit pseudoscalar

**Arithmetic**
- Addition, subtraction, negation (`+`, `-`, unary `-`)
- Scalar multiplication and division (`*`, `/`)
- Equality and approximate equality (`==`, `approx_eq()`)

**Norms and Inverses**
- `reverse()`, `tilde()` — reverse operation
- `norm()`, `norm_squared()` — magnitude
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

**Duality**
- `dual()` — left dual (A * I⁻¹)
- `undual()` — reverse dual (A * I)
- `right_dual()` — right dual (I⁻¹ * A)

**Geometric Operations**
- `reflect(n)` — reflection across hyperplane perpendicular to n
- `project(B)` — projection onto blade B
- `reject(B)` — rejection from blade B

**Infrastructure**
- Python test suite (188 tests)
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

