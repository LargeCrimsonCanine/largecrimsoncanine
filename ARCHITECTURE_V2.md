# LargeCrimsonCanine v2.0 Architecture Plan

**Vision**: Be the NumPy of Geometric Algebra — the default library everyone reaches for.

**Goal**: Make every other GA library obsolete by being faster, more flexible, and easier to use.

---

## Table of Contents

1. [Competitive Landscape](#competitive-landscape)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Implementation Phases](#implementation-phases)
5. [API Design](#api-design)
6. [Testing Strategy](#testing-strategy)
7. [Parallelization Plan](#parallelization-plan)

---

## Competitive Landscape

### Libraries to Surpass

| Library | Language | Strengths | Weaknesses |
|---------|----------|-----------|------------|
| **Kingdon** | Python | Symbolic, type-agnostic, Cl(p,q,r), ML integration | Pure Python speed |
| **clifford** | Python | Mature, NumPy-based, Cl(p,q,r) | Slow, complex API |
| **galgebra** | Python | Symbolic (SymPy), educational | Not for numerical work |
| **ganja.js** | JavaScript | Visualization, PGA focus | JS only |
| **klein** | C++ | Extremely fast PGA | PGA only, C++ |
| **geometric_algebra** | Rust | Rust, type-safe | Limited features |

### Our Advantages to Exploit

1. **Rust backend** — Native speed without C++ complexity
2. **PyO3 bindings** — Seamless Python integration
3. **Clean slate** — No legacy API baggage
4. **Comprehensive API** — Already 795 tests, 100+ methods

### Gaps to Close

1. Cl(p,q,r) metric signatures (PGA, CGA, spacetime)
2. Batched/vectorized operations (NumPy arrays as coefficients)
3. ML framework integration (PyTorch, JAX)
4. Symbolic capabilities (code generation, not runtime interpretation)
5. Visualization

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LargeCrimsonCanine v2.0                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Python API Layer                            │   │
│  │  • lcc.Algebra("PGA3D") / lcc.Algebra(3,0,1)                       │   │
│  │  • lcc.Multivector(algebra, coefficients)                          │   │
│  │  • Operator overloading: *, ^, |, ~, etc.                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Algebra Registry                              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ Cl(n,0,0)│ │ PGA(n)   │ │ CGA(n)   │ │Spacetime │ │ Custom   │  │   │
│  │  │ Euclidean│ │Cl(n,0,1) │ │Cl(n+1,1) │ │Cl(1,3)/  │ │Cl(p,q,r) │  │   │
│  │  │          │ │          │ │          │ │Cl(3,1)   │ │          │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  │                                                                     │   │
│  │  Each algebra contains:                                             │   │
│  │  • Metric signature (p, q, r)                                       │   │
│  │  • Pre-computed Cayley table (product signs)                        │   │
│  │  • Sparsity masks (which products are non-zero)                     │   │
│  │  • Blade names (e1, e2, e12, e+, e-, eo, ei, etc.)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Coefficient Backend Layer                        │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│  │  │  Scalar  │ │  Array   │ │  Tensor  │ │ Symbolic │               │   │
│  │  │  (f64)   │ │ (ndarray)│ │(PyTorch) │ │ (codegen)│               │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │   │
│  │       │             │            │            │                     │   │
│  │       └─────────────┴────────────┴────────────┘                     │   │
│  │                           │                                         │   │
│  │              Rust trait: Coefficient                                │   │
│  │              • add, sub, mul, div, neg                              │   │
│  │              • zero, one                                            │   │
│  │              • sqrt, sin, cos, exp, log (for rotors)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Core Operations (Rust)                         │   │
│  │                                                                     │   │
│  │  Products:           Transforms:         Analysis:                  │   │
│  │  • geometric         • reverse           • grade extraction         │   │
│  │  • outer (wedge)     • involute          • norm                     │   │
│  │  • inner (contract)  • conjugate         • inverse                  │   │
│  │  • regressive        • dual              • exp/log                  │   │
│  │  • commutator        • sandwich          • sqrt/pow                 │   │
│  │                                                                     │   │
│  │  SIMD acceleration (when coefficients are arrays)                   │   │
│  │  Rayon parallelism (for large batches)                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Algebra Registry (`src/algebra/`)

**Purpose**: Define and cache algebra structures for any Cl(p,q,r).

**Files**:
- `src/algebra/mod.rs` — Module exports
- `src/algebra/signature.rs` — Metric signature (p, q, r)
- `src/algebra/cayley.rs` — Cayley table generation and caching
- `src/algebra/registry.rs` — Named algebra registry (PGA3D, CGA, etc.)
- `src/algebra/blades.rs` — Blade naming and indexing

**Key Structures**:

```rust
/// Metric signature for Cl(p,q,r)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Signature {
    pub p: usize,  // Positive (square to +1)
    pub q: usize,  // Negative (square to -1)
    pub r: usize,  // Zero (square to 0, degenerate)
}

impl Signature {
    pub fn euclidean(n: usize) -> Self { Signature { p: n, q: 0, r: 0 } }
    pub fn pga(n: usize) -> Self { Signature { p: n, q: 0, r: 1 } }
    pub fn cga(n: usize) -> Self { Signature { p: n + 1, q: 1, r: 0 } }
    pub fn spacetime_sta() -> Self { Signature { p: 1, q: 3, r: 0 } }
    pub fn spacetime_aps() -> Self { Signature { p: 3, q: 0, r: 0 } } // APS = Algebra of Physical Space

    pub fn dimension(&self) -> usize { self.p + self.q + self.r }
    pub fn num_blades(&self) -> usize { 1 << self.dimension() }

    /// Returns the square of basis vector i (0-indexed)
    /// Returns +1 for i < p, -1 for p <= i < p+q, 0 for i >= p+q
    pub fn basis_square(&self, i: usize) -> f64 {
        if i < self.p { 1.0 }
        else if i < self.p + self.q { -1.0 }
        else { 0.0 }
    }
}

/// Pre-computed algebra structure
pub struct Algebra {
    pub signature: Signature,
    pub cayley_signs: Vec<f64>,      // Flattened 2D: [a * num_blades + b] -> sign
    pub cayley_blades: Vec<usize>,   // Flattened 2D: [a * num_blades + b] -> result blade
    pub blade_names: Vec<String>,    // Human-readable names
    pub grade_masks: Vec<usize>,     // Bitmask for each grade
}
```

**Cayley Table Generation**:

```rust
/// Compute geometric product of two basis blades under signature
pub fn blade_product_signed(a: usize, b: usize, sig: &Signature) -> (usize, f64) {
    let mut sign = compute_reorder_sign(a, b);  // From current algebra.rs
    let result_blade = a ^ b;

    // Apply metric: for each basis vector that appears in both a and b,
    // we get a contraction with that basis vector's square
    let common = a & b;
    for i in 0..sig.dimension() {
        if (common >> i) & 1 == 1 {
            sign *= sig.basis_square(i);
        }
    }

    (result_blade, sign)
}
```

### 2. Generic Multivector (`src/multivector/`)

**Purpose**: Support different coefficient types (f64, arrays, tensors).

**Files**:
- `src/multivector/mod.rs` — Module exports
- `src/multivector/traits.rs` — Coefficient trait definition
- `src/multivector/scalar.rs` — f64 coefficient implementation
- `src/multivector/array.rs` — ndarray coefficient implementation (future)
- `src/multivector/core.rs` — Generic Multivector<C> implementation
- `src/multivector/ops.rs` — Operator implementations
- `src/multivector/python.rs` — PyO3 bindings

**Key Structures**:

```rust
/// Trait for coefficient types
pub trait Coefficient: Clone + Send + Sync {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    fn scale(&self, s: f64) -> Self;
    fn sqrt(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn abs(&self) -> Self;
    fn is_zero(&self, tol: f64) -> bool;
}

impl Coefficient for f64 { /* ... */ }
// Future: impl Coefficient for ndarray::Array1<f64> { /* ... */ }
// Future: impl Coefficient for PyTorchTensor { /* ... */ }

/// Generic multivector over coefficient type C
pub struct Multivector<C: Coefficient> {
    pub algebra: Arc<Algebra>,
    pub coeffs: Vec<C>,
}
```

**For Python (PyO3)**, we expose a concrete type:

```rust
/// Python-exposed multivector (f64 coefficients for now)
#[pyclass(name = "Multivector", eq, hash, frozen)]
pub struct PyMultivector {
    inner: Multivector<f64>,
}
```

### 3. Named Algebras (`src/algebras/`)

**Purpose**: Provide convenient constructors and specialized methods for common algebras.

**Files**:
- `src/algebras/mod.rs` — Re-exports
- `src/algebras/euclidean.rs` — Cl(n,0,0)
- `src/algebras/pga.rs` — Projective GA Cl(n,0,1)
- `src/algebras/cga.rs` — Conformal GA Cl(n+1,1,0)
- `src/algebras/sta.rs` — Spacetime Algebra Cl(1,3,0)

**Example (PGA)**:

```rust
// src/algebras/pga.rs

/// PGA convenience constructors and operations
impl Algebra {
    /// Create 2D PGA: Cl(2,0,1) - for 2D Euclidean geometry
    pub fn pga2d() -> Arc<Self> {
        Self::new(Signature::pga(2))
    }

    /// Create 3D PGA: Cl(3,0,1) - for 3D Euclidean geometry
    pub fn pga3d() -> Arc<Self> {
        Self::new(Signature::pga(3))
    }
}

/// PGA-specific multivector methods
impl<C: Coefficient> Multivector<C> {
    /// Create a point in PGA (grade-3 in PGA3D)
    pub fn pga_point(algebra: &Arc<Algebra>, x: C, y: C, z: C) -> Self {
        // Point = e123 + x*e032 + y*e013 + z*e021
        // (using standard PGA conventions)
        todo!()
    }

    /// Create a plane in PGA (grade-1 in PGA3D)
    pub fn pga_plane(algebra: &Arc<Algebra>, a: C, b: C, c: C, d: C) -> Self {
        // Plane = a*e1 + b*e2 + c*e3 + d*e0
        todo!()
    }

    /// Create a line in PGA (grade-2 in PGA3D)
    pub fn pga_line_from_points(p1: &Self, p2: &Self) -> Self {
        p1.regressive(p2)
    }

    /// Create a motor (rigid transformation) from rotor and translation
    pub fn pga_motor(rotor: &Self, translation: &Self) -> Self {
        todo!()
    }
}
```

### 4. Visualization (`lcc/viz/` - Python)

**Purpose**: ganja.js integration for Jupyter notebooks.

**Files**:
- `lcc/viz/__init__.py`
- `lcc/viz/ganja.py` — ganja.js wrapper
- `lcc/viz/jupyter.py` — Jupyter widget integration

**Approach**: Wrap ganja.js or pyganja, convert our multivectors to their format.

### 5. Code Generation (`lcc-codegen/` - separate crate)

**Purpose**: Generate optimized Rust code for specific algebras.

**Use Cases**:
1. Pre-generate optimized implementations for common algebras
2. Users can generate code for custom algebras
3. Symbolic simplification at compile time, not runtime

**Output**: Rust source files with specialized implementations.

---

## Implementation Phases

### Phase 1: Metric Signatures (Foundation)

**Deliverables**:
1. `Signature` struct with p, q, r
2. `blade_product_signed()` respecting metric
3. `Algebra` struct with cached Cayley table
4. Update `Multivector` to hold `Arc<Algebra>` instead of `dims: usize`
5. Migrate all 795 existing tests to work with new structure
6. Add tests for PGA and CGA basics

**Breaking Changes**:
- `Multivector.from_vector([1,2,3])` becomes `Multivector.from_vector(algebra, [1,2,3])`
- Or: Keep convenience API with default Euclidean algebra

**Parallel Work Streams**:
- Stream A: `src/algebra/` module (signature, cayley, registry)
- Stream B: Update `src/multivector.rs` to use Algebra
- Stream C: Named algebra conveniences (pga, cga constructors)
- Stream D: Test migration and new algebra tests

### Phase 2: Named Algebras & Conveniences

**Deliverables**:
1. `Algebra.euclidean(n)`, `Algebra.pga(n)`, `Algebra.cga(n)`
2. PGA-specific methods (point, plane, line, motor)
3. CGA-specific methods (point, sphere, circle, pair)
4. Blade naming conventions per algebra
5. Pretty-printing with algebra-aware names

**Parallel Work Streams**:
- Stream A: Euclidean conveniences (what we have now, polished)
- Stream B: PGA implementation and tests
- Stream C: CGA implementation and tests
- Stream D: STA (spacetime algebra) implementation

### Phase 3: Performance Optimization

**Deliverables**:
1. SIMD acceleration for products (using `std::simd` or `packed_simd`)
2. Sparse multivector representation option
3. Lazy evaluation for chains of operations
4. Rayon parallelism for batched operations
5. Benchmarks vs Kingdon, clifford, klein

**Parallel Work Streams**:
- Stream A: SIMD geometric product
- Stream B: Sparse representation
- Stream C: Benchmark suite
- Stream D: Rayon integration

### Phase 4: Batched/Array Coefficients

**Deliverables**:
1. `Coefficient` trait implementation for ndarray
2. Broadcasting semantics (one algebra, many instances)
3. Efficient memory layout for batched operations
4. NumPy interop via PyO3

**Parallel Work Streams**:
- Stream A: ndarray Coefficient impl
- Stream B: Batched product operations
- Stream C: NumPy bindings
- Stream D: Benchmarks vs vectorized Kingdon

### Phase 5: ML Integration

**Deliverables**:
1. PyTorch tensor coefficient support
2. Autograd through GA operations
3. JAX integration (via numpy bridge or direct)
4. Example notebooks: GA for neural networks

**Parallel Work Streams**:
- Stream A: PyTorch Coefficient impl
- Stream B: Autograd wrappers
- Stream C: JAX bridge
- Stream D: ML example notebooks

### Phase 6: Symbolic & Code Generation

**Deliverables**:
1. `lcc-codegen` crate for generating optimized Rust
2. Symbolic expression type for GA formulas
3. Compile symbolic to fast numerical code
4. CLI tool: `lcc-codegen --algebra PGA3D --output pga3d.rs`

**Parallel Work Streams**:
- Stream A: Symbolic expression AST
- Stream B: Code generation backend
- Stream C: CLI tool
- Stream D: Pre-generated common algebras

### Phase 7: Visualization & Polish

**Deliverables**:
1. ganja.js integration
2. Jupyter widget support
3. Comprehensive documentation
4. Tutorial notebooks
5. Website and branding

---

## API Design

### Python API (Target)

```python
import largecrimsoncanine as lcc

# Algebras
R3 = lcc.Algebra.euclidean(3)       # Cl(3,0,0)
PGA3D = lcc.Algebra.pga(3)          # Cl(3,0,1)
CGA = lcc.Algebra.cga(3)            # Cl(4,1,0)
custom = lcc.Algebra(2, 1, 0)       # Cl(2,1,0) - Minkowski 2+1

# Multivectors
v = lcc.vector(R3, [1, 2, 3])       # Vector in R3
R = lcc.rotor(R3, axis=[0,0,1], angle=0.5)  # Rotor

# PGA specifics
p = lcc.pga.point(PGA3D, 1, 2, 3)   # Point
plane = lcc.pga.plane(PGA3D, 0, 0, 1, 5)  # z = 5
line = p & q                         # Join (regressive product)
motor = R * T                        # Motor = rotation * translation

# CGA specifics
p = lcc.cga.point(CGA, 1, 2, 3)     # Conformal point
sphere = lcc.cga.sphere(CGA, center, radius)
circle = sphere1 ^ sphere2           # Intersection

# Operations (same across all algebras)
ab = a * b          # Geometric product
a_wedge_b = a ^ b   # Outer product
a_dot_b = a | b     # Inner product (left contraction)
a_meet_b = a & b    # Regressive product (meet/join)
~a                  # Reverse
a.dual()            # Dual
a.sandwich(b)       # a * b * ~a
a.exp()             # Exponential
a.log()             # Logarithm

# Batched operations (coefficients are arrays)
import numpy as np
points = lcc.vector(R3, np.random.randn(1000, 3))  # 1000 vectors
rotated = R.sandwich(points)  # Rotate all 1000 at once

# PyTorch integration
import torch
points_t = lcc.vector(R3, torch.randn(1000, 3, requires_grad=True))
loss = rotated.norm().sum()
loss.backward()  # Gradients flow through GA operations
```

### Backward Compatibility

The current API should continue to work:

```python
# Old API (still works, uses default Euclidean algebra)
v = lcc.Multivector.from_vector([1, 2, 3])  # Implicitly Cl(3,0,0)
e1 = lcc.Multivector.e1(3)                  # Implicitly Cl(3,0,0)

# New API (explicit algebra)
R3 = lcc.Algebra.euclidean(3)
v = lcc.Multivector.from_vector(R3, [1, 2, 3])
e1 = R3.e1()
```

---

## Testing Strategy

### Test Categories

1. **Unit Tests** (Rust): Test individual functions in isolation
2. **Integration Tests** (Rust): Test module interactions
3. **Python Tests** (pytest): Test PyO3 bindings and Python API
4. **Property Tests** (proptest): Algebraic identities hold for random inputs
5. **Benchmark Tests**: Performance regression detection

### Algebraic Identities to Test

For any algebra Cl(p,q,r), these must hold:

```
Associativity:     (a * b) * c == a * (b * c)
Distributivity:    a * (b + c) == a*b + a*c
Grade structure:   grade(a ^ b) == grade(a) + grade(b)  (when non-zero)
Reverse:           ~~a == a
                   ~(a*b) == ~b * ~a
Metric:            e_i * e_i == signature[i]  (+1, -1, or 0)
Orthogonality:     e_i * e_j == -e_j * e_i  (when i != j)
Inverse:           a * a^(-1) == 1  (for invertible a)
Dual:              a.dual().dual() == ±a  (sign depends on dimension)
Exponential:       exp(log(R)) == R  (for rotors)
                   log(exp(B)) == B  (for bivectors with |B| < π)
```

### Test Matrix

| Algebra | Dimensions | Signature | Special Tests |
|---------|------------|-----------|---------------|
| R2 | 2 | (2,0,0) | 2D rotations, complex number isomorphism |
| R3 | 3 | (3,0,0) | 3D rotations, quaternion isomorphism |
| R4 | 4 | (4,0,0) | 4D rotations |
| PGA2D | 3 | (2,0,1) | 2D points, lines, motors |
| PGA3D | 4 | (3,0,1) | 3D points, lines, planes, motors |
| CGA2D | 4 | (3,1,0) | Conformal 2D, circles |
| CGA3D | 5 | (4,1,0) | Conformal 3D, spheres |
| STA | 4 | (1,3,0) | Spacetime, Lorentz transforms |
| Cl(2,1,0) | 3 | (2,1,0) | Minkowski 2+1 |

---

## Parallelization Plan

### Phase 1 Parallel Streams

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 1                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stream A                    Stream B                           │
│  ─────────                   ─────────                          │
│  src/algebra/                Update Multivector                 │
│  • signature.rs              • Add algebra field                │
│  • cayley.rs                 • Update geometric_product         │
│  • registry.rs               • Update all ops to use algebra    │
│  • blades.rs                                                    │
│         │                            │                          │
│         └──────────┬─────────────────┘                          │
│                    │                                            │
│                    ▼                                            │
│              Integration                                        │
│              • Wire algebra into multivector                    │
│              • Update Python bindings                           │
│                    │                                            │
│         ┌─────────┴─────────┐                                   │
│         │                   │                                   │
│         ▼                   ▼                                   │
│    Stream C            Stream D                                 │
│    ─────────           ─────────                                │
│    Named Algebras      Test Migration                           │
│    • euclidean.rs      • Update 795 tests                       │
│    • pga.rs            • Add PGA tests                          │
│    • cga.rs            • Add CGA tests                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Dependency Graph

```
signature.rs ──┐
               ├──► cayley.rs ──┐
blades.rs ─────┘                │
                                ├──► registry.rs ──► multivector update
                                │
current multivector.rs ─────────┘
                                          │
                                          ▼
                                    Python bindings
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                 ▼
                   euclidean.rs       pga.rs           cga.rs
                        │                 │                 │
                        ▼                 ▼                 ▼
                   R3 tests          PGA tests         CGA tests
```

### Agent Task Specifications

**Agent A: Algebra Module**
```
CREATE src/algebra/mod.rs
CREATE src/algebra/signature.rs implementing:
  - Signature struct {p, q, r}
  - euclidean(n), pga(n), cga(n) constructors
  - basis_square(i) method
  - dimension(), num_blades() methods

CREATE src/algebra/cayley.rs implementing:
  - blade_product_signed(a, b, sig) -> (blade, sign)
  - compute_cayley_table(sig) -> (Vec<usize>, Vec<f64>)

CREATE src/algebra/blades.rs implementing:
  - blade_name(index, sig) -> String
  - parse_blade_name(name, sig) -> Option<usize>

CREATE src/algebra/registry.rs implementing:
  - Algebra struct with cached cayley table
  - Algebra::new(sig) with caching
  - Thread-safe algebra cache (lazy_static or once_cell)

TEST: All functions with unit tests
TEST: Verify e_i * e_i = signature[i] for various signatures
TEST: Verify e_i * e_j = -e_j * e_i for i != j
```

**Agent B: Multivector Update**
```
MODIFY src/multivector.rs:
  - Add algebra: Arc<Algebra> field (or reference)
  - Remove dims: usize field (get from algebra)
  - Update geometric_product to use algebra.cayley_*
  - Update all methods that use dims to use algebra.signature.dimension()
  - Keep backward compatibility: from_vector([1,2,3]) uses default Euclidean

ENSURE: All existing functionality preserved
ENSURE: No performance regression for Euclidean case
```

**Agent C: Named Algebras**
```
CREATE src/algebras/mod.rs
CREATE src/algebras/euclidean.rs with convenience methods
CREATE src/algebras/pga.rs with:
  - pga_point(), pga_plane(), pga_line()
  - pga_motor(), pga_translator(), pga_rotor()
CREATE src/algebras/cga.rs with:
  - cga_point(), cga_sphere(), cga_plane()
  - cga_circle(), cga_point_pair()

TEST: PGA3D geometric constructions
TEST: CGA3D conformal operations
```

**Agent D: Test Migration**
```
UPDATE tests/test_basic.py:
  - Ensure all 795 tests pass with new structure
  - Add explicit algebra where needed, or verify defaults work

ADD tests/test_pga.py:
  - PGA point/line/plane operations
  - Motor composition
  - Meet and join operations

ADD tests/test_cga.py:
  - Conformal point representation
  - Sphere/circle intersections
  - Round and flat objects
```

---

## Success Criteria

### Phase 1 Complete When:
- [ ] All existing 795 tests pass
- [ ] `lcc.Algebra.euclidean(3)` works
- [ ] `lcc.Algebra.pga(3)` works
- [ ] `lcc.Algebra(p, q, r)` works for any signature
- [ ] Basis vectors square correctly: `e_i * e_i == sig[i]`
- [ ] PGA3D points and planes can be created and manipulated
- [ ] CGA3D points and spheres can be created

### Library "Wins" When:
- [ ] Faster than Kingdon on numerical benchmarks
- [ ] Supports all algebras Kingdon supports
- [ ] Has ML integration (PyTorch tensors)
- [ ] Has visualization
- [ ] Documentation is excellent
- [ ] Community adoption (GitHub stars, PyPI downloads)

---

## File Structure (Target)

```
largecrimsoncanine/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── algebra/
│   │   ├── mod.rs
│   │   ├── signature.rs
│   │   ├── cayley.rs
│   │   ├── blades.rs
│   │   └── registry.rs
│   ├── multivector/
│   │   ├── mod.rs
│   │   ├── traits.rs        # Coefficient trait
│   │   ├── scalar.rs        # f64 impl
│   │   ├── core.rs          # Multivector<C>
│   │   ├── ops.rs           # Operations
│   │   └── python.rs        # PyO3 bindings
│   ├── algebras/
│   │   ├── mod.rs
│   │   ├── euclidean.rs
│   │   ├── pga.rs
│   │   ├── cga.rs
│   │   └── sta.rs
│   └── (future: simd/, batch/, symbolic/)
├── tests/
│   ├── test_basic.py        # Existing tests
│   ├── test_pga.py
│   ├── test_cga.py
│   ├── test_sta.py
│   └── test_algebra.py
├── lcc/                      # Python package
│   ├── __init__.py
│   ├── viz/
│   │   ├── __init__.py
│   │   └── ganja.py
│   └── notebooks/
├── benches/
│   ├── geometric_product.rs
│   ├── vs_kingdon.py
│   └── vs_clifford.py
└── lcc-codegen/              # Future: code generation crate
    ├── Cargo.toml
    └── src/
```

---

## Notes

- Keep all commits professional and buttoned up
- No time estimates in documentation
- Focus on correctness first, then performance
- Maintain backward compatibility where possible
- Every feature needs tests
- Benchmark against competition regularly
