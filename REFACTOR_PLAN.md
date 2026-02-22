# Multivector Refactor Plan: dims -> Algebra

**Date**: 2026-02-21
**Goal**: Replace `dims: usize` with `algebra: Arc<Algebra>` to support mixed-signature algebras (PGA, CGA, STA).

---

## 1. Current State

### Struct Definition (lines 18-27)

```rust
#[pyclass(eq, hash, frozen)]
#[derive(Debug, Clone)]
pub struct Multivector {
    pub coeffs: Vec<f64>,
    pub dims: usize,
}
```

### Key Limitations

1. **Euclidean Only**: The current `blade_product()` in `algebra.rs` assumes all basis vectors square to +1
2. **No Metric Awareness**: Operations like `geometric_product`, `norm_squared`, etc. cannot handle PGA's null vectors or STA's timelike vectors
3. **Dimension vs Signature**: `dims` tells us the vector space dimension but not the metric signature (p, q, r)

### Current algebra.rs Functions Used

| Function | Location | Purpose |
|----------|----------|---------|
| `blade_product(a, b)` | lines 23-27 | Returns (result_blade, sign) assuming Euclidean |
| `blade_grade(index)` | lines 7-9 | Returns grade (popcount) - metric-independent |
| `reverse_sign(grade)` | lines 36-43 | Returns sign for reverse - metric-independent |
| `grade_involution_sign(grade)` | lines 54-60 | Returns sign for involute - metric-independent |
| `clifford_conjugate_sign(grade)` | lines 75-81 | Returns sign for conjugate - metric-independent |

---

## 2. Changes Needed

### 2.1 Functions Using `dims` Field Access

These functions read `self.dims` and need to use `self.algebra.signature.dimension()` or a convenience accessor:

| Function | Line | Usage Pattern |
|----------|------|---------------|
| `PartialEq::eq` | 31 | `self.dims == other.dims` |
| `Hash::hash` | 39 | `self.dims.hash(state)` |
| `to_vector_coords` | 155 | `(0..self.dims).map(...)` |
| `__repr__` | 1601 | `format!(..., self.dims)` |
| `dimension()` | 1777 | returns `self.dims` |
| `dims()` | 1783 | returns `self.dims` |

### 2.2 Constructors Taking `dims: usize`

These static methods take `dims: usize` as a parameter and need overloads or default algebra:

| Function | Line | Signature |
|----------|------|-----------|
| `zero` | 255 | `fn zero(dims: usize)` |
| `from_scalar` | 275 | `fn from_scalar(value: f64, dims: usize)` |
| `from_list` | 308 | `fn from_list(coeffs: Vec<f64>)` - infers dims from len |
| `from_vector` | 337 | `fn from_vector(coords: Vec<f64>)` - infers dims from len |
| `basis` | 361 | `fn basis(index: usize, dims: usize)` |
| `pseudoscalar` | 390 | `fn pseudoscalar(dims: usize)` |
| `e1` | 412 | `fn e1(dims: usize)` |
| `e2` | 426 | `fn e2(dims: usize)` |
| `e3` | 445 | `fn e3(dims: usize)` |
| `e4` | 465 | `fn e4(dims: usize)` |
| `e12` | 486 | `fn e12(dims: usize)` |
| `e23` | 510 | `fn e23(dims: usize)` |
| `e31` | 535 | `fn e31(dims: usize)` |
| `random` | 581 | `fn random(dims: usize)` |
| `random_vector` | 614 | `fn random_vector(dims: usize)` |
| `random_rotor` | 649 | `fn random_rotor(dims: usize)` |
| `from_bivector` | 1044 | `fn from_bivector(components: Vec<f64>, dims: usize)` |

### 2.3 Constructors Creating Multivector Structs Inline

These locations create `Multivector { coeffs, dims }` directly and need updating:

| Location | Lines | Count |
|----------|-------|-------|
| Constructors returning Ok(Multivector { coeffs, dims }) | ~50 locations | Many |
| Helper methods creating return values | throughout | Many |

Key locations (grepped `dims: self.dims` and `dims:` patterns):
- Lines: 99, 264, 284, 325, 350, 378, 400, 496, 569, 815, 839, 1100, 1123, 1158, 1199, 1219, 1236, 1274, 1322, 1382, 1447, 1881, 1904, 1913, 1935, 2047, 2070, 2094, 2110, 2130, 2162, 2184, 2212, 2231, 2252, 2273, 2286, 2297, 2308, 2486, 2522, 2562, 2587, 2605

### 2.4 Functions Calling `blade_product`

These are the critical functions that need metric-aware product:

| Function | Line | Purpose |
|----------|------|---------|
| `geometric_product` | 1248-1276 | Core operation - calls `algebra::blade_product(i, j)` at line 1267 |
| `outer_product` | 1292-1324 | Wedge product - calls at line 1315 |
| `left_contraction` | 1342-1384 | Inner product - calls at line 1371 |
| `right_contraction` | 1407-1447 | Right contraction - calls at line 1436 |
| `scalar_product` | 1467-1495 | Dot product - calls at line 1486 |
| `norm_squared` | 2648-2670 | Norm computation - calls at line 2661 |

### 2.5 Dimension Compatibility Checks

These check `self.dims != other.dims` and need to check algebra compatibility:

| Function | Line | Error Pattern |
|----------|------|---------------|
| `geometric_product` | 1249 | dimension mismatch error |
| `outer_product` | 1293 | dimension mismatch error |
| `left_contraction` | 1343 | dimension mismatch error |
| `right_contraction` | 1408 | dimension mismatch error |
| `scalar_product` | 1468 | dimension mismatch error |
| `commutator` | 1515 | dimension mismatch error |
| `anticommutator` | 1537 | dimension mismatch error |
| `regressive` | 1566 | dimension mismatch error |
| `__add__` | 1866 | dimension mismatch error |
| `__sub__` | 1889 | dimension mismatch error |
| `approx_eq` | 1944 | dimension mismatch (silent false) |
| `div` | 2762 | dimension mismatch error |
| `sandwich` | 2798 | dimension mismatch error |
| `slerp` | 3250 | dimension mismatch error |
| `lerp` | 3285 | dimension mismatch error |
| Plus ~30 more geometric operations... | Various | Various |

---

## 3. Backward Compatibility Strategy

### 3.1 Python API Compatibility

**Goal**: Keep `Multivector.from_vector([1,2,3])` working with implicit Euclidean algebra.

**Strategy**: Use a global default algebra registry with lazy initialization:

```rust
// In src/algebra/registry.rs
use once_cell::sync::Lazy;
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::RwLock;

static EUCLIDEAN_ALGEBRAS: Lazy<RwLock<HashMap<usize, Arc<Algebra>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub fn get_euclidean(n: usize) -> Arc<Algebra> {
    {
        let cache = EUCLIDEAN_ALGEBRAS.read().unwrap();
        if let Some(alg) = cache.get(&n) {
            return alg.clone();
        }
    }
    let alg = Arc::new(Algebra::new(Signature::euclidean(n)));
    EUCLIDEAN_ALGEBRAS.write().unwrap().insert(n, alg.clone());
    alg
}
```

### 3.2 Constructor Patterns

**Pattern A: Explicit algebra (new API)**
```rust
pub fn from_vector_with_algebra(algebra: Arc<Algebra>, coords: Vec<f64>) -> PyResult<Self>
```

**Pattern B: Implicit Euclidean (backward compatible)**
```rust
pub fn from_vector(coords: Vec<f64>) -> PyResult<Self> {
    let dims = coords.len();
    let algebra = get_euclidean(dims);
    Self::from_vector_with_algebra(algebra, coords)
}
```

### 3.3 Python Bindings

For PyO3, we need to handle both cases:

```python
# Old API (still works)
v = Multivector.from_vector([1, 2, 3])  # Euclidean Cl(3,0,0)

# New API
R3 = Algebra.euclidean(3)
v = Multivector.from_vector(R3, [1, 2, 3])

# Or via algebra methods
v = R3.vector([1, 2, 3])
```

**Implementation**: Use `#[pyo3(signature = ...)]` with optional first argument:

```rust
#[staticmethod]
#[pyo3(signature = (coords, algebra=None))]
pub fn from_vector(
    coords: Vec<f64>,
    algebra: Option<&PyAlgebra>,
) -> PyResult<Self> {
    let alg = match algebra {
        Some(a) => a.inner.clone(),
        None => get_euclidean(coords.len()),
    };
    // ... implementation
}
```

---

## 4. Struct Changes

### 4.1 New Multivector Definition

```rust
use std::sync::Arc;

#[pyclass(eq, hash, frozen)]
#[derive(Debug, Clone)]
pub struct Multivector {
    pub coeffs: Vec<f64>,
    pub algebra: Arc<Algebra>,
}

impl Multivector {
    /// Convenience accessor for backward compatibility
    pub fn dims(&self) -> usize {
        self.algebra.signature.dimension()
    }

    /// Check if two multivectors are in the same algebra
    pub fn same_algebra(&self, other: &Multivector) -> bool {
        Arc::ptr_eq(&self.algebra, &other.algebra)
            || self.algebra.signature == other.algebra.signature
    }
}
```

### 4.2 Hash and Eq Changes

```rust
impl PartialEq for Multivector {
    fn eq(&self, other: &Self) -> bool {
        self.algebra.signature == other.algebra.signature
            && self.coeffs == other.coeffs
    }
}

impl std::hash::Hash for Multivector {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.algebra.signature.hash(state);  // Hash the signature, not Arc pointer
        for &c in &self.coeffs {
            c.to_bits().hash(state);
        }
    }
}
```

### 4.3 Product Implementation Changes

```rust
pub fn geometric_product(&self, other: &Multivector) -> PyResult<Self> {
    if !self.same_algebra(other) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "algebra mismatch: left operand is {} but right operand is {}",
            self.algebra.signature, other.algebra.signature
        )));
    }

    let size = self.coeffs.len();
    let mut result = vec![0.0f64; size];
    let n = self.algebra.signature.dimension();

    for (i, &a) in self.coeffs.iter().enumerate() {
        if a == 0.0 { continue; }
        for (j, &b) in other.coeffs.iter().enumerate() {
            if b == 0.0 { continue; }
            // Use algebra's Cayley table instead of blade_product
            let idx = i * (1 << n) + j;
            let blade = self.algebra.cayley_blades[idx];
            let sign = self.algebra.cayley_signs[idx];
            result[blade] += sign * a * b;
        }
    }

    Ok(Multivector {
        coeffs: result,
        algebra: self.algebra.clone(),
    })
}
```

---

## 5. Risk Assessment

### 5.1 Breaking Changes

| Category | Impact | Mitigation |
|----------|--------|------------|
| `Multivector { coeffs, dims }` syntax | Internal code only | Search and replace |
| `self.dims` field access | ~150 locations | Add `dims()` method, search/replace |
| Serialization (`__getstate__`, `__reduce__`) | Pickle compatibility | Add migration code |
| Hash values change | Set/dict keys | Document as breaking |

### 5.2 Tests That Will Fail Initially

1. **All 795 existing tests** - until backward compatibility layer works
2. **Serialization tests** - `__getstate__` returns `(coeffs, dims)` currently
3. **Equality tests** - hash/eq implementation changes
4. **Repr tests** - `__repr__` format changes

### 5.3 Performance Risks

1. **Arc cloning overhead**: Each operation clones `Arc<Algebra>` - negligible (atomic increment)
2. **Cayley table lookup**: May be faster than computing signs at runtime
3. **Memory**: Cached algebras stay in memory forever - acceptable for common algebras

### 5.4 Compatibility Risks

1. **Algebra equality**: Two algebras with same signature should be compatible
2. **Cross-algebra operations**: What happens with `euclidean(3) * pga(3)`? - Error, clearly
3. **Default algebra leaking**: Code assuming Euclidean might break silently for non-Euclidean - mitigated by explicit errors

---

## 6. Migration Order

### Phase 1: Foundation (No Breaking Changes)

1. **Create `src/algebra/` module**
   - `signature.rs`: Signature struct
   - `cayley.rs`: Cayley table generation
   - `registry.rs`: Algebra struct and caching
   - `blades.rs`: Blade naming

2. **Add to `src/lib.rs`**
   ```rust
   mod algebra;
   pub use algebra::{Signature, Algebra};
   ```

3. **Test algebra module in isolation**
   - Verify Cayley tables match blade_product for Euclidean
   - Add tests for PGA, CGA signatures

### Phase 2: Dual Structure (Backward Compatible)

1. **Add algebra field alongside dims**
   ```rust
   pub struct Multivector {
       pub coeffs: Vec<f64>,
       pub dims: usize,  // Keep for now
       algebra_opt: Option<Arc<Algebra>>,  // New, optional
   }
   ```

2. **Add helper methods**
   ```rust
   fn get_algebra(&self) -> Arc<Algebra> {
       self.algebra_opt.clone().unwrap_or_else(|| get_euclidean(self.dims))
   }
   ```

3. **Update product functions to use `get_algebra()`**
   - geometric_product
   - outer_product
   - left_contraction
   - right_contraction
   - scalar_product
   - norm_squared

4. **Run all 795 tests** - should pass

### Phase 3: New Constructors

1. **Add algebra-aware constructors**
   - `Multivector::with_algebra(algebra, coeffs)`
   - `Multivector::vector(algebra, coords)`
   - etc.

2. **Add Python Algebra class**
   ```python
   alg = lcc.Algebra.euclidean(3)
   alg = lcc.Algebra.pga(3)
   alg = lcc.Algebra(3, 0, 1)  # Cl(3,0,1)
   ```

3. **Add algebra constructors to Python**
   ```python
   v = alg.vector([1, 2, 3])
   e1 = alg.basis(1)
   ```

### Phase 4: Remove dims Field

1. **Update struct to final form**
   ```rust
   pub struct Multivector {
       pub coeffs: Vec<f64>,
       pub algebra: Arc<Algebra>,
   }
   ```

2. **Replace all `self.dims` with `self.dims()` method**

3. **Update all constructors**

4. **Update serialization**

5. **Run all tests**

### Phase 5: New Algebra Tests

1. **Add PGA3D test suite**
   - Point/line/plane creation
   - Meet and join operations
   - Motor composition

2. **Add CGA3D test suite**
   - Conformal point representation
   - Sphere/circle operations

3. **Add STA test suite**
   - Spacetime vectors
   - Lorentz transformations

---

## 7. File Changes Summary

| File | Changes |
|------|---------|
| `src/algebra.rs` | Keep existing functions, add metric-aware variants |
| `src/algebra/mod.rs` | NEW - module exports |
| `src/algebra/signature.rs` | NEW - Signature struct |
| `src/algebra/cayley.rs` | NEW - Cayley table generation |
| `src/algebra/registry.rs` | NEW - Algebra struct, caching |
| `src/algebra/blades.rs` | NEW - Blade naming |
| `src/multivector.rs` | Major refactor - ~150 changes |
| `src/lib.rs` | Add algebra module exports |
| `tests/test_basic.py` | No changes needed if backward compatible |
| `tests/test_pga.py` | NEW - PGA tests |
| `tests/test_cga.py` | NEW - CGA tests |
| `tests/test_algebra.py` | NEW - Algebra unit tests |

---

## 8. Success Criteria

- [ ] All 795 existing tests pass
- [ ] `Multivector.from_vector([1, 2, 3])` still works (Euclidean default)
- [ ] `Algebra.euclidean(3)` creates Cl(3,0,0)
- [ ] `Algebra.pga(3)` creates Cl(3,0,1)
- [ ] `Algebra(p, q, r)` creates arbitrary Cl(p,q,r)
- [ ] PGA null vectors square to 0
- [ ] STA timelike vectors square to -1
- [ ] Products use cached Cayley tables
- [ ] No performance regression for Euclidean operations

---

## Appendix A: Line Numbers Reference

Complete grep results for `self.dims` and `dims:` patterns:

```
26:    pub dims: usize,
31:        self.dims == other.dims && self.coeffs == other.coeffs
39:        self.dims.hash(state);
155:        (0..self.dims).map(|i| self.coeffs[1 << i]).collect()
255:    pub fn zero(dims: usize) -> PyResult<Self> {
275:    pub fn from_scalar(value: f64, dims: usize) -> PyResult<Self> {
323:        // Compute dims: len = 2^dims, so dims = log2(len)
361:    pub fn basis(index: usize, dims: usize) -> PyResult<Self> {
390:    pub fn pseudoscalar(dims: usize) -> PyResult<Self> {
412:    pub fn e1(dims: usize) -> PyResult<Self> {
426:    pub fn e2(dims: usize) -> PyResult<Self> {
445:    pub fn e3(dims: usize) -> PyResult<Self> {
465:    pub fn e4(dims: usize) -> PyResult<Self> {
486:    pub fn e12(dims: usize) -> PyResult<Self> {
510:    pub fn e23(dims: usize) -> PyResult<Self> {
535:    pub fn e31(dims: usize) -> PyResult<Self> {
569:        Ok(Multivector { coeffs, dims: 3 })
581:    pub fn random(dims: usize) -> PyResult<Self> {
614:    pub fn random_vector(dims: usize) -> PyResult<Self> {
649:    pub fn random_rotor(dims: usize) -> PyResult<Self> {
815:        Ok(Multivector { coeffs, dims: 3 })
839:        Multivector { coeffs, dims: 3 }
851:        if self.dims != 3 {
878:        if self.dims != 3 {
1000:        if self.dims != 3 {
1044:    pub fn from_bivector(components: Vec<f64>, dims: usize) -> PyResult<Self> {
1086:        if k > self.dims {
1109:        if k > self.dims {
1144:        if k > self.dims {
1179:        if k > self.dims {
1249:        if self.dims != other.dims {
1293:        if self.dims != other.dims {
1343:        if self.dims != other.dims {
1408:        if self.dims != other.dims {
1468:        if self.dims != other.dims {
1515:        if self.dims != other.dims {
1537:        if self.dims != other.dims {
1566:        if self.dims != other.dims {
1601:        format!("Multivector({:?}, dims={})", self.coeffs, self.dims)
1777:        self.dims
1783:        self.dims
1866:        if self.dims != other.dims {
1889:        if self.dims != other.dims {
... (continues for ~100 more lines)
```

## Appendix B: blade_product Call Sites

```
1267:                let (blade, sign) = algebra::blade_product(i, j);
1315:                let (blade, sign) = algebra::blade_product(i, j);
1371:                let (blade, sign) = algebra::blade_product(i, j);
1436:                let (blade, sign) = algebra::blade_product(i, j);
1486:                let (blade, sign) = algebra::blade_product(i, j);
2661:                let (blade, sign) = algebra::blade_product(i, j);
```

All 6 locations need to be updated to use `self.algebra.product(i, j)` or Cayley table lookup.
