# Contributing to LargeCrimsonCanine

## Code Comment Conventions

Comments in this codebase follow strict conventions. Please read this before contributing.

### What belongs in a comment

- What a non-obvious piece of code does
- Why a particular implementation decision was made (especially when alternatives exist)
- Mathematical references with precise locations (chapter, section, equation number)
- Safety invariants that must be preserved

### What does not belong in a comment

- Restatements of what the code obviously does
- Personality, voice, or editorial opinion
- Speculation about future changes (use GitHub issues instead)

### Format

Rust doc comments use `///` for public API and `//` for inline explanation.

```rust
/// Computes the geometric product of two multivectors.
///
/// The geometric product is the fundamental operation of Clifford algebra.
/// For basis vectors e_i and e_j: e_i * e_j = e_ij if i != j, scalar if i == j.
///
/// Reference: Dorst et al. ch.3 [VERIFY]
pub fn geometric_product(&self, rhs: &Multivector) -> Multivector {
    // Grade structure is preserved; we iterate over all grade combinations.
    // This naive implementation is O(n^2) in the number of basis blades.
    // See issue #X for the planned optimization.
    ...
}
```

### Citation policy

All mathematical references must be marked [VERIFY] until independently confirmed. Do not add citations from memory. 

## Development Setup

Requires Rust (stable) and Python 3.8+.

```bash
git clone https://github.com/LargeCrimsonCanine/largecrimsoncanine
cd largecrimsoncanine
pip install maturin
maturin develop
```

## Pull Request Process

1. Open an issue first for non-trivial changes
2. Tests required for all new functionality
3. Doc comments required for all public API
4. Citations marked [VERIFY] until confirmed

## License

By contributing you agree your contributions are licensed under MIT.

