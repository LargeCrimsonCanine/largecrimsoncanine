# Changelog

All notable changes to LargeCrimsonCanine will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core `Multivector` type with coefficient array storage
- Geometric product (`*` operator)
- Outer (wedge) product (`^` operator)
- Grade projection
- Scalar extraction
- Blade grade and product primitives in `algebra.rs`
- Python test suite
- PyO3 bindings scaffold
- ARCHITECTURE.md documenting design decisions
- CONTRIBUTING.md with comment conventions and citation policy

### Known Limitations
- Euclidean metric only (Cl(p,q,r) support planned for v0.2)
- Runtime dimension checking (compile-time const generics planned)
- Inner product, reverse, conjugate, dual, norm not yet implemented

