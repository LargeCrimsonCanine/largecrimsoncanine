use pyo3::prelude::*;

// Keep the old algebra module for backward compatibility
mod algebra_legacy;

// New algebra module with Cl(p,q,r) support
pub mod algebra;

// SIMD-accelerated operations
pub mod simd;

// Batched operations with NumPy support
mod batch;

mod multivector;
mod pyalgebra;

// PGA convenience methods (extends Multivector)
mod pga;

// CGA convenience methods (extends Multivector)
mod cga;

// STA convenience methods (extends Multivector)
mod sta;

pub use multivector::Multivector;
pub use pyalgebra::PyAlgebra;
pub use batch::PyMultivectorBatch;

// Re-export algebra types at crate level for convenience
pub use algebra::{Algebra, Signature};

/// Python module entry point.
///
/// Registers all types and functions exposed to Python.
#[pymodule]
#[pyo3(name = "largecrimsoncanine")]
fn largecrimsoncanine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Multivector>()?;
    m.add_class::<PyAlgebra>()?;
    m.add_class::<PyMultivectorBatch>()?;
    Ok(())
}
