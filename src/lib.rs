use pyo3::prelude::*;

// Keep the old algebra module for backward compatibility
mod algebra_legacy;

// New algebra module with Cl(p,q,r) support
pub mod algebra;

mod multivector;
mod pyalgebra;

pub use multivector::Multivector;
pub use pyalgebra::PyAlgebra;

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
    Ok(())
}
