use pyo3::prelude::*;

mod algebra;
mod multivector;

pub use multivector::Multivector;

/// Python module entry point.
///
/// Registers all types and functions exposed to Python.
#[pymodule]
fn largecrimsoncanine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Multivector>()?;
    Ok(())
}

