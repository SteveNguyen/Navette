//! Python bindings of navette-core: the extension module `navette._core`.
//!
//! One file per core module, each with a `register` function adding its classes and functions to the Python
//! module. To bind a new class, edit that module's file only.

use pyo3::prelude::*;

mod collision_monitor;
mod controller;
mod convert;
mod exploration;
mod geometry;
mod navigator;
mod occupancy_grid;
mod scan;
mod state_estimation;
mod velocity_smoother;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    geometry::register(m)?;
    scan::register(m)?;
    occupancy_grid::register(m)?;
    collision_monitor::register(m)?;
    velocity_smoother::register(m)?;
    state_estimation::register(m)?;
    exploration::register(m)?;
    controller::register(m)?;
    navigator::register(m)?;
    Ok(())
}
