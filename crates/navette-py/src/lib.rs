//! Python bindings of navette-core: the extension module `navette._core`.
//!
//! One file per core module, each with a `register` function adding its classes and functions to the Python
//! module. To bind a new class, edit that module's file only.
//!
//! The GIL: the calls that can take long (FreeSpace.scan, Navigator.step / step_mask / observe,
//! OccupancyGrid.update / replay, FrontierPlanner.plan, FrontierExplore's steps) release it while the core
//! works (`Python::detach`), so that a slow plan does not stall the runtime's command thread (the base's
//! dead-man is fed from it) or the status page. The object stays mutably borrowed meanwhile: using it, or
//! the grid / planner it lends itself, from another thread during such a call raises RuntimeError
//! ("Already borrowed") instead of waiting. One thread per navigator.

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
