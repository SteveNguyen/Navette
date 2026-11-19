//! Navette core: everything between the floor mask and the base command, in pure Rust.
//!
//! Port of a Python prototype, whose outputs on seeded synthetic inputs are the regression oracle
//! (tests/golden). Module names follow Nav2 where it has an equivalent.

pub mod collision_monitor;
pub mod controller;
pub mod exploration;
pub mod geometry;
pub mod navigator;
pub mod occupancy_grid;
pub mod scan;
pub mod state_estimation;
pub mod velocity_smoother;
