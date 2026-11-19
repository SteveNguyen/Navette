//! Bindings of navette_core::controller: `ExploreParams`, `ArcController`.

use pyo3::prelude::*;

use navette_core::controller::dwa::{ArcController, ExploreParams, FixedGoal};
use navette_core::geometry::Pose2;

use crate::collision_monitor::{safety_params, PySafetyParams};
use crate::convert::param_err;
use crate::occupancy_grid::PyOccupancyGrid;
use crate::scan::PyScan;

/// Arc controller parameters (plain attributes, like a dataclass):
///   v_set            m/s, candidate forward speeds
///   w_max, n_w       candidate yaw rates: n_w values over [-w_max, w_max] rad/s
///   horizon_s        s, the heading error is evaluated after this long on the arc
///   k_heading, k_clear, k_speed   score weights
///   clear_cap        m of free arc that counts as fully clear
///   spin_above_deg   goal farther than this off the heading: turn in place,
///   spin_done_deg    until within this
///   w_spin           rad/s, yaw rate of the turns in place
///   blocked_s        s without an admissible arc or without motion: blocked
///   rotate_in_place  no admissible arc but the goal more than spin_done_deg off: turn toward it at once
///   moving_v         m/s, applied speed from which the robot counts as moving (below it for blocked_s:
///                    blocked); keep it under the slowest speed the robot is sent
/// ValueError on an invalid value (an empty v_set or a speed <= 0 in it, n_w = 0, clear_cap or w_spin <= 0,
/// a negative or non-finite value): at construction, and where the parameters are used (the attributes can
/// be set afterwards).
#[pyclass(name = "ExploreParams", module = "navette", get_all, set_all, eq)]
#[derive(Clone, PartialEq)]
pub struct PyExploreParams {
    v_set: Vec<f64>,
    w_max: f64,
    n_w: usize,
    horizon_s: f64,
    k_heading: f64,
    k_clear: f64,
    k_speed: f64,
    clear_cap: f64,
    spin_above_deg: f64,
    spin_done_deg: f64,
    w_spin: f64,
    blocked_s: f64,
    rotate_in_place: bool,
    moving_v: f64,
}

impl PyExploreParams {
    /// Copy into the core type, validated (the attributes are settable from Python).
    pub fn to_core(&self) -> PyResult<ExploreParams> {
        let p = ExploreParams {
            v_set: self.v_set.clone(),
            w_max: self.w_max,
            n_w: self.n_w,
            horizon_s: self.horizon_s,
            k_heading: self.k_heading,
            k_clear: self.k_clear,
            k_speed: self.k_speed,
            clear_cap: self.clear_cap,
            spin_above_deg: self.spin_above_deg,
            spin_done_deg: self.spin_done_deg,
            w_spin: self.w_spin,
            blocked_s: self.blocked_s,
            rotate_in_place: self.rotate_in_place,
            moving_v: self.moving_v,
        };
        p.validate().map_err(param_err)?;
        Ok(p)
    }

    pub fn from_core(p: &ExploreParams) -> Self {
        Self {
            v_set: p.v_set.clone(),
            w_max: p.w_max,
            n_w: p.n_w,
            horizon_s: p.horizon_s,
            k_heading: p.k_heading,
            k_clear: p.k_clear,
            k_speed: p.k_speed,
            clear_cap: p.clear_cap,
            spin_above_deg: p.spin_above_deg,
            spin_done_deg: p.spin_done_deg,
            w_spin: p.w_spin,
            blocked_s: p.blocked_s,
            rotate_in_place: p.rotate_in_place,
            moving_v: p.moving_v,
        }
    }
}

#[pymethods]
impl PyExploreParams {
    #[new]
    #[pyo3(signature = (v_set=vec![0.1, 0.2, 0.3], w_max=1.2, n_w=13, horizon_s=1.0, k_heading=1.0, k_clear=0.6,
                        k_speed=0.3, clear_cap=1.0, spin_above_deg=70.0, spin_done_deg=20.0, w_spin=1.0,
                        blocked_s=1.0, rotate_in_place=false, moving_v=0.05))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        v_set: Vec<f64>,
        w_max: f64,
        n_w: usize,
        horizon_s: f64,
        k_heading: f64,
        k_clear: f64,
        k_speed: f64,
        clear_cap: f64,
        spin_above_deg: f64,
        spin_done_deg: f64,
        w_spin: f64,
        blocked_s: f64,
        rotate_in_place: bool,
        moving_v: f64,
    ) -> PyResult<Self> {
        let p = Self {
            v_set,
            w_max,
            n_w,
            horizon_s,
            k_heading,
            k_clear,
            k_speed,
            clear_cap,
            spin_above_deg,
            spin_done_deg,
            w_spin,
            blocked_s,
            rotate_in_place,
            moving_v,
        };
        p.to_core()?;
        Ok(p)
    }

    fn __repr__(&self) -> String {
        format!(
            "ExploreParams(v_set={:?}, w_max={:?}, n_w={}, horizon_s={:?}, k_heading={:?}, k_clear={:?}, \
             k_speed={:?}, clear_cap={:?}, spin_above_deg={:?}, spin_done_deg={:?}, w_spin={:?}, \
             blocked_s={:?}, rotate_in_place={}, moving_v={:?})",
            self.v_set,
            self.w_max,
            self.n_w,
            self.horizon_s,
            self.k_heading,
            self.k_clear,
            self.k_speed,
            self.clear_cap,
            self.spin_above_deg,
            self.spin_done_deg,
            self.w_spin,
            self.blocked_s,
            if self.rotate_in_place { "True" } else { "False" },
            self.moving_v
        )
    }
}

/// The given parameters (validated), or `default`.
pub fn explore_params(
    p: Option<PyRef<'_, PyExploreParams>>,
    default: ExploreParams,
) -> PyResult<ExploreParams> {
    p.map_or(Ok(default), |p| p.to_core())
}

/// Arc selection toward a goal heading (see the core module's documentation): each step tries (v, w) arcs
/// against the scan and the map and returns the best admissible one, or a turn in place.
///   state          "DRIVE" or "SPIN"
///   goal           the goal heading of the last step (fixed frame, rad)
///   blocked_since  since when (s) there has been no admissible arc or no motion; None: not blocked
#[pyclass(name = "ArcController", module = "navette")]
pub struct PyArcController {
    inner: ArcController,
}

#[pymethods]
impl PyArcController {
    #[new]
    #[pyo3(signature = (p=None, safety=None))]
    fn new(
        p: Option<PyRef<'_, PyExploreParams>>,
        safety: Option<PyRef<'_, PySafetyParams>>,
    ) -> PyResult<Self> {
        let params = explore_params(p, ExploreParams::default())?;
        Ok(Self {
            inner: ArcController::new(params, safety_params(safety)?).map_err(param_err)?,
        })
    }

    #[getter]
    fn state(&self) -> &'static str {
        self.inner.state.as_str()
    }

    #[getter]
    fn goal(&self) -> Option<f64> {
        self.inner.goal
    }

    #[getter]
    fn blocked_since(&self) -> Option<f64> {
        self.inner.blocked_since
    }

    /// Requested (v, w) at time t (s) from pose (x, y, theta) toward the fixed-frame heading `goal` (rad).
    /// scan: this frame's Scan; grid: the OccupancyGrid; v_applied: the speed command sent last step (m/s).
    /// Blocked for blocked_s: the controller turns to `goal` in place.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        t: f64,
        pose: [f64; 3],
        scan: PyRef<'_, PyScan>,
        grid: PyRef<'_, PyOccupancyGrid>,
        v_applied: f64,
        goal: f64,
    ) -> PyResult<(f64, f64)> {
        let scan = scan.to_core(py)?;
        let pose = Pose2::new(pose[0], pose[1], pose[2]);
        Ok(self
            .inner
            .step(&mut FixedGoal(goal), t, pose, Some(&scan), &grid.inner, v_applied))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExploreParams>()?;
    m.add_class::<PyArcController>()?;
    Ok(())
}
