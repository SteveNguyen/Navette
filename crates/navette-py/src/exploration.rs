//! Bindings of navette_core::exploration as a numpy API: `FrontierParams`, `frontier_cells`,
//! `distance_transform`, `dijkstra`, `FrontierPlanner`, `FrontierExplore`.
//! A pose is an (x, y, theta) sequence, points are (N, 2) arrays in the fixed frame, grids are 2-D arrays
//! with [i, j] = the cell at x = x0 + i res, y = y0 + j res.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use navette_core::controller::dwa::{ExploreParams, GoalSource};
use navette_core::exploration::frontier::{self, FrontierParams, FrontierPlanner};
use navette_core::exploration::frontier_explore::FrontierExplore;
use navette_core::geometry::{Point2, Pose2};

use crate::collision_monitor::{safety_params, sector_mask, PySafetyParams};
use crate::controller::{explore_params, PyExploreParams};
use crate::convert::{param_err, rows_to_py};
use crate::occupancy_grid::PyOccupancyGrid;
use crate::scan::PyScan;

fn pose2(p: [f64; 3]) -> Pose2 {
    Pose2::new(p[0], p[1], p[2])
}

fn point_to_py(py: Python<'_>, p: Point2) -> Bound<'_, PyArray1<f64>> {
    PyArray1::from_slice(py, &p)
}

/// Frontier planner parameters (plain attributes, like a dataclass):
///   clear             m, distance paths keep from obstacle cells: robot radius + margin
///   soft_m, soft_k    cells cost up to 1 + soft_k per metre within soft_m beyond `clear`
///   blind             m, radius around the robot the camera cannot see
///   min_frontier_m    frontier groups shorter than this are dropped
///   min_unknown_m2    smaller unknown holes are no frontier
///   turn_k            m of path cost per rad the robot must turn toward the frontier
///   replan_s          s between plans
///   lookahead         m, the goal heading points at the path point this far ahead
///   keep_ratio        switch target only if the new one costs < keep_ratio x the current (0: commit)
///   target_timeout_s  a target not reached after this long is banned,
///   ban_s, ban_r      for ban_s, within ban_r of it
/// ValueError on an invalid value (negative, not finite, soft_m or target_timeout_s = 0): at construction,
/// and where the parameters are used (the attributes can be set afterwards).
#[pyclass(name = "FrontierParams", module = "navette", get_all, set_all, eq)]
#[derive(Clone, PartialEq)]
pub struct PyFrontierParams {
    clear: f64,
    soft_m: f64,
    soft_k: f64,
    blind: f64,
    min_frontier_m: f64,
    min_unknown_m2: f64,
    turn_k: f64,
    replan_s: f64,
    lookahead: f64,
    keep_ratio: f64,
    target_timeout_s: f64,
    ban_s: f64,
    ban_r: f64,
}

impl PyFrontierParams {
    /// Copy into the core type, validated (the attributes are settable from Python).
    pub fn to_core(&self) -> PyResult<FrontierParams> {
        let p = FrontierParams {
            clear: self.clear,
            soft_m: self.soft_m,
            soft_k: self.soft_k,
            blind: self.blind,
            min_frontier_m: self.min_frontier_m,
            min_unknown_m2: self.min_unknown_m2,
            turn_k: self.turn_k,
            replan_s: self.replan_s,
            lookahead: self.lookahead,
            keep_ratio: self.keep_ratio,
            target_timeout_s: self.target_timeout_s,
            ban_s: self.ban_s,
            ban_r: self.ban_r,
        };
        p.validate().map_err(param_err)?;
        Ok(p)
    }

    fn from_core(p: &FrontierParams) -> Self {
        Self {
            clear: p.clear,
            soft_m: p.soft_m,
            soft_k: p.soft_k,
            blind: p.blind,
            min_frontier_m: p.min_frontier_m,
            min_unknown_m2: p.min_unknown_m2,
            turn_k: p.turn_k,
            replan_s: p.replan_s,
            lookahead: p.lookahead,
            keep_ratio: p.keep_ratio,
            target_timeout_s: p.target_timeout_s,
            ban_s: p.ban_s,
            ban_r: p.ban_r,
        }
    }
}

#[pymethods]
impl PyFrontierParams {
    #[new]
    #[pyo3(signature = (clear=0.18, soft_m=0.3, soft_k=3.0, blind=0.5, min_frontier_m=0.25, min_unknown_m2=0.1,
                        turn_k=0.5, replan_s=1.0, lookahead=0.6, keep_ratio=0.0, target_timeout_s=30.0,
                        ban_s=60.0, ban_r=0.5))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        clear: f64,
        soft_m: f64,
        soft_k: f64,
        blind: f64,
        min_frontier_m: f64,
        min_unknown_m2: f64,
        turn_k: f64,
        replan_s: f64,
        lookahead: f64,
        keep_ratio: f64,
        target_timeout_s: f64,
        ban_s: f64,
        ban_r: f64,
    ) -> PyResult<Self> {
        let p = Self {
            clear,
            soft_m,
            soft_k,
            blind,
            min_frontier_m,
            min_unknown_m2,
            turn_k,
            replan_s,
            lookahead,
            keep_ratio,
            target_timeout_s,
            ban_s,
            ban_r,
        };
        p.to_core()?;
        Ok(p)
    }

    fn __repr__(&self) -> String {
        format!(
            "FrontierParams(clear={:?}, soft_m={:?}, soft_k={:?}, blind={:?}, min_frontier_m={:?}, \
             min_unknown_m2={:?}, turn_k={:?}, replan_s={:?}, lookahead={:?}, keep_ratio={:?}, \
             target_timeout_s={:?}, ban_s={:?}, ban_r={:?})",
            self.clear,
            self.soft_m,
            self.soft_k,
            self.blind,
            self.min_frontier_m,
            self.min_unknown_m2,
            self.turn_k,
            self.replan_s,
            self.lookahead,
            self.keep_ratio,
            self.target_timeout_s,
            self.ban_s,
            self.ban_r
        )
    }
}

/// The given parameters (validated), or the defaults.
pub fn frontier_params(p: Option<PyRef<'_, PyFrontierParams>>) -> PyResult<FrontierParams> {
    p.map_or_else(|| Ok(FrontierParams::default()), |p| p.to_core())
}

/// A 2-D array as a row-major Vec and its shape (h, w).
fn grid_of<T: numpy::Element + Copy>(a: &PyReadonlyArray2<'_, T>) -> (Vec<T>, usize, usize) {
    let a = a.as_array();
    let (h, w) = a.dim();
    (a.iter().copied().collect(), h, w)
}

fn same_shape(a: (usize, usize), b: (usize, usize), names: &str) -> PyResult<()> {
    if a != b {
        return Err(PyValueError::new_err(format!(
            "{names} must have the same shape, got {a:?} and {b:?}"
        )));
    }
    Ok(())
}

/// Boolean grid of the frontier cells: free cells with an unknown (neither obstacle nor free) 4-neighbour,
/// without the 8-connected groups smaller than min_cells. Unknown blobs (4-connected) smaller than
/// min_unknown_cells make no frontier. obst, free: bool grids of the same shape.
#[pyfunction]
#[pyo3(signature = (obst, free, min_cells, min_unknown_cells=0))]
fn frontier_cells<'py>(
    py: Python<'py>,
    obst: PyReadonlyArray2<'py, bool>,
    free: PyReadonlyArray2<'py, bool>,
    min_cells: usize,
    min_unknown_cells: usize,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let ((obst, h, w), (free, hf, wf)) = (grid_of(&obst), grid_of(&free));
    same_shape((h, w), (hf, wf), "obst and free")?;
    let fr = frontier::frontier_cells(&obst, &free, h, w, min_cells, min_unknown_cells);
    PyArray1::from_vec(py, fr).reshape([h, w])
}

/// Exact Euclidean distance (in cells) from every cell to the nearest True cell of the bool grid `obst`
/// (scipy.ndimage.distance_transform_edt(~obst)); without any, the distance to cell (-1, 0).
#[pyfunction]
fn distance_transform<'py>(
    py: Python<'py>,
    obst: PyReadonlyArray2<'py, bool>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (obst, h, w) = grid_of(&obst);
    PyArray1::from_vec(py, frontier::distance_transform(&obst, h, w)).reshape([h, w])
}

/// Cheapest path cost from the cell `start` = (i, j) to every cell, moving between 8-neighbours that are
/// both True in the bool grid `trav`; an edge costs its length (res, or res sqrt(2)) x the mean of the float
/// grid `cost` over its two cells. Returns (cost grid, inf where not reached; predecessor grid: the row-major
/// index i * w + j of the previous cell on the path, -1 for the start and where not reached).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn dijkstra<'py>(
    py: Python<'py>,
    trav: PyReadonlyArray2<'py, bool>,
    cost: PyReadonlyArray2<'py, f64>,
    res: f64,
    start: (usize, usize),
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>)> {
    let ((trav, h, w), (cost, hc, wc)) = (grid_of(&trav), grid_of(&cost));
    same_shape((h, w), (hc, wc), "trav and cost")?;
    if start.0 >= h || start.1 >= w {
        return Err(PyValueError::new_err(format!(
            "start {start:?} is outside the ({h}, {w}) grid"
        )));
    }
    let (d, pred) = frontier::dijkstra(&trav, &cost, h, w, res, start.0 * w + start.1);
    let pred: Vec<i64> = pred
        .into_iter()
        .map(|p| if p == frontier::NO_PRED { -1 } else { i64::from(p) })
        .collect();
    Ok((
        PyArray1::from_vec(py, d).reshape([h, w])?,
        PyArray1::from_vec(py, pred).reshape([h, w])?,
    ))
}

/// Frontier planner on an OccupancyGrid (see the core module's documentation).
///   p         FrontierParams (a copy)
///   path      (N, 2) fixed-frame points from the robot to the target; None without a plan
///   target    (2,) fixed-frame point; None without a plan
///   done      the last plan found no reachable frontier
///   banned    [(time of the ban, (2,) point)], oldest first
///   frontier  (N, 2) frontier cell centres of the last plan (for display)
#[pyclass(name = "FrontierPlanner", module = "navette")]
pub struct PyFrontierPlanner {
    pub(crate) inner: FrontierPlanner,
}

#[pymethods]
impl PyFrontierPlanner {
    #[new]
    #[pyo3(signature = (p=None))]
    fn new(p: Option<PyRef<'_, PyFrontierParams>>) -> PyResult<Self> {
        Ok(Self {
            inner: FrontierPlanner::new(frontier_params(p)?).map_err(param_err)?,
        })
    }

    #[getter]
    fn p(&self) -> PyFrontierParams {
        PyFrontierParams::from_core(&self.inner.params)
    }

    #[getter]
    fn path<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        self.inner.path().map(|p| rows_to_py(py, p)).transpose()
    }

    #[getter]
    fn target<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.target().map(|t| point_to_py(py, t))
    }

    #[getter]
    fn done(&self) -> bool {
        self.inner.done()
    }

    #[getter]
    fn banned<'py>(&self, py: Python<'py>) -> Vec<(f64, Bound<'py, PyArray1<f64>>)> {
        self.inner
            .banned()
            .iter()
            .map(|&(t, q)| (t, point_to_py(py, q)))
            .collect()
    }

    #[getter]
    fn frontier<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        rows_to_py(py, self.inner.frontier())
    }

    /// Give the current target up for ban_s.
    fn ban(&mut self, t: f64) {
        self.inner.ban(t);
    }

    /// Choose the target and the path from pose on grid at time t (s).
    /// near: robot-frame bearings (rad) of the CURRENT scan's obstacles filling the image bottom
    /// (Scan.near), near_half: their sector half-width (rad); used for this plan only.
    #[pyo3(signature = (t, pose, grid, near=None, near_half=0.0))]
    fn plan(
        &mut self,
        py: Python<'_>,
        t: f64,
        pose: [f64; 3],
        grid: PyRef<'_, PyOccupancyGrid>,
        near: Option<Vec<f64>>,
        near_half: f64,
    ) {
        let (inner, grid, near) = (&mut self.inner, &grid.inner, near.unwrap_or_default());
        // without the GIL (see lib.rs)
        py.detach(|| inner.plan(t, pose2(pose), grid, &near, near_half));
    }

    /// Fixed-frame heading (rad) to the path point `lookahead` ahead; None without a path.
    fn heading(&self, pose: [f64; 3]) -> Option<f64> {
        self.inner.heading(pose2(pose))
    }
}

/// Frontier exploration: the arc controller steered along the frontier planner's path, with the look-around
/// before giving up and the recovery turn when blocked (see the core module's documentation).
///   p        ExploreParams (a copy); default: ExploreParams(rotate_in_place=True)
///   planner  the FrontierPlanner
///   state    "DRIVE", "SPIN", "LOOK" or "DONE"
///   goal     the goal heading of the last step (fixed frame, rad)
///   scan     the last scan: its image-bottom sectors for the planner, the side of the recovery turn
///   recover  fixed-frame heading (rad) of the recovery turn in progress, or None
///   fails    blocks on the current target
#[pyclass(name = "FrontierExplore", module = "navette")]
pub struct PyFrontierExplore {
    inner: FrontierExplore,
    /// The planner as a Python object. It holds the planner between calls; `with` lends it to `inner`.
    planner: Py<PyFrontierPlanner>,
}

impl PyFrontierExplore {
    /// Run `f` on the behaviour, its planner in place, without the GIL (see lib.rs). The planner object
    /// stays mutably borrowed meanwhile.
    fn with<R: Send>(&mut self, py: Python<'_>, f: impl FnOnce(&mut FrontierExplore) -> R + Send) -> R {
        let mut planner = self.planner.borrow_mut(py);
        std::mem::swap(&mut self.inner.goals.planner, &mut planner.inner);
        let inner = &mut self.inner;
        let r = py.detach(|| f(inner));
        std::mem::swap(&mut self.inner.goals.planner, &mut planner.inner);
        r
    }
}

#[pymethods]
impl PyFrontierExplore {
    #[new]
    #[pyo3(signature = (p=None, safety=None, fp=None))]
    fn new(
        py: Python<'_>,
        p: Option<PyRef<'_, PyExploreParams>>,
        safety: Option<PyRef<'_, PySafetyParams>>,
        fp: Option<PyRef<'_, PyFrontierParams>>,
    ) -> PyResult<Self> {
        let default = ExploreParams {
            rotate_in_place: true,
            ..Default::default()
        };
        let safety = safety_params(safety)?;
        let fp = frontier_params(fp)?;
        Ok(Self {
            inner: FrontierExplore::new(explore_params(p, default)?, safety, fp).map_err(param_err)?,
            planner: Py::new(
                py,
                PyFrontierPlanner {
                    inner: FrontierPlanner::new(fp).map_err(param_err)?,
                },
            )?,
        })
    }

    #[getter]
    fn p(&self) -> PyExploreParams {
        PyExploreParams::from_core(&self.inner.controller.params)
    }

    #[getter]
    fn planner(&self, py: Python<'_>) -> Py<PyFrontierPlanner> {
        self.planner.clone_ref(py)
    }

    #[getter]
    fn state(&self) -> &'static str {
        self.inner.state().as_str()
    }

    #[getter]
    fn goal(&self) -> Option<f64> {
        self.inner.controller.goal
    }

    #[getter]
    fn scan(&self, py: Python<'_>) -> Option<PyScan> {
        self.inner.goals.scan.as_ref().map(|s| PyScan::from_core(py, s))
    }

    /// The scan is copied.
    #[setter]
    fn set_scan(&mut self, py: Python<'_>, scan: Option<PyRef<'_, PyScan>>) -> PyResult<()> {
        self.inner.goals.scan = scan.map(|s| s.to_core(py)).transpose()?;
        Ok(())
    }

    #[getter]
    fn recover(&self) -> Option<f64> {
        self.inner.goals.recover
    }

    #[setter]
    fn set_recover(&mut self, recover: Option<f64>) {
        self.inner.goals.recover = recover;
    }

    #[getter]
    fn fails(&self) -> u32 {
        self.inner.goals.fails()
    }

    /// Plan from pose, with the image-bottom obstacles of the last scan.
    #[pyo3(name = "_plan")]
    fn plan(&mut self, py: Python<'_>, t: f64, pose: [f64; 3], grid: PyRef<'_, PyOccupancyGrid>) {
        let grid = &grid.inner;
        self.with(py, |fe| fe.goals.plan(t, pose2(pose), grid));
    }

    /// The goal heading (fixed frame, rad): the recovery turn in progress, else along the planner's path
    /// (planning first if there is none).
    fn choose_goal(
        &mut self,
        py: Python<'_>,
        t: f64,
        pose: [f64; 3],
        grid: PyRef<'_, PyOccupancyGrid>,
    ) -> f64 {
        let grid = &grid.inner;
        self.with(py, |fe| fe.goals.choose_goal(t, pose2(pose), grid))
    }

    /// The way is blocked: start the recovery turn; the second block on the same target bans it. With
    /// ExploreParams.recover_replan the recovery heading is chosen by the next choose_goal (after a replan).
    fn on_blocked(&mut self, py: Python<'_>, t: f64, pose: [f64; 3]) {
        self.with(py, |fe| fe.goals.on_blocked(t, pose2(pose)));
    }

    /// Requested (v, w) at time t (s) from pose (x, y, theta). scan: this frame's Scan (None: the last one
    /// is kept); grid: the OccupancyGrid; v_applied: the speed command sent last step (m/s); near_known:
    /// per-sector bool of this scan, blind segment explained by the map (used with ExploreParams.near_known);
    /// w_meas: the measured yaw rate (rad/s), for the end of the turns in place (ExploreParams.spin_lead_s).
    #[pyo3(signature = (t, pose, scan, grid, v_applied, near_known=None, w_meas=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        t: f64,
        pose: [f64; 3],
        scan: Option<PyRef<'_, PyScan>>,
        grid: PyRef<'_, PyOccupancyGrid>,
        v_applied: f64,
        near_known: Option<PyReadonlyArray1<'_, bool>>,
        w_meas: f64,
    ) -> PyResult<(f64, f64)> {
        let scan = scan.map(|s| s.to_core(py)).transpose()?;
        let near_known = match (&scan, near_known) {
            (Some(s), mask) => sector_mask(mask, s.len(), "near_known")?,
            (None, Some(_)) => {
                return Err(PyValueError::new_err(
                    "near_known is per sector of a scan: give the scan",
                ))
            }
            (None, None) => None,
        };
        let grid = &grid.inner;
        Ok(self.with(py, |fe| {
            fe.step(
                t,
                pose2(pose),
                scan.as_ref(),
                grid,
                v_applied,
                w_meas,
                near_known.as_deref(),
            )
        }))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFrontierParams>()?;
    m.add_class::<PyFrontierPlanner>()?;
    m.add_class::<PyFrontierExplore>()?;
    m.add_function(wrap_pyfunction!(frontier_cells, m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform, m)?)?;
    m.add_function(wrap_pyfunction!(dijkstra, m)?)?;
    Ok(())
}
