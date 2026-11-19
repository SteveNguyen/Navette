//! Bindings of navette_core::navigator: `Navigator`, `NavOut`.

use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use navette_core::geometry::Pose2;
use navette_core::navigator::{Behaviour, NavOut, Navigator, NavigatorParams};
use navette_core::occupancy_grid::{OccupancyGrid, OccupancyGridParams};
use navette_core::scan::Scan;

use crate::collision_monitor::{safety_params, PySafetyParams};
use crate::controller::{explore_params, PyExploreParams};
use crate::convert::{param_err, rows_to_py};
use crate::exploration::{frontier_params, PyFrontierParams};
use crate::occupancy_grid::PyOccupancyGrid;
use crate::scan::{PyFreeSpace, PyScan};
use crate::state_estimation::PyTiltTrust;

fn pose2(p: [f64; 3]) -> Pose2 {
    Pose2::new(p[0], p[1], p[2])
}

/// The result of one Navigator step (plain attributes, like a dataclass):
///   v, w          what to send: m/s and rad/s (> 0 counter-clockwise), after the collision monitor and the
///                 resume gate
///   v_req, w_req  what the behaviour asked for
///   state         the behaviour's state: "DRIVE", "SPIN", "LOOK", "DONE" or "FORWARD"
///   finished      exploration is over: DONE for 3 s
///   blind_stop    forward motion is stopped only by the never-seen blind zone below the image (a look down
///                 at it, Navigator.observe, would help)
#[pyclass(name = "NavOut", module = "navette", get_all, set_all)]
pub struct PyNavOut {
    v: f64,
    w: f64,
    v_req: f64,
    w_req: f64,
    state: String,
    finished: bool,
    blind_stop: bool,
}

impl From<NavOut> for PyNavOut {
    fn from(o: NavOut) -> Self {
        Self {
            v: o.v,
            w: o.w,
            v_req: o.v_req,
            w_req: o.w_req,
            state: o.state.as_str().to_owned(),
            finished: o.finished,
            blind_stop: o.blind_stop,
        }
    }
}

#[pymethods]
impl PyNavOut {
    #[new]
    #[pyo3(signature = (v, w, v_req, w_req, state, finished, blind_stop=false))]
    fn new(v: f64, w: f64, v_req: f64, w_req: f64, state: String, finished: bool, blind_stop: bool) -> Self {
        Self {
            v,
            w,
            v_req,
            w_req,
            state,
            finished,
            blind_stop,
        }
    }

    fn __repr__(&self) -> String {
        let py_bool = |b| if b { "True" } else { "False" };
        format!(
            "NavOut(v={:?}, w={:?}, v_req={:?}, w_req={:?}, state={:?}, finished={}, blind_stop={})",
            self.v,
            self.w,
            self.v_req,
            self.w_req,
            self.state,
            py_bool(self.finished),
            py_bool(self.blind_stop)
        )
    }
}

/// One navigation step per frame: occupancy grid -> behaviour -> collision monitor -> resume gate (see the
/// core module's documentation).
///   safety      SafetyParams (default: SafetyParams())
///   behaviour   "explore": frontier exploration; "forward": a constant straight request of forward_v (m/s)
///               through the collision monitor (dry runs, stop tests)
///   odometry    False when the pose does not move with the robot (no base, a carried dry run): the map
///               would smear, so "forward" then uses each frame's scan alone ("explore" always uses the map)
///   resume_n    steps allowing forward motion needed to resume after a stop
///   grid        the OccupancyGrid to map into (default: OccupancyGrid()); it is the `grid` attribute
///   frontier    FrontierParams (default: FrontierParams())
///   explore     ExploreParams (default: ExploreParams(rotate_in_place=True))
///   tilt_trust  a TiltTrust whose parameters the navigator's own takes (default: TiltTrust())
/// Attributes: grid; frontier (N, 2), path (N, 2) or None, target (2,) or None: the planner's frontier cell
/// centres, path and target in the fixed frame, for display; safety, behaviour, odometry, resume_n,
/// forward_v as given.
/// ValueError if a parameter is invalid (the parameter objects are checked again here: their attributes can
/// be set after they were built).
/// Threads: step, step_mask and observe release the GIL while they work (a plan on a large map takes a
/// while; other threads run meanwhile). One thread per navigator: using it or its grid from another thread
/// during such a call raises RuntimeError.
#[pyclass(name = "Navigator", module = "navette")]
pub struct PyNavigator {
    inner: Navigator,
    /// The map as a Python object. It holds the grid between calls; `with` lends it to `inner`, which keeps
    /// a one-cell grid meanwhile.
    grid: Py<PyOccupancyGrid>,
}

impl PyNavigator {
    /// Run `f` on the navigator, its grid in place, without the GIL (see lib.rs). The grid object stays
    /// mutably borrowed meanwhile: it holds the stand-in grid, which nobody must read.
    fn with<R: Send>(&mut self, py: Python<'_>, f: impl FnOnce(&mut Navigator) -> R + Send) -> R {
        let mut grid = self.grid.borrow_mut(py);
        std::mem::swap(self.inner.grid_mut(), &mut grid.inner);
        let inner = &mut self.inner;
        let r = py.detach(|| f(inner));
        std::mem::swap(self.inner.grid_mut(), &mut grid.inner);
        r
    }
}

/// The scan as the core type; the map needs two sectors to know their width.
fn core_scan(py: Python<'_>, scan: &PyScan) -> PyResult<Scan> {
    let scan = scan.to_core(py)?;
    if scan.len() < 2 {
        return Err(PyValueError::new_err("the scan needs at least two sectors"));
    }
    Ok(scan)
}

#[pymethods]
impl PyNavigator {
    #[new]
    #[pyo3(signature = (safety=None, behaviour="explore", odometry=true, resume_n=2, forward_v=0.15, grid=None,
                        frontier=None, explore=None, tilt_trust=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        safety: Option<PyRef<'_, PySafetyParams>>,
        behaviour: &str,
        odometry: bool,
        resume_n: u32,
        forward_v: f64,
        grid: Option<Py<PyOccupancyGrid>>,
        frontier: Option<PyRef<'_, PyFrontierParams>>,
        explore: Option<PyRef<'_, PyExploreParams>>,
        tilt_trust: Option<PyRef<'_, PyTiltTrust>>,
    ) -> PyResult<Self> {
        let behaviour = match behaviour {
            "explore" => Behaviour::Explore,
            "forward" => Behaviour::Forward,
            other => {
                return Err(PyValueError::new_err(format!(
                    "behaviour must be 'explore' or 'forward', got {other:?}"
                )))
            }
        };
        let grid = match grid {
            Some(grid) => grid,
            None => Py::new(
                py,
                PyOccupancyGrid {
                    inner: OccupancyGrid::new(OccupancyGridParams::default()).map_err(param_err)?,
                },
            )?,
        };
        let defaults = NavigatorParams::default();
        let grid_params = *grid.borrow(py).inner.params();
        let params = NavigatorParams {
            safety: safety_params(safety)?,
            behaviour,
            odometry,
            resume_n,
            forward_v,
            // the navigator's own grid only stands in for the Python object's between calls: one cell
            grid: OccupancyGridParams {
                size_m: grid_params.res,
                ..grid_params
            },
            frontier: frontier_params(frontier)?,
            explore: explore_params(explore, defaults.explore)?,
            tilt_trust: tilt_trust.map_or(defaults.tilt_trust, |t| {
                let t = &t.inner;
                navette_core::state_estimation::tilt_trust::TiltTrust {
                    motion: 0.0,
                    sigma: t.s0,
                    ..*t
                }
            }),
        };
        Ok(Self {
            inner: Navigator::new(params).map_err(param_err)?,
            grid,
        })
    }

    #[getter]
    fn grid(&self, py: Python<'_>) -> Py<PyOccupancyGrid> {
        self.grid.clone_ref(py)
    }

    #[getter]
    fn frontier<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        rows_to_py(py, self.inner.frontier())
    }

    #[getter]
    fn path<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        self.inner.path().map(|p| rows_to_py(py, p)).transpose()
    }

    #[getter]
    fn target<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.target().map(|t| PyArray1::from_slice(py, &t))
    }

    #[getter]
    fn safety(&self) -> PySafetyParams {
        PySafetyParams::from_core(&self.inner.params().safety)
    }

    #[getter]
    fn behaviour(&self) -> &'static str {
        match self.inner.params().behaviour {
            Behaviour::Explore => "explore",
            Behaviour::Forward => "forward",
        }
    }

    #[getter]
    fn odometry(&self) -> bool {
        self.inner.params().odometry
    }

    #[getter]
    fn resume_n(&self) -> u32 {
        self.inner.params().resume_n
    }

    #[getter]
    fn forward_v(&self) -> f64 {
        self.inner.params().forward_v
    }

    /// One step -> NavOut. t time (s), dt since the last step; pose (x, y, theta) in the fixed frame; scan:
    /// this frame's Scan; cam_h camera height (m); a, v, w measured base motion (forward acceleration m/s^2,
    /// speed m/s, yaw rate rad/s); v_prev the speed command sent last step.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        t: f64,
        dt: f64,
        pose: [f64; 3],
        scan: PyRef<'_, PyScan>,
        cam_h: f64,
        a: f64,
        v: f64,
        w: f64,
        v_prev: f64,
    ) -> PyResult<PyNavOut> {
        let scan = core_scan(py, &scan)?;
        Ok(self
            .with(py, |nav| {
                nav.step(t, dt, pose2(pose), &scan, cam_h, a, v, w, v_prev)
            })
            .into())
    }

    /// step() from the floor mask -> (NavOut, Scan). fs: the FreeSpace that turns the mask into the scan;
    /// floor: bool mask (h, w), True = floor; up_c: 'up' in the camera frame; then as step().
    #[allow(clippy::too_many_arguments)]
    fn step_mask(
        &mut self,
        py: Python<'_>,
        mut fs: PyRefMut<'_, PyFreeSpace>,
        floor: PyReadonlyArray2<'_, bool>,
        up_c: [f64; 3],
        t: f64,
        dt: f64,
        pose: [f64; 3],
        cam_h: f64,
        a: f64,
        v: f64,
        w: f64,
        v_prev: f64,
    ) -> PyResult<(PyNavOut, PyScan)> {
        let mask = fs.mask(&floor)?;
        let fs_inner = &mut fs.inner;
        let (out, scan) = self
            .with(py, |nav| {
                nav.step_mask(fs_inner, &mask, &up_c, t, dt, pose2(pose), cam_h, a, v, w, v_prev)
            })
            .map_err(|e| PyValueError::new_err(e.0))?;
        Ok((out.into(), PyScan::from_core(py, &scan)))
    }

    /// A head-glance frame (robot holding still, head looking down at the blind zone): into the map only,
    /// always as a keyframe; no behaviour, no command.
    fn observe(
        &mut self,
        py: Python<'_>,
        t: f64,
        pose: [f64; 3],
        scan: PyRef<'_, PyScan>,
        cam_h: f64,
    ) -> PyResult<()> {
        let scan = core_scan(py, &scan)?;
        self.with(py, |nav| nav.observe(t, pose2(pose), &scan, cam_h));
        Ok(())
    }

    /// Forget the map and the behaviour's state: the robot was moved by hand.
    fn reset(&mut self, py: Python<'_>) {
        self.with(py, Navigator::reset);
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNavOut>()?;
    m.add_class::<PyNavigator>()?;
    Ok(())
}
