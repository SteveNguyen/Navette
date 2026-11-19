//! Bindings of navette_core::occupancy_grid as a numpy API: `OccupancyGrid`.
//! A pose is an (x, y, theta) sequence, points are (N, 2) arrays, cells (N, 2) integer arrays.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use navette_core::geometry::{Point2, Pose2};
use navette_core::occupancy_grid::{OccupancyGrid, OccupancyGridParams};

use crate::convert::{param_err, rows, rows_to_py};
use crate::scan::PyScan;

fn pose2(p: [f64; 3]) -> Pose2 {
    Pose2::new(p[0], p[1], p[2])
}

/// Log-odds occupancy grid in the fixed frame, fed by scan keyframes (see the core module's documentation).
///   thr             a cell is obstacle if l > thr, free if l < -thr, unknown otherwise
///   clamp           bound of |l|: ~4 contrary keyframes flip a cell
///   sigma_ref       range error (m) at which evidence counts half
///   max_obstacle_m  obstacles at or beyond this range are not written
///   robot_radius    footprint marked free at each keyframe
///   key_m, key_deg, key_s   a keyframe is taken when the robot moved, turned or waited this much
///   max_keyframes   keyframes kept for replay(); beyond it the oldest are dropped (bounded memory: ~30 MB at
///                   4000). The grid keeps their evidence; only a later replay() rebuilds without them.
/// ValueError on an invalid parameter: res, size_m, clamp or sigma_ref <= 0, more than 4096 cells per side
/// (size_m / res), a negative threshold, range or keyframe distance, anything not finite.
#[pyclass(name = "OccupancyGrid", module = "navette")]
pub struct PyOccupancyGrid {
    pub(crate) inner: OccupancyGrid,
}

impl PyOccupancyGrid {
    /// The grid's origin, or the error of a frame query made before the first update.
    fn origin(&self) -> PyResult<Point2> {
        self.inner
            .origin()
            .ok_or_else(|| PyValueError::new_err("the grid has no origin before its first update"))
    }

    fn grid<'py, T: numpy::Element>(&self, py: Python<'py>, v: Vec<T>) -> PyResult<Bound<'py, PyArray2<T>>> {
        let n = self.inner.n();
        PyArray1::from_vec(py, v).reshape([n, n])
    }
}

#[pymethods]
impl PyOccupancyGrid {
    #[new]
    #[pyo3(signature = (res=0.05, size_m=24.0, clamp=1.5, thr=0.2, sigma_ref=0.10, max_obstacle_m=2.5,
                        robot_radius=0.13, key_m=0.1, key_deg=5.0, key_s=0.5, max_keyframes=4000))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        res: f64,
        size_m: f64,
        clamp: f64,
        thr: f64,
        sigma_ref: f64,
        max_obstacle_m: f64,
        robot_radius: f64,
        key_m: f64,
        key_deg: f64,
        key_s: f64,
        max_keyframes: usize,
    ) -> PyResult<Self> {
        let params = OccupancyGridParams {
            res,
            size_m,
            clamp,
            thr,
            sigma_ref,
            max_obstacle_m,
            robot_radius,
            key_m,
            key_deg,
            key_s,
            max_keyframes,
        };
        Ok(Self {
            inner: OccupancyGrid::new(params).map_err(param_err)?,
        })
    }

    /// Cell size (m).
    #[getter]
    fn res(&self) -> f64 {
        self.inner.params().res
    }

    /// Cells per side.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    #[getter]
    fn thr(&self) -> f64 {
        self.inner.params().thr
    }

    /// Most keyframes kept for replay().
    #[getter]
    fn max_keyframes(&self) -> usize {
        self.inner.params().max_keyframes
    }

    /// Fixed-frame x of the corner of cell [0, 0]; None before the first update.
    #[getter]
    fn x0(&self) -> Option<f64> {
        self.inner.origin().map(|o| o[0])
    }

    /// Fixed-frame y of the corner of cell [0, 0]; None before the first update.
    #[getter]
    fn y0(&self) -> Option<f64> {
        self.inner.origin().map(|o| o[1])
    }

    /// Log-odds (n, n) float32, a copy: [i, j] = cell at x = x0 + i res, y = y0 + j res.
    #[getter]
    fn log_odds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.grid(py, self.inner.log_odds().to_vec())
    }

    /// Poses (x, y, theta) of the keyframes, oldest first.
    #[getter]
    fn keyframes(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .keyframes()
            .iter()
            .map(|k| (k.pose.x, k.pose.y, k.pose.theta))
            .collect()
    }

    /// Replace the keyframe poses (one (x, y, theta) per keyframe), e.g. by corrected ones; replay() then
    /// rebuilds the grid from them.
    fn set_keyframe_poses(&mut self, poses: Vec<[f64; 3]>) -> PyResult<()> {
        let keyframes = self.inner.keyframes_mut();
        if poses.len() != keyframes.len() {
            return Err(PyValueError::new_err(format!(
                "{} poses for {} keyframes",
                poses.len(),
                keyframes.len()
            )));
        }
        for (k, p) in keyframes.iter_mut().zip(poses) {
            k.pose = pose2(p);
        }
        Ok(())
    }

    /// Cells (N, 2) int64 holding the fixed-frame points xy (N, 2); outside [0, n) for points outside the
    /// grid.
    fn cell<'py>(
        &self,
        py: Python<'py>,
        xy: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.origin()?;
        let pts: Vec<Point2> = rows(&xy, "xy")?;
        let ij: Vec<[i64; 2]> = pts.into_iter().map(|p| self.inner.cell(p)).collect();
        PyArray1::from_slice(py, ij.as_flattened()).reshape([ij.len(), 2])
    }

    /// Fixed-frame centres (N, 2) of the cells ij (N, 2) int64.
    fn centres<'py>(
        &self,
        py: Python<'py>,
        ij: PyReadonlyArray2<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.origin()?;
        let ij = ij.as_array();
        if ij.ncols() != 2 {
            return Err(PyValueError::new_err(format!(
                "ij must have shape (N, 2), got {:?}",
                ij.shape()
            )));
        }
        let c: Vec<Point2> = ij
            .rows()
            .into_iter()
            .map(|r| self.inner.centre([r[0], r[1]]))
            .collect();
        rows_to_py(py, &c)
    }

    /// Integrate the scan if it is a new keyframe (always with force: the frames of a head glance, standing
    /// still, see what the driving frames cannot). Returns True if it was.
    /// sigma: tilt error bound (rad), None = unweighted. h: camera height (m). max_range: trusted range (m),
    /// None = all. The scan is copied: changing it afterwards does not change the keyframe.
    #[pyo3(signature = (t, pose, scan, sigma=None, h=0.35, max_range=None, force=false))]
    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        py: Python<'_>,
        t: f64,
        pose: [f64; 3],
        scan: PyRef<'_, PyScan>,
        sigma: Option<f64>,
        h: f64,
        max_range: Option<f64>,
        force: bool,
    ) -> PyResult<bool> {
        let scan = scan.to_core(py)?;
        if scan.len() < 2 {
            return Err(PyValueError::new_err("the scan needs at least two sectors"));
        }
        let inner = &mut self.inner;
        // without the GIL (see lib.rs)
        Ok(py.detach(|| inner.update(t, pose2(pose), &scan, sigma, h, max_range, force)))
    }

    /// Replace the whole map (tests, tools): log_odds (n, n) float32, with the corner of cell [0, 0] at
    /// (x0, y0). The keyframes are forgotten.
    fn set_map(&mut self, x0: f64, y0: f64, log_odds: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let n = self.inner.n();
        if log_odds.as_array().dim() != (n, n) {
            return Err(PyValueError::new_err(format!(
                "log_odds must have shape ({n}, {n})"
            )));
        }
        let l: Vec<f32> = log_odds.as_array().iter().copied().collect();
        self.inner.set_map([x0, y0], &l);
        Ok(())
    }

    /// Rebuild the grid from the keyframes (after their poses were corrected).
    fn replay(&mut self, py: Python<'_>) {
        let inner = &mut self.inner;
        py.detach(|| inner.replay()); // without the GIL (see lib.rs)
    }

    /// Clear the map and forget the keyframes; the next update centres the grid again.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// (obstacle, free) boolean (n, n) grids; the rest is unknown.
    #[allow(clippy::type_complexity)]
    fn masks<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<bool>>)> {
        let (obstacle, free) = self.inner.masks();
        Ok((self.grid(py, obstacle)?, self.grid(py, free)?))
    }

    /// Per sector: True if it is a near sector (obstacle filling the image bottom) that the map explains: its
    /// blind segment, from the robot edge r0 (default: robot_radius) to the closest visible point, has been
    /// observed, and the map holds an obstacle on it, up to closest + explain_m. Call it BEFORE update() with
    /// the same scan, so the scan cannot explain itself.
    #[pyo3(signature = (pose, scan, r0=None, explain_m=0.1))]
    fn blind_known<'py>(
        &self,
        py: Python<'py>,
        pose: [f64; 3],
        scan: PyRef<'py, PyScan>,
        r0: Option<f64>,
        explain_m: f64,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let known = self
            .inner
            .blind_known(pose2(pose), &scan.to_core(py)?, r0, explain_m);
        Ok(PyArray1::from_vec(py, known))
    }

    /// Centres (N, 2) of the obstacle cells closer than max_range (m) to the robot, in the robot frame.
    #[pyo3(signature = (pose, max_range=1.5))]
    fn obstacle_points<'py>(
        &self,
        py: Python<'py>,
        pose: [f64; 3],
        max_range: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        rows_to_py(py, &self.inner.obstacle_points(pose2(pose), max_range))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOccupancyGrid>()?;
    Ok(())
}
