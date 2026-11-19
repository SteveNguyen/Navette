//! Bindings of navette_core::collision_monitor as a numpy API: `SafetyParams`, `boundary_points`, `arc`,
//! `stopping_distance`, `free_arc_length`, `free_arc_lengths`, `safe_command`. Points are (N, 2) arrays in
//! the robot frame (x forward, y left); per-sector masks are bool arrays with one entry per scan sector.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use navette_core::collision_monitor::{self as core, SafetyParams};

use crate::convert::{rows, rows_to_py};
use crate::scan::PyScan;

/// Robot footprint and stopping model of the collision monitor (plain attributes, like a dataclass):
///   radius       m, robot footprint
///   margin       m, extra clearance
///   v_max        m/s, highest forward speed
///   decel        m/s^2, braking deceleration
///   latency      s, camera frame -> command applied
///   unseen_m     m, boundary put in sectors the camera does not see
///   reverse_max  m/s, allowed reverse speed (blind)
///   step         m, arc sampling
///   n_speeds     candidate speeds between 0 and the request
#[pyclass(name = "SafetyParams", module = "navette", get_all, set_all, eq)]
#[derive(PartialEq)]
pub struct PySafetyParams {
    radius: f64,
    margin: f64,
    v_max: f64,
    decel: f64,
    latency: f64,
    unseen_m: f64,
    reverse_max: f64,
    step: f64,
    n_speeds: usize,
}

impl PySafetyParams {
    /// Copy into the core type (what the Rust side of every other binding consumes).
    pub fn to_core(&self) -> SafetyParams {
        SafetyParams {
            radius: self.radius,
            margin: self.margin,
            v_max: self.v_max,
            decel: self.decel,
            latency: self.latency,
            unseen_m: self.unseen_m,
            reverse_max: self.reverse_max,
            step: self.step,
            n_speeds: self.n_speeds,
        }
    }
}

#[pymethods]
impl PySafetyParams {
    #[new]
    #[pyo3(signature = (radius=0.13, margin=0.05, v_max=0.4, decel=0.5, latency=0.15, unseen_m=0.35,
                        reverse_max=0.0, step=0.02, n_speeds=20))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        radius: f64,
        margin: f64,
        v_max: f64,
        decel: f64,
        latency: f64,
        unseen_m: f64,
        reverse_max: f64,
        step: f64,
        n_speeds: usize,
    ) -> Self {
        Self {
            radius,
            margin,
            v_max,
            decel,
            latency,
            unseen_m,
            reverse_max,
            step,
            n_speeds,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SafetyParams(radius={:?}, margin={:?}, v_max={:?}, decel={:?}, latency={:?}, unseen_m={:?}, \
             reverse_max={:?}, step={:?}, n_speeds={})",
            self.radius,
            self.margin,
            self.v_max,
            self.decel,
            self.latency,
            self.unseen_m,
            self.reverse_max,
            self.step,
            self.n_speeds
        )
    }
}

/// The given parameters, or the defaults.
fn params(p: Option<PyRef<'_, PySafetyParams>>) -> SafetyParams {
    p.map_or_else(SafetyParams::default, |p| p.to_core())
}

/// A per-sector bool array as a Vec, checked against the scan's number of sectors.
fn sector_mask(
    mask: Option<PyReadonlyArray1<'_, bool>>,
    n: usize,
    name: &str,
) -> PyResult<Option<Vec<bool>>> {
    let Some(mask) = mask else { return Ok(None) };
    let mask = mask.as_array().to_vec();
    if mask.len() != n {
        return Err(PyValueError::new_err(format!(
            "{name} has {} entries, the scan has {n} sectors",
            mask.len()
        )));
    }
    Ok(Some(mask))
}

/// (N, 2) points the robot must keep clear of, one per sector of a Scan.
/// near_m: where to put obstacles that run off the image bottom (`near`: the camera cannot see how close
/// they are). None = at the closest visible point (optimistic); the collision monitor puts them at the
/// clearance radius. near_mask: per-sector bool, only these near sectors get near_m (None = all).
#[pyfunction]
#[pyo3(signature = (scan, unseen_m, near_m=None, near_mask=None))]
fn boundary_points<'py>(
    py: Python<'py>,
    scan: PyRef<'py, PyScan>,
    unseen_m: f64,
    near_m: Option<f64>,
    near_mask: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let scan = scan.to_core(py)?;
    let near_mask = sector_mask(near_mask, scan.len(), "near_mask")?;
    rows_to_py(
        py,
        &core::boundary_points(&scan, unseen_m, near_m, near_mask.as_deref()),
    )
}

/// Robot centre (N, 2) after the arc lengths s (N,) on a path of curvature curv (1/m, + = left), from the
/// origin heading +x.
#[pyfunction]
fn arc<'py>(
    py: Python<'py>,
    curv: f64,
    s: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rows_to_py(py, &core::arc(curv, &s.as_array().to_vec()))
}

/// Distance travelled before standing still from speed v: v * latency + v^2 / (2 decel).
#[pyfunction]
fn stopping_distance(v: f64, p: PyRef<'_, PySafetyParams>) -> f64 {
    core::stopping_distance(v, &p.to_core())
}

/// Arc length the centre can travel at curvature curv before coming within radius + margin of a point of
/// `points` (N, 2), looking s_max ahead (default: the stopping distance at v_max + 0.1 m).
#[pyfunction]
#[pyo3(signature = (points, curv, p, s_max=None))]
fn free_arc_length(
    points: PyReadonlyArray2<'_, f64>,
    curv: f64,
    p: PyRef<'_, PySafetyParams>,
    s_max: Option<f64>,
) -> PyResult<f64> {
    Ok(core::free_arc_length(
        &rows(&points, "points")?,
        curv,
        &p.to_core(),
        s_max,
    ))
}

/// free_arc_length for several curvatures at once (same results).
#[pyfunction]
fn free_arc_lengths<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    curvs: Vec<f64>,
    p: PyRef<'py, PySafetyParams>,
    s_max: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let free = core::free_arc_lengths(&rows(&points, "points")?, &curvs, &p.to_core(), s_max);
    Ok(PyArray1::from_vec(py, free))
}

/// (v, w) requested -> (v, w) allowed, plus the free arc length at the allowed speed (m, for display).
/// p: SafetyParams (default: SafetyParams()).
/// extra_points: (N, 2) more points to keep clear of, robot frame (the occupancy grid's obstacles).
/// near_known: per-sector bool, blind segment already observed by the map: those near sectors are not put
/// at radius + margin. Pass it only with the same map's extra_points.
#[pyfunction]
#[pyo3(signature = (v, w, scan, p=None, extra_points=None, near_known=None))]
fn safe_command(
    py: Python<'_>,
    v: f64,
    w: f64,
    scan: PyRef<'_, PyScan>,
    p: Option<PyRef<'_, PySafetyParams>>,
    extra_points: Option<PyReadonlyArray2<'_, f64>>,
    near_known: Option<PyReadonlyArray1<'_, bool>>,
) -> PyResult<(f64, f64, f64)> {
    let scan = scan.to_core(py)?;
    let extra = match &extra_points {
        Some(e) => Some(rows(e, "extra_points")?),
        None => None,
    };
    let near_known = sector_mask(near_known, scan.len(), "near_known")?;
    Ok(core::safe_command(
        v,
        w,
        &scan,
        &params(p),
        extra.as_deref(),
        near_known.as_deref(),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySafetyParams>()?;
    m.add_function(wrap_pyfunction!(boundary_points, m)?)?;
    m.add_function(wrap_pyfunction!(arc, m)?)?;
    m.add_function(wrap_pyfunction!(stopping_distance, m)?)?;
    m.add_function(wrap_pyfunction!(free_arc_length, m)?)?;
    m.add_function(wrap_pyfunction!(free_arc_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(safe_command, m)?)?;
    Ok(())
}
