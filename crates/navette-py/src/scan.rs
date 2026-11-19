//! Bindings of navette_core::scan as a numpy API: `FreeSpace`, `Scan`,
//! `drive_mask`, `seg_probs`.

use numpy::{
    Element, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use navette_core::scan::{self, FreeSpace, Scan, ScanParams, ShapeError};

use crate::convert::rows;

fn shape_err(e: ShapeError) -> PyErr {
    PyValueError::new_err(e.0)
}

/// `numpy.asarray(obj, dtype=T)` as a 1-D array: no copy if `obj` already is one.
fn as_array1<'py, T: Element>(obj: &Bound<'py, PyAny>) -> PyResult<Py<PyArray1<T>>> {
    let py = obj.py();
    let a = py
        .import("numpy")?
        .call_method1("asarray", (obj, T::get_dtype(py)))?;
    Ok(a.cast_into::<PyArray1<T>>()?.unbind())
}

fn to_vec<T: Element + Copy>(py: Python<'_>, a: &Py<PyArray1<T>>) -> Vec<T> {
    a.bind(py).readonly().as_array().to_vec()
}

/// Per-bearing scan (one entry per sector):
///   bearing_deg  sector centres (deg), 0 = forward, positive = left
///   seen         some image pixel looks at the floor in this bearing (False = outside the view: unknown)
///   obstacle     distance to the nearest obstacle (m); inf if none within max_range
///   near         the obstacle covers the closest point the camera sees in this bearing
///   floor        farthest floor point seen (m), capped at max_range
///   closest      closest floor point the camera sees (m, inf if not seen); default: inf everywhere
///
/// Like a dataclass, it holds the numpy arrays it is given (no copy when the dtype already is
/// float64 / bool), so they can be modified in place: `scan.near[k] = True`.
#[pyclass(name = "Scan", module = "navette")]
pub struct PyScan {
    #[pyo3(get, set)]
    bearing_deg: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    seen: Py<PyArray1<bool>>,
    #[pyo3(get, set)]
    obstacle: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    near: Py<PyArray1<bool>>,
    #[pyo3(get, set)]
    floor: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    closest: Py<PyArray1<f64>>,
}

impl PyScan {
    pub fn from_core(py: Python<'_>, s: &Scan) -> Self {
        Self {
            bearing_deg: PyArray1::from_slice(py, &s.bearing_deg).unbind(),
            seen: PyArray1::from_slice(py, &s.seen).unbind(),
            obstacle: PyArray1::from_slice(py, &s.obstacle).unbind(),
            near: PyArray1::from_slice(py, &s.near).unbind(),
            floor: PyArray1::from_slice(py, &s.floor).unbind(),
            closest: PyArray1::from_slice(py, &s.closest).unbind(),
        }
    }

    /// Copy into the core type (what the Rust side of every other binding consumes).
    pub fn to_core(&self, py: Python<'_>) -> PyResult<Scan> {
        let s = Scan {
            bearing_deg: to_vec(py, &self.bearing_deg),
            seen: to_vec(py, &self.seen),
            obstacle: to_vec(py, &self.obstacle),
            near: to_vec(py, &self.near),
            floor: to_vec(py, &self.floor),
            closest: to_vec(py, &self.closest),
        };
        let n = s.len();
        if [
            s.seen.len(),
            s.obstacle.len(),
            s.near.len(),
            s.floor.len(),
            s.closest.len(),
        ] != [n; 5]
        {
            return Err(PyValueError::new_err("Scan arrays must all have the same length"));
        }
        Ok(s)
    }
}

#[pymethods]
impl PyScan {
    #[new]
    #[pyo3(signature = (bearing_deg, seen, obstacle, near, floor, closest=None))]
    fn new(
        bearing_deg: &Bound<'_, PyAny>,
        seen: &Bound<'_, PyAny>,
        obstacle: &Bound<'_, PyAny>,
        near: &Bound<'_, PyAny>,
        floor: &Bound<'_, PyAny>,
        closest: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let py = bearing_deg.py();
        let bearing_deg = as_array1::<f64>(bearing_deg)?;
        let closest = match closest {
            Some(c) => as_array1(c)?,
            // not given: inf = "no floor point seen", the safe reading (the whole range is blind)
            None => PyArray1::from_vec(py, vec![f64::INFINITY; bearing_deg.bind(py).len()]).unbind(),
        };
        Ok(Self {
            bearing_deg,
            seen: as_array1(seen)?,
            obstacle: as_array1(obstacle)?,
            near: as_array1(near)?,
            floor: as_array1(floor)?,
            closest,
        })
    }

    /// Usable free distance = min(obstacle, floor); nan where not seen.
    fn free<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray1::from_vec(py, self.to_core(py)?.free()))
    }
}

/// Floor mask -> Scan. rays, rays_low: (h*w, 3) viewing rays of the mask pixels, through the pixel centre and
/// through its lower edge (navette.rays.mask_rays with dv = 0 and dv = 0.5); mask_shape = (h, w).
/// min_blob_px: obstacle blobs smaller than this (mask pixels) are ignored as noise.
/// near_tol: an obstacle within near_tol (m) of the closest visible point counts as `near`.
#[pyclass(name = "FreeSpace", module = "navette")]
pub struct PyFreeSpace {
    pub(crate) inner: FreeSpace,
}

impl PyFreeSpace {
    /// The mask as a row-major slice: the array's own memory if it is C-contiguous, else a copy in logical
    /// (row-major) order. `as_slice` alone also accepts a Fortran-contiguous array, whose memory is
    /// column-major: same shape and values, but read as another picture.
    pub(crate) fn mask<'a>(
        &self,
        floor: &'a PyReadonlyArray2<'_, bool>,
    ) -> PyResult<std::borrow::Cow<'a, [bool]>> {
        let (h, w) = self.inner.shape();
        if floor.shape() != [h, w] {
            return Err(PyValueError::new_err(format!(
                "floor mask has shape {:?}, expected ({h}, {w})",
                floor.shape()
            )));
        }
        Ok(match floor.as_slice() {
            Ok(s) if floor.is_c_contiguous() => s.into(),
            _ => floor.as_array().iter().copied().collect(),
        })
    }
}

#[pymethods]
impl PyFreeSpace {
    #[new]
    #[pyo3(signature = (rays, rays_low, mask_shape=scan::MASK_SHAPE, sector_deg=2.0, max_range=2.5, min_blob_px=10,
                        near_tol=0.05))]
    fn new(
        rays: PyReadonlyArray2<'_, f64>,
        rays_low: PyReadonlyArray2<'_, f64>,
        mask_shape: (usize, usize),
        sector_deg: f64,
        max_range: f64,
        min_blob_px: usize,
        near_tol: f64,
    ) -> PyResult<Self> {
        let params = ScanParams {
            sector_deg,
            max_range,
            min_blob_px,
            near_tol,
        };
        let inner = FreeSpace::new(
            rows(&rays, "rays")?,
            rows(&rays_low, "rays_low")?,
            mask_shape,
            params,
        )
        .map_err(shape_err)?;
        Ok(Self { inner })
    }

    /// Number of sectors.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Sector centres (deg).
    #[getter]
    fn bearing_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.bearing_deg())
    }

    /// Mask shape (h, w).
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn sector_deg(&self) -> f64 {
        self.inner.params().sector_deg
    }

    #[getter]
    fn max_range(&self) -> f64 {
        self.inner.params().max_range
    }

    #[getter]
    fn min_blob_px(&self) -> usize {
        self.inner.params().min_blob_px
    }

    #[getter]
    fn near_tol(&self) -> f64 {
        self.inner.params().near_tol
    }

    /// Non-floor pixels without the blobs (8-connected) smaller than min_blob_px.
    fn obstacles<'py>(
        &mut self,
        py: Python<'py>,
        floor: PyReadonlyArray2<'py, bool>,
    ) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let mask = self.mask(&floor)?;
        let (h, w) = self.inner.shape();
        PyArray1::from_slice(py, self.inner.obstacles(&mask).map_err(shape_err)?).reshape([h, w])
    }

    /// Floor range (nan for rays at or above the horizon) and bearing sector of gravity-frame rays (N, 3).
    #[allow(clippy::type_complexity)]
    fn project<'py>(
        &self,
        py: Python<'py>,
        rays_g: PyReadonlyArray2<'py, f64>,
        h: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
        let (r, sec) = self.inner.project(&rows(&rays_g, "rays_g")?, h);
        let sec = sec.into_iter().map(|s| s as i64).collect();
        Ok((PyArray1::from_vec(py, r), PyArray1::from_vec(py, sec)))
    }

    /// floor: bool mask (h, w), True = floor. up_c: 'up' in the camera frame. h: camera height (m).
    fn scan(
        &mut self,
        py: Python<'_>,
        floor: PyReadonlyArray2<'_, bool>,
        up_c: [f64; 3],
        h: f64,
    ) -> PyResult<PyScan> {
        let mask = self.mask(&floor)?;
        let s = self.inner.scan(&mask, &up_c, h).map_err(shape_err)?;
        Ok(PyScan::from_core(py, &s))
    }
}

/// Pixels the robot may drive on: floor (p_floor > thr), minus floor with something low above it
/// (p_under > under_thr). under_thr None, or a floor-only model (p_under None): floor alone.
/// p_floor, p_under: float32 arrays of the same shape.
#[pyfunction]
#[pyo3(signature = (p_floor, p_under, thr, under_thr))]
fn drive_mask<'py>(
    py: Python<'py>,
    p_floor: PyReadonlyArrayDyn<'py, f32>,
    p_under: Option<PyReadonlyArrayDyn<'py, f32>>,
    thr: f64,
    under_thr: Option<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let pf: Vec<f32> = p_floor.as_array().iter().copied().collect();
    let pu: Option<Vec<f32>> = match &p_under {
        Some(u) if u.shape() != p_floor.shape() => {
            return Err(PyValueError::new_err(
                "p_floor and p_under must have the same shape",
            ));
        }
        Some(u) => Some(u.as_array().iter().copied().collect()),
        None => None,
    };
    // thresholds to f32, as numpy does when comparing a float32 array with a Python float
    let m = scan::drive_mask(&pf, pu.as_deref(), thr as f32, under_thr.map(|t| t as f32));
    PyArray1::from_vec(py, m).reshape(p_floor.shape())
}

/// Floor model logits (C, H, W) float32 -> (floor prob, under prob or None). Floor = softmax over channels 0-1
/// only; a 3-channel model's channel 2 is the 'under' logit, read with a sigmoid.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn seg_probs<'py>(
    py: Python<'py>,
    z: PyReadonlyArray3<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Option<Bound<'py, PyArray2<f32>>>)> {
    let [c, h, w] = *z.shape() else { unreachable!() };
    if c < 2 {
        return Err(PyValueError::new_err("logits must have at least 2 channels"));
    }
    let flat: Vec<f32> = z.as_array().iter().copied().collect();
    let (floor, under) = scan::seg_probs(&flat, h * w);
    let under = match under {
        Some(u) => Some(PyArray1::from_vec(py, u).reshape([h, w])?),
        None => None,
    };
    Ok((PyArray1::from_vec(py, floor).reshape([h, w])?, under))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScan>()?;
    m.add_class::<PyFreeSpace>()?;
    m.add_function(wrap_pyfunction!(drive_mask, m)?)?;
    m.add_function(wrap_pyfunction!(seg_probs, m)?)?;
    Ok(())
}
