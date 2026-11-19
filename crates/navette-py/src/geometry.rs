//! Bindings of navette_core::geometry, with numpy conventions: points and rays are (N, 2) / (N, 3)
//! arrays, a pose is an (x, y, theta) sequence.

use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use navette_core::geometry::{self, Pose2};

use crate::convert::{rows, rows_to_py};

/// Rotation R_gc (rows = g axes expressed in the camera frame), so v_g = R_gc @ v_c.
#[pyfunction]
fn gravity_frame<'py>(py: Python<'py>, up_c: [f64; 3]) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rows_to_py(py, &geometry::gravity_frame(&up_c))
}

/// 'up' in the camera frame (OpenCV: y down, z forward) for a camera pitched down by pitch_down_deg, no roll.
#[pyfunction]
fn up_from_pitch<'py>(py: Python<'py>, pitch_down_deg: f64) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, &geometry::up_from_pitch(pitch_down_deg))
}

/// Floor points (N, 2) = (forward, left) in metres in the g frame, for camera-frame viewing rays (N, 3).
/// Rays that do not hit the floor (at or above the horizon) give NaN.
#[pyfunction]
fn rays_to_floor<'py>(
    py: Python<'py>,
    rays: PyReadonlyArray2<'py, f64>,
    up_c: [f64; 3],
    h: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rows_to_py(py, &geometry::rays_to_floor(&rows(&rays, "rays")?, &up_c, h))
}

/// Horizontal distance from the camera foot point to 3D points p_c (N, 3).
#[pyfunction]
fn horizontal_range<'py>(
    py: Python<'py>,
    p_c: PyReadonlyArray2<'py, f64>,
    up_c: [f64; 3],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(PyArray1::from_vec(
        py,
        geometry::horizontal_range(&rows(&p_c, "p_c")?, &up_c),
    ))
}

/// Wrap an angle (rad) to [-pi, pi).
#[pyfunction]
fn wrap_angle(a: f64) -> f64 {
    geometry::wrap_angle(a)
}

/// Robot-frame points (N, 2) -> fixed frame, for the robot pose (x, y, theta).
#[pyfunction]
fn to_fixed<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<'py, f64>,
    pose: [f64; 3],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pose = Pose2::new(pose[0], pose[1], pose[2]);
    let out: Vec<_> = rows(&pts, "pts")?.into_iter().map(|p| pose.to_fixed(p)).collect();
    rows_to_py(py, &out)
}

/// Fixed-frame points (N, 2) -> robot frame, for the robot pose (x, y, theta).
#[pyfunction]
fn to_robot<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<'py, f64>,
    pose: [f64; 3],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pose = Pose2::new(pose[0], pose[1], pose[2]);
    let out: Vec<_> = rows(&pts, "pts")?.into_iter().map(|p| pose.to_robot(p)).collect();
    rows_to_py(py, &out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gravity_frame, m)?)?;
    m.add_function(wrap_pyfunction!(up_from_pitch, m)?)?;
    m.add_function(wrap_pyfunction!(rays_to_floor, m)?)?;
    m.add_function(wrap_pyfunction!(horizontal_range, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_angle, m)?)?;
    m.add_function(wrap_pyfunction!(to_fixed, m)?)?;
    m.add_function(wrap_pyfunction!(to_robot, m)?)?;
    Ok(())
}
