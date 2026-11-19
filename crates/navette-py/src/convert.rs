//! numpy <-> core conversions shared by the binding files.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// (N, K) float64 array -> N rows of K values (K = 3: rays, points; K = 2: planar points).
pub fn rows<const K: usize>(a: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Vec<[f64; K]>> {
    if a.shape()[1] != K {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, {K}), got {:?}",
            a.shape()
        )));
    }
    Ok(a.as_array()
        .rows()
        .into_iter()
        .map(|r| std::array::from_fn(|k| r[k]))
        .collect())
}

/// N rows of K values -> (N, K) float64 array.
pub fn rows_to_py<'py, const K: usize>(
    py: Python<'py>,
    rows: &[[f64; K]],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    PyArray1::from_slice(py, rows.as_flattened()).reshape([rows.len(), K])
}
