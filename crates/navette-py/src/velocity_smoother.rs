//! Bindings of navette_core::velocity_smoother: `ResumeGate`.

use pyo3::prelude::*;

use navette_core::velocity_smoother::ResumeGate;

/// Stop at once, resume only after n consecutive steps allowing forward motion (anti-chatter, after the
/// collision monitor). Call it once per step: `v = gate(v_req, v_ok)`.
///   n        steps allowing forward motion needed to resume after a stop (1 = no gate)
///   stopped  the collision monitor cut a forward request to 0 and motion has not resumed since
///   clear    steps allowing forward motion since the stop
#[pyclass(name = "ResumeGate", module = "navette")]
pub struct PyResumeGate {
    inner: ResumeGate,
}

#[pymethods]
impl PyResumeGate {
    #[new]
    #[pyo3(signature = (n=2))]
    fn new(n: u32) -> Self {
        Self {
            inner: ResumeGate::new(n),
        }
    }

    #[getter]
    fn n(&self) -> u32 {
        self.inner.n
    }

    #[setter]
    fn set_n(&mut self, n: u32) {
        self.inner.n = n;
    }

    #[getter]
    fn stopped(&self) -> bool {
        self.inner.stopped()
    }

    #[getter]
    fn clear(&self) -> u32 {
        self.inner.clear()
    }

    /// v allowed after the gate. A stop = the collision monitor cut a forward request to 0.
    fn __call__(&mut self, v_req: f64, v_ok: f64) -> f64 {
        self.inner.step(v_req, v_ok)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResumeGate>()?;
    Ok(())
}
