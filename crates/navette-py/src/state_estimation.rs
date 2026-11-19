//! Bindings of navette_core::state_estimation: `UpFilter` (attitude), `TiltTrust` (tilt_trust), `G`.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use navette_core::state_estimation::attitude::{UpFilter, UpFilterParams};
use navette_core::state_estimation::tilt_trust::TiltTrust;
use navette_core::state_estimation::G;

/// Gravity 'up' in IMU axes from gyro + accelerometer samples (complementary filter on the gravity vector,
/// with gyro bias tracking).
///   tau_s        time constant of the pull toward the accelerometer
///   bias_tau_s   the bias follows the accelerometer correction with the time constant tau_s * bias_tau_s
///   acc_gate     the correction gain falls linearly to 0 when ||a| / g - 1| reaches acc_gate
///   g            |a| at rest (m/s^2)
///   history_s    history kept for up_at
///   bias0        initial gyro bias (rad/s)
///   still_*      still detection (rate rad/s, relative |a| error, duration) and its bias time constant
///   rate_gate    (rad/s, None = off) the correction gain also falls linearly to 0 as the peak-held
///                rotation rate (decay rate_tau_s) reaches rate_gate
/// State: u ('up', None before the first sample), b (gyro bias, rad/s), t (last sample time).
#[pyclass(name = "UpFilter", module = "navette")]
pub struct PyUpFilter {
    inner: UpFilter,
}

#[pymethods]
impl PyUpFilter {
    #[new]
    #[pyo3(signature = (tau_s=1.0, bias_tau_s=20.0, acc_gate=0.1, g=G, history_s=0.5, bias0=None,
                        still_rate=UpFilterParams::default().still_rate, still_acc=0.02, still_s=0.5,
                        still_bias_tau_s=1.0, rate_gate=None, rate_tau_s=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        tau_s: f64,
        bias_tau_s: f64,
        acc_gate: f64,
        g: f64,
        history_s: f64,
        bias0: Option<[f64; 3]>,
        still_rate: f64,
        still_acc: f64,
        still_s: f64,
        still_bias_tau_s: f64,
        rate_gate: Option<f64>,
        rate_tau_s: f64,
    ) -> Self {
        let params = UpFilterParams {
            tau_s,
            bias_tau_s,
            acc_gate,
            g,
            history_s,
            still_rate,
            still_acc,
            still_s,
            still_bias_tau_s,
            rate_gate,
            rate_tau_s,
        };
        Self {
            inner: UpFilter::new(params, bias0),
        }
    }

    /// One IMU sample at time t (s): gyro in rad/s, acc in m/s^2, both in IMU axes. Returns u.
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        gyro: [f64; 3],
        acc: [f64; 3],
    ) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.update(t, gyro, acc))
    }

    /// 'up' (IMU axes) at time t within the kept history, linearly interpolated; clamped at the ends.
    fn up_at<'py>(&self, py: Python<'py>, t: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.inner.up_at(t) {
            Some(u) => Ok(PyArray1::from_slice(py, &u)),
            None => Err(PyValueError::new_err("no IMU sample yet")),
        }
    }

    /// 'up' in IMU axes (a copy); None before the first sample.
    #[getter]
    fn u<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.u.map(|u| PyArray1::from_slice(py, &u))
    }

    #[setter]
    fn set_u(&mut self, u: Option<[f64; 3]>) {
        self.inner.u = u;
    }

    /// Gyro bias (rad/s), a copy.
    #[getter]
    fn b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.b)
    }

    #[setter]
    fn set_b(&mut self, b: [f64; 3]) {
        self.inner.b = b;
    }

    /// Time of the last sample.
    #[getter]
    fn t(&self) -> Option<f64> {
        self.inner.t()
    }

    /// Time the current still period started (None: not still).
    #[getter]
    fn still_since(&self) -> Option<f64> {
        self.inner.still_since()
    }

    /// Peak-held rotation rate (rad/s); stays 0 without a rate gate.
    #[getter]
    fn rate_lp(&self) -> f64 {
        self.inner.rate_lp()
    }

    /// Gain of the accelerometer correction: 1 / tau_s.
    #[getter]
    fn kp(&self) -> f64 {
        1.0 / self.inner.params.tau_s
    }

    /// Gain of the bias correction: 1 / (tau_s * bias_tau_s).
    #[getter]
    fn ki(&self) -> f64 {
        1.0 / (self.inner.params.tau_s * self.inner.params.bias_tau_s)
    }

    #[getter]
    fn acc_gate(&self) -> f64 {
        self.inner.params.acc_gate
    }

    #[getter]
    fn g(&self) -> f64 {
        self.inner.params.g
    }

    #[getter]
    fn history_s(&self) -> f64 {
        self.inner.params.history_s
    }

    #[getter]
    fn still_rate(&self) -> f64 {
        self.inner.params.still_rate
    }

    #[getter]
    fn still_acc(&self) -> f64 {
        self.inner.params.still_acc
    }

    #[getter]
    fn still_s(&self) -> f64 {
        self.inner.params.still_s
    }

    #[getter]
    fn still_bias_tau_s(&self) -> f64 {
        self.inner.params.still_bias_tau_s
    }

    #[getter]
    fn rate_gate(&self) -> Option<f64> {
        self.inner.params.rate_gate
    }

    #[getter]
    fn rate_tau_s(&self) -> f64 {
        self.inner.params.rate_tau_s
    }
}

/// How far the scan can be trusted, given how wrong the camera 'up' may be right now: a tilt error bound
/// from the base motion (sigma, rad) and the range it allows.
///   sigma0_deg  constant floor of the bound (calibration, noise)
///   tau_s       decay of the motion term (the 'up' filter's time constant)
///   tol_m       range error accepted at the trusted range
///   timing_s    frame / IMU timing error
/// State: s0 (rad), motion (rad, peak-held acceleration-induced bound), sigma (rad, current bound).
#[pyclass(name = "TiltTrust", module = "navette")]
pub struct PyTiltTrust {
    pub(crate) inner: TiltTrust,
}

#[pymethods]
impl PyTiltTrust {
    #[new]
    #[pyo3(signature = (sigma0_deg=0.5, tau_s=1.0, tol_m=0.10, timing_s=0.01))]
    fn new(sigma0_deg: f64, tau_s: f64, tol_m: f64, timing_s: f64) -> Self {
        Self {
            inner: TiltTrust::new(sigma0_deg, tau_s, tol_m, timing_s),
        }
    }

    /// a forward acceleration (m/s^2), v speed (m/s), w yaw rate (rad/s), gyro_rate head rotation rate
    /// (rad/s, default |w|). Returns the current tilt error bound (rad).
    #[pyo3(signature = (dt, a, v, w, gyro_rate=None))]
    fn update(&mut self, dt: f64, a: f64, v: f64, w: f64, gyro_rate: Option<f64>) -> f64 {
        self.inner.update(dt, a, v, w, gyro_rate)
    }

    /// Trusted range (m) for camera height h under the current bound.
    fn max_range(&self, h: f64) -> f64 {
        self.inner.max_range(h)
    }

    #[getter]
    fn s0(&self) -> f64 {
        self.inner.s0
    }

    #[getter]
    fn tau_s(&self) -> f64 {
        self.inner.tau_s
    }

    #[getter]
    fn tol_m(&self) -> f64 {
        self.inner.tol_m
    }

    #[getter]
    fn timing_s(&self) -> f64 {
        self.inner.timing_s
    }

    #[getter]
    fn motion(&self) -> f64 {
        self.inner.motion
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUpFilter>()?;
    m.add_class::<PyTiltTrust>()?;
    m.add("G", G)?;
    Ok(())
}
