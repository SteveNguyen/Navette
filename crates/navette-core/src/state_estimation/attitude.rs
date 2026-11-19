//! Gravity 'up' estimate from an IMU: gyro propagation + accelerometer correction.
//!
//! Only the tilt is needed for floor projection (heading is irrelevant), so the state is the unit 'up'
//! vector u in IMU axes (the accelerometer direction when still) plus a gyro bias b. Per IMU sample
//! (complementary / Mahony-style filter on the gravity vector):
//!
//! ```text
//! predict : u <- Rot(-(w - b) dt) u     a fixed world vector rotates opposite to the body, in body axes
//!                                       (full 3D: a head nod is not a single-axis rotation)
//! correct : e = u x a/|a|               rotation that would bring u onto the accelerometer direction
//!           u <- Rot(k e dt) u          k = 1/tau_s, reduced when |a| is far from g (linear
//!                                       acceleration, impacts)
//!           b <- b + e dt / (tau_s * bias_tau_s)
//!                                       slow gyro bias tracking (the bias about the vertical is
//!                                       unobservable and does not affect u)
//! still   : when |w - b| < still_rate and |a| ~ g for still_s, the gyro reads its bias directly:
//!           b <- b + (w - b) dt / still_bias_tau_s   (all 3 axes, ~1 s)
//! ```
//!
//! Measured on Reachy Mini's head IMU: gyro bias ~0.43 deg/s, stable to 0.02 deg/s between runs. Without
//! fast bias learning the error grows to ~bias * tau_s (3.5 deg at tau_s = 8). On a balancing base the head
//! is never perfectly still: the still update may then rarely fire, so pass `bias0` (measured at startup or
//! stored) when available.
//!
//! tau_s is the time constant of the pull toward the accelerometer: short = accelerometer errors under
//! motion leak in (accelerometer alone: up to 1.8 deg off when nodding at 1 Hz), long = gyro scale / bias
//! errors accumulate.
//!
//! Camera frames lag the IMU (~38 ms on Reachy Mini): use [`UpFilter::up_at`] at t_frame - latency.
//!
//! Same equations in the same order as the Python prototype, in f64. Squares are x * x where the prototype
//! calls libm's pow, so the rotation rate may differ by one ulp.

use std::collections::VecDeque;

use super::G;
use crate::geometry::Vec3;

/// `u` rotated by the rotation vector `r` (axis r/|r|, angle |r|), Rodrigues' formula:
/// u cos + (k x u) sin + k (k . u)(1 - cos), k = r / |r|.
fn rotate(r: Vec3, u: Vec3) -> Vec3 {
    let th = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    if th < 1e-15 {
        return u;
    }
    let k = [r[0] / th, r[1] / th, r[2] / th];
    let (c, s) = (th.cos(), th.sin());
    let d = (k[0] * u[0] + k[1] * u[1] + k[2] * u[2]) * (1.0 - c);
    [
        u[0] * c + (k[1] * u[2] - k[2] * u[1]) * s + k[0] * d,
        u[1] * c + (k[2] * u[0] - k[0] * u[2]) * s + k[1] * d,
        u[2] * c + (k[0] * u[1] - k[1] * u[0]) * s + k[2] * d,
    ]
}

#[inline]
fn norm(v: Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// [`UpFilter`] parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpFilterParams {
    /// Time constant (s) of the pull toward the accelerometer.
    pub tau_s: f64,
    /// The gyro bias follows the accelerometer correction with the time constant tau_s * bias_tau_s.
    pub bias_tau_s: f64,
    /// The correction gain falls linearly to 0 when | |a| / g - 1 | reaches acc_gate.
    pub acc_gate: f64,
    /// |a| at rest (m/s^2): the accelerometer's own reading of gravity (its scale may be ~1 % off).
    pub g: f64,
    /// Length (s) of the history kept for [`UpFilter::up_at`].
    pub history_s: f64,
    /// Still detection: rotation rate below this (rad/s) ...
    pub still_rate: f64,
    /// ... and | |a| / g - 1 | below this ...
    pub still_acc: f64,
    /// ... for this long (s).
    pub still_s: f64,
    /// Time constant (s) of the bias update when still.
    pub still_bias_tau_s: f64,
    /// (rad/s, None = off) The correction gain also falls linearly to 0 as the peak-held rotation rate
    /// reaches rate_gate. A shaking head has accelerometer errors whose magnitude stays near g; a good gyro
    /// (400 Hz stream) can carry the estimate meanwhile.
    pub rate_gate: Option<f64>,
    /// Decay time constant (s) of the peak-held rotation rate.
    pub rate_tau_s: f64,
}

impl Default for UpFilterParams {
    fn default() -> Self {
        Self {
            tau_s: 1.0,
            bias_tau_s: 20.0,
            acc_gate: 0.1,
            g: G,
            history_s: 0.5,
            still_rate: 2f64.to_radians(),
            still_acc: 0.02,
            still_s: 0.5,
            still_bias_tau_s: 1.0,
            rate_gate: None,
            rate_tau_s: 1.0,
        }
    }
}

/// 'Up' in IMU axes, from gyro + accelerometer samples. See the module documentation.
#[derive(Clone, Debug)]
pub struct UpFilter {
    pub params: UpFilterParams,
    /// 'Up' in IMU axes (unit vector); None before the first sample.
    pub u: Option<Vec3>,
    /// Gyro bias (rad/s).
    pub b: Vec3,
    /// Time of the last sample.
    t: Option<f64>,
    /// Time the current still period started.
    still_since: Option<f64>,
    /// Peak-held |w - b| (rad/s), only maintained with a rate gate.
    rate_lp: f64,
    /// (t, u) of the last history_s, for latency-compensated queries.
    hist: VecDeque<(f64, Vec3)>,
}

impl UpFilter {
    /// `bias0`: initial gyro bias (rad/s), zero if None.
    pub fn new(params: UpFilterParams, bias0: Option<Vec3>) -> Self {
        Self {
            params,
            u: None,
            b: bias0.unwrap_or([0.0; 3]),
            t: None,
            still_since: None,
            rate_lp: 0.0,
            hist: VecDeque::new(),
        }
    }

    /// Time of the last sample.
    pub fn t(&self) -> Option<f64> {
        self.t
    }

    /// Time the current still period started (None: not still).
    pub fn still_since(&self) -> Option<f64> {
        self.still_since
    }

    /// Peak-held rotation rate (rad/s); stays 0 without a rate gate.
    pub fn rate_lp(&self) -> f64 {
        self.rate_lp
    }

    /// One IMU sample at time `t` (s): `gyro` in rad/s, `acc` in m/s^2, both in IMU axes. Returns u.
    /// A sample not later than the previous one is ignored.
    pub fn update(&mut self, t: f64, gyro: Vec3, acc: Vec3) -> Vec3 {
        let p = &self.params;
        let a_norm = norm(acc);
        let (Some(mut u), Some(t_prev)) = (self.u, self.t) else {
            // start from the accelerometer
            let u = [acc[0] / a_norm, acc[1] / a_norm, acc[2] / a_norm];
            self.u = Some(u);
            self.t = Some(t);
            self.push(t, u);
            return u;
        };
        let dt = t - t_prev;
        self.t = Some(t);
        if dt <= 0.0 {
            return u;
        }
        let mut b = self.b;
        let acc_err = (a_norm / p.g - 1.0).abs();

        let rate = norm([gyro[0] - b[0], gyro[1] - b[1], gyro[2] - b[2]]); // |w - b|
        let still = rate < p.still_rate && acc_err < p.still_acc;
        if !still {
            self.still_since = None;
        } else if self.still_since.is_none() {
            self.still_since = Some(t);
        }
        if let Some(since) = self.still_since {
            if t - since >= p.still_s {
                let k = (dt / p.still_bias_tau_s).min(1.0);
                for j in 0..3 {
                    b[j] += (gyro[j] - b[j]) * k;
                }
            }
        }

        // predict
        let w = [gyro[0] - b[0], gyro[1] - b[1], gyro[2] - b[2]];
        u = rotate([-w[0] * dt, -w[1] * dt, -w[2] * dt], u);
        // correct
        let mut trust = (1.0 - acc_err / p.acc_gate).max(0.0);
        if let Some(rate_gate) = p.rate_gate {
            let r = norm(w);
            // peak hold: close at once when motion starts, reopen with rate_tau_s once it stops
            // (a plain low-pass let the first rate_tau_s of bad accelerometer into u and the bias)
            self.rate_lp = if r > self.rate_lp {
                r
            } else {
                self.rate_lp + (r - self.rate_lp) * (dt / p.rate_tau_s).min(1.0)
            };
            trust *= (1.0 - self.rate_lp / rate_gate).max(0.0);
        }
        if trust > 0.0 {
            let n = [acc[0] / a_norm, acc[1] / a_norm, acc[2] / a_norm];
            // e = u x a/|a|, |e| = sin(angle between u and the accelerometer)
            let e = [
                u[1] * n[2] - u[2] * n[1],
                u[2] * n[0] - u[0] * n[2],
                u[0] * n[1] - u[1] * n[0],
            ];
            let c = 1.0 / p.tau_s * trust * dt;
            u = rotate([c * e[0], c * e[1], c * e[2]], u);
            let c = 1.0 / (p.tau_s * p.bias_tau_s) * trust * dt;
            for j in 0..3 {
                b[j] += c * e[j];
            }
        }
        let n = norm(u);
        let u = [u[0] / n, u[1] / n, u[2] / n];
        self.u = Some(u);
        self.b = b;
        self.push(t, u);
        u
    }

    fn push(&mut self, t: f64, u: Vec3) {
        self.hist.push_back((t, u));
        while self.hist.front().is_some_and(|h| h.0 < t - self.params.history_s) {
            self.hist.pop_front();
        }
    }

    /// 'Up' (IMU axes) at time `t` within the kept history, linearly interpolated between samples (then
    /// normalised); clamped at the ends. None before the first sample.
    pub fn up_at(&self, t: f64) -> Option<Vec3> {
        let (first, last) = (self.hist.front()?, self.hist.back()?);
        let u = if t >= last.0 {
            last.1
        } else if t < first.0 {
            first.1
        } else {
            // the samples around t: hist[j].t <= t < hist[j + 1].t
            let j = self.hist.partition_point(|h| h.0 <= t) - 1;
            let ((t0, u0), (t1, u1)) = (self.hist[j], self.hist[j + 1]);
            // numpy.interp: slope * (t - t0) + u0
            std::array::from_fn(|k| (u1[k] - u0[k]) / (t1 - t0) * (t - t0) + u0[k])
        };
        let n = norm(u);
        Some([u[0] / n, u[1] / n, u[2] / n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn angle_deg(a: Vec3, b: Vec3) -> f64 {
        let c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (norm(a) * norm(b));
        c.clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// 'Up' in the axes of an IMU tilted by `deg` about its x axis.
    fn up_tilted(deg: f64) -> Vec3 {
        let a = deg.to_radians();
        [0.0, a.sin(), a.cos()]
    }

    #[test]
    fn rotate_is_rodrigues() {
        // 90 deg about z: x -> y
        let v = rotate([0.0, 0.0, std::f64::consts::FRAC_PI_2], [1.0, 0.0, 0.0]);
        assert!(v[0].abs() < 1e-15 && (v[1] - 1.0).abs() < 1e-15 && v[2] == 0.0);
        // about a general axis: the length and the component along the axis are kept
        let (r, u) = ([0.3, -0.2, 0.5], [0.1, 0.9, -0.4]);
        let v = rotate(r, u);
        assert!((norm(v) - norm(u)).abs() < 1e-15);
        let along = |x: Vec3| r[0] * x[0] + r[1] * x[1] + r[2] * x[2];
        assert!((along(v) - along(u)).abs() < 1e-15);
        // and rotating back gives u again
        let back = rotate([-0.3, 0.2, -0.5], v);
        assert!(angle_deg(back, u) < 1e-6 && angle_deg(v, u) > 10.0);
        assert_eq!(rotate([0.0; 3], u), u);
    }

    #[test]
    fn starts_from_the_accelerometer_and_ignores_old_samples() {
        let mut f = UpFilter::new(UpFilterParams::default(), None);
        assert!(f.u.is_none() && f.up_at(0.0).is_none());
        let u = f.update(1.0, [0.0; 3], [0.0, 0.0, 2.0 * G]);
        assert_eq!(u, [0.0, 0.0, 1.0]);
        // not later than the previous sample: ignored, whatever it says
        assert_eq!(f.update(1.0, [1.0, 0.0, 0.0], [G, 0.0, 0.0]), u);
        assert_eq!(f.update(0.5, [1.0, 0.0, 0.0], [G, 0.0, 0.0]), u);
        assert_eq!(f.t(), Some(0.5));
    }

    #[test]
    fn gyro_alone_carries_the_estimate() {
        // accelerometer in free fall (gain 0): rotating the body by +0.2 rad about x turns 'up' by -0.2 rad
        let mut f = UpFilter::new(UpFilterParams::default(), None);
        f.update(0.0, [0.0; 3], [0.0, 0.0, G]);
        for k in 1..=100 {
            f.update(k as f64 * 0.01, [0.2, 0.0, 0.0], [0.0; 3]);
        }
        let u = f.u.unwrap();
        assert!(
            (u[1] - 0.2f64.sin()).abs() < 1e-12 && (u[2] - 0.2f64.cos()).abs() < 1e-12,
            "{u:?}"
        );
        assert_eq!(f.b, [0.0; 3]);
    }

    #[test]
    fn converges_to_the_accelerometer_and_learns_the_bias_when_still() {
        let bias = [0.0068, 0.0028, -0.0016]; // ~0.43 deg/s
        let up = up_tilted(10.0);
        let acc = [up[0] * G, up[1] * G, up[2] * G];
        let mut f = UpFilter::new(UpFilterParams::default(), None);
        f.update(0.0, bias, [0.0, 0.0, G]); // starts 10 deg off
        for k in 1..=1000 {
            f.update(k as f64 * 0.01, bias, acc);
        }
        assert!(angle_deg(f.u.unwrap(), up) < 0.05);
        assert!(f.still_since().is_some());
        for (b, bias) in f.b.iter().zip(bias) {
            assert!((b - bias).abs() < 1e-4, "{:?}", f.b); // all 3 axes, incl. about the vertical
        }
    }

    #[test]
    fn accelerometer_is_gated_by_its_magnitude_and_by_the_rotation_rate() {
        let wrong = up_tilted(30.0);
        // |a| 20 % off g: no correction at all
        let mut f = UpFilter::new(UpFilterParams::default(), None);
        f.update(0.0, [0.0; 3], [0.0, 0.0, G]);
        for k in 1..=100 {
            f.update(k as f64 * 0.01, [0.0; 3], wrong.map(|x| x * 1.2 * G));
        }
        assert_eq!(f.u.unwrap(), [0.0, 0.0, 1.0]);
        // |a| = g but the head turns fast (about the vertical: 'up' does not move): the rate gate holds
        let gated = UpFilterParams {
            rate_gate: Some(0.5),
            ..Default::default()
        };
        for (params, moved) in [(gated, false), (UpFilterParams::default(), true)] {
            let mut f = UpFilter::new(params, None);
            f.update(0.0, [0.0; 3], [0.0, 0.0, G]);
            for k in 1..=100 {
                f.update(k as f64 * 0.01, [0.0, 0.0, 1.0], wrong.map(|x| x * G));
            }
            assert_eq!(angle_deg(f.u.unwrap(), [0.0, 0.0, 1.0]) > 1.0, moved);
        }
    }

    #[test]
    fn up_at_interpolates_the_history_and_clamps() {
        let mut f = UpFilter::new(UpFilterParams::default(), None);
        let mut ups = vec![];
        let dt = 1.0 / 32.0; // exact in binary: the sample times below are exact
        for k in 0..100 {
            let t = k as f64 * dt;
            // |a| = 0 after the first sample: gyro only
            let acc = if k == 0 { [0.0, 0.0, G] } else { [0.0; 3] };
            ups.push(f.update(t, [0.5 * (3.0 * t).cos(), 0.0, 0.0], acc));
        }
        // history_s = 0.5 = 16 dt: samples 83..=99 are kept
        let mid = f.up_at(90.5 * dt).unwrap();
        let want = std::array::from_fn(|j| (ups[90][j] + ups[91][j]) / 2.0);
        assert!(angle_deg(mid, want) < 1e-6);
        assert!(angle_deg(f.up_at(90.0 * dt).unwrap(), ups[90]) < 1e-6);
        assert!(angle_deg(f.up_at(90.0 * dt).unwrap(), ups[91]) > 0.1);
        assert_eq!(f.up_at(5.0), f.up_at(99.0 * dt)); // after the last sample: the last
        assert_eq!(f.up_at(0.0), f.up_at(83.0 * dt)); // before the history: its first sample
        assert!(angle_deg(f.up_at(0.0).unwrap(), ups[83]) < 1e-6);
        assert!(angle_deg(f.up_at(0.0).unwrap(), ups[82]) > 0.01);
    }
}
