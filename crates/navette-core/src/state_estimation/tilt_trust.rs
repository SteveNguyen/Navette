//! How far the scan can be trusted, given how wrong the camera 'up' may be right now.
//!
//! Range error from a tilt error s (rad) for a floor point at d, camera height h: s_d = (d^2 + h^2) / h * s
//! (d = h / tan(depression)). Keeping s_d <= tol gives the trusted range d_max = sqrt(tol h / s - h^2).
//!
//! The tilt error bound s comes from the base motion: the 'up' filter is pulled toward atan(a / g) of error
//! by a forward acceleration a (the lean of a balancing base fools the accelerometer) and toward
//! atan(v w / g) by the centripetal acceleration of a turn; it builds up and fades with the filter time
//! constant, so the bound is a peak hold of those terms decaying with tau_s. Plus a constant floor
//! (calibration, noise) and the frame / IMU timing error times the head rotation rate.
//!
//! Used to gate what goes into the map; the collision monitor keeps the full scan (near field: small
//! errors). a, v, w come from the base odometry.

use super::G;

/// Tilt error bound of a moving base. See the module documentation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TiltTrust {
    /// Constant floor of the bound (rad): calibration, noise.
    pub s0: f64,
    /// Decay time constant (s) of the motion term: the 'up' filter's.
    pub tau_s: f64,
    /// Range error (m) accepted at the trusted range.
    pub tol_m: f64,
    /// Frame / IMU timing error (s).
    pub timing_s: f64,
    /// Peak-held acceleration-induced error bound (rad).
    pub motion: f64,
    /// Current tilt error bound (rad). s0 before the first update (a frame can come before any motion data).
    pub sigma: f64,
}

impl Default for TiltTrust {
    fn default() -> Self {
        Self::new(0.5, 1.0, 0.10, 0.01)
    }
}

impl TiltTrust {
    pub fn new(sigma0_deg: f64, tau_s: f64, tol_m: f64, timing_s: f64) -> Self {
        let s0 = sigma0_deg.to_radians();
        Self {
            s0,
            tau_s,
            tol_m,
            timing_s,
            motion: 0.0,
            sigma: s0,
        }
    }

    /// One step of `dt` seconds: `a` forward acceleration (m/s^2), `v` speed (m/s), `w` yaw rate (rad/s),
    /// `gyro_rate` head rotation rate (rad/s, None: |w|). Returns the current tilt error bound (rad).
    pub fn update(&mut self, dt: f64, a: f64, v: f64, w: f64, gyro_rate: Option<f64>) -> f64 {
        let m = (a.abs() / G).atan() + ((v * w).abs() / G).atan();
        let held = self.motion * (-dt / self.tau_s).exp();
        self.motion = if held > m { held } else { m };
        let rate = gyro_rate.unwrap_or(w).abs();
        self.sigma = self.s0 + self.motion + rate * self.timing_s;
        self.sigma
    }

    /// Trusted range (m) for the camera height `h` under the current bound.
    pub fn max_range(&self, h: f64) -> f64 {
        (self.tol_m * h / self.sigma - h * h).max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: f64 = 0.35;

    #[test]
    fn range_error_at_the_trusted_range_equals_the_tolerance() {
        let mut tt = TiltTrust::default();
        assert_eq!(tt.sigma, 0.5f64.to_radians()); // usable before the first update
        let s = tt.update(0.1, 0.0, 0.0, 0.0, None);
        let d = tt.max_range(H);
        assert!(((d * d + H * H) / H * s - tt.tol_m).abs() < 1e-9);
        assert!(1.8 < d && d < 2.1); // 0.5 deg floor, 10 cm: ~2 m
    }

    #[test]
    fn acceleration_shrinks_the_range_then_it_recovers() {
        let mut tt = TiltTrust::default();
        tt.update(0.1, 0.0, 0.0, 0.0, None);
        let rest = tt.max_range(H);
        tt.update(0.1, 0.5, 0.2, 0.0, None);
        let accel = tt.max_range(H);
        assert!(accel < 0.5 * rest);
        for _ in 0..10 {
            tt.update(0.1, 0.0, 0.3, 0.0, None); // 1 s later: partly recovered, not yet back
        }
        assert!(accel < tt.max_range(H) && tt.max_range(H) < rest);
        for _ in 0..50 {
            tt.update(0.1, 0.0, 0.3, 0.0, None);
        }
        assert!((tt.max_range(H) - rest).abs() < 0.05);
    }

    #[test]
    fn turning_counts_like_a_sideways_acceleration() {
        let (mut a, mut b) = (TiltTrust::default(), TiltTrust::default());
        a.update(0.1, 0.3, 0.0, 0.0, None);
        b.update(0.1, 0.0, 0.3, 1.0, Some(0.0)); // centripetal 0.3 m/s^2
        assert!((a.sigma - b.sigma).abs() < 1e-12);
        // the head rotation rate (default |w|) adds the timing term
        let mut c = TiltTrust::default();
        c.update(0.1, 0.0, 0.3, -1.0, None);
        assert!((c.sigma - b.sigma - 0.01).abs() < 1e-12);
    }

    #[test]
    fn large_bound_means_no_trusted_range() {
        let mut tt = TiltTrust::default();
        tt.update(0.1, 9.81, 0.0, 0.0, None); // 45 deg of possible error
        assert_eq!(tt.max_range(H), 0.0);
    }
}
