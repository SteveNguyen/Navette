//! Floor-plane geometry for a single camera, and planar poses.
//!
//! Frames
//!   camera (c): OpenCV convention, x right, y down, z along the optical axis.
//!   gravity-aligned (g): origin at the camera centre, z up, x = optical axis projected on the floor
//!       (forward), y = left. Only the camera TILT matters here; heading is arbitrary for a single-frame
//!       floor projection.
//!   robot: planar, x forward, y left (the g frame seen from above).
//!   fixed: planar odometry frame; a [`Pose2`] is the robot's pose in it.
//!
//! Core idea: knowing the "up" direction in the camera frame and the camera height h above the floor, every
//! pixel that sees the floor maps to a metric floor point.
//!
//! The lens model is not here: the core takes per-pixel viewing rays (see `navette.rays` on the Python side).

use std::f64::consts::{PI, TAU};

pub type Vec3 = [f64; 3];
/// Row-major 3x3 matrix.
pub type Mat3 = [Vec3; 3];
/// Planar point (x, y) in metres.
pub type Point2 = [f64; 2];

#[inline]
pub fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalized(v: &Vec3) -> Vec3 {
    let n = dot(v, v).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Rotation R_gc (rows = g axes expressed in the camera frame), so v_g = R_gc * v_c.
/// `up_c` need not be normalised. Degenerate (NaN rows x and y) only if the camera looks straight down.
pub fn gravity_frame(up_c: &Vec3) -> Mat3 {
    let z = normalized(up_c);
    // optical axis (0, 0, 1) projected onto the floor plane
    let oz = z[2];
    let x = normalized(&[-oz * z[0], -oz * z[1], 1.0 - oz * z[2]]);
    let y = cross(&z, &x); // left
    [x, y, z]
}

/// 'up' in the camera frame (OpenCV: y down, z forward) for a camera pitched down by `pitch_down_deg`, no
/// roll. Looking down, the optical axis points away from 'up': up . z = -sin(pitch).
pub fn up_from_pitch(pitch_down_deg: f64) -> Vec3 {
    let p = pitch_down_deg.to_radians();
    [0.0, -p.cos(), -p.sin()]
}

/// Floor points (forward, left) in metres in the g frame, one per camera-frame viewing ray, for a camera at
/// height `h`. Rays that do not hit the floor (at or above the horizon) give NaN.
pub fn rays_to_floor(rays_c: &[Vec3], up_c: &Vec3, h: f64) -> Vec<Point2> {
    let r = gravity_frame(up_c);
    rays_c
        .iter()
        .map(|ray| {
            let z = dot(&r[2], ray);
            // floor is the plane z = -h
            let s = if z < 0.0 { -h / z } else { f64::NAN };
            [dot(&r[0], ray) * s, dot(&r[1], ray) * s]
        })
        .collect()
}

/// Horizontal distance from the camera foot point to 3D points `p_c` (camera frame).
pub fn horizontal_range(p_c: &[Vec3], up_c: &Vec3) -> Vec<f64> {
    let up = normalized(up_c);
    p_c.iter()
        .map(|p| {
            let k = dot(p, &up);
            let d = [p[0] - k * up[0], p[1] - k * up[1], p[2] - k * up[2]];
            dot(&d, &d).sqrt()
        })
        .collect()
}

/// Wrap an angle (rad) to [-pi, pi). `(a + pi) mod (2 pi) - pi` with a floored modulo, so pi maps to -pi.
#[inline]
pub fn wrap_angle(a: f64) -> f64 {
    (a + PI).rem_euclid(TAU) - PI
}

/// Pose of the robot in the fixed (odometry) frame: position (m) and heading (rad, counter-clockwise,
/// 0 = along fixed x). In Python: an `(x, y, theta)` sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose2 {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Pose2 {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta }
    }

    /// Robot-frame point (x forward, y left) -> fixed frame.
    #[inline]
    pub fn to_fixed(&self, p: Point2) -> Point2 {
        let (s, c) = self.theta.sin_cos();
        [self.x + c * p[0] - s * p[1], self.y + s * p[0] + c * p[1]]
    }

    /// Fixed-frame point -> robot frame.
    #[inline]
    pub fn to_robot(&self, p: Point2) -> Point2 {
        let (s, c) = self.theta.sin_cos();
        let (dx, dy) = (p[0] - self.x, p[1] - self.y);
        [c * dx + s * dy, -s * dx + c * dy]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_frame_is_a_rotation_with_z_up_and_x_forward() {
        for up in [up_from_pitch(20.0), [0.3, -2.0, -0.7], [-0.1, -0.9, 0.2]] {
            let r = gravity_frame(&up);
            for i in 0..3 {
                for j in 0..3 {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((dot(&r[i], &r[j]) - want).abs() < 1e-12);
                }
            }
            let n = dot(&up, &up).sqrt();
            assert!((dot(&r[2], &up) / n - 1.0).abs() < 1e-12); // z = up
            assert!(r[0][2] > 0.0); // x = the optical axis, minus its vertical part
            assert!(dot(&cross(&r[0], &r[1]), &r[2]) > 0.0); // right-handed: y = left
        }
    }

    #[test]
    fn floor_point_straight_ahead() {
        // camera 0.3 m up, pitched 30 deg down: the optical axis hits the floor at h / tan(30 deg)
        let p = rays_to_floor(&[[0.0, 0.0, 1.0]], &up_from_pitch(30.0), 0.3);
        assert!((p[0][0] - 0.3 / 30f64.to_radians().tan()).abs() < 1e-12 && p[0][1].abs() < 1e-12);
        // a ray to the right (camera +x) lands at negative y (y = left)
        assert!(rays_to_floor(&[[0.2, 0.0, 1.0]], &up_from_pitch(30.0), 0.3)[0][1] < 0.0);
        // level camera: the optical axis is on the horizon
        assert!(rays_to_floor(&[[0.0, 0.0, 1.0]], &up_from_pitch(0.0), 0.3)[0][0].is_nan());
    }

    #[test]
    fn wrap_angle_range() {
        assert_eq!(wrap_angle(0.0), 0.0);
        assert_eq!(wrap_angle(PI), -PI); // [-pi, pi)
        assert!((wrap_angle(3.0 * PI + 0.1) - (-PI + 0.1)).abs() < 1e-12);
        assert!((wrap_angle(-PI - 0.1) - (PI - 0.1)).abs() < 1e-12);
    }

    #[test]
    fn pose_transforms_are_inverse() {
        let pose = Pose2::new(1.0, -2.0, 0.7);
        let p = [0.4, 0.3];
        let q = pose.to_robot(pose.to_fixed(p));
        assert!((q[0] - p[0]).abs() < 1e-12 && (q[1] - p[1]).abs() < 1e-12);
        // heading +90 deg: robot forward is fixed +y
        let f = Pose2::new(1.0, 2.0, PI / 2.0).to_fixed([1.0, 0.0]);
        assert!((f[0] - 1.0).abs() < 1e-12 && (f[1] - 3.0).abs() < 1e-12);
    }
}
