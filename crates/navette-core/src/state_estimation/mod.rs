//! State estimation: camera attitude and how far to trust it.

pub mod attitude;
pub mod tilt_trust;

/// Standard gravity (m/s^2).
pub const G: f64 = 9.81;
