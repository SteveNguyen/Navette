"""Floor-plane geometry against a synthetic camera. Pixels go through navette.rays.pixel_rays first: the
lens model is outside the core."""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from conftest import camera  # noqa: E402
from navette import gravity_frame, horizontal_range, rays_to_floor, to_fixed, to_robot, up_from_pitch, wrap_angle  # noqa: E402
from navette.rays import pixel_rays  # noqa: E402

K = np.array([[333.6, 0, 320.0], [0, 333.6, 180.0], [0, 0, 1]])
D = np.zeros(5)


def synthetic_camera(pitch_down_deg, roll_deg, h):
    """World: z up, x forward. Camera at (0,0,h), looking forward, pitched down."""
    return camera(pitch_down_deg, roll_deg)[0], np.array([0.0, 0.0, h])


def project(points_w, R_wc, c_w):
    p_c = (points_w - c_w) @ R_wc  # world -> camera
    uv, _ = cv2.projectPoints(p_c, np.zeros(3), np.zeros(3), K, D)
    return uv.reshape(-1, 2), p_c


def pixels_to_floor(uv, K, D, up_c, h):
    return rays_to_floor(pixel_rays(uv, K, D), up_c, h)


def test_floor_projection_recovers_ground_points():
    R_wc, c_w = synthetic_camera(25.0, 0.0, 0.30)
    pts = np.array([[0.6, 0.0, 0], [1.0, 0.2, 0], [1.5, -0.3, 0], [2.0, 0.1, 0]])
    uv, _ = project(pts, R_wc, c_w)
    up_c = R_wc.T @ np.array([0, 0, 1.0])
    est = pixels_to_floor(uv, K, D, up_c, 0.30)
    # No roll and the camera looks along world x, so g frame == world frame here.
    assert np.allclose(est, pts[:, :2], atol=1e-6)


def test_up_from_pitch_matches_a_pitched_camera():
    for pitch in (0.0, 15.0, 30.0):
        R_wc, _ = synthetic_camera(pitch, 0.0, 0.3)
        assert np.allclose(up_from_pitch(pitch), R_wc.T @ np.array([0, 0, 1.0]), atol=1e-12)


def test_points_above_horizon_are_nan():
    R_wc, c_w = synthetic_camera(5.0, 0.0, 0.30)
    up_c = R_wc.T @ np.array([0, 0, 1.0])
    est = pixels_to_floor(np.array([[320.0, 0.0]]), K, D, up_c, 0.30)  # top row, above horizon
    assert np.all(np.isnan(est))


def test_range_is_invariant_to_roll():
    R_wc, c_w = synthetic_camera(30.0, 8.0, 0.30)
    pts = np.array([[0.8, 0.1, 0], [1.2, -0.2, 0]])
    uv, p_c = project(pts, R_wc, c_w)
    up_c = R_wc.T @ np.array([0, 0, 1.0])
    est = pixels_to_floor(uv, K, D, up_c, 0.30)
    assert np.allclose(np.hypot(est[:, 0], est[:, 1]), np.hypot(pts[:, 0], pts[:, 1]), atol=1e-6)
    assert np.allclose(horizontal_range(p_c, up_c), np.hypot(pts[:, 0], pts[:, 1]), atol=1e-9)


# --- frames, poses, angles --------------------------------------------------------------------------------


def test_gravity_frame_matches_the_numpy_formula():
    for up_c in (camera(25, 8)[1], 3.0 * camera(40, -12)[1], np.array([0.1, -0.9, 0.3])):
        z = up_c / np.linalg.norm(up_c)
        x = np.array([0.0, 0.0, 1.0]) - z[2] * z
        x /= np.linalg.norm(x)
        assert np.allclose(gravity_frame(up_c), np.stack([x, np.cross(z, x), z]), atol=1e-15)


def test_pose_transforms_match_the_numpy_formulas():
    rng = np.random.default_rng(0)
    pts, pose = rng.normal(size=(6, 2)), (0.4, -1.3, 2.1)
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    fixed = to_fixed(pts, pose)
    assert np.allclose(fixed, np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1),
                       atol=1e-15)
    assert np.allclose(to_robot(fixed, np.array(pose)), pts, atol=1e-12)


def test_wrap_angle_is_the_floored_modulo():
    for a in (0.0, 1.0, -1.0, math.pi, -math.pi, 3.5, -3.5, 7 * math.pi + 0.2, -9.1):
        assert wrap_angle(a) == pytest.approx((a + math.pi) % (2 * math.pi) - math.pi, abs=1e-12)
        assert -math.pi <= wrap_angle(a) < math.pi
