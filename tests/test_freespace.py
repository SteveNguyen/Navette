"""Scan of rendered synthetic scenes (ideal lens), and the Python API of FreeSpace / Scan."""

import numpy as np
import pytest

pytest.importorskip("cv2")  # lens model of the rays (navette.rays)

from conftest import camera, free_space, ideal_calib  # noqa: E402
from navette import FreeSpace, Scan, drive_mask, seg_probs  # noqa: E402
from navette.rays import MASK_H, MASK_W, mask_rays  # noqa: E402

H_CAM = 0.25  # m


def render(cal, R_wc, is_floor):
    """Floor mask of a scene: a pixel is floor if its ray hits the floor at a point where is_floor(x, y)."""
    rw = mask_rays(cal) @ R_wc.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -H_CAM / rw[:, 2]
    x, y = rw[:, 0] * t, rw[:, 1] * t
    down = rw[:, 2] < 0
    return (down & is_floor(np.where(down, x, 0), np.where(down, y, 0))).reshape(MASK_H, MASK_W)


def test_wall_distance_per_bearing(tmp_path):
    cal = ideal_calib(tmp_path)
    for pitch, roll in [(20, 0), (25, 8), (15, -10)]:
        R_wc, up = camera(pitch, roll)
        d = 1.2  # wall perpendicular to the forward direction
        sc = free_space(cal).scan(render(cal, R_wc, lambda x, y: x < d), up, H_CAM)
        # Scan bearings are relative to the camera heading (optical axis on the floor); the roll applied about
        # the world x axis after the pitch turns that heading by yaw0 from the world x axis.
        opt = R_wc[:, 2]
        yaw0 = np.degrees(np.arctan2(opt[1], opt[0]))
        lo, hi = sc.bearing_deg - 1 + yaw0, sc.bearing_deg + 1 + yaw0  # sector, in world bearings
        nearest = np.where((lo < 0) & (hi > 0), 0, np.minimum(np.abs(lo), np.abs(hi)))  # wall point nearest
        expect = d / np.cos(np.radians(nearest))
        m = sc.seen & np.isfinite(sc.obstacle)
        assert m.sum() >= 20, (pitch, roll, m.sum())  # wall seen over a wide range of bearings
        err = sc.obstacle[m] - expect[m]  # 1 mask row = 2.6 cm of floor at 1.2 m
        assert np.all(np.abs(err) < 0.02), (pitch, roll, err.min(), err.max())
        assert not sc.near[m].any()
        free = sc.free()[m]  # last floor pixel centre: up to half a pixel short of the obstacle (conservative)
        assert np.all((free <= sc.obstacle[m]) & (free > sc.obstacle[m] - 0.03))


def test_empty_floor_is_free_to_max_range(tmp_path):
    cal = ideal_calib(tmp_path)
    R_wc, up = camera(20, 0)  # horizon in view: pixels above it are not floor but must not count
    sc = free_space(cal).scan(render(cal, R_wc, lambda x, y: np.ones_like(x, bool)), up, H_CAM)
    assert np.all(np.isinf(sc.obstacle))
    ahead = np.abs(sc.bearing_deg) < 30  # at the view edges only the image bottom corners see floor, nearby
    assert np.allclose(sc.free()[ahead], 2.5)
    assert sc.seen[np.abs(sc.bearing_deg) < 30].all()  # ~76 deg horizontal field of view
    assert not sc.seen[np.abs(sc.bearing_deg) > 60].any()
    assert np.isnan(sc.free()[~sc.seen]).all()


def test_box_on_the_left_is_at_positive_bearings(tmp_path):
    cal = ideal_calib(tmp_path)
    R_wc, up = camera(20, 0)
    box = lambda x, y: ~((x > 0.8) & (x < 1.0) & (y > 0.2) & (y < 0.5))  # near face at x = 0.8, left side
    sc = free_space(cal).scan(render(cal, R_wc, box), up, H_CAM)
    b = sc.bearing_deg[np.isfinite(sc.obstacle)]
    assert b.min() > 9 and b.max() < 33  # atan2(0.2, 1.0) = 11 deg .. atan2(0.5, 0.8) = 32 deg
    face = (sc.bearing_deg > 16) & (sc.bearing_deg < 30)
    expect = 0.8 / np.cos(np.radians(sc.bearing_deg[face] - 1))
    assert np.all(np.abs(sc.obstacle[face] - expect) < 0.02)


def test_obstacle_filling_the_image_bottom_is_near(tmp_path):
    cal = ideal_calib(tmp_path)
    R_wc, up = camera(20, 0)
    sc = free_space(cal).scan(render(cal, R_wc, lambda x, y: ~(np.abs(y) < 0.1)), up, H_CAM)  # a strip ahead
    ahead = np.abs(sc.bearing_deg) < 1.5
    assert sc.near[ahead].all() and np.isfinite(sc.obstacle[ahead]).all()
    assert not sc.near[sc.seen & (np.abs(sc.bearing_deg) > 30)].any()


def test_small_blobs_are_ignored(tmp_path):
    cal = ideal_calib(tmp_path)
    R_wc, up = camera(20, 0)
    floor = render(cal, R_wc, lambda x, y: np.ones_like(x, bool))
    floor[150:152, 100:103] = False  # 6-pixel speck on the floor
    assert np.isinf(free_space(cal).scan(floor, up, H_CAM).obstacle).all()
    assert np.isfinite(free_space(cal, min_blob_px=1).scan(floor, up, H_CAM).obstacle).any()


def test_seg_probs_floor_uses_channels_0_1_only():
    # the under logit (channel 2) must not leak into the floor probability
    z2 = np.random.default_rng(0).normal(0, 3, (2, 4, 5)).astype(np.float32)
    z3 = np.concatenate([z2, np.full((1, 4, 5), 10.0, np.float32)])
    p2, u2 = seg_probs(z2)
    p3, u3 = seg_probs(z3)
    assert u2 is None and np.allclose(p2, p3) and np.allclose(u3, 1 / (1 + np.exp(-10.0)))
    # same values as the numpy expression
    e = np.exp(z2 - z2.max(0))
    assert np.allclose(p2, e[1] / e.sum(0), atol=1e-6)


def test_under_band_becomes_an_obstacle_at_its_near_edge(tmp_path):
    # open floor; the model says 'under' for the floor 0.8-1.5 m ahead (e.g. under a bed): the scan puts the
    # obstacle at 0.8 m, where the overhang starts; ignoring the under output leaves the floor free
    cal = ideal_calib(tmp_path)
    R_wc, up = camera(20, 0)
    floor = render(cal, R_wc, lambda x, y: np.ones_like(x, bool))
    under = render(cal, R_wc, lambda x, y: (x > 0.8) & (x < 1.5))
    p_floor, p_under = floor.astype(np.float32), under.astype(np.float32)
    fs = free_space(cal)
    sc = fs.scan(drive_mask(p_floor, p_under, 0.8, 0.5), up, H_CAM)
    ahead = np.abs(sc.bearing_deg) < 10
    assert np.all(np.abs(sc.obstacle[ahead] - 0.8 / np.cos(np.radians(np.abs(sc.bearing_deg[ahead]) - 1))) < 0.03)
    assert np.isinf(fs.scan(drive_mask(p_floor, p_under, 0.8, None), up, H_CAM).obstacle[ahead]).all()
    assert np.isinf(fs.scan(drive_mask(p_floor, None, 0.8, 0.5), up, H_CAM).obstacle[ahead]).all()


# --- the Python API of the bindings -----------------------------------------------------------------------


def test_scan_built_by_hand_behaves_like_a_dataclass():
    bear = (np.arange(180) + 0.5) * 2.0 - 180
    seen = np.abs(bear) < 40
    obstacle = np.where(np.abs(bear) < 10, 1.0, np.inf)
    sc = Scan(bear, seen, obstacle, np.zeros(180, bool), np.full(180, 2.5))
    assert sc.obstacle is obstacle and sc.seen is seen  # the arrays are held, not copied
    assert np.isinf(sc.closest).all()  # default
    assert np.array_equal(sc.free(), np.where(seen, np.minimum(obstacle, 2.5), np.nan), equal_nan=True)
    sc.near[np.abs(bear) < 4] = True  # in-place edits are seen, as with the dataclass
    sc.obstacle[90] = 0.5
    assert sc.near.sum() == 4 and sc.free()[90] == 0.5
    sc.floor = np.full(180, 0.3)
    assert np.all(sc.free()[seen] == 0.3)
    assert Scan(list(bear), list(seen), list(obstacle), [0] * 180, [2] * 180).floor.dtype == np.float64  # coerced
    with pytest.raises(ValueError):
        Scan(bear, seen[:5], obstacle, np.zeros(180, bool), np.full(180, 2.5)).free()


def test_wrong_shapes_raise(tmp_path):
    rays = np.ones((6, 3))
    with pytest.raises(ValueError):
        FreeSpace(rays, rays, mask_shape=(2, 4))
    with pytest.raises(ValueError):
        FreeSpace(np.ones((6, 2)), np.ones((6, 2)), mask_shape=(2, 3))
    fs = FreeSpace(rays, rays, mask_shape=(2, 3))
    with pytest.raises(ValueError):
        fs.scan(np.ones((3, 2), bool), [0, -1, 0], 0.3)
    # a non-contiguous mask is accepted
    big = np.ones((4, 6), bool)
    assert fs.scan(big[::2, ::2], [0, -1, -0.3], 0.3).seen.shape == (180,)
