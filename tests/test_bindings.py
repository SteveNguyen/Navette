"""The bindings' handling of numpy arrays: an array is read by its values, whatever its memory layout."""

import numpy as np
import pytest

import navette
from navette.testing.fakes import camera_rotation, pinhole_rays

SHAPE = (96, 160)
SCAN_FIELDS = ("bearing_deg", "seen", "obstacle", "near", "floor", "closest")


def layouts(a):
    """The same values in other memory layouts: Fortran order, a transposed view of a C array (what `x.T > thr`
    gives), a strided view, a negative stride, a non-contiguous slice of a larger array."""
    big = np.zeros((a.shape[0] + 2, a.shape[1] + 3), a.dtype)
    big[1:-1, 2:-1] = a
    out = {"fortran": np.asfortranarray(a), "transposed_view": np.ascontiguousarray(a.T).T,
           "strided": np.repeat(a, 2, axis=1)[:, ::2], "reversed": np.ascontiguousarray(a[::-1])[::-1],
           "slice": big[1:-1, 2:-1]}
    for name, b in out.items():
        assert b.shape == a.shape and np.array_equal(b, a) and not b.flags.c_contiguous, name
    return out


def same_scan(a, b):
    return all(np.array_equal(getattr(a, k), getattr(b, k), equal_nan=True) for k in SCAN_FIELDS)


@pytest.fixture
def view():
    """(FreeSpace, 'up' in the camera frame, a floor mask with a box on the left and a wall far on the
    right): nothing symmetric, so a mask read in the wrong order gives another scan."""
    fs = navette.FreeSpace(pinhole_rays(SHAPE, 104.0), pinhole_rays(SHAPE, 104.0, dv=0.5), SHAPE)
    up = camera_rotation(20.0).T @ np.array([0, 0, 1.0])
    mask = np.ones(SHAPE, bool)
    mask[30:70, 20:60] = False
    mask[:50, 100:] = False
    return fs, up, mask


def test_floor_mask_is_read_by_value_whatever_its_layout(view):
    """A Fortran-ordered mask (or any non-C-contiguous one) used to be read in memory order: same shape, same
    values, another scan (an obstacle 0.46 m ahead read at 0.70 m)."""
    fs, up, mask = view
    ref, obst = fs.scan(mask, up, 0.35), fs.obstacles(mask)
    assert np.isfinite(ref.obstacle).sum() > 10
    for name, m in layouts(mask).items():
        assert same_scan(fs.scan(m, up, 0.35), ref), name
        assert np.array_equal(fs.obstacles(m), obst), name
        nav = navette.Navigator(behaviour="forward")
        out, scan = nav.step_mask(fs, m, tuple(up), 0.0, 0.1, (0, 0, 0), 0.35, 0, 0, 0, 0)
        assert same_scan(scan, ref), name


def test_other_2d_arrays_are_read_by_value_whatever_their_layout(view):
    fs, up, mask = view
    rng = np.random.default_rng(1)
    rays, low = pinhole_rays(SHAPE, 104.0), pinhole_rays(SHAPE, 104.0, dv=0.5)
    ref = fs.scan(mask, up, 0.35)
    logits, p = rng.random((3, *SHAPE)).astype(np.float32), rng.random(SHAPE).astype(np.float32)
    grid = navette.OccupancyGrid(size_m=2.0)
    log_odds = rng.normal(size=(grid.n, grid.n)).astype(np.float32)
    pts = rng.random((7, 2)) * 0.6
    sp = navette.SafetyParams()
    obst, free = log_odds > 0.5, log_odds < -0.5
    cost = 1.0 + rng.random(log_odds.shape)
    for name in layouts(mask):
        lay = lambda a, name=name: layouts(a)[name]  # noqa: E731
        # FreeSpace's rays (N, 3)
        fs2 = navette.FreeSpace(lay(rays), lay(low), SHAPE)
        assert same_scan(fs2.scan(mask, up, 0.35), ref), name
        r, sec = fs.project(rays, 0.35)
        r2, sec2 = fs.project(lay(rays), 0.35)
        assert np.array_equal(r, r2, equal_nan=True) and np.array_equal(sec, sec2), name
        # probabilities and masks
        assert np.array_equal(navette.drive_mask(lay(p), None, 0.5, None), navette.drive_mask(p, None, 0.5, None)), name
        assert np.array_equal(navette.drive_mask(p, lay(p), 0.5, 0.5), navette.drive_mask(p, p, 0.5, 0.5)), name
        assert np.array_equal(navette.seg_probs(np.asfortranarray(logits))[0], navette.seg_probs(logits)[0])
        # the grid: map, points, cells
        grid.set_map(0.0, 0.0, lay(log_odds))
        assert np.array_equal(grid.log_odds, log_odds), name
        assert np.array_equal(grid.cell(lay(pts)), grid.cell(pts)), name
        ij = grid.cell(pts)
        assert np.array_equal(grid.centres(lay(ij)), grid.centres(ij)), name
        # the collision monitor's points
        assert navette.free_arc_length(lay(pts), 0.5, sp) == navette.free_arc_length(pts, 0.5, sp), name
        assert np.array_equal(navette.free_arc_lengths(lay(pts), [0.0, 1.0], sp, 1.0),
                              navette.free_arc_lengths(pts, [0.0, 1.0], sp, 1.0)), name
        assert navette.safe_command(0.3, 0.2, ref, sp, lay(pts)) == navette.safe_command(0.3, 0.2, ref, sp, pts), name
        # the planner's grids
        assert np.array_equal(navette.frontier_cells(lay(obst), lay(free), 2), navette.frontier_cells(obst, free, 2)), name
        assert np.array_equal(navette.distance_transform(lay(obst)), navette.distance_transform(obst)), name
        d, pred = navette.dijkstra(~obst, cost, 0.05, (3, 4))
        d2, pred2 = navette.dijkstra(lay(~obst), lay(cost), 0.05, (3, 4))
        assert np.array_equal(d, d2) and np.array_equal(pred, pred2), name
