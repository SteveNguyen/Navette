"""navette._core.LikelihoodGrid / match_points: the binding of the correlative scan matcher (the algorithm is
tested in the core, crates/navette-core/src/slam/scan_matcher.rs). Synthetic rooms only."""

import math

import numpy as np
import pytest
from navette._core import LikelihoodGrid, match_points, to_robot

# L-shaped room, 4 x 3.5 m (no symmetry)
L_ROOM = [(-2.0, -1.75), (2.0, -1.75), (2.0, 0.25), (0.0, 0.25), (0.0, 1.75), (-2.0, 1.75)]


def walls(vertices, spacing, noise, rng, closed=True):
    """Points every ~spacing m along the polyline, with Gaussian noise (m)."""
    v = np.asarray(vertices, float)
    ends = np.vstack([v[1:], v[:1]]) if closed else v[1:]
    pts = []
    for a, b in zip(v, ends):
        m = round(float(np.linalg.norm(b - a)) / spacing)
        f = np.arange(m)[:, None] / m
        pts.append(a + f * (b - a))
    pts = np.vstack(pts)
    return pts + noise * rng.standard_normal(pts.shape)


def test_grid_values_and_frame():
    rng = np.random.default_rng(0)
    pts = walls(L_ROOM, 0.03, 0.0, rng)
    g = LikelihoodGrid(pts, res=0.05, sigma=0.05, margin=0.15)
    assert (g.res, g.sigma) == (0.05, 0.05)
    assert g.x0 == pytest.approx(-2.15) and g.y0 == pytest.approx(-1.9)
    v = g.values
    # the bounding box plus margins: 4.3 x 3.8 m, 86-87 x 76-77 cells (the last one holds max + margin)
    assert v.dtype == np.float32 and v.ndim == 2
    assert v.shape[0] in (86, 87) and v.shape[1] in (76, 77)
    assert v.max() <= 1.0 and v.min() == 0.0
    # on a wall: high; in the middle of the room: 0
    i, j = int((-2.0 - g.x0) / 0.05), int((0.0 - g.y0) / 0.05)
    assert v[i, j] > 0.6
    i, j = int((-1.0 - g.x0) / 0.05), int((-0.75 - g.y0) / 0.05)
    assert v[i, j] == 0.0


def test_match_recovers_a_known_offset():
    rng = np.random.default_rng(1)
    g = LikelihoodGrid(walls(L_ROOM, 0.03, 0.01, rng))
    truth = (0.23, -0.31, math.radians(12.0))
    query = to_robot(walls(L_ROOM, 0.03, 0.015, rng), truth)
    r = match_points(g, query, (0.0, 0.0, 0.0))
    x, y, th = r["pose"]
    assert math.hypot(x - truth[0], y - truth[1]) < 0.05
    assert abs(th - truth[2]) < math.radians(1.5)
    assert r["score"] > 0.6 and r["n"] == len(query)
    cov = r["covariance"]
    assert cov.shape == (3, 3) and np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    lo, hi = r["xy_eigenvalues"]
    assert 0 < lo <= hi and math.sqrt(hi) < 0.05  # well constrained


def test_corridor_covariance_is_long_along_the_corridor():
    rng = np.random.default_rng(2)

    def corridor(half_len, noise):
        return np.vstack([
            walls([(-half_len, -0.6), (half_len, -0.6)], 0.03, noise, rng, closed=False),
            walls([(-half_len, 0.6), (half_len, 0.6)], 0.03, noise, rng, closed=False),
        ])

    g = LikelihoodGrid(corridor(4.0, 0.01))
    truth = (0.1, 0.05, math.radians(2.0))
    r = match_points(g, to_robot(corridor(1.5, 0.015), truth), (0.0, 0.0, 0.0), dx=0.5, dy=0.5)
    cov = r["covariance"]
    assert abs(r["pose"][1] - truth[1]) < 0.03
    assert cov[0, 0] > 50 * cov[1, 1]
    assert math.sqrt(r["xy_eigenvalues"][1]) > 0.2 and abs(r["xy_major_axis"]) < math.radians(5)


def test_guess_outside_the_window_scores_low():
    rng = np.random.default_rng(3)
    g = LikelihoodGrid(walls(L_ROOM, 0.03, 0.01, rng))
    query = to_robot(walls(L_ROOM, 0.03, 0.015, rng), (1.3, -1.1, 0.0))
    assert match_points(g, query, (1.2, -1.0, 0.1))["score"] > 0.6
    assert match_points(g, query, (0.0, 0.0, 0.0))["score"] < 0.35


def test_argument_errors():
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        LikelihoodGrid(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        LikelihoodGrid(pts, res=0.0)
    with pytest.raises(ValueError):
        LikelihoodGrid(np.full((2, 2), np.nan))
    g = LikelihoodGrid(pts)
    with pytest.raises(ValueError):
        match_points(g, pts, (0.0, 0.0, 0.0), dx=-1.0)
    with pytest.raises(ValueError):
        match_points(g, np.zeros((2, 3)), (0.0, 0.0, 0.0))
    # no point: the guess, score 0
    r = match_points(g, np.zeros((0, 2)), (0.5, 0.0, 0.0))
    assert r["pose"] == (0.5, 0.0, 0.0) and r["score"] == 0.0 and r["n"] == 0
