"""navette's collision monitor against the Python prototype's outputs on the same inputs
(tests/golden/collision_monitor_*.npz, see tests/golden/README.md): random scans (seen / unseen / near /
inf / NaN mixes) x requests x optional map points and near_known, for three parameter sets."""

import numpy as np
import pytest

from conftest import golden_files
from navette import (SafetyParams, Scan, arc, boundary_points, free_arc_length, free_arc_lengths, safe_command,
                     stopping_distance)

# Lengths and coordinates: sin / cos / hypot need not be bit-identical between maths libraries, and the
# prototype squares v_max with libm's pow in the default look-ahead (x * x here: one ulp off for some v_max).
# On the machine that wrote the goldens the match is exact (0.0) for every value of these files.
RTOL = 1e-12

FILES = golden_files("collision_monitor")
PARAMS = ("radius", "margin", "v_max", "decel", "latency", "unseen_m", "reverse_max", "step")


def load(path):
    g = np.load(path)
    p = SafetyParams(**{k: float(g[k]) for k in PARAMS}, n_speeds=int(g["n_speeds"]))
    return g, p


def scan_of(g, k):
    return Scan(g["bearing_deg"], g["seen"][k], g["obstacle"][k], g["near"][k], g["floor"][k], g["closest"][k])


def extra_of(g, k):
    return g["extra"][g["extra_start"][k]:g["extra_start"][k + 1]] if g["has_extra"][k] else None


def points_of(g, k):
    """The points the arcs of case k were checked against: boundary (near at the clearance) + map points."""
    extra = extra_of(g, k)
    return g["bp_near"][k] if extra is None else np.concatenate([g["bp_near"][k], extra])


def assert_close(got, want, what):
    """Same inf / nan pattern, finite values within RTOL."""
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0, equal_nan=True, err_msg=str(what))


def test_golden_files_exist():
    assert len(FILES) >= 3


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_safe_command(path):
    g, p = load(path)
    for k in range(len(g["v"])):
        v, w = float(g["v"][k]), float(g["w"][k])
        known = g["near_known"][k] if g["has_known"][k] else None
        got = safe_command(v, w, scan_of(g, k), p, extra_of(g, k), known)
        what = (k, str(g["kind"][k]), v, w)
        assert got[0] == g["safe"][k, 0], what  # the same candidate speed, exactly
        assert got[1] == g["safe"][k, 1] == w, what
        assert_close(got[2], g["safe"][k, 2], what)


def test_safe_command_default_params():
    """Without parameters: SafetyParams(), the first file's."""
    g, p = load(FILES[0])
    assert p == SafetyParams()
    for k in range(40):
        got = safe_command(float(g["v"][k]), float(g["w"][k]), scan_of(g, k))
        want = safe_command(float(g["v"][k]), float(g["w"][k]), scan_of(g, k), p)
        assert got == want


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_boundary_points(path):
    g, p = load(path)
    clear = p.radius + p.margin
    for k in range(len(g["v"])):
        sc = scan_of(g, k)
        assert_close(boundary_points(sc, p.unseen_m), g["bp_plain"][k], k)
        assert_close(boundary_points(sc, p.unseen_m, near_m=clear), g["bp_near"][k], k)
        assert_close(boundary_points(sc, p.unseen_m, near_m=clear, near_mask=g["mask"][k]), g["bp_mask"][k], k)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_free_arc_lengths(path):
    g, p = load(path)
    for k in range(len(g["v"])):
        points, s_max = points_of(g, k), float(g["s_max"][k])
        got = free_arc_lengths(points, g["curvs"][k], p, s_max)
        assert got.shape == g["fal"][k].shape
        assert_close(got, g["fal"][k], k)
        assert np.array_equal(free_arc_lengths(points, list(g["curvs"][k]), p, s_max), got)  # a list, as well
    assert len(free_arc_lengths(points, [], p, s_max)) == 0


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_free_arc_length(path):
    g, p = load(path)
    assert g["single_valid"].sum() > 200
    for k in np.nonzero(g["single_valid"])[0]:  # (not the cases with a NaN point, see test_nan_points_are_ignored)
        points, s_max = points_of(g, k), float(g["s_max"][k])
        for j, c in enumerate(g["curvs"][k]):
            assert_close(free_arc_length(points, float(c), p, s_max=s_max), g["fa"][k, j], (k, j))
            assert_close(free_arc_length(points, float(c), p), g["fa_default"][k, j], (k, j))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_nan_points_are_ignored(path):
    """Tolerated difference: with a NaN point (a seen sector whose range is NaN), the prototype's single-arc
    function finds every arc free; navette ignores the NaN point, as the batched function of both does."""
    g, p = load(path)
    cases = np.nonzero(~g["single_valid"])[0]
    assert len(cases) > 5
    for k in cases:
        points, s_max = points_of(g, k), float(g["s_max"][k])
        got = [free_arc_length(points, float(c), p, s_max=s_max) for c in g["curvs"][k]]
        assert_close(got, g["fal"][k], k)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_arc_and_stopping_distance(path):
    g, p = load(path)
    for c, want in zip(g["arc_curv"], g["arc_xy"]):
        got = arc(float(c), g["arc_s"])
        assert got.shape == want.shape
        assert_close(got, want, c)
    assert_close([stopping_distance(float(v), p) for v in g["stop_v"]], g["stop_d"], "stopping_distance")
