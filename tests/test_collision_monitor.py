"""Collision monitor through the Python API: the Python prototype's own tests, same assertions."""

import numpy as np
import pytest

from navette import (SafetyParams, Scan, boundary_points, free_arc_length, free_arc_lengths, safe_command,
                     stopping_distance)

P = SafetyParams()
BEAR = (np.arange(180) + 0.5) * 2.0 - 180  # 2-deg sectors, as FreeSpace's


def scan(free_at=lambda b: 2.5, fov=38.0):
    """Synthetic scan: sectors within +-fov seen; free distance free_at(bearing) (obstacle if < 2.5)."""
    seen = np.abs(BEAR) < fov
    d = np.array([free_at(b) for b in BEAR], float)
    obstacle = np.where(d < 2.5, d, np.inf)
    return Scan(BEAR, seen, obstacle, np.zeros(180, bool), np.full(180, 2.5))


def test_open_floor_allows_full_speed():
    v, w, _ = safe_command(0.4, 0.0, scan())
    assert v == 0.4 and w == 0.0


def test_wall_ahead_slows_then_stops():
    # 0.15 m: already inside radius + margin (0.18 m) -> no forward motion at all
    speeds = [safe_command(0.4, 0.0, scan(lambda b: d))[0] for d in (1.0, 0.35, 0.25, 0.15)]
    assert speeds[0] == 0.4  # 1 m: stopping distance at 0.4 m/s is 0.22 m, plenty of room
    assert 0.4 > speeds[1] > speeds[2] > speeds[3] == 0.0
    # the allowed speed stops before radius + margin from the wall; the next candidate speed would not
    room = 0.35 - P.radius - P.margin
    v = speeds[1]
    assert stopping_distance(v, P) <= room
    assert stopping_distance(v + 0.4 / P.n_speeds, P) > room - P.step


def test_obstacle_on_the_left_blocks_left_turn_only():
    # everything seen; obstacle on the left side at 0.3 m, bearings 60-80 deg (>= 0.26 m beside a straight path)
    sc = scan(lambda b: 0.3 if 60 < b < 80 else 2.5, fov=180)
    assert safe_command(0.4, 0.0, sc)[0] == 0.4
    # radius 0.15 m arcs: the left one comes within 0.18 m of it ~0.18 m along, before it could stop (0.22 m)
    assert safe_command(0.4, 2.67, sc)[0] < 0.4
    assert safe_command(0.4, -2.67, sc)[0] == 0.4  # mirror arc to the right: clear


def test_no_driving_into_unseen_sides():
    # open floor, but a radius 0.1 m arc to the right reaches the unseen side within the stopping distance
    assert safe_command(0.4, -4.0, scan())[0] < 0.4
    assert safe_command(0.4, -4.0, scan(fov=180))[0] == 0.4  # same arc, everything seen: allowed


def test_obstacle_running_off_the_image_bottom_blocks_forward():
    sc = scan(lambda b: 0.45 if abs(b) < 6 else 2.5)  # 0.45 m = closest visible point: may be much closer
    near = Scan(sc.bearing_deg, sc.seen, sc.obstacle, np.isfinite(sc.obstacle), sc.floor)
    assert safe_command(0.3, 0.0, sc)[0] == 0.3  # same distance, but its foot is seen: room to stop
    assert safe_command(0.3, 0.0, near)[0] == 0.0


def test_turn_in_place_and_reverse():
    assert safe_command(0.0, 1.0, scan(lambda b: 0.2)) == (0.0, 1.0, 0.0)  # wall close: still may turn
    assert safe_command(-0.2, 0.0, scan())[0] == 0.0  # no camera behind


# ---- the search over candidate speeds equals the one-candidate-at-a-time loop (the specification)


def reference(v, w, scan, p, extra=None):
    """The loop the search replaces, kept here as the specification."""
    if v <= 0:
        return max(v, -p.reverse_max), w, 0.0
    points = boundary_points(scan, p.unseen_m, near_m=p.radius + p.margin)
    if extra is not None and len(extra):
        points = np.concatenate([points, extra])
    for vc in np.linspace(min(v, p.v_max), 0, p.n_speeds, endpoint=False):
        s_free = free_arc_length(points, w / vc, p)
        if stopping_distance(vc, p) <= s_free:
            return float(vc), w, float(s_free)
    return 0.0, w, 0.0


def random_scan(rng):
    seen = np.abs(BEAR) < 55
    floor = np.where(seen, rng.uniform(0.3, 2.5, 180), 0.0)
    obst = np.where(seen & (rng.random(180) < 0.3), rng.uniform(0.2, 2.0, 180), np.inf)
    near = np.isfinite(obst) & (rng.random(180) < 0.1)
    return Scan(BEAR, seen, obst, near, floor, np.where(seen, 0.44, np.inf))


def test_free_arc_lengths_matches_free_arc_length():
    """The batched helper (safe_command, a controller's arc scoring) equals the per-arc function."""
    rng = np.random.default_rng(1)
    p = SafetyParams(latency=0.73, decel=0.16)
    for _ in range(200):
        pts = rng.uniform(-1.5, 1.5, (int(rng.integers(1, 80)), 2))
        curvs = rng.uniform(-6, 6, 9)
        curvs[0] = 0.0
        s_max = float(rng.uniform(0.3, 1.2))
        got = free_arc_lengths(pts, curvs, p, s_max)
        want = [free_arc_length(pts, c, p, s_max=s_max) for c in curvs]
        assert np.allclose(got, want, atol=1e-9), (got, want)


def test_search_matches_the_loop():
    rng = np.random.default_rng(0)
    for p in (SafetyParams(), SafetyParams(latency=0.73, decel=0.16)):
        for _ in range(300):
            scan = random_scan(rng)
            extra = rng.uniform(-1.5, 1.5, (int(rng.integers(0, 30)), 2))
            v, w = float(rng.uniform(-0.1, 0.4)), float(rng.choice([0.0, rng.uniform(-1, 1)]))
            a, b = safe_command(v, w, scan, p, extra), reference(v, w, scan, p, extra)
            assert np.allclose(a, b, atol=1e-9), (v, w, a, b)


# ---- the Python API itself


def test_safety_params_is_a_plain_value():
    p = SafetyParams()
    assert (p.radius, p.margin, p.v_max, p.decel, p.latency, p.unseen_m, p.reverse_max, p.step, p.n_speeds) == (
        0.13, 0.05, 0.4, 0.5, 0.15, 0.35, 0.0, 0.02, 20)
    q = SafetyParams(latency=1.53, decel=0.64)
    assert (q.latency, q.decel, q.radius) == (1.53, 0.64, 0.13)
    assert p == SafetyParams() and p != q
    assert repr(q) == ("SafetyParams(radius=0.13, margin=0.05, v_max=0.4, decel=0.64, latency=1.53, unseen_m=0.35, "
                       "reverse_max=0.0, step=0.02, n_speeds=20)")
    q.latency, q.decel, q.n_speeds = 0.15, 0.5, 20  # attributes are writable
    assert q == p
    q.reverse_max = 0.1
    assert safe_command(-0.2, 0.0, scan(), q)[0] == -0.1 and safe_command(-0.2, 0.0, scan(), p)[0] == 0.0


def test_near_known_keeps_the_visible_distance():
    """A near sector whose blind segment the map has observed is not put at the clearance radius; the map's
    own points then decide."""
    sc = scan(lambda b: 0.45 if abs(b) < 6 else 2.5)
    near = Scan(sc.bearing_deg, sc.seen, sc.obstacle, np.isfinite(sc.obstacle), sc.floor)
    known = np.ones(180, bool)
    assert safe_command(0.3, 0.0, near, P, None, known)[0] == 0.3
    assert safe_command(0.3, 0.0, near, P, np.zeros((0, 2)), ~known)[0] == 0.0
    assert safe_command(0.3, 0.0, near, P, extra_points=np.array([[0.15, 0.0]]), near_known=known)[0] == 0.0


def test_wrong_shapes_are_errors():
    with pytest.raises(ValueError):
        safe_command(0.3, 0.0, scan(), P, None, np.ones(10, bool))
    with pytest.raises(ValueError):
        boundary_points(scan(), 0.35, near_m=0.18, near_mask=np.ones(10, bool))
    with pytest.raises(ValueError):
        safe_command(0.3, 0.0, scan(), P, np.zeros((4, 3)))
    with pytest.raises(ValueError):
        free_arc_lengths(np.zeros((4, 3)), [0.0], P, 0.5)
