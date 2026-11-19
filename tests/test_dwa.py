"""Arc controller on synthetic scans (the Python prototype's tests of its arc selection, with the goal heading
given: the prototype chose it on a local map, which the frontier planner replaced)."""

import math

import numpy as np
import pytest

from navette import ArcController, ExploreParams, OccupancyGrid, Scan

BEAR = (np.arange(180) + 0.5) * 2.0 - 180


def scan(obst_at=lambda b: np.inf, fov=38.0, closest=0.4):
    seen = np.abs(BEAR) < fov
    obst = np.array([obst_at(b) for b in BEAR], float)
    return Scan(BEAR, seen, obst, np.zeros(180, bool), np.full(180, 2.5), np.full(180, closest))


def run(sc, goal, pose=(0, 0, 0), v_applied=0.3, t=0.0, ex=None):
    """One step toward the fixed-frame heading `goal` (rad), on an empty map."""
    ex = ex or ArcController()
    return ex.step(t, pose, sc, OccupancyGrid(), v_applied, goal), ex


def test_open_floor_drives_straight_at_cruise():
    (v, w), _ = run(scan(), 0.0)
    assert v == 0.3 and abs(w) < 0.21


def test_curves_around_an_obstacle_instead_of_stopping():
    # a box 0.8 m ahead, slightly left: keep moving, steer right
    (v, w), _ = run(scan(lambda b: 0.8 if -4 < b < 14 else np.inf), 0.0)
    assert v > 0 and w < 0


def test_goal_behind_spins_in_place():
    # wall everywhere in view at 0.5 m, the goal behind
    (v, w), ex = run(scan(lambda b: 0.5), math.pi, t=1.0)
    assert v == 0.0 and abs(w) > 0 and ex.state == "SPIN"


def test_rotate_in_place_when_no_arc_and_goal_off_ahead():
    # obstacle at 0.2 m all around (no admissible arc, even the tightest), goal 45 deg left (below
    # spin_above): without rotate_in_place the controller waits for the blocked timer; with it (as the
    # frontier exploration sets it) it turns toward the goal without counting as blocked
    sc = scan(lambda b: 0.2, fov=181, closest=0.15)
    for rip, expected in ((False, (0.0, 0.0)), (True, (0.0, ExploreParams().w_spin))):
        ex = ArcController(ExploreParams(rotate_in_place=rip))
        (v, w), ex = run(sc, math.radians(45), ex=ex, v_applied=0.0)
        assert (v, w) == expected
        assert (ex.blocked_since is None) == rip


def test_blocked_for_blocked_s_turns_to_the_goal_again():
    # obstacle at 0.2 m all around: no admissible arc. After blocked_s the controller asks its goal source
    # for another goal and turns to it; with a fixed goal, that is a turn in place toward the same heading.
    ex = ArcController()
    sc = scan(lambda b: 0.2, fov=181, closest=0.15)
    cmds = [ex.step(k * 0.1, (0, 0, 0), sc, OccupancyGrid(), 0.0, 0.0) for k in range(12)]
    assert cmds[:11] == [(0.0, 0.0)] * 11 and ex.blocked_since is None
    assert cmds[11] == (0.0, ExploreParams().w_spin) and ex.state == "SPIN"


def test_near_known_uses_the_collision_monitor_s_near_rule():
    # a wall at an angle touching the image bottom on the left (near sectors at 0.4 m), open ahead
    near = (BEAR > 10) & (BEAR < 38)
    sc = Scan(BEAR, np.abs(BEAR) < 38, np.where(near, 0.4, np.inf), near, np.full(180, 2.5), np.full(180, 0.4))
    known, unknown = np.ones(180, bool), np.zeros(180, bool)
    # default (the prototype): every near sector at the clearance radius, whatever the mask: no arc
    assert ArcController().step(0.0, (0, 0, 0), sc, OccupancyGrid(), 0.3, 0.0) == (0.0, 0.0)
    assert ArcController().step(0.0, (0, 0, 0), sc, OccupancyGrid(), 0.3, 0.0, known) == (0.0, 0.0)
    on = ExploreParams(near_known=True)
    assert on.near_known and not ExploreParams().near_known and "near_known=True" in repr(on)
    # on: only the sectors the map does not explain
    assert ArcController(on).step(0.0, (0, 0, 0), sc, OccupancyGrid(), 0.3, 0.0, unknown) == (0.0, 0.0)
    v, _ = ArcController(on).step(0.0, (0, 0, 0), sc, OccupancyGrid(), 0.3, 0.0, near_known=known)
    assert v > 0
    with pytest.raises(ValueError, match="near_known has 3 entries"):
        ArcController(on).step(0.0, (0, 0, 0), sc, OccupancyGrid(), 0.3, 0.0, np.ones(3, bool))


def turn_rates(params, headings_deg, goals_deg, w_meas=0.0):
    """Yaw rates commanded on open floor from these headings toward these goals (deg), one step each."""
    ex = ArcController(params)
    return [ex.step(k * 0.1, (0, 0, math.radians(h)), scan(), OccupancyGrid(), 0.0, math.radians(g),
                    w_meas=w_meas)[1] for k, (h, g) in enumerate(zip(headings_deg, goals_deg))]


def test_latched_turn_toward_a_goal_behind():
    # the goal ~180 deg off jitters across +-180 (replans): the prototype reverses at each crossing
    goals = [178, -178, 178, -178]
    assert turn_rates(ExploreParams(), [0] * 4, goals) == [1.0, -1.0, 1.0, -1.0]
    latched = ExploreParams(spin_flip_deg=45.0)
    assert turn_rates(latched, [0] * 4, goals) == [1.0] * 4
    # a real goal switch (more than spin_flip_deg) reverses; so does an overshoot (40 deg past the goal)
    assert turn_rates(latched, [0, 0], [178, -100]) == [1.0, -1.0]
    assert turn_rates(latched, [0, 130], [90, 90]) == [1.0, -1.0]


def test_turn_ends_on_the_predicted_heading_and_slows_down():
    # spinning to 90 deg, at 64 deg turning at 0.5 rad/s: 0.5 s ahead it is within spin_done (20 deg)
    assert turn_rates(ExploreParams(), [0, 64], [90, 90], w_meas=0.5) == [1.0, 1.0]
    w = turn_rates(ExploreParams(spin_lead_s=0.5), [0, 64], [90, 90], w_meas=0.5)
    assert w[0] == 1.0 and w[1] < 1.0  # an arc: the turn ended
    # slowdown: sqrt(2 a angle) between spin_w_min and w_spin (a goal 30 deg off: rotate in place, no arc)
    sc = scan(lambda b: 0.2, fov=181, closest=0.15)
    slow = ExploreParams(rotate_in_place=True, spin_accel=0.5, spin_w_min=0.3)
    (v, w), _ = run(sc, math.radians(30), ex=ArcController(slow), v_applied=0.0)
    assert v == 0.0 and w == pytest.approx(math.sqrt(2 * 0.5 * math.radians(30)))
    (v, w), _ = run(sc, math.radians(30), ex=ArcController(ExploreParams(rotate_in_place=True)), v_applied=0.0)
    assert w == 1.0


def test_blocked_only_without_progress():
    # no admissible arc, nothing applied, the robot creeping 2 cm per step: blocked after 1 s by the
    # prototype's rule, never by the progress rule (5 cm counts); standing still, blocked after blocked_s
    sc = scan(lambda b: 0.2, fov=181, closest=0.15)

    def first_block(p, dx):
        ex = ArcController(p)
        for k in range(60):
            ex.step(k * 0.1, (dx * k, 0, 0), sc, OccupancyGrid(), 0.0, 0.0)
            if ex.state == "SPIN":
                return k
        return None

    progress = ExploreParams(blocked_progress=True, blocked_s=3.0)
    assert (progress.progress_m, progress.progress_deg) == (0.05, 17.0) and not ExploreParams().blocked_progress
    assert first_block(ExploreParams(), 0.02) == 11
    assert first_block(progress, 0.02) is None
    assert first_block(progress, 0.0) == 31


def test_map_obstacles_count_as_the_scan_s():
    # open scan, but the map holds a wall 0.28 m ahead: too close to stop from cruise speed
    grid = OccupancyGrid(size_m=6.0)
    l = np.zeros((grid.n, grid.n), np.float32)
    l[grid.n // 2 + 5, grid.n // 2 - 30:grid.n // 2 + 30] = 2.0
    grid.set_map(-3.0, -3.0, l)
    ex = ArcController()
    v, w = ex.step(0.0, (0, 0, 0), scan(), grid, 0.3, 0.0)
    v_open, w_open = ArcController().step(0.0, (0, 0, 0), scan(), OccupancyGrid(), 0.3, 0.0)
    assert (v_open, w_open) == (0.3, 0.0) and v < 0.3


def test_parameters_are_dataclass_like():
    p = ExploreParams(v_set=(0.1, 0.25), rotate_in_place=True)
    assert (p.v_set, p.n_w, p.w_max, p.rotate_in_place) == ([0.1, 0.25], 13, 1.2, True)
    assert ExploreParams().v_set == [0.1, 0.2, 0.3] and not ExploreParams().rotate_in_place
    p.w_spin = 0.5
    assert p == ExploreParams(v_set=(0.1, 0.25), rotate_in_place=True, w_spin=0.5) and p != ExploreParams()
    assert "rotate_in_place=True" in repr(p) and "moving_v=0.05" in repr(p)
    assert ExploreParams().moving_v == 0.05 and ExploreParams(moving_v=0.02) != ExploreParams()
