"""navette.OccupancyGrid: the Python prototype's grid tests and the grid part of its blind-zone tests, same
assertions, plus the binding's own behaviour (keyframe poses, reset, argument checks)."""

import math

import numpy as np
import pytest

from navette import OccupancyGrid, Scan

BEAR = (np.arange(180) + 0.5) * 2.0 - 180


def scan(obst_at=lambda b: np.inf, fov=38.0, closest=0.4, floor=2.5):
    seen = np.abs(BEAR) < fov
    obst = np.array([obst_at(b) for b in BEAR], float)
    return Scan(BEAR, seen, obst, np.zeros(180, bool), np.full(180, floor), np.full(180, closest))


WALL = lambda b: 1.02 if abs(b) < 4 else np.inf  # noqa: E731


def status(g, xy):
    obst, free = g.masks()
    i, j = g.cell(np.array([xy]))[0]
    return "obstacle" if obst[i, j] else "free" if free[i, j] else "unknown"


def test_one_keyframe_marks_free_obstacle_footprint_and_leaves_the_rest_unknown():
    g = OccupancyGrid(size_m=8)
    assert g.update(0.0, (0, 0, 0), scan(WALL))
    assert status(g, (1.02, 0.0)) == "obstacle"
    assert status(g, (0.7, 0.0)) == "free"
    assert status(g, (0.05, 0.0)) == "free"      # under the robot
    assert status(g, (0.3, 0.0)) == "unknown"    # blind zone in front of the robot
    assert status(g, (0.0, 1.0)) == "unknown"    # outside the view


def test_image_bottom_obstacle_is_not_persisted_in_the_blind_zone():
    # obstacle filling the image bottom straight ahead, seen from several headings while turning on the spot:
    # only the seen point (0.4 m) is recorded, the blind zone between it and the robot stays unknown (writing
    # the "may be anywhere there" guess walled the robot in)
    g = OccupancyGrid(size_m=8)
    for k, yaw in enumerate(np.radians([0, 30, 60, 90])):
        s = scan(lambda b: 0.42 if abs(b) < 4 else np.inf)
        s.near[np.abs(BEAR) < 4] = True
        g.update(k * 1.0, (0, 0, yaw), s)
        if k == 0:
            assert status(g, (0.42, 0.0)) == "obstacle"  # the seen point (later views see it as floor)
    for yaw in np.radians([0, 30, 60, 90]):
        assert status(g, (0.25 * math.cos(yaw), 0.25 * math.sin(yaw))) == "unknown"


def test_keyframes_skip_repeated_views():
    g = OccupancyGrid(size_m=8)
    assert g.update(0.0, (0, 0, 0), scan())
    assert not g.update(0.1, (0.02, 0, 0), scan())        # barely moved, too soon
    assert g.update(0.2, (0.15, 0, 0), scan())            # moved 15 cm
    assert g.update(0.3, (0.15, 0, math.radians(10)), scan())  # turned 10 deg
    assert g.update(0.9, (0.15, 0, math.radians(10)), scan())  # 0.6 s later


def test_uncertain_far_evidence_needs_more_views():
    g = OccupancyGrid(size_m=8)
    g.update(0.0, (0, 0, 0), scan(), sigma=math.radians(2.0))
    assert status(g, (0.6, 0.0)) == "free"        # near: range error small, full weight
    assert status(g, (2.0, 0.0)) == "unknown"     # far: ~0.4 m range error, weak evidence
    for k in range(1, 30):
        g.update(k * 1.0, (0, 0, 0), scan(), sigma=math.radians(2.0))
    assert status(g, (2.0, 0.0)) == "free"


def test_max_range_ignores_far_cells():
    g = OccupancyGrid(size_m=8)
    g.update(0.0, (0, 0, 0), scan(lambda b: 1.52 if abs(b) < 4 else np.inf), max_range=1.0)
    assert status(g, (0.7, 0.0)) == "free"
    assert status(g, (1.3, 0.0)) == "unknown"
    assert status(g, (1.52, 0.0)) == "unknown"


def test_long_seen_obstacle_flips_after_a_few_free_views():
    g = OccupancyGrid(size_m=8)
    for k in range(20):
        g.update(k * 1.0, (0, 0, 0), scan(WALL))
    n = 0
    while status(g, (1.02, 0.0)) == "obstacle":
        n += 1
        g.update(20.0 + n, (0, 0, 0), scan())
    assert n <= 4  # clamp: certainty is bounded


def test_replay_rebuilds_the_same_grid():
    g = OccupancyGrid(size_m=8)
    g.update(0.0, (0, 0, 0), scan(WALL))
    g.update(1.0, (0.3, 0.1, 0.4), scan(), sigma=0.01)
    before = g.log_odds.copy()
    g.replay()
    assert np.array_equal(before, g.log_odds)


# --- blind zone: the map side of the prototype's blind-zone tests (the commands are the collision monitor's) ---

FOV, CLOSEST = 55.0, 0.44  # camera half-width (deg), closest visible floor (m)


def open_scan(far=2.0):
    """Floor visible from CLOSEST to `far` in the camera's sectors, no obstacle."""
    seen = np.abs(BEAR) < FOV
    return Scan(BEAR, seen, np.full(180, np.inf), np.zeros(180, bool), np.where(seen, far, 0.0),
                np.where(seen, CLOSEST, np.inf))


def side_near_scan(bearing=40.0):
    """Open floor, plus an obstacle touching the image bottom (near) in the sectors around `bearing` deg:
    e.g. a door frame at the image's bottom corner while passing a doorway."""
    s = open_scan()
    k = np.abs(BEAR - bearing) < 3
    s.obstacle[k], s.near[k], s.floor[k] = CLOSEST, True, CLOSEST
    return s


def approach_scan_with(fixed_xy, from_pose):
    """Open floor seen from from_pose, plus the obstacle at the fixed-frame point fixed_xy (in view)."""
    s = open_scan()
    dx, dy = fixed_xy[0] - from_pose[0], fixed_xy[1] - from_pose[1]
    d, b = np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx))
    k = np.abs(BEAR - b) < 1.5
    s.obstacle[k], s.floor[k] = d, d
    return s


DOOR = (CLOSEST * np.cos(np.radians(40.0)), CLOSEST * np.sin(np.radians(40.0)))  # side_near_scan's obstacle


def test_near_side_obstacle_is_known_once_the_map_has_seen_it_and_the_blind_zone():
    scan = side_near_scan()
    grid = OccupancyGrid()
    assert not grid.blind_known((0.0, 0.0, 0.0), scan).any()  # nothing seen yet (start): keep the guess
    assert grid.obstacle_points((0.0, 0.0, 0.0)).shape == (0, 2)
    # driving in: from 0.5 m back, the floor around the current position and the door frame were in view
    grid.update(0.0, (-0.5, 0.0, 0.0), approach_scan_with(DOOR, (-0.5, 0.0, 0.0)))
    known = grid.blind_known((0.0, 0.0, 0.0), scan)  # asked before the current frame goes in
    assert known[np.abs(BEAR - 40.0) < 3].any() and not known[~scan.near].any()
    # the map's points stand in for the guess: the door frame, in the robot frame
    pts = grid.obstacle_points((0.0, 0.0, 0.0))
    assert len(pts) and np.abs(pts - DOOR).max() < 0.05


def test_an_obstacle_appearing_in_a_seen_free_blind_zone_is_not_known():
    """Someone steps in front after the floor there was seen free. The map does not hold that obstacle, so the
    image-bottom guess stays (with 'seen' alone the robot drove through)."""
    grid = OccupancyGrid()
    for k, x in enumerate(np.arange(-1.0, 0.01, 0.1)):  # driving 1 m over open floor
        grid.update(0.5 * k, (x, 0.0, 0.0), open_scan())
    now = open_scan()
    ahead = np.abs(BEAR) < 8
    now.obstacle[ahead], now.near[ahead], now.floor[ahead] = CLOSEST, True, CLOSEST
    known = grid.blind_known((0.0, 0.0, 0.0), now)
    assert not known.any()
    assert len(grid.obstacle_points((0.0, 0.0, 0.0))) == 0


def test_an_obstacle_the_map_saw_in_the_blind_zone_is_in_its_points():
    """A box 0.3 m ahead, seen while approaching, now under the image bottom: the map's points hold it."""
    grid = OccupancyGrid()
    approach = open_scan()
    ahead = np.abs(BEAR) < 6
    approach.obstacle[ahead], approach.floor[ahead] = 0.8, 0.8  # from 0.5 m back: box at 0.8 m
    grid.update(0.0, (-0.5, 0.0, 0.0), approach)
    pts = grid.obstacle_points((0.0, 0.0, 0.0))
    assert len(pts) and np.abs(pts[:, 0] - 0.3).max() < 0.05 and np.abs(pts[:, 1]).max() < 0.15


def test_first_frame_is_a_keyframe():
    """The frames of a glance at START reach the grid before any step."""
    grid = OccupancyGrid()
    grid.update(0.0, (0.0, 0.0, 0.0), open_scan(), h=0.33, force=True)
    assert len(grid.keyframes) == 1


# --- binding ---


def test_attributes_and_frames():
    g = OccupancyGrid(size_m=8)
    assert (g.res, g.n, g.thr, g.x0, g.y0) == (0.05, 160, 0.2, None, None)
    assert OccupancyGrid().n == 480
    with pytest.raises(ValueError):
        g.cell(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError):
        g.centres(np.array([[0, 0]]))
    g.update(0.0, np.array([1.0, -2.0, 0.3]), scan())  # a pose may be any sequence
    assert (g.x0, g.y0) == (-3.0, -6.0)  # centred on the first pose
    ij = g.cell(np.array([[1.0, -2.0], [-3.01, 2.01]]))
    assert ij.dtype == np.int64 and ij.tolist() == [[80, 80], [-1, 160]]
    assert np.allclose(g.centres(ij), [[1.025, -1.975], [-3.025, 2.025]], atol=1e-12)
    assert g.log_odds.shape == (160, 160) and g.log_odds.dtype == np.float32
    g.log_odds[:] = 7  # a copy
    assert g.log_odds.max() < 1


def test_keyframe_poses_can_be_corrected_before_replay():
    g = OccupancyGrid(size_m=8)
    g.update(0.0, (0, 0, 0), scan(WALL))
    g.update(1.0, (0.3, 0.1, 0.4), scan())
    assert g.keyframes == [(0.0, 0.0, 0.0), (0.3, 0.1, 0.4)]
    with pytest.raises(ValueError):
        g.set_keyframe_poses([(0, 0, 0)])
    g.set_keyframe_poses([(0.0, 1.0, 0.0), (0.3, 0.1, 0.4)])  # first keyframe 1 m to the left: its wall follows
    assert g.keyframes[0] == (0.0, 1.0, 0.0)
    assert status(g, (1.02, 0.0)) == "obstacle"  # not before replay()
    g.replay()
    assert status(g, (1.02, 1.0)) == "obstacle" and status(g, (1.02, 0.0)) != "obstacle"


def test_the_keyframe_holds_a_copy_of_the_scan():
    g = OccupancyGrid(size_m=8)
    s = scan(WALL)
    g.update(0.0, (0, 0, 0), s)
    before = g.log_odds
    s.obstacle[:] = 0.6
    g.replay()
    assert np.array_equal(g.log_odds, before)


def test_reset_clears_the_map_and_the_next_update_recentres():
    g = OccupancyGrid(size_m=8)
    g.update(0.0, (0, 0, 0), scan(WALL))
    g.reset()
    assert g.x0 is None and g.keyframes == [] and not g.log_odds.any()
    assert g.update(0.0, (5.0, 5.0, 0.0), scan())  # a first keyframe again
    assert (g.x0, g.y0) == (1.0, 1.0)


def test_outside_the_grid_is_ignored_and_unknown():
    g = OccupancyGrid(size_m=8)
    g.update(0.0, (0, 0, 0), scan())
    before = g.log_odds
    assert g.update(1.0, (50.0, 0.0, 0.0), scan(WALL))  # 50 m away: nothing of it is inside
    assert np.array_equal(g.log_odds, before)
    assert g.obstacle_points((50.0, 0.0, 0.0)).shape == (0, 2)
    near = scan(WALL)
    near.near[np.abs(BEAR) < 4] = True
    assert not g.blind_known((50.0, 0.0, 0.0), near).any()


def test_a_scan_needs_two_sectors():
    one = Scan([0.0], [True], [np.inf], [False], [1.0], [0.4])
    with pytest.raises(ValueError):
        OccupancyGrid().update(0.0, (0, 0, 0), one)
