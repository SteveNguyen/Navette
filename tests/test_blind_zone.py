"""Blind-zone rule across the modules (the Python prototype's tests): the map explains an obstacle filling the
image bottom (OccupancyGrid.blind_known) and the collision monitor then keeps its visible distance
(safe_command near_known); the Navigator flags a stop caused only by the never-seen blind zone, and a glance
frame clears it."""

import numpy as np

from navette import Navigator, OccupancyGrid, SafetyParams, Scan, safe_command

BEAR = (np.arange(180) + 0.5) * 2.0 - 180
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


def test_near_side_obstacle_stops_without_the_map_and_not_once_the_map_has_seen_it_and_the_blind_zone():
    p = SafetyParams()
    scan = side_near_scan()
    assert safe_command(0.15, 0.0, scan, p)[0] == 0.0  # the guess: somewhere in the blind zone, at the edge
    grid = OccupancyGrid()
    assert not grid.blind_known((0.0, 0.0, 0.0), scan).any()  # nothing seen yet (start): keep the guess
    # driving in: from 0.5 m back, the floor around the current position and the door frame were in view
    grid.update(0.0, (-0.5, 0.0, 0.0), approach_scan_with(DOOR, (-0.5, 0.0, 0.0)))
    known = grid.blind_known((0.0, 0.0, 0.0), scan)  # asked before the current frame goes in
    assert known[np.abs(BEAR - 40.0) < 3].any() and not known[~scan.near].any()
    v = safe_command(0.15, 0.0, scan, p, grid.obstacle_points((0.0, 0.0, 0.0)), known)[0]
    assert v == 0.15


def test_an_obstacle_appearing_in_a_seen_free_blind_zone_still_stops():
    """Someone steps in front after the floor there was seen free. The map does not hold that obstacle, so the
    image-bottom guess stays (with 'seen' alone the robot drove through)."""
    p, grid = SafetyParams(), OccupancyGrid()
    for k, x in enumerate(np.arange(-1.0, 0.01, 0.1)):  # driving 1 m over open floor
        grid.update(0.5 * k, (x, 0.0, 0.0), open_scan())
    now = open_scan()
    ahead = np.abs(BEAR) < 8
    now.obstacle[ahead], now.near[ahead], now.floor[ahead] = CLOSEST, True, CLOSEST
    known = grid.blind_known((0.0, 0.0, 0.0), now)
    assert not known.any()
    assert safe_command(0.15, 0.0, now, p, grid.obstacle_points((0.0, 0.0, 0.0)), known)[0] == 0.0


def test_an_obstacle_the_map_saw_in_the_blind_zone_still_stops():
    """A box 0.3 m ahead, seen while approaching, now under the image bottom: the map's points stop the robot."""
    p, grid = SafetyParams(), OccupancyGrid()
    approach = open_scan()
    ahead = np.abs(BEAR) < 6
    approach.obstacle[ahead], approach.floor[ahead] = 0.8, 0.8  # from 0.5 m back: box at 0.8 m
    grid.update(0.0, (-0.5, 0.0, 0.0), approach)
    now = open_scan()
    now.obstacle[ahead], now.near[ahead], now.floor[ahead] = CLOSEST, True, CLOSEST
    known = grid.blind_known((0.0, 0.0, 0.0), now)
    assert safe_command(0.15, 0.0, now, p, grid.obstacle_points((0.0, 0.0, 0.0)), known)[0] == 0.0


def test_navigator_flags_a_blind_stop_and_a_glance_frame_clears_it():
    """forward + odometry: a near side obstacle with an empty map stops the robot, flagged blind_stop; a head-down
    frame (Navigator.observe) sees the floor up to the robot and the obstacle where it is; then it goes."""
    nav = Navigator(behaviour="forward", odometry=True, resume_n=1)
    out = nav.step(0.0, 0.1, (0.0, 0.0, 0.0), side_near_scan(), 0.37, 0.0, 0.0, 0.0, 0.0)
    assert out.v == 0.0 and out.blind_stop
    down = approach_scan_with(DOOR, (0.0, 0.0, 0.0))
    down.floor[down.seen & ~np.isfinite(down.obstacle)] = 0.6
    down.closest[down.seen] = 0.15  # head down: floor visible from 0.15 m
    nav.observe(0.5, (0.0, 0.0, 0.0), down, 0.33)
    out = nav.step(1.0, 0.1, (0.0, 0.0, 0.0), side_near_scan(), 0.37, 0.0, 0.0, 0.0, 0.0)
    assert out.v == 0.15 and not out.blind_stop


def test_a_stop_by_a_seen_obstacle_is_not_a_blind_stop():
    """An obstacle read right at the robot (not `near`: its distance is known) stops it: no blind_stop. Nor
    without odometry, where there is no map to ask."""
    wall = open_scan()
    ahead = np.abs(BEAR) < 20
    wall.obstacle[ahead], wall.floor[ahead] = 0.17, 0.17
    nav = Navigator(behaviour="forward", odometry=True, resume_n=1)
    out = nav.step(0.0, 0.1, (0.0, 0.0, 0.0), wall, 0.37, 0.0, 0.0, 0.0, 0.0)
    assert out.v == 0.0 and not out.blind_stop
    nav = Navigator(behaviour="forward", odometry=False, resume_n=1)
    out = nav.step(0.0, 0.1, (0.0, 0.0, 0.0), side_near_scan(), 0.37, 0.0, 0.0, 0.0, 0.0)
    assert out.v == 0.0 and not out.blind_stop


def test_observe_before_any_step():
    """The frames of a glance at START reach Navigator.observe before its first step."""
    nav = Navigator(behaviour="explore", odometry=True)
    nav.observe(0.0, (0.0, 0.0, 0.0), open_scan(), 0.33)
    assert len(nav.grid.keyframes) == 1
