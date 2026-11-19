"""Frontier planner and frontier exploration on hand-built maps (the Python prototype's tests)."""

import math

import numpy as np

from navette import ExploreParams, FrontierExplore, FrontierParams, FrontierPlanner, OccupancyGrid, Scan, frontier_cells


def edit(g, change):
    """Change the grid's log-odds in place: change(l, x, y) with the cell centres x, y (n, n)."""
    c = g.centres(np.argwhere(np.ones((g.n, g.n), bool))).reshape(g.n, g.n, 2)
    l = g.log_odds
    change(l, c[..., 0], c[..., 1])
    g.set_map(g.x0, g.y0, l)


def room(door_w=0.8, open_right=False):
    """6 x 6 m grid, origin at the centre. Known free room x, y in [-1, 1] with walls; a door of width door_w in
    the wall at x = +1 leads to unknown space; optionally the wall at y = -1 is missing (unknown beyond)."""
    g = OccupancyGrid(size_m=6.0)
    g.set_map(-3.0, -3.0, np.zeros((g.n, g.n), np.float32))

    def build(l, x, y):
        inside = (np.abs(x) < 1) & (np.abs(y) < 1)
        wall = (np.abs(x) < 1.1) & (np.abs(y) < 1.1) & ~inside
        wall &= ~((x > 0.95) & (np.abs(y) < door_w / 2))
        if open_right:
            wall &= ~(y < -0.95)
        l[inside] = -2.0
        l[wall] = 2.0

    edit(g, build)
    return g


def test_frontier_is_the_door():
    g = room()
    obst, free = g.masks()
    fr = frontier_cells(obst, free, 5)
    pts = g.centres(np.argwhere(fr))
    assert len(pts) and np.all(pts[:, 0] > 0.9) and np.all(np.abs(pts[:, 1]) < 0.4)


def test_small_unknown_hole_is_no_frontier():
    g = room(door_w=0.0)
    i, j = g.cell(np.array([[0.3, 0.3]]))[0]

    def hole(l, x, y):
        l[i:i + 4, j:j + 4] = 0.0  # 0.2 x 0.2 m unknown hole in the free room

    edit(g, hole)
    obst, free = g.masks()
    assert frontier_cells(obst, free, 5).any()
    assert not frontier_cells(obst, free, 5, 40).any()


def test_commits_to_its_target():
    g = room(open_right=True)
    pl = FrontierPlanner()
    pl.plan(0.0, (0.0, 0.3, 0.0), g)  # facing the door, nearer to it
    first = pl.target.copy()
    assert first[0] > 0.9
    pl.plan(1.0, (0.0, -0.4, -math.pi / 2), g)  # now nearer to and facing the open side: keeps the door
    assert np.hypot(*(pl.target - first)) < 0.3


def test_plans_through_the_door_keeping_off_the_walls():
    g = room()
    pl = FrontierPlanner()
    pl.plan(0.0, (-0.5, 0.5, 0.0), g)
    assert not pl.done
    assert pl.target[0] > 0.9 and abs(pl.target[1]) < 0.4
    obst, _ = g.masks()
    walls = g.centres(np.argwhere(obst))
    d = np.min(np.hypot(pl.path[:, None, 0] - walls[None, :, 0], pl.path[:, None, 1] - walls[None, :, 1]), 1)
    assert d[np.hypot(*(pl.path - [-0.5, 0.5]).T) > 0.2].min() >= FrontierParams().clear - 1e-6
    h = pl.heading((-0.5, 0.5, 0.0))
    assert -math.pi / 2 < h < 0.1  # toward the door: right and ahead


def test_too_narrow_door_means_done():
    g = room(door_w=0.25)  # narrower than 2 x clear
    pl = FrontierPlanner()
    pl.plan(0.0, (0.0, 0.0, 0.0), g)
    assert pl.done and pl.path is None


def test_closed_room_is_done():
    g = room(door_w=0.0)
    pl = FrontierPlanner()
    pl.plan(0.0, (0.0, 0.0, 0.0), g)
    assert pl.done


def test_prefers_the_nearest_frontier_and_bans_move_on():
    g = room(open_right=True)  # door at x = +1 and the whole y = -1 side open
    pl = FrontierPlanner()
    pl.plan(0.0, (0.0, -0.4, 0.0), g)
    assert pl.target[1] < -0.9  # the open side is nearer than the door
    first = pl.target.copy()
    pl.ban(1.0)
    pl.plan(1.0, (0.0, -0.4, 0.0), g)
    assert np.hypot(*(pl.target - first)) >= FrontierParams().ban_r


def test_no_frontier_turns_around_once_before_done():
    g = room(door_w=0.0)  # closed room: nothing to explore
    fe = FrontierExplore()
    cmds = [fe.step(k * 0.1, (0.0, 0.0, 0.0), None, g, 0.0) for k in range(100)]
    n_look = sum(1 for v, w in cmds if v == 0.0 and w > 0)
    assert abs(n_look * 0.1 * fe.p.w_spin - 2 * math.pi) < 0.2  # one full turn
    assert fe.state == "DONE" and cmds[-1] == (0.0, 0.0)


def bottom_scan(half_deg=4.0, at=0.45):
    """Scan with an obstacle filling the image bottom within half_deg of straight ahead (Scan.near), free
    floor elsewhere up to 0.9 m (stays inside room())."""
    b = (np.arange(180) + 0.5) * 2.0 - 180
    seen = np.abs(b) < 38
    ahead = np.abs(b) < half_deg
    return Scan(b, seen, np.where(ahead, at, np.inf), ahead.copy(), np.full(180, 0.9), np.full(180, at))


def test_image_bottom_obstacle_in_the_current_scan_sends_the_path_around():
    # Robot at the origin facing the door (+x), floor around it never seen (blind disk unknown).
    # Without that obstacle the path leaves straight ahead through the blind disk; with the current scan's
    # image-bottom obstacle straight ahead it leaves sideways (planner and collision monitor agree).
    g = room()

    def blind_disk(l, x, y):
        l[np.hypot(x, y) < 0.5] = 0.0

    edit(g, blind_disk)
    pose = (0.0, 0.0, 0.0)

    def first_bearings(path):
        q = path[(np.hypot(*path.T) > 0.2) & (np.hypot(*path.T) < 0.45)]
        return np.degrees(np.arctan2(q[:, 1], q[:, 0]))

    pl = FrontierPlanner()
    pl.plan(0.0, pose, g)
    assert np.abs(first_bearings(pl.path)).min() < 10  # straight ahead
    fe = FrontierExplore()  # the behaviour passes the scan's image-bottom sectors
    fe.scan = bottom_scan()
    fe._plan(0.0, pose, g)
    assert fe.planner.path is not None and not fe.planner.done
    assert np.abs(first_bearings(fe.planner.path)).min() > 15  # around the obstacle


def test_recovery_turn_is_chosen_once_and_held():
    # blocked: turn 90 deg toward the side with more free floor in view, and keep that goal while turning
    # even if the view changes (re-choosing every step made it flip left/right in place)
    g = room()
    fe = FrontierExplore()
    s = bottom_scan()
    s.floor[s.bearing_deg < 0] = 0.3  # less free floor on the right
    fe.scan = s
    fe.choose_goal(0.0, (0.0, 0.0, 0.0), g)
    fe.on_blocked(0.5, (0.0, 0.0, 0.0))
    assert abs(fe.recover - math.pi / 2) < 1e-6  # left
    fe.scan = bottom_scan()  # view changes: right now as free as left
    for yaw in (0.2, 0.5, 0.9):
        assert fe.choose_goal(1.0, (0.0, 0.0, yaw), g) == fe.recover
    h = fe.choose_goal(2.0, (0.0, 0.0, math.pi / 2), g)  # reached: back to the planner's heading
    assert fe.recover is None and h == fe.planner.heading((0.0, 0.0, math.pi / 2))


def test_replanned_recovery_turns_toward_the_new_path_and_holds():
    # blocked while heading at 115 deg (the robot faces it): the prototype turns 90 deg to the freer side; with
    # recover_replan the recovery replans and turns toward the new path (the door, ~0 deg)
    g = room()
    up = (0.0, 0.0, 2.0)
    s = bottom_scan()
    s.floor[s.bearing_deg < 0] = 0.3  # less free floor on the right
    for replan, want in ((False, 2.0 + math.pi / 2 - 2 * math.pi), (True, None)):
        fe = FrontierExplore(ExploreParams(rotate_in_place=True, recover_replan=replan, recover_hold_s=2.0))
        fe.scan = s
        fe.on_blocked(0.5, up)
        h = fe.choose_goal(0.5, up, g)
        if want is None:  # the path's heading, ~0
            assert h == fe.planner.heading(up) and abs(h) < 0.2
        else:
            assert abs(h - want) < 1e-9
    # blocked toward the door (0 deg) and the replan says the same: 90 deg the freer way (no turn before),
    # then for recover_hold_s the goal stays there instead of turning back toward the door
    fe = FrontierExplore(ExploreParams(rotate_in_place=True, recover_replan=True, recover_hold_s=2.0))
    fe.scan = s
    fe.choose_goal(0.0, (0.0, 0.0, 0.0), g)
    fe.on_blocked(0.5, (0.0, 0.0, 0.0))
    assert abs(fe.choose_goal(0.5, (0.0, 0.0, 0.0), g) - math.pi / 2) < 1e-9
    left = (0.0, 0.0, math.pi / 2)
    assert abs(fe.choose_goal(4.0, left, g) - math.pi / 2) < 1e-9 and fe.recover is None  # turned; held
    assert fe.choose_goal(6.0, left, g) == fe.planner.heading(left)  # 2 s later: back to the path


def test_recovery_turn_is_not_interrupted_by_the_look_around():
    # planner done (closed room) while a recovery turn to the right is under way: keep turning right, no LOOK
    g = room(door_w=0.0)
    fe = FrontierExplore()
    fe.recover = -math.pi / 2
    for k in range(5):
        v, w = fe.step(k * 0.1, (0.0, 0.0, 0.0), None, g, 0.0)
        assert fe.planner.done and fe.state != "LOOK" and v == 0.0 and w < 0


def test_second_block_on_the_same_target_bans_it():
    g = room(open_right=True)
    fe = FrontierExplore()
    pose = (0.0, 0.0, 0.0)
    fe.choose_goal(0.0, pose, g)
    first = fe.planner.target.copy()
    fe.on_blocked(0.5, pose)
    assert fe.planner.banned == []  # first block: just the recovery turn
    fe.recover = None
    fe.on_blocked(1.0, pose)
    assert len(fe.planner.banned) == 1 and np.hypot(*(fe.planner.banned[0][1] - first)) < 1e-6


def test_target_timeout_bans_it_but_only_banned_ones_left_is_not_done():
    g = room()
    pl = FrontierPlanner(FrontierParams(target_timeout_s=5.0))
    pl.plan(0.0, (0.0, 0.0, 0.0), g)
    first = pl.target.copy()
    pl.plan(6.0, (0.0, 0.0, 0.0), g)  # not reached: banned; the door is the only frontier, so it is retried
    assert not pl.done and pl.banned == []
    assert np.hypot(*(pl.target - first)) < 0.5


def test_the_planner_object_of_the_behaviour_is_live():
    # fe.planner is the behaviour's planner, not a copy: a ban through it is seen by the next plan
    g = room(open_right=True)
    fe = FrontierExplore()
    fe.choose_goal(0.0, (0.0, -0.4, 0.0), g)
    planner, first = fe.planner, fe.planner.target.copy()
    planner.ban(0.5)
    assert len(fe.planner.banned) == 1 and fe.planner.path is None
    fe.choose_goal(1.0, (0.0, -0.4, 0.0), g)  # no path: plans again, away from the banned target
    assert np.hypot(*(planner.target - first)) >= FrontierParams().ban_r


def test_parameters_are_dataclass_like():
    p = FrontierParams(lookahead=0.4)
    assert (p.lookahead, p.clear, p.blind, p.replan_s, p.keep_ratio) == (0.4, 0.18, 0.5, 1.0, 0.0)
    p.ban_r = 0.3
    assert p == FrontierParams(lookahead=0.4, ban_r=0.3) and p != FrontierParams()
    assert FrontierPlanner(p).p == p and "ban_r=0.3" in repr(p)
