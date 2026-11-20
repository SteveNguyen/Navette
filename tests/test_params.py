"""Parameter validation: an invalid value is a ValueError naming the parameter where the object is built or
the parameters are used, never a hang (arc sampling with step = 0, an infinite look-ahead with decel = 0), a
panic or a huge allocation (a grid of size / 0 cells)."""

import math

import numpy as np
import pytest

import navette
from navette import (ArcController, ExploreParams, FreeSpace, FrontierExplore, FrontierParams, FrontierPlanner,
                     Navigator, OccupancyGrid, SafetyParams, Scan)
from navette.testing.fakes import pinhole_rays

NAN, INF = math.nan, math.inf
BEAR = (np.arange(180) + 0.5) * 2.0 - 180


def wall_scan(d=0.3, fov=38.0):
    seen = np.abs(BEAR) < fov
    return Scan(BEAR.copy(), seen, np.where(seen, d, np.inf), np.zeros(180, bool), np.where(seen, d, 0.0),
                np.where(seen, 0.25, np.inf))


SAFETY_BAD = [("step", 0.0), ("step", -0.02), ("step", NAN), ("decel", 0.0), ("decel", -1.0), ("decel", INF),
              ("latency", -0.1), ("latency", NAN), ("radius", -0.1), ("radius", INF), ("margin", -0.01),
              ("unseen_m", -0.1), ("unseen_m", NAN), ("v_max", 0.0), ("v_max", -0.4), ("reverse_max", -0.1),
              ("n_speeds", 0)]


@pytest.mark.parametrize("key, value", SAFETY_BAD)
def test_safety_params_are_checked_when_built(key, value):
    with pytest.raises(ValueError, match=f"SafetyParams.{key}"):
        SafetyParams(**{key: value})


@pytest.mark.parametrize("key, value", SAFETY_BAD)
def test_safety_params_set_afterwards_are_checked_where_they_are_used(key, value):
    """The attributes are settable: step = 0 or decel = 0 set after the constructor used to hang
    safe_command (an endless arc)."""
    p = SafetyParams()
    setattr(p, key, value)
    pts = np.array([[0.5, 0.0]])
    for use in (lambda: navette.safe_command(0.15, 0.0, wall_scan(), p),
                lambda: navette.free_arc_length(pts, 0.0, p),
                lambda: navette.free_arc_lengths(pts, [0.0, 1.0], p, 1.0),
                lambda: navette.stopping_distance(0.2, p),
                lambda: Navigator(p),
                lambda: Navigator(p, behaviour="forward"),
                lambda: ArcController(None, p),
                lambda: FrontierExplore(None, p)):
        with pytest.raises(ValueError, match=f"SafetyParams.{key}"):
            use()
    setattr(p, key, getattr(SafetyParams(), key))
    assert navette.safe_command(0.15, 0.0, wall_scan(1.0), p)[0] > 0  # valid again: used


@pytest.mark.parametrize("kw, message", [
    ({"res": 0.0}, "OccupancyGridParams.res"), ({"res": -0.05}, "res"), ({"res": NAN}, "res"),
    ({"size_m": 0.0}, "size_m"), ({"size_m": -1.0}, "size_m"), ({"size_m": NAN}, "size_m"),
    ({"size_m": 1e6}, "cells per side"), ({"res": 1e-6}, "cells per side"), ({"size_m": 0.01}, "cells per side"),
    ({"clamp": 0.0}, "clamp"), ({"clamp": NAN}, "clamp"), ({"thr": -0.2}, "thr"), ({"sigma_ref": 0.0}, "sigma_ref"),
    ({"key_m": -0.1}, "key_m"), ({"key_deg": -5.0}, "key_deg"), ({"key_s": NAN}, "key_s"),
    ({"robot_radius": -0.1}, "robot_radius"), ({"max_obstacle_m": NAN}, "max_obstacle_m"),
])
def test_occupancy_grid_params(kw, message):
    with pytest.raises(ValueError, match=message):
        OccupancyGrid(**kw)


def test_occupancy_grid_size_limits():
    assert OccupancyGrid(size_m=0.05).n == 1 and OccupancyGrid().n == 480
    assert OccupancyGrid(res=0.1, size_m=409.6).n == 4096  # the largest
    with pytest.raises(ValueError, match="4097"):
        OccupancyGrid(res=0.1, size_m=409.7)


@pytest.mark.parametrize("kw, message", [
    ({"sector_deg": 0.0}, "ScanParams.sector_deg"), ({"sector_deg": -2.0}, "sector_deg"),
    ({"sector_deg": 361.0}, "sector_deg"), ({"sector_deg": NAN}, "sector_deg"), ({"sector_deg": 1e-9}, "sector_deg"),
    ({"max_range": 0.0}, "max_range"), ({"max_range": -1.0}, "max_range"), ({"max_range": INF}, "max_range"),
    ({"near_tol": -0.1}, "near_tol"),
])
def test_free_space_params(kw, message):
    shape = (4, 6)
    rays = pinhole_rays(shape, 5.0)
    with pytest.raises(ValueError, match=message):
        FreeSpace(rays, rays, shape, **kw)
    assert FreeSpace(rays, rays, shape, sector_deg=360.0).n == 1


FRONTIER_BAD = [("replan_s", NAN), ("replan_s", -1.0), ("lookahead", NAN), ("lookahead", -0.6), ("clear", -0.1),
                ("soft_m", 0.0), ("soft_k", -1.0), ("blind", INF), ("min_frontier_m", NAN), ("turn_k", -0.5),
                ("target_timeout_s", 0.0), ("ban_s", NAN), ("ban_r", -0.5), ("keep_ratio", NAN)]


@pytest.mark.parametrize("key, value", FRONTIER_BAD)
def test_frontier_params(key, value):
    with pytest.raises(ValueError, match=f"FrontierParams.{key}"):
        FrontierParams(**{key: value})
    p = FrontierParams()
    setattr(p, key, value)  # set afterwards: caught where it is used
    for use in (lambda: FrontierPlanner(p), lambda: FrontierExplore(None, None, p), lambda: Navigator(frontier=p)):
        with pytest.raises(ValueError, match=f"FrontierParams.{key}"):
            use()


EXPLORE_BAD = [("v_set", []), ("v_set", [0.1, 0.0]), ("v_set", [-0.1]), ("v_set", [NAN]), ("n_w", 0),
               ("w_max", -1.2), ("w_max", NAN), ("clear_cap", 0.0), ("w_spin", 0.0), ("w_spin", NAN),
               ("blocked_s", NAN), ("blocked_s", -1.0), ("horizon_s", INF), ("k_heading", NAN),
               ("spin_done_deg", -1.0), ("moving_v", -0.05), ("moving_v", NAN),
               ("spin_flip_deg", -1.0), ("spin_lead_s", NAN), ("spin_accel", -0.1), ("spin_w_min", 0.0),
               ("spin_w_min", INF), ("progress_m", -0.05), ("progress_deg", NAN),
               ("recover_hold_s", -1.0), ("look360_m", -1.0), ("look360_m", NAN)]


@pytest.mark.parametrize("key, value", EXPLORE_BAD)
def test_explore_params(key, value):
    with pytest.raises(ValueError, match=f"ExploreParams.{key}"):
        ExploreParams(**{key: value})
    p = ExploreParams()
    setattr(p, key, value)
    for use in (lambda: ArcController(p), lambda: FrontierExplore(p), lambda: Navigator(explore=p)):
        with pytest.raises(ValueError, match=f"ExploreParams.{key}"):
            use()


def test_navigator_own_params():
    with pytest.raises(ValueError, match="forward_v"):
        Navigator(behaviour="forward", forward_v=NAN)
    with pytest.raises(ValueError, match="TiltTrust.tau_s"):
        Navigator(tilt_trust=navette.TiltTrust(tau_s=0.0))
    with pytest.raises(ValueError, match="behaviour"):
        Navigator(behaviour="backward")
    nav = Navigator(SafetyParams(n_speeds=1), explore=ExploreParams(v_set=[0.05], n_w=1))  # the smallest valid
    assert nav.step(0.0, 0.1, (0, 0, 0), wall_scan(2.0), 0.35, 0, 0, 0, 0).state in ("DRIVE", "SPIN", "LOOK")
