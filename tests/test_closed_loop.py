"""Closed-loop properties with navette.Navigator itself driving the 2-D simulator (tests/sim2d.py), in the
rooms, with the seeds and the parameters of the golden sequences (tests/golden/navigator_*.npz), which hold
what the Python prototype reached there:

  - no collision, in any room;
  - exploration finishes where the prototype's does, within the same step budget;
  - the explored free area is within 5 % of the prototype's;
  - the forward behaviour stops before the box at the distance the collision monitor plans;
  - a stop caused by the never-seen blind zone is flagged, and a glance clears it;
  - following a box that moves away step by step, the restart hysteresis (resume_margin) turns the sub-second
    start-stop bursts into fewer, longer ones (not a golden: the prototype had none).
  - in a 1 m corridor with a turn and clutter, with the reference robot's config and its latency (commands
    0.46 s late, yaw rate lagging 0.2 s, speeds capped as the runtime does), the exploration switches of
    [explore] (near_known, latched turns, progress-based blocks, replanned recovery) turn less in place and
    reverse their turns less than the prototype's behaviour, with no collision and the same coverage;
  - single-frame near flicker (the floor model's false patches at the image bottom): with the reference robot's
    near persistence (2 of 3) the robot drives through it, and still stops for a box that fills the image bottom.

Unlike tests/test_navigator_golden.py (recorded inputs replayed), the loop is closed here: a different command
changes every later frame.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sim2d
from conftest import golden_files
from navette import Navigator, SafetyParams, Scan

SAFETY = ("radius", "margin", "v_max", "decel", "latency", "unseen_m", "reverse_max", "step")
FILES = golden_files("navigator")
EXPLORE = [p for p in FILES if "forward" not in p.stem]
FORWARD_BOX = [p for p in FILES if "forward" in p.stem and "side_box" not in p.stem]


def setup(path):
    """(reference, navigator, simulator) of a golden sequence: the same room, seed, noise and parameters."""
    g = dict(np.load(path))
    safety = SafetyParams(**{k: float(g[k]) for k in SAFETY}, n_speeds=int(g["n_speeds"]))
    nav = Navigator(safety, behaviour=str(g["behaviour"]), odometry=bool(g["odometry"]), resume_n=int(g["resume_n"]),
                    forward_v=float(g["forward_v"]))
    sim = sim2d.Sim(*sim2d.room(str(g["room"])), seed=int(g["seed"]), noise=float(g["noise"]))
    return g, nav, sim


def run(nav, sim, steps, glance=True):
    def step(t, dt, pose, scan, cam_h, a, v, w, v_prev):
        return nav.step(t, dt, pose, Scan(**scan), cam_h, a, v, w, v_prev)

    def observe(t, pose, scan, cam_h):
        nav.observe(t, pose, Scan(**scan), cam_h)

    return sim2d.run(sim, step, observe if glance else None, steps=steps)


def test_rooms():
    assert {p.stem.removeprefix("navigator_").rsplit("_", 1)[0] for p in EXPLORE} == {
        "open", "doorway", "clutter", "wall", "small"}
    assert len(FORWARD_BOX) >= 4


@pytest.mark.parametrize("path", EXPLORE, ids=lambda p: p.stem)
def test_exploration(path):
    g, nav, sim = setup(path)
    frames, summary = run(nav, sim, int(g["steps"]))
    assert not summary["collided"], summary
    assert summary["min_clearance"] > nav.safety.radius + 0.1  # well clear of the walls, as the prototype
    if g["finished_at"] >= 0:  # the prototype explored the room within the budget: so does navette
        assert summary["finished_at"] is not None, f"not finished in {int(g['steps'])} steps"
        assert frames[-1]["out"].state == "DONE" and nav.target is None
    free = int(nav.grid.masks()[1].sum())
    assert abs(free - int(g["free_cells"])) <= 0.05 * int(g["free_cells"]), (free, int(g["free_cells"]))
    assert summary["travelled"] > 3.0


@pytest.mark.parametrize("path", FORWARD_BOX, ids=lambda p: p.stem)
def test_forward_stops_before_the_box(path):
    g, nav, sim = setup(path)
    sp = nav.safety
    frames, summary = run(nav, sim, int(g["steps"]))
    outs = [f["out"] for f in frames if f["op"] == "step"]
    assert not summary["collided"] and summary["travelled"] > 1.0
    assert max(o.v for o in outs) == nav.forward_v  # reached the requested speed
    assert all(o.v == 0.0 for o in outs[-20:]) and abs(sim.v) < 1e-3  # and ended at rest
    # the box's near face is at x = 3.0; the robot drives along y = 1.5 toward it
    gap = 3.0 - sim.pose[0]
    assert abs(sim.pose[1] - 1.5) < 0.05 and abs(gap - sim.clearance()) < 0.01
    if nav.odometry:
        # The map explains the box once it is under the image, so the robot stops by the collision monitor's
        # plan: the centre never nearer than radius + margin, and no farther than that + the stopping
        # distance of the slowest speed candidate tried (forward_v / n_speeds) + the arc sampling step
        # + the range noise at 0.5 m.
        slack = sp.latency * nav.forward_v / sp.n_speeds + sp.step + 0.03
        assert sp.radius + sp.margin <= gap <= sp.radius + sp.margin + slack, gap
    else:
        # No map: the box filling the image bottom (within near_tol of the closest visible floor, 0.47 m) may
        # be anywhere in the blind zone, so the robot stops as soon as it does: one frame's travel past that.
        assert sim.closest - nav.forward_v * 2 * sim2d.DT - 0.02 <= gap <= sim.closest + sim.near_tol, gap
    assert gap == pytest.approx(float(g["min_clearance"]), abs=0.02)  # as the prototype


def test_blind_stop_is_flagged_and_a_glance_clears_it():
    path = next(p for p in FILES if "side_box" in p.stem)
    # without a glance: a box in the never-seen blind zone holds the robot for good, and says why
    g, nav, sim = setup(path)
    frames, summary = run(nav, sim, 40, glance=False)
    assert all(f["out"].blind_stop and f["out"].v == 0.0 and f["out"].v_req > 0 for f in frames)
    assert summary["travelled"] == 0.0
    # with one: the frames looking down put the box and the floor around it in the map; the robot goes
    g, nav, sim = setup(path)
    frames, summary = run(nav, sim, int(g["steps"]))
    ops = [f["op"] for f in frames]
    assert ops[:4] == ["step", "observe", "observe", "step"] and ops.count("observe") == 2
    outs = [f["out"] for f in frames if f["op"] == "step"]
    # Passing the box, a sector the glance did not cover holds the robot again for a moment (too close to the
    # last glance for another one); it clears as the box leaves the view.
    blind = [o.blind_stop for o in outs]
    assert blind[0] and not blind[1] and sum(blind[1:]) <= 3
    assert outs[1].v == nav.forward_v and not summary["collided"]
    assert sim.pose[0] > 1.56 + sim.radius  # past the box (x up to 1.56), 0.25 m to its left
    assert summary["min_clearance"] == pytest.approx(0.25, abs=0.01)
    assert summary["travelled"] == pytest.approx(float(g["travelled"]), abs=0.02)  # as the prototype


def follow_receding_box(resume_margin):
    """Forward (0.15 m/s, resume_n = 4 as the reference robot) toward a box that moves 5 cm away every second
    from t = 1 s: the robot catches up, stops at the edge of the blocked zone, and starts again as the box
    recedes. Scan alone (odometry=False: a robot at rest adds no keyframe, so with the map a moved box would
    stay in it), and a camera seeing the floor down to 0.1 m (sim closest), so that the collision monitor's
    clearance decides the stops, not the near rule (binary, whatever the margin).
    Returns (burst durations in s: runs of steps with v > 0, the run summary)."""
    walls, _ = sim2d.room("box")
    sim = sim2d.Sim(walls, (2.2, 1.5, 0.0), closest=0.1)  # box face 0.8 m ahead
    box = sim.walls[4:].copy()
    nav = Navigator(behaviour="forward", odometry=False, resume_n=4, resume_margin=resume_margin)

    def step(t, dt, pose, scan, cam_h, a, v, w, v_prev):
        out = nav.step(t, dt, pose, Scan(**scan), cam_h, a, v, w, v_prev)
        sim.walls[4:] = box + np.array([0.05, 0.0, 0.05, 0.0]) * int(t)  # for the next frame
        return out

    frames, summary = sim2d.run(sim, step, steps=200)
    moving = np.array([f["out"].v > 0 for f in frames], int)
    edges = np.flatnonzero(np.diff(np.r_[0, moving, 0]))  # starts and ends of the runs of moving steps
    return (edges[1::2] - edges[::2]) * sim2d.DT, summary


def test_restart_hysteresis_avoids_short_bursts():
    plain, plain_run = follow_receding_box(0.0)
    assert not plain_run["collided"]
    # without it: stop-and-go, a burst under 0.5 s for each 5 cm step of the box (the worst case of a
    # balancing base, which overshoots most after a short burst)
    assert len(plain) > 15 and plain[1:].max() < 0.5, plain
    hyst, hyst_run = follow_receding_box(0.1)
    assert not hyst_run["collided"]
    # with 0.10 m: it waits for 10 cm more room (two steps of the box), then drives: no burst under 1 s
    assert 2 <= len(hyst) < len(plain) / 2 and hyst.min() >= 1.0, hyst
    # and it keeps up with the box: within the 10 cm it waits for
    assert hyst_run["travelled"] > plain_run["travelled"] - 0.1


# ------------------------------------------------------------------ the exploration switches, with latency

ROBOT_TOML = Path(__file__).parents[1] / "robots" / "reachy_poussah.toml"
# the [explore] switches at the core's defaults: the prototype's behaviour
PROTOTYPE = {"near_known": False, "spin_flip_deg": 0.0, "spin_lead_s": 0.0, "spin_accel": 0.0, "blocked_s": 1.0,
             "blocked_progress": False, "recover_replan": False, "recover_hold_s": 0.0}
SWITCHES = {"A": ("near_known",), "B": ("spin_flip_deg", "spin_lead_s", "spin_accel"),
            "D": ("blocked_s", "blocked_progress"), "C": ("recover_replan", "recover_hold_s")}
DELAY = 4  # frames: commands reach the base ~0.46 s after their frame (the robot: 0.4-0.9 s)
W_LAG_S = 0.2  # s, the yaw rate's first-order lag


def corridor():
    """An L-shaped corridor 1 m wide (4.5 m, a left turn, 3 m) into a 3 x 2 m room; small boxes (12 cm: bottles,
    shoes) along the walls, and one in the room. Start at its closed end, facing along it."""
    outline = sim2d.polygon((0, 0), (4.5, 0), (4.5, 4.0), (5.5, 4.0), (5.5, 6.0), (2.5, 6.0), (2.5, 4.0), (3.5, 4.0),
                            (3.5, 1.0), (0, 1.0))
    s = 0.12
    clutter = [sim2d.rect(1.6, 0.0, 1.6 + s, s), sim2d.rect(2.7, 1.0 - s, 2.7 + s, 1.0),
               sim2d.rect(4.5 - s, 2.0, 4.5, 2.0 + s), sim2d.rect(3.5, 3.0, 3.5 + s, 3.0 + s),
               sim2d.rect(4.2, 5.2, 4.5, 5.5)]
    return np.vstack([outline, *clutter]), (0.5, 0.5, 0.0)


def explore_corridor(switches, steps=2000, seed=1):
    """The reference robot's navigator (loop.new_navigator) with these [explore] switches in the corridor: noisy
    ranges, the camera's blind zone (floor seen from 0.47 m), DELAY and W_LAG_S, commands capped as the runtime's
    controller does (cap_speed to controller.v_max, w_max). Returns the metrics until finished (or `steps`)."""
    from navette import config
    from navette.runtime import loop
    from navette.runtime.controller import cap_speed

    cfg = config.load(ROBOT_TOML)
    cfg = cfg.replace("explore", **switches)
    nav = loop.new_navigator(cfg, "explore", cfg.behaviour.forward_v, True)
    sim = sim2d.Sim(*corridor(), seed=seed, noise=0.01, w_lag_s=W_LAG_S)
    cmds = []

    def step(t, dt, pose, scan, cam_h, a, v, w, v_prev):
        out = nav.step(t, dt, pose, Scan(**scan), cam_h, a, v, w, v_prev)
        v, w = cap_speed(out.v, out.w, cfg.controller.v_max, cfg.controller.w_max)
        cmds.append((v, w))
        return SimpleNamespace(v=v, w=w, finished=out.finished, blind_stop=out.blind_stop)

    _, summary = sim2d.run(sim, step, steps=steps, delay=DELAY)
    n = len(cmds) if summary["finished_at"] is None else summary["finished_at"] + 1
    turns = [np.sign(w) for v, w in cmds[:n] if v == 0.0 and w != 0.0]  # in place
    return dict(reversals=int(sum(a != b for a, b in zip(turns, turns[1:]))), spin=len(turns) / n,
                blocks=nav.blocks, free=int(nav.grid.masks()[1].sum()),
                finished_s=None if summary["finished_at"] is None else summary["finished_at"] * sim2d.DT,
                collided=summary["collided"], min_clearance=summary["min_clearance"])


def test_exploration_switches_turn_less_in_a_cluttered_corridor_with_latency():
    """Off (the core's defaults, the prototype's behaviour) against on (the reference robot's config). Measured
    when this was written: reversals 8 -> 3, time turning in place 39 % -> 16 %, blocks 6 -> 0, free cells
    4850 -> 4858, finished in 137 -> 171 s (on drives more: 17 m instead of 11), no collision."""
    off, on = explore_corridor(PROTOTYPE), explore_corridor({})
    print(f"\noff {off}\non  {on}")
    assert not off["collided"] and not on["collided"]
    assert on["min_clearance"] > 0.13 + 0.05  # the collision monitor's clearance (radius + margin) held
    assert on["reversals"] < off["reversals"] and on["spin"] < off["spin"]
    assert on["blocks"] < off["blocks"]
    assert on["free"] >= 0.95 * off["free"]
    assert off["finished_s"] is not None and on["finished_s"] is not None


def forward_with_flicker(near_k, near_n, seed=3, p=0.3):
    """The reference robot's navigator, forward behaviour, scan alone (odometry=False: no map to explain the box,
    so the near rule decides the final stop), toward the box 2.2 m ahead. On a random third of the frames, never
    two within 3 frames, a false patch fills the image bottom for that frame only (the floor model's flicker): 2
    to 3 sectors at a random bearing within 30 deg of straight ahead read near, at the closest visible floor.
    Returns (steps' v, the gap to the box at each step, frames with a flicker, the run summary)."""
    from navette import config
    from navette.runtime import loop

    cfg = config.load(ROBOT_TOML).replace("perception", near_k=near_k, near_n=near_n)
    nav = loop.new_navigator(cfg, "forward", cfg.behaviour.forward_v, False)
    sim = sim2d.Sim(*sim2d.room("box"), seed=1, noise=0.01)
    rng = np.random.default_rng(seed)
    vs, gaps, flickers, last = [], [], [], 2  # the first flicker from frame 5 on

    def step(t, dt, pose, scan, cam_h, a, v, w, v_prev):
        nonlocal last
        k = len(vs)
        if k - last >= 3 and rng.random() < p:
            last = k
            flickers.append(k)
            b = rng.uniform(-30.0, 26.0)
            s = (sim2d.BEARING_DEG >= b) & (sim2d.BEARING_DEG <= b + 4.0) & scan["seen"]
            scan = {**scan, "obstacle": np.where(s, scan["closest"], scan["obstacle"]),
                    "floor": np.where(s, scan["closest"], scan["floor"]), "near": scan["near"] | s}
        out = nav.step(t, dt, pose, Scan(**scan), cam_h, a, v, w, v_prev)
        vs.append(out.v)
        gaps.append(3.0 - pose[0])
        return out

    _, summary = sim2d.run(sim, step, steps=200)
    return np.array(vs), np.array(gaps), flickers, summary


def test_near_persistence_ignores_a_single_frame_flicker_and_still_stops_for_a_real_obstacle():
    """Floor-model flicker (single-frame near patches) on the way to a box. Without near persistence (1 of 1,
    the core's default) the first flicker stops the robot far from the box, and every later one again; with
    the reference robot's 2 of 3, the robot drives through them without a stop and stops where the box fills
    the image bottom, one frame later at most, without touching it."""
    sim = sim2d.Sim(*sim2d.room("box"))
    reach = sim.closest + sim.near_tol  # the box fills the image bottom from this gap on
    v_off, gap_off, flick_off, off = forward_with_flicker(1, 1)
    v_on, gap_on, flick_on, on = forward_with_flicker(2, 3)
    assert not off["collided"] and not on["collided"]
    assert len(flick_on) > 10  # every one of them is in the robot's way or near it
    first_stop = lambda v: int(np.flatnonzero((v == 0.0) & (np.arange(len(v)) > 0))[0])  # noqa: E731
    # off: stopped by a flicker, more than 1 m from the box
    k = first_stop(v_off)
    assert k in flick_off and gap_off[k] > 1.0
    # on: no stop before the box, then stopped by it for good
    k = first_stop(v_on)
    assert gap_on[k] <= reach and (v_on[k:] == 0.0).all()
    # one frame (2 of 3) after the box first filled the image bottom, at 0.15 m/s: within 2 frames' travel
    assert gap_on[-1] >= reach - 2 * 0.15 * sim2d.DT - 0.02 - 0.03, gap_on[-1]  # 0.03: range noise at 0.5 m
    assert on["min_clearance"] > 0.13 + 0.05 + 0.1


@pytest.mark.parametrize("group", SWITCHES)
def test_each_exploration_switch_alone_is_collision_free(group):
    switches = {k: v for k, v in PROTOTYPE.items() if k not in SWITCHES[group]}  # this group as configured
    r = explore_corridor(switches)
    assert not r["collided"] and r["min_clearance"] > 0.18, r
    assert r["free"] >= 0.95 * explore_corridor(PROTOTYPE)["free"], r
