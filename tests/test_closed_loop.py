"""Closed-loop properties with navette.Navigator itself driving the 2-D simulator (tests/sim2d.py), in the
rooms, with the seeds and the parameters of the golden sequences (tests/golden/navigator_*.npz), which hold
what the Python prototype reached there:

  - no collision, in any room;
  - exploration finishes where the prototype's does, within the same step budget;
  - the explored free area is within 5 % of the prototype's;
  - the forward behaviour stops before the box at the distance the collision monitor plans;
  - a stop caused by the never-seen blind zone is flagged, and a glance clears it.

Unlike tests/test_navigator_golden.py (recorded inputs replayed), the loop is closed here: a different command
changes every later frame.
"""

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
