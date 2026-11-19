"""navette.Navigator against the Python prototype's closed-loop sequences (tests/golden/navigator_*.npz, see
tests/golden/README.md): the prototype drove the 2-D simulator (tests/sim2d.py) in several rooms, with both
behaviours and two sets of SafetyParams; every input of every call was recorded with its output.

The recorded inputs are replayed through navette.Navigator and EVERY step is compared exactly (==, no
tolerance): the command (v, w), the request (v_req, w_req), the state, finished, blind_stop, and the
planner's target, path length and frontier size; then the final grid, bit for bit. No step is exempted.
The planner's equal-cost choices (Dijkstra predecessors, the cheapest frontier) follow the prototype's order,
see crates/navette-core/src/exploration/frontier.rs.

Float results come from the same operations in the same order, but sin / cos / atan2 / hypot need not be
bit-identical between maths libraries; on the machine that wrote the goldens (x86-64, glibc) nothing differs.
"""

import numpy as np
import pytest

import sim2d
from conftest import golden_files
from navette import Navigator, SafetyParams, Scan

FILES = golden_files("navigator")
SAFETY = ("radius", "margin", "v_max", "decel", "latency", "unseen_m", "reverse_max", "step")
OUT = ("v", "w", "v_req", "w_req", "state", "finished", "blind_stop")


def navigator(g):
    """The Navigator of a golden file."""
    safety = SafetyParams(**{k: float(g[k]) for k in SAFETY}, n_speeds=int(g["n_speeds"]))
    return Navigator(safety, behaviour=str(g["behaviour"]), odometry=bool(g["odometry"]), resume_n=int(g["resume_n"]),
                     forward_v=float(g["forward_v"]))


def scan_of(g, i):
    """The scan of frame i, from what the simulator's scans are stored as (the generator checked that it gives
    the scan back bit for bit)."""
    return Scan(**sim2d.scan_arrays(g["seen_idx"], g["floor"][i], float(g["closest"][i]), float(g["max_range"][i]),
                                    float(g["near_tol"])))


def replay(g, nav):
    """Replay the frames of a golden file through nav. Yields (frame index, NavOut) for the steps."""
    step_in = [g[k].tolist() for k in ("t", "dt", "cam_h", "a", "v", "w", "v_prev")]
    for i, (op, pose) in enumerate(zip(g["op"], g["pose"])):
        t, dt, cam_h, a, v, w, v_prev = (x[i] for x in step_in)
        if op == 1:
            nav.observe(t, tuple(pose), scan_of(g, i), cam_h)
        else:
            yield i, nav.step(t, dt, tuple(pose), scan_of(g, i), cam_h, a, v, w, v_prev)


def test_golden_files_exist():
    names = {p.stem for p in FILES}
    assert len(FILES) >= 12
    # both behaviours, the default and the live SafetyParams, a run to the end, a glance, no odometry
    assert {"navigator_small_default", "navigator_small_live", "navigator_forward_box_default",
            "navigator_forward_side_box_glance", "navigator_forward_no_odometry"} <= names


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_sequence(path):
    g = dict(np.load(path))
    nav = navigator(g)
    want = {k: g[f"out_{k}"].tolist() for k in OUT}
    differ = []  # (frame, what, navette, prototype)

    def check(i, what, got, ref):
        if got != ref:
            differ.append((i, what, got, ref))

    for i, out in replay(g, nav):
        for k in OUT:
            check(i, k, getattr(out, k), want[k][i])
        # the planner, for diagnosis: a difference here shows before it reaches a command
        target = nav.target
        check(i, "target", [np.nan, np.nan] if target is None else target.tolist(),
              g["target"][i].tolist() if np.isfinite(g["target"][i, 0]) else [np.nan, np.nan])
        path_points = nav.path
        check(i, "path_len", -1 if path_points is None else len(path_points), int(g["path_len"][i]))
        check(i, "n_frontier", len(nav.frontier), int(g["n_frontier"][i]))
    frames = sorted({d[0] for d in differ})
    assert not differ, f"{len(frames)} of {(g['op'] == 0).sum()} steps differ, first: {differ[:8]}"

    # the map built on the way
    grid = nav.grid
    lo = np.zeros(int(g["n"]) ** 2, np.float32)
    lo[g["lo_idx"]] = g["lo_val"]
    assert np.array_equal(grid.log_odds.reshape(-1), lo)
    if len(g["lo_idx"]):
        assert (grid.x0, grid.y0) == (float(g["x0"]), float(g["y0"]))
    obst, free = grid.masks()
    assert (int(free.sum()), int(obst.sum())) == (int(g["free_cells"]), int(g["obst_cells"]))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_sequences_cover_the_behaviours(path):
    """What each kind of sequence must exercise, so that an exact replay means something."""
    g = np.load(path)
    states, name = set(g["out_state"][g["op"] == 0].tolist()), path.stem
    if str(g["behaviour"]) == "forward":
        assert states == {"FORWARD"}
        moving = g["out_v"] > 0
        assert moving.any()
        if "side_box" not in name:
            assert not moving[g["op"] == 0][-1]  # ended stopped before the box
    else:
        assert {"DRIVE", "SPIN"} <= states and g["planned"].sum() > 20
    if any(room in name for room in ("small", "open", "wall")):  # explored to the end
        assert {"LOOK", "DONE"} <= states and g["finished_at"] > 0 and g["out_finished"].any()
    if "glance" in name:
        assert (g["op"] == 1).sum() == 2 and g["out_blind_stop"].any()
