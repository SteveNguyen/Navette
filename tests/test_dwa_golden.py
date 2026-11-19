"""navette.ArcController against the Python prototype's arc selection on the same inputs
(tests/golden/dwa_arcs.npz, see tests/golden/README.md): sequences of random scans over maps with random
obstacles, toward given goal headings, for several ExploreParams and SafetyParams.

Everything is compared exactly: the chosen arc (v, w) is one of the candidates, the same one on ties; the
state and the blocked timer.
"""

import numpy as np

from conftest import GOLDEN
from navette import ArcController, ExploreParams, OccupancyGrid, SafetyParams, Scan

EXPLORE = ("w_max", "n_w", "horizon_s", "k_heading", "k_clear", "k_speed", "clear_cap", "spin_above_deg",
           "spin_done_deg", "w_spin", "blocked_s")
SAFETY = ("radius", "margin", "v_max", "decel", "latency", "unseen_m", "reverse_max", "step")


def test_arc_sequences():
    g = dict(np.load(GOLDEN / "dwa_arcs.npz"))  # every array read once
    res, size_m = float(g["res"]), float(g["size_m"])
    controllers, grids = {}, {}
    arcs = spins = 0
    for i, q in enumerate(g["step_seq"]):
        if q not in controllers:
            explore = dict(zip(EXPLORE, g["seq_explore"][q].tolist()))
            explore["n_w"] = int(explore["n_w"])
            p = ExploreParams(v_set=tuple(g["seq_v_set"][q]), rotate_in_place=bool(g["seq_rotate"][q]), **explore)
            safety = SafetyParams(**dict(zip(SAFETY, g["seq_safety"][q].tolist())), n_speeds=int(g["seq_n_speeds"][q]))
            controllers[q] = ArcController(p, safety)
            grid = OccupancyGrid(res=res, size_m=size_m)
            l = np.zeros(grid.n * grid.n, np.float32)
            l[g["map_cells"][g["map_ptr"][q]:g["map_ptr"][q + 1]]] = 2.0
            grid.set_map(-size_m / 2, -size_m / 2, l.reshape(grid.n, grid.n))
            grids[q] = grid
        c = controllers[q]
        scan = Scan(g["bearing_deg"], g["seen"][i], g["obstacle"][i], g["near"][i], g["floor"][i], g["closest"][i])
        out = c.step(float(g["t"][i]), tuple(g["pose"][i]), scan, grids[q], float(g["v_applied"][i]), float(g["goal"][i]))
        what = (i, int(q))
        assert out == tuple(g["cmd"][i]), what
        assert c.state == g["state"][i], what
        assert c.goal == g["goal"][i], what
        want = g["blocked_since"][i]
        assert c.blocked_since is None if np.isnan(want) else c.blocked_since == want, what
        arcs += out[0] > 0
        spins += c.state == "SPIN"
    assert arcs > 300 and spins > 100  # both branches are exercised
    assert np.isfinite(g["blocked_since"]).sum() > 100
