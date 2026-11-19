"""navette's frontier planner against the Python prototype's outputs on the same inputs
(tests/golden/frontier_*.npz, see tests/golden/README.md).

frontier_fields.npz: the building blocks on random and structured grids: frontier_cells, the distance
transform (scipy.ndimage.distance_transform_edt) and the Dijkstra cost and predecessor fields
(scipy.sparse.csgraph.dijkstra on the prototype's 8-connected graph). All exact, the predecessors among
equal-cost paths included.
frontier_planner.npz: FrontierPlanner call sequences (plan with image-bottom sectors, ban, heading) on
hand-built rooms and random maps, with several parameter sets. Exact: done, the target, the path cell for
cell, the frontier cells, the number of bans; the heading (one atan2): 1e-12.
"""

import numpy as np
import pytest

from conftest import GOLDEN
from navette import FrontierParams, FrontierPlanner, OccupancyGrid, dijkstra, distance_transform, frontier_cells

PARAMS = ("clear", "soft_m", "soft_k", "blind", "min_frontier_m", "min_unknown_m2", "turn_k", "replan_s", "lookahead",
          "keep_ratio", "target_timeout_s", "ban_s", "ban_r")


@pytest.fixture(scope="module")
def fields():
    return dict(np.load(GOLDEN / "frontier_fields.npz"))  # every array read once


def test_frontier_cells(fields):
    g = fields
    assert g["fc_out"].any(axis=(1, 2)).sum() > 20  # cases with a frontier
    for k in range(len(g["fc_obst"])):
        fr = frontier_cells(g["fc_obst"][k], g["fc_free"][k], int(g["fc_min_cells"][k]), int(g["fc_min_unknown"][k]))
        assert fr.dtype == bool and np.array_equal(fr, g["fc_out"][k]), k
    # min_unknown_cells defaults to 0
    k = int(np.flatnonzero(g["fc_min_unknown"] == 0)[0])
    assert np.array_equal(frontier_cells(g["fc_obst"][k], g["fc_free"][k], int(g["fc_min_cells"][k])), g["fc_out"][k])


@pytest.mark.parametrize("shape", "abc")
def test_distance_transform(fields, shape):
    obst, want = fields[f"dt_obst_{shape}"], fields[f"dt_out_{shape}"]
    assert not obst[0].any()  # the case without any obstacle is in
    for k in range(len(obst)):
        d = distance_transform(obst[k])
        assert d.dtype == np.float64 and np.array_equal(d, want[k]), k


def test_dijkstra(fields):
    g = fields
    ties = 0
    for k in range(len(g["dj_trav"])):
        d, pred = dijkstra(g["dj_trav"][k], g["dj_cost"][k], float(g["dj_res"][k]), tuple(g["dj_start"][k]))
        assert np.array_equal(d, g["dj_d"][k]), k  # inf where not reached
        assert pred.dtype == np.int64 and np.array_equal(pred, g["dj_pred"][k]), k
        ties += count_ties(g["dj_trav"][k], g["dj_cost"][k], float(g["dj_res"][k]), d)
    assert ties > 300  # cells whose predecessor was decided by the order of the queue alone


def count_ties(trav, cost, res, d):
    """Number of cells reached at their cost through two neighbours that have the same cost themselves: there,
    the predecessor depends on the order in which equal costs leave the queue."""
    h, w = trav.shape
    same = np.zeros((h, w), bool)  # two neighbours giving the cell its cost are at the same cost
    first = np.full((h, w), np.nan)  # cost of the first neighbour giving the cell its cost
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) == (0, 0):
                continue
            a = (slice(max(0, -di), h - max(0, di)), slice(max(0, -dj), w - max(0, dj)))  # the cell
            b = (slice(max(0, di), h - max(0, -di)), slice(max(0, dj), w - max(0, -dj)))  # its neighbour
            with np.errstate(invalid="ignore"):
                through = d[b] + res * np.hypot(di, dj) * (cost[a] + cost[b]) / 2
                gives = trav[a] & trav[b] & np.isfinite(d[a]) & (through == d[a]) & (d[a] > 0)
            same[a] |= gives & (first[a] == d[b])
            first[a] = np.where(gives & np.isnan(first[a]), d[b], first[a])
    return int(same.sum())


def test_planner():
    g = dict(np.load(GOLDEN / "frontier_planner.npz"))
    grids = []
    for l in g["maps"]:
        grid = OccupancyGrid(res=float(g["res"]), size_m=float(g["size_m"]))
        grid.set_map(float(g["x0"]), float(g["y0"]), l)
        grids.append(grid)
    planners = {}
    assert g["done"].sum() > 20 and g["has_path"].sum() > 300 and g["n_banned"].max() >= 2
    for i, (c, kind, t, pose) in enumerate(zip(g["op_case"], g["op_kind"], g["op_t"].tolist(), g["op_pose"])):
        if c not in planners:
            planners[c] = FrontierPlanner(FrontierParams(**dict(zip(PARAMS, g["case_params"][c].tolist()))))
        pl, grid = planners[c], grids[g["case_map"][c]]
        if kind == 1:
            pl.ban(t)
        else:
            near = g["op_near"][i]
            pl.plan(t, tuple(pose), grid, near[np.isfinite(near)], float(g["op_near_half"][i]))
        what = (i, int(c))
        assert pl.done == g["done"][i], what
        assert len(pl.banned) == g["n_banned"][i], what
        if np.isnan(g["target"][i, 0]):
            assert pl.target is None, what
        else:
            assert np.array_equal(pl.target, g["target"][i]), what
        if g["has_path"][i]:
            assert np.array_equal(pl.path, g["path_xy"][g["path_ptr"][i]:g["path_ptr"][i + 1]]), what
        else:
            assert pl.path is None, what
        assert np.array_equal(pl.frontier, g["frontier_xy"][g["frontier_ptr"][i]:g["frontier_ptr"][i + 1]]), what
        for at, want in ((pose, g["heading"][i]), (g["pose2"][i], g["heading2"][i])):
            h = pl.heading(tuple(at))
            if np.isnan(want):
                assert h is None, what
            else:
                assert h == pytest.approx(want, rel=0, abs=1e-12), what
