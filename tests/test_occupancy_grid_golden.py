"""navette.OccupancyGrid against the Python prototype's outputs on the same sequences
(tests/golden/occupancy_grid_*.npz, see tests/golden/README.md).

A file is a grid configuration, a list of update() calls and the prototype's outputs at stages: "pre" (before
any update), "s<step>" (after that step) and "replay" (after the last step, with corrected keyframe poses).

Everything is compared exactly: the keyframe decisions, the float32 log-odds (same operations in the same
order), the masks, blind_known. A cell index comes from float64 sin / cos, which need not be bit-identical
between maths libraries: there, a sample on a cell border could change cell. On the machine that wrote the
goldens (x86-64, glibc) nothing differs. Points are float64 cell centres: 1e-9.
"""

import numpy as np
import pytest

from conftest import golden_files
from navette import OccupancyGrid, Scan

POINT_ATOL = 1e-9
PARAMS = ("res", "size_m", "clamp", "thr", "sigma_ref", "max_obstacle_m", "robot_radius", "key_m", "key_deg", "key_s")
SCAN_FIELDS = ("seen", "obstacle", "near", "floor", "closest")

FILES = golden_files("occupancy_grid")


def opt(x):
    """nan stands for None in the files."""
    return None if np.isnan(x) else float(x)


def check_stage(g, grid, stage):
    """The grid's state and query outputs against the stage's."""
    n = int(g["n"])
    want = np.zeros(n * n, np.float32)
    want[g[f"{stage}_lo_idx"]] = g[f"{stage}_lo_val"]
    got = grid.log_odds
    assert got.dtype == np.float32 and got.shape == (n, n)
    assert np.array_equal(got.reshape(-1), want), stage
    obst, free = grid.masks()
    assert obst.dtype == bool and obst.shape == (n, n) and free.shape == (n, n)
    assert np.array_equal(np.flatnonzero(obst), g[f"{stage}_obst_idx"]), stage
    assert np.array_equal(np.flatnonzero(free), g[f"{stage}_free_idx"]), stage
    for q, (pose, r0, explain_m) in enumerate(zip(g[f"{stage}_bq_pose"], g[f"{stage}_bq_r0"], g[f"{stage}_bq_explain"])):
        scan = Scan(g["bearing_deg"], *(g[f"{stage}_bq_{f}"][q] for f in SCAN_FIELDS))
        known = grid.blind_known(tuple(pose), scan, opt(r0), float(explain_m))
        assert known.dtype == bool and np.array_equal(known, g[f"{stage}_blind"][q]), (stage, q)
    ptr = g[f"{stage}_op_ptr"]
    for q, (pose, max_range) in enumerate(zip(g[f"{stage}_op_pose"], g[f"{stage}_op_range"])):
        pts = grid.obstacle_points(tuple(pose), float(max_range))
        want = g[f"{stage}_op_pts"][ptr[q]:ptr[q + 1]]
        # the same cells in the same order (by cell), so also the same set
        assert pts.shape == want.shape, (stage, q)
        np.testing.assert_allclose(pts, want, rtol=0, atol=POINT_ATOL, err_msg=str((stage, q)))


def test_golden_files_exist():
    assert len(FILES) >= 4


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_sequence(path):
    g = np.load(path)
    grid = OccupancyGrid(**{k: float(g[k]) for k in PARAMS})
    assert grid.n == int(g["n"]) and grid.res == float(g["res"]) and grid.thr == float(g["thr"])
    stage_at = dict(zip(g["stage_step"].tolist()[:-1], g["stages"].tolist()[:-1]))  # the last one is the replay
    assert grid.x0 is None and grid.y0 is None
    check_stage(g, grid, stage_at[-1])  # before any update
    for i, (t, pose, sigma, h, max_range, force) in enumerate(zip(g["t"], g["pose"], g["sigma"], g["h"],
                                                                  g["max_range"], g["force"])):
        scan = Scan(g["bearing_deg"], *(g[f][i] for f in SCAN_FIELDS))
        is_key = grid.update(float(t), tuple(pose), scan, opt(sigma), float(h), opt(max_range), bool(force))
        assert is_key == g["is_key"][i], i
        if i in stage_at:
            check_stage(g, grid, stage_at[i])
    assert (grid.x0, grid.y0) == (float(g["x0"]), float(g["y0"]))
    assert len(grid.keyframes) == g["is_key"].sum()
    assert np.array_equal(np.array(grid.keyframes), g["pose"][g["is_key"]])

    # frames
    ij = grid.cell(g["cell_xy"])
    assert ij.dtype == np.int64 and np.array_equal(ij, g["cell_ij"])
    assert np.array_equal(grid.centres(g["cell_ij"]), g["cell_centres"])

    # replay: the same grid from the same keyframes, another one from corrected poses
    before = grid.log_odds
    grid.replay()
    assert np.array_equal(grid.log_odds, before)
    grid.set_keyframe_poses(g["corrected_pose"])
    assert np.array_equal(np.array(grid.keyframes), g["corrected_pose"])
    grid.replay()
    assert g["stages"][-1] == "replay"
    check_stage(g, grid, "replay")

    # reset: as new
    grid.reset()
    assert grid.x0 is None and len(grid.keyframes) == 0 and not grid.log_odds.any()
    check_stage(g, grid, stage_at[-1])
