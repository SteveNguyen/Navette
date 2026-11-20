"""navette.slam.data (Sequence save / load, Scan rebuild), the simulated SLAM scenario (tests/sim_slam.py) and
navette.slam.extract on a synthetic run folder. Synthetic data only."""
import math

import numpy as np
import pytest
from sim_slam import V_MAX, scenario

from navette.slam import metrics as M
from navette.slam.data import SCAN_KEYS, Keyframe, Sequence, scan_dict, to_scan

PERFECT = dict(scale=0, yaw_scale=0, bias_deg_min=0, noise_s=0, noise_th=0, noise_th0_deg=0)


@pytest.fixture(scope="module")
def seq():
    return scenario(seed=0)


def assert_same(a, b):
    assert len(a) == len(b) and a.meta == b.meta
    for k in ("t", "odom", "cam_h", "cam_heading", "glance", "bearing_deg", "gt"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k), err_msg=k)
    for k in SCAN_KEYS:
        np.testing.assert_array_equal(a.scans[k], b.scans[k], err_msg=k)
        assert a.scans[k].dtype == b.scans[k].dtype


def test_save_load_round_trip(seq, tmp_path):
    seq.save(tmp_path / "s.npz")
    assert_same(Sequence.load(tmp_path / "s.npz"), seq)


def test_round_trip_without_ground_truth(tmp_path):
    s = scenario(seed=0)
    frames = [Keyframe(**{**f.__dict__, "gt": None}) for f in list(s)[:20]]
    frames[3].glance, frames[4].cam_heading = True, 0.05
    a = Sequence.from_keyframes(frames, {"source": "test"})
    a.save(tmp_path / "a.npz")
    b = Sequence.load(tmp_path / "a.npz")
    assert b.gt is None and b.glance[3] and b[4].cam_heading == 0.05
    assert b.scan_poses()[4, 2] == pytest.approx(b.odom[4, 2] + 0.05)
    assert b[4].scan_pose() == pytest.approx(tuple(b.scan_poses()[4]))


def test_frames_rebuild_a_navette_scan(seq):
    scan = seq.navette_scan(100)
    d = seq.scan(100)
    for k in ("bearing_deg",) + SCAN_KEYS:
        np.testing.assert_array_equal(np.asarray(getattr(scan, k)), d[k])
    np.testing.assert_array_equal(np.asarray(scan.free()), np.asarray(to_scan(scan_dict(scan)).free()))


def test_scenario_is_deterministic(seq):
    assert_same(scenario(seed=0), seq)
    other = scenario(seed=1)
    assert not np.array_equal(other.odom[: len(seq)][:100], seq.odom[:100])  # the seed changes the noise


def test_scenario_ground_truth_drive(seq):
    """A loop at <= 0.15 m/s back to the start, 8 Hz, spins of a full turn."""
    gt = seq.gt
    assert np.diff(seq.t) == pytest.approx(0.125)
    step = np.hypot(*np.diff(gt[:, :2], axis=0).T)
    assert step.max() <= V_MAX * 0.125 + 1e-9
    assert 10.0 < M.path_length(gt) < 14.0
    assert math.dist(gt[0, :2], gt[-1, :2]) < 0.1  # ends at its start: a loop closure
    spins = seq.meta["spins"]
    assert len(spins) >= 4 and spins[0][0] == 0
    for a, b in spins:
        turned = np.sum(M.wrap(np.diff(gt[a:b + 1, 2])))
        assert turned > 2 * math.pi * 0.99, (a, b)
        assert M.path_length(gt[a:b + 1]) < 0.1  # in place (a little coasting at its start)
    assert seq.meta["min_clearance"] > 0.2


def test_scenario_scans_see_the_walls(seq):
    seen, obst = seq.scans["seen"], np.isfinite(seq.scans["obstacle"])
    assert (seen.sum(1) == 34).all()  # sector centres within +-35 deg: -33, -31, ... 33
    assert (obst.sum(1) > 0).all() and np.median(obst.sum(1)) > 10
    assert (seq.scans["obstacle"][obst] >= 0.34).all() and (seq.scans["floor"][seen] <= 2.5).all()


def test_scenario_odometry_drifts_plausibly(seq):
    """Default drift model: a few % of the path at the end, not metres; perfect odometry is the truth."""
    s = M.summary(seq.odom, seq.gt)
    assert 0.5 < s["end_pct"] < 10
    assert 0.03 < s["ate_rmse"] < 0.5
    assert 2 < s["end_rot_deg"] < 30
    perfect = scenario(seed=0, odom_kw=PERFECT)
    assert M.summary(perfect.odom, perfect.gt)["ate_anchored_max"] < 1e-9


def test_extract_from_a_synthetic_run(tmp_path):
    """The extraction replays a run's perception without a navigator: the robot's stepped frames and glance
    frames, odometry poses, scans of the floor mask."""
    pytest.importorskip("cv2")
    from test_poussah_replay import FRAMES_T, R_CAM_IMU, ROBOT_TOML, V, X, all_floor, ideal_calib, make_run

    from navette import config
    from navette.rays import mask_rays
    from navette.robots.reachy_poussah import recording
    from navette.slam.extract import extract
    from navette.testing.fakes import FakeUpFilter

    run = tmp_path / "run"
    steps = {k: (0, 0.0, "GLANCE" if k == 4 else "DRIVE") for k in range(10) if k != 2}
    make_run(run, steps=steps)
    cfg = config.load(ROBOT_TOML).replace("camera", body_z=0.3)
    calib = ideal_calib(tmp_path)
    robot = recording.open_replay_robot(cfg, str(run), model=all_floor,
                                        calib=(R_CAM_IMU, X, mask_rays(calib), mask_rays(calib, dv=0.5)))
    seq = extract(robot, cfg, str(run), FakeUpFilter, log=lambda m: None)
    robot.close()
    assert len(seq) == 9 and seq.gt is None
    assert list(np.flatnonzero(seq.glance)) == [3]  # frame 4, the 4th kept
    t_cap = FRAMES_T[[k for k in range(10) if k != 2]] - 0.038  # capture time: arrival minus camera lag
    assert seq.t == pytest.approx(t_cap)
    assert seq.odom[:, 0] == pytest.approx(V * (FRAMES_T[[k for k in range(10) if k != 2]] - 9.6), abs=V * 0.021)
    assert seq.cam_h == pytest.approx(0.3 + 0.0512, abs=1e-3)
    assert seq.scans["seen"].any(1).all()  # an all-floor mask: floor seen ahead
    seq.save(tmp_path / "s.npz")
    assert_same(Sequence.load(tmp_path / "s.npz"), seq)
