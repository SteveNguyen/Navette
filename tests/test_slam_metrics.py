"""navette.slam.metrics on trajectories with known errors."""
import numpy as np
import pytest

from navette.slam import metrics as M


def square_loop(n_side=50, side=2.0):
    """A 2 x 2 m square driven counter-clockwise, heading along the motion, ending at the start: 4 * side m."""
    s = np.linspace(0, side, n_side, endpoint=False)
    legs = [np.stack([s, np.zeros_like(s)], 1), np.stack([np.full_like(s, side), s], 1),
            np.stack([side - s, np.full_like(s, side)], 1), np.stack([np.zeros_like(s), side - s], 1)]
    xy = np.vstack(legs + [[[0.0, 0.0]]])
    th = np.concatenate([np.full(n_side, a) for a in (0, np.pi / 2, np.pi, -np.pi / 2)] + [[0.0]])
    return np.column_stack([xy, th])


def odometry_with_yaw_drift(ref, rad_per_m):
    """ref's motion re-integrated with a heading error growing by rad_per_m per metre of travel."""
    est = [ref[0].copy()]
    for a, b in zip(ref[:-1], ref[1:]):
        m = M.relative(a, b)
        m[2] += rad_per_m * np.hypot(m[0], m[1])
        est.append(M.compose(est[-1], m))
    return np.array(est)


def test_compose_inverse_relative():
    a, b = np.array([1.0, 2.0, 0.7]), np.array([-0.3, 0.5, -2.0])
    assert M.compose(a, M.inverse(a)) == pytest.approx([0, 0, 0], abs=1e-12)
    assert M.compose(a, M.relative(a, b)) == pytest.approx(b)
    # a unit step forward from a pose facing +y lands at +1 in y
    assert M.compose([0, 0, np.pi / 2], [1, 0, 0]) == pytest.approx([0, 1, np.pi / 2], abs=1e-12)


def test_identical_trajectories_have_no_error():
    ref = square_loop()
    s = M.summary(ref, ref, delta=1.0)
    assert s["path_m"] == pytest.approx(8.0)
    for k in ("ate_rmse", "ate_max", "ate_anchored_rmse", "rpe_trans_mean", "rpe_rot_deg_per_m", "end_m",
              "end_rot_deg"):
        assert s[k] == pytest.approx(0, abs=1e-9), k
    assert M.return_to_start(ref) == pytest.approx((0, 0), abs=1e-9)


def test_rigidly_moved_copy_has_no_error_after_alignment():
    ref = square_loop()
    T = np.array([3.0, -1.5, 2.1])
    est = M.transform(T, ref)
    a = M.ate(est, ref)
    assert a["rmse"] == pytest.approx(0, abs=1e-9) and a["rot_rmse"] == pytest.approx(0, abs=1e-9)
    assert M.compose(a["T"], T) == pytest.approx([0, 0, 0], abs=1e-9)  # the alignment undoes T
    assert M.ate(est, ref, aligned=False)["rmse"] > 1.0
    # relative errors and anchored drift do not depend on where est starts
    assert M.rpe(est, ref, 0.5)["trans_max"] == pytest.approx(0, abs=1e-9)
    assert M.drift(est, ref)["end_m"] == pytest.approx(0, abs=1e-9)


def test_alignment_is_least_squares_without_scale():
    """A scaled copy is not made to fit: the remaining error is the scale's."""
    ref = square_loop()
    est = ref.copy()
    est[:, :2] *= 1.1
    a = M.ate(est, ref)
    assert a["T"][2] == pytest.approx(0, abs=1e-9)  # symmetric: no rotation helps
    centre = ref[:, :2].mean(0)
    d = np.hypot(*(ref[:, :2] - centre).T)
    assert a["rmse"] == pytest.approx(0.1 * np.sqrt(np.mean(d * d)), rel=1e-6)


def test_known_scale_error_gives_the_expected_rpe_and_drift():
    """Odometry over-reading distance by 2 %: RPE 2 % of delta on straight legs, end error ~2 % of a leg."""
    ref = square_loop(n_side=200)
    est = [ref[0].copy()]
    for a, b in zip(ref[:-1], ref[1:]):
        m = M.relative(a, b)
        m[:2] *= 1.02
        est.append(M.compose(est[-1], m))
    est = np.array(est)
    r = M.rpe(est, ref, 0.5)
    assert r["trans_max"] == pytest.approx(0.02 * 0.5, rel=0.05)
    assert r["rot_mean"] == pytest.approx(0, abs=1e-9)
    # the scaled square is the true one scaled about its start: the end returns to the start too
    assert M.drift(est, ref)["end_m"] == pytest.approx(0, abs=1e-6)
    assert M.return_to_start(est) == pytest.approx((0, 0), abs=1e-6)


def test_known_heading_drift_gives_the_expected_rotation_rpe():
    ref = square_loop(n_side=200)
    rate = np.radians(2.0)  # 2 deg per metre
    est = odometry_with_yaw_drift(ref, rate)
    r = M.rpe(est, ref, 1.0)
    assert r["rot_mean"] == pytest.approx(rate * 1.0, rel=0.02)
    assert r["rot_per_m"] == pytest.approx(rate, rel=0.02)
    d = M.drift(est, ref)
    assert np.degrees(d["end_rot"]) == pytest.approx(16.0, rel=0.01)  # 8 m at 2 deg/m
    assert d["pct"] == pytest.approx(100 * d["end_m"] / 8.0)
    assert d["end_m"] > 0.1
    # without ground truth, a loop ending at its start shows the same end error
    assert M.return_to_start(est) == pytest.approx((d["end_m"], d["end_rot"]), rel=1e-6)


def test_rpe_pairs_follow_the_reference_distance():
    ref = np.column_stack([np.arange(11) * 0.1, np.zeros(11), np.zeros(11)])  # 1 m in 0.1 m steps
    i, j = M.rpe_pairs(ref, 0.25)  # (not 0.3: a multiple of the step would compare float sums)
    assert list(i) == list(range(8)) and list(j - i) == [3] * 8
    assert M.rpe(ref, ref, 5.0)["n"] == 0  # shorter than delta: no pair
