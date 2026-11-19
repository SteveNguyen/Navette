"""navette.UpFilter on synthetic IMU data: known orientation, gyro bias, linear acceleration.
The Python prototype's own tests, same assertions."""

import numpy as np
import pytest

from navette import G, UpFilter

RATE = 50.0
AXIS = np.array([0.8, -0.6, 0.0])  # nod axis: not a single IMU axis


def rodrigues(r):
    """Rotation matrix of the rotation vector r (axis r / |r|, angle |r|)."""
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def angle_deg(a, b):
    """Angle in degrees between rows of a and b (N, 3)."""
    a = np.asarray(a, float) / np.linalg.norm(a, axis=-1, keepdims=True)
    b = np.asarray(b, float) / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def run(imu_t, gyro, acc, query_t, **kw):
    """Filter a whole log, return 'up' (IMU axes) at each query time (causal estimate)."""
    f = UpFilter(**kw)
    ups = np.array([f.update(t, w, a) for t, w, a in zip(imu_t, gyro, acc)])
    out = np.array([[np.interp(q, imu_t, ups[:, j]) for j in range(3)] for q in query_t])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def simulate(seconds=60.0, amp=0.15, freq=0.5, bias=(0.0, 0.0, 0.0), lin_acc=0.0, seed=0, lead_s=0.0, rate=RATE):
    """Body rotates about a fixed axis by amp*sin(2 pi f (t - lead_s)) after lead_s still seconds,
    base tilted 10 deg.

    Returns t, gyro, acc (IMU = body axes) and the true 'up' in body axes."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / rate)
    tilt = rodrigues(np.radians([10.0, 0, 0]))
    on = t >= lead_s
    th = on * amp * np.sin(2 * np.pi * freq * (t - lead_s))
    dth = on * amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * (t - lead_s))
    lin_acc = on * lin_acc
    gyro, acc, up = [], [], []
    for k in range(len(t)):
        R_wb = tilt @ rodrigues(AXIS * th[k])
        u = R_wb.T @ np.array([0, 0, 1.0])
        # Linear acceleration of the IMU in world axes, in phase with the angular acceleration.
        a_lin = lin_acc[k] * np.array([np.sin(2 * np.pi * freq * (t[k] - lead_s)), 0, 0])
        gyro.append(AXIS * dth[k] + np.asarray(bias) + rng.normal(0, 0.002, 3))
        acc.append(R_wb.T @ (np.array([0, 0, G]) + a_lin) + rng.normal(0, 0.02, 3))
        up.append(u)
    return t, np.array(gyro), np.array(acc), np.array(up)


def test_converges_from_wrong_start():
    t, w, a, up = simulate(seconds=120, amp=0.0)  # > a few bias_tau_s: the start error leaks into b first
    f = UpFilter(tau_s=1.0)
    f.update(t[0], w[0], a[0])
    f.u = rodrigues(np.radians([0, 20.0, 0])) @ f.u  # 20 deg off
    for k in range(1, len(t)):
        f.update(t[k], w[k], a[k])
    assert angle_deg(f.u[None], up[-1][None])[0] < 0.1


def test_tracks_bias_perpendicular_to_gravity():
    bias = np.array([0.01, -0.008, 0.005])  # ~0.6 deg/s
    t, w, a, up = simulate(seconds=200, amp=0.0, bias=bias)
    f = UpFilter(tau_s=1.0, bias_tau_s=10.0)
    for k in range(len(t)):
        f.update(t[k], w[k], a[k])
    u = up[-1]
    perp = lambda v: v - (v @ u) * u  # noqa: E731  only this part is observable
    assert np.linalg.norm(perp(f.b) - perp(bias)) < 0.001
    assert angle_deg(f.u[None], u[None])[0] < 0.1


def test_learns_full_bias_when_still():
    bias = np.radians([0.39, 0.16, -0.09])  # Reachy Mini's head gyro bias
    t, w, a, up = simulate(seconds=5, amp=0.0, bias=bias)
    f = UpFilter(tau_s=8.0)
    for k in range(len(t)):
        f.update(t[k], w[k], a[k])
    assert np.linalg.norm(f.b - bias) < np.radians(0.02)  # all 3 axes, incl. about the vertical


def test_beats_accelerometer_under_linear_acceleration():
    # 1 m/s^2 linear acceleration (~6 deg accelerometer error) while nodding.
    t, w, a, up = simulate(seconds=60, amp=0.1, freq=1.0, lin_acc=1.0)
    est = run(t, w, a, t, tau_s=2.0)
    err_f = angle_deg(est[500:], up[500:])
    err_a = angle_deg(a[500:], up[500:])
    assert np.percentile(err_f, 95) < 0.3 * np.percentile(err_a, 95)


def test_rate_gate_blocks_accelerometer_leak_during_fast_rotation():
    # Fast nod (peak ~57 deg/s) with 3 m/s^2 linear acceleration: the accelerometer is badly off,
    # the gyro is clean. With the rate gate the estimate follows the gyro. Starts still (as the
    # robot does): motion from t = 0 would let the first 0.25 s of bad accelerometer into the bias.
    # 400 Hz as a FIFO stream: at 50 Hz the gyro integration itself is off by ~1 deg here, and the
    # gate is only meant for a good gyro.
    t, w, a, up = simulate(seconds=30, amp=0.15, freq=1.0, lin_acc=3.0, lead_s=3.0, rate=400.0)
    m = t > 4.0
    err_open = angle_deg(run(t, w, a, t, tau_s=1.0)[m], up[m])
    err_gate = angle_deg(run(t, w, a, t, tau_s=1.0, rate_gate=np.radians(30))[m], up[m])
    assert np.percentile(err_gate, 95) < 0.5 * np.percentile(err_open, 95)


def test_up_at_interpolates_history():
    t, w, a, up = simulate(seconds=5, amp=0.1, freq=1.0)
    f = UpFilter(tau_s=1.0)
    ups = [f.update(t[k], w[k], a[k]) for k in range(len(t))]
    q = t[-10] + 0.5 / RATE
    expect = (ups[-10] + ups[-9]) / 2
    assert angle_deg(f.up_at(q)[None], expect[None])[0] < 1e-6


# ---- the Python API itself


def test_parameters_and_state_attributes():
    f = UpFilter()
    assert (f.kp, f.ki, f.acc_gate, f.g, f.history_s) == (1.0, 1.0 / 20.0, 0.1, G, 0.5) and G == 9.81
    assert (f.still_rate, f.still_acc, f.still_s, f.still_bias_tau_s) == (np.radians(2.0), 0.02, 0.5, 1.0)
    assert (f.rate_gate, f.rate_tau_s, f.rate_lp) == (None, 1.0, 0.0)
    assert f.u is None and f.t is None and f.still_since is None and np.array_equal(f.b, np.zeros(3))
    with pytest.raises(ValueError):
        f.up_at(0.0)  # no sample yet
    f = UpFilter(tau_s=2.0, bias_tau_s=5.0, g=9.9, bias0=[0.1, 0.2, 0.3], rate_gate=0.5)
    assert (f.kp, f.ki, f.g, f.rate_gate) == (0.5, 0.1, 9.9, 0.5) and np.array_equal(f.b, [0.1, 0.2, 0.3])
    u = f.update(3.0, np.zeros(3), np.array([0.0, 0.0, 9.9]))
    assert np.array_equal(u, [0, 0, 1]) and np.array_equal(f.u, u) and f.t == 3.0
    u[0] = 5.0  # a copy: the filter's state is not touched
    assert np.array_equal(f.u, [0, 0, 1])
    f.b = np.zeros(3)
    assert np.array_equal(f.b, np.zeros(3))


def test_samples_that_are_not_measurements_are_ignored():
    """A zero or non-finite sample used to make 'up' NaN for good (and a NaN 'up' makes the scan see nothing)."""
    up, zero = np.array([0.0, 0.0, G]), np.zeros(3)
    f = UpFilter()
    for acc in (zero, np.full(3, np.nan), np.array([np.inf, 0, 0])):
        assert f.update(0.0, zero, acc) is None and f.u is None and f.t is None  # no 'up' yet
    with pytest.raises(ValueError):
        f.up_at(0.0)
    assert np.array_equal(f.update(0.01, zero, up), [0, 0, 1])  # the first valid sample starts the filter
    for k, (gyro, acc) in enumerate([(np.full(3, np.nan), up), (zero, np.full(3, np.nan)), (zero, zero),
                                     (np.array([0, np.inf, 0]), up)]):
        u = f.update(0.02 + 0.01 * k, gyro, acc)
        assert np.isfinite(u).all() and np.isfinite(f.u).all() and np.isfinite(f.b).all()
    assert np.array_equal(f.up_at(1.0), [0, 0, 1])
