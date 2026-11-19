"""navette.TiltTrust: the Python prototype's own tests (same assertions), and its outputs on random base
motions (tests/golden/tilt_trust_motion.npz, see tests/golden/README.md)."""

import math

import numpy as np

from conftest import golden_files
from navette import TiltTrust

H = 0.35

# sigma, motion, range: exact (0.0) on the machine that wrote the goldens (atan, exp, sqrt of the same libm)
RTOL = 1e-12


def test_range_error_at_the_trusted_range_equals_the_tolerance():
    tt = TiltTrust()
    s = tt.update(0.1, 0.0, 0.0, 0.0)
    d = tt.max_range(H)
    assert abs((d * d + H * H) / H * s - tt.tol_m) < 1e-9
    assert 1.8 < d < 2.1  # 0.5 deg floor, 10 cm: ~2 m


def test_acceleration_shrinks_the_range_then_it_recovers():
    tt = TiltTrust()
    tt.update(0.1, 0.0, 0.0, 0.0)
    rest = tt.max_range(H)
    tt.update(0.1, 0.5, 0.2, 0.0)
    accel = tt.max_range(H)
    assert accel < 0.5 * rest
    for _ in range(10):  # 1 s later: partly recovered (filter time constant), not yet back
        tt.update(0.1, 0.0, 0.3, 0.0)
    assert accel < tt.max_range(H) < rest
    for _ in range(50):
        tt.update(0.1, 0.0, 0.3, 0.0)
    assert abs(tt.max_range(H) - rest) < 0.05


def test_turning_counts_like_a_sideways_acceleration():
    a, b = TiltTrust(), TiltTrust()
    a.update(0.1, 0.3, 0.0, 0.0)
    b.update(0.1, 0.0, 0.3, 1.0, gyro_rate=0.0)  # centripetal 0.3 m/s^2
    assert math.isclose(a.sigma, b.sigma)


def test_attributes():
    tt = TiltTrust()
    assert (tt.s0, tt.tau_s, tt.tol_m, tt.timing_s, tt.motion) == (math.radians(0.5), 1.0, 0.10, 0.01, 0.0)
    assert tt.sigma == tt.s0  # usable before the first update
    tt = TiltTrust(sigma0_deg=1.0, tau_s=2.0, tol_m=0.2, timing_s=0.0)
    assert (tt.s0, tt.tau_s, tt.tol_m, tt.timing_s) == (math.radians(1.0), 2.0, 0.2, 0.0)


def test_golden():
    (path,) = golden_files("tilt_trust")
    g = np.load(path)
    assert g["sigma"].shape[1] >= 100 and not np.isnan(g["gyro_rate"]).all()
    for q in range(len(g["h"])):
        tt = TiltTrust(sigma0_deg=float(g["sigma0_deg"][q]), tau_s=float(g["tau_s"][q]), tol_m=float(g["tol_m"][q]),
                       timing_s=float(g["timing_s"][q]))
        h = float(g["h"][q])
        assert tt.sigma == g["sigma_start"][q] and tt.max_range(h) == g["range_start"][q]
        for k in range(g["sigma"].shape[1]):
            args = [float(g[name][q, k]) for name in ("dt", "a", "v", "w")]
            rate = g["gyro_rate"][q, k]
            s = tt.update(*args) if np.isnan(rate) else tt.update(*args, gyro_rate=float(rate))
            got = (s, tt.sigma, tt.motion, tt.max_range(h))
            want = (g["sigma"][q, k], g["sigma"][q, k], g["motion"][q, k], g["max_range"][q, k])
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=0, err_msg=str((q, k)))
