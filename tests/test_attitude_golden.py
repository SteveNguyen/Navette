"""navette.UpFilter against the Python prototype's outputs on synthetic IMU streams
(tests/golden/attitude_streams.npz, see tests/golden/README.md): rest, slow tilt, nod, vibration, gyro bias,
rate gate, irregular timestamps, impacts. 'up' and the bias are compared after every sample."""

import numpy as np
import pytest

from conftest import golden_files
from navette import UpFilter

# The filter is a long recursion of sin / cos / sqrt: one ulp anywhere stays in the state. On the machine
# that wrote the goldens the match is exact (0.0) for update(); up_at() is within one ulp (the prototype
# normalises it with a BLAS dot product).
ATOL = 1e-12

PARAMS = ("tau_s", "bias_tau_s", "acc_gate", "g", "history_s", "still_rate", "still_acc", "still_s",
          "still_bias_tau_s", "rate_tau_s")

(PATH,) = golden_files("attitude")
G = np.load(PATH)
STREAMS = range(len(G["names"]))


def make(s):
    kw = {k: float(G[k][s]) for k in PARAMS}
    return UpFilter(**kw, rate_gate=None if np.isnan(G["rate_gate"][s]) else float(G["rate_gate"][s]),
                    bias0=G["bias0"][s] if G["has_bias0"][s] else None)


def test_golden_file():
    assert len(G["names"]) >= 8 and G["t"].shape[1] >= 1000
    assert (np.diff(G["t"], axis=1) <= 0).sum() > 50  # repeated / backward timestamps are in
    assert not np.isnan(G["rate_gate"]).all() and G["has_bias0"].any()


@pytest.mark.parametrize("s", STREAMS, ids=lambda s: str(G["names"][s]))
def test_stream(s):
    f = make(s)
    t, gyro, acc = G["t"][s], G["gyro"][s], G["acc"][s]
    every = int(G["query_every"])
    for k in range(len(t)):
        u = f.update(float(t[k]), gyro[k], acc[k])
        np.testing.assert_allclose(u, G["up"][s, k], rtol=0, atol=ATOL, err_msg=f"up, sample {k}")
        np.testing.assert_allclose(f.b, G["bias"][s, k], rtol=0, atol=ATOL, err_msg=f"bias, sample {k}")
        # states: exact
        assert (f.still_since is not None) == G["still"][s, k], k
        assert f.still_since == G["still_since"][s, k] or not G["still"][s, k], k
        assert f.t == t[k], k
        np.testing.assert_allclose(f.rate_lp, G["rate_lp"][s, k], rtol=0, atol=ATOL, err_msg=f"rate_lp, sample {k}")
        if (k + 1) % every == 0:
            q = (k + 1) // every - 1
            for tq, want in zip(G["query_t"][s, q], G["query_up"][s, q]):
                np.testing.assert_allclose(f.up_at(float(tq)), want, rtol=0, atol=ATOL, err_msg=f"up_at, sample {k}")
