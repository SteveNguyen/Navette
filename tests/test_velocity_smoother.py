"""navette.ResumeGate: the Python prototype's test, and its outputs on random sequences
(tests/golden/velocity_smoother_gate.npz, see tests/golden/README.md)."""

import numpy as np

from conftest import golden_files
from navette import ResumeGate


def test_resume_gate_stops_at_once_and_resumes_after_n_clear_steps():
    g = ResumeGate(2)
    assert g(0.15, 0.15) == 0.15      # going: unchanged
    assert g(0.15, 0.0) == 0.0        # stop: at once
    assert g(0.15, 0.15) == 0.0       # 1st clear step: still stopped
    assert g(0.15, 0.15) == 0.15      # 2nd: resume
    assert g(0.15, 0.0) == 0.0
    assert g(0.15, 0.15) == 0.0
    assert g(0.15, 0.0) == 0.0        # noise in between restarts the count
    assert g(0.15, 0.1) == 0.0
    assert g(0.15, 0.1) == 0.1
    assert ResumeGate(1)(0.15, 0.15) == 0.15


def test_state_attributes():
    g = ResumeGate()
    assert (g.n, g.stopped, g.clear) == (2, False, 0)
    g(0.15, 0.0)
    g(0.15, 0.15)
    assert (g.stopped, g.clear) == (True, 1)
    g.n = 1  # writable: the next clear step resumes
    assert g(0.15, 0.15) == 0.15 and not g.stopped


def test_golden():
    (path,) = golden_files("velocity_smoother")
    g = np.load(path)
    assert g["v_out"].shape[1] >= 100
    for n, v_req, v_ok, v_out, stopped, clear in zip(g["n"], g["v_req"], g["v_ok"], g["v_out"], g["stopped"],
                                                    g["clear"]):
        gate = ResumeGate(int(n))
        for k in range(len(v_req)):
            what = (int(n), k)
            assert gate(float(v_req[k]), float(v_ok[k])) == v_out[k], what
            assert gate.stopped == stopped[k] and gate.clear == clear[k], what
