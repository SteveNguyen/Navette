"""Reachy Mini side of the adapter without hardware: BMI088 FIFO parsing (imu_fifo.py), sample time
reconstruction (streaming and batch), IMU source fallback, state recorder, antenna push."""
import json
import os
import signal
import sys
import time

import numpy as np
import pytest

from navette.robots.reachy_poussah import imu as lr
from navette.robots.reachy_poussah import robot as rb
from navette.robots.reachy_poussah.imu_fifo import parse_acc_fifo, parse_gyro_fifo
from navette.runtime import cli

# ------------------------------------------------------------------ FIFO bytes


def le16(v):
    return list(int(v).to_bytes(2, "little", signed=True))


def test_parse_acc_fifo_frames_sensortime_and_overread():
    buf = [0x84] + le16(100) + le16(-200) + le16(10920)
    buf += [0x40, 3]  # skip frame: 3 frames dropped
    buf += [0x84] + le16(-1) + le16(0) + le16(32767)
    buf += [0x44, 0x01, 0x02, 0x03]  # sensortime 0x030201
    buf += [0x80, 0x00, 0x00]  # over-read
    frames, times, skipped = parse_acc_fifo(buf)
    assert frames == [(100, -200, 10920), (-1, 0, 32767)]
    assert times == [0x030201]
    assert skipped == 3


def test_parse_acc_fifo_truncated_frame_stops():
    buf = [0x84] + le16(1) + le16(2) + le16(3) + [0x84, 0x00, 0x01]
    frames, _, _ = parse_acc_fifo(buf)
    assert frames == [(1, 2, 3)]


def test_parse_gyro_fifo():
    buf = le16(1) + le16(-2) + le16(3) + le16(-32768) + le16(0) + le16(32767)
    assert parse_gyro_fifo(buf) == [(1, -2, 3), (-32768, 0, 32767)]


def test_sample_times_recovered_from_jittery_reads():
    rng = np.random.default_rng(1)
    f_true, a_true = 401.3, 10.0012  # sensor clock slightly off nominal, unknown phase
    t_read = 10.02 + np.cumsum(rng.uniform(0.015, 0.025, 6000))  # 50 Hz-ish reads with jitter
    n_avail = np.floor((t_read - a_true) * f_true).astype(int) + 1  # samples taken before each read
    counts = np.diff(np.r_[0, n_avail])
    t, f = lr.fifo_sample_times(t_read, counts)
    true = a_true + np.arange(n_avail[-1]) / f_true
    assert abs(f - f_true) < 0.05
    assert np.max(np.abs(t - true)) < 0.5e-3  # better than 0.5 ms


# ------------------------------------------------------------------ sample times, streaming

ACC_HZ = 2.0  # varying test signal: acc z = G0 + sin(2 pi ACC_HZ t), gyro z = 10 deg/s * cos(...)
G0 = 9.80665


def acc_true(t):
    return np.stack([np.zeros_like(t), np.zeros_like(t), G0 + np.sin(2 * np.pi * ACC_HZ * t)], -1)


def synthetic_reads(f_true=401.3, a_true=10.0012, n=1500, seed=1, stall_at=None, stall_s=0.6):
    """Reads at ~50 Hz with jitter; optional stall (the reader descheduled). Returns (t_read, counts, true
    sample times of the frames each read returns)."""
    rng = np.random.default_rng(seed)
    dts = rng.uniform(0.015, 0.025, n)
    if stall_at is not None:
        dts[stall_at] = stall_s
    t_read = 10.02 + np.cumsum(dts)
    n_avail = np.floor((t_read - a_true) * f_true).astype(int) + 1  # samples taken before each read
    return t_read, np.diff(np.r_[0, n_avail]), a_true + np.arange(n_avail[-1]) / f_true


def capped(counts, cap):
    """FIFO of `cap` frames in stream mode: the oldest are dropped. (got, lost) per read."""
    lost = np.maximum(counts - cap, 0)
    return counts - lost, lost


def test_fifo_clock_streaming_matches_truth_and_is_causal():
    t_read, counts, true = synthetic_reads()
    clk = lr.FifoClock()
    times = []
    for t, n in zip(t_read, counts):
        ts = clk.add(t, n)
        assert len(ts) == n and (n == 0 or ts[-1] <= t)  # never after the read that returned them
        times.extend(ts)
    times = np.array(times)
    assert np.all(np.diff(times) > 0)
    late = true > true[0] + 2.0  # after 2 s of reads (the fit window filling)
    assert np.max(np.abs(times[late] - true[late])) < 1e-3
    assert abs(clk.f - 401.3) < 0.5


def test_fifo_clock_reset_after_overrun():
    t_read, counts, _ = synthetic_reads(n=300)
    clk = lr.FifoClock()
    for t, n in zip(t_read[:200], counts[:200]):
        last = clk.add(t, n)[-1] if n else None
    clk.reset()
    ts = clk.add(t_read[220], 100)  # a full FIFO after lost samples
    assert ts[-1] <= t_read[220] and ts[0] > last and len(ts) == 100


def fifo_records(t_read, g_counts, a_counts, g_true, a_true, gov=None, an=None):
    """imu_fifo.py records: gyro z 10 deg/s, acc from acc_true at each frame's true time (raw int16)."""
    head = {"header": True, "gyro_odr_hz": 400, "acc_odr_hz": 400, "gyro_lsb_per_dps": 65.536,
            "acc_lsb_per_g": 10920, "g": G0}
    recs, ig, ia = [head], 0, 0
    for k, (t, ng, na) in enumerate(zip(t_read, g_counts, a_counts)):
        g = [0, 0, int(round(10 * 65.536))] * int(ng)
        a = np.round(acc_true(a_true[ia:ia + na]) / G0 * 10920).astype(int).ravel().tolist()
        recs.append({"t0": float(t), "t": float(t) + 0.002, "g": g, "gov": bool(gov[k]) if gov is not None else False,
                     "a": a, "st": [], "an": int(an[k]) if an is not None else 7 * int(na)})
        ig, ia = ig + ng, ia + na
    return recs


def test_fifo_imu_pairs_a_varying_accelerometer_at_the_gyro_times():
    """Accelerometer on its own clock (398 Hz vs the gyro's 401.3), interpolated at the gyro sample times."""
    t_read, g_counts, g_true = synthetic_reads(n=300)
    _, a_counts, a_true = synthetic_reads(f_true=398.0, a_true=10.0007, n=300)
    imu = lr.FifoImu()
    s = [x for rec in fifo_records(t_read, g_counts, a_counts, g_true, a_true) for x in imu.add(rec)]
    assert len(s) == g_counts.sum()
    t = np.array([x[0] for x in s])
    acc = np.array([x[2] for x in s])
    late = t > t[0] + 2.0
    assert np.max(np.abs(t[late] - g_true[late])) < 1e-3
    # within ~1 ms of timing and the last acc sample held for at most one read
    assert np.median(np.abs(acc[late, 2] - acc_true(g_true[late])[:, 2])) < 0.02
    assert s[-1][1] == pytest.approx(np.radians([0, 0, 10.0]), abs=1e-3)


def test_fifo_imu_rotates_into_the_camera_frame():
    """The Imu contract wants camera axes: R (IMU axes -> camera axes) is applied to both vectors."""
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # IMU z (up) -> camera -y (up)
    t_read, counts, true = synthetic_reads(n=60)
    plain, rotated = lr.FifoImu(), lr.FifoImu(R)
    for rec in fifo_records(t_read, counts, counts, true, true):
        a, b = plain.add(rec), rotated.add(rec)
        assert len(a) == len(b)
        for (ta, ga, aa), (tb, gb, ab) in zip(a, b):
            assert ta == tb and gb == pytest.approx(R @ ga) and ab == pytest.approx(R @ aa)
    assert b[-1][2][1] < -8.0 and b[-1][1][1] == pytest.approx(-np.radians(10.0), abs=1e-3)


def stall_records():
    """A 0.6 s stall overflows both FIFOs (gyro 100 frames, accelerometer 146 frames)."""
    t_read, counts, true = synthetic_reads(f_true=400.0, a_true=10.0, n=600, stall_at=300)
    g_got, g_lost = capped(counts, 100)
    a_got, a_lost = capped(counts, 146)
    keep_g = np.repeat(np.arange(len(counts)), counts)  # read index of each true sample

    def returned(got):  # true times of the frames each read actually returns (the newest `got`)
        out = []
        for k in range(len(counts)):
            idx = np.flatnonzero(keep_g == k)
            out.extend(true[idx[len(idx) - got[k]:]])
        return np.array(out)
    g_true, a_true = returned(g_got), returned(a_got)
    recs = fifo_records(t_read, g_got, a_got, g_true, a_true, gov=g_lost > 0, an=7 * a_got)
    return recs, g_true, a_true, int(g_got[:301].sum())


def test_fifo_imu_resets_both_clocks_after_a_stall():
    recs, g_true, a_true, i_stall = stall_records()
    imu = lr.FifoImu()
    s = [x for rec in recs for x in imu.add(rec)]
    t = np.array([x[0] for x in s])
    acc = np.array([x[2] for x in s])
    after = slice(i_stall + 400, None)  # 1 s after the stall
    assert np.median(np.abs(t[after] - g_true[after])) < 2e-3
    # paired accelerometer right: a 138 ms accelerometer clock error gives errors up to ~1.5 m/s^2 here
    assert np.median(np.abs(acc[after, 2] - acc_true(g_true[after])[:, 2])) < 0.05


def test_fifo_overruns():
    assert lr.fifo_overruns({"gov": True, "an": 56}) == (True, True)
    assert lr.fifo_overruns({"gov": False, "an": 1020}) == (False, True)  # accelerometer FIFO near full
    assert lr.fifo_overruns({"gov": False, "an": 56}) == (False, False)


def test_load_imu_fifo_segments_at_overruns(tmp_path):
    """The batch loader (replay) restarts its fit after a FIFO overflow too."""
    recs, g_true, a_true, i_stall = stall_records()
    p = tmp_path / "imu_fifo.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    t_g, gyro, t_a, acc, info = lr.load_imu_fifo(str(p))
    assert len(t_g) == len(g_true) and len(t_a) == len(a_true)
    assert np.median(np.abs(t_g[i_stall:] - g_true[i_stall:])) < 1e-3
    assert np.median(np.abs(t_a[i_stall:] - a_true[i_stall:])) < 1e-3
    assert np.max(np.abs(t_a[:i_stall - 200] - a_true[:i_stall - 200])) < 1e-3
    assert abs(info["gyro_hz"] - 400.0) < 0.5


def test_jsonl_tail_returns_complete_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    tail = lr.JsonlTail(str(p))
    assert tail.poll() == []  # not there yet
    with open(p, "w") as f:
        f.write('{"a": 1}\n{"a": ')
        f.flush()
        assert tail.poll() == [{"a": 1}]
        f.write('2}\n')
        f.flush()
        assert tail.poll() == [{"a": 2}]
    tail.close()


# ------------------------------------------------------------------ IMU source

class FakeRecorder:
    def __init__(self):
        self.imu = [(1.0 + k * 0.02, np.zeros(3), np.array([0, 0, 9.8])) for k in range(5)]

    def imu_after(self, t):
        return [s for s in self.imu if s[0] > t]


def fake_python(tmp_path, body, name="fake_python"):
    """A stand-in for the daemon's python: a shell script that ignores its arguments."""
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(0o755)
    return str(p)


def test_imu_source_falls_back_when_the_sidecar_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "DAEMON_PY", str(tmp_path / "no_such_python"))
    rec, logs = FakeRecorder(), []
    src = lr.ImuSource(str(tmp_path), rec, logs.append)
    assert src.name == "daemon" and "cannot start" in logs[0] and src.t_fallback is not None
    rec.imu.append((time.monotonic() + 1, np.zeros(3), np.array([0, 0, 9.8])))  # newer than the fallback
    assert len(src.read()) == 1 and src.read() == []
    src.stop()


def test_imu_source_fallback_samples_are_rotated_too(tmp_path):
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    rec = FakeRecorder()
    src = lr.ImuSource(str(tmp_path), rec, lambda m: None, R=R, python=str(tmp_path / "no_such_python"))
    rec.imu.append((time.monotonic() + 1, np.array([0, 0, 0.1]), np.array([0, 0, 9.8])))
    (t, gyro, acc), = src.read()
    assert acc == pytest.approx([0, -9.8, 0]) and gyro == pytest.approx([0, -0.1, 0])
    src.stop()


def test_imu_source_falls_back_when_the_sidecar_exits(tmp_path, monkeypatch):
    """This interpreter has no smbus3: the real imu_fifo.py exits at once with an error."""
    monkeypatch.setattr(lr, "DAEMON_PY", sys.executable)
    logs = []
    src = lr.ImuSource(str(tmp_path), FakeRecorder(), logs.append)
    assert src.name == "fifo"
    src.proc.wait(10)
    src.read()
    assert src.name == "daemon" and "exited" in logs[0]
    assert "smbus3" in open(tmp_path / "imu_fifo.out").read()
    src.stop()


def test_imu_source_reads_the_sidecar_log_as_it_grows(tmp_path, monkeypatch):
    """The test writes the stand-in sidecar's log in pieces (lines cut anywhere): every sample comes out once,
    in time order, as the lines complete."""
    monkeypatch.setattr(lr, "DAEMON_PY", fake_python(tmp_path, "exec sleep 5"))
    src = lr.ImuSource(str(tmp_path), FakeRecorder(), lambda m: None)
    t_read, counts, g_true = synthetic_reads(n=60)
    text = "\n".join(json.dumps(r) for r in fifo_records(t_read, counts, counts, g_true, g_true)) + "\n"
    cuts = [0, 1000, 1001, 5000, 9000, 9001, len(text) // 2, len(text) - 3, len(text)]
    got, sizes = [], []
    with open(src.path, "w") as f:
        for a, b in zip(cuts[:-1], cuts[1:]):
            f.write(text[a:b])
            f.flush()
            new = src.read()
            sizes.append(len(new))
            got += new
    assert src.name == "fifo" and len(got) == counts.sum()
    assert sum(1 for n in sizes if n > 0) >= 4  # delivered progressively, not at the end
    t = np.array([x[0] for x in got])
    assert np.all(np.diff(t) > 0)
    src.stop()
    assert src.proc.poll() is not None


def test_imu_source_stop_escalates_sigint_then_sigterm(tmp_path, monkeypatch):
    """A sidecar that does not exit on SIGINT gets SIGTERM (still restores), SIGKILL only last."""
    monkeypatch.setattr(lr, "DAEMON_PY", fake_python(tmp_path, "trap '' INT\nexec sleep 30"))
    src = lr.ImuSource(str(tmp_path), FakeRecorder(), lambda m: None)
    time.sleep(0.2)  # let the shell set its trap
    src.stop()
    assert src.proc.returncode == -signal.SIGTERM


HELPER = """
import os, sys, time
from navette.robots.reachy_poussah import imu as lr
lr.DAEMON_PY = {fake!r}
class R:
    def imu_after(self, t):
        return []
src = lr.ImuSource({out!r}, R(), print)
time.sleep(0.5)
os._exit(0)  # dies without ImuSource.stop()
"""


def test_sidecar_gets_sigint_when_the_loop_process_dies(tmp_path):
    """PR_SET_PDEATHSIG: the sidecar does not outlive a crashed / killed loop process."""
    import subprocess

    pid_file = tmp_path / "pid"
    fake = fake_python(tmp_path, f"echo $$ > {pid_file}\nexec sleep 30")
    script = tmp_path / "helper.py"
    script.write_text(HELPER.format(fake=fake, out=str(tmp_path)))
    subprocess.run([sys.executable, str(script)], timeout=20, check=True)
    pid = int(pid_file.read_text())
    gone = False
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            gone = True
            break
        time.sleep(0.05)
    if not gone:
        os.kill(pid, signal.SIGKILL)
    assert gone


@pytest.mark.parametrize("stop_on_signals", [cli.stop_on_signals,
                                             __import__("navette.robots.reachy_poussah.imu_fifo", fromlist=["x"]).stop_on_signals])
def test_stop_on_signals_turns_sigterm_and_sighup_into_keyboardinterrupt(stop_on_signals):
    """Both the runtime's CLI and the IMU reader (its own copy: it runs without navette)."""
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        stop_on_signals()
        for s in old:
            with pytest.raises(KeyboardInterrupt):
                os.kill(os.getpid(), s)
                time.sleep(1)  # the handler runs at the next bytecode
    finally:
        for s, h in old.items():
            signal.signal(s, h)


# ------------------------------------------------------------------ SDK side

class FlakyMini:
    """SDK whose IMU read fails the first 3 times."""

    def __init__(self):
        self.n = 0

    @property
    def imu(self):
        self.n += 1
        if self.n <= 3:
            raise RuntimeError("sdk hiccup")
        return {"accelerometer": [0, 0, 9.8], "gyroscope": [0, 0, 0.001 * self.n], "quaternion": [1, 0, 0, 0]}

    def get_current_head_pose(self):
        return np.eye(4)

    def get_current_joint_positions(self):
        return [0.0] * 7, [0.0, 0.0]


def test_state_recorder_survives_sdk_errors(tmp_path):
    """One SDK error does not end the state log / the IMU fallback."""
    logs = []
    rec = rb.StateRecorder(FlakyMini(), str(tmp_path), logs.append)
    rec.start()
    time.sleep(0.2)
    rec.stop_event.set()
    rec.join(1)
    assert rec.errors == 3 and "sdk hiccup" in logs[0]
    assert len(rec.imu) > 10
    kinds = [json.loads(ln)["kind"] for ln in open(tmp_path / "state.jsonl")]
    assert kinds.count("imu") > 10 and "head_pose" in kinds


def test_antenna_push():
    rest = rb.ANTENNA_REST
    assert not rb.antenna_pushed(rest)
    assert not rb.antenna_pushed([rest[0] + 0.3, rest[1] - 0.3])
    assert rb.antenna_pushed([rest[0] - 0.5, rest[1]])
    assert rb.antenna_pushed([rest[0], rest[1] + 0.45])
    p = rb.AntennaPush()
    seq = [rest, [rest[0] + 0.6, rest[1]], [rest[0] + 0.7, rest[1]], rest, [rest[0], rest[1] - 0.6]]
    assert [p.update(x) for x in seq] == [False, True, False, False, True]  # one STOP per push


def test_camera_reads_stamps_and_saves_frames(tmp_path):
    """ReachyCamera on a stand-in SDK: capture time = arrival minus the camera's lag, the head pose times the
    hand-eye matrix, every frame saved with its arrival time."""
    cv2 = pytest.importorskip("cv2")

    class Media:
        frames = [np.full((720, 1280, 3), 90, np.uint8), None, np.full((720, 1280, 3), 200, np.uint8)]

        def get_frame(self):
            return self.frames.pop(0)

    class Mini:
        media = Media()

        def get_current_head_pose(self):
            T = np.eye(4)
            T[2, 3] = 0.1
            return T

    X = np.eye(4)
    X[0, 3] = 0.04
    seen = []

    def model(bgr):
        seen.append(bgr.shape)
        return np.ones((rb.MASK_H, rb.MASK_W), bool), np.zeros((rb.MASK_H, rb.MASK_W), bool)

    cam = rb.ReachyCamera(Mini(), model, X, None, None, str(tmp_path), latency_s=0.038)
    t0 = time.monotonic()
    a, none, b = cam.read(), cam.read(), cam.read()
    assert none is None and (a.index, b.index) == (0, 1) and t0 - 0.038 <= a.t < b.t <= time.monotonic() - 0.038
    assert a.T_base_cam[:3, 3] == pytest.approx([0.04, 0, 0.1]) and a.image.shape == (360, 640, 3)
    assert set(a.ms) == {"head", "save"}
    floor = cam.segment(a)
    assert floor.all() and a.under is not None and seen == [(360, 640, 3)]
    cam.close()
    rows = open(tmp_path / "frames.csv").read().split()
    assert rows[0] == "idx,t" and len(rows) == 3 and float(rows[1].split(",")[1]) == pytest.approx(a.t + 0.038, abs=1e-6)
    assert cv2.imread(str(tmp_path / "frames" / "000001.jpg")).mean() == pytest.approx(200, abs=2)
