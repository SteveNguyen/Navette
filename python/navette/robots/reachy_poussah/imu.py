"""Head IMU of the Reachy Mini (BMI088) for the Imu contract: the 400 Hz FIFO reader (imu_fifo.py, a separate
process in the daemon's venv) read from its log as it grows, with the daemon's own 50 Hz IMU as the fallback.

The FIFO gives samples in batches, without times: FifoClock rebuilds them (causal, streaming) on the robot;
load_imu_fifo does the same over a whole log for the replay (batch fit).
"""
import collections
import json
import math
import os
import signal
import subprocess
import time

import numpy as np

DAEMON_PY = "/venvs/mini_daemon/bin/python"  # has smbus3 (the apps' venv does not)
SIDECAR_TIMEOUT_S = 2.0  # no new sidecar data for this long (or it exited): fall back to the daemon IMU
ACC_FULL_BYTES = 1000  # accelerometer FIFO (1024 B, 7 B per frame) this full at a read: it may have overflowed


def fifo_overruns(rec):
    """(gyro frames lost, accelerometer frames lost) before this imu_fifo.py read. Gyro: its overrun flag (gov).
    Accelerometer: no flag; its byte count `an` near the 1 KB capacity (stream mode drops the oldest frames), or
    the gyro overrun (the gyro FIFO, 100 frames = 0.25 s, fills before the 146-frame accelerometer one)."""
    g = bool(rec["gov"])
    return g, g or rec.get("an", 0) >= ACC_FULL_BYTES


# ------------------------------------------------------------------ streaming (robot)

class FifoClock:
    """Causal sample times of one BMI088 FIFO stream, from (read start time, frames returned) per read: the
    streaming version of fifo_sample_times (same model, fitted on the last `window` reads only).

    Sample k (counted from the first) is at a + k / f. f: least-squares slope of read time vs cumulative count
    (nominal f0 until min_reads reads). a: every sample a read returns was taken before that read started, so
    a <= t_read_i - (C_i - 1) / f for every read i; the tightest bound over the window is taken. The times of
    the new samples are therefore never after their read (causal). Kept strictly increasing: a later, better
    fit may place a new sample before one already returned; it is then put just after it (sub-ms)."""

    def __init__(self, f0=400.0, window=500, min_reads=20):
        self.f0, self.min_reads = f0, min_reads
        # the last `window` reads (t_read, cumulative count after the read) in ring buffers: the fit and the
        # bound below are order-free, and numpy arrays avoid rebuilding 500-element lists per read
        self.t_buf, self.c_buf = np.empty(window), np.empty(window)
        self.n, self.i = 0, 0
        self.count = 0
        self.t_last = -math.inf
        self.f = f0

    def reset(self):
        """Samples were lost (FIFO overrun): the count no longer gives the time. Restart the fit."""
        self.n, self.i = 0, 0
        self.count = 0

    def add(self, t_read, n):
        """n samples returned by the read that started at t_read -> their times (array of n)."""
        if n <= 0:
            return np.empty(0)
        self.count += n
        self.t_buf[self.i], self.c_buf[self.i] = t_read, self.count
        self.i = (self.i + 1) % len(self.t_buf)
        self.n = min(self.n + 1, len(self.t_buf))
        t, C = self.t_buf[:self.n], self.c_buf[:self.n]
        if self.n >= self.min_reads:
            # least-squares slope of t vs C, closed form (as np.polyfit(C, t, 1)[0], ~20x cheaper on the CM4:
            # two clocks per 20 ms record cost ~29 ms per camera frame with polyfit)
            dc = C - C.mean()
            self.f = float(np.dot(dc, dc) / np.dot(dc, t - t.mean()))
        a = np.min(t - (C - 1) / self.f)
        times = a + np.arange(self.count - n, self.count) / self.f
        times = np.minimum(times, t_read)  # causal by construction, up to float rounding (1e-15 s)
        times = np.maximum(times, self.t_last + 1e-6 * np.arange(1, n + 1))
        self.t_last = times[-1]
        return times


class JsonlTail:
    """New complete lines of a jsonl file another process is writing (polled, no thread)."""

    def __init__(self, path):
        self.path, self.f, self.buf = path, None, ""

    def poll(self):
        if self.f is None:
            if not os.path.exists(self.path):
                return []
            self.f = open(self.path)
        self.buf += self.f.read()
        *lines, self.buf = self.buf.split("\n")
        return [json.loads(ln) for ln in lines if ln.strip()]

    def close(self):
        if self.f is not None:
            self.f.close()


class FifoImu:
    """imu_fifo.py records -> (t, gyro rad/s, acc m/s^2) samples at the gyro's times, as load_imu_fifo
    (scales from the header line; acc linearly interpolated at the gyro times, held past its last sample:
    the accelerometer runs on its own clock and FIFO). R: optional 3x3 rotation applied to both vectors
    (IMU axes -> camera axes), a whole record at a time."""

    def __init__(self, R=None):
        self.head, self.R = None, R
        self.g_clock, self.a_clock = FifoClock(), FifoClock()
        self.acc_t = collections.deque(maxlen=64)  # recent accelerometer samples for the interpolation
        self.acc = collections.deque(maxlen=64)

    def add(self, rec):
        if rec.get("header"):
            self.head = rec
            return []
        if self.head is None:
            return []
        g = np.array(rec["g"], float).reshape(-1, 3)
        a = np.array(rec["a"], float).reshape(-1, 3)
        g_lost, a_lost = fifo_overruns(rec)  # frames lost before these: the counts no longer give the times
        if g_lost:
            self.g_clock.reset()
        if a_lost:
            self.a_clock.reset()
        t_g = self.g_clock.add(rec["t0"], len(g))
        t_a = self.a_clock.add(rec["t0"], len(a))
        self.acc_t.extend(t_a)
        self.acc.extend(a / self.head["acc_lsb_per_g"] * self.head["g"])
        if not self.acc:
            return []
        at, av = np.array(self.acc_t), np.array(self.acc)
        acc_g = np.stack([np.interp(t_g, at, av[:, j]) for j in range(3)], axis=1)
        gyro = np.radians(g / self.head["gyro_lsb_per_dps"])
        if self.R is not None:
            gyro, acc_g = gyro @ self.R.T, acc_g @ self.R.T
        return list(zip(t_g, gyro, acc_g))


class ImuSource:
    """The Imu contract. Samples from the imu_fifo.py sidecar (400 Hz), started here with the daemon venv's
    python and read from its jsonl as it grows; or, if it fails (exits, or no data for SIDECAR_TIMEOUT_S), the
    daemon's 50 Hz IMU that the state thread keeps (StateRecorder.imu_after). `name` says which one is in use.
    R: 3x3 rotation IMU axes -> camera axes (None: samples stay in IMU axes)."""

    def __init__(self, out_dir, recorder, log, R=None, python=None):
        self.recorder, self.log, self.R = recorder, log, R
        self.path = os.path.join(out_dir, "imu_fifo.jsonl")
        self.tail, self.fifo, self.proc = JsonlTail(self.path), FifoImu(R), None
        self.name, self.t_data, self.t_sdk = "fifo", time.monotonic(), -math.inf
        self.t_fallback = None  # host time the daemon IMU took over (meta.json: the replay switches there too)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imu_fifo.py")
        self.out_f = open(os.path.join(out_dir, "imu_fifo.out"), "w")
        try:
            # same process group: a Ctrl-C in the terminal reaches it too (SIGINT: it restores the registers);
            # PR_SET_PDEATHSIG: SIGINT as well if this process dies without stopping it (kill -9, crash).
            # --duration bounds an orphan anyway (4 h).
            self.proc = subprocess.Popen([python or DAEMON_PY, script, "--out", self.path, "--duration", "14400"],
                                         stdout=self.out_f, stderr=subprocess.STDOUT, preexec_fn=_sigint_on_parent_death)
        except OSError as e:
            self._fallback(f"cannot start: {e}")

    def _fallback(self, why):
        self.log(f"IMU: sidecar failed ({why}): using the daemon's 50 Hz IMU")
        self.name = "daemon"
        self.t_sdk = self.t_fallback = time.monotonic()  # samples from now on (older ones would step it back)
        self.stop()

    def read(self):
        """New samples [(t, gyro, acc)] since the last call."""
        if self.name == "fifo":
            out = [s for rec in self.tail.poll() for s in self.fifo.add(rec)]
            now = time.monotonic()
            if out:
                self.t_data = now
                return out
            if self.proc.poll() is not None:
                self._fallback(f"exited with {self.proc.returncode}, see imu_fifo.out")
            elif now - self.t_data > SIDECAR_TIMEOUT_S:
                self._fallback(f"no data for {SIDECAR_TIMEOUT_S} s")
            else:
                return []
        out = self.recorder.imu_after(self.t_sdk)
        if out:
            self.t_sdk = out[-1][0]
            if self.R is not None:
                out = [(t, self.R @ g, self.R @ a) for t, g, a in out]
        return out

    def stop(self):
        """SIGINT, then SIGTERM (both make imu_fifo.py restore the BMI088 registers in its finally), SIGKILL only
        as a last resort: it would leave the daemon's IMU at 400 Hz with the FIFOs on until the daemon restarts."""
        for sig, wait_s in ((signal.SIGINT, 3.0), (signal.SIGTERM, 3.0), (signal.SIGKILL, 3.0)):
            if self.proc is None or self.proc.poll() is not None:
                break
            self.proc.send_signal(sig)
            try:
                self.proc.wait(wait_s)
            except subprocess.TimeoutExpired:
                self.log(f"IMU sidecar still running after {signal.Signals(sig).name}")
        self.tail.close()
        self.out_f.close()


def _sigint_on_parent_death():
    """preexec_fn of the sidecar (runs in the child): prctl(PR_SET_PDEATHSIG, SIGINT). The kernel sends it when
    the thread that started the child exits, so ImuSource is created on the thread that runs the loop."""
    import ctypes

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGINT)  # 1 = PR_SET_PDEATHSIG


# ------------------------------------------------------------------ batch (replay)

def fifo_sample_times(t_read, counts):
    """Times of FIFO samples from (read start time, frames returned) per read.

    Samples are evenly spaced at an unknown rate f (sensor clock), sample k at a + k / f.
    f: least-squares slope of read time vs cumulative count. a: every sample a read returns
    was taken before that read started, so a <= t_read_i - (C_i - 1) / f for all i; the
    tightest bound is taken (reads land at random phases, so some land just after a sample).
    """
    t_read, counts = np.asarray(t_read, float), np.asarray(counts)
    C = np.cumsum(counts)  # samples available after read i
    m = counts > 0
    f = 1.0 / np.polyfit(C[m], t_read[m], 1)[0]
    a = np.min(t_read[m] - (C[m] - 1) / f)
    return a + np.arange(C[-1]) / f, f


def segmented_sample_times(t_read, counts, resets, f0):
    """fifo_sample_times per run of reads between losses (resets[i]: frames lost before read i), so a stall
    that overflowed the FIFO does not shift every later sample. A segment with fewer than 3 non-empty reads
    uses the nominal rate f0. Returns (times, rate of the longest segment)."""
    counts = np.asarray(counts)
    starts = [0] + [i for i in range(1, len(counts)) if resets[i]] + [len(counts)]
    times, rates = [], []
    for i0, i1 in zip(starts[:-1], starts[1:]):
        t, c = np.asarray(t_read[i0:i1], float), counts[i0:i1]
        if np.count_nonzero(c) >= 3:
            ts, f = fifo_sample_times(t, c)
        elif c.sum():
            C, m = np.cumsum(c), c > 0
            f = f0
            ts = np.min(t[m] - (C[m] - 1) / f) + np.arange(C[-1]) / f
        else:
            continue
        times.append(ts)
        rates.append((len(ts), f))
    return np.concatenate(times) if times else np.empty(0), max(rates)[1] if rates else f0


def load_imu_fifo(path):
    """imu_fifo.py log -> (t_gyro, gyro rad/s, t_acc, acc m/s^2, info). Times in time.monotonic s."""
    with open(path) as f:
        head = json.loads(f.readline())
        recs = [json.loads(line) for line in f]
    g = np.array([v for r in recs for v in r["g"]], float).reshape(-1, 3)
    a = np.array([v for r in recs for v in r["a"]], float).reshape(-1, 3)
    t0 = [r["t0"] for r in recs]
    lost = [fifo_overruns(r) for r in recs]
    t_g, f_g = segmented_sample_times(t0, [len(r["g"]) // 3 for r in recs], [x[0] for x in lost],
                                      head.get("gyro_odr_hz", 400.0))
    t_a, f_a = segmented_sample_times(t0, [len(r["a"]) // 3 for r in recs], [x[1] for x in lost],
                                      head.get("acc_odr_hz", 400.0))
    gyro = np.radians(g / head["gyro_lsb_per_dps"])
    acc = a / head["acc_lsb_per_g"] * head["g"]
    info = {"gyro_hz": f_g, "acc_hz": f_a, "gyro_overruns": sum(r["gov"] for r in recs), "reads": len(recs)}
    return t_g, gyro, t_a, acc, info
