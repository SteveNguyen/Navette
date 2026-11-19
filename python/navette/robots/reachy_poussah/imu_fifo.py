"""Head IMU (BMI088) at 400 Hz through its FIFOs, next to the running daemon. Runs on the robot.

Stand-alone script (standard library + smbus3), run by imu.ImuSource with the daemon venv's python: it does
not import navette. It does not touch the daemon: the daemon only writes the
BMI088 configuration at start and then reads its data registers once per 50 Hz loop; it keeps
working while this script raises the output rate and drains the FIFOs. Each i2c transfer is
atomic in the kernel, so the two processes do not corrupt each other's reads.

    /venvs/mini_daemon/bin/python imu_fifo.py --out /tmp/imu_fifo.jsonl --duration 120
    (the daemon venv has smbus3; apps_venv does not)

Registers from Bosch's BMI08x Sensor API (bmi08_defs.h v1.9.0). Only the documented bits are
changed (read-modify-write); everything is restored on exit (a daemon restart also restores it).

Log (jsonl): a header line with the scales, then one line per read:
  {"t0": before, "t": after (time.monotonic, s), "g": [x, y, z, ...] raw int16 gyro frames,
   "gov": gyro FIFO overrun, "a": [x, y, z, ...] raw int16 accel frames, "st": [sensortime ...],
   "an": accel FIFO byte count}
"""

import argparse
import json
import signal
import time

ACC_ADDR, GYRO_ADDR = 0x18, 0x69  # as in the daemon's bmi088 driver (params.py)
# accelerometer
ACC_CONF = 0x40  # bits 7:4 bandwidth (0xA normal), 3:0 ODR (0x0A = 400 Hz)
ACC_FIFO_LEN0 = 0x24  # 14-bit byte count in LEN0 | (LEN1 & 0x3F) << 8
ACC_FIFO_DATA = 0x26
ACC_FIFO_CONFIG_0 = 0x48  # bit 0: 0 stream, 1 fifo
ACC_FIFO_CONFIG_1 = 0x49  # bit 6: accel data in FIFO
ACC_HDR_DATA, ACC_HDR_TIME, ACC_HDR_SKIP, ACC_HDR_CFG, ACC_HDR_DROP = 0x84, 0x44, 0x40, 0x48, 0x50
SENSORTIME_S = 39.0625e-6
# gyroscope
GYRO_BW = 0x10  # 0x03: ODR 400 Hz, 47 Hz filter; 0x05 (daemon default): ODR 100 Hz, 12 Hz
GYRO_FIFO_STATUS = 0x0E  # bits 6:0 frame count, bit 7 overrun
GYRO_FIFO_CONFIG_1 = 0x3E  # bits 7:6 mode (0b10 stream), bits 1:0 data select (0 = xyz)
GYRO_FIFO_DATA = 0x3F

PERIOD_S = 0.02
SETTLE_S = 0.3
OVERREAD = 4  # extra accel bytes: the chip appends a sensortime frame when the FIFO runs empty


def parse_acc_fifo(buf):
    """Headered accelerometer FIFO bytes -> (list of (x, y, z) int16, list of sensortimes, n skipped).

    Stops at the first over-read byte (0x80) or an unknown header."""
    frames, times, skipped = [], [], 0
    i = 0
    while i < len(buf):
        h = buf[i] & 0xFC  # bits 1:0 are interrupt tags, not used here
        if h == ACC_HDR_DATA and i + 7 <= len(buf):
            v = [int.from_bytes(bytes(buf[i + 1 + 2 * k:i + 3 + 2 * k]), "little", signed=True) for k in range(3)]
            frames.append(tuple(v))
            i += 7
        elif h == ACC_HDR_TIME and i + 4 <= len(buf):
            times.append(buf[i + 1] | buf[i + 2] << 8 | buf[i + 3] << 16)
            i += 4
        elif h in (ACC_HDR_SKIP, ACC_HDR_CFG) and i + 2 <= len(buf):
            skipped += buf[i + 1] if h == ACC_HDR_SKIP else 0
            i += 2
        elif h == ACC_HDR_DROP:
            i += 1
        else:  # 0x80 over-read, or a truncated frame
            break
    return frames, times, skipped


def parse_gyro_fifo(buf):
    """Gyro FIFO bytes (xyz frames of 6 bytes, no header) -> list of (x, y, z) int16."""
    n = len(buf) // 6
    return [tuple(int.from_bytes(bytes(buf[6 * j + 2 * k:6 * j + 2 * k + 2]), "little", signed=True)
                  for k in range(3)) for j in range(n)]


def _interrupt(signum, frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


def stop_on_signals():
    """SIGTERM and SIGHUP (kill, ssh session lost) raise KeyboardInterrupt like Ctrl-C, so the finally blocks
    run (here: restore the registers). Main thread only."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)


def main():
    from smbus3 import SMBus, i2c_msg  # robot only (daemon venv)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, required=True, help="seconds")
    args = ap.parse_args()
    stop_on_signals()

    bus = SMBus(4)

    def rd(addr, reg, n):
        w, r = i2c_msg.write(addr, [reg]), i2c_msg.read(addr, n)
        bus.i2c_rdwr(w, r)
        return list(r)

    def rmw(addr, reg, mask, value):
        old = bus.read_byte_data(addr, reg)
        bus.write_byte_data(addr, reg, (old & ~mask & 0xFF) | (value & mask))
        return old

    saved = {}
    try:
        saved[(GYRO_ADDR, GYRO_BW)] = rmw(GYRO_ADDR, GYRO_BW, 0x0F, 0x03)
        saved[(GYRO_ADDR, GYRO_FIFO_CONFIG_1)] = rmw(GYRO_ADDR, GYRO_FIFO_CONFIG_1, 0xC3, 0x80)
        saved[(ACC_ADDR, ACC_CONF)] = rmw(ACC_ADDR, ACC_CONF, 0x0F, 0x0A)
        saved[(ACC_ADDR, ACC_FIFO_CONFIG_0)] = rmw(ACC_ADDR, ACC_FIFO_CONFIG_0, 0x01, 0x00)
        saved[(ACC_ADDR, ACC_FIFO_CONFIG_1)] = rmw(ACC_ADDR, ACC_FIFO_CONFIG_1, 0x40, 0x40)
        print("registers before:", {f"{a:#x}/{r:#x}": f"{v:#04x}" for (a, r), v in saved.items()}, flush=True)
        time.sleep(SETTLE_S)  # the gyro filter rings for tens of ms after the rate change (up to 160 deg/s)
        # The gyro FIFO (100 frames) overflowed while settling and its overrun flag latches (R7: set on
        # every read); writing FIFO_CONFIG_1 again clears the FIFO and the flag.
        bus.write_byte_data(GYRO_ADDR, GYRO_FIFO_CONFIG_1, bus.read_byte_data(GYRO_ADDR, GYRO_FIFO_CONFIG_1))
        # line-buffered: imu.ImuSource reads this file as it grows (a default 8 KB buffer = ~0.4 s late)
        with open(args.out, "w", buffering=1) as f:
            f.write(json.dumps({"header": True, "gyro_odr_hz": 400, "acc_odr_hz": 400,
                                "gyro_lsb_per_dps": 65.536, "acc_lsb_per_g": 10920, "g": 9.80665,
                                "sensortime_s": SENSORTIME_S, "saved": {f"{a}/{r}": v for (a, r), v in saved.items()}})
                    + "\n")
            t_end = time.monotonic() + args.duration
            next_t = time.monotonic()
            n_g = n_a = n_ov = 0
            first = True
            while time.monotonic() < t_end:
                next_t += PERIOD_S
                t0 = time.monotonic()
                st = rd(GYRO_ADDR, GYRO_FIFO_STATUS, 1)[0]
                cnt, gov = st & 0x7F, bool(st & 0x80)
                gb = rd(GYRO_ADDR, GYRO_FIFO_DATA, 6 * cnt) if cnt else []
                l0, l1 = rd(ACC_ADDR, ACC_FIFO_LEN0, 2)
                alen = l0 | (l1 & 0x3F) << 8
                ab = rd(ACC_ADDR, ACC_FIFO_DATA, alen + OVERREAD) if alen else []
                t1 = time.monotonic()
                if not first:  # the first read drains what accumulated before (unknown start time)
                    g = parse_gyro_fifo(gb)
                    a, times, _ = parse_acc_fifo(ab)
                    n_g, n_a, n_ov = n_g + len(g), n_a + len(a), n_ov + gov
                    f.write(json.dumps({"t0": t0, "t": t1, "g": [v for fr in g for v in fr], "gov": gov,
                                        "a": [v for fr in a for v in fr], "st": times, "an": alen}) + "\n")
                first = False
                time.sleep(max(0.0, next_t - time.monotonic()))
        print(f"done: {n_g} gyro, {n_a} accel frames, {n_ov} gyro overruns", flush=True)
    finally:
        # a second signal (terminal Ctrl-C, then ImuSource's own SIGINT) must not cut the restore short
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        for (addr, reg), v in saved.items():
            bus.write_byte_data(addr, reg, v)
        bus.close()
        print("registers restored", flush=True)


if __name__ == "__main__":
    main()
