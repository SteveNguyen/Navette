"""Extract a SLAM Sequence (navette.slam.data) from a recorded robot run, on a laptop, by replaying it through
the same perception as navette.runtime.replay (adapter's open_replay camera / IMU / base, LiveUp, camera_pose,
FreeSpace, OdometryLocalization) but without any navigator: what is kept per frame does not depend on a
navigator's decisions.

    uv run --project NAVETTE --extra reachy-poussah python -m navette.slam.extract
        --robot NAVETTE/robots/reachy_poussah.toml RUN_DIR OUT.npz          (one command line)

Frames kept: as the replay, a robot run's steps.csv decides (only the frames the robot stepped, plus its head
glance frames, flagged `glance`); without one, every frame with a pose and an 'up'. No ground truth.
Runs are private (CLAUDE.md): write OUT outside the repository.
"""
import argparse
import importlib
import os

import numpy as np

from navette.interfaces import Odometry
from navette.runtime.localization import OdometryLocalization
from navette.runtime.loop import camera_pose, default_up_filter, finite_odometry, free_space, usable_camera_pose
from navette.runtime.replay import robot_steps
from navette.runtime.up import LiveUp
from navette.slam.data import Keyframe, Sequence, scan_dict


def extract(robot, cfg, run_dir, make_up_filter=default_up_filter, log=print):
    """Sequence of a replayed run. robot: the adapter's open_replay(cfg, run_dir)."""
    camera, imu, base = robot.camera, robot.imu, robot.base
    follow = robot_steps(run_dir)
    fs = free_space(camera)
    up = LiveUp(make_up_filter, cfg.perception.up_g_s, cfg.perception.imu_max_age_s, g=getattr(imu, "g", None))
    loc = OdometryLocalization()
    frames, skipped = [], 0
    while True:
        try:
            frame = camera.read()
        except EOFError:
            break
        t, idx = frame.t, frame.index
        up.add(imu.read())  # every frame, stepped or not: the attitude filter needs all samples
        odom = finite_odometry(base.odometry()) if base is not None else Odometry(t, 0.0, 0.0, 0.0, 0.0, 0.0)
        loc.update(odom)
        if follow is not None and idx not in follow:
            continue
        up_c = up.up_at(t)
        if up_c is None or loc.pose() is None:  # the robot took no step either
            skipped += 1
            continue
        cam_h, cam_heading = camera_pose(frame.T_base_cam, up_c, cfg.camera.body_z)
        if not usable_camera_pose(cam_h, cam_heading):
            skipped += 1
            continue
        scan = fs.scan(camera.segment(frame), up_c, cam_h)
        glance = follow is not None and follow[idx][2] == "GLANCE"
        frames.append(Keyframe(t=t, odom=loc.pose(), scan=scan_dict(scan), cam_h=cam_h, cam_heading=cam_heading,
                               glance=glance))
    log(f"{os.path.basename(os.path.normpath(run_dir))}: {len(frames)} frames "
        f"({sum(f.glance for f in frames)} glance), {skipped} skipped (no 'up' / pose / camera pose), "
        f"pose {'odometry' if base is not None else 'fixed (no base in the recording)'}")
    meta = dict(source="navette.slam.extract", run=os.path.basename(os.path.normpath(run_dir)),
                robot_steps=follow is not None, base=base is not None, imu=getattr(imu, "name", ""))
    return Sequence.from_keyframes(frames, meta)


def main(argv=None):
    from navette import config

    ap = argparse.ArgumentParser(prog="python -m navette.slam.extract", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, help="robot config (TOML)")
    ap.add_argument("run", help="recorded run folder (with frames.csv)")
    ap.add_argument("out", help="output .npz")
    args = ap.parse_args(argv)
    cfg = config.load(args.robot)
    adapter = importlib.import_module(cfg.robot.adapter)
    robot = adapter.open_replay(cfg, args.run)
    try:
        seq = extract(robot, cfg, args.run)
    finally:
        robot.close()
    seq.save(args.out)
    d = np.diff(seq.odom[:, :2], axis=0)
    print(f"{args.out}: {len(seq)} frames, {seq.t[-1] - seq.t[0]:.0f} s, odometry path "
          f"{np.hypot(d[:, 0], d[:, 1]).sum():.1f} m")


if __name__ == "__main__":
    main()
