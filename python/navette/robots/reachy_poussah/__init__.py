"""Adapter: Reachy Mini on the Poussah self-balancing base (config: robots/reachy_poussah.toml).

  robot.py         the live robot: camera + floor model, head IMU, head glance, antenna STOP, recordings
  poussah_base.py  the Poussah base behind the Base contract
  poussah_link.py  the base's wire protocol (Bluetooth LE / USB), clock sync, wheel odometry, link log
  imu.py           head IMU sources and FIFO sample timing; imu_fifo.py: the 400 Hz reader (separate process)
  recording.py     a recorded run as a Robot, for the replay
  app.py           the loop as a Reachy Mini app

Extras: `navette[reachy-poussah]` (OpenCV, onnxruntime, bleak, pyserial, fastapi, uvicorn); the Reachy Mini
SDK comes with the robot. Nothing here is imported until open / open_replay is called.
"""


def open(cfg, out_dir, dry_run=False, use_base=True, log=print, **kw):
    """The live robot (navette.interfaces.Robot). See robot.open_robot."""
    from navette.robots.reachy_poussah.robot import open_robot

    return open_robot(cfg, out_dir, dry_run=dry_run, use_base=use_base, log=log, **kw)


def open_replay(cfg, run_dir, **kw):
    """A recorded run as a Robot. See recording.open_replay_robot."""
    from navette.robots.reachy_poussah.recording import open_replay_robot

    return open_replay_robot(cfg, run_dir, **kw)
