"""The generic runtime: everything between a robot adapter (navette.interfaces) and the core's Navigator.

  loop.py          the perception loop, run folder (meta.json, steps.csv, commands.csv)
  controller.py    START / STOP state machine, watchdogs, command thread
  up.py            'up' at a frame's capture time from the IMU samples
  localization.py  the one source of the pose (odometry today)
  glance.py        head glance at the blind zone, for robots with a Head
  replay.py        a recorded run through the navigator again
  page.py          status page (START / STOP / status / preview); preview.py: its picture
  cli.py           `python -m navette.runtime --robot robots/<robot>.toml ...`

No robot SDK and no optional dependency is imported at module level (OpenCV: preview only; fastapi: page only).
"""
