"""The navigation loop on a robot: camera -> floor mask -> scan -> navigator -> controller -> base.

One process owns everything. Threads:
  perception (caller's thread)  frame -> odometry -> IMU 'up' -> floor mask -> scan -> Navigator.step ->
                                Controller.set_nav
  command    (controller.cmd_hz) Controller.tick: watchdogs, START / STOP sequencing, Base.command with the
                                latest navigator output
  page                          the status page (navette.runtime.page), served by the caller
  + the adapter's own threads (base link, IMU reader, stop button...)

Run folder (written here; the adapter adds its own recordings next to them, see its module):
  meta.json      clock, config, "args" (behaviour, forward_v, dry_run, base: whether there is one), "robot"
                 (the adapter's data, refreshed at the end), frames (count)
  steps.csv      one row per navigation step, STEP_COLS + LIVE_COLS (a replay follows frame, run, v_prev, state)
  commands.csv   every call the controller made to the base (or would have: no base / dry run)
"""
import csv
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from navette import interfaces as itf
from navette.runtime import controller as ctl
from navette.runtime.glance import Glance
from navette.runtime.localization import OdometryLocalization
from navette.runtime.up import LiveUp

# steps.csv columns shared by the robot run and the replay (step_row)
STEP_COLS = ["t", "frame", "x", "y", "heading_deg", "cam_h", "pitch_down_deg", "v", "w", "v_req", "w_req", "state",
             "seg_ms", "scan_ms", "nav_ms"]
# robot-only columns: controller state, navigator run id (+1 per START) and the v_prev passed (the replay
# follows both), IMU source, latency from the frame's capture to set_nav (ms; for choosing perception.stale_s),
# and where the loop's time goes outside the mask / scan / navigator: waiting for the frame, the adapter's
# own stages inside Camera.read (Frame.ms: head pose read, frame save), the IMU read + filter, and the
# previous iteration's preview render
LIVE_COLS = ["ctrl", "run", "v_prev", "imu", "lat_ms", "wait_ms", "head_ms", "imu_ms", "save_ms", "preview_ms"]
STOP_WAIT_S = 4.0  # at exit: time given to the command thread to finish the stop (park)


@dataclass
class NavOut:
    """What the runtime reads of a navigator output (navette.Navigator.step returns an object with these
    fields); also built here for the zero-velocity hold of a head glance."""
    v: float          # m/s, what to send
    w: float          # rad/s, > 0 counter-clockwise
    v_req: float      # what the behaviour asked for, before the collision monitor
    w_req: float
    state: str        # behaviour state (GLANCE: set by the loop)
    finished: bool    # exploration done
    blind_stop: bool = False  # forward motion stopped only by the never-seen blind zone (a head glance helps)


def camera_pose(T_base_cam, up_c, body_z):
    """(camera height m, camera heading rad in the base frame) from the camera pose in the base frame.
    A body can rest tilted (the reference robot: ~7 deg), so the height is taken along the measured 'up', not
    the base z axis. Heading: the optical axis (camera z) projected on the base's horizontal plane (x forward,
    y left)."""
    up_body = T_base_cam[:3, :3] @ up_c
    fwd = T_base_cam[:3, 2]
    return body_z + float(up_body @ T_base_cam[:3, 3]), math.atan2(fwd[1], fwd[0])


def finite_odometry(odom):
    """odom, or None if one of its values is not finite: that is no odometry (no pose, so no step: a NaN pose
    or velocity would go through the map and the tilt trust)."""
    return odom if odom is not None and all(math.isfinite(x) for x in odom) else None


def usable_camera_pose(cam_h, cam_heading):
    """The camera is above the floor at a finite height and heading. Otherwise the scan means nothing (it
    sees nothing at all with a NaN height), and the loop takes no step."""
    return math.isfinite(cam_h) and cam_h > 0 and math.isfinite(cam_heading)


def camera_mount_warning(T_base_cam, radius):
    """A warning if the camera's mounting is outside what the navigator supports (the Camera contract in
    navette.interfaces), else None. Limits: MAX_CAMERA_YAW_DEG of yaw, half the footprint radius of
    horizontal offset."""
    yaw, offset = itf.camera_mount(T_base_cam)
    if abs(yaw) > itf.MAX_CAMERA_YAW_DEG:
        return (f"WARNING: the camera looks {yaw:+.0f} deg off the base's forward axis (limit "
                f"{itf.MAX_CAMERA_YAW_DEG:.0f}): the navigator takes straight ahead in the image for the "
                f"driving direction, the robot may drive toward what it does not look at")
    if offset > 0.5 * radius:
        return (f"WARNING: the camera is {offset:.2f} m from the base centre horizontally (limit {0.5 * radius:.2f},"
                f" half the footprint radius): the offset is ignored, ranges are off by up to that much")
    return None


def pitch_down_deg(up_c):
    """Optical axis angle below the horizon (deg) from 'up' in the camera frame (z optical axis)."""
    return math.degrees(math.asin(float(np.clip(-up_c[2], -1, 1))))


class BaseMotion:
    """(a, v, w) for the navigator's tilt trust, from the base's measured velocity (Odometry.v, w): a = dv/dt
    between steps, low-passed with tau_s (a balancing base rocks its wheels back and forth). Measured, not
    commanded: the reference ramps to 0 well before the base stops, and is 0 when a person pushes or carries
    the robot. One odometry sample per step: the base's telemetry is sampled at the frame rate (aliasing
    accepted)."""

    def __init__(self, tau_s=0.2):
        self.tau_s, self.v_prev, self.a = tau_s, None, 0.0

    def update(self, odom, dt):
        """odom: navette.interfaces.Odometry, or None (no base): still."""
        if odom is None:
            return 0.0, 0.0, 0.0
        if self.v_prev is not None and dt > 0:
            self.a += ((odom.v - self.v_prev) / dt - self.a) * min(1.0, dt / self.tau_s)
        self.v_prev = odom.v
        return self.a, odom.v, odom.w


def step_row(t_rel, idx, pose, cam_h, pitch, out, ms):
    """One steps.csv row (STEP_COLS); pose (x, y, heading) as passed to the navigator, ms (seg, scan, nav)."""
    return [f"{t_rel:.3f}", idx, f"{pose[0]:.3f}", f"{pose[1]:.3f}", f"{math.degrees(pose[2]):.1f}", f"{cam_h:.3f}",
            f"{pitch:.1f}", f"{out.v:.3f}", f"{out.w:.3f}", f"{out.v_req:.3f}", f"{out.w_req:.3f}", out.state,
            f"{ms[0]:.1f}", f"{ms[1]:.1f}", f"{ms[2]:.1f}"]


class Status:
    """What the perception loop shows on the page (written by one thread, read by the page: plain attributes)."""

    def __init__(self):
        self.hz, self.frames, self.imu, self.cam_h, self.pitch = 0.0, 0, "", None, None
        self.nav_state, self.preview, self.message = "", None, "starting"

    def snapshot(self):
        return {"hz": round(self.hz, 1), "frames": self.frames, "imu": self.imu, "nav_state": self.nav_state,
                "cam_h": self.cam_h, "pitch": self.pitch, "message": self.message}


MOVING_V = 0.05  # m/s: applied speed from which the arc controller counts the robot as moving (its default)


def navigator_params(cfg):
    """The core's parameter objects for a robot config: {"safety": SafetyParams, "grid": OccupancyGrid's
    keyword arguments, "frontier": FrontierParams, "explore": ExploreParams}. Everything that depends on the
    robot's geometry and speeds is derived here, in one place, from the config; the rest keeps the core's
    defaults (tuning, not geometry):
      footprint.radius            -> the footprint the grid marks free
      footprint.radius + margin   -> the distance the planner's paths keep from obstacles (the collision
                                     monitor's clearance)
      camera.blind_m              -> the planner's blind disk
      explore.v_set               -> the arc controller's candidate speeds, without those above safety.v_max
                                     (the collision monitor would cut them; if none is left: safety.v_max)
      "moving" threshold          -> MOVING_V, or half the slowest speed the robot is sent if that is less
                                     (the slowest candidate, or controller.v_max): a slow robot is not
                                     "blocked" for driving slowly
    The controller's cap (controller.v_max) is applied to the command afterwards (Controller, cap_speed), as
    the Python prototype did: it does not remove candidates here."""
    import navette

    radius, margin = cfg.footprint.radius, cfg.footprint.margin
    v_set = [v for v in cfg.explore.v_set if v <= cfg.safety.v_max] or [cfg.safety.v_max]
    moving_v = min(MOVING_V, 0.5 * min(min(v_set), cfg.controller.v_max))
    return {"safety": navette.SafetyParams(**cfg.safety_params()),
            "grid": {"robot_radius": radius},
            "frontier": navette.FrontierParams(clear=radius + margin, blind=cfg.camera.blind_m),
            "explore": navette.ExploreParams(v_set=v_set, moving_v=moving_v, rotate_in_place=True)}


def new_navigator(cfg, behaviour, forward_v, odometry):
    """A navette.Navigator for a robot config (navigator_params) with its resume gate (resume_n, and the
    restart hysteresis resume_margin). The single place where the runtime builds one."""
    import navette

    p = navigator_params(cfg)
    return navette.Navigator(p["safety"], behaviour=behaviour, odometry=odometry, resume_n=cfg.behaviour.resume_n,
                             forward_v=forward_v, grid=navette.OccupancyGrid(**p["grid"]), frontier=p["frontier"],
                             explore=p["explore"], resume_margin=cfg.behaviour.resume_margin)


def default_navigator(cfg, behaviour, forward_v):
    """run()'s make_navigator: odometry -> new_navigator."""
    new_navigator(cfg, behaviour, forward_v, True)  # a bad parameter fails here, before the robot is opened
    return lambda odometry: new_navigator(cfg, behaviour, forward_v, odometry)


def default_up_filter(g):
    import navette

    return navette.UpFilter(g=g)


def free_space(camera):
    from navette import FreeSpace

    return FreeSpace(np.asarray(camera.rays, float), np.asarray(camera.rays_low, float), tuple(camera.mask_shape))


def write_meta(out_dir, meta):
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def run(robot, cfg, out_dir, stop_event, make_navigator, make_up_filter=default_up_filter, behaviour=None,
        forward_v=None, dry_run=False, web_app=None, on_controller=None, log=print):
    """The robot loop until stop_event.

    robot: navette.interfaces.Robot (already open). out_dir: the run folder (exists).
    make_navigator(odometry: bool) -> a navette.Navigator (step, observe, grid); a new one at every START.
    behaviour, forward_v: only recorded in meta.json (the replay reads them); make_navigator carries them.
    web_app: a FastAPI app to add the page routes to (served by the caller), or None.
    on_controller(ctrl): called once the Controller exists (other START / STOP sources)."""
    camera, imu, base = robot.camera, robot.imu, robot.base
    fs = free_space(camera)
    meta = {"clock": "time.monotonic", "started_wall": datetime.now().isoformat(), "config": cfg.as_dict(),
            "args": {"behaviour": behaviour or cfg.behaviour.default,
                     "forward_v": cfg.behaviour.forward_v if forward_v is None else forward_v,
                     "dry_run": dry_run, "base": base is not None},
            "robot": robot.meta()}
    write_meta(out_dir, meta)
    status = Status()
    threads_stop = threading.Event()
    files, ctrl, n_steps = [], None, 0
    try:
        cmd_f = open(os.path.join(out_dir, "commands.csv"), "w", newline="", buffering=1)
        files.append(cmd_f)
        cmd_w = csv.writer(cmd_f)
        cmd_w.writerow(["t", "ctrl", "cmd", "v", "w", "seq", "dry", "reason"])

        def on_command(t, state, cmd, v, w, seq, reason):
            cmd_w.writerow([f"{t:.4f}", state, cmd, f"{v:.4f}", f"{w:.4f}", "" if seq is None else seq,
                            int(dry_run or base is None), reason])

        ctrl = ctl.Controller(base, cfg.controller.v_max, cfg.controller.w_max, dry_run,
                              stale_s=cfg.perception.stale_s, start_timeout_s=cfg.controller.start_timeout_s,
                              on_command=on_command, log=log, page_timeout_s=cfg.page.timeout_s)
        threading.Thread(target=ctl.command_loop, args=(ctrl, threads_stop, log, cfg.controller.cmd_hz),
                         daemon=True).start()
        robot.on_stop(ctrl.stop)
        if on_controller is not None:
            on_controller(ctrl)
        if web_app is not None:
            from navette.runtime.page import add_routes

            add_routes(web_app, ctrl, status, cfg.page.timeout_s, cfg.page.preview_s)

        steps_f = open(os.path.join(out_dir, "steps.csv"), "w", newline="", buffering=1)
        files.append(steps_f)
        steps_w = csv.writer(steps_f)
        steps_w.writerow(STEP_COLS + LIVE_COLS)
        up = LiveUp(make_up_filter, cfg.perception.up_g_s, cfg.perception.imu_max_age_s)
        loc = OdometryLocalization()
        # odometry moves the pose only with a base that drives: not without a base, nor in a dry run (the robot
        # is carried while the base's wheels stay still: the map would smear)
        odometry = base is not None and not dry_run
        nav, nav_run, motion = make_navigator(odometry), ctrl.run, BaseMotion()
        glance, glance_pending = None, False
        if robot.head is not None:
            g = cfg.glance
            glance = Glance(robot.head, g.n_frames, g.settle_s, g.cooldown_s, g.moved_m, g.turned_deg, g.retries)
        hold = NavOut(0.0, 0.0, 0.0, 0.0, "GLANCE", False)  # zero velocity while the head glances
        preview = _preview_or_none(log)
        t0, t_prev, t_preview, preview_ms, w0 = None, None, 0.0, 0.0, None
        mount_checked = False
        status.message = "ready" + (" (DRY RUN)" if dry_run else "")
        log(status.message)
        while not stop_event.is_set():
            w0 = time.perf_counter() if w0 is None else w0  # from the end of the last frame's work
            frame = camera.read()  # blocks a few tens of ms at most; None: no new frame
            if frame is None:
                continue
            w1 = time.perf_counter()
            if not mount_checked:  # once: a warning, not a refusal (a head on a neck sits a few degrees off)
                mount_checked = True
                warning = camera_mount_warning(frame.T_base_cam, cfg.footprint.radius)
                if warning:
                    log(warning)
            # the base right now: the pose of this frame (a frame late by the camera's lag: 6 mm at 0.15 m/s)
            odom = finite_odometry(base.odometry()) if base is not None else \
                itf.Odometry(frame.t, 0.0, 0.0, 0.0, 0.0, 0.0)
            loc.update(odom)
            in_read = sum(frame.ms.values())  # the adapter's own stages inside read()
            wait_ms, w0 = (w1 - w0) * 1e3 - in_read, None
            t = frame.t
            w2 = time.perf_counter()
            up.add(imu.read())
            status.imu = imu.name
            w3 = time.perf_counter()
            status.frames = frame.index + 1
            up_c = up.up_at(t)
            # no step (no output: the stale-perception watchdog halts a drive) while the IMU is not started,
            # stale or gives no finite 'up', without a finite odometry, or with a camera pose that is not
            # usable: a navigator fed with any of these would answer from a scan that sees nothing
            if up_c is None or loc.pose() is None:
                status.message = "IMU not ready or stale" if up_c is None else "no finite odometry from the base"
                continue
            cam_h, cam_heading = camera_pose(frame.T_base_cam, up_c, cfg.camera.body_z)
            if not usable_camera_pose(cam_h, cam_heading):
                status.message = f"camera pose not usable (height {cam_h:.3f} m)"
                continue
            x, y, th = loc.pose()
            pose = (x, y, th + cam_heading)  # heading: the base's + the camera's in the base frame
            if ctrl.run != nav_run:  # START: a new exploration from here, with an empty map
                nav, nav_run, t_prev = make_navigator(odometry), ctrl.run, None
                glance_pending = glance is not None  # the map is empty: look at the blind zone before moving
            if glance_pending:
                # hold until the base is ready (STARTING -> DRIVING: a stand-up moves the camera a lot), then glance
                if ctrl.state == ctl.DRIVING and glance.begin(t, pose, force=True):
                    glance_pending = False
                elif ctrl.state not in (ctl.STARTING, ctl.DRIVING):  # stopped meanwhile
                    glance_pending = False
            g = glance.frame(t, pose) if glance is not None else None
            if glance is not None and glance.failed:
                log(f"glance: head not back at the navigation pose after {glance.retries} retries")
                glance.failed = False
            if g is None and glance_pending:
                g = "move"  # hold, frame not used
            ms_tail = [f"{wait_ms:.1f}", f"{frame.ms.get('head', 0.0):.1f}", f"{(w3 - w2) * 1e3:.1f}",
                       f"{frame.ms.get('save', 0.0):.1f}"]
            if g is not None:  # glancing: hold still; settled head-down frames go into the map only
                t0 = t if t0 is None else t0
                if g == "look":
                    c0 = time.perf_counter()
                    floor = camera.segment(frame)
                    c1 = time.perf_counter()
                    scan = fs.scan(floor, up_c, cam_h)
                    c2 = time.perf_counter()
                    nav.observe(t, pose, scan, cam_h)
                    steps_w.writerow(step_row(t - t0, frame.index, pose, cam_h, pitch_down_deg(up_c), hold,
                                              ((c1 - c0) * 1e3, (c2 - c1) * 1e3, (time.perf_counter() - c2) * 1e3))
                                     + [ctrl.state, nav_run, "0.0000", imu.name, ""] + ms_tail + [""])
                ctrl.set_nav(hold, t, nav_run)
                status.nav_state = "GLANCE"
                continue
            c0 = time.perf_counter()
            floor = camera.segment(frame)
            c1 = time.perf_counter()
            scan = fs.scan(floor, up_c, cam_h)
            c2 = time.perf_counter()
            dt = 0.1 if t_prev is None else t - t_prev
            # the measured motion matters to a balancing base only (it pitches when it accelerates)
            a, v, w = motion.update(odom, dt) if base is not None and base.balancing else (0.0, 0.0, 0.0)
            v_prev = ctrl.sent[0]
            out = nav.step(t, dt, pose, scan, cam_h, a, v, w, v_prev)
            c3 = time.perf_counter()
            # age measured from the frame's capture; outputs of a navigator from before a START that happened
            # during this step are dropped by the controller (run id)
            ctrl.set_nav(out, t, nav_run)
            lat_ms = (time.monotonic() - t) * 1e3
            if glance is not None and out.blind_stop and ctrl.state in (ctl.STARTING, ctl.DRIVING):
                glance.begin(t, pose)  # stopped by a never-seen blind zone: look at it (cooldown inside)
            t0 = t if t0 is None else t0
            pitch = pitch_down_deg(up_c)
            steps_w.writerow(step_row(t - t0, frame.index, pose, cam_h, pitch, out,
                                      ((c1 - c0) * 1e3, (c2 - c1) * 1e3, (c3 - c2) * 1e3))
                             + [ctrl.state, nav_run, f"{v_prev:.4f}", imu.name, f"{lat_ms:.1f}"] + ms_tail
                             + [f"{preview_ms:.1f}"])
            n_steps += 1
            if t_prev is not None and t > t_prev:
                status.hz = 0.9 * status.hz + 0.1 / (t - t_prev)
            t_prev = t
            status.nav_state = out.state
            status.cam_h, status.pitch = round(cam_h, 3), round(pitch, 1)
            status.message = "DRY RUN" if dry_run else ""
            preview_ms = 0.0
            if preview is not None and frame.image is not None and t - t_preview > cfg.page.preview_s:
                t_preview = t
                p0 = time.perf_counter()
                snap = ctrl.snapshot()
                status.preview = preview.jpeg(preview.view(
                    frame.image, floor, frame.under, out, cam_h, pitch, nav, pose, scan,
                    f"{snap['state']} sent v {snap['v']:.2f} w {snap['w']:+.2f} {status.hz:.1f} Hz"
                    + (" DRY" if dry_run else ""))) or status.preview
                preview_ms = (time.perf_counter() - p0) * 1e3  # logged with the next step
    finally:
        log("exit: stopping")
        if ctrl is not None:
            ctrl.stop("exit")
            t_end = time.monotonic() + STOP_WAIT_S  # the command thread parks the base once it has stopped
            while ctrl.state != ctl.IDLE and time.monotonic() < t_end:
                time.sleep(0.05)
        threads_stop.set()
        for f in files:
            f.close()
        meta.update(frames=status.frames, steps=n_steps, robot=robot.meta())
        write_meta(out_dir, meta)
        log(f"run folder {out_dir}: {status.frames} frames, {n_steps} steps, IMU {imu.name}")


def _preview_or_none(log):
    """The preview module if OpenCV is there (optional dependency), else None: no preview on the page."""
    try:
        from navette.runtime import preview
    except ImportError as e:
        log(f"no preview on the page: {e}")
        return None
    return preview
