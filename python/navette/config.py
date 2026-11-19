"""Robot config: one TOML file per robot (robots/<name>.toml) -> typed parameters.

Every robot parameter the runtime uses comes from this file; none has a default in the code, so a missing or
misspelled key is an error at load time, not a silent fallback. The `[adapter]` table is the adapter's own
(link type, model and calibration files...): it is passed to the adapter as a dict, with the entries named in
`adapter.paths` resolved against the config file's folder.

    cfg = load("robots/reachy_poussah.toml")
    navette.SafetyParams(**cfg.safety_params())
"""
import dataclasses
import math
import os
import tomllib
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotCfg:
    name: str
    adapter: str   # module with open(cfg, out_dir, ...) and open_replay(cfg, run_dir): navette.interfaces.Robot


@dataclass(frozen=True)
class Footprint:
    radius: float  # m, circle around the base frame's origin that contains the robot
    margin: float  # m, extra clearance kept by the collision monitor


@dataclass(frozen=True)
class Safety:
    """Collision monitor. Stopping distance planned at speed v: v * latency + v^2 / (2 decel), with
    latency = perception.stale_s + base_lag_s (Config.safety_params)."""
    v_max: float        # m/s, highest speed the monitor plans for (the controller's cap is controller.v_max)
    decel: float        # m/s^2, measured braking of the base
    base_lag_s: float   # s, measured delay between a stop command and the start of braking
    unseen_m: float     # m, where an obstacle is assumed in bearings the camera does not see
    reverse_max: float  # m/s, allowed reverse speed (the camera does not look back)
    step: float         # m, arc sampling
    n_speeds: int       # candidate speeds between 0 and the request


@dataclass(frozen=True)
class Perception:
    stale_s: float         # s, halt if the newest navigator output's frame was captured longer ago than this
    imu_max_age_s: float   # s, no navigation step if the newest IMU sample is older than this at the frame
    up_g_s: float          # s of IMU samples used to measure |g| before the attitude filter starts


@dataclass(frozen=True)
class CameraCfg:
    body_z: float   # m, height of the base frame's origin (Frame.T_base_cam's reference) above the floor
    blind_m: float  # m, radius around the robot the camera cannot see at its navigation pose (the planner's
    #                 blind disk: about where the lowest image row meets the floor)


@dataclass(frozen=True)
class Explore:
    """Exploration behaviour (the arc controller)."""
    v_set: tuple  # m/s, candidate forward speeds. Those above safety.v_max are dropped (the collision monitor
    #               would cut them). controller.v_max is NOT applied here: the command is scaled down to it
    #               afterwards, so a candidate above it only says "as fast as allowed"
    near_known: bool  # the arcs put an obstacle off the image bottom at the clearance radius only where the map
    #                   does not explain it (the collision monitor's rule); false: everywhere (the core default)
    spin_flip_deg: float  # deg, latched turns in place: reverse only if the goal moved more than this since the
    #                       turn started (or on an overshoot); 0 = no latch (the core default)
    spin_lead_s: float    # s, the end of a turn in place is tested on heading + measured yaw rate x this
    spin_accel: float     # rad/s^2, slow down before the end of a turn in place (sqrt(2 a angle)); 0 = off
    spin_w_min: float     # rad/s, the lowest yaw rate of that slowdown (> 0)


@dataclass(frozen=True)
class Behaviour:
    default: str      # "explore" (frontier exploration) or "forward" (straight, through the collision monitor)
    forward_v: float  # m/s, the forward behaviour's request
    resume_n: int     # clear steps needed to move again after a stop (velocity smoother)
    resume_margin: float  # m, restart hysteresis: after a stop, a clear step must also fit with this much more
    #                       clearance than the collision monitor's margin (0 = none)


@dataclass(frozen=True)
class ControllerCfg:
    v_max: float            # m/s, cap on what is sent to the base, after the collision monitor
    w_max: float            # rad/s
    cmd_hz: float           # command rate; must stay well above 1 / (the base's dead-man time)
    start_timeout_s: float  # s, halt if the base is not DRIVING this long after START


@dataclass(frozen=True)
class Page:
    port: int          # status page port
    timeout_s: float   # s, halt a run if the page has not polled for this long; 0 = off
    preview_s: float   # s between two previews (camera + map picture)


@dataclass(frozen=True)
class Glance:
    """Head glance at the ground just ahead (only used if the adapter provides a Head)."""
    n_frames: int       # settled head-down frames put into the map
    settle_s: float     # s waited after each head move
    cooldown_s: float   # s before another glance, unless the robot moved
    moved_m: float      # m: moved this far since the last glance = moved
    turned_deg: float   # deg: turned this much = moved
    retries: int        # times the move back is sent again if the head is not back


@dataclass(frozen=True)
class Config:
    path: str
    robot: RobotCfg
    footprint: Footprint
    safety: Safety
    perception: Perception
    camera: CameraCfg
    explore: Explore
    behaviour: Behaviour
    controller: ControllerCfg
    page: Page
    glance: Glance
    adapter: dict

    def safety_params(self):
        """Keyword arguments of navette.SafetyParams. The latency is the oldest perception the watchdog lets
        drive (perception.stale_s) plus the base's own lag: raising one raises the planned stopping distance."""
        s = self.safety
        return {"radius": self.footprint.radius, "margin": self.footprint.margin, "v_max": s.v_max,
                "decel": s.decel, "latency": self.perception.stale_s + s.base_lag_s, "unseen_m": s.unseen_m,
                "reverse_max": s.reverse_max, "step": s.step, "n_speeds": s.n_speeds}

    def replace(self, section, **values):
        """A copy with some values of one section changed (command line overrides)."""
        return dataclasses.replace(self, **{section: dataclasses.replace(getattr(self, section), **values)})

    def as_dict(self):
        return dataclasses.asdict(self)


_SECTIONS = {"robot": RobotCfg, "footprint": Footprint, "safety": Safety, "perception": Perception,
             "camera": CameraCfg, "explore": Explore, "behaviour": Behaviour, "controller": ControllerCfg, "page": Page,
             "glance": Glance}
_NUMBER = {"float": (int, float), "int": (int,), "str": (str,)}
# Ranges: every number is finite and >= 0; these are > 0 (a zero would divide, never time out, or stop
# nothing), and the page port is a port.
_POSITIVE = {"safety": ("v_max", "decel", "step", "n_speeds"), "perception": ("stale_s", "imu_max_age_s"),
             "camera": ("body_z",), "explore": ("spin_w_min",), "controller": ("v_max", "w_max", "cmd_hz", "start_timeout_s"),
             "page": ("port",), "glance": ("n_frames",)}


def _section(name, cls, raw):
    """One table -> its dataclass: every field present, no other key, the right type (ints accepted as
    floats; bool is not a number)."""
    names = {f.name: f.type if isinstance(f.type, str) else f.type.__name__ for f in dataclasses.fields(cls)}
    missing, extra = sorted(set(names) - set(raw)), sorted(set(raw) - set(names))
    if missing or extra:
        raise ValueError(f"[{name}]: " + "; ".join(
            f"{what} {keys}" for what, keys in (("missing", missing), ("unknown", extra)) if keys))
    for key, kind in names.items():
        v = raw[key]
        if kind == "tuple":  # a list of speeds
            if not isinstance(v, list) or not v or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in v):
                raise ValueError(f"[{name}] {key} = {v!r}: expected a non-empty list of numbers > 0")
            continue
        if kind == "bool":  # a switch: true or false only
            if not isinstance(v, bool):
                raise ValueError(f"[{name}] {key} = {v!r}: expected bool")
            continue
        if isinstance(v, bool) or not isinstance(v, _NUMBER[kind]):
            raise ValueError(f"[{name}] {key} = {v!r}: expected {kind}")
        if kind != "str":
            positive = key in _POSITIVE.get(name, ())
            if not (math.isfinite(v) and (v > 0 if positive else v >= 0)):
                raise ValueError(f"[{name}] {key} = {v!r}: must be finite and {'> 0' if positive else '>= 0'}")
    if name == "page" and raw["port"] > 65535:
        raise ValueError(f"[page] port = {raw['port']!r}: must be at most 65535")
    convert = {"float": float, "tuple": lambda v: tuple(float(x) for x in v)}
    return cls(**{k: convert.get(names[k], lambda v: v)(v) for k, v in raw.items()})


def load(path):
    """Read and check a robot config."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = sorted(set(raw) - set(_SECTIONS) - {"adapter"})
    absent = sorted(set(_SECTIONS) - set(raw))
    if unknown or absent:
        raise ValueError(f"{path}: " + "; ".join(
            f"{what} {keys}" for what, keys in (("missing tables", absent), ("unknown tables", unknown)) if keys))
    try:
        parts = {name: _section(name, cls, raw[name]) for name, cls in _SECTIONS.items()}
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    if parts["behaviour"].default not in ("explore", "forward"):
        raise ValueError(f"{path}: [behaviour] default must be 'explore' or 'forward'")
    adapter = dict(raw.get("adapter", {}))
    folder = os.path.dirname(os.path.abspath(path))
    for key in adapter.get("paths", []):  # files of the adapter, relative to the config file
        adapter[key] = os.path.normpath(os.path.join(folder, adapter[key]))
    return Config(path=os.path.abspath(path), adapter=adapter, **parts)
