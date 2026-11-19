"""Robot config loading (navette.config) and the reference robot's file (robots/reachy_poussah.toml)."""
import importlib
import os
import re
from pathlib import Path

import pytest

from navette import config
from navette.runtime import cli

ROBOT_TOML = Path(__file__).parents[1] / "robots" / "reachy_poussah.toml"


def test_reachy_poussah_values():
    """The values measured on the robot (stopping distance fit, latency, watchdog and page timeouts)."""
    cfg = config.load(ROBOT_TOML)
    assert cfg.robot.name == "reachy_poussah" and cfg.robot.adapter == "navette.robots.reachy_poussah"
    assert (cfg.footprint.radius, cfg.footprint.margin) == (0.13, 0.05)
    assert (cfg.safety.decel, cfg.safety.base_lag_s, cfg.perception.stale_s) == (0.64, 0.63, 0.9)
    assert cfg.safety_params() == {"radius": 0.13, "margin": 0.05, "v_max": 0.4, "decel": 0.64,
                                   "latency": pytest.approx(0.9 + 0.63), "unseen_m": 0.35, "reverse_max": 0.0,
                                   "step": 0.02, "n_speeds": 20}
    assert (cfg.controller.v_max, cfg.controller.w_max, cfg.controller.cmd_hz) == (0.15, 0.5, 20.0)
    assert cfg.controller.start_timeout_s == 10.0 and cfg.perception.imu_max_age_s == 0.1
    assert (cfg.page.port, cfg.page.timeout_s, cfg.page.preview_s) == (8090, 2.0, 2.0)
    assert (cfg.behaviour.default, cfg.behaviour.forward_v) == ("explore", 0.15)
    # restart hysteresis: stop-and-go bursts overshoot on the balancing base
    assert (cfg.behaviour.resume_n, cfg.behaviour.resume_margin) == (4, 0.10)
    assert cfg.camera.body_z == 0.325 and cfg.camera.blind_m == 0.5
    assert cfg.explore.v_set == (0.1, 0.2, 0.3)
    # the exploration fixes measured on the robot's runs, on (the core's defaults are off)
    assert cfg.explore.near_known is True
    assert (cfg.explore.spin_flip_deg, cfg.explore.spin_lead_s, cfg.explore.spin_accel, cfg.explore.spin_w_min) == (
        45.0, 0.5, 0.5, 0.3)
    assert (cfg.explore.blocked_s, cfg.explore.blocked_progress, cfg.explore.progress_m, cfg.explore.progress_deg) == (
        3.0, True, 0.05, 17.0)
    assert (cfg.explore.recover_replan, cfg.explore.recover_hold_s) == (False, 0.0)  # C off: unproven
    # near persistence: the floor model's single-frame flicker at the image bottom (the core's default: 1 of 1)
    assert (cfg.perception.near_k, cfg.perception.near_n) == (2, 3)
    assert cfg.teleop.timeout_s == 0.5  # operator dead-man of the phone joystick
    assert cfg.glance.n_frames == 2 and isinstance(cfg.glance.n_frames, int)
    a = cfg.adapter
    assert a["glance_pitch_deg"] == 0  # the head glance is off: it upset the balancing base
    assert (a["base_link"], a["camera_latency_s"], a["model_threads"], a["nav_pitch_deg"]) == ("ble", 0.038, 2, 15.0)


def test_planned_stopping_distance():
    """v * latency + v^2 / (2 decel): 0.25 m at 0.15 m/s, 0.34 at 0.2, 0.53 at 0.3."""
    p = config.load(ROBOT_TOML).safety_params()
    stop = lambda v: v * p["latency"] + v * v / (2 * p["decel"])
    assert [round(stop(v), 2) for v in (0.15, 0.2, 0.3)] == [0.25, 0.34, 0.53]


def test_adapter_paths_are_relative_to_the_config_file_and_the_adapter_imports():
    cfg = config.load(ROBOT_TOML)
    folder = os.path.dirname(cfg.path)
    for key in cfg.adapter["paths"]:
        assert os.path.isabs(cfg.adapter[key]) and cfg.adapter[key].startswith(folder)
    assert cfg.adapter["model"] == os.path.join(folder, "models", "floor.onnx")
    adapter = importlib.import_module(cfg.robot.adapter)  # without the robot SDK, bleak or onnxruntime
    assert callable(adapter.open) and callable(adapter.open_replay)


def edited(tmp_path, old, new, count=1):
    text = ROBOT_TOML.read_text()
    assert len(re.findall(old, text, flags=re.M)) == count, old
    p = tmp_path / "robot.toml"
    p.write_text(re.sub(old, new, text, flags=re.M))
    return p


@pytest.mark.parametrize("old, new, message", [
    (r"^decel = 0.64$", "", r"\[safety\]: missing \['decel'\]"),                    # no silent default
    (r"^decel = 0.64$", "decel = 0.64\ndecell = 1.0", r"\[safety\]: unknown \['decell'\]"),  # a typo is not ignored
    (r"^decel = 0.64$", 'decel = "0.64"', r"\[safety\] decel = '0.64': expected float"),
    (r"^decel = 0.64$", "decel = true", r"expected float"),
    (r"^resume_n = 4$", "resume_n = 2.5", r"\[behaviour\] resume_n = 2.5: expected int"),
    (r"^resume_margin = 0.10$", "", r"\[behaviour\]: missing \['resume_margin'\]"),
    (r"^resume_margin = 0.10$", "resume_margin = \"0.1\"", r"\[behaviour\] resume_margin = '0.1': expected float"),
    (r'^default = "explore"$', 'default = "v3"', r"'explore', 'forward' or 'teleop'"),
    (r"^timeout_s = 0.5$", "", r"\[teleop\]: missing \['timeout_s'\]"),
    (r"^timeout_s = 0.5$", "timeout_s = 0.0", r"\[teleop\] timeout_s = 0.0: must be finite and > 0"),
    (r"^timeout_s = 0.5$", "timeout_s = nan", r"\[teleop\] timeout_s = nan"),
    (r"^\[teleop\]$", "", r"missing tables \['teleop'\]"),
    (r"^\[page\]$", "[pages]", r"missing tables \['page'\]; unknown tables \['pages'\]"),
    (r"^v_set = \[0.1, 0.2, 0.3\]$", "v_set = []", r"\[explore\] v_set = \[\]: expected a non-empty list"),
    (r"^v_set = \[0.1, 0.2, 0.3\]$", "v_set = [0.1, 0.0]", r"\[explore\] v_set"),
    (r"^v_set = \[0.1, 0.2, 0.3\]$", "v_set = 0.1", r"\[explore\] v_set = 0.1: expected a non-empty list"),
    (r"^v_set = \[0.1, 0.2, 0.3\]$", 'v_set = ["0.1"]', r"\[explore\] v_set"),
    (r"^blind_m = 0.5$", "blind_m = -0.5", r"\[camera\] blind_m"),
    (r"^near_known = true$", "", r"\[explore\]: missing \['near_known'\]"),
    (r"^near_known = true$", "near_known = 1", r"\[explore\] near_known = 1: expected bool"),
    (r"^near_known = true$", 'near_known = "true"', r"\[explore\] near_known = 'true': expected bool"),
    (r"^spin_w_min = 0.3$", "spin_w_min = 0.0", r"\[explore\] spin_w_min = 0.0: must be finite and > 0"),
    (r"^spin_lead_s = 0.5$", "spin_lead_s = -0.5", r"\[explore\] spin_lead_s = -0.5: must be finite and >= 0"),
    (r"^spin_flip_deg = 45.0$", "spin_flip_deg = nan", r"\[explore\] spin_flip_deg = nan"),
    (r"^blocked_progress = true$", "blocked_progress = 0", r"\[explore\] blocked_progress = 0: expected bool"),
    (r"^progress_m = 0.05$", "progress_m = -0.05", r"\[explore\] progress_m = -0.05: must be finite and >= 0"),
    (r"^blocked_s = 3.0$", "blocked_s = inf", r"\[explore\] blocked_s = inf"),
    (r"^recover_replan = false$", "", r"\[explore\]: missing \[.recover_replan.\]"),
    (r"^recover_hold_s = 0.0$", "recover_hold_s = -2.0", r"\[explore\] recover_hold_s = -2.0: must be finite and >= 0"),
    # ranges: the key is named
    (r"^decel = 0.64$", "decel = 0.0", r"\[safety\] decel = 0.0: must be finite and > 0"),
    (r"^step = 0.02$", "step = 0", r"\[safety\] step = 0: must be finite and > 0"),
    (r"^step = 0.02$", "step = -0.02", r"\[safety\] step"),
    (r"^n_speeds = 20$", "n_speeds = 0", r"\[safety\] n_speeds = 0: must be finite and > 0"),
    (r"^v_max = 0.4$", "v_max = nan", r"\[safety\] v_max = nan"),
    (r"^base_lag_s = 0.63$", "base_lag_s = -0.1", r"\[safety\] base_lag_s = -0.1: must be finite and >= 0"),
    (r"^radius = 0.13 ", "radius = -0.13 ", r"\[footprint\] radius"),
    (r"^margin = 0.05$", "margin = inf", r"\[footprint\] margin = inf"),
    (r"^stale_s = 0.9$", "stale_s = 0", r"\[perception\] stale_s"),
    (r"^body_z = 0.325$", "body_z = 0.0", r"\[camera\] body_z"),
    (r"^cmd_hz = 20.0 ", "cmd_hz = 0.0 ", r"\[controller\] cmd_hz"),
    (r"^v_max = 0.15 ", "v_max = -0.15 ", r"\[controller\] v_max"),
    (r"^port = 8090$", "port = 0", r"\[page\] port"),
    (r"^port = 8090$", "port = 70000", r"\[page\] port = 70000: must be at most 65535"),
    (r"^timeout_s = 2.0$", "timeout_s = -1.0", r"\[page\] timeout_s"),
    (r"^n_frames = 2$", "n_frames = 0", r"\[glance\] n_frames"),
    (r"^resume_n = 4$", "resume_n = -1", r"\[behaviour\] resume_n"),
    (r"^resume_margin = 0.10$", "resume_margin = -0.1", r"\[behaviour\] resume_margin = -0.1: must be finite and >= 0"),
    (r"^resume_margin = 0.10$", "resume_margin = nan", r"\[behaviour\] resume_margin = nan"),
    (r"^resume_margin = 0.10$", "resume_margin = inf", r"\[behaviour\] resume_margin = inf"),
    (r"^near_k = 2$", "", r"\[perception\]: missing \['near_k'\]"),
    (r"^near_k = 2$", "near_k = 0", r"\[perception\] near_k = 0: must be finite and > 0"),
    (r"^near_k = 2$", "near_k = 2.0", r"\[perception\] near_k = 2.0: expected int"),
    (r"^near_k = 2$", "near_k = 4", r"\[perception\] near_k = 4: must be at most near_n \(3\)"),
    (r"^near_n = 3$", "near_n = 0", r"\[perception\] near_n = 0: must be finite and > 0"),
])
def test_bad_config_is_an_error(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        config.load(edited(tmp_path, old, new))


def test_teleop_can_be_the_default_behaviour(tmp_path):
    cfg = config.load(edited(tmp_path, r'^default = "explore"$', 'default = "teleop"'))
    assert cfg.behaviour.default == "teleop"


def test_cli_teleop_and_session():
    parse = cli.build_parser().parse_args
    args = parse(["--robot", "r.toml", "--out", "x", "--behaviour", "teleop", "--session", "home-1_a"])
    assert (args.behaviour, args.session) == ("teleop", "home-1_a")
    assert parse(["--robot", "r.toml", "--out", "x"]).session is None
    assert parse(["--robot", "r.toml", "--out", "x", "--session", "a" * 32]).session == "a" * 32
    for bad in ["", "Home", "-home", "_a", "a b", "a/b", "a" * 33, "é", "a.b"]:
        with pytest.raises(SystemExit):
            parse(["--robot", "r.toml", "--out", "x", "--session", bad])
    with pytest.raises(SystemExit):
        parse(["--robot", "r.toml", "--out", "x", "--behaviour", "joystick"])


def test_ints_are_accepted_as_floats(tmp_path):
    cfg = config.load(edited(tmp_path, r"^timeout_s = 2.0$", "timeout_s = 0"))
    assert cfg.page.timeout_s == 0.0 and isinstance(cfg.page.timeout_s, float)


def test_replace_and_command_line_overrides():
    cfg = config.load(ROBOT_TOML)
    assert cfg.replace("controller", v_max=0.3).controller.v_max == 0.3 and cfg.controller.v_max == 0.15
    args = cli.build_parser().parse_args(["--robot", str(ROBOT_TOML), "--out", "x", "--stale-s", "0.6", "--v-max", "0.2",
                                          "--page-timeout-s", "0", "--port", "8000", "--w-max", "0.4"])
    cfg = cli.load_config(args)
    assert (cfg.controller.v_max, cfg.controller.w_max, cfg.page.timeout_s, cfg.page.port) == (0.2, 0.4, 0.0, 8000)
    # the collision monitor's latency follows the staleness limit: the two cannot disagree
    assert cfg.safety_params()["latency"] == pytest.approx(0.6 + 0.63)
    plain = cli.load_config(cli.build_parser().parse_args(["--robot", str(ROBOT_TOML), "--out", "x"]))
    assert plain == config.load(ROBOT_TOML)
    assert cfg.as_dict()["safety"]["decel"] == 0.64
