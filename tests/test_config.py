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
    assert (cfg.behaviour.default, cfg.behaviour.forward_v, cfg.behaviour.resume_n) == ("explore", 0.15, 2)
    assert cfg.camera.body_z == 0.325
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
    (r"^resume_n = 2$", "resume_n = 2.5", r"\[behaviour\] resume_n = 2.5: expected int"),
    (r'^default = "explore"$', 'default = "v3"', r"'explore' or 'forward'"),
    (r"^\[page\]$", "[pages]", r"missing tables \['page'\]; unknown tables \['pages'\]"),
])
def test_bad_config_is_an_error(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        config.load(edited(tmp_path, old, new))


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
