# Installing Navette

Two machines: a **laptop** (Linux, x86-64) builds and tests, and the **robot** (Reachy Mini Wireless, Raspberry
Pi CM4, 64-bit Raspberry Pi OS) runs. The robot has too little disk space for a Rust toolchain, so the wheel is
cross-built on the laptop and copied over. Nothing here touches the robot's own Python environments.

Reference setup: Reachy Mini (Wireless) on the Poussah base. Commands assume the robot answers as
`reachy-mini.local` and its user is `pollen`.

**Base firmware**: `hoverboard-nav` branch of `pollen-robotics/firmware_poussah`, commit 8c704f1 or later (it loads
its Mini settings from its seat switch, gives the gyro heading and the ready-to-stand flag). Build and flash over
USB from that repository (see its `HOVERBOARD.md`): `pio run -e hoverboard`, then `tools/flash.sh` (keeps the
base's own calibration). The base must also be calibrated (`[hello] setup=7`): see `HOVERBOARD.md`, "Per-base
setup". Older firmware works only with the settings line in the config (`[adapter] stand_settings`, the comment
there gives it) and without the gyro heading (heading then drifts ~30 deg per 2 m).

## 1. Laptop

Prerequisites: [uv](https://docs.astral.sh/uv/), [rustup](https://rustup.rs/) (stable toolchain), git, and the
Hugging Face CLI for the floor model (`uv tool install huggingface_hub`, gives `hf`).

```bash
git clone https://github.com/SteveNguyen/Navette.git navette
uv sync --project navette                       # builds the extension, installs the dev tools
uv run --project navette pytest -q navette/tests    # ~1.5 min, no hardware
cargo test --manifest-path navette/Cargo.toml
```

Build the robot's wheel (aarch64, abi3: any CPython >= 3.10, glibc >= 2.17):

```bash
rustup target add aarch64-unknown-linux-gnu      # once
uv tool run --with ziglang maturin build --release --target aarch64-unknown-linux-gnu --zig \
    -m navette/crates/navette-py/Cargo.toml
# -> navette/target/wheels/navette-0.1.0-cp310-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
```

## 2. Files the robot needs

The robot config `robots/reachy_poussah.toml` names four files in its `[adapter]` table, relative to the config:

| Key | File | Where it comes from |
|---|---|---|
| `model` | `models/floor.onnx` | the floor segmentation model of the `floor_seg` project, `v7_floor_under` for now (public development repo): `hf download SteveNguyen/reachy-mini-floor-seg-dev --include "v7_floor_under/floor.onnx" --local-dir <dir>` |
| `calib` | `calib/calib.json` | camera intrinsics `{"image_resolution", "K", "D"}` of the head camera |
| `r_cam_imu` | `calib/R_cam_imu.json` | rotation from the head IMU's axes to the camera's |
| `handeye` | `calib/handeye_X.json` | camera pose in the head frame |

The three calibration files are in the repository, `robots/calib/`: they are the reference unit's, and fit another
Reachy Mini of the same hardware closely enough to start (same lens, same head). The calibration tools are not
part of Navette yet. Only the floor model has to be fetched.

Check `[adapter]` in the config: `base_link` (`"ble"` or `"usb"`), `ble_name` (the base's Bluetooth name, from its
`[hello] name=`), `usb_port`. A base with another seat height (the reference one is 4 cm): `[camera] body_z` =
0.177 + seat height (m); every distance the camera measures scales with it.

## 3. Robot

Copy the wheel, the config and its files (the robot keeps them under `~/navette/`):

```bash
ssh pollen@reachy-mini.local "mkdir -p ~/navette/wheel ~/navette/pylib ~/navette/robot/models ~/navette/robot/calib ~/navette/runs"
scp navette/target/wheels/navette-*.whl pollen@reachy-mini.local:navette/wheel/
scp navette/robots/reachy_poussah.toml pollen@reachy-mini.local:navette/robot/
scp floor.onnx pollen@reachy-mini.local:navette/robot/models/
scp navette/robots/calib/*.json pollen@reachy-mini.local:navette/robot/calib/
```

Install into a private folder, without dependencies: the robot's app environment (`/venvs/apps_venv`) already
has numpy, OpenCV, onnxruntime, FastAPI and uvicorn, and they must not be replaced. Only the Bluetooth client is
missing:

```bash
ssh pollen@reachy-mini.local
/venvs/apps_venv/bin/python -m pip install --no-deps --upgrade --target ~/navette/pylib \
    ~/navette/wheel/navette-0.1.0-cp310-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
/venvs/apps_venv/bin/python -m pip install --target ~/navette/pylib bleak    # once; brings dbus-fast
```

(A later `--upgrade` of the wheel replaces the `navette` package only.) Check:

```bash
PYTHONPATH=~/navette/pylib /venvs/apps_venv/bin/python -c "import navette, bleak; print(navette.Navigator())"
PYTHONPATH=~/navette/pylib /venvs/apps_venv/bin/python -c \
    "from navette.config import load; load('$HOME/navette/robot/reachy_poussah.toml'); print('config ok')"
```

Optional, once per software version: the test suite on the robot (~7 min; copy `navette/tests` over first). It
checks that the CM4's maths library gives the same results as the laptop:

```bash
PYTHONPATH=~/navette/pylib /venvs/apps_venv/bin/python -m pytest -q -p no:cacheprovider ~/navette/tests
```

## 4. Before each session

- **Wake the robot**: its motors must be enabled (from the dashboard, or
  `curl -X POST http://localhost:8000/api/motors/set_mode/enabled` on the robot). With the motors off the head
  cannot reach its navigation pose and the run refuses to start.
- **No other app** may hold the robot (`curl http://localhost:8000/api/apps/current-app-status` says `null`).
- **Base**: powered, sitting still for a few seconds after boot (it learns its gyro offset), Mini on top
  (seat switch: `[diag] mini=1`).
- **Disk**: a run writes ~20 MB per minute of frames; keep an eye on `df -h /` and copy runs off (below).
- **Room**: ~0.3 m free in front of and behind the robot at START (the stand-up sways ~±14 cm).

## 5. Run

On the robot:

```bash
cd ~/navette
PYTHONPATH=~/navette/pylib /venvs/apps_venv/bin/python -m navette.runtime \
    --robot ~/navette/robot/reachy_poussah.toml --out ~/navette/runs --behaviour explore
```

- The head moves to its navigation pose at once. **The base does not move until START.**
- Open the status page on a phone or laptop on the same network: `http://reachy-mini.local:8090/` (Android does
  not resolve `.local` names: use the robot's IP address; reload the page if it was open from an earlier run).
  START, STOP; pushing an antenna is also a STOP.
- **What exploration looks like**: after START the base stands up (it lurches back, then sways forward, ~±14 cm)
  and settles ~2 s; then it explores at up to 0.15 m/s. It often turns on the spot: the camera sees only ~70 deg
  ahead and nothing closer than ~0.3 m, so it turns to look before it drives. It stops close to obstacles and
  turns away; it never reverses. When nothing is left to explore it turns once all round, then stops by itself
  ("exploration finished") and sits down. The phone page shows the camera with the floor in green and the map.
- **Known limits** (floor model): doorsteps, thresholds and some doormats read as obstacles (the robot may not
  leave a room); small false obstacles appear on shiny or sunlit tiles; floor under low furniture is not always
  recognised. The robot then stops or turns away: it is a limit, not a fault. Tell us, with the run folder.
- The robot halts by itself if the page stops polling for 2 s, if perception goes stale, or on a base fault.
- `--behaviour forward`: straight ahead through the collision monitor. `--dry-run`: everything runs, nothing is
  sent to the base. `--no-base`: no base at all (head and camera only).
- **Driving by hand (recording sessions)**: `--behaviour teleop --session <name>`, e.g.

  ```bash
  PYTHONPATH=~/navette/pylib /venvs/apps_venv/bin/python -m navette.runtime \
      --robot ~/navette/robot/reachy_poussah.toml --out ~/navette/runs --behaviour teleop --session alice-flat-1
  ```

  The page then shows a joystick under START / STOP: press START, then one thumb on the disk, up = forward,
  sideways = turn; lift it and the robot stops (it also stops by itself 0.5 s after the last joystick update,
  `[teleop] timeout_s`). The speed is capped as in exploration (0.15 m/s, 0.5 rad/s) and there is no reverse (the
  camera only looks forward): turn round instead. The requests still go through the collision monitor: under the
  joystick, "joystick" is what you ask, "allowed" what the robot may do, **HELD** when it holds you before an
  obstacle (turn away). Every watchdog stays on (silent page, stale perception, base fault). The session name
  (lower case letters, digits, `_` and `-`, up to 32 characters, starting with a letter or digit) goes into the
  run's `meta.json` (`"session"`) for the dataset import.
- Ctrl-C ends the program (the base is stopped and sat down). If the ssh session drops, the program ends too and
  the base's own dead-man stops it; to run independently of the ssh session, start it with `setsid -f` and a log
  file.

## 6. After a run

Each run is a folder `~/navette/runs/<date-time>/` (frames, IMU, base link log, `steps.csv`, `commands.csv`,
`meta.json`). `steps.csv` has one row per navigation step: pose, the camera's height, pitch (`pitch_down_deg`),
roll (`roll_deg`) and 'up' in the camera frame (`up_x`, `up_y`, `up_z`; these four columns come last, and runs
recorded before them have none), the commands. Recorded frames show the inside of a home: keep them private. Copy it to the laptop and free the
robot's disk:

```bash
scp -r pollen@reachy-mini.local:navette/runs/<date-time> <private folder>/
ssh pollen@reachy-mini.local "rm -rf ~/navette/runs/<date-time>"
```

Replay it on the laptop (the model and calibration files beside a copy of the config, as on the robot):

```bash
uv run --project navette --extra reachy-poussah python -m navette.runtime \
    --robot <config folder>/reachy_poussah.toml --replay <private folder>/<date-time> --out <output folder>
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `head not at the navigation pose ... motors disabled?` | wake the robot (section 4) |
| START refused: `not ready to stand` / `sitting down` | the base is still settling on its rest after a sit: wait a second, press START again |
| START refused: `setup=...` | the base is not calibrated (firmware setup must be 7) |
| `no [nav] from the base` | base off, out of Bluetooth range, or `ble_name` wrong |
| page unreachable from a phone | `.local` name on Android: use the IP; the robot's WiFi can also drop (the run then halts by itself) |
| robot halted, page and ssh unreachable, head still holding | the robot's WiFi dropped (seen twice on the reference unit, during runs with the Bluetooth base link); the program keeps running and logging; it does not come back by itself: restart the Mini (head on the body), then copy the run |
| no joystick on the page in a teleop run | the browser shows an earlier run's page: reload it |
