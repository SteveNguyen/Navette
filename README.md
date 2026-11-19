# Navette

A small navigation stack for robots with one camera: from a floor mask to a safe base command. A Rust core
with Python bindings, light enough for a Raspberry Pi CM4; a Python runtime around it (loop, START / STOP,
watchdogs, logs, replay, phone page); one adapter and one config file per robot. First robot: Reachy Mini on
the Poussah self-balancing base. Plan and decisions: [docs/PLAN.md](docs/PLAN.md).

What it does each frame: floor mask -> per-bearing scan (a virtual laser scan) -> occupancy grid ->
behaviour (frontier exploration, or straight ahead) -> collision monitor -> velocity smoother -> (v, w).

## Layout

    crates/navette-core/     the core: pure Rust, no dependency (scan, occupancy grid, collision monitor,
                             velocity smoother, attitude, tilt trust, frontier exploration, arc controller,
                             navigator)
    crates/navette-py/       PyO3 bindings (abi3 wheel): module navette._core, re-exported by navette
    python/navette/          Python package
      interfaces.py            the robot contract: Camera, Imu, Base, optional Head
      config.py                robot config (TOML) -> typed parameters
      rays.py                  viewing rays of an OpenCV-calibrated camera (the core takes rays, not a lens model)
      runtime/                 generic loop, controller, replay, status page, command line
      robots/reachy_poussah/   the adapter of the first robot
      testing/                 fake robot, conformance checks for an adapter
    robots/reachy_poussah.toml  the first robot's config
    tests/                   pytest: unit, golden (tests/golden, see its README), closed loop, end to end
    benches/                 timing scripts

The core alone, from Python:

```python
from navette import FreeSpace, Navigator, SafetyParams
from navette.rays import mask_rays

fs = FreeSpace(mask_rays("calib.json"), mask_rays("calib.json", dv=0.5))  # once
nav = Navigator(SafetyParams(), behaviour="explore")
scan = fs.scan(floor_mask, up_c, cam_h)  # bool (192, 320) mask, 'up' in the camera frame, camera height (m)
out = nav.step(t, dt, pose, scan, cam_h, a, v, w, v_prev)  # out.v, out.w: what to send; out.state
```

## Build

Needs a Rust toolchain and [uv](https://docs.astral.sh/uv/). From the repository root:

    uv sync                # creates .venv, builds the extension (release) and installs it, editable

`uv sync` / `uv run` rebuild the extension when a Rust source changes.

## Test

    cargo test                                        # the core's own tests
    uv run --with fastapi --with httpx pytest         # everything else (without fastapi: the page tests skip)
    cargo clippy --workspace --all-targets -- -D warnings
    cargo fmt --all

- Golden tests compare every module with the outputs of the Python prototype it was ported from, on seeded
  synthetic inputs ([tests/golden/README.md](tests/golden/README.md)).
- `tests/test_end_to_end.py` runs the runtime with the real core on a fake robot in a simulated room: no
  hardware.
- `tests/test_real_runs.py` replays recorded robot runs and compares them with the prototype's replay and
  with what the robot logged. Local only (the recordings are private): skipped unless `NAVETTE_RUNS` is set,
  see its docstring.

## Run on a robot

On the robot: the wheel (below), the robot's config and the files its `[adapter]` table names, beside the
config (for the Reachy Mini: the floor model and three calibration files, see `robots/reachy_poussah.toml`).

    pip install "navette-0.1.0-cp310-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl[reachy-poussah]"
    python -m navette.runtime --robot robots/reachy_poussah.toml --out RUNS --dry-run --no-base
    python -m navette.runtime --robot robots/reachy_poussah.toml --out RUNS

(`pip install --no-deps <wheel>` leaves the environment's own numpy, OpenCV... untouched, if the robot's
SDK already brings them.) Nothing moves before START on the status page (`http://<robot>:8090/`); STOP, a watchdog (stale perception,
base fault, page silent) or the end of the exploration stops the base. `--dry-run`: everything runs, nothing
is sent to the base; `--no-base`: no base at all. `--behaviour forward` drives straight through the collision
monitor instead of exploring. Each run writes a folder (`RUNS/<date-time>/`: frames, IMU, base lines,
`steps.csv`, `commands.csv`) that replays on a laptop:

    uv run --extra reachy-poussah python -m navette.runtime --robot robots/reachy_poussah.toml --replay RUNS/<run> --out OUT

## Adding a robot

1. Write an adapter module: a `Camera`, an `Imu`, a `Base` (and a `Head` if the camera can look down) as
   [python/navette/interfaces.py](python/navette/interfaces.py) specifies them (units, frames, clock, threads,
   the base's dead-man), and two functions, `open(cfg, out_dir, dry_run, use_base, log)` and
   `open_replay(cfg, run_dir)`, that return them as a `Robot`. `navette.robots.reachy_poussah` is the example.
2. Write its config, `robots/<name>.toml`: every key of `robots/reachy_poussah.toml` is required
   (`[robot] adapter` is the module's name; `[adapter]` is the adapter's own table).
3. Run the conformance checks on the real devices before the runtime ever drives them:

   ```python
   from navette.testing import conformance
   conformance.check_camera(robot.camera)
   conformance.check_imu(robot.imu)    # robot standing still
   conformance.check_base(robot.base)  # ROBOT MOTION: ~0.1 m forward, a ~0.3 rad turn, then stop and park
   ```

   They check units, signs, timestamps, the session and the dead-man; `tests/test_conformance_fake.py` shows
   them catching wrong adapters.
4. First runs: `--dry-run`, then `--behaviour forward` toward an obstacle, then exploration.

## Bench

| Script | Times |
|--------|-------|
| `uv run python benches/bench_step.py` | the whole per-frame core path, floor mask -> command, on recorded closed-loop sequences with a 320x192 mask per frame: `FreeSpace.scan` + `Navigator.step`, and `Navigator.step_mask` |
| `uv run python benches/bench_scan.py` | `FreeSpace.scan` on 320x192 masks |
| `uv run python benches/bench_occupancy_grid.py` | `OccupancyGrid`: blind_known, update, obstacle_points, replay over a drive through a room |
| `uv run python benches/bench_collision_monitor.py` | `safe_command` (passing, slowed, blocked requests) and a controller's arc scoring |
| `uv run python benches/bench_navigator.py` | `Navigator.step` over recorded closed-loop sequences, steps with a plan apart from the others; the frontier planner's parts, on a room and on a 20 x 20 m hall (its worst case) |

`bench_step.py` on a laptop (x86-64), ms per frame, median / p95 / max, against the Python prototype on the
same masks:

| Sequence | navette | prototype |
|----------|---------|-----------|
| exploration, cluttered room | 1.41 / 2.06 / 4.11 | 2.94 / 5.56 / 9.62 |
| exploration, corridor and doorway | 1.42 / 2.01 / 3.56 | 3.77 / 6.14 / 9.47 |
| forward toward a box | 1.90 / 1.95 / 3.62 | 3.70 / 4.42 / 6.16 |

The scan is 1.35 ms of it (median); the navigator's step 0.02 to 0.06 ms (p95 0.7 ms, a step with a plan).
`step_mask` saves nothing measurable over `scan` + `step`: the runtime uses the two calls, which keeps their
times apart in the run log. Not measured on the CM4 yet.

## Wheel for the robot (aarch64)

The robot has no room for a toolchain: the wheel is cross-built, with the Rust target and zig as the linker.
From the repository root:

    rustup target add aarch64-unknown-linux-gnu      # once
    uv tool run --with ziglang maturin build --release --target aarch64-unknown-linux-gnu --zig

It writes `target/wheels/navette-0.1.0-cp310-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl`: abi3
(any CPython >= 3.10), glibc >= 2.17 (64-bit Raspberry Pi OS), ~0.5 MB, with the extension
(`navette/_core.abi3.so`) and the whole Python package. The robot configs (`robots/`) are not in the wheel.

## Licence

Apache-2.0, see [LICENSE](LICENSE).
