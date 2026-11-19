# Navette: plan

Navette is a small navigation stack for robots with one camera: a Rust core with Python bindings, light
enough for a Raspberry Pi CM4, generic enough to move from one robot to another. First robot: Reachy Mini on
the Poussah self-balancing base. It is the rewrite of a Python prototype, which is the behaviour reference
for the regression tests.

## Goals

1. **Light.** One navigation step (floor mask -> command) under 10 ms on the CM4 (the prototype: 83 ms median).
2. **Generic.** Nothing robot-specific in the core. One config file per robot; camera, IMU and base behind
   small interfaces with a written contract and a conformance test suite.
3. **No behaviour regression.** Same outputs as the prototype on the same inputs, checked by tests.
4. **Ready for SLAM.** See "Localization and SLAM" below.

## Layout

```
navette/
  Cargo.toml                 workspace
  crates/navette-core/       pure Rust, no Python, no heavy dependency
    src/geometry.rs            gravity frame, poses, angles
    src/scan.rs                floor mask -> per-bearing scan (a virtual laser scan)
    src/occupancy_grid.rs      log-odds grid, keyframes, blind-zone query
    src/collision_monitor.rs   safe command, free arc lengths
    src/velocity_smoother.rs   resume gate (stop at once, restart after n clear steps)
    src/state_estimation/      attitude ('up' from the IMU), tilt_trust (range trust of a balancing base)
    src/exploration/frontier.rs  frontier planner (Dijkstra, distance transform, labelling)
    src/controller/dwa.rs      arc sampling and scoring
    src/navigator.rs           one step: grid -> behaviour -> collision monitor -> smoother
  crates/navette-py/         PyO3 bindings (abi3 wheel), module navette._core
  python/navette/            Python package
    __init__.py                re-exports the core classes
    config.py                  robot config (TOML) -> parameter objects
    interfaces.py              Camera, Imu, Base protocols (the contract); optional Head
    runtime/                   generic loop, START/STOP controller, watchdogs, logging, replay, phone page
    robots/reachy_poussah/     camera + floor model + head IMU (Reachy Mini SDK), Poussah BLE base driver
    testing/                   fake robot, conformance tests any adapter can be run against
  robots/reachy_poussah.toml robot config (footprint, speeds, latency, decel, camera height, calibration)
  tests/                     pytest: unit tests, golden tests, closed-loop tests
  benches/                   step timing
```

Names follow Nav2 where it has an equivalent (scan, occupancy grid, collision monitor, velocity smoother,
controller, navigator, base driver). `attitude` and `tilt_trust` have none.

Out of scope: simulation, calibration, data collection and model training tools.

## Decisions

- **Rust core, Python shell.** Camera, robot SDK, BLE and the floor model stay in Python; everything between
  the mask and the command is Rust. Later the loop itself can move to Rust without changing the core.
- **Two binding levels.** Per-module classes (numpy in / out) for tests and tools, and `Navigator.step(...)`,
  one call per frame, for the robot loop.
- **Lens model outside the core.** The core takes per-pixel viewing rays (N x 3), computed once by the
  adapter. Any lens works, and the core needs no OpenCV.
- **The floor model sits behind `Camera`** (the adapter returns a mask): a robot with a depth camera or a
  lidar can feed a mask or a scan without touching the core.
- **abi3 wheel**, cross-built for aarch64 (the Mini has no room for a toolchain).
- **Robot parameters come from the config file**, not from constants in code.
- **Apache-2.0.**

## Robot contract (python/navette/interfaces.py)

| Interface | Methods | Contract |
|---|---|---|
| `Camera` | `read() -> (t, floor_mask)`, `rays` | bool mask; per-pixel rays; `t` = capture time, monotonic clock |
| `Imu` | `read() -> [(t, gyro, acc)]` | camera frame, rad/s, m/s^2, all samples since the last call |
| `Base` | `odometry() -> (t, x, y, theta, v, w)`, `command(v, w)`, `start()`, `stop()`, `status()` | x forward, theta counter-clockwise, m, rad; the base stops by itself when commands stop (dead-man); faults are reported |
| `Head` (optional) | look down / back | only for robots with a movable camera |

A new robot = these classes + a config file, validated by `navette.testing`'s conformance tests.

## Localization and SLAM

Today the pose is the base's odometry and the map is an occupancy grid in the odometry frame. The final goal
is SLAM and localization on top; whether it fits on the CM4 is open. What the design keeps ready for it:

- **The pose has one source, behind an interface** (`localization/`): odometry now, gyro-fused heading next,
  a SLAM pose later. Nothing else in the stack reads the wheels directly.
- **Frames as REP 105**: `odom` (continuous, drifts) and `map` (corrected, may jump). The grid lives in `map`;
  with no SLAM, `map` = `odom`. The collision monitor only uses the robot frame, so a pose jump cannot make
  it unsafe.
- **The grid keeps its keyframes** (pose, scan, weights) and can be rebuilt from them after a pose-graph
  correction (`replay()`), so a SLAM back end only has to return corrected keyframe poses.
- **The runtime can log everything a SLAM front end needs** (frames, IMU, odometry, keyframes), so SLAM can
  first be developed offline on recorded runs, and the CM4 question answered by measurement (on board, or
  off board on a laptop over WiFi, or on a companion computer).

Not built now: no SLAM code, no map persistence.

## Regression tests

1. **Unit tests** of each module through the Python API (the prototype's own tests, same assertions).
2. **Golden tests, per module.** Reference outputs of the prototype on seeded synthetic inputs (.npz in
   tests/golden/). Exact for booleans, indices and states; 1e-6 for f64; 1e-4 for values that go through f32.
3. **Golden sequences, end to end.** A synthetic 2-D room (ray-cast scans, seeded) driven closed-loop by the
   prototype; the recorded per-step inputs are replayed through the Navigator and every output compared.
   Planner ties (Dijkstra predecessors, argmin) may differ: the test demands the same collision-monitor
   output for the same request on every step and the same state sequence; any tolerated difference is listed
   explicitly in the test, not hidden by a loose threshold.
4. **Closed-loop properties** in the 2-D room with Navette driving: no collision, exploration finishes,
   coverage within a few percent of the reference.
5. **Real runs (local only).** If `NAVETTE_RUNS` points at recorded robot runs, replay them and compare.
   Skipped otherwise. Recorded frames are private and never committed.
6. **Rust unit tests** (`cargo test`) for the core's own invariants.
7. **Adapter conformance tests** against the fake robot (units, signs, timestamps, dead-man).

## Phases

| # | Work | Verified by |
|---|------|-------------|
| 0 | Repo, workspace, bindings build, geometry + scan | cargo test, pytest, scan goldens |
| 1 | occupancy_grid; collision_monitor + velocity_smoother; state_estimation (in parallel) | unit tests + goldens |
| 2 | exploration/frontier, controller/dwa, navigator | goldens, golden sequences, closed loop |
| 3 | Python runtime: config, interfaces, loop + controller, replay, phone page, Poussah driver, Reachy adapter, conformance tests | runtime tests with the fake robot |
| 4 | Benchmarks, aarch64 wheel | bench report |
| 5 | Independent review, fixes | review findings closed |
| 6 | On the robot: timing, then the staged live tests again | needs a go |

Phases 1 and 3 run in parallel after phase 0.
