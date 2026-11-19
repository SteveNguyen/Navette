# Navette: plan

Navette is a small navigation stack for robots with one camera: a Rust core with Python bindings, light
enough for a Raspberry Pi CM4, generic enough to move from one robot to another. First robot: Reachy Mini on
the Poussah self-balancing base. It is the rewrite of a Python prototype, which is the behaviour reference
for the regression tests.

## Goals

1. **Light.** One navigation step (floor mask -> command) under 10 ms on the CM4 (the prototype: 83 ms median).
   On a laptop: 1.4 ms median against the prototype's 2.9 to 3.8 (README, Bench); the CM4 is phase 6.
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
    src/exploration/frontier_explore.rs  exploration behaviour: the arc controller along the planner's path,
                               look-around, recovery when blocked
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
  tests/                     pytest: unit tests, golden tests, closed-loop tests, end to end on the fake robot
  benches/                   timing of each module and of the whole step
```

Names follow Nav2 where it has an equivalent (scan, occupancy grid, collision monitor, velocity smoother,
controller, navigator, base driver). `attitude` and `tilt_trust` have none.

Out of scope: simulation, calibration, data collection and model training tools.

## Decisions

- **Rust core, Python shell.** Camera, robot SDK, BLE and the floor model stay in Python; everything between
  the mask and the command is Rust. Later the loop itself can move to Rust without changing the core.
- **Two binding levels.** Per-module classes (numpy in / out) for tests and tools, and `Navigator.step(...)`,
  one call per frame, for the robot loop. The loop calls `FreeSpace.scan` then `Navigator.step`:
  `Navigator.step_mask` (both in one call) measured no faster, and two calls keep their times apart in the
  run log.
- **Lens model outside the core.** The core takes per-pixel viewing rays (N x 3), computed once by the
  adapter. Any lens works, and the core needs no OpenCV.
- **The floor model sits behind `Camera`** (the adapter returns a mask): a robot with a depth camera or a
  lidar can feed a mask or a scan without touching the core.
- **abi3 wheel**, cross-built for aarch64 (the Mini has no room for a toolchain).
- **Robot parameters come from the config file**, not from constants in code: the collision monitor's, and
  what the grid, the planner and the arc controller need of the robot's geometry and speeds (footprint,
  clearance, blind distance, candidate speeds), derived in one place (`runtime/loop.py`, `navigator_params`).
  The rest of their parameters is tuning and keeps the core's defaults.
- **Parameters are validated** where an object is built (`validate()` in the core, ValueError in Python, the
  key named when it comes from the config): a bad value is an error, not a hang or a check that never fires.
- **One thread per navigator.** The long calls of the bindings release the GIL (the command thread and the
  page keep running during a slow plan); an object used from another thread during a call raises.
- **Apache-2.0.**

## Robot contract (python/navette/interfaces.py)

| Interface | Methods | Contract |
|---|---|---|
| `Camera` | `read() -> Frame` (`t`, `index`, `T_base_cam`), `segment(frame) -> floor mask`, `rays`, `rays_low`, `mask_shape` | C-contiguous bool mask; per-pixel rays; `t` = capture time, monotonic clock; mounted looking along the base's +x (yaw within 10 deg), close to the vertical through the base centre |
| `Imu` | `read() -> [(t, gyro, acc)]` | camera frame, rad/s, m/s^2, all samples since the last call, every one a finite measurement |
| `Base` | `odometry() -> (t, x, y, theta, v, w)`, `command(v, w)`, `start()`, `stop()`, `park()`, `status()`, `balancing` | x forward, theta counter-clockwise, m, rad; the base stops by itself when commands stop (dead-man; afterwards it resumes on the next command or reports a fault); `stop()` brakes and latches; faults are reported |
| `Head` (optional) | `look_down()`, `look_back()`, `is_back()` | only for robots with a movable camera |

A new robot = these classes + a config file, validated by `navette.testing`'s conformance checks (README,
Adding a robot).

Camera mounting: the navigator's robot frame is the camera's heading frame at the base centre; of
`T_base_cam` it uses the heading and the height. A camera looking sideways, or far from the base centre, is
not supported yet (the conformance check fails, the loop warns): a general camera-to-base transform of the
scan is future work.

## Localization and SLAM

Today the pose is the base's odometry and the map is an occupancy grid in the odometry frame. The final goal
is SLAM and localization on top; whether it fits on the CM4 is open. What the design keeps ready for it:

- **The pose has one source, behind an interface** (`runtime/localization.py`): odometry now, gyro-fused heading next,
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
   Exact on every step, the planner's ties included (Dijkstra predecessors and the cheapest frontier follow
   the prototype's order): command, request, state, target, path length, frontier size, and the final grid
   bit for bit. No step is exempted.
4. **Closed-loop properties** in the 2-D room with Navette driving: no collision, exploration finishes,
   coverage within a few percent of the reference.
5. **End to end, no hardware** (`tests/test_end_to_end.py`): the runtime's loop and controller with the real
   Navigator and UpFilter on the fake robot, whose camera renders the floor mask of a simulated room: START
   explores without collision, to its end; STOP, stale perception and a silent page stop it; the forward
   behaviour stops before a box.
6. **Real runs (local only)** (`tests/test_real_runs.py`). If `NAVETTE_RUNS` points at recorded robot runs:
   the core on the prototype's replay inputs, the runtime's replay against the prototype's replay step by
   step, and against what the robot logged. Skipped otherwise. Recorded frames are private and never
   committed. On the 4 runs recorded with their frames (8559 steps): every output, plan and final grid
   identical to the prototype's replay; 4 scans differ in one sector by 1 to 7 mm (a pixel within one float32
   ulp of a sector boundary: numpy's float32 arctan2 is not correctly rounded, libm's is).
7. **Rust unit tests** (`cargo test`) for the core's own invariants.
8. **Adapter conformance tests** against the fake robot (units, signs, timestamps, dead-man).

## Deliberate differences from the prototype

The core gives the prototype's outputs on the prototype's inputs (the goldens). Where it departs, it is on
inputs the prototype handled unsafely, and on purpose:

- **Attitude filter** (`state_estimation/attitude.rs`): a sample with a non-finite value is ignored, and the
  filter does not start from an accelerometer reading of zero norm (it waits for the next sample). In the
  prototype one such sample made 'up' NaN for good. `update` returns None until the filter has an 'up'.
- **Navigator** (`navigator.rs`): a step whose scan has no seen sector returns v = 0 (w, the request and the
  state as computed; not a `blind_stop`). The collision monitor, unchanged, puts the boundary of unseen
  sectors at `unseen_m` all round, which lets a robot that sees no floor at all creep forward for ever.
- **Restart hysteresis** (`navigator.rs`, `resume_margin`; active only when > 0, 0 by default so the goldens
  are unchanged): after a stop, a step counts toward resuming only if the request also fits with the collision
  monitor's margin + `resume_margin` (same scan and map), `resume_n` steps in a row; stops stay immediate at
  the normal margin. A self-balancing base still travels the lag it built after any stop, most after a short
  burst, and the stop-and-go at the edge of a blocked zone produces such bursts (the reference robot: 0.10 m,
  `resume_n` 4). Stops decided by the `near` rule (an obstacle off the image bottom that the map does not
  explain) do not depend on the margin, so the hysteresis does not change them.
- **Runtime** (`runtime/up.py`, `runtime/loop.py`): no navigation step without a finite 'up', a finite
  odometry and a finite camera height above the floor; the stale-perception watchdog then halts a drive.
- **Parameters**: invalid values are refused (the prototype looped for ever on an arc step of 0).
- **Occupancy grid**: only the last `max_keyframes` keyframes (4000) are kept. The grid is the same; a
  `replay()` rebuilds it from the keyframes kept.
- **Bindings**: a floor mask is read by its values whatever its memory layout.

Unchanged on purpose: the controller's speed cap (`controller.v_max`) scales the command after the collision
monitor and does not limit the arc controller's candidate speeds (`explore.v_set`), as in the prototype's
live loop. The planner's search stops once every frontier cell is settled: same plans.

## Phases

| # | Work | Verified by | Status |
|---|------|-------------|--------|
| 0 | Repo, workspace, bindings build, geometry + scan | cargo test, pytest, scan goldens | done |
| 1 | occupancy_grid; collision_monitor + velocity_smoother; state_estimation (in parallel) | unit tests + goldens | done |
| 2 | exploration/frontier, controller/dwa, navigator | goldens, golden sequences, closed loop | done |
| 3 | Python runtime: config, interfaces, loop + controller, replay, phone page, Poussah driver, Reachy adapter, conformance tests | runtime tests with the fake robot | done |
| 4 | Integration: the runtime on the real core, end to end without hardware, replay of real runs against the prototype; benchmarks; aarch64 wheel | end-to-end tests, real-run comparison, bench table (README) | done; the wheel is built and inspected, not run on an aarch64 machine yet |
| 5 | Independent review, fixes | review findings closed, each with a test (bad IMU sample, mask layout, parameter validation, keyframe bound, contract and conformance gaps, geometry from the config, GIL) | done |
| 6 | On the robot: conformance checks, timing, then the staged live tests again | needs a go | not started |

Phases 1 and 3 ran in parallel after phase 0.
