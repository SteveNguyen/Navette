# Navette: notes for coding agents

Navette is a navigation stack for small robots with one camera: a Rust core (`crates/navette-core`), PyO3
bindings (`crates/navette-py`, module `navette._core`) and a Python runtime (`python/navette`). First robot:
Reachy Mini on the Poussah self-balancing base. Start with `README.md`, then `docs/PLAN.md` (design, decisions,
deliberate behaviour choices) and `docs/INSTALL.md` (laptop, robot, running, troubleshooting).

## Safety rules (a real robot drives on this code, in homes)

- **Never move the robot without an explicit "go" from the human for that action**: starting the runtime moves
  the head; START drives the base; enabling the motors, conformance checks (`tools/check_poussah_base.py`) and
  head moves all count. Prefer `--dry-run` / `--no-base` first. Stopping (STOP, Ctrl-C) is always fine.
- Recorded runs (`runs/<date-time>/`: frames, logs) show the inside of people's homes: **never commit them, never
  upload them, never copy them into the repository**. Tests use synthetic data only (`tests/sim2d.py`, the fake
  robot in `navette.testing`, `tests/golden/`).
- Do not weaken a safety layer to make something drive: the collision monitor, the restart gate, the watchdogs
  (stale perception, phone-page dead-man, base faults) and the base's own dead-man stay on.

## Working on the code

```bash
uv sync                                   # builds the extension (rebuilds when Rust sources change)
uv run pytest -q tests                    # ~1.5 min, no hardware; must stay green
cargo test                                # core unit tests
cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check
uv tool run --with ziglang maturin build --release --target aarch64-unknown-linux-gnu --zig \
    -m crates/navette-py/Cargo.toml       # the robot's wheel (see docs/INSTALL.md)
```

- `tests/golden/*.npz` are reference outputs of the core on synthetic inputs, compared exactly: do not
  regenerate them. A behaviour change goes behind a parameter whose core default keeps the old behaviour (golden
  tests unchanged); a robot config (`robots/*.toml`) turns it on, with a comment giving the measured reason, and
  `docs/PLAN.md` lists it.
- Robot specifics live in an adapter (`python/navette/robots/<robot>/`) behind the contract in
  `python/navette/interfaces.py`; a new robot is an adapter plus a config, checked with
  `navette.testing.conformance`.
- Names follow Nav2 where it has an equivalent (scan, occupancy grid, collision monitor, velocity smoother,
  controller, navigator).
- Comment the why of non-obvious code (frames, units, measured facts). Keep changes minimal.
- Frames: x forward, y left, theta counter-clockwise; metres, seconds, radians; `time.monotonic` clock.
