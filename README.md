# Navette

Monocular floor-mask navigation: a Rust core with Python bindings, light enough for a Raspberry Pi CM4.
Plan, layout and decisions: [docs/PLAN.md](docs/PLAN.md).

- `crates/navette-core`: pure Rust, no dependency. Everything between the floor mask and the base command.
- `crates/navette-py`: PyO3 bindings (abi3 wheel), Python module `navette._core`, re-exported by `navette`.
- `python/navette`: Python package. `navette.rays` computes the viewing rays of an OpenCV-calibrated camera
  (optional dependency `navette[opencv]`); the core itself takes rays, not a lens model.

```python
from navette import FreeSpace
from navette.rays import mask_rays

fs = FreeSpace(mask_rays("calib.json"), mask_rays("calib.json", dv=0.5))  # once
scan = fs.scan(floor_mask, up_c, h)  # bool (192, 320) mask, 'up' in the camera frame, camera height (m)
scan.free()  # usable free distance per bearing
```

## Build

Needs a Rust toolchain and [uv](https://docs.astral.sh/uv/). From the repository root:

    uv sync                # creates .venv, builds the extension (release) and installs it, editable

`uv sync` / `uv run` rebuild the extension when a Rust source changes.

## Test

    cargo test             # core unit tests
    uv run pytest          # unit tests + golden tests (tests/golden, see its README)
    cargo clippy --workspace --all-targets
    cargo fmt --all

## Bench

| Script | Times |
|--------|-------|
| `uv run python benches/bench_scan.py` | `FreeSpace.scan` on 320x192 masks |
| `uv run python benches/bench_occupancy_grid.py` | `OccupancyGrid`: blind_known, update, obstacle_points, replay over a drive through a room |
| `uv run python benches/bench_collision_monitor.py` | `safe_command` (passing, slowed, blocked requests) and a controller's arc scoring |
| `uv run python benches/bench_navigator.py` | `Navigator.step` over recorded closed-loop sequences, steps with a plan apart from the others; the frontier planner's parts, on a room and on a 20 x 20 m hall (its worst case) |

## Licence

Apache-2.0, see [LICENSE](LICENSE).
