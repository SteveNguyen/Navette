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

    uv run python benches/bench_scan.py

## Licence

Apache-2.0, see [LICENSE](LICENSE).
