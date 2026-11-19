"""FreeSpace.scan timing on the cases of tests/golden/scan_real_lens.npz (320x192 masks).

    uv run python benches/bench_scan.py
"""

import time
from pathlib import Path

import numpy as np

from navette import FreeSpace

REPS = 30  # scans per case

g = np.load(Path(__file__).resolve().parents[1] / "tests/golden/scan_real_lens.npz")
fs = FreeSpace(g["rays"], g["rays_low"])

# the straight-down case is left out: nothing is projected (NaN frame)
cases = [(m, up, float(h)) for n, m, up, h in zip(g["names"], g["floor_mask"], g["up_c"], g["h"])
         if "straight_down" not in n]
per_case = []
for m, up, h in cases:
    fs.scan(m, up, h)  # warm up
    t = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fs.scan(m, up, h)
        t.append(time.perf_counter() - t0)
    per_case.append(np.median(t) * 1e3)
per_case = np.array(per_case)
print(f"navette: FreeSpace.scan over {len(cases)} cases x {REPS} reps, ms per scan (per-case medians): "
      f"median {np.median(per_case):.3f}, min {per_case.min():.3f}, max {per_case.max():.3f}")
