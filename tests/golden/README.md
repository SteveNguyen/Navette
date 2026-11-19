# Golden files

Reference outputs of the Python prototype on seeded synthetic inputs, one set of files per module
(`<module>_*.npz`). Each file holds the inputs, the parameters and the prototype's outputs; the golden tests
(`tests/test_<module>_golden.py`) replay the inputs through navette and compare. Synthetic data only.

Tolerances:

| Values | Tolerance |
|--------|-----------|
| booleans, indices, sectors, states, inf / nan pattern | exact |
| float64 | 1e-6 |
| values that go through float32 | 1e-4 at most; scan ranges: 1e-6 (bit-identical on x86-64 / glibc) |

| Files | Content |
|-------|---------|
| `scan_real_lens.npz` | FreeSpace: Reachy Mini lens, 320x192 masks, default parameters, 36 cases |
| `scan_pinhole_s{1,3,7}.npz` | FreeSpace: ideal lens, small masks, sector 1 / 3 / 7 deg and other parameters, 15 cases each |
