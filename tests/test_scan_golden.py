"""navette.FreeSpace against the Python prototype's outputs on the same inputs (tests/golden/scan_*.npz,
see tests/golden/README.md). No OpenCV needed: the rays are in the files."""

import numpy as np
import pytest

from conftest import golden_files
from navette import FreeSpace

# Ranges go through float32 atan2 / hypot, which need not be bit-identical between maths libraries (numpy's
# SIMD loops, glibc on x86-64 and aarch64). On the machine that wrote the goldens the match is exact (0.0).
RANGE_ATOL = 1e-6

FILES = golden_files("scan")


def load(path):
    g = np.load(path)
    fs = FreeSpace(g["rays"], g["rays_low"], mask_shape=tuple(g["mask_shape"]), sector_deg=float(g["sector_deg"]),
                   max_range=float(g["max_range"]), min_blob_px=int(g["min_blob_px"]), near_tol=float(g["near_tol"]))
    return g, fs


def assert_ranges(got, want, what):
    """Same inf / nan pattern, finite values within RANGE_ATOL."""
    np.testing.assert_allclose(got, want, rtol=0, atol=RANGE_ATOL, equal_nan=True, err_msg=str(what))


def test_golden_files_exist():
    assert len(FILES) >= 4


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_sectors(path):
    g, fs = load(path)
    assert fs.n == len(g["bearing_deg"])
    assert np.array_equal(fs.bearing_deg, g["bearing_deg"])
    assert fs.shape == tuple(g["mask_shape"])


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_obstacles(path):
    g, fs = load(path)
    for name, floor, want in zip(g["names"], g["floor_mask"], g["obst"]):
        got = fs.obstacles(floor)
        assert got.dtype == bool and np.array_equal(got, want), name


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_project(path):
    g, fs = load(path)
    r, sec = fs.project(g["proj_rays_g"], float(g["proj_h"]))
    assert np.array_equal(sec, g["proj_sec"])
    assert_ranges(r, g["proj_r"], "project")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_scan(path):
    g, fs = load(path)
    for i, name in enumerate(g["names"]):
        sc = fs.scan(g["floor_mask"][i], g["up_c"][i], float(g["h"][i]))
        assert np.array_equal(sc.bearing_deg, g["bearing_deg"]), name
        assert np.array_equal(sc.seen, g["seen"][i]), name
        assert np.array_equal(sc.near, g["near"][i]), name
        for field in ("obstacle", "floor", "closest"):
            assert_ranges(getattr(sc, field), g[field][i], (name, field))
        assert_ranges(sc.free(), g["free"][i], (name, "free"))
