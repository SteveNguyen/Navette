"""Navette: monocular floor-mask navigation, Rust core with Python bindings.

Everything the Rust core exports (navette._core) is re-exported here: a class bound in
crates/navette-py appears as navette.<Class> without editing this file.
"""

from navette._core import *  # noqa: F401,F403
