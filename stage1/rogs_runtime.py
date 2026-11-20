#!/usr/bin/env python3
"""Load the patched RoGS RGB/label rasterizers in the required order."""
import sys
from pathlib import Path

BRED_ROOT = Path(__file__).resolve().parents[1]
VENDOR = BRED_ROOT / "vendor"
ROGS = VENDOR / "RoGS"
LABEL_FIX = VENDOR / "diff_gs_label"
FIX = VENDOR / "ortho_gaussian_rasterization"
for path in (ROGS, LABEL_FIX, FIX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Cache the patched modules before rogs_adapter adjusts sys.path for RoGS.
import diff_gaussian_rasterization  # noqa: F401,E402
import diff_gs_label  # noqa: F401,E402
import diff_gaussian_rasterization.scene.cameras  # noqa: F401,E402
