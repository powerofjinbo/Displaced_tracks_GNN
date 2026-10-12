"""ACTS geometry identifiers and particle barcodes, and the detector regions of the generic detector.

Geometry id bit layout: the pre-"extra" ACTS ``GeometryID`` layout: volume [56, 64), boundary [48, 56),
layer [36, 48), approach [28, 36), sensitive [0, 28). Checked on the data (2026-10-02): boundary and approach are 0
for every hit and bits [0, 28) take values 1..3108 in ttbar event 0, using bits below 8 as well, so the newer layout
(sensitive [8, 28) + extra [0, 8)) does not apply. The module is (volume, layer, sensitive) = the full id.

Barcode bit layout (ACTS ``Barcode`` as used in these files, brief §2.3): vertex primary [52, 64), vertex secondary
[40, 52), particle [24, 40), generation [16, 24), sub-particle [0, 16).
"""

from __future__ import annotations

import numpy as np

PIXEL_VOLUMES = (7, 8, 9)
SHORT_STRIP_VOLUMES = (12, 13, 14)
LONG_STRIP_VOLUMES = (16, 17, 18)
BARREL_VOLUMES = (8, 13, 17)         # the middle volume of each subsystem (endcaps -z / +z on either side)
# Median radii of the 10 barrel layers (brief §2.3; pixel 4, short strip 4, long strip 2)
BARREL_RADII_MM = (32.1, 72.0, 116.0, 172.0, 260.2, 360.0, 500.2, 660.1, 820.1, 1020.1)


def decode_geometry_id(gid: np.ndarray) -> dict[str, np.ndarray]:
    gid = np.asarray(gid, dtype=np.int64).view(np.uint64)
    return {
        "volume": (gid >> np.uint64(56)).astype(np.int64),
        "boundary": ((gid >> np.uint64(48)) & np.uint64(0xFF)).astype(np.int64),
        "layer": ((gid >> np.uint64(36)) & np.uint64(0xFFF)).astype(np.int64),
        "approach": ((gid >> np.uint64(28)) & np.uint64(0xFF)).astype(np.int64),
        "sensitive": (gid & np.uint64(0x0FFFFFFF)).astype(np.int64),
    }


def decode_barcode(pid: np.ndarray) -> dict[str, np.ndarray]:
    pid = np.asarray(pid, dtype=np.int64).view(np.uint64)
    return {
        "vprim": ((pid >> np.uint64(52)) & np.uint64(0xFFF)).astype(np.int64),
        "vsec": ((pid >> np.uint64(40)) & np.uint64(0xFFF)).astype(np.int64),
        "particle": ((pid >> np.uint64(24)) & np.uint64(0xFFFF)).astype(np.int64),
        "gen": ((pid >> np.uint64(16)) & np.uint64(0xFF)).astype(np.int64),
        "sub": (pid & np.uint64(0xFFFF)).astype(np.int64),
    }


def is_pixel(volume: np.ndarray) -> np.ndarray:
    return np.isin(volume, PIXEL_VOLUMES)


def is_barrel(volume: np.ndarray) -> np.ndarray:
    return np.isin(volume, BARREL_VOLUMES)


def cylindrical_to_cartesian(x_rphiz: np.ndarray) -> np.ndarray:
    """(r, phi, z) float32 rows -> (x, y, z) float64 rows."""
    r = x_rphiz[:, 0].astype(np.float64)
    phi = x_rphiz[:, 1].astype(np.float64)
    return np.stack([r * np.cos(phi), r * np.sin(phi), x_rphiz[:, 2].astype(np.float64)], axis=1)
