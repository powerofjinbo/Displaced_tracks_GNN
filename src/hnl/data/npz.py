"""Reader for the Exa.TrkX HNL / ttbar npz events (Zenodo 10.5281/zenodo.6812533, X. Ju, v2.0).

Each event is an uncompressed ``np.savez`` file ``<sample>/<i>.npz`` with keys (brief §2.3):

    x                     float32 [N, 3]  hit position in cylindrical coordinates (r [mm], phi [rad], z [mm])
    hid                   int64   [N]     a permutation of 0..N-1 (hit index in geometry-id order, not row order)
    pid                   int64   [N]     ACTS barcode of the particle that made the hit (no noise hits)
    layers                int64   [N]     ACTS geometry id
    layerless_true_edges  int64   [2, E]  true edges as ROW indices (not hid values)
    particles             float64 [n, 16] particle table, columns PARTICLE_COLUMNS

Column 12 (``parent_pid``) is the parent's signed PDG code, not a particle id. It is 0 for every particle in the
15 GeV / 100 mm sample (unlabeled). vx, vy, vz, vt are the particle's END state, not its production point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DATA_ROOT = os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt")
UNPACKED = os.path.join(DATA_ROOT, "unpacked")

PARTICLE_COLUMNS = (
    "particle_id", "particle_type", "process", "vx", "vy", "vz", "vt", "px", "py", "pz",
    "m", "q", "parent_pid", "pt", "radius", "eta",
)
COL = {name: i for i, name in enumerate(PARTICLE_COLUMNS)}
NPZ_KEYS = {
    "x": "float32", "hid": "int64", "pid": "int64", "layers": "int64",
    "layerless_true_edges": "int64", "particles": "float64",
}


@dataclass(frozen=True)
class Sample:
    key: str
    dirname: str
    n_events: int
    kind: str                     # "ttbar" or "hnl"
    labeled: bool                 # particles[:, 12] carries the parent PDG code
    hnl_mass_gev: float | None = None
    hnl_ctau_mm: float | None = None
    root: str | None = None       # data root of the sample when it is not UNPACKED (the generated samples)


SAMPLES: dict[str, Sample] = {
    s.key: s
    for s in (
        Sample("ttbar", "ttbar_nopileup_10K", 10000, "ttbar", True),
        Sample("hnl10_100", "WmuHNL10GeV100mm_nopileup_10K", 10000, "hnl", True, 10.0, 100.0),
        Sample("hnl15_100", "WmuHNL15GeV100mm_nopileup_5K", 5000, "hnl", False, 15.0, 100.0),
        Sample("hnl15_200", "WmuHNL15GeV200mm_nopileup_10K", 10000, "hnl", True, 15.0, 200.0),
    )
}

# The samples produced in this project with ACTS v15 (docs/new_samples_spec.md; evaluated under preregistration
# addendum G, docs/preregistration_new_sample.md). Registered only when HNL_GENERATED=1, so that nothing that iterates
# over SAMPLES changes otherwise. parent_pid is 0 everywhere (unlabeled); truth lives in the samples' sidecars.
GENERATED_ROOT = os.path.join(DATA_ROOT, "generated")
GENERATED: dict[str, Sample] = {
    s.key: s
    for s in (
        Sample("generic_iso", "generic_iso_v1", 8000, "generic", False, root=GENERATED_ROOT),
        Sample("generic_dense", "generic_dense_v1", 8000, "generic", False, root=GENERATED_ROOT),
        Sample("hss15_100", "hss15_100_v1", 1500, "generic", False, root=GENERATED_ROOT),
        Sample("hss40_300", "hss40_300_v1", 1500, "generic", False, root=GENERATED_ROOT),
    )
}
if os.environ.get("HNL_GENERATED") == "1":
    SAMPLES.update(GENERATED)


def event_path(sample: str, index: int, root: str = UNPACKED) -> str:
    s = SAMPLES[sample]
    if not 0 <= index < s.n_events:
        raise IndexError(f"{sample}: event {index} outside 0..{s.n_events - 1}")
    return os.path.join(s.root if s.root is not None else root, s.dirname, f"{index}.npz")


def load_event(sample: str, index: int, root: str = UNPACKED, check: bool = True) -> dict[str, np.ndarray]:
    """Load one event as a dict of arrays (copies; the file is closed on return)."""
    path = event_path(sample, index, root)
    with np.load(path, allow_pickle=False) as f:
        ev = {k: f[k] for k in f.files}
    if check:
        got = {k: str(v.dtype) for k, v in ev.items()}
        if got != NPZ_KEYS:
            raise ValueError(f"{path}: unexpected keys/dtypes {got}")
    return ev


def particle_ids(particles: np.ndarray) -> np.ndarray:
    """Particle ids as int64. Stored as float64; exact because every id is < 2**53 (checked here)."""
    col = particles[:, COL["particle_id"]]
    if np.any(np.abs(col) >= 2.0**53):
        raise ValueError("particle_id not exactly representable in float64")
    ids = col.astype(np.int64)
    if not np.array_equal(ids.astype(np.float64), col):
        raise ValueError("non-integer particle_id")
    return ids
