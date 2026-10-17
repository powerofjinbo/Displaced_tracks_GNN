"""Host tables of the overlay sets (docs/preregistration_pileup.md sections 2, 3 and 9.4). H-val, closure and pilot hosts
are tp_v1 val events in derived/tp_v1. H-test hosts are training_part03 events in derived/tp03_v1, which H-T1 writes;
the H-test overlay reads its hosts from there, named explicitly (section 9.4, per-set checks). `host_root` resolves and
checks the root a driver is given: for htest it must be named and must be derived/tp03_v1; for the other sets it
defaults to derived/tp_v1 and must be that root. A dry root (tests, scratch runs) is accepted with dry=True for the
pilot set only (k = 9; as hnl.overlay.truth_checks.set_host_root), and only when its directory name is tp_v1: H-val,
closure and H-test always name their production host root.
"""

from __future__ import annotations

import os

from hnl.trackml.split import EXT

DERIVED_TP = os.path.join(EXT, "derived", "tp_v1")          # = hnl.trackml.throughput.DERIVED_TP
DERIVED_TP03 = os.path.join(EXT, "derived", "tp03_v1")
SET_HOST_ROOT = {"hval": DERIVED_TP, "closure": DERIVED_TP, "pilot": DERIVED_TP, "htest": DERIVED_TP03}
DRY_SETS = ("pilot",)                                        # = hnl.overlay.truth_checks.DRY_HOST_ROOT_SETS


def host_root(set_name: str, root: str | None = None, dry: bool = False) -> str:
    """The host root of a set (module docstring); raises for a root that is not the set's."""
    if set_name not in SET_HOST_ROOT:
        raise KeyError(f"no host root for set {set_name!r}")
    want = SET_HOST_ROOT[set_name]
    if root is None:
        if set_name == "htest":
            raise ValueError(f"the H-test overlay names its host root explicitly ({want})")
        return want
    if dry:
        if set_name not in DRY_SETS:
            raise ValueError(f"a dry host root is for the pilot set only, not {set_name} (it reads {want})")
        if os.path.basename(os.path.normpath(root)) != os.path.basename(want):
            raise ValueError(f"dry host root {root} of set {set_name} must be a directory named "
                             f"{os.path.basename(want)}")
        return os.path.abspath(root)
    if os.path.realpath(root) != os.path.realpath(want):
        raise ValueError(f"set {set_name} reads its hosts from {want}, not {root}")
    return want


def same_root(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and os.path.realpath(a) == os.path.realpath(b)
