"""Truth information per (acorn event id, particle id) for evaluation tables (docs/definitions.md).

acorn event ids are integers code * 1e8 + index (hnl.data.acorn_store). Columns: event_id, sample, event, particle_id,
category (hnl / w / other), is_target, pdg, q, pt, eta, n_hits, n_pixel_hits, r_first, d0 (truth-hit fit, mm), rv
(two-track vertex radius of the HNL-daughter pair, mm; NaN if none), d0_bin, rv_bin, and for HNL daughters
pair_angle (3D opening angle between the two daughters' table momenta, rad; NaN unless the event has exactly two
HNL daughters, both with non-zero table momentum).
"""

from __future__ import annotations

import functools
import os

import numpy as np
import pandas as pd

from hnl.data.acorn_store import SAMPLE_CODE
from hnl.truth.report import D0_BINS, RV_BINS

CODE_SAMPLE = {v: k for k, v in SAMPLE_CODE.items()}
DERIVED = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived")
COLUMNS = ["event", "particle_id", "category", "is_target", "pdg", "q", "pt", "eta", "n_hits", "n_pixel_hits",
           "r_first", "px", "py", "pz"]


def split_event_id(event_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(event_id, dtype=np.int64)
    return e // 100_000_000, e % 100_000_000


def particle_table(event_ids) -> pd.DataFrame:
    from hnl.data.convert import read_table
    from hnl.graphs.window_study import truth_particles
    codes, idx = split_event_id(np.unique(np.asarray(event_ids, dtype=np.int64)))
    out = []
    for c in np.unique(codes):
        sample = CODE_SAMPLE[int(c)]
        ev = idx[codes == c]
        p = read_table(os.path.join(DERIVED, "events", "v1"), "particles", sample, events=ev, columns=COLUMNS)
        t = truth_particles(sample)
        p = p.merge(t[t.event.isin(ev)], on=["event", "particle_id"], how="left")
        p["sample"] = sample
        p["event_id"] = np.int64(c) * 100_000_000 + p.event.to_numpy(np.int64)
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    p["is_target"] = p.is_target.astype(bool)
    p["pair_angle"] = np.nan
    h = p[p.category == "hnl"]
    two = h.groupby("event_id").particle_id.transform("size") == 2
    h = h[two]
    if len(h):
        g = h.sort_values(["event_id", "particle_id"])
        a, b = g.iloc[0::2], g.iloc[1::2]
        va = a[["px", "py", "pz"]].to_numpy(float)
        vb = b[["px", "py", "pz"]].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):     # a partner with zero table momentum -> NaN
            cosang = (va * vb).sum(1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1))
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        p.loc[a.index, "pair_angle"] = ang
        p.loc[b.index, "pair_angle"] = ang
    p["d0_bin"] = pd.cut(p.d0.abs(), D0_BINS, right=False)
    p["rv_bin"] = pd.cut(p.rv, RV_BINS, right=False)
    return p


@functools.lru_cache(maxsize=4)
def _particle_table_cached(events: tuple) -> pd.DataFrame:
    return particle_table(np.array(events, dtype=np.int64))


def particle_table_cached(event_ids) -> pd.DataFrame:
    """particle_table with a small in-process cache for read-out tools that evaluate several outputs of the same val
    events (the returned frame is shared: do not modify it in place)."""
    return _particle_table_cached(tuple(np.unique(np.asarray(event_ids, dtype=np.int64)).tolist()))
