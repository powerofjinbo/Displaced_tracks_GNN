"""Track building on stitched TrackML scores (docs/trackml_plan.md sections 5 and 6.7).

Inputs per event: the compact tables (hits; layer id = volume * 1000 + layer; module azimuth phi_c from detectors.csv,
TrackML numbering, for the learned merging) and the scored full-event edges of hnl.trackml.infer (a < b, score).
Builders: the HNL families and grids (hnl.eval.trackbuild_val.configs: cc, walkmergeX, mwslayer, mwslayermergeX,
mws2passmergeX, mwslmerge; plan section 5 WP-A), with the algorithms imported unchanged (hnl.trackbuild.builders,
hnl.trackbuild.mutex_ws, hnl.trackbuild.learned_merge, hnl.trackbuild.merge_fragments._merge through
hnl.trackml.seqmerge). A TrackML event has about 10^4 fragments, and the O(T^2) Python loops that look for merge
candidates would take hours, so the candidate searches are re-implemented by vectorised blocks with the same
floating-point operations in the same order (exact; the tests compare them with the originals): merge_fragments_fast
(hnl.trackbuild.merge_fragments.merge_fragments, both modes) and lm_candidates (hnl.trackbuild.learned_merge.candidates).
The helix merging then computes every candidate's fit values exactly as _merge's batch of all candidates would (the
sequential pass only for the pairs its cuts could pass), runs _merge's acceptance loop on them, and calls the unchanged
_merge on the resulting merges (hnl.trackml.seqmerge; same labels; about 3e6 refits per event otherwise, gate G3).
The hand-cut merging families carry a cut-set grid parameter (H: HNL cuts; C: per |eta| bin, builders.merge_cuts).
Matching: every metric of hnl.trackml.evaluate.

    python -m hnl.trackml.trackbuild grid --scores DIR --events val --out DIR [--builders ...] [--merge-model M]
                                          [--condition full] [--allow-test] [--block K --block-size N]
    python -m hnl.trackml.trackbuild grid-merge --blocks DIR [DIR ...] --events val --out DIR
        (pools the per-event counts of event blocks of one grid into the files of an unsplit grid)
    python -m hnl.trackml.trackbuild lmerge-data --scores DIR --events merge_train --out PAIRS.npz
        (then `python -m hnl.trackbuild.learned_merge train --data PAIRS.npz --out MODEL.joblib`, unchanged)
    python -m hnl.trackml.trackbuild select --grid DIR [--grid-nonoise DIR] [--grid-1gev DIR] --out SELECTION.json
    python -m hnl.trackml.trackbuild timing --scores DIR --events dev --out DIR     (gate G3: minutes per event and
                                          configuration on 16 CPUs, exactness of the fast merging on one event)
    python -m hnl.trackml.trackbuild merge-check --scores DIR --events 1000 --out DIR   (fast vs original merging)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.trackbuild import builders as TB
from hnl.trackbuild import learned_merge as LM
from hnl.trackbuild.merge_fragments import _volume_from_radius
from hnl.trackml import config as CFG
from hnl.trackml import events as E
from hnl.trackml.seqmerge import BinnedCuts, merge_sequential

DEFAULT_FAMILIES = ("cc", "walkmergeX", "mwslayer", "mwslayermergeX", "mws2passmergeX", "mwslmerge")
SUPPORTED = ("cc", "walk", "walkmerge", "mwslayer", "mwslayermerge", "mws2pass", "mws2passmerge", "mwslmerge")

# Cut sets of the hand-cut merging families (DECISIONS 2026-10-05, merge-cut calibration): a trailing grid parameter
# of their configurations, 0 = H (the HNL cuts in every |eta| bin), 1 = C (per |eta| bin of the union,
# builders.merge_cuts). Configurations without it are H.
MERGE_FAMILIES = {"walkmerge": 4, "mwslayermerge": 2, "mws2passmerge": 3}     # family -> index of the cut-set flag
CUT_SET_FLAGS = {"H": 0.0, "C": 1.0}
_CUTS = {}
# processes for the exact pre-fit of the sequential pass (hnl.trackml.seqmerge.prefit; same values in any process):
# 1 in the grid, whose workers already fill the CPUs; gate_g3.sbatch gives each build its CPUs (plan G3: 16 CPUs)
PREFIT_PROCS = int(os.environ.get("HNL_TML_PREFIT_PROCS", "1"))


def cut_set(name: str) -> BinnedCuts | None:
    """None for H (scalar HNL cuts, the configuration's own max_rms); the BinnedCuts of builders.merge_cuts for C."""
    if name == "H":
        return None
    if name not in _CUTS:
        mc = CFG.load()["builders"]["merge_cuts"]
        _CUTS[name] = BinnedCuts.from_tsv(CFG.repo_path(mc["table"]), mc["max_rms"], mc["max_zrms"], mc["max_out"],
                                          name=name)
    return _CUTS[name]


def cut_set_of(cfg: tuple) -> str:
    """Cut set of a configuration tuple (H unless its merging family carries the C flag)."""
    pos = MERGE_FAMILIES.get(cfg[0])
    if pos is None or len(cfg) <= pos:
        return "H"
    flag = float(cfg[pos])
    for k, v in CUT_SET_FLAGS.items():
        if flag == v:
            return k
    raise ValueError(f"unknown cut-set flag {flag} in {cfg}")


# ------------------------------------------------------------------------------------------------ fast exact merging
def _tracks_of(labels: np.ndarray, trk: np.ndarray) -> dict:
    """{track: its hit indices in increasing order} (as np.flatnonzero(labels == t)), by one sort."""
    ok = np.flatnonzero(labels >= 0)
    o = ok[np.argsort(labels[ok], kind="stable")]
    lab = labels[o]
    starts = np.searchsorted(lab, trk, side="left")
    ends = np.searchsorted(lab, trk, side="right")
    return {t: o[s:e] for t, s, e in zip(trk.tolist(), starts, ends)}


def _candidates_seq(trk, hits, R, phi, max_dphi, max_gap_r, block=1024):
    """merge_fragments' sequential candidates: (a, b), a != b, in the loops' order (a, then b, in trk order)."""
    inner = np.empty(len(trk), dtype=np.int64)
    outer = np.empty(len(trk), dtype=np.int64)
    for i, t in enumerate(trk.tolist()):
        idx = hits[t]
        o = idx[np.argsort(R[idx])]
        inner[i], outer[i] = o[0], o[-1]
    R_in, R_out, phi_in, phi_out = R[inner], R[outer], phi[inner], phi[outer]
    out = []
    T = len(trk)
    for i0 in range(0, T, block):
        i1 = min(i0 + block, T)
        dR = R_in[None, :] - R_out[i0:i1, None]
        ok = ~((dR < -20.0) | (dR > max_gap_r))
        dphi = np.abs(np.remainder(phi_in[None, :] - phi_out[i0:i1, None] + np.pi, 2 * np.pi) - np.pi)
        ok &= ~(dphi > max_dphi)
        ok[np.arange(i1 - i0), np.arange(i0, i1)] = False
        ia, ib = np.nonzero(ok)
        out += list(zip(trk[ia + i0], trk[ib]))
    return out


_ANY = {}


def _any_block(i0: int):
    """One block of rows of the any-order candidate test (a worker inherits the fragment fits by fork)."""
    T, kp, kz, Rmed, rmed, max_dphi_dir, max_dcot, block = _ANY["args"]
    i1 = min(i0 + block, T)
    Rm = 0.5 * (Rmed[i0:i1, None] + Rmed[None, :])
    rm = 0.5 * (rmed[i0:i1, None] + rmed[None, :])
    fa = kp[i0:i1, 0, None] * Rm + kp[i0:i1, 1, None]
    fb = kp[None, :, 0] * Rm + kp[None, :, 1]
    dphi = np.abs(np.remainder(fa - fb + np.pi, 2 * np.pi) - np.pi)
    za = kz[i0:i1, 0, None] * rm + kz[i0:i1, 1, None]
    zb = kz[None, :, 0] * rm + kz[None, :, 1]
    dz = np.abs(za - zb) / np.maximum(rm, 1.0)
    ok = ~(dphi > max_dphi_dir) & ~(dz > max_dcot)
    ok &= np.arange(T)[None, :] > np.arange(i0, i1)[:, None]
    ia, ib = np.nonzero(ok)
    return ia + i0, ib


def _candidates_any_fast(trk, hits, r, phi, z, R, max_dphi_dir, max_dcot, block=1024, n_proc: int = 1):
    """merge_fragments._candidates_any: per fragment the same line fits, then every pair i < j tested by blocks with
    the same operations (np.polyval of a degree-1 fit = p0 * x + p1). n_proc > 1: the blocks in parallel processes
    (each block's arithmetic is the same; the pairs come back in the same order)."""
    T = len(trk)
    kp = np.empty((T, 2))
    kz = np.empty((T, 2))
    Rmed = np.empty(T)
    rmed = np.empty(T)
    for i, t in enumerate(trk.tolist()):
        idx = hits[t]
        Rt, ph, rt, zt = R[idx], np.unwrap(phi[idx][np.argsort(R[idx])]), r[idx], z[idx]
        Rs = np.sort(Rt)
        if len(idx) >= 2 and Rs[-1] - Rs[0] > 1e-6:
            k_phi = np.polyfit(Rs, ph, 1)
            k_z = np.polyfit(rt, zt, 1)
        else:
            k_phi = np.array([0.0, ph[0]])
            k_z = np.array([zt[0] / max(rt[0], 1e-6), 0.0])
        kp[i], kz[i], Rmed[i], rmed[i] = k_phi, k_z, np.median(Rt), np.median(rt)
    starts = list(range(0, T, block))
    _ANY["args"] = (T, kp, kz, Rmed, rmed, max_dphi_dir, max_dcot, block)
    try:
        if n_proc > 1 and len(starts) > 1:
            import concurrent.futures as cf
            import multiprocessing as mp
            with cf.ProcessPoolExecutor(max_workers=n_proc, mp_context=mp.get_context("fork")) as ex:
                parts = list(ex.map(_any_block, starts))
        else:
            parts = [_any_block(i0) for i0 in starts]
    finally:
        _ANY.clear()
    out = []
    for ia, ib in parts:
        out += list(zip(trk[ia], trk[ib]))
    return out


def any_cot(r, phi, z, labels, trk, volume, cand, n_proc: int = 1, seed: int = 0) -> np.ndarray:
    """cot theta of every any-order candidate's union fit exactly as _merge's batch of all of them gives it (bit for
    bit), by hnl.trackml.seqmerge's size-class pre-fit with the batch's padded width and s-z iteration count (the
    union's |eta| bin of cut set C)."""
    from hnl.trackml import seqmerge as SM
    tab = SM.Table(r, phi, z, labels, trk, volume)
    a = np.searchsorted(trk, np.array([c[0] for c in cand]))
    b = np.searchsorted(trk, np.array([c[1] for c in cand]))
    sizes = tab.fn[a] + tab.fn[b]
    if (sizes < 3).any():
        raise RuntimeError("any_cot: candidates with fewer than 3 hits")
    h = int(np.flatnonzero(sizes == sizes.max())[0])
    W, _, _ = SM._witnesses(tab, a, b, np.zeros(len(a), dtype=bool), h, r, phi, z, volume, 4096, seed)
    return SM.prefit(tab, a, b, np.arange(len(a)), h, W, r, phi, z, volume, n_proc=n_proc)["cot"]


def merge_fragments_fast(r, phi, z, labels, volume=None, min_frag: int = 2, max_dphi: float = 0.3,
                         max_gap_r: float = 400.0, max_rms: float = 1.0, max_zrms: float = 5.0, max_out: int = 1,
                         mode: str = "sequential", max_dphi_dir: float = 0.05, max_dcot: float = 0.05,
                         min_span_mm: float = 50.0, cuts: BinnedCuts | None = None) -> np.ndarray:
    """hnl.trackbuild.merge_fragments.merge_fragments with the candidate search vectorised (same result). cuts (a
    BinnedCuts, cut set C): every candidate is held to the cuts of its union's |eta| bin (asinh |cot| of the union fit)
    instead of max_rms / max_zrms / max_out: the unchanged _merge gets them as arrays. Both passes go through
    hnl.trackml.seqmerge.merge_exact: the candidates' fit values bit for bit as in _merge's batch of all of them (exact
    pre-fit), _merge's acceptance loop for the merges, and the unchanged _merge on those merges + the pinning pairs.
    PREFIT_PROCS processes share the any-order candidate search and the pre-fits (same results)."""
    labels = np.asarray(labels).copy()
    r, phi, z = (np.asarray(x, dtype=np.float64) for x in (r, phi, z))
    if volume is None:
        volume = _volume_from_radius(r, z)
    R = np.hypot(r, z)
    ok = labels >= 0
    if not ok.any():
        return labels
    trk, cnt = np.unique(labels[ok], return_counts=True)
    trk = trk[cnt >= min_frag]
    if len(trk) < 2:
        return labels
    if mode != "any":
        # hnl.trackml.seqmerge (G3): exact pre-filter and pre-fit, same labels as the unfiltered pass
        return merge_sequential(r, phi, z, labels, volume, min_frag, max_dphi, max_gap_r, max_rms, max_zrms, max_out,
                                min_span_mm=min_span_mm, cuts=cuts, n_proc=PREFIT_PROCS)
    hits = _tracks_of(labels, trk)
    cand = _candidates_any_fast(trk, hits, r, phi, z, R, max_dphi_dir, max_dcot, n_proc=PREFIT_PROCS)
    if not cand:
        return labels
    # _merge(cand, ...) through hnl.trackml.seqmerge.merge_exact: every candidate's fit values bit for bit as in
    # _merge's batch (exact pre-fit), _merge's any-order acceptance loop for the merges, the unchanged _merge on them
    from hnl.trackml.seqmerge import merge_exact
    cuts = BinnedCuts.uniform(max_rms, max_zrms, max_out) if cuts is None else cuts
    a = np.searchsorted(trk, np.array([c[0] for c in cand]))
    b = np.searchsorted(trk, np.array([c[1] for c in cand]))
    out, _ = merge_exact(r, phi, z, labels, trk, volume, a, b, np.arange(len(a)), cuts, min_span_mm, False,
                         n_proc=PREFIT_PROCS)
    return out


def lm_candidates(labels, r, phi, z, block: int = 1024) -> np.ndarray:
    """hnl.trackbuild.learned_merge.candidates with the all-pairs window test by blocks (same result)."""
    labels = np.asarray(labels)
    R = np.hypot(r, z)
    ok = labels >= 0
    trk, inv, cnt = np.unique(labels[ok], return_inverse=True, return_counts=True)
    keep = cnt >= 2
    if keep.sum() < 2:
        return np.zeros((0, 2), dtype=np.int64)
    T = len(trk)
    idx = np.flatnonzero(ok)
    ph = phi[idx]
    c = np.bincount(inv, weights=np.cos(ph), minlength=T)
    s = np.bincount(inv, weights=np.sin(ph), minlength=T)
    mean_phi = np.arctan2(s, c)
    dph = np.remainder(ph - mean_phi[inv] + np.pi, 2 * np.pi) - np.pi
    Ri, ri, zi = R[idx], r[idx], z[idx]
    n = cnt.astype(np.float64)

    def line(xv, yv):
        sx, sy = np.bincount(inv, xv, T), np.bincount(inv, yv, T)
        sxx, sxy = np.bincount(inv, xv * xv, T), np.bincount(inv, xv * yv, T)
        det = n * sxx - sx * sx
        slope = np.where(np.abs(det) > 1e-9, (n * sxy - sx * sy) / np.where(np.abs(det) > 1e-9, det, 1), 0.0)
        icpt = (sy - slope * sx) / n
        return slope, icpt
    kp, bp = line(Ri, dph)
    kz, bz = line(ri, zi)
    Rm = np.bincount(inv, Ri, T) / n
    rm = np.bincount(inv, ri, T) / n
    sel = np.flatnonzero(keep)
    A, Bv, Q = [], [], []
    for i0 in range(0, len(sel), block):
        sa = sel[i0:i0 + block]
        a_, b_ = np.meshgrid(sa, sel, indexing="ij")
        up = a_ < b_
        a_, b_ = a_[up], b_[up]
        Rp, rp = 0.5 * (Rm[a_] + Rm[b_]), 0.5 * (rm[a_] + rm[b_])
        pa = mean_phi[a_] + kp[a_] * Rp + bp[a_]
        pb = mean_phi[b_] + kp[b_] * Rp + bp[b_]
        dphi = np.abs(np.remainder(pa - pb + np.pi, 2 * np.pi) - np.pi)
        dz = np.abs(kz[a_] * rp + bz[a_] - kz[b_] * rp - bz[b_]) / np.maximum(rp, 1.0)
        win = (dphi < LM.MAX_DPHI) & (dz < LM.MAX_DZ)
        A.append(a_[win])
        Bv.append(b_[win])
        Q.append((dphi + dz)[win])
    a_, b_, q = np.concatenate(A), np.concatenate(Bv), np.concatenate(Q)
    if len(a_) == 0:
        return LM.end_candidates(labels, r, phi, z)
    rank = np.zeros(len(q), dtype=bool)
    for end in (a_, b_):
        o = np.lexsort((q, end))
        first = np.r_[True, end[o][1:] != end[o][:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
        rank[o[(np.arange(len(o)) - start) < LM.MAX_PARTNERS]] = True
    a_, b_ = a_[rank], b_[rank]
    pairs = np.stack([trk[a_], trk[b_]], 1).astype(np.int64)
    return np.unique(np.concatenate([pairs, LM.end_candidates(labels, r, phi, z)]), axis=0)


# ------------------------------------------------------------------------------------------------ geometry and inputs
_DET = None


def module_phi_c(volume, layer, module) -> np.ndarray:
    """Azimuth of each hit's module centre from detectors.csv (TrackML numbering; replaces
    hnl.geometry.resolution.module_phi_c, whose envelopes use our ACTS endcap numbering)."""
    global _DET
    from hnl.trackml.convert import load_detectors, module_key
    if _DET is None:
        d = load_detectors()
        _DET = (d["key"], np.arctan2(d["c"][:, 1], d["c"][:, 0]))
    keys, phic = _DET
    k = module_key(volume, layer, module)
    pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
    if not (keys[pos] == k).all():
        raise KeyError(f"{int((keys[pos] != k).sum())} hits on modules missing from detectors.csv")
    return phic[pos]


def geometry(ev: E.Event) -> dict:
    vol = ev.volume.astype(np.int64)
    return {"volume": vol, "layer_id": ev.layer_id, "phi_c": module_phi_c(ev.volume, ev.layer, ev.module)}


def read_scores(directory: str, events, condition: str = "full") -> dict[int, tuple]:
    """{event: (a, b, score)} from the scored-edge parquet files of hnl.trackml.infer (KeyError if one is missing;
    ValueError if a file was scored in another condition)."""
    import pyarrow.parquet as pq
    want = {int(e) for e in events}
    out = {}
    for f in sorted(os.listdir(directory)):
        if not f.endswith(".parquet"):
            continue
        with open(os.path.join(directory, f + ".json")) as fh:
            side = json.load(fh)
        if side.get("condition", "full") != condition:
            raise ValueError(f"{f}: scored in condition {side.get('condition')}, not {condition}")
        have = {int(e) for e in side["events"]}
        sel = sorted(want & have)
        if not sel:
            continue
        t = pq.read_table(os.path.join(directory, f), filters=[("event", "in", sel)],
                          columns=["event", "a", "b", "score"]).to_pandas()
        for e, g in t.groupby("event"):
            out[int(e)] = (g.a.to_numpy(np.int64), g.b.to_numpy(np.int64), g.score.to_numpy(np.float64))
    missing = sorted(want - set(out))
    if missing:
        raise KeyError(f"no scores for events {missing[:10]}")
    return out


# ------------------------------------------------------------------------------------------------ builders
class EventState:
    """Per-event inputs and caches shared by the configurations of one event."""

    def __init__(self, ev: E.Event, a, b, sc, merge_model=None):
        self.ev, self.n = ev, ev.n
        self.a, self.b, self.sc = np.asarray(a), np.asarray(b), np.asarray(sc, dtype=np.float64)
        self.r, self.phi, self.z = (np.asarray(x, dtype=np.float64) for x in (ev.r, ev.phi, ev.z))
        self.R = np.hypot(self.r, self.z)
        self.L = ev.layer_id
        self.merge_model = merge_model
        self._lm = None
        self._geo = None

    @property
    def geo(self):
        if self._geo is None:
            self._geo = geometry(self.ev)
        return self._geo

    def merged(self, lab, max_rms=1.0, cuts: BinnedCuts | None = None):
        """Helix fragment merging (any order, then sequential): the HNL cuts with this max_rms, or a cut set."""
        lab = merge_fragments_fast(self.r, self.phi, self.z, lab, max_rms=max_rms, mode="any", cuts=cuts)
        return merge_fragments_fast(self.r, self.phi, self.z, lab, max_rms=max_rms, cuts=cuts)

    def lmerge_state(self):
        if self._lm is None:
            if self.merge_model is None:
                raise ValueError("mwslmerge needs --merge-model")
            first = LM.first_pass(self.n, self.a, self.b, self.sc, self.L)
            pairs = lm_candidates(first, self.r, self.phi, self.z)
            X = LM.features(first, pairs, self.r, self.phi, self.z, self.geo)
            p = self.merge_model.predict_proba(X)[:, 1] if len(pairs) else np.zeros(0)
            self._lm = (first, pairs, p)
        return self._lm


def build(cfg: tuple, st: EventState) -> np.ndarray:
    """Track labels of one configuration (hnl.eval.trackbuild_val semantics for the supported families)."""
    from hnl.trackbuild.mutex_ws import mws_layer, mws_two_pass
    kind = cfg[0]
    n, a, b, sc = st.n, st.a, st.b, st.sc
    if kind == "cc":
        return TB.cc(n, a, b, sc, cfg[1])
    if kind == "walk":
        return TB.walkthrough(n, a, b, sc, st.R, cfg[1], cfg[2])
    cuts = cut_set(cut_set_of(cfg)) if kind in MERGE_FAMILIES else None   # C: per |eta| bin; H: None
    if kind == "walkmerge":
        lab = TB.walkthrough(n, a, b, sc, st.R, cfg[1], cfg[2])
        return st.merged(lab, cfg[3], cuts)
    if kind in ("mwslayer", "mwslayermerge"):
        lab = mws_layer(n, a, b, sc, st.L, cfg[1])
        return st.merged(lab, cuts=cuts) if kind == "mwslayermerge" else lab
    if kind in ("mws2pass", "mws2passmerge"):
        lab = mws_two_pass(n, a, b, sc, st.L, cfg[1], cfg[2])
        return st.merged(lab, cuts=cuts) if kind == "mws2passmerge" else lab
    if kind == "mwslmerge":
        first, pairs, p = st.lmerge_state()
        return LM.apply(first, pairs, p, cfg[1], st.L)
    raise ValueError(f"builder family {kind!r} is not supported on TrackML ({SUPPORTED})")


def explicit_configs(spec: str) -> tuple[list[tuple], list]:
    """Configurations from JSON (a file or a string): [{"name", "family", "params": "p1,p2"}, ...] or [[family, p1, ...]]."""
    frozen = json.load(open(spec)) if os.path.exists(spec) else json.loads(spec)
    cfgs, names = [], []
    for d in frozen:
        if isinstance(d, dict):
            cfgs.append((d["family"], *[float(x) for x in str(d["params"]).split(",") if x != ""]))
            names.append(d.get("name"))
        else:
            cfgs.append((d[0], *[float(x) for x in d[1:]]))
            names.append(None)
    bad = sorted({c[0] for c in cfgs} - set(SUPPORTED))
    if bad:
        raise SystemExit(f"unsupported families {bad}")
    return cfgs, names


def grid_configs(builders, cut_sets=("H",)) -> list[tuple]:
    """The HNL grid of these families (hnl.eval.trackbuild_val.configs); every configuration of a hand-cut merging
    family once per cut set, with its flag appended (builders.merge_cut_sets: a grid parameter, DECISIONS 2026-10-05)."""
    from hnl.eval.trackbuild_val import configs
    cfgs = configs(list(builders))
    bad = sorted({c[0] for c in cfgs} - set(SUPPORTED))
    if bad:
        raise ValueError(f"unsupported families {bad}")
    out = []
    for c in cfgs:
        if c[0] in MERGE_FAMILIES:
            if len(c) != MERGE_FAMILIES[c[0]]:
                raise ValueError(f"unexpected parameters for {c[0]}: {c}")
            out += [(*c, CUT_SET_FLAGS[s]) for s in cut_sets]
        else:
            out.append(c)
    return out


# ------------------------------------------------------------------------------------------------ grid
def _grid_events(paths_args):
    from hnl.trackml.evaluate import event_counts
    events, scores_dir, cfgs, merge_model, condition = paths_args[:5]
    verbose = len(paths_args) > 5 and bool(paths_args[5])
    model = None
    if merge_model:
        import joblib
        model = joblib.load(merge_model)
    sc_all = read_scores(scores_dir, events, condition)
    counts, parts, timing = [], [], []
    for ev in E.iter_events(events):
        ev = E.with_condition(ev, condition)
        a, b, sc = sc_all[ev.event]
        st = EventState(ev, a, b, sc, model)
        hid = ev.full_rows + 1
        for i, cfg in enumerate(cfgs):
            t0 = time.time()
            lab = build(cfg, st)
            dt = time.time() - t0
            if verbose:
                print(json.dumps({"event": ev.event, "config": i, "family": cfg[0],
                                  "params": [float(x) for x in cfg[1:]], "seconds": round(dt, 2)}), flush=True)
            # no TrackML score in C-1gev (plan section 4.2 P0: its normalisation changes)
            c, rows = event_counts(ev.pid, lab, ev.particles, hid, None if condition == "1gev" else ev.weight)
            counts.append({"event": ev.event, "config": i, **c})
            rows.insert(0, "config", np.int16(i))
            rows.insert(0, "event", np.int32(ev.event))
            parts.append(rows)
            timing.append({"event": ev.event, "config": i, "family": cfg[0], "seconds": round(dt, 2)})
    return pd.DataFrame(counts), pd.concat(parts, ignore_index=True), pd.DataFrame(timing)


def cmd_grid(a, cfg):
    import concurrent.futures as cf
    names = None
    if a.configs:                         # explicit (frozen) configurations
        cfgs, names = explicit_configs(a.configs)
        builders = sorted({c[0] for c in cfgs})
    else:
        builders = a.builders.split(",") if a.builders else list(cfg["builders"]["families"])
        cfgs = grid_configs(builders, cfg["builders"].get("merge_cut_sets", ["H"]))
    if any(c[0] == "mwslmerge" for c in cfgs) and not a.merge_model:
        raise SystemExit("mwslmerge needs --merge-model")
    events = CFG.event_set(a.events, cfg, allow_test=a.allow_test)[:a.max_events]
    if a.block is not None:               # one block of the event list (grid-merge pools the blocks)
        events = events[a.block * a.block_size:(a.block + 1) * a.block_size]
        if not events:
            raise SystemExit(f"block {a.block} of size {a.block_size} is empty")
    os.makedirs(a.out, exist_ok=True)
    chunks = [(events[k:k + a.chunk], a.scores, cfgs, a.merge_model, a.condition)
              for k in range(0, len(events), a.chunk)]
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_grid_events, chunks))
    counts = pd.concat([r[0] for r in res], ignore_index=True)
    counts.to_parquet(os.path.join(a.out, "per_event.parquet"), index=False)
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(os.path.join(a.out, "per_target.parquet"),
                                                                 index=False)
    pd.concat([r[2] for r in res], ignore_index=True).to_parquet(os.path.join(a.out, "timing.parquet"), index=False)
    with open(os.path.join(a.out, "configs.json"), "w") as fh:
        json.dump([[x if isinstance(x, str) else float(x) for x in c] for c in cfgs], fh)
    if names:
        with open(os.path.join(a.out, "config_names.json"), "w") as fh:
            json.dump(names, fh)
    summ = summarize_grid(counts, cfgs)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with open(os.path.join(a.out, "grid.json"), "w") as fh:
        json.dump({"events": len(events), "condition": a.condition, "scores": a.scores, "merge_model": a.merge_model,
                   "builders": builders, "elapsed_s": round(time.time() - t0, 1), "block": a.block,
                   "block_size": a.block_size if a.block is not None else None,
                   "slurm_job": os.environ.get("SLURM_JOB_ID")}, fh, indent=1)
    print(summ.round(4).to_string(index=False))


def cmd_grid_merge(a, cfg):
    """The files of an unsplit grid from its event blocks: same configurations, every event of the set exactly once."""
    cfgs = None
    for d in a.blocks:
        with open(os.path.join(d, "configs.json")) as fh:
            c = json.load(fh)
        if cfgs is not None and c != cfgs:
            raise SystemExit(f"{d}: configurations differ from {a.blocks[0]}")
        cfgs = c
    meta = [json.load(open(os.path.join(d, "grid.json"))) for d in a.blocks]
    if len({(m["condition"], m["scores"], m["merge_model"]) for m in meta}) != 1:
        raise SystemExit("blocks differ in condition, scores or merge model")
    tabs = {n: pd.concat([pd.read_parquet(os.path.join(d, n + ".parquet")) for d in a.blocks], ignore_index=True)
            for n in ("per_event", "per_target", "timing")}
    counts = tabs["per_event"]
    want = sorted(int(e) for e in CFG.event_set(a.events, cfg, allow_test=a.allow_test)[:a.max_events])
    per = counts.groupby("config").event.agg(lambda e: sorted(int(x) for x in e))
    if len(per) != len(cfgs) or any(v != want for v in per):
        raise SystemExit(f"blocks do not hold every event of {a.events} exactly once for every configuration")
    os.makedirs(a.out, exist_ok=True)
    for n, t in tabs.items():
        t.to_parquet(os.path.join(a.out, n + ".parquet"), index=False)
    with open(os.path.join(a.out, "configs.json"), "w") as fh:
        json.dump(cfgs, fh)
    names = [json.load(open(p)) if os.path.exists(p) else None
             for p in (os.path.join(d, "config_names.json") for d in a.blocks)]
    if any(n != names[0] for n in names):
        raise SystemExit("blocks differ in configuration names")
    if names[0] is not None:
        with open(os.path.join(a.out, "config_names.json"), "w") as fh:
            json.dump(names[0], fh)
    summ = summarize_grid(counts, cfgs)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with open(os.path.join(a.out, "grid.json"), "w") as fh:
        json.dump({**{k: meta[0][k] for k in ("condition", "scores", "merge_model", "builders")}, "events": len(want),
                   "blocks": a.blocks, "elapsed_s": [m["elapsed_s"] for m in meta],
                   "slurm_job": [m["slurm_job"] for m in meta]}, fh, indent=1)
    print(summ.round(4).to_string(index=False))


def summarize_grid(counts: pd.DataFrame, cfgs) -> pd.DataFrame:
    from hnl.trackml.evaluate import pooled
    rows = []
    for i, g in counts.groupby("config"):
        c = cfgs[int(i)]
        rows.append({"config": int(i), "family": c[0], "params": ",".join(f"{x:g}" for x in c[1:]),
                     "n_events": len(g), "n_targets": int(g.n_target.sum()), **pooled(g)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ selection
def builder_rule(summ: pd.DataFrame, fr: float, eff: str = "ours_eff_target", fake: str = "ours_fake_rate",
                 n_col: str = "n_targets") -> tuple[pd.Series, bool]:
    """Plan section 5 WP-A with the one-standard-error tie (DECISIONS 2026-10-04 20:18 UTC): among configurations with
    fake <= fr, those within one binomial SE of the best efficiency are tied and the lowest fake rate wins (then the
    higher efficiency); if none reaches fr, the lowest-fake-rate configuration, flagged. Returns (row, reached)."""
    ok = summ[summ[fake] <= fr + 1e-12]
    if ok.empty:
        return summ.sort_values([fake, eff], ascending=[True, False]).iloc[0], False
    best = ok[eff].max()
    se = float(np.sqrt(best * (1 - best) / max(float(ok.loc[ok[eff].idxmax(), n_col]), 1.0)))
    tied = ok[ok[eff] >= best - se - 1e-12]
    return tied.sort_values([fake, eff], ascending=[True, False]).iloc[0], True


def select(full: pd.DataFrame, fr: float, nonoise: pd.DataFrame | None = None, onegev: pd.DataFrame | None = None,
           refs=None) -> dict:
    """WP-A (per family, then across families: R6 on the same numbers), WP-S (highest val TrackML score) and one WP-R
    per reference (plan section 5)."""
    from hnl.trackml.evaluate import REFERENCES
    refs = refs or REFERENCES
    out = {"fr_target": fr, "families": {}}
    fam_rows = []
    for fam, g in full.groupby("family"):
        row, reached = builder_rule(g, fr)
        out["families"][fam] = {"config": int(row.config), "params": row.params, "reached_fr": reached,
                                "eff_target": float(row.ours_eff_target), "fake_rate": float(row.ours_fake_rate)}
        fam_rows.append(row)
    fr_df = pd.DataFrame(fam_rows)
    row, reached = builder_rule(fr_df, fr)
    out["WP-A"] = {"config": int(row.config), "family": row.family, "params": row.params, "reached_fr": reached,
                   "flag": None if reached else "no configuration reaches FR_T: lowest-fake-rate fallback"}
    # sensitivity, reported only (plan section 5 FR_T): FR_T,CC = the val fake rate of CC at its highest-target-
    # efficiency cut, and WP-A's choice under it (same rule, per family then across families)
    cc = full[full.family == "cc"]
    if not cc.empty:
        fr_cc = float(cc.loc[cc.ours_eff_target.idxmax(), "ours_fake_rate"])
        rows_cc = [builder_rule(g, fr_cc)[0] for _, g in full.groupby("family")]
        row_cc, reached_cc = builder_rule(pd.DataFrame(rows_cc), fr_cc)
        out["WP-A-FRcc"] = {"fr_target_cc": fr_cc, "config": int(row_cc.config), "family": row_cc.family,
                            "params": row_cc.params, "reached_fr": reached_cc,
                            "same_as_WP-A": int(row_cc.config) == int(row.config)}
    if "P0_trackml_score" in full and full.P0_trackml_score.notna().any():
        s = full.loc[full.P0_trackml_score.idxmax()]
        out["WP-S"] = {"config": int(s.config), "family": s.family, "params": s.params,
                       "trackml_score": float(s.P0_trackml_score)}

    def wpr(df, eff, fake, limit, higher_is_better_fake=False):
        if df is None:
            return None
        ok = df[(df[fake] >= limit - 1e-12) if higher_is_better_fake else (df[fake] <= limit + 1e-12)]
        if ok.empty:
            return {"reached": False}
        r_ = ok.loc[ok[eff].idxmax()]
        return {"reached": True, "config": int(r_.config), "family": r_.family, "params": r_.params,
                "eff": float(r_[eff]), "fake_metric": float(r_[fake])}
    out["WP-R"] = {
        # J-purity over tracks matched to a selected particle in Ju et al.'s base selection (|eta| < 4, pT > 0.1 GeV;
        # their purity sentence follows "without any fiducial pT cut")
        "P1": wpr(nonoise, "P1_eps_tech_pt05", "P1_purity_J_pt01", refs["P1"]["purity"][0], True),
        "P2": wpr(full, "P2_eff", "P2_fr_liu", refs["P2"]["full"]["EC-GNN"][1]),
        "P3": wpr(full, "P3_efficiency", "P3_fake", refs["P3"]["fake"][0]),
        "P4": wpr(onegev, "P3_efficiency", "P3_fake", refs["P4"]["EC"]["fake"][0]),
    }
    return out


def cmd_select(a, cfg):
    load = lambda d: pd.read_csv(os.path.join(d, "summary.csv")) if d else None
    res = select(load(a.grid), float(cfg["builders"]["fr_target"]), load(a.grid_nonoise), load(a.grid_1gev))
    res.update({"grid": a.grid, "grid_nonoise": a.grid_nonoise, "grid_1gev": a.grid_1gev})
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------------------------------------ learned merging data
def _lmerge_events(args):
    events, scores_dir = args
    sc_all = read_scores(scores_dir, events)
    Xs, ys, es = [], [], []
    for ev in E.iter_events(events, particles=False):
        a, b, sc = sc_all[ev.event]
        st = EventState(ev, a, b, sc)
        first = LM.first_pass(st.n, st.a, st.b, st.sc, st.L)
        pairs = lm_candidates(first, st.r, st.phi, st.z)
        X = LM.features(first, pairs, st.r, st.phi, st.z, st.geo)
        y = LM.labels_for(first, pairs, ev.pid) if len(pairs) else np.zeros(0, dtype=bool)
        Xs.append(X)
        ys.append(y)
        es.append(np.full(len(y), int(ev.eid), dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(es)


def cmd_lmerge_data(a, cfg):
    import concurrent.futures as cf
    events = CFG.event_set(a.events, cfg)
    chunks = [(events[k:k + 5], a.scores) for k in range(0, len(events), 5)]
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_lmerge_events, chunks))
    X, y, ev = (np.concatenate([r[i] for r in res]) for i in range(3))
    np.savez(a.out, X=X, y=y, event=ev, features=np.array(LM.FEATURES))
    print(json.dumps({"events": len(events), "pairs": int(len(y)), "positive_frac": float(y.mean()) if len(y) else 0}))


# ------------------------------------------------------------------------------------------------ gate G3
def merge_check(scores: str, event: int, s0: float = 0.2, original: bool = True) -> dict:
    """The fast fragment merging against the original hnl.trackbuild.merge_fragments (both modes) on the first-pass
    labels of the same-layer MWS (s0) of one full event: identical labels, fragment count and run times."""
    from hnl.trackbuild.merge_fragments import merge_fragments
    from hnl.trackbuild.mutex_ws import mws_layer
    ev = next(E.iter_events([event], particles=False))
    a_, b_, sc = read_scores(scores, [event])[ev.event]
    st = EventState(ev, a_, b_, sc)
    t0 = time.time()
    lab = mws_layer(st.n, st.a, st.b, st.sc, st.L, s0)
    out = {"event": event, "s0": s0, "mws_layer_s": round(time.time() - t0, 1),
           "n_fragments": int(len(np.unique(lab[lab >= 0])))}
    for mode in ("any", "sequential"):
        t0 = time.time()
        fast = merge_fragments_fast(st.r, st.phi, st.z, lab, mode=mode)
        out[f"{mode}_fast_s"] = round(time.time() - t0, 1)
        out[f"{mode}_merged_fragments"] = int(len(np.unique(lab[lab >= 0])) - len(np.unique(fast[fast >= 0])))
        if original:
            t0 = time.time()
            ref = merge_fragments(st.r, st.phi, st.z, lab, mode=mode)
            out[f"{mode}_original_s"] = round(time.time() - t0, 1)
            out[f"{mode}_identical"] = bool(np.array_equal(fast, ref))
        print(json.dumps(out), flush=True)
    return out


def merge_profile(scores: str, event: int, s0: float = 0.2) -> dict:
    """Why helix fragment merging is expensive on a TrackML event: fragments after the same-layer MWS (s0), the candidate
    pairs of both modes (fast search, timed) and the size of the refit that hnl.trackbuild.merge_fragments._merge does
    at once (every candidate union padded to the longest one: candidates x longest union x 8 B per array)."""
    from hnl.trackbuild.mutex_ws import mws_layer
    ev = next(E.iter_events([event], particles=False))
    a_, b_, sc = read_scores(scores, [event])[ev.event]
    st = EventState(ev, a_, b_, sc)
    lab = mws_layer(st.n, st.a, st.b, st.sc, st.L, s0)
    ok = lab >= 0
    trk, cnt = np.unique(lab[ok], return_counts=True)
    trk = trk[cnt >= 2]
    size = dict(zip(trk.tolist(), cnt[cnt >= 2].tolist()))
    hits = _tracks_of(lab, trk)
    out = {"event": event, "s0": s0, "n_fragments": int(len(trk)), "max_fragment_hits": int(max(size.values()))}
    for mode in ("any", "sequential"):
        t0 = time.time()
        if mode == "any":
            cand = _candidates_any_fast(trk, hits, st.r, st.phi, st.z, st.R, 0.05, 0.05)
        else:
            cand = _candidates_seq(trk, hits, st.R, st.phi, 0.3, 400.0)
        dt = time.time() - t0
        u = np.array([size[int(x)] + size[int(y)] for x, y in cand]) if cand else np.zeros(1, dtype=int)
        out.update({f"{mode}_search_s": round(dt, 1), f"{mode}_candidates": len(cand),
                    f"{mode}_union_hits_sum": int(u.sum()), f"{mode}_union_hits_max": int(u.max()),
                    f"{mode}_padded_gb_per_array": round(len(cand) * int(u.max()) * 8 / 1e9, 2)})
    print(json.dumps(out), flush=True)
    return out


def cmd_merge_profile(a, cfg):
    res = merge_profile(a.scores, int(CFG.event_set(a.events, cfg)[0]), a.s0)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "merge_profile.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def cmd_merge_check(a, cfg):
    res = merge_check(a.scores, int(CFG.event_set(a.events, cfg)[0]), a.s0, not a.fast_only)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "merge_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def cmd_timing(a, cfg):
    """Gate G3 (plan section 6.7): per event and configuration, the build time (printed per configuration as it runs);
    with --check-original, merge_check on the first event."""
    if a.configs:
        cfgs, _ = explicit_configs(a.configs)
    else:
        builders = a.builders.split(",") if a.builders else list(cfg["builders"]["families"])
        if "mwslmerge" in builders and not a.merge_model:
            builders = [b for b in builders if b != "mwslmerge"]
        cfgs = grid_configs(builders, cfg["builders"].get("merge_cut_sets", ["H"]))
    events = CFG.event_set(a.events, cfg)[:int(a.max_events or cfg["builders"]["gate_g3"]["events"])]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "configs.json"), "w") as fh:
        json.dump([[x if isinstance(x, str) else float(x) for x in c] for c in cfgs], fh)
    res = _grid_events((events, a.scores, cfgs, a.merge_model, "full", True))
    tm = res[2]
    res[0].to_parquet(os.path.join(a.out, "per_event.parquet"), index=False)
    tm.to_parquet(os.path.join(a.out, "timing.parquet"), index=False)
    exact = merge_check(a.scores, events[0]) if a.check_original else None
    limit = 60.0 * float(cfg["builders"]["gate_g3"]["max_minutes_per_event_config"])
    per = tm.groupby("family").seconds.describe()
    ok_exact = exact is None or all(exact.get(f"{m}_identical", True) for m in ("any", "sequential"))
    out = {"events": events, "max_seconds": float(tm.seconds.max()), "limit_seconds": limit,
           "gate_g3_pass": bool(tm.seconds.max() <= limit and ok_exact),
           "per_family": per.round(2).to_dict(orient="index"), "merge_exactness": exact,
           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    with open(os.path.join(a.out, "gate_g3.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--scores", required=True)
    g.add_argument("--events", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--builders", default=None)
    g.add_argument("--configs", default=None, help="explicit configurations (JSON file or string), e.g. the frozen "
                   "ones of `python -m hnl.trackml.freeze configs`; replaces --builders")
    g.add_argument("--merge-model", default=None)
    g.add_argument("--condition", default="full", choices=E.CONDITIONS)
    g.add_argument("--allow-test", action="store_true")
    g.add_argument("--max-events", type=int, default=None)
    g.add_argument("--chunk", type=int, default=2)
    g.add_argument("--block", type=int, default=None, help="only events [K*N, (K+1)*N) of the set")
    g.add_argument("--block-size", type=int, default=10)
    g.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    gm = sub.add_parser("grid-merge")
    gm.add_argument("--blocks", nargs="+", required=True)
    gm.add_argument("--events", required=True)
    gm.add_argument("--out", required=True)
    gm.add_argument("--allow-test", action="store_true")
    gm.add_argument("--max-events", type=int, default=None)
    d = sub.add_parser("lmerge-data")
    d.add_argument("--scores", required=True)
    d.add_argument("--events", default="merge_train")
    d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    s = sub.add_parser("select")
    s.add_argument("--grid", required=True)
    s.add_argument("--grid-nonoise", default=None)
    s.add_argument("--grid-1gev", default=None)
    s.add_argument("--out", required=True)
    t = sub.add_parser("timing")
    t.add_argument("--scores", required=True)
    t.add_argument("--events", default="dev")
    t.add_argument("--out", required=True)
    t.add_argument("--builders", default=None)
    t.add_argument("--configs", default=None, help="explicit configurations (JSON file or string), replaces --builders")
    t.add_argument("--merge-model", default=None)
    t.add_argument("--check-original", action="store_true")
    t.add_argument("--max-events", type=int, default=None)
    mp = sub.add_parser("merge-profile")
    mp.add_argument("--scores", required=True)
    mp.add_argument("--events", required=True)
    mp.add_argument("--out", required=True)
    mp.add_argument("--s0", type=float, default=0.2)
    mc = sub.add_parser("merge-check")
    mc.add_argument("--scores", required=True)
    mc.add_argument("--events", required=True)
    mc.add_argument("--out", required=True)
    mc.add_argument("--s0", type=float, default=0.2)
    mc.add_argument("--fast-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = CFG.load(a.config)
    {"grid": cmd_grid, "grid-merge": cmd_grid_merge, "lmerge-data": cmd_lmerge_data, "select": cmd_select, "timing": cmd_timing,
     "merge-check": cmd_merge_check, "merge-profile": cmd_merge_profile}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
