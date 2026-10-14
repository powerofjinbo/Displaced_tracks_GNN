"""Stage-1 C on val: alternative track builders on the scored graphs of a run (no retraining).

For every event of an extracted gnn_scores/valset.tar and every builder configuration on a fixed grid, the track labels
are matched exactly as in hnl.eval.evaluate_gnn (FastMatch): per particle dm_matched and n_sm_tracks, per event
n_tracks_ge5, n_fake and the HNL pair flags. The configuration of each builder is chosen afterwards on val by
hnl.eval.trackbuild_select (fake rate <= FR*, then the highest primary efficiency: the multi-threshold builder rule of
DECISIONS.md, compared with CC at the run's preregistered §4 cut).

    python -m hnl.eval.trackbuild_val --graphs DIR --out OUT [--builders cc,jr,walk,ccwalk,...,walktripmerge]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.eval import matching as M
from hnl.eval.evaluate_gnn import _event_id, event_truth
from hnl.graphs import ceiling as C
from hnl.trackbuild import builders as TB

GRID = np.round(np.arange(0.1, 0.951, 0.1), 2)


def configs(builders: list[str]) -> list[tuple]:
    out = []
    if "cc" in builders:
        out += [("cc", c) for c in np.round(np.arange(0.05, 0.951, 0.05), 2)]
    if "jr" in builders:
        out += [("jr", c, j) for c, j in itertools.product(GRID, GRID) if j <= c]
    if "walk" in builders:
        out += [("walk", m, a) for m, a in itertools.product(GRID, np.round(np.r_[GRID, 0.99], 2)) if a >= m]
    if "ccmerge" in builders:
        out += [("ccmerge", c, m) for c in np.round(np.arange(0.3, 0.951, 0.1), 2) for m in (0.5, 1.0, 2.0)]
    if "walkmerge" in builders:
        out += [("walkmerge", m, a, rms) for m in (0.1, 0.2, 0.3) for a in (0.3, 0.5, 0.7, 0.9) if a >= m
                for rms in (0.5, 1.0)]
    if "ccmergeany" in builders:
        out += [("ccmergeany", c, m) for c in np.round(np.arange(0.3, 0.951, 0.1), 2) for m in (0.5, 1.0)]
    if "walkmerge2" in builders:             # looser merging: z rms (long strips), R gap, direction agreement
        out += [("walkmerge2", m, 0.5, zr, gap, dd) for m in (0.1, 0.2) for zr in (5.0, 10.0) for gap in (400.0, 800.0)
                for dd in (0.05, 0.1)]
    if "walktripmerge" in builders:
        out += [("walktripmerge", m, 0.5, ang, dcot) for m in (0.05, 0.1, 0.2) for ang in (0.1, 0.2, 0.4)
                for dcot in (0.1, 0.3)]
    if "walkmergeS" in builders:             # compact walkthrough + merging grid for Stage-1 runs (val read-out)
        out += [("walkmerge", m, 0.5, 1.0) for m in (0.1, 0.2)]
    if "walkmergeX" in builders:             # superset of walkmergeS, extended at the upper edge (DECISIONS 2026-10-04)
        out += [("walkmerge", m, a, 1.0) for m in (0.1, 0.2, 0.3, 0.4, 0.5) for a in (0.5, 0.7) if a >= m]
    if "mws" in builders:                    # Mutex Watershed on the signed graph edges; beta = -logit(c)
        out += [("mws", c) for c in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    if "mwslift" in builders:                # learned same-particle pair head on graph + lifted pairs ($HNL_MWS_MODEL)
        out += [("mwslift", c) for c in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    if "mwsliftmerge" in builders:           # + helix fragment merging (any order, then sequential; rms 1.0)
        out += [("mwsliftmerge", c) for c in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    if "mwslayer" in builders:               # Mutex Watershed, only same-layer pairs repel (hit layers from the hits table)
        out += [("mwslayer", c) for c in (0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)]
    if "mwslayermerge" in builders:
        out += [("mwslayermerge", c) for c in (0.03, 0.05, 0.1, 0.2)]
    if "mwslayermergeX" in builders:         # extended at the upper edge (DECISIONS 2026-10-04)
        out += [("mwslayermerge", c) for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
    if "mws2pass" in builders:               # second same-layer MWS pass at a lower s0 on the leftover hits
        out += [("mws2pass", c1, c2) for c1 in (0.1, 0.2) for c2 in (0.01, 0.02, 0.03, 0.05)]
    if "mws2passmerge" in builders:          # + helix fragment merging
        out += [("mws2passmerge", c1, c2) for c1 in (0.1, 0.2) for c2 in (0.01, 0.02, 0.03, 0.05)]
    if "mws2passmergeX" in builders:         # extended at the upper edge (DECISIONS 2026-10-04)
        out += [("mws2passmerge", c1, c2) for c1 in (0.2, 0.3, 0.4, 0.5) for c2 in (0.02, 0.05, 0.1)]
    if "mwsrep" in builders:                 # GNN attraction, repulsion from the pair head's confident pairs
        out += [("mwsrep", c, p) for c in (0.03, 0.05, 0.1, 0.2) for p in (0.003, 0.01, 0.03, 0.1)]
    if "wmattach" in builders:               # walkthrough + merging, then learned graph-free attachment
        out += [("wmattach", t) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    if "mwsattach" in builders:              # same-layer MWS (s0 = 0.1), then attachment ($HNL_MWSATTACH_MODEL)
        out += [("mwsattach", t) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    if "mwsmattach" in builders:             # same-layer MWS (s0 = 0.2) + merging, then attachment
        out += [("mwsmattach", t) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    if "mwslmattach" in builders:            # same-layer MWS + learned merging (tau 0.1), then attachment
        out += [("mwslmattach", t) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    if "mwslmerge" in builders:              # same-layer MWS (s0 = 0.2), then learned fragment merging ($HNL_MERGE_MODEL)
        out += [("mwslmerge", t) for t in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)]
    if "ldc" in builders:                    # learned dendrogram cut; model from --ldc-model / $HNL_LDC_MODEL
        out += [("ldc", t) for t in np.round(np.arange(0.05, 0.951, 0.05), 2)]
    if "vetowalkmerge" in builders:          # vertex-aware models only (edge_vtx_scores)
        out += [("vetowalkmerge", m, 0.5, 1.0, v) for m in (0.1, 0.2) for v in (0.3, 0.5, 0.7, 0.9)]
    if "vetocc" in builders:
        out += [("vetocc", c, v) for c in np.round(np.arange(0.3, 0.951, 0.1), 2) for v in (0.3, 0.5, 0.7, 0.9)]
    if "ccwalk" in builders:
        # acorn's regime: a low CC cut, walk thresholds at or above it (below the cut they would have no effect)
        out += [("ccwalk", c, m, a) for c in (0.01, 0.1, 0.3) for m in (0.1, 0.3, 0.5) if m >= c
                for a in (0.5, 0.7, 0.9) if a >= m]
    # acorn's own CCandWalk code (hnl.trackbuild.acorn_walk; acwalk = plain walk as in acorn's CTD 2023 config,
    # acwalkg = score-guided walk, guidance mode 2, as in its Rel. 24 / ACAT 2025 configs), parameters (score_cut_cc,
    # walk min, walk add): the walk's min threshold on CC's 0.05 grid at acorn's score_cut_cc 0.01 and add 0.6, add
    # 0.4 / 0.8 at min 0.05-0.2, and CC + walk at one threshold (0.1, 0.3). Feedback round 1 (strongest baselines).
    acw = ([(0.01, m, 0.6) for m in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)]
           + [(0.01, m, a) for m in (0.05, 0.1, 0.2) for a in (0.4, 0.8)] + [(c, c, 0.6) for c in (0.1, 0.3)])
    for fam in ("acwalk", "acwalkg"):
        if fam in builders:
            out += [(fam, *p) for p in acw]
    if "acwalkX" in builders:                # extension at the lower edge (acorn (ttbar) chose min 0.02, the grid's
        # lowest): CC and walk at one threshold c <= 0.01 (c = 0: every edge kept, the walk never stops on score)
        out += [(fam, c, c, 0.6) for fam in ("acwalk", "acwalkg")
                for c in (0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.0)]
    return out


def build(cfg, n, a, b, sc, R, hits=None, vtx=None):
    """Track labels of one builder configuration. vtx: per-edge cross-daughter scores of a vertex-aware model
    (hnl.models.vertex_gnn), needed by the veto families, which drop edges with vtx > v before building."""
    kind = cfg[0]
    if kind.startswith("veto"):
        if vtx is None:
            raise ValueError(f"{kind} needs edge_vtx_scores (a vertex-aware model's scores)")
        keep = np.asarray(vtx, dtype=np.float64) <= float(cfg[-1])
        return build((kind[4:], *cfg[1:-1]), n, a[keep], b[keep], sc[keep], R, hits)
    if kind == "cc":
        return TB.cc(n, a, b, sc, cfg[1])
    if kind == "jr":
        return TB.junction_removal(n, a, b, sc, R, cfg[1], cfg[2])
    if kind == "walk":
        return TB.walkthrough(n, a, b, sc, R, cfg[1], cfg[2])
    if kind == "ccwalk":
        return TB.cc_and_walk(n, a, b, sc, R, cfg[1], cfg[2], cfg[3])
    if kind == "ccmerge":
        from hnl.trackbuild.merge_fragments import merge_fragments
        r, phi, z = hits
        return merge_fragments(r, phi, z, TB.cc(n, a, b, sc, cfg[1]), max_rms=cfg[2])
    if kind == "walkmerge":
        from hnl.trackbuild.merge_fragments import merge_fragments
        r, phi, z = hits
        lab = TB.walkthrough(n, a, b, sc, R, cfg[1], cfg[2])
        lab = merge_fragments(r, phi, z, lab, max_rms=cfg[3], mode="any")
        return merge_fragments(r, phi, z, lab, max_rms=cfg[3])
    if kind == "walktripmerge":
        from hnl.trackbuild.merge_fragments import merge_fragments
        r, phi, z = hits
        xyz = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
        lab = TB.walkthrough_triplet(n, a, b, sc, R, xyz, cfg[1], cfg[2], max_angle=cfg[3], max_dcot=cfg[4])
        lab = merge_fragments(r, phi, z, lab, max_rms=1.0, mode="any")
        return merge_fragments(r, phi, z, lab, max_rms=1.0)
    if kind == "walkmerge2":
        from hnl.trackbuild.merge_fragments import merge_fragments
        r, phi, z = hits
        lab = TB.walkthrough(n, a, b, sc, R, cfg[1], cfg[2])
        opt = dict(max_rms=1.0, max_zrms=cfg[3], max_gap_r=cfg[4], max_dphi_dir=cfg[5], max_dcot=cfg[5])
        lab = merge_fragments(r, phi, z, lab, mode="any", **opt)
        return merge_fragments(r, phi, z, lab, **opt)
    if kind == "ccmergeany":
        from hnl.trackbuild.merge_fragments import merge_fragments
        r, phi, z = hits
        lab = merge_fragments(r, phi, z, TB.cc(n, a, b, sc, cfg[1]), max_rms=cfg[2], mode="any")
        return merge_fragments(r, phi, z, lab, max_rms=cfg[2])
    raise ValueError(kind)


def load_scored(path: str, with_vtx: bool = False):
    """One scored graph: (event id, pid, r, phi, z, R, a, b, score) with float64 coordinates and every unordered edge
    once (a < b) with the highest score of its directions, as in hnl.eval.evaluate_gnn. with_vtx: also the
    cross-daughter score of that same directed edge (edge_vtx_scores; None if the graph has none)."""
    from acorn.utils.loading_utils import load_pyg
    g = load_pyg(path[:-3] if path.endswith(".gz") else path, weights_only=False)
    eid = _event_id(g)
    pid = g.hit_particle_id.numpy()
    n = len(pid)
    r = (g.r if "r" in g else g.hit_r).numpy().astype(np.float64)
    z = (g.z if "z" in g else g.hit_z).numpy().astype(np.float64)
    phi = (g.phi if "phi" in g else g.hit_phi).numpy().astype(np.float64)
    R = np.hypot(r, z)
    ei = g.edge_index.numpy()
    sc = (g.scores if "scores" in g else g.edge_scores).numpy()
    a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    code = a.astype(np.int64) * n + b
    order = np.lexsort((-sc, code))
    first = np.ones(len(code), dtype=bool)
    first[1:] = code[order][1:] != code[order][:-1]
    k = order[first]
    if with_vtx:
        vtx = g.edge_vtx_scores.numpy()[k] if "edge_vtx_scores" in g else None
        return eid, pid, r, phi, z, R, a[k], b[k], sc[k], vtx
    return eid, pid, r, phi, z, R, a[k], b[k], sc[k]


_LDC = None
_ATT = {}
_MWS = {}


def _mws_model(lifted: bool):
    if lifted not in _MWS:
        from hnl.trackbuild.mutex_ws import MWS
        path = os.environ.get("HNL_MWS_MODEL") if lifted else None
        if lifted and not path:
            raise ValueError("mwslift configurations need --mws-model (or $HNL_MWS_MODEL)")
        _MWS[lifted] = MWS(path)
    return _MWS[lifted]


ATTACH_FAMILIES = {"wmattach": ("walkmerge", "HNL_ATTACH_MODEL", "--attach-model"),
                   "mwsattach": ("mwslayer", "HNL_MWSATTACH_MODEL", "--mwsattach-model"),
                   "mwsmattach": ("mwslayermerge", "HNL_MWSMATTACH_MODEL", "--mwsmattach-model"),
                   "mwslmattach": ("mwslmerge", "HNL_MWSLMATTACH_MODEL", "--mwslmattach-model")}


def _attach_model(family: str):
    if family not in _ATT:
        from hnl.trackbuild.attach import Attacher
        kind, env, flag = ATTACH_FAMILIES[family]
        path = os.environ.get(env)
        if not path:
            raise ValueError(f"{family} configurations need {flag} (or ${env})")
        _ATT[family] = Attacher(path)
        if _ATT[family].first_pass != kind:
            raise ValueError(f"{path}: trained on the {_ATT[family].first_pass} first pass, {family} uses {kind}")
    return _ATT[family]


def _ldc_model():
    global _LDC
    if _LDC is None:
        from hnl.trackbuild.ldc import LDC
        path = os.environ.get("HNL_LDC_MODEL")
        if not path:
            raise ValueError("ldc configurations need --ldc-model (or $HNL_LDC_MODEL)")
        _LDC = LDC(path)
    return _LDC


LAYER_KINDS = ("mwslayer", "mwslayermerge", "mwslift", "mwsliftmerge", "mwsrep", "mws2pass", "mws2passmerge")
GEO_KINDS = (*ATTACH_FAMILIES, "mwslmerge")  # need the hits' module geometry (hnl.geometry.resolution)
_MERGE = None


def _merge_model():
    global _MERGE
    if _MERGE is None:
        from hnl.trackbuild.learned_merge import Merger
        path = os.environ.get("HNL_MERGE_MODEL")
        if not path:
            raise ValueError("mwslmerge configurations need --merge-model (or $HNL_MERGE_MODEL)")
        _MERGE = Merger(path)
    return _MERGE


def evaluate_event(path: str, cfgs: list[tuple], hnl_pairs: dict, keep_ids: dict | None = None, layers=None,
                   geometry=None, labels_out: dict | None = None):
    """labels_out: if given, receives the track labels of every configuration, keyed (event id, config index)."""
    eid, pid, r, phi, z, R, a, b, sc, vtx = load_scored(path, with_vtx=True)
    L = None if layers is None else layers[int(eid)]
    if L is not None:
        assert len(L) == len(pid)
    geo = None if geometry is None else geometry[int(eid)]
    if geo is not None:
        assert len(geo["volume"]) == len(pid)
    n = len(pid)
    fm = M.FastMatch(pid)
    pair = hnl_pairs.get(eid)
    dm = np.zeros((len(cfgs), fm.P), dtype=bool)
    nsm = np.zeros((len(cfgs), fm.P), dtype=np.int16)
    rows = []
    ldc_state = None
    att_state = {}
    merge_state = None
    mws_state = {}
    acorn_state = None
    for i, cfg in enumerate(cfgs):
        if cfg[0] in ("acwalk", "acwalkg"):  # acorn's CCandWalk on the stored graph (stored edge order), per event
            if acorn_state is None:
                from hnl.trackbuild.acorn_walk import AcornCCWalk
                acorn_state = AcornCCWalk.from_path(path)
                if acorn_state.n != n:
                    raise ValueError(f"{path}: {acorn_state.n} hits for acorn's builder, {n} here")
            lab = acorn_state.labels(cfg[1], cfg[2], cfg[3], guided=cfg[0] == "acwalkg")
        elif cfg[0] == "ldc":                # one dendrogram and one q per event, any theta
            if ldc_state is None:
                ldc_state = _ldc_model().scores(n, a, b, sc, r, phi, z)
            from hnl.trackbuild.ldc import LDC
            lab = LDC.labels(*ldc_state, cfg[1])
        elif cfg[0] in ("mws", "mwslift", "mwsliftmerge", "mwsrep"):   # one set of signed weights per event
            from hnl.trackbuild.mutex_ws import MWS, _logit
            lifted = cfg[0] != "mws"
            if lifted not in mws_state:
                mws_state[lifted] = _mws_model(lifted).prepare(n, a, b, sc, r, phi, z, L)
            if cfg[0] == "mwsrep":
                lab = MWS.labels_rep(mws_state[lifted], a, b, sc, cfg[1], cfg[2])
            else:
                lab = MWS.labels(mws_state[lifted], -float(_logit(cfg[1])))
            if cfg[0] == "mwsliftmerge":
                from hnl.trackbuild.merge_fragments import merge_fragments
                lab = merge_fragments(r, phi, z, lab, max_rms=1.0, mode="any")
                lab = merge_fragments(r, phi, z, lab, max_rms=1.0)
        elif cfg[0] in ("mws2pass", "mws2passmerge"):
            from hnl.trackbuild.mutex_ws import mws_two_pass
            lab = mws_two_pass(n, a, b, sc, L, cfg[1], cfg[2])
            if cfg[0] == "mws2passmerge":
                from hnl.trackbuild.merge_fragments import merge_fragments
                lab = merge_fragments(r, phi, z, lab, max_rms=1.0, mode="any")
                lab = merge_fragments(r, phi, z, lab, max_rms=1.0)
        elif cfg[0] in ("mwslayer", "mwslayermerge"):
            from hnl.trackbuild.mutex_ws import mws_layer
            lab = mws_layer(n, a, b, sc, L, cfg[1])
            if cfg[0] == "mwslayermerge":
                from hnl.trackbuild.merge_fragments import merge_fragments
                lab = merge_fragments(r, phi, z, lab, max_rms=1.0, mode="any")
                lab = merge_fragments(r, phi, z, lab, max_rms=1.0)
        elif cfg[0] == "mwslmerge":          # one first pass and one p per event, any tau
            from hnl.trackbuild import learned_merge as LM
            if merge_state is None:
                if geo is None:
                    raise ValueError("mwslmerge needs the hits' module geometry (hnl.geometry.resolution.hit_geometry)")
                first = LM.first_pass(n, a, b, sc, geo["layer_id"])
                merge_state = (first, *_merge_model().prepare(first, r, phi, z, geo))
            lab = LM.apply(merge_state[0], merge_state[1], merge_state[2], cfg[1], geo["layer_id"])
        elif cfg[0] in ATTACH_FAMILIES:      # one first pass and one p per event and family, any tau
            from hnl.trackbuild import attach as AT
            if cfg[0] not in att_state:
                if geo is None:
                    raise ValueError(f"{cfg[0]} needs the hits' module geometry (hnl.geometry.resolution.hit_geometry)")
                kind = ATTACH_FAMILIES[cfg[0]][0]
                first = AT.first_pass(n, a, b, sc, R, (r, phi, z), kind, geo["layer_id"], geo)
                att_state[cfg[0]] = (first, *_attach_model(cfg[0]).prepare(first, r, phi, z, a, b, sc, geo))
            st = att_state[cfg[0]]
            lab = AT.apply(st[0], st[1], st[2], cfg[1])
        else:
            lab = build(cfg, n, a, b, sc, R, (r, phi, z), vtx)
        if labels_out is not None:
            labels_out[(eid, i)] = np.asarray(lab)
        m = fm.match(lab)
        dm[i], nsm[i] = m["dm_matched"], m["n_sm_tracks"]
        row = {"event_id": int(eid), "config": i, "n_tracks_ge5": m["n_tracks_ge5"], "n_fake": m["n_fake"]}
        if pair is not None:
            ia, ib = np.searchsorted(fm.particles, pair)
            row["pair_both"] = bool(m["dm_matched"][ia] and m["dm_matched"][ib])
            row["pair_merged"] = fm.merged(m, *pair)
        rows.append(row)
    sel = np.ones(fm.P, dtype=bool) if keep_ids is None else np.isin(fm.particles, keep_ids.get(eid, []))
    P = int(sel.sum())
    part = pd.DataFrame({"event_id": np.int64(eid), "particle_id": np.tile(fm.particles[sel].astype(np.int64), len(cfgs)),
                         "config": np.repeat(np.arange(len(cfgs), dtype=np.int16), P), "dm_matched": dm[:, sel].ravel(),
                         "n_sm_tracks": nsm[:, sel].ravel()})
    return part, pd.DataFrame(rows)


def _chunk(args):
    paths, cfgs, pairs, keep = args
    layers = geometry = None
    if any(c[0] in GEO_KINDS for c in cfgs):  # one read of the hits table gives both
        from hnl.geometry.resolution import hit_geometry
        geometry = hit_geometry([int(_eid_of(p)) for p in paths])
        layers = {e: g["layer_id"] for e, g in geometry.items()}
    elif any(c[0] in LAYER_KINDS for c in cfgs):
        from hnl.trackbuild.mutex_ws import hit_layers
        layers = hit_layers([int(_eid_of(p)) for p in paths])
    res = [evaluate_event(p, cfgs, pairs, keep, layers, geometry) for p in paths]
    return pd.concat([r[0] for r in res], ignore_index=True), pd.concat([r[1] for r in res], ignore_index=True)


def _eid_of(path: str) -> str:
    return os.path.basename(path).split("event")[-1].split(".")[0]


def main(argv=None):
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--builders", default="cc,jr,walk,ccwalk,ccmerge,walkmerge,ccmergeany")
    ap.add_argument("--config", default=None,
                    help="exactly one configuration 'family,p1,p2,...' instead of the grids (the one fixed on val, for "
                         "the one-time test evaluation)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--ldc-model", default=None, help="joblib model of hnl.trackbuild.ldc (for the 'ldc' family)")
    ap.add_argument("--attach-model", default=None, help="joblib model of hnl.trackbuild.attach ('wmattach')")
    ap.add_argument("--mwsattach-model", default=None,
                    help="joblib model of hnl.trackbuild.attach trained on the mwslayer first pass ('mwsattach')")
    ap.add_argument("--merge-model", default=None, help="joblib model of hnl.trackbuild.learned_merge ('mwslmerge')")
    ap.add_argument("--mwslmattach-model", default=None,
                    help="joblib model of hnl.trackbuild.attach trained on the mwslmerge first pass ('mwslmattach')")
    ap.add_argument("--mwsmattach-model", default=None,
                    help="joblib model of hnl.trackbuild.attach trained on the mwslayermerge first pass ('mwsmattach')")
    ap.add_argument("--mws-model", default=None, help="joblib lifted-edge model of hnl.trackbuild.mutex_ws ('mwslift')")
    a = ap.parse_args(argv)
    if a.ldc_model:
        os.environ["HNL_LDC_MODEL"] = os.path.abspath(a.ldc_model)       # inherited by the worker processes
    if a.attach_model:
        os.environ["HNL_ATTACH_MODEL"] = os.path.abspath(a.attach_model)
    if a.mwsattach_model:
        os.environ["HNL_MWSATTACH_MODEL"] = os.path.abspath(a.mwsattach_model)
    if a.mwsmattach_model:
        os.environ["HNL_MWSMATTACH_MODEL"] = os.path.abspath(a.mwsmattach_model)
    if a.merge_model:
        os.environ["HNL_MERGE_MODEL"] = os.path.abspath(a.merge_model)
    if a.mwslmattach_model:
        os.environ["HNL_MWSLMATTACH_MODEL"] = os.path.abspath(a.mwslmattach_model)
    if a.mws_model:
        os.environ["HNL_MWS_MODEL"] = os.path.abspath(a.mws_model)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))[:a.max_events]
    eids = [os.path.basename(p).split("event")[-1].split(".")[0] for p in paths]
    pairs, keep = event_truth(eids)
    if a.config:
        parts = a.config.split(",")
        cfgs = [(parts[0], *[float(x) for x in parts[1:]])]
    else:
        cfgs = configs(a.builders.split(","))
    chunks = [(paths[k:k + 10], cfgs, pairs, keep) for k in range(0, len(paths), 10)]
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        res = list(ex.map(_chunk, chunks))
    pd.concat([r[0] for r in res], ignore_index=True).to_parquet(os.path.join(a.out, "per_particle.parquet"),
                                                                  index=False)
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(os.path.join(a.out, "per_event.parquet"),
                                                                  index=False)
    with open(os.path.join(a.out, "configs.json"), "w") as f:
        json.dump([list(map(lambda x: x if isinstance(x, str) else float(x), c)) for c in cfgs], f)
    print(json.dumps({"n_events": len(paths), "n_configs": len(cfgs), "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
