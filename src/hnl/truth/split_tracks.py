"""d0 resolution from split-track consistency (brief §5b, second of three resolution estimates).

For every target particle (pT > 1 GeV, |eta| < 4) with >= 8 hits, the chain-ordered hits are split into an inner and
an outer half; each half is fitted exactly like the full track (muons: fixed R from the table pT; others: free
curvature). The spread of d0(inner) - d0(outer) (and of each half against the full fit) measures the d0 precision on
real hits, multiple scattering and energy loss included. Halves have half the hits and a shorter lever arm, so these
spreads are upper bounds on the full-track resolution.

    python -m hnl.truth.split_tracks --out <dir> [--samples hnl10_100,ttbar] [--events 2000]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from hnl.data.npz import COL, load_event, particle_ids
from hnl.geometry.ids import cylindrical_to_cartesian, decode_geometry_id
from hnl.truth import helix as hx
from hnl.truth import labels as L
from hnl.truth.fit_events import _pad_like, hit_scales

D0_BINS = [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf]


def _fit(p: hx.Padded, R0, sigma, muon, scale_t):
    f0 = hx.fit_circle(p, R0, sigma, scale=scale_t)
    pg = hx.perigee(f0.xc, f0.yc, f0.R, f0.sigma)
    k0 = sigma / R0
    fix = hx.fit_perigee(p, pg.d0, pg.phi0, k0, free_kappa=False, scale=scale_t)
    free = hx.fit_perigee(p, fix.d0, fix.phi0, k0, free_kappa=True, scale=scale_t)
    return np.where(muon, fix.d0, free.d0), np.where(muon, fix.converged, free.converged)


def split_event(ev: dict, min_hits: int = 8) -> pd.DataFrame:
    x, pid, te, P = ev["x"], ev["pid"], ev["layerless_true_edges"], ev["particles"]
    ids = particle_ids(P)
    order = hx.order_hits(pid, te)
    pid_o = pid[order]
    up, cnt = np.unique(pid_o, return_counts=True)
    sorter = np.argsort(ids)
    rows_p = sorter[np.searchsorted(ids, up, sorter=sorter)]
    tgt = (P[rows_p, COL["pt"]] > 1.0) & (np.abs(P[rows_p, COL["eta"]]) < 4.0) & (cnt >= min_hits)
    keep_ids = up[tgt]
    if not len(keep_ids):
        return pd.DataFrame()
    xyz = cylindrical_to_cartesian(x)
    vol = decode_geometry_id(ev["layers"])["volume"]
    st, _ = hit_scales(vol)
    sel = np.isin(pid_o, keep_ids)
    rows = order[sel]
    tid = pid_o[sel]
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    n = np.diff(np.r_[starts, len(tid)])
    pos = np.arange(len(tid)) - np.repeat(starts, n)
    half = pos >= np.repeat(n // 2, n)                       # outer half: positions >= n // 2
    prow = sorter[np.searchsorted(ids, tid[starts], sorter=sorter)]
    pt, q = P[prow, COL["pt"]], P[prow, COL["q"]]
    R0 = hx.MM_PER_GEV * pt
    sigma = np.where(q < 0, 1, -1).astype(np.int64)          # sigma = -sign(q)
    muon = np.abs(P[prow, COL["particle_type"]]) == 13
    out = {"particle_id": tid[starts], "pdg": P[prow, COL["particle_type"]].astype(np.int64), "pt": pt, "n_hits": n}
    for tag, m in (("full", np.ones(len(tid), bool)), ("inner", ~half), ("outer", half)):
        p = hx.pad_tracks(tid[m], xyz[rows[m]], rows=rows[m])
        d0, conv = _fit(p, R0, sigma, muon, _pad_like(p, st))
        out[f"d0_{tag}"], out[f"conv_{tag}"] = d0, conv
    df = pd.DataFrame(out)
    lab = L.labels_from_table(P)
    df["is_hnl"] = lab["hnl_daughter"][prow]
    df["is_w"] = lab["w_muon"][prow]
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", default="hnl10_100,ttbar")
    ap.add_argument("--events", type=int, default=2000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from hnl.data.splits import dropped_events
    drop = dropped_events()
    frames = []
    for s in a.samples.split(","):
        for i in range(a.events):
            if i in drop[s]:
                continue
            d = split_event(load_event(s, i))
            if len(d):
                frames.append(d.assign(sample=s, event=i))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(os.path.join(a.out, "split_tracks.parquet"), index=False)
    df["d0_bin"] = pd.cut(df.d0_full.abs(), D0_BINS, right=False)
    df["cat"] = np.where(df.is_hnl, "hnl", np.where(df.is_w, "w", np.where(df.pdg.abs() == 13, "mu",
                                                                             np.where(df.pdg.abs() == 11, "e", "had"))))
    rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan
    summ = (df.assign(dio=df.d0_inner - df.d0_outer, dif=df.d0_inner - df.d0_full, dof=df.d0_outer - df.d0_full)
            .groupby(["cat", "d0_bin"], observed=True)
            .agg(n=("dio", "size"), sig_inner_minus_outer=("dio", rs), sig_inner_minus_full=("dif", rs),
                 sig_outer_minus_full=("dof", rs)))
    summ.to_csv(os.path.join(a.out, "split_tracks_summary.csv"))
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
