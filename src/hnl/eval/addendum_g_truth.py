"""Truth tables for addendum G step T2 that T1 does not write. Truth only: hits, particle tables, truth sidecars and the
hit-level truth fits of hnl.truth.fit_events; no model output exists or is read. Definitions: docs/definitions.md,
docs/addendum_g_t2_readouts.md.

points        one HNL point of hnl.data.shards.POINTS (evaluation only; docs/hnl_points_spec.md), every event of
              splits/hnl_<m>_<ctau>_all.txt, read through the tar reader; one row per particle with hits:
                table values (pdg, q, pT, eta: final state, docs/definitions.md section 4), hits, pixel hits, category
                (column 12 as in the labeled Zenodo samples, hnl.truth.labels.labels_from_table: "hnl" = N daughter,
                |parent| = 50; "w" = W muon, |parent| = 24 and |PDG| = 13; else "other"), target flag;
                exact quantities from the truth sidecar (production vertex, initial momentum): d0_exact = |d0| of the
                helix through the production vertex in the 2 T field (hnl.eval.addendum_g_t1.d0_exact, as T1),
                r_prod = production radius, rv_exact = r_prod of an N daughter (the N decay radius), NaN otherwise;
                hit-fit quantities of docs/definitions.md section 3, computed exactly as derived/truth_fits/v1 and the
                census of results/hnl_points_truth_v1.md (hnl.truth.fit_events.fit_event on the whole event, field
                sign -1): d0_fit = |d0| (muons fixed-R, others free curvature; NaN below 3 hits); rv_fit = radius of
                the two-track vertex of the event's N-daughter pair (fit_events.pair_vertices, both daughters fitted,
                the vertex accepted, else NaN: the "no vertex" category).
              Populations: in_pop = P_HNL (fitted), N-daughter targets with d0_fit >= 10 mm or rv_fit >= 100 mm (the
              primary population of the Zenodo study, counted in results/hnl_points_truth_v1.md); in_pop_exact =
              P_HNL (exact), N-daughter targets with d0_exact >= 10 mm or rv_exact >= 100 mm; prompt = W-muon targets.
              -> derived/generated_v1/t2/truth_<point>.parquet (+ .json with the counts)
fits          generic_iso, one list (val, or union_test = the claims' test events): the hit-fit |d0| and fitted r_v of
              the gun particles (RESULTS T1 deviation 15, "P_G under the hit-fit |d0| and the fitted r_v of
              docs/definitions.md"): fit_event on the whole event as above; d0_fit as above; rv_fit of an LLP = the
              two-track vertex (hnl.truth.helix.two_track_vertex, the choice rule of docs/definitions.md section 3) of
              its two prongs' fitted circles, each prong's circle from the fit that gives its |d0| (fixed-R for muons,
              free curvature otherwise); NaN when a prong has no fit or the vertex is not accepted. Events below 8000
              from the npz files, the extension's (>= 8000) through the tar reader.
              -> derived/generated_v1/t2/fits_generic_iso_<list>.parquet
zenodo-shape  the r_v shape of the Zenodo HNL primary population, the target of the r_v reweighting of deviation 15:
              N-daughter targets of the labeled hnl10_100 and hnl15_200 samples, train and val splits only (no test
              event), with fitted |d0| >= 10 mm or fitted r_v >= 100 mm (derived/truth_fits/v1 via
              hnl.truth.report.load), histogram of the fitted r_v (accepted vertices) in 20 mm bins over [0, 800) mm;
              daughters without a vertex and with r_v >= 800 mm are counted beside it.
              -> results/addendum_g_zenodo_rv_shape_v1.csv

    python -m hnl.eval.addendum_g_truth points --sample hnl5_50 [--workers 8]
    python -m hnl.eval.addendum_g_truth fits --list val|union_test [--workers 8]
    python -m hnl.eval.addendum_g_truth zenodo-shape
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from hnl.eval.addendum_g import DATA, POINTS_ORDER, REPO, SPLIT_STEM, T2, X1_DIR, X1_FIRST, _git, split_events

FIELD_SIGN = -1                               # derived/truth_fits/v1 (docs/definitions.md section 3)
RV_SHAPE_EDGES = np.arange(0.0, 800.0 + 1e-9, 20.0)
SHAPE_CSV = os.path.join(REPO, "results", "addendum_g_zenodo_rv_shape_v1.csv")
ZENODO_SHAPE_SAMPLES = ("hnl10_100", "hnl15_200")
ZENODO_SHAPE_SPLITS = ("train", "val")


def _fits(ev: dict) -> pd.DataFrame:
    from hnl.truth.fit_events import fit_event
    from hnl.truth.report import truth_d0
    df, _ = fit_event(ev, FIELD_SIGN)
    df["d0_fit"] = truth_d0(df).abs()
    return df


# ------------------------------------------------------------------------------------------------- HNL points
def point_event(ev: dict, i: int) -> pd.DataFrame:
    from hnl.data.npz import COL, particle_ids
    from hnl.geometry.ids import PIXEL_VOLUMES, decode_barcode, decode_geometry_id
    from hnl.truth import labels as L
    from hnl.truth.fit_events import pair_vertices
    P, pid = ev["particles"], ev["pid"]
    ids = particle_ids(P)
    up, cnt = np.unique(pid, return_counts=True)
    order = np.argsort(ids)
    rows = order[np.searchsorted(ids, up, sorter=order)]
    vol = decode_geometry_id(ev["layers"])["volume"]
    upix, cpix = np.unique(pid[np.isin(vol, PIXEL_VOLUMES)], return_counts=True)
    lab = L.labels_from_table(P)
    cat = np.where(lab["hnl_daughter"], "hnl", np.where(lab["w_muon"], "w", "other"))[rows]
    bc = decode_barcode(up.astype(np.int64))
    t = pd.DataFrame({
        "event": np.int32(i), "particle_id": up.astype(np.int64), "pdg": P[rows, COL["particle_type"]].astype(np.int64),
        "q": P[rows, COL["q"]].astype(np.float32), "pt": P[rows, COL["pt"]], "eta": P[rows, COL["eta"]],
        "n_hits": cnt.astype(np.int32),
        "n_pixel_hits": pd.Series(cpix, index=upix).reindex(up).fillna(0).astype(np.int32).to_numpy(),
        "category": cat, "vsec": bc["vsec"].astype(np.int32), "gen": bc["gen"].astype(np.int32)})
    df = _fits(ev)
    t = t.merge(df[["particle_id", "d0_fit"]], on="particle_id", how="left", validate="one_to_one")
    v = pair_vertices(df, "label_hnl")
    t["rv_fit"] = np.nan
    if len(v) and bool(v.ok.iloc[0]):
        t.loc[t.particle_id.isin([int(v.pid_a.iloc[0]), int(v.pid_b.iloc[0])]), "rv_fit"] = float(v.rv.iloc[0])
    return t


def _point_chunk(args):
    directory, events = args
    from hnl.data import shards
    out = []
    for i, ev in shards.iter_events_from(directory, events, check=True):
        out.append(point_event(ev, i))
    if sorted(int(x) for x in pd.concat(out).event.unique()) != sorted(int(e) for e in events):
        raise RuntimeError(f"{directory}: missing events in a chunk")
    return pd.concat(out, ignore_index=True)


def point_truth(sample: str, workers: int) -> tuple[pd.DataFrame, dict]:
    from hnl.data import shards
    from hnl.eval.addendum_g_t1 import d0_exact
    directory = shards.sample_dir(sample)
    events = split_events(sample, "all")
    chunks = [(directory, [int(e) for e in events[k:k + 100]]) for k in range(0, len(events), 100)]
    with cf.ProcessPoolExecutor(max_workers=max(workers, 1)) as ex:
        t = pd.concat(list(ex.map(_point_chunk, chunks)), ignore_index=True)
    side = []
    for f in sorted(os.listdir(directory + "_truth")):
        if f.startswith("truth_") and f.endswith(".npz"):
            with np.load(os.path.join(directory + "_truth", f), allow_pickle=False) as z:
                side.append(pd.DataFrame({"event": z["p_event"].astype(np.int32),
                                          "particle_id": z["p_particle_id"].astype(np.int64),
                                          **{f"p_{k}": z[f"p_{k}"] for k in ("vx", "vy", "vz", "px", "py", "pz")},
                                          "p_pdg": z["p_particle_type"]}))
    side = pd.concat(side, ignore_index=True)
    if side.duplicated(["event", "particle_id"]).any():
        raise SystemExit(f"{sample}: duplicate (event, particle) in the truth sidecars")
    t = t.merge(side, on=["event", "particle_id"], how="left", validate="one_to_one")
    chk = {"particles_with_hits": len(t), "without_initial_state": int(t.p_px.isna().sum()),
           "sidecar_pdg_mismatch": int((t.p_pdg.notna() & (t.p_pdg != t.pdg)).sum())}
    t = t.drop(columns=["p_pdg"])
    t["r_prod"] = np.hypot(t.p_vx, t.p_vy)
    t["d0_exact"] = d0_exact(t.p_vx.to_numpy(), t.p_vy.to_numpy(), t.p_px.to_numpy(), t.p_py.to_numpy(),
                             t.q.to_numpy())
    hd = (t.category == "hnl").to_numpy()
    t["rv_exact"] = np.where(hd, t.r_prod, np.nan)
    t["is_target"] = (t.pt > 1.0) & (np.abs(t.eta) < 4.0) & (t.n_hits >= 5)
    t["in_pop"] = t.is_target & hd & ((t.d0_fit >= 10.0) | (t.rv_fit >= 100.0))
    t["in_pop_exact"] = t.is_target & hd & ((t.d0_exact >= 10.0) | (t.rv_exact >= 100.0))
    t["prompt"] = t.is_target & (t.category == "w")
    d = t[hd]
    pair = d.groupby("event").r_prod.agg(["size", "min", "max"])
    chk.update({"events": int(len(events)), "n_daughter_rows": int(hd.sum()),
                "daughter_pairs_with_unequal_r_prod_mm": int(((pair["size"] == 2) & (pair["max"] - pair["min"] > 1e-3))
                                                             .sum()),
                "n_daughter_targets": int((hd & t.is_target).sum()), "P_HNL_fitted": int(t.in_pop.sum()),
                "P_HNL_exact": int(t.in_pop_exact.sum()), "P_HNL_both": int((t.in_pop & t.in_pop_exact).sum()),
                "w_muon_targets": int(t.prompt.sum()),
                "daughter_targets_without_fitted_vertex": int((hd & t.is_target & t.rv_fit.isna()).sum())})
    return t, chk


def cmd_points(a) -> None:
    os.makedirs(T2, exist_ok=True)
    t, chk = point_truth(a.sample, a.workers)
    path = os.path.join(T2, f"truth_{a.sample}.parquet")
    t.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    from hnl.eval.addendum_g_t1 import sha256_file
    meta = {"sample": a.sample, "split_list": f"splits/{SPLIT_STEM[a.sample]}_all.txt", "checks": chk,
            "truth_table": path, "truth_table_sha256": sha256_file(path), "git": _git(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(T2, f"truth_{a.sample}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


# ------------------------------------------------------------------------------------------------- generic_iso fits
def llp_vertices(df: pd.DataFrame, gun: pd.DataFrame) -> pd.DataFrame:
    """Fitted r_v of each LLP of one event with two fitted prongs. df: fit_event rows; gun: (particle_id, llp) of the
    event's gun particles with hits."""
    from hnl.truth import helix as hx
    g = gun.merge(df, on="particle_id", how="inner")
    rows = []
    for llp, p in g.groupby("llp"):
        if len(p) != 2:
            continue
        h = []
        for _, r in p.iterrows():
            tag = "fix" if abs(int(r.pdg)) == 13 else "free"
            h.append({k: np.array([r[f"{tag}_{c}"]]) for k, c in (("xc", "xc"), ("yc", "yc"), ("R", "R"),
                                                                  ("sigma", "sigma"), ("x_pca", "xpca"),
                                                                  ("y_pca", "ypca"), ("z0", "z0"), ("cot", "cot"))})
        v = hx.two_track_vertex(h[0], h[1], np.array([p.r_first.min()]))
        rows.append({"llp": llp, "rv_fit_any": float(np.hypot(v.x, v.y)[0]), "rv_ok": bool(v.ok[0])})
    out = pd.DataFrame(rows, columns=["llp", "rv_fit_any", "rv_ok"])
    return out.astype({"llp": gun.llp.dtype, "rv_fit_any": float, "rv_ok": bool})


def _fits_chunk(args):
    from hnl.data import npz, shards
    events, gun = args
    out = []
    small = [e for e in events if e < X1_FIRST]
    big = [e for e in events if e >= X1_FIRST]
    src = [(e, npz.load_event("generic_iso", e)) for e in small]
    if big:
        src += list(shards.iter_events_from(X1_DIR, big, check=True))
    for e, ev in src:
        df = _fits(ev)
        ge = gun[gun.event == e][["particle_id", "llp"]]
        f = ge.merge(df[["particle_id", "pdg", "d0_fit", "n_hits"]], on="particle_id", how="left")
        f["event"] = np.int32(e)
        v = llp_vertices(df, ge)
        f = f.merge(v, on="llp", how="left")
        f["rv_ok"] = f.rv_ok.fillna(False).astype(bool)
        f["rv_fit"] = np.where(f.rv_ok, f.rv_fit_any, np.nan)
        out.append(f)
    return pd.concat(out, ignore_index=True)


def cmd_fits(a) -> None:
    from hnl.eval.addendum_g import _truth
    events = split_events("generic_iso", a.list)
    t = _truth(["generic_iso"])
    gun = t[t.gun.fillna(False).astype(bool) & t.event.isin(events)][["event", "particle_id", "llp"]]
    if a.list == "union_test" and not (events >= X1_FIRST).any():
        raise SystemExit("union_test without extension events")
    chunks = []
    for k in range(0, len(events), 50):
        ev = [int(e) for e in events[k:k + 50]]
        chunks.append((ev, gun[gun.event.isin(ev)]))
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        f = pd.concat(list(ex.map(_fits_chunk, chunks)), ignore_index=True)
    if len(f) != len(gun):
        raise SystemExit(f"{len(f)} fitted rows for {len(gun)} gun particles")
    os.makedirs(T2, exist_ok=True)
    path = os.path.join(T2, f"fits_generic_iso_{a.list}.parquet")
    f.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    from hnl.eval.addendum_g_t1 import sha256_file
    meta = {"list": a.list, "events": int(len(events)), "gun_particles_with_hits": int(len(gun)),
            "without_fit": int(f.d0_fit.isna().sum()), "llp_vertices_accepted": int(f.drop_duplicates(
                ["event", "llp"]).rv_ok.sum()), "table": path, "sha256": sha256_file(path), "git": _git(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with open(path.replace(".parquet", ".json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1))


# ------------------------------------------------------------------------------------------------- Zenodo r_v shape
def zenodo_shape() -> pd.DataFrame:
    from hnl.data.splits import read_split
    from hnl.truth.report import load
    fits = os.path.join(DATA, "derived", "truth_fits", "v1")
    lo, hi = RV_SHAPE_EDGES[:-1], RV_SHAPE_EDGES[1:]
    tab = pd.DataFrame({"lo": lo, "hi": hi})
    extra = []
    for s in ZENODO_SHAPE_SAMPLES:
        p, _ = load(fits, s)
        ev = np.concatenate([read_split(s, sp) for sp in ZENODO_SHAPE_SPLITS])
        p = p[p.event.isin(ev) & p.is_hnl & p.is_target]
        pop = p[(p.d0.abs() >= 10.0) | (p.rv_cat >= 100.0)]
        r = pop.rv_cat.to_numpy()
        tab[f"n_{s}"] = np.histogram(r[np.isfinite(r)], RV_SHAPE_EDGES)[0]
        extra.append({"sample": s, "population": len(pop), "no_vertex": int(np.isnan(r).sum()),
                      "rv_ge_800": int((r >= 800).sum())})
    tab["n"] = tab[[f"n_{s}" for s in ZENODO_SHAPE_SAMPLES]].sum(1)
    tab["fraction"] = tab.n / tab.n.sum()
    return tab, extra


def cmd_zenodo_shape(a) -> None:
    tab, extra = zenodo_shape()
    head = ("# Zenodo HNL primary population, r_v shape (target of the deviation-15 r_v reweighting): N-daughter targets "
            "of hnl10_100 and hnl15_200, train + val splits (no test event), fitted |d0| >= 10 mm or fitted r_v >= 100 "
            "mm; fitted r_v of accepted vertices in 20 mm bins (derived/truth_fits/v1). Outside the histogram: "
            + "; ".join(f"{e['sample']}: population {e['population']}, no vertex {e['no_vertex']}, r_v >= 800 mm "
                        f"{e['rv_ge_800']}" for e in extra) + f". hnl.eval.addendum_g_truth zenodo-shape, git {_git()[:7]}\n")
    with open(SHAPE_CSV, "w") as f:
        f.write(head)
        tab.to_csv(f, index=False)
    print(head + tab.to_string())


def read_shape() -> pd.DataFrame:
    return pd.read_csv(SHAPE_CSV, comment="#")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("points")
    p.add_argument("--sample", required=True, choices=list(POINTS_ORDER))
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p = sub.add_parser("fits")
    p.add_argument("--list", required=True, choices=["val", "union_test"])
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    sub.add_parser("zenodo-shape")
    a = ap.parse_args(argv)
    {"points": cmd_points, "fits": cmd_fits, "zenodo-shape": cmd_zenodo_shape}[a.cmd](a)


if __name__ == "__main__":
    main()
