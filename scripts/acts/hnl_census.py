"""Truth census of HNL events (W -> mu N, N -> mu mu nu) in the Zenodo npz format, for the closure of the HNL
generator settings against the Zenodo samples (docs/hnl_points_spec.md; results/hnl_points_closure_v1.md). No model
is run; only particle tables, hits and truth fits are read.

Inputs:
  zenodo   labeled Zenodo samples (hnl10_100, hnl15_200), the events of the given splits only (train and val; the
           test splits are never read). Truth fits (d0, r_v) come from derived/truth_fits/v1, which hnl.truth.fit_events
           made with the same function this script applies to our events (checked with --refit-check).
  shards   our tar-sharded events (hnl.data.shards) with their truth sidecars: the same census, truth fits computed
           here with hnl.truth.fit_events.fit_event / pair_vertices, plus the generator truth (true r_v, true d0).
  gen      generator level only (hnl_pythia.cc or ACTS particles_gen CSVs, no simulation): kinematics of the W muon
           and the N daughters, multiplicities of generated particles.

Output: <out>_events.parquet (one row per event), <out>_signal.parquet (one row per W muon and N daughter),
<out>_ue.json (pT / eta histograms of particles with hits, parent-code counts).

    python scripts/acts/hnl_census.py zenodo --sample hnl10_100 --splits train,val --out OUT [--workers N]
    python scripts/acts/hnl_census.py shards --dir SAMPLEDIR --out OUT [--workers N]
    python scripts/acts/hnl_census.py gen --dir GENDIR [--dir ...] --out OUT
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from hnl.data.npz import COL, load_event, particle_ids  # noqa: E402
from hnl.data.splits import read_split  # noqa: E402
from hnl.geometry.ids import decode_barcode, decode_geometry_id, is_pixel  # noqa: E402

MU_MASS = 0.1056584
MM_PER_GEV = 1000.0 / (0.299792458 * 2.0)     # helix radius per GeV of pT in 2 T
FITS = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived", "truth_fits",
                    "v1")
PT_EDGES = np.logspace(-1, 2.5, 36)
ETA_EDGES = np.linspace(-4, 4, 33)


def four(px, py, pz, m):
    return np.array([np.sqrt(px * px + py * py + pz * pz + m * m), px, py, pz])


def inv_mass(a, b):
    s = a + b
    return float(np.sqrt(max(s[0] ** 2 - s[1] ** 2 - s[2] ** 2 - s[3] ** 2, 0.0)))


def angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))


def true_d0(x, y, px, py, q):
    """|d0| of a helix in 2 T along +z through (x, y) with transverse momentum (px, py) and charge q."""
    pt = np.hypot(px, py)
    R = MM_PER_GEV * pt
    ux, uy = px / pt, py / pt
    s = 1.0 if q > 0 else -1.0       # q > 0 turns clockwise seen from +z: centre on the right of the motion
    cx, cy = x + s * R * uy, y - s * R * ux
    return abs(np.hypot(cx, cy) - R)


def event_census(ev: dict, event: int) -> tuple[dict, list[dict], dict]:
    P = ev["particles"]
    ids = particle_ids(P)
    bc = decode_barcode(ids)
    pdg = P[:, COL["particle_type"]].astype(np.int64)
    parent = P[:, COL["parent_pid"]].astype(np.int64)
    q = P[:, COL["q"]]
    gen = bc["gen"]
    vsec = bc["vsec"]
    pid = ev["pid"]
    up, cnt = np.unique(pid, return_counts=True)
    nh = dict(zip(up.tolist(), cnt.tolist()))
    vol = decode_geometry_id(ev["layers"])["volume"]
    pu, pc = np.unique(pid[is_pixel(vol)], return_counts=True)
    npix = dict(zip(pu.tolist(), pc.tolist()))
    hits = np.array([nh.get(int(i), 0) for i in ids])
    g0 = gen == 0
    e = {
        "event": event, "n_hits": len(pid), "n_rows": len(P), "n_phits": len(up),
        "n_gen0_rows": int(g0.sum()), "n_gen0_phits": int((g0 & (hits > 0)).sum()),
        "n_sec_phits": int((~g0 & (hits > 0)).sum()),
        "n_rows_k0s": int((g0 & (pdg == 310)).sum()), "n_rows_lambda": int((g0 & (np.abs(pdg) == 3122)).sum()),
        "n_rows_k0l": int((g0 & (pdg == 130)).sum()), "n_parent_k0s": int((g0 & (parent == 310)).sum()),
        "n_parent_lambda": int((g0 & (np.abs(parent) == 3122)).sum()),
        "n_sec_parent_nonzero": int((~g0 & (parent != 0)).sum()),
        "n_vsec_gen0": int(len(np.unique(vsec[g0]))),
    }
    sig = []
    w = np.flatnonzero((np.abs(parent) == 24) & (np.abs(pdg) == 13))
    d = np.flatnonzero(np.abs(parent) == 50)
    e["n_w"], e["n_d"] = len(w), len(d)
    e["w_sign"] = int(np.sign(parent[w[0]])) if len(w) else 0
    e["n_sign"] = int(np.sign(parent[d[0]])) if len(d) else 0
    e["d_all_muons"] = bool(len(d) and np.all(np.abs(pdg[d]) == 13))
    e["d_opposite"] = bool(len(d) == 2 and q[d[0]] * q[d[1]] < 0)
    e["d_same_vsec"] = bool(len(d) == 2 and vsec[d[0]] == vsec[d[1]])
    e["w_vsec"] = int(vsec[w[0]]) if len(w) else -1
    e["w_alone_vsec1"] = bool(len(w) == 1 and vsec[w[0]] == 1 and np.sum(g0 & (vsec == 1) & (np.abs(pdg) == 13)) == 1)
    if len(d):
        at = g0 & (vsec == vsec[d[0]])
        at[d] = False
        others = pdg[at]
        e["n_other_at_dvsec"] = int(len(others))
        e["other_at_dvsec"] = ",".join(str(x) for x in sorted(others.tolist()))
        e["parent_other_at_dvsec"] = ",".join(str(x) for x in sorted(parent[at].tolist()))
    else:
        e["n_other_at_dvsec"], e["other_at_dvsec"], e["parent_other_at_dvsec"] = 0, "", ""
    if len(w):
        k = w[0]
        e["n_other_at_wvsec"] = int(np.sum(g0 & (vsec == vsec[k])) - 1)
        e["w_photons_parent24"] = int(np.sum(g0 & (pdg == 22) & (np.abs(parent) == 24)))
    mom = P[:, [COL["px"], COL["py"], COL["pz"]]]
    for role, rows in (("w", w), ("d", d)):
        for k in rows:
            p3 = mom[k]
            sig.append({
                "event": event, "role": role, "particle_id": int(ids[k]), "pdg": int(pdg[k]), "q": float(q[k]),
                "parent": int(parent[k]), "pt": float(P[k, COL["pt"]]), "eta": float(P[k, COL["eta"]]),
                "phi": float(np.arctan2(p3[1], p3[0])), "p": float(np.linalg.norm(p3)), "n_hits": int(hits[k]),
                "n_pixel_hits": int(npix.get(int(ids[k]), 0)), "vsec": int(vsec[k]), "vt_end": float(P[k, COL["vt"]]),
                "r_end": float(np.hypot(P[k, COL["vx"]], P[k, COL["vy"]])), "z_end": float(P[k, COL["vz"]]),
            })
    if len(d) == 2:
        a, b = d
        fa, fb = four(*mom[a], MU_MASS), four(*mom[b], MU_MASS)
        e["m_mumu"] = inv_mass(fa, fb)
        e["open_angle"] = angle(mom[a], mom[b])
        ea, eb = P[a, COL["eta"]], P[b, COL["eta"]]
        dphi = np.angle(np.exp(1j * (np.arctan2(mom[a][1], mom[a][0]) - np.arctan2(mom[b][1], mom[b][0]))))
        e["dR"] = float(np.hypot(ea - eb, dphi))
        s = mom[a] + mom[b]
        e["pair_pt"] = float(np.hypot(s[0], s[1]))
    else:
        e["m_mumu"] = e["open_angle"] = e["dR"] = e["pair_pt"] = np.nan
    ue = {
        "pt": np.histogram(P[hits > 0, COL["pt"]], PT_EDGES)[0],
        "eta": np.histogram(P[(hits > 0) & np.isfinite(P[:, COL["eta"]]), COL["eta"]], ETA_EDGES)[0],
        "parent": collections.Counter(parent[g0].tolist()),
        "parent_photon": collections.Counter(parent[g0 & (pdg == 22)].tolist()),
        "pdg_gen0_hits": collections.Counter(np.abs(pdg[g0 & (hits > 0)]).tolist()),
    }
    return e, sig, ue


def add_fits(ev: dict, event: int, sig: list[dict], e: dict, field_sign: int = -1):
    """Truth fits with the function of derived/truth_fits/v1 (hnl.truth.fit_events)."""
    from hnl.truth.fit_events import fit_event, pair_vertices
    from hnl.truth.report import truth_d0
    df, _ = fit_event(ev, field_sign)
    df["d0"] = truth_d0(df)
    by = df.set_index("particle_id")
    for s in sig:
        if s["particle_id"] in by.index:
            r = by.loc[s["particle_id"]]
            s["d0"], s["r_first"] = float(r.d0), float(r.r_first)
        else:
            s["d0"], s["r_first"] = np.nan, np.nan
    v = pair_vertices(df, "label_hnl")
    e["rv"] = float(v.rv.iloc[0]) if len(v) and bool(v.ok.iloc[0]) else np.nan
    e["rv_any"] = float(v.rv.iloc[0]) if len(v) else np.nan


def merge_hist(acc, ue):
    for k, v in ue.items():
        if isinstance(v, collections.Counter):
            acc.setdefault(k, collections.Counter()).update(v)
        else:
            acc[k] = acc.get(k, 0) + v


# ---- inputs -------------------------------------------------------------------------------------------------------

def _zenodo_chunk(args):
    sample, events, refit = args
    E, S, U = [], [], {}
    for i in events:
        ev = load_event(sample, int(i))
        e, sig, ue = event_census(ev, int(i))
        if refit:
            add_fits(ev, int(i), sig, e)
        E.append(e)
        S += sig
        merge_hist(U, ue)
    return E, S, U


def _shard_chunk(args):
    directory, events = args
    from hnl.data.shards import iter_events_from
    E, S, U = [], [], {}
    for i, ev in iter_events_from(directory, events):
        e, sig, ue = event_census(ev, i)
        add_fits(ev, i, sig, e)
        E.append(e)
        S += sig
        merge_hist(U, ue)
    return E, S, U


def run_parallel(fn, chunks, workers):
    E, S, U = [], [], {}
    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(fn, chunks))
    else:
        results = [fn(c) for c in chunks]
    for e, s, u in results:
        E += e
        S += s
        merge_hist(U, u)
    return pd.DataFrame(E), pd.DataFrame(S), U


def zenodo(sample: str, splits: list[str], workers: int, refit: bool):
    events = np.sort(np.concatenate([read_split(sample, s) for s in splits]))
    chunks = [(sample, events[k:k + 200], refit) for k in range(0, len(events), 200)]
    E, S, U = run_parallel(_zenodo_chunk, chunks, workers)
    E["split"] = ""
    for s in splits:
        E.loc[E.event.isin(read_split(sample, s)), "split"] = s
    # truth fits of the full Zenodo sample (same function, derived/truth_fits/v1)
    fp = pd.read_parquet(os.path.join(FITS, f"particles_{sample}.parquet"),
                         columns=["event", "particle_id", "pdg", "fix_d0", "free_d0", "r_first"])
    fp = fp[fp.event.isin(events)]
    fp["d0_parquet"] = np.where(fp.pdg.abs() == 13, fp.fix_d0, fp.free_d0)
    S = S.merge(fp[["event", "particle_id", "d0_parquet", "r_first"]].rename(columns={"r_first": "r_first_parquet"}),
                on=["event", "particle_id"], how="left")
    v = pd.read_parquet(os.path.join(FITS, f"vertices_{sample}.parquet"))
    v = v[v.event.isin(events)]
    E = E.merge(pd.DataFrame({"event": v.event, "rv_parquet": np.where(v.ok, v.rv, np.nan), "rv_any_parquet": v.rv}),
                on="event", how="left")
    if not refit:
        S["d0"], S["r_first"] = S.d0_parquet, S.r_first_parquet
        E["rv"], E["rv_any"] = E.rv_parquet, E.rv_any_parquet
    return E, S, U


def shards(directory: str, workers: int):
    from hnl.data.shards import event_indices_from
    idx = event_indices_from(directory)
    chunks = [(directory, idx[k:k + 100].tolist()) for k in range(0, len(idx), 100)]
    E, S, U = run_parallel(_shard_chunk, chunks, workers)
    # generator truth from the sidecars
    tdir = directory.rstrip("/") + "_truth"
    G, Pi = [], []
    for f in sorted(glob.glob(os.path.join(tdir, "truth_*.npz"))):
        with np.load(f, allow_pickle=False) as z:
            G.append(pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith("g_")}))
            Pi.append(pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith("p_")}))
    G = pd.concat(G, ignore_index=True)
    Pi = pd.concat(Pi, ignore_index=True)
    G["rv_true"] = np.hypot(G.n_dec_x, G.n_dec_y)
    G["zv_true"] = G.n_dec_z
    G["n_eta"] = np.arcsinh(G.n_pz / np.hypot(G.n_px, G.n_py))
    G["L_true"] = np.sqrt((G.n_dec_x - G.n_prod_x) ** 2 + (G.n_dec_y - G.n_prod_y) ** 2 + (G.n_dec_z - G.n_prod_z) ** 2)
    G["betagamma"] = np.sqrt(G.n_px ** 2 + G.n_py ** 2 + G.n_pz ** 2) / G.n_m
    E = E.merge(G[["event", "rv_true", "zv_true", "n_eta", "L_true", "betagamma", "n_tau", "w_charge", "n_tries"]],
                on="event", how="left")
    Pi = Pi.rename(columns={"vx": "vx0", "vy": "vy0", "vz": "vz0", "px": "px0", "py": "py0", "pz": "pz0"})
    S = S.merge(Pi[["event", "particle_id", "vx0", "vy0", "vz0", "px0", "py0", "pz0"]], on=["event", "particle_id"],
                how="left")
    S["d0_true"] = [true_d0(x, y, px, py, qq) if np.isfinite(x) else np.nan
                    for x, y, px, py, qq in zip(S.vx0, S.vy0, S.px0, S.py0, S.q)]
    S["r_prod_true"] = np.hypot(S.vx0, S.vy0)
    return E, S, U


def gen_level(dirs: list[str]):
    """Generator level: one row per event from particles_gen + parents CSVs (+ hnl_gen truth when present)."""
    E, S = [], []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "event*-particles_gen.csv")))
        truth = {}
        for f in glob.glob(os.path.join(d, "hnl_gen_*.csv")):
            t = pd.read_csv(f)
            truth.update({int(r.event): r for r in t.itertuples()})
        for f in files:
            ev = int(os.path.basename(f)[5:14])
            g = pd.read_csv(f, dtype={"particle_id": np.uint64})
            pf = f.replace("particles_gen", "parents")
            par = pd.read_csv(pf, dtype={"particle_id": np.uint64}) if os.path.exists(pf) else None
            ids = g.particle_id.to_numpy(np.uint64)
            vsec = ((ids >> np.uint64(40)) & np.uint64(0xFFF)).astype(np.int64)
            p = np.sqrt(g.px ** 2 + g.py ** 2 + g.pz ** 2).to_numpy()
            pt = np.hypot(g.px, g.py).to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                eta = np.arcsinh(g.pz.to_numpy() / pt)
            chg = g.q.to_numpy() != 0
            e = {"dir": d, "event": ev, "n_gen": len(g), "n_gen_p05": int((p > 0.5).sum()),
                 "n_ch_pt05_eta25": int((chg & (pt > 0.5) & (np.abs(eta) < 2.5)).sum()),
                 "n_ch_p05": int((chg & (p > 0.5)).sum()), "n_k0s": int((g.particle_type == 310).sum()),
                 "n_lambda": int((g.particle_type.abs() == 3122).sum()), "n_vsec": int(len(np.unique(vsec))),
                 "vsec_max": int(vsec.max())}
            # barcode structure without the parents (also for ACTS-generator output): muons at vertex 1 and the
            # brief's rule (a vertex > 1 holding exactly two particles, opposite-charge muons)
            pdg_ = g.particle_type.to_numpy()
            q_ = g.q.to_numpy()
            e["n_mu_vsec1"] = int(np.sum((vsec == 1) & (np.abs(pdg_) == 13)))
            vv, vc = np.unique(vsec, return_counts=True)
            n_rule = 0
            for v_ in vv[(vc == 2) & (vv > 1)]:
                k_ = np.flatnonzero(vsec == v_)
                if np.all(np.abs(pdg_[k_]) == 13) and q_[k_[0]] * q_[k_[1]] < 0:
                    n_rule += 1
            e["n_rule_pairs"] = n_rule
            pdg = g.particle_type.to_numpy()
            if par is not None:
                parent = par.set_index("particle_id").loc[g.particle_id, "parent_pdg"].to_numpy()
                w = np.flatnonzero((np.abs(parent) == 24) & (np.abs(pdg) == 13))
                dd = np.flatnonzero(np.abs(parent) == 50)
                e["roles"] = "parents"
            else:
                # ACTS-generator output has no parents: the W muon is the muon at vertex 1, the N daughters the
                # brief's pair (when exactly one of each exists)
                parent = np.zeros(len(g), dtype=np.int64)
                w = np.flatnonzero((vsec == 1) & (np.abs(pdg) == 13))
                w = w if len(w) == 1 else w[:0]
                dd = np.array([], dtype=np.int64)
                if n_rule == 1:
                    for v_ in vv[(vc == 2) & (vv > 1)]:
                        k_ = np.flatnonzero(vsec == v_)
                        if np.all(np.abs(pdg_[k_]) == 13) and q_[k_[0]] * q_[k_[1]] < 0:
                            dd = k_
                e["roles"] = "rule"
            e.update({"n_w": len(w), "n_d": len(dd), "w_vsec": int(vsec[w[0]]) if len(w) else -1,
                      "w_alone_vsec1": bool(len(w) == 1 and vsec[w[0]] == 1 and
                                            np.sum((vsec == 1) & (np.abs(pdg) == 13)) == 1),
                      "d_alone_vsec": bool(len(dd) == 2 and vsec[dd[0]] == vsec[dd[1]] and
                                           np.sum(vsec == vsec[dd[0]]) == 2),
                      "n_parent_k0s": int((parent == 310).sum())})
            mom = g[["px", "py", "pz"]].to_numpy()
            if len(dd) == 2:
                a, b = dd
                e["m_mumu"] = inv_mass(four(*mom[a], MU_MASS), four(*mom[b], MU_MASS))
                e["open_angle"] = angle(mom[a], mom[b])
            for role, rows in (("w", w), ("d", dd)):
                for k in rows:
                    S.append({"dir": d, "event": ev, "role": role, "pdg": int(pdg[k]), "q": float(g.q.iloc[k]),
                              "parent": int(parent[k]), "pt": float(pt[k]), "eta": float(eta[k]),
                              "p": float(p[k]), "r_prod": float(np.hypot(g.vx.iloc[k], g.vy.iloc[k])),
                              "z_prod": float(g.vz.iloc[k]),
                              "d0_true": float(true_d0(g.vx.iloc[k], g.vy.iloc[k], g.px.iloc[k], g.py.iloc[k],
                                                       g.q.iloc[k]))})
            if ev in truth:
                t = truth[ev]
                pn = np.array([t.n_px, t.n_py, t.n_pz])
                e.update({"w_charge": int(t.w_charge), "rv_true": float(np.hypot(t.n_dec_x, t.n_dec_y)),
                          "zv_true": float(t.n_dec_z), "n_eta": float(np.arcsinh(pn[2] / np.hypot(pn[0], pn[1]))),
                          "L_true": float(np.sqrt((t.n_dec_x - t.n_prod_x) ** 2 + (t.n_dec_y - t.n_prod_y) ** 2
                                                  + (t.n_dec_z - t.n_prod_z) ** 2)),
                          "n_tau": float(t.n_tau), "betagamma": float(np.sqrt(t.n_px ** 2 + t.n_py ** 2 + t.n_pz ** 2)
                                                                       / t.n_m), "n_tries": int(t.n_tries)})
            E.append(e)
    return pd.DataFrame(E), pd.DataFrame(S), {}


def jsonable(U):
    out = {}
    for k, v in U.items():
        if isinstance(v, collections.Counter):
            out[k] = {str(a): int(b) for a, b in v.most_common()}
        else:
            out[k] = np.asarray(v).tolist()
    out["pt_edges"] = PT_EDGES.tolist()
    out["eta_edges"] = ETA_EDGES.tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["zenodo", "shards", "gen"])
    ap.add_argument("--sample")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--dir", action="append")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--refit-check", type=int, default=0,
                    help="zenodo: also refit the first N events and compare with derived/truth_fits/v1")
    a = ap.parse_args(argv)
    if a.mode == "zenodo":
        splits = a.splits.split(",")
        if "test" in splits:
            raise SystemExit("the test splits are not read")
        E, S, U = zenodo(a.sample, splits, a.workers, refit=False)
        if a.refit_check:
            ev0 = np.sort(np.concatenate([read_split(a.sample, s) for s in splits]))[: a.refit_check]
            e2, s2, _ = _zenodo_chunk((a.sample, ev0, True))
            s2 = pd.DataFrame(s2).merge(S[["event", "particle_id", "d0_parquet", "r_first_parquet"]],
                                        on=["event", "particle_id"])
            e2 = pd.DataFrame(e2).merge(E[["event", "rv_parquet"]], on="event")
            chk = {"n_particles": len(s2), "max_abs_d0_diff": float(np.nanmax(np.abs(s2.d0 - s2.d0_parquet))),
                   "n_d0_nan_mismatch": int((s2.d0.isna() != s2.d0_parquet.isna()).sum()),
                   "n_events": len(e2), "max_abs_rv_diff": float(np.nanmax(np.abs(e2.rv - e2.rv_parquet))),
                   "n_rv_nan_mismatch": int((e2.rv.isna() != e2.rv_parquet.isna()).sum())}
            with open(a.out + "_refit_check.json", "w") as f:
                json.dump(chk, f, indent=1)
            print("refit check", chk)
    elif a.mode == "shards":
        E, S, U = shards(a.dir[0], a.workers)
    else:
        E, S, U = gen_level(a.dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    E.to_parquet(a.out + "_events.parquet", index=False)
    S.to_parquet(a.out + "_signal.parquet", index=False)
    with open(a.out + "_ue.json", "w") as f:
        json.dump(jsonable(U), f)
    print(f"{a.mode}: {len(E)} events, {len(S)} signal particles -> {a.out}_*")


if __name__ == "__main__":
    main()
