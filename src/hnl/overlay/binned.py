"""Efficiencies of addendum H section 7 in bins (docs/preregistration_pileup.md section 7), on one overlay set: eps(P_P),
eps(P_D) and the efficiency of all gun targets, in C-full, C-1GeV and C-hs, at WP-A, WP-S and WP-R-P3, in
  - the bins of addendum G section 7: |d0|, r_v, first-hit radius, |d0| / r_first and pT (hnl.overlay.h1.BINS), and
    the number of other targets within Delta R < 0.05 of the initial direction (`neighbours` and NB_BINS, the
    definition of hnl.eval.addendum_g_summary repeated here so that no addendum-G module joins the bound code; counted
    over every target of the full event, gun or host, as addendum G counts every target of its events);
  - signed eta (ETA_BINS);
  - species (particle_type of the throughput side table ptype: e, mu, pi, K, p, other);
  - local pileup hit density: the mean number of host hits, noise included, per module over the modules that hold the
    target's hits, in C-full, in the tertile bins committed at H-T0.5 (configs/trackml/overlay_version_h_v1.json,
    density_bins; results/pileup_hp/h_t05_truth_density_bins.py computes the same quantity).
The H1 code bins only the P_P difference Delta in the first five variables. Intervals: 68 % Clopper-Pearson on k of n
per bin. No claim rests on these numbers.

Inputs: the truth of the stored set (hnl.overlay.h1.load_truth and gun_targets; ptype, particles and hits of the overlay
root) and the double-majority flag of each gun target per condition and working point from hnl.overlay.h1.load_outputs,
the H1 code's own reader: the grids' per_target rows after hnl.overlay.tracks.grid_provenance, and H-val in C-full
from its sealed directory through hnl.overlay.blind.sealed_per_target (the step-5 read). Grants: kind eval in every
condition (dry on the pilot), with --out inside each grant's out; on H-val and H-test also the set's launch record
(HNL_H_LAUNCH, decision D3).

Writes under --out only, refusing to overwrite: targets.parquet (one row per gun target: flags, variables and the dm
flag per condition and working point) and summary.json.

    python -m hnl.overlay.binned run --set hval|htest --out DIR
    python -m hnl.overlay.binned run --set pilot --out DIR --sealed DIR --expect SHA --grid-1gev DIR --grid-hs DIR
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from hnl.overlay import blind as B
from hnl.overlay import grant as G
from hnl.overlay import h1 as H

WPS = ("WP-A", "WP-S", "WP-R-P3")
PILOT_WPS = ("WP-R-P3",)                     # the pilot's grids hold the dry configurations only
CONDS = ("full", "1gev", "hs")
POPULATIONS = ("P_P", "P_D", "all gun targets")
ETA_BINS = [-4.0, -2.5, -1.5, -0.8, 0.0, 0.8, 1.5, 2.5, 4.0]
NB_BINS = [0, 1, 2, 3, 5, np.inf]                     # = hnl.eval.addendum_g_summary.NB_BINS
SPECIES = {11: "e", 13: "mu", 211: "pi", 321: "K", 2212: "p"}
DR_NEIGHBOUR = 0.05


def _density_edges(path: str = os.path.join(B.REPO, "configs", "trackml", "overlay_version_h_v1.json")) -> list:
    with open(path) as fh:
        e = json.load(fh)["density_bins"]["edges"]
    return [np.inf if x is None else float(x) for x in e]


def variables(density_edges=None) -> dict:
    """{name: (column, edges or None for a category)}."""
    v = {k: (col, list(edges)) for k, (col, edges) in H.BINS.items()}
    v[f"other targets within dR < {DR_NEIGHBOUR}"] = ("n_neigh", list(NB_BINS))
    v["signed eta"] = ("eta", ETA_BINS)
    v["species"] = ("species", None)
    v["local pileup hit density [host hits per module]"] = ("density", density_edges or _density_edges())
    return v


# ------------------------------------------------------------------------------------------------ truth
def neighbours(t: pd.DataFrame) -> np.ndarray:
    """Per row: other targets of the same event within Delta R < 0.05 of the initial momentum direction (columns event,
    p_px, p_py, p_pz, is_target); the code of hnl.eval.addendum_g_summary.neighbours."""
    out = np.zeros(len(t), dtype=np.int64)
    p = np.hypot(t.p_px.to_numpy(), t.p_py.to_numpy())
    eta = np.arcsinh(t.p_pz.to_numpy() / np.maximum(p, 1e-12))
    phi = np.arctan2(t.p_py.to_numpy(), t.p_px.to_numpy())
    tgt = t.is_target.to_numpy(bool)
    ev = t.event.to_numpy()
    order = np.argsort(ev, kind="stable")
    bounds = np.flatnonzero(np.r_[True, ev[order][1:] != ev[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        ti = idx[tgt[idx]]
        if len(ti) < 2:
            continue
        de = eta[idx][:, None] - eta[ti][None, :]
        dp = np.remainder(phi[idx][:, None] - phi[ti][None, :] + np.pi, 2 * np.pi) - np.pi
        close = (de ** 2 + dp ** 2) < DR_NEIGHBOUR ** 2
        out[idx] = close.sum(1) - tgt[idx].astype(int)
    return out


def truth(root: str, events) -> pd.DataFrame:
    """Every gun target of the events with the H1 flags and the variables of the module docstring (truth only)."""
    from hnl.trackml.convert import read_table
    parts, ghits = H.load_truth(root, events)
    g = H.gun_targets(parts, ghits)
    g = g[g.target.astype(bool)].reset_index(drop=True)
    ev = sorted(int(e) for e in events)
    sp, dens, neigh = [], [], []
    for a in range(0, len(ev), 20):
        part = ev[a:a + 20]
        tg = g[g.event.isin(part)][["event", "particle_id"]]
        y = read_table("ptype", part, ["event", "particle_id", "particle_type"], root=root)
        sp.append(tg.merge(y, on=["event", "particle_id"], how="left"))
        h = read_table("hits", part, ["event", "volume", "layer", "module", "pid"], root=root)
        gun = B.is_gun(h.pid.to_numpy(np.int64))
        key = ["event", "volume", "layer", "module"]
        host = h[~gun].groupby(key, sort=False).size().rename("n_host").reset_index()
        th = h[gun].merge(tg, left_on=["event", "pid"], right_on=["event", "particle_id"])
        mods = th[["event", "particle_id", "volume", "layer", "module"]].drop_duplicates()
        mods = mods.merge(host, on=key, how="left").fillna({"n_host": 0})
        dens.append(mods.groupby(["event", "particle_id"]).n_host.mean().rename("density").reset_index())
        p = read_table("particles", part, ["event", "particle_id", "px", "py", "pz", "is_target"], root=root)
        p = p.rename(columns={"px": "p_px", "py": "p_py", "pz": "p_pz"}).reset_index(drop=True)
        p["n_neigh"] = neighbours(p)
        neigh.append(p[["event", "particle_id", "n_neigh"]].merge(tg, on=["event", "particle_id"]))
    keys = ["event", "particle_id"]
    for extra in (pd.concat(sp), pd.concat(dens), pd.concat(neigh)):
        g = g.merge(extra.astype({"event": np.int64, "particle_id": np.int64}), on=keys, how="left",
                    validate="one_to_one")
    if g.particle_type.isna().any() or g.density.isna().any() or g.n_neigh.isna().any():
        raise ValueError("a gun target without its particle_type, hits or particles row")
    g["species"] = g.particle_type.abs().astype(np.int64).map(SPECIES).fillna("other")
    g["population_pp"] = g.pp.astype(bool)
    g["population_pd"] = g.pd.astype(bool)
    return g


# ------------------------------------------------------------------------------------------------ statistics
def cp68(k, n) -> tuple[float, float]:
    from scipy.stats import beta
    if n == 0:
        return float("nan"), float("nan")
    a = (1.0 - 0.6827) / 2.0
    lo = float(beta.ppf(a, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1.0 - a, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def _cell(found: np.ndarray) -> dict:
    n, k = int(len(found)), int(found.sum())
    lo, hi = cp68(k, n)
    return {"n": n, "k": k, "eff": k / n if n else float("nan"), "lo": lo, "hi": hi}


def binned(t: pd.DataFrame, dm_col: str, var: tuple) -> list[dict]:
    col, edges = var
    found = t[dm_col].to_numpy(bool)
    if edges is None:
        cats = t[col].astype(str).to_numpy()
        return [{"bin": c, **_cell(found[cats == c])} for c in sorted(set(cats))]
    x = t[col].to_numpy(np.float64)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        out.append({"bin": f"[{lo:g}, {hi:g})", **_cell(found[m])})
    return out


def summarize(t: pd.DataFrame, wps, density_edges=None) -> dict:
    pops = {"P_P": t.population_pp.to_numpy(bool), "P_D": t.population_pd.to_numpy(bool),
            "all gun targets": np.ones(len(t), bool)}
    vs = variables(density_edges)
    out = {}
    for wp in wps:
        for c in CONDS:
            col = f"dm_{c}_{wp}"
            r = {}
            for pop, m in pops.items():
                q = t[m]
                r[pop] = {"all": _cell(q[col].to_numpy(bool)), "bins": {name: binned(q, col, v) for name, v in vs.items()}}
            out.setdefault(wp, {})[c] = r
    return out


# ------------------------------------------------------------------------------------------------ run
def run(set_name: str, out: str, sealed: str | None = None, expect: str | None = None, grid_1gev: str | None = None,
        grid_hs: str | None = None, root: str | None = None, repo: str = B.REPO) -> dict:
    from hnl.overlay import hstats as HS
    split = B.load_split()
    k = B.set_index(set_name, split)
    if set_name not in ("hval", "htest", "pilot"):
        raise PermissionError(f"no section-7 efficiencies for set {set_name!r}")
    if k == 9:
        if B._under(out, B.RUNS_H):
            raise PermissionError("placeholder outputs never go under the production runs root")
        if not (sealed and expect and grid_1gev and grid_hs):
            raise PermissionError("the pilot takes --sealed, --expect, --grid-1gev and --grid-hs (placeholders)")
        grids = {"1gev": grid_1gev, "hs": grid_hs}
    else:
        if any(x is not None for x in (sealed, grid_1gev, grid_hs, root)):
            raise PermissionError("production sets read the grids of hnl.overlay.chain paths and, on H-val, their "
                                  "sealed directory only")
        from hnl.overlay.chain import paths
        grids = {c: paths(set_name, c)["grid"] for c in (("1gev", "hs") if k == 0 else CONDS)}
        sealed = B.sealed_dir("hval", "full", B.RUNS_H) if k == 0 else None
    if os.path.exists(os.path.join(out, "summary.json")) or os.path.exists(os.path.join(out, "targets.parquet")):
        raise SystemExit(f"{out} holds the binned efficiencies already: refusing to overwrite them")
    events = H.set_events(set_name, split)
    for c in CONDS:
        G.check(events, "dry" if k == 9 else "eval", c, out=out)
    if k in (0, 2):
        bad = B.check_launch(set_name, os.environ.get("HNL_H_LAUNCH"), B.runs_root(repo), repo)
        if bad:
            raise PermissionError(f"{set_name}: refused by its launch record: {bad[:5]}")
    root = B.overlay_root(k, root) or B.DERIVED_OVERLAY
    t = truth(root, events)
    wps = PILOT_WPS if k == 9 else WPS
    sources = {}
    for wp in wps:
        outs, src = H.load_outputs(set_name, wp, grids, sealed, expect, root if k == 9 else None, split)
        for c in CONDS:
            t[f"dm_{c}_{wp}"] = HS.attach_dm(t, outs[c], "event", "particle_id", f"{set_name} {c} {wp}",
                                             t.target.astype(bool)).to_numpy()
        sources[wp] = src
    rec = {"set": set_name, "k": k, "events": len(events), "root": root, "working_points": list(wps),
           "conditions": list(CONDS), "populations": list(POPULATIONS),
           "variables": {k_: {"column": c, "edges": e} for k_, (c, e) in variables().items()},
           "n_gun_targets": int(len(t)), "n_pp": int(t.population_pp.sum()), "n_pd": int(t.population_pd.sum()),
           "sources": sources, "summary": summarize(t, wps),
           "git": B._git_head(), "code_sha256": B.sha256_file(os.path.abspath(__file__)),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(out, exist_ok=True)
    keep = ["event", "particle_id", "llp", "iso", "pp", "pd", "soft_sibling", "region", "abs_d0", "rv", "r_first",
            "ratio", "pt", "eta", "particle_type", "species", "density", "n_neigh"]
    t[[c for c in keep if c in t.columns] + [c for c in t.columns if c.startswith("dm_")]].to_parquet(
        os.path.join(out, "targets.parquet"), index=False)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=_json_default)
    return rec


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON: {type(o)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--out", required=True)
    for x in ("sealed", "expect", "grid-1gev", "grid-hs", "root"):
        r.add_argument(f"--{x}", default=None)
    a = ap.parse_args(argv)
    rec = run(a.set, a.out, a.sealed, a.expect, a.grid_1gev, a.grid_hs, a.root)
    print(json.dumps({k: rec[k] for k in ("set", "events", "n_gun_targets", "n_pp", "n_pd")}))


if __name__ == "__main__":
    main()
