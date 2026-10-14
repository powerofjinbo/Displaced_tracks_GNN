"""Gate C1 of the pileup overlay (docs/pileup_overlay_plan.md section 10): cells and resolution of the digitiser
(hnl.overlay.digitise) against TrackML training hits, per volume in bins of local incidence. Truth only; no model
output is read.

Samples.
- reference: hits of TrackML throughput calibration events (training; derived/tp_v1 hits and truth tables).
- gun: pions and muons simulated over TrackML's angular range (`particles`, then ActsExampleFatrasGeneric of the
  patched build with the TrackML-like field map, slurm/overlay/c1.sbatch), digitised with the per-cell charge factor.
- replay (diagnostic, no pass rule): the truth points of the reference hits digitised again, which isolates the
  digitiser from the simulation.

Per hit: tan_u = du/dw and tan_v = dv/dw of the true direction in the TrackML module frame, |tan| = tan of the
incidence angle, the cell features, and the residual of the reported position against the true point moved along the
true direction onto the hit's plane (du, dv). Quantities and their binning variable (bins `EDGES`, signed tan_u in the
barrel volumes, where the Lorentz drift breaks the symmetry):
  span0, MAD(du), mean(du) vs tan_u;  span1, MAD(dv) (pixels) vs |tan_v|;  charge x cos(incidence) (pixels) vs |tan|.
Local incidence has two angles, and the gun fills them differently from TrackML (more forward and displaced tracks).
The comparison is therefore made at matched incidence: in every bin the gun hits are reweighted to TrackML's joint
distribution of the binning variable (reference quartiles in the bin) and the other angle (the `EDGES` bins of |tan_v|,
or of tan_u for the v quantities), on the cells both samples populate (`match_weights`). The unmatched comparison
(plain bins of one variable) is kept as a record: it fails where the second angle differs (results/pileup_hp).
Pass rules of section 10 (fixed at design v2):
- every bin with at least 200 hits in both samples (on the common support): mean span0 and span1 within 10 %, pixel
  charge x cos within 5 %, MAD(du) within 20 % (all volumes), MAD(dv) within 20 % (pixels);
- mean u bias within 1 um: the bin means of du averaged with the reference's bin counts as weights (counted bins);
- effective thickness within 0.04 mm: t_eff = pitch x slope of a straight-line fit of span against the drift width
  variable (|tan_u - tan(LA)| for span0, |tan_v| for the pixel span1) over |x| < 3 (span0) and < 8 (span1).
A failure of the MAD rule is fixed by the smear of section 4.4 (an emulation change).

    python -m hnl.overlay.c1 particles --out DIR --events N --seed S [--per-event 50]
    python -m hnl.overlay.c1 run --sim-dir DIR --lookup MAP.csv --out-json J [--ref-events 40] [--seed S]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.overlay import digitise as D

TP = os.path.join(D.EXT, "derived", "tp_v1")
SPLIT_TP = os.path.join(D.REPO, "configs", "trackml", "split_tp_v1.json")
VOLUMES = (7, 8, 9, 12, 13, 14, 16, 17, 18)
_ABS = [0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4, 3.2, 5, 8, 16, np.inf]
EDGES = {"abs": np.array(_ABS), "signed": np.array([-x for x in _ABS[:0:-1]] + _ABS)}
MIN_HITS = 200
RULES = {"span0": ("rel", 0.10), "span1": ("rel", 0.10), "charge_cos": ("rel", 0.05), "mad_u": ("rel", 0.20),
         "mad_v": ("rel", 0.20)}
BIAS_TOL_MM = 0.001
TEFF_TOL_MM = 0.04
TEFF_RANGE = {"span0": 3.0, "span1": 8.0}
MASS = {13: 0.105658, 211: 0.139570}
COLS = ["particle_id", "particle_type", "process", "vx", "vy", "vz", "vt", "px", "py", "pz", "m", "q"]


# ------------------------------------------------------------------------------------------------ hit tables
def mad(x) -> float:
    x = np.asarray(x, np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def features(volume, tan_u, tan_v, span0, span1, charge, du, dv) -> pd.DataFrame:
    """The per-hit table the comparison reads."""
    tan_u, tan_v = np.asarray(tan_u, np.float64), np.asarray(tan_v, np.float64)
    tan = np.hypot(tan_u, tan_v)
    return pd.DataFrame({"volume": np.asarray(volume, np.int64), "tan_u": tan_u, "abs_tan_v": np.abs(tan_v),
                         "abs_tan": tan, "span0": np.asarray(span0, np.float64), "span1": np.asarray(span1, np.float64),
                         "charge_cos": np.asarray(charge, np.float64) / np.sqrt(1 + tan * tan),
                         "du": np.asarray(du, np.float64), "dv": np.asarray(dv, np.float64)})


def calib_events(n: int) -> list[int]:
    """The first n calibration events of split_tp_v1 (training events with stored truth)."""
    with open(SPLIT_TP) as fh:
        s = json.load(fh)
    calib = sorted(int(e) for e in s["calib"])
    held = set(int(e) for e in s["val"]) | set(int(e) for e in s["test"])
    ev = calib[:n]
    if set(ev) & held or not set(ev) <= set(int(e) for e in s["train"]):
        raise RuntimeError("calibration events must be training events")
    return ev


def trackml_hits(events, modules: D.Modules, root: str = TP) -> pd.DataFrame:
    """Particle hits of TrackML events with their truth: the columns of `features` plus the truth point and
    momentum and the module ids (for the replay)."""
    from hnl.trackml.convert import read_table
    H = []
    for k in range(0, len(events), 10):
        part = list(events[k:k + 10])
        h = read_table("hits", part, ["event", "row", "x", "y", "z", "volume", "layer", "module", "pid", "ncells",
                                      "charge", "span0", "span1"], root=root)
        t = read_table("truth", part, root=root)
        H.append(h[h.pid != 0].merge(t, on=["event", "row"], how="inner"))
    h = pd.concat(H, ignore_index=True)
    row = modules.index(h.volume, h.layer, h.module)
    if (row < 0).any():
        raise RuntimeError("TrackML hit on a module missing from detectors.csv")
    R, c = modules.R[row], modules.c[row]
    lh = np.einsum("nij,ni->nj", R, h[["x", "y", "z"]].to_numpy(np.float64) - c)
    lt = np.einsum("nij,ni->nj", R, h[["tx", "ty", "tz"]].to_numpy(np.float64) - c)
    ld = np.einsum("nij,ni->nj", R, h[["tpx", "tpy", "tpz"]].to_numpy(np.float64))
    ok = np.abs(ld[:, 2]) > D.MIN_COS * np.linalg.norm(ld, axis=1)
    h, lh, lt, ld = h[ok].reset_index(drop=True), lh[ok], lt[ok], ld[ok]
    tu, tv = ld[:, 0] / ld[:, 2], ld[:, 1] / ld[:, 2]
    s = lh[:, 2] - lt[:, 2]
    du = lh[:, 0] - (lt[:, 0] + s * tu)
    dv = lh[:, 1] - (lt[:, 1] + s * tv)
    f = features(h.volume, tu, tv, h.span0, h.span1, h.charge, du, dv)
    for k in ("event", "volume", "layer", "module", "pid", "tx", "ty", "tz", "tpx", "tpy", "tpz"):
        f["t_" + k] = h[k].to_numpy()
    return f


def digitised_features(dig: D.Digitised) -> pd.DataFrame:
    h = dig.hits
    return features(h.volume_id, h.tan_u, h.tan_v, h.span0, h.span1, h.charge, h.u - h.u_true, h.v - h.v_true)


def replay(ref: pd.DataFrame, modules: D.Modules, seed: int | None) -> pd.DataFrame:
    """Digitise the truth points of the reference hits again, per event with the charge factors of
    event_rng(seed, event); seed None: bare path lengths (no per-cell factor)."""
    out = []
    for e, g in ref.groupby("t_event", sort=True):
        sim = pd.DataFrame({k: g["t_" + k].to_numpy() for k in ("tx", "ty", "tz", "tpx", "tpy", "tpz")})
        sim["particle_id"] = g.t_pid.to_numpy()
        rng = None if seed is None else D.event_rng(seed, int(e))
        dig = D.digitise_on_modules(sim, g.t_volume.to_numpy(), g.t_layer.to_numpy(), g.t_module.to_numpy(), modules,
                                    rng=rng)
        out.append(digitised_features(dig))
    return pd.concat(out, ignore_index=True)


def sim_events(sim: str, events=None):
    """(event number, sim hits) of a directory of CsvSimHitWriter files or of a file packed by
    hnl.overlay.fatras_io (column event)."""
    if sim.endswith(".parquet"):
        t = pd.read_parquet(sim)
        for e, g in list(t.groupby("event", sort=True))[:events]:
            yield int(e), g.reset_index(drop=True)
        return
    for f in sorted(f for f in os.listdir(sim) if f.endswith("-hits.csv"))[:events]:
        yield D.event_number(f), D.read_sim_event(os.path.join(sim, f))


def sim_hits(sim: str, lookup, modules: D.Modules, seed: int, events=None) -> tuple[pd.DataFrame, dict]:
    """Digitise every event of `sim` (see sim_events); the per-hit features and the summed digitiser stats."""
    out, stats, n = [], {}, 0
    for e, sh in sim_events(sim, events):
        dig = D.digitise(sh, lookup, modules, rng=D.event_rng(seed, e))
        out.append(digitised_features(dig))
        for k, v in dig.stats.items():
            stats[k] = max(stats.get(k, 0), v) if k.startswith("max") else stats.get(k, 0) + v
        n += 1
    stats["events"] = n
    return pd.concat(out, ignore_index=True), stats


# ------------------------------------------------------------------------------------------------ comparison
QUANTITIES = {  # name: (column, statistic, binning axis, pixels only)
    "span0": ("span0", "mean", "u", False), "mad_u": ("du", "mad", "u", False), "bias_u": ("du", "mean", "u", False),
    "span1": ("span1", "mean", "v", False), "mad_v": ("dv", "mad", "v", True),
    "charge_cos": ("charge_cos", "mean", "t", True)}
OTHER_AXIS = {"u": "v", "v": "u", "t": "v"}
N_SUB = 4                       # sub-bins of the binning variable (reference quartiles) for the matching


def axis_values(f: pd.DataFrame, axis: str, volume: int) -> tuple[np.ndarray, np.ndarray]:
    """Binning variable and bin edges: tan_u (signed in the barrel volumes), |tan_v| or |tan|."""
    if axis == "u":
        if volume in D.BARREL_VOLUMES:
            return f.tan_u.to_numpy(), EDGES["signed"]
        return np.abs(f.tan_u.to_numpy()), EDGES["abs"]
    return (f.abs_tan_v.to_numpy() if axis == "v" else f.abs_tan.to_numpy()), EDGES["abs"]


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])


def wstat(x: np.ndarray, w: np.ndarray, stat: str) -> float:
    if not len(x) or w.sum() <= 0:
        return float("nan")
    if stat == "mean":
        return float(np.average(x, weights=w))
    m = wquantile(x, w, 0.5)
    return 1.4826 * wquantile(np.abs(x - m), w, 0.5)


def match_weights(xr, yr, xt, yt, y_edges) -> tuple[np.ndarray, np.ndarray]:
    """Weights that give the test hits of one bin the reference's joint distribution of (x, y): cells of N_SUB
    reference-quantile sub-bins of x (bounded by the reference's smallest and largest x) times the y bins. Cells
    without hits on both sides get weight 0 on both sides (common support); reference weights are 1 on the support."""
    qe = np.quantile(xr, np.linspace(0, 1, N_SUB + 1)) if len(xr) else np.zeros(2)
    qe[-1] = np.nextafter(qe[-1], np.inf)
    ny = len(y_edges) + 1
    cr = np.searchsorted(qe, xr, side="right") * ny + np.digitize(yr, y_edges)
    ct = np.searchsorted(qe, xt, side="right") * ny + np.digitize(yt, y_edges)
    m = (N_SUB + 2) * ny
    nr, nt = np.bincount(cr, minlength=m), np.bincount(ct, minlength=m)
    common = (nr > 0) & (nt > 0)
    wr = common[cr].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        wt = np.where(common[ct], nr[ct] / np.maximum(nt[ct], 1), 0.0)
    return wr, wt


def bin_rows(ref: pd.DataFrame, test: pd.DataFrame, volume: int, q: str, matched: bool) -> list[dict]:
    """One row per bin of the binning variable: hits and statistic of both samples, and with `matched` the reference
    share on the common support (coverage)."""
    col, stat, axis, _ = QUANTITIES[q]
    xr, edges = axis_values(ref, axis, volume)
    xt, _ = axis_values(test, axis, volume)
    yr, y_edges = axis_values(ref, OTHER_AXIS[axis], volume)
    yt, _ = axis_values(test, OTHER_AXIS[axis], volume)
    br, bt = np.digitize(xr, edges) - 1, np.digitize(xt, edges) - 1
    vr, vt = ref[col].to_numpy(), test[col].to_numpy()
    rows = []
    for k in range(len(edges) - 1):
        ir, it = np.flatnonzero(br == k), np.flatnonzero(bt == k)
        if not len(ir) and not len(it):
            continue
        if matched and len(ir) and len(it):
            wr, wt = match_weights(xr[ir], yr[ir], xt[it], yt[it], y_edges)
        else:
            wr, wt = np.ones(len(ir)), np.ones(len(it))
        rows.append({"lo": float(edges[k]), "hi": float(edges[k + 1]),
                     "n_ref": int((wr > 0).sum()), "ref": wstat(vr[ir], wr, stat),
                     "n_test": int((wt > 0).sum()), "test": wstat(vt[it], wt, stat),
                     "coverage": float(wr.mean()) if len(wr) else float("nan")})
    return rows


def t_eff(g: pd.DataFrame, volume: int) -> dict:
    """Effective thickness [mm] from the slope of span against the drift-width variable (module docstring)."""
    pu = {7: 0.05, 8: 0.05, 9: 0.05, 12: 0.08, 13: 0.08, 14: 0.08, 16: 0.12, 17: 0.12, 18: 0.12}[volume]
    tan_la = np.tan(D.LORENTZ_ANGLE.get(volume, 0.0))
    out = {}
    fits = [("span0", np.abs(g.tan_u.to_numpy() - tan_la), pu)]
    if volume in D.PIXEL_VOLUMES:
        fits.append(("span1", g.abs_tan_v.to_numpy(), 0.05625))
    for name, x, pitch in fits:
        sel = x < TEFF_RANGE[name]
        if sel.sum() > 100:
            slope, icpt = np.polyfit(x[sel], g[name].to_numpy()[sel], 1)
            out[name] = {"t_eff_mm": float(slope * pitch), "intercept": float(icpt), "n": int(sel.sum())}
    return out


def compare(ref: pd.DataFrame, test: pd.DataFrame, matched: bool = True) -> dict:
    """C1 rules (module docstring) for two per-hit tables. matched: the test hits of every bin are reweighted to the
    reference's joint distribution of both incidence angles in the bin (`match_weights`), so that the comparison is
    made at the same local incidence; unmatched: plain bins of the one binning variable."""
    res = {"matched": matched, "volumes": {}, "n_bins_counted": 0, "n_bins_failed": 0}
    all_pass = True
    for v in VOLUMES:
        rv, tv = ref[ref.volume == v], test[test.volume == v]
        if not len(rv) or not len(tv):
            continue
        out = {"bins": {}, "pass": True}
        for q, (_, _, _, pix_only) in QUANTITIES.items():
            if pix_only and v not in D.PIXEL_VOLUMES:
                continue
            rows = bin_rows(rv, tv, v, q, matched)
            for r in rows:
                r["counted"] = bool(r["n_ref"] >= MIN_HITS and r["n_test"] >= MIN_HITS)
            if q == "bias_u":
                cnt = [r for r in rows if r["counted"]]
                if cnt:
                    w = np.array([r["n_ref"] for r in cnt], float)
                    b_ref = float(np.sum(w * [r["ref"] for r in cnt]) / w.sum())
                    b_test = float(np.sum(w * [r["test"] for r in cnt]) / w.sum())
                    ok = abs(b_test - b_ref) <= BIAS_TOL_MM
                    out["bias_u"] = {"ref_um": b_ref * 1e3, "test_um": b_test * 1e3,
                                     "diff_um": (b_test - b_ref) * 1e3, "n_bins": len(cnt), "pass": bool(ok)}
                    out["pass"] &= bool(ok)
                continue
            _, tol = RULES[q]
            for r in rows:
                r["rel_diff"] = (r["test"] - r["ref"]) / r["ref"] if r["counted"] and r["ref"] != 0 else float("nan")
                r["pass"] = bool(abs(r["rel_diff"]) <= tol) if r["counted"] else None
                if r["counted"]:
                    res["n_bins_counted"] += 1
                    res["n_bins_failed"] += int(not r["pass"])
                    out["pass"] &= r["pass"]
            out["bins"][q] = rows
        te = {}
        a, b = t_eff(rv, v), t_eff(tv, v)
        for name in ("span0", "span1"):
            if name in a and name in b:
                ok = abs(b[name]["t_eff_mm"] - a[name]["t_eff_mm"]) <= TEFF_TOL_MM
                te[name] = {"ref_mm": a[name]["t_eff_mm"], "test_mm": b[name]["t_eff_mm"],
                            "diff_mm": b[name]["t_eff_mm"] - a[name]["t_eff_mm"], "pass": bool(ok)}
                out["pass"] &= bool(ok)
        out["t_eff"] = te
        out["summary"] = summary(out)
        res["volumes"][v] = out
        all_pass &= out["pass"]
    res["pass"] = bool(all_pass)
    return res


def summary(vol: dict) -> dict:
    """Per quantity: counted bins, failed bins, the largest |relative difference|, the reference-weighted ratio and
    the smallest coverage of a counted bin."""
    s = {}
    for q, bins in vol["bins"].items():
        c = [b for b in bins if b["counted"]]
        if not c:
            s[q] = {"counted": 0}
            continue
        w = np.array([b["n_ref"] for b in c], float)
        s[q] = {"counted": len(c), "failed": int(sum(not b["pass"] for b in c)),
                "max_abs_rel_diff": float(max(abs(b["rel_diff"]) for b in c)),
                "weighted_ratio": float(np.sum(w * [b["test"] for b in c]) / np.sum(w * [b["ref"] for b in c])),
                "min_coverage": float(min(b["coverage"] for b in c))}
    return s


def overall(f: pd.DataFrame) -> dict:
    """Per volume, without incidence bins: hits, MAD of du and dv [um] and mean spans (descriptive)."""
    return {int(v): {"n": int(len(g)), "mad_u_um": mad(g.du) * 1e3, "mad_v_um": mad(g.dv) * 1e3,
                     "span0": float(g.span0.mean()), "span1": float(g.span1.mean())}
            for v, g in f.groupby("volume")}


def bins_frame(results: dict) -> pd.DataFrame:
    """All bins of several comparisons {sample: compare(...)} as one table."""
    rows = []
    for name, res in results.items():
        for v, r in res["volumes"].items():
            for q, bins in r["bins"].items():
                for b in bins:
                    rows.append({"sample": name, "volume": int(v), "quantity": q, **b})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ gun particles
def make_particles(out: str, events: int, seed: int, per_event: int = 50) -> dict:
    """Pions and muons (+-211, +-13, equal shares) over TrackML's angular range: eta ~ U(-4, 4), phi ~ U(0, 2 pi),
    pT ~ logU(0.15, 20) GeV. Half start at a prompt vertex (x = y = 0, z ~ N(0, 55 mm)), half at a displaced vertex
    placed as the gun of spec v1 places its LLP decays (r_v ~ U(1, 800) mm, z_v = z_PV + r_v sinh(eta_LLP) with
    eta_LLP ~ U(-3, 3), redrawn while |z_v| > 2900 mm), with an independent direction. ACTS CsvParticleReader layout."""
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    pdgs = np.array([211, -211, 13, -13])
    for e in range(events):
        n = per_event
        pdg = pdgs[rng.integers(0, 4, n)]
        q = np.where(np.isin(pdg, (211, -13)), 1, -1)
        m = np.array([MASS[abs(int(p))] for p in pdg])
        pt = np.exp(rng.uniform(np.log(0.15), np.log(20.0), n))
        eta = rng.uniform(-4, 4, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        zpv = rng.normal(0, 55.0)
        vx, vy, vz = np.zeros(n), np.zeros(n), np.full(n, zpv)
        disp = np.arange(n) >= n // 2
        for i in np.flatnonzero(disp):
            while True:
                rv, el, fv = rng.uniform(1, 800), rng.uniform(-3, 3), rng.uniform(0, 2 * np.pi)
                zv = zpv + rv * np.sinh(el)
                if abs(zv) <= 2900:
                    break
            vx[i], vy[i], vz[i] = rv * np.cos(fv), rv * np.sin(fv), zv
        pid = (np.uint64(1) << np.uint64(52)) | (np.arange(1, n + 1, dtype=np.uint64) << np.uint64(24))
        df = pd.DataFrame({"particle_id": pid, "particle_type": pdg, "process": 0, "vx": vx, "vy": vy, "vz": vz,
                           "vt": 0.0, "px": pt * np.cos(phi), "py": pt * np.sin(phi), "pz": pt * np.sinh(eta),
                           "m": m, "q": q})
        df[COLS].to_csv(os.path.join(out, f"event{e:09d}-particles.csv"), index=False, float_format="%.10g")
    return {"events": events, "per_event": per_event, "seed": seed}


# ------------------------------------------------------------------------------------------------ driver
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_run(a):
    t0 = time.time()
    mods = D.Modules.load()
    lk = D.load_lookup(a.lookup)
    ev = calib_events(a.ref_events)
    ref = trackml_hits(ev, mods)
    print(f"[c1] reference: {len(ref)} hits of {len(ev)} calibration events ({time.time() - t0:.0f} s)", flush=True)
    out = {"plan": "docs/pileup_overlay_plan.md section 10, C1 (design v2, bd7e357)",
           "reference": {"events": ev, "n_hits": int(len(ref)), "root": TP},
           "lookup": {"path": os.path.abspath(a.lookup), "sha256": sha256(a.lookup)},
           "detectors_sha256": sha256(D.DETECTORS), "cell_sigma": D.CELL_SIGMA, "seed": a.seed,
           "code_sha256": {os.path.basename(f): sha256(f) for f in (D.__file__, __file__)},
           "rules": {"min_hits_per_bin": MIN_HITS, "rel": RULES, "bias_tol_mm": BIAS_TOL_MM,
                     "t_eff_tol_mm": TEFF_TOL_MM, "t_eff_range": TEFF_RANGE, "match_sub_bins": N_SUB},
           "edges": {k: [float(x) for x in e] for k, e in EDGES.items()}}
    rep, rep0 = replay(ref, mods, a.seed), replay(ref, mods, None)
    cmp = {"replay": compare(ref, rep), "replay_no_cell_factor": compare(ref, rep0)}
    out["overall"] = {"reference": overall(ref), "replay": overall(rep), "replay_no_cell_factor": overall(rep0)}
    print(f"[c1] replay done ({time.time() - t0:.0f} s)", flush=True)
    if a.sim_dir:
        gun, st = sim_hits(a.sim_dir, lk, mods, a.seed)
        out["gun"] = {"sim_dir": os.path.abspath(a.sim_dir), "n_hits": int(len(gun)), "digitiser_stats": st}
        out["overall"]["gun"] = overall(gun)
        cmp["gun"] = compare(ref, gun, matched=True)
        cmp["gun_unmatched"] = compare(ref, gun, matched=False)
        print(f"[c1] gun: {len(gun)} hits ({time.time() - t0:.0f} s) pass={cmp['gun']['pass']} "
              f"unmatched pass={cmp['gun_unmatched']['pass']}", flush=True)
    out["bins_csv"] = os.path.basename(os.path.splitext(a.out_json)[0] + "_bins.csv")
    out["compare"] = {k: {**{kk: vv for kk, vv in c.items() if kk != "volumes"},
                          "volumes": {v: {kk: vv for kk, vv in r.items() if kk != "bins"}
                                      for v, r in c["volumes"].items()}} for k, c in cmp.items()}
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    bins_frame(cmp).to_csv(os.path.splitext(a.out_json)[0] + "_bins.csv", index=False, float_format="%.6g")
    print(f"[c1] wrote {a.out_json}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("particles")
    p.add_argument("--out", required=True)
    p.add_argument("--events", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--per-event", type=int, default=50)
    r = sub.add_parser("run")
    r.add_argument("--sim-dir", default=None)
    r.add_argument("--lookup", required=True)
    r.add_argument("--out-json", required=True)
    r.add_argument("--ref-events", type=int, default=40)
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "particles":
        print(make_particles(a.out, a.events, a.seed, a.per_event))
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
