"""Claim H1 of addendum H (docs/preregistration_pileup.md section 6) and the gun numbers reported with it (section 7).
This module evaluates outputs it is given; it runs no model. It is tested on truth-made placeholder outputs.

Gun targets (`gun_targets`), from the stored truth of the overlay events (particles table; hits for the regions):
- gun particles: vertex field (particle_id >> 52) in 1001-1010, by this field alone (not the converter's category);
  isolated 1001-1008, dense 1009-1010;
- targets: the stored is_target (pT > 1 GeV, |eta| < 4, at least 5 hits written in the overlay event);
- P_P: isolated gun targets with |d0| >= 10 mm or r_v >= 100 mm (the converter's d0 and r_v); P_D: the same cuts on
  the dense component;
- region (forward fallback, decision D12): central when none of the target's hits lies at |z| > 1 m, otherwise
  forward +z or -z by the sign of eta;
- soft sibling: another gun particle of the same LLP (same vertex field) with pT <= 1 GeV and at least one hit.

H1 (`h1`): Delta = eps(P_P, C-full) - eps(P_P, C-1GeV) [points], DM at WP-A, on the same targets: the P_P lists of
the two conditions must be equal, every P_P target must have an output row in both (`pair_conditions`). sigma: the
ddof-1 standard deviation of Delta over B = 500 paired event-bootstrap resamplings with the index table of
hnl.trackml.evaluate.bootstrap (default_rng(12345).integers(0, n, size=(500, n)) over the n events of the set in
increasing overlay id; hnl.overlay.hstats, checked against evaluate.bootstrap itself on every run). Each resampling
recomputes both pooled efficiencies on the same events. The outcome inequalities use the unrounded Delta and sigma,
with delta = 1.0 point:
  holds                                         Delta - 2 sigma > -delta
  does not hold, resolved below the margin      Delta + 2 sigma < -delta
  does not hold, inconclusive                   otherwise
Fewer than MIN_PP = 1,000 P_P targets (after any restriction): H1 is evaluated as written and labelled underpowered.
A restriction of section 9.3 (regions) is applied before everything; a failed flag gate is carried as a flag; a
descriptive H (section 9.3) leaves H1 undecided.

Reported without a claim (`report`): eps of P_P, P_D and all gun targets in each condition given; Delta without the P_P
targets that have a soft sibling, and their number; Delta in the bins of addendum G section 7 (|d0|, r_v, r_first,
|d0|/r_first, pT) and the bound on Delta of the navigator losses (`nav_loss_bound`: lost share times the largest
|Delta_bin - Delta| over the bins with at least NAV_MIN_PP = 20 P_P targets; the other bins are listed, decision D5).

Inputs (`load_outputs`). C-full on H-val comes from the sealed directory alone, through
hnl.overlay.blind.sealed_per_target (open_sealed: the committed overlay version record and the manifest sha256 that
RESULTS.md records; an --expect given must agree with it); the frozen chain's per_target tables are never read for
k = 0 in C-full. C-1GeV and C-hs on H-val, and every condition on H-test, come from the grids of hnl.overlay.tracks
(hnl.overlay.tracks.grid_provenance, the check hnl.overlay.v1 makes on the closure grids: every event of the set,
the frozen classifier's scores in that condition in every block, the learned merge model of selection_v1.json, built on
the production overlay tables; never a placeholder). The pilot (k = 9) takes truth-made placeholder outputs only: a
scratch sealed directory of hnl.overlay.blind (C-full, with --expect its manifest sha256) and placeholder grids of
hnl.overlay.tracks (C-1GeV, C-hs), at a working point without a learned component (WP-R-P3); its output is titled DRY
RUN. The H-test hosts are those of hnl.overlay.split_h.htest_hosts (dropped hosts replaced by their spares).

Enforcement. `run` reads every condition of its set or nothing: H-val and the pilot need the C-1GeV and C-hs grids
(C-full from the sealed directory), H-test the C-full, C-1GeV and C-hs grids; a missing one is refused before any
grant, launch or marker check, so a decisive run never goes without C-hs. slurm/overlay/h1.sbatch calls
slurm/overlay/guard_h.sh --grant for every condition, and `run` checks a grant (hnl.overlay.grant) for each: kind eval
on H-val and H-test (made under the set's launch token), dry on the pilot, whose out holds the output JSON. On H-val
and H-test `run` also requires the set's launch record to accept HNL_H_LAUNCH with every pinned file unchanged
(hnl.overlay.blind.check_launch), and writes its output once per working point (hnl.overlay.blind.write_once: a
pending record, then the output placed exclusively, then the consumed marker <runs>/launch/<set>.consumed.<WP>.json;
all three are checked before any input is read; a run that stops before its pending record may be repeated, and from
the pending record on the working point is consumed). The output of a decisive run (hval, htest) has one fixed path,
results/pileup_hp/h1_<set>_<WP>_full-1gev-hs_v1.json (`decisive_out`); any other --out-json is refused. Its P_P
restriction, failed flag gates, navigator-loss share and descriptive state come only from a committed record
(--restriction-record: a tracked file under results/pileup_hp/, committed unchanged, written by the code of decision
D7 (hnl.overlay.d7record), and pinned by the set's launch record: on H-val an H-val record results/pileup_hp/h_t05_*,
on H-test an H-T1 record results/pileup_hp/h_t1_* that R15 names; a record of the other prefix is refused on each,
`record_prefix`); --restriction, --flags, --lost-share and --descriptive are refused on hval and htest whenever
they are given, whatever their value (an empty or zero value included), and stay free on the pilot.

    python -m hnl.overlay.h1 run --set hval --sealed DIR --grid-1gev DIR --grid-hs DIR --restriction-record R \
        --out-json results/pileup_hp/h1_hval_<WP>_full-1gev-hs_v1.json
    python -m hnl.overlay.h1 run --set htest --grid-full DIR --grid-1gev DIR --grid-hs DIR --restriction-record R \
        --out-json results/pileup_hp/h1_htest_<WP>_full-1gev-hs_v1.json
    python -m hnl.overlay.h1 out-path --set hval|htest --wp WP              (prints the fixed output path)
    python -m hnl.overlay.h1 run --set pilot --root ROOT --sealed DIR --expect SHA --grid-1gev DIR --grid-hs DIR \
        --out-json J                                                   (placeholder outputs; WP-R-P3 by default)
    python -m hnl.overlay.h1 dry-run --set pilot --root ROOT --out-json J     (truth-made placeholder dm flags)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.overlay import hstats as HS
from hnl.overlay import split_h as SH
from hnl.trackml.split import EXT, REPO

MARGIN = 1.0                                     # delta [points], decision D13
MIN_PP = 1000
NAV_MIN_PP = 20                                  # P_P targets a bin needs to enter the navigator-loss bound (D5)
GUN_FIRST, N_ISO, N_DENSE = 1001, 8, 2           # = hnl.overlay.gun VERTEX0 + 1, N_ISO, N_DENSE
PP_D0_MM, PP_RV_MM = 10.0, 100.0
FORWARD_Z_MM = 1000.0
REGIONS = ("central", "fwd+", "fwd-")
CONDITIONS = ("full", "1gev", "hs")
SPLIT_H = os.path.join(REPO, "configs", "trackml", "split_h_v1.json")
DERIVED_OVERLAY = os.path.join(EXT, "derived", "overlay_v1")     # = hnl.overlay.writer.DERIVED_OVERLAY
OVERLAY_FIRST_ID = 50_000_000
SET_INDEX = {"hval": 0, "closure": 1, "htest": 2, "pilot": 9}
DECISIVE = ("hval", "htest")
RESULTS_DIR = os.path.join(REPO, "results", "pileup_hp")       # where the decisive outputs go (`decisive_out`)
RECORD_DIR = "results/pileup_hp/"                               # committed records of a decisive run's inputs (D7)
RECORD_KEYS = ("restriction", "flags", "lost_share")
WP_NAMES = ("WP-A", "WP-S", "WP-R-P1", "WP-R-P2", "WP-R-P3", "WP-R-P4")       # frozen working points (selection_v1)
# bins of addendum G section 7 (hnl.eval.addendum_g_summary)
BINS = {"|d0| [mm]": ("abs_d0", [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf]),
        "r_v [mm]": ("rv", [0, 10, 30, 100, 200, 300, 450, 650, np.inf]),
        "first-hit radius [mm]": ("r_first", [0, 50, 100, 200, 300, 500, 700, np.inf]),
        "|d0| / r_first": ("ratio", [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.01]),
        "pT [GeV]": ("pt", [1, 2, 5, 10, 20, np.inf])}


# ------------------------------------------------------------------------------------------------ truth
def gun_targets(particles: pd.DataFrame, hits: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every gun particle of the events (module docstring) with its flags. particles: event, particle_id, pt, eta, d0,
    rv, n_hits, is_target, r_first; hits: event, pid, z (for the regions; None: region unknown)."""
    pid = particles.particle_id.to_numpy(np.int64)
    v = pid >> 52
    g = particles[(v >= GUN_FIRST) & (v < GUN_FIRST + N_ISO + N_DENSE)].copy()
    g["event"] = g.event.astype(np.int64)
    g["particle_id"] = g.particle_id.astype(np.int64)
    g["llp"] = (g.particle_id.to_numpy(np.int64) >> 52) - (GUN_FIRST - 1)
    g["iso"] = g.llp <= N_ISO
    g["abs_d0"] = g.d0.abs().astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        g["ratio"] = g.abs_d0 / g.r_first.astype(np.float64)
    disp = (g.abs_d0 >= PP_D0_MM) | (g.rv >= PP_RV_MM)
    tgt = g.is_target.astype(bool)
    g["target"] = tgt
    g["pp"] = tgt & g.iso & disp
    g["pd"] = tgt & ~g.iso & disp
    soft = (g.pt <= 1.0) & (g.n_hits >= 1)
    n_soft = g.assign(s=soft).groupby(["event", "llp"]).s.transform("sum").to_numpy()
    g["soft_sibling"] = (n_soft - soft.to_numpy(int)) > 0
    if hits is not None:
        far = hits[np.abs(hits.z.to_numpy(np.float64)) > FORWARD_Z_MM]
        fwd = pd.MultiIndex.from_arrays([far.event.to_numpy(np.int64), far.pid.to_numpy(np.int64)])
        f = pd.MultiIndex.from_arrays([g.event.to_numpy(), g.particle_id.to_numpy()]).isin(fwd)
        g["region"] = np.where(~f, "central", np.where(g.eta.to_numpy() > 0, "fwd+", "fwd-"))
    else:
        g["region"] = None
    return g.reset_index(drop=True)


def restrict(g: pd.DataFrame, regions) -> pd.DataFrame:
    """P_P restricted to the given regions (section 9.3); None: unrestricted."""
    if regions is None:
        return g
    bad = sorted(set(regions) - set(REGIONS))
    if bad or g.region.isna().any():
        raise ValueError(f"restriction {regions}: unknown regions {bad} or targets without a region")
    return g.assign(pp=g.pp & g.region.isin(list(regions)))


def pair_conditions(g: pd.DataFrame, outputs: dict, column: str = "pp") -> pd.DataFrame:
    """The targets of `column` with dm in each condition of outputs {condition: (event, particle_id, dm)}: the target
    lists of the conditions must be the same, so each target needs a row in every condition."""
    t = g[g[column]].reset_index(drop=True)
    for cond, tab in outputs.items():
        k = pd.MultiIndex.from_arrays([tab.event.to_numpy(np.int64), tab.particle_id.to_numpy(np.int64)])
        if k.duplicated().any():
            raise ValueError(f"{cond}: a target appears twice")
        t[f"dm_{cond}"] = HS.attach_dm(t, tab, "event", "particle_id", f"{column} in {cond}", np.ones(len(t), bool))
    return t


def per_event(t: pd.DataFrame, events, cols) -> np.ndarray:
    ev = pd.Index(np.asarray(sorted(int(e) for e in events), np.int64))
    pos = ev.get_indexer(t.event.to_numpy(np.int64))
    if (pos < 0).any():
        raise ValueError("targets of events outside the event list")
    return np.stack([np.bincount(pos, minlength=len(ev))] + [np.bincount(
        pos, weights=t[c].to_numpy(bool).astype(np.float64), minlength=len(ev)) for c in cols], 1)


# ------------------------------------------------------------------------------------------------ H1
def outcome(delta: float, sigma: float, margin: float = MARGIN) -> str:
    """The outcome rule of section 6 on the unrounded Delta and sigma."""
    if delta - 2 * sigma > -margin:
        return "holds"
    if delta + 2 * sigma < -margin:
        return "does not hold, resolved below the margin"
    return "does not hold, inconclusive"


def h1(t: pd.DataFrame, events, flags=(), descriptive: bool = False, margin: float = MARGIN) -> dict:
    """H1 on the paired P_P targets t (`pair_conditions` with full and 1gev) of the events (all events of the set)."""
    events = sorted(int(e) for e in events)
    c = per_event(t, events, ["dm_full", "dm_1gev"])
    r = HS.paired_delta(c[:, 0], c[:, 1], c[:, 2])
    n_pp = r["n"]
    out = {"n_events": len(events), "n_pp": n_pp, "eps_full": r["eps_a"], "eps_1gev": r["eps_b"],
           "delta": r["delta"], "sigma": r["sigma"], "B": r["B"], "seed": r["seed"], "ddof": 1, "margin": margin,
           "index_table_check": {k: HS.check_index_table(c[:, 0], c[:, i]) for k, i in (("full", 1), ("1gev", 2))},
           "underpowered": bool(n_pp < MIN_PP), "flags": list(flags)}
    if not all(x["equal"] for x in out["index_table_check"].values()):
        raise RuntimeError(f"index table differs from hnl.trackml.evaluate.bootstrap: {out['index_table_check']}")
    if descriptive:
        out["outcome"] = "not decided (H descriptive, section 9.3)"
    elif not n_pp:
        out["outcome"] = "not decided (no P_P target)"
    else:
        out["outcome"] = outcome(out["delta"], out["sigma"], margin)
    label = out["outcome"]
    if out["underpowered"]:
        label += "; underpowered (fewer than 1,000 P_P targets)"
    if flags:
        label += "; flags: " + ", ".join(flags)
    out["label"] = label
    return out


def nav_loss_bound(lost_share: float, bins: dict, delta: float, min_n: int = NAV_MIN_PP) -> dict:
    """Section 7: lost share times the largest |Delta_bin - Delta| over the bins (`binned_delta`) with at least min_n
    P_P targets (decision D5); the bins with fewer targets (but at least one) are listed and do not enter the max."""
    d, low = [], []
    for name, v in bins.items():
        for b in v:
            if not b.get("n") or not np.isfinite(b["delta"]):
                continue
            if b["n"] >= min_n:
                d.append(abs(b["delta"] - delta))
            else:
                low.append({"variable": name, "bin": b.get("bin"), "n": int(b["n"]), "delta": b["delta"]})
    m = max(d) if d else float("nan")
    return {"lost_share": lost_share, "max_abs_bin_deviation": m, "bound": lost_share * m if d else float("nan"),
            "min_pp_per_bin": min_n, "bins_used": len(d), "bins_below_min": low}


def binned_delta(t: pd.DataFrame, events) -> dict:
    """Delta in the bins of addendum G section 7 (paired bootstrap sigma per bin)."""
    out = {}
    for name, (col, edges) in BINS.items():
        rows = []
        b = pd.cut(t[col].astype(np.float64), edges, right=False)
        for k, g in t.groupby(b, observed=False):
            if len(g):
                c = per_event(g, events, ["dm_full", "dm_1gev"])
                r = HS.paired_delta(c[:, 0], c[:, 1], c[:, 2])
                rows.append({"bin": str(k), "n": r["n"], "eps_full": r["eps_a"], "eps_1gev": r["eps_b"],
                             "delta": r["delta"], "sigma": r["sigma"]})
            else:
                rows.append({"bin": str(k), "n": 0, "delta": float("nan")})
        out[name] = rows
    return out


def efficiencies(g: pd.DataFrame, outputs: dict, events) -> dict:
    """eps [percent] of P_P, P_D and all gun targets in every condition given, with the bootstrap sigma (reported
    without a claim)."""
    out = {}
    for pop in ("pp", "pd", "target"):
        t = pair_conditions(g, outputs, pop)
        rec = {"n": int(len(t))}
        for cond in outputs:
            c = per_event(t, events, [f"dm_{cond}"])
            e, s, _ = HS.bootstrap_stat(c, lambda x: HS.pct(x[..., 1], x[..., 0]))
            rec[cond] = {"eps": e, "sigma": s}
        out[pop] = rec
    return out


def report(g: pd.DataFrame, outputs: dict, events, restriction=None, flags=(), descriptive: bool = False,
           lost_share: float | None = None) -> dict:
    """H1 and the gun numbers of section 7 for one working point (outputs: {condition: (event, particle_id, dm)},
    conditions full and 1gev required, hs optional)."""
    if not {"full", "1gev"} <= set(outputs):
        raise ValueError("H1 needs the outputs in C-full and C-1GeV")
    events = sorted(int(e) for e in events)
    gr = restrict(g, restriction)
    t = pair_conditions(gr, {k: outputs[k] for k in ("full", "1gev")}, "pp")
    out = {"restriction": list(restriction) if restriction else None, "h1": h1(t, events, flags, descriptive)}
    if restriction:
        tu = pair_conditions(g, {k: outputs[k] for k in ("full", "1gev")}, "pp")
        out["h1_all_pp"] = h1(tu, events, flags, descriptive)
    ns = t[~t.soft_sibling]
    c = per_event(ns, events, ["dm_full", "dm_1gev"])
    out["without_soft_siblings"] = {"pp_with_soft_sibling": int(t.soft_sibling.sum()),
                                    **HS.paired_delta(c[:, 0], c[:, 1], c[:, 2])}
    out["bins"] = binned_delta(t, events)
    if lost_share is not None:
        out["nav_loss_bound"] = nav_loss_bound(lost_share, out["bins"], out["h1"]["delta"])
    out["efficiencies"] = efficiencies(g, outputs, events)
    out["pp_by_region"] = ({r: int((t.region == r).sum()) for r in REGIONS} if t.region.notna().all() else None)
    return out


# ------------------------------------------------------------------------------------------------ inputs
def set_events(set_name: str, split: dict | None = None, path: str = SPLIT_H) -> list[int]:
    """The overlay ids of a set in increasing order: hval and closure from their val lists, pilot from its hosts, htest
    from the evaluated H-test hosts (hnl.overlay.split_h.htest_hosts: the list recorded at H-T1, each host dropped by a
    per-event check replaced by its spare as htest_dropped records it)."""
    if split is None:
        with open(path) as fh:
            split = json.load(fh)
    if set_name in ("hval", "closure"):
        hosts = split[set_name]
    elif set_name == "pilot":
        hosts = split["sets"]["pilot"]["hosts"]
    elif set_name == "htest":
        hosts = SH.htest_hosts(split)                         # LookupError before H-T1
    else:
        raise KeyError(set_name)
    k = SET_INDEX[set_name]
    ids = sorted(OVERLAY_FIRST_ID + 1_000_000 * k + int(h) for h in hosts)
    if len(set(ids)) != len(ids):
        raise ValueError("a host appears twice")
    return ids


def load_truth(root: str, events) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stored particles and the hits (event, pid, z) of the gun particles of the events."""
    from hnl.trackml.convert import read_table
    ev = sorted(int(e) for e in events)
    P, H = [], []
    for a in range(0, len(ev), 20):
        part = ev[a:a + 20]
        p = read_table("particles", part, ["event", "particle_id", "pt", "eta", "d0", "rv", "n_hits", "is_target",
                                           "r_first"], root=root)
        v = p.particle_id.to_numpy(np.int64) >> 52
        P.append(p[(v >= GUN_FIRST) & (v < GUN_FIRST + N_ISO + N_DENSE)])
        h = read_table("hits", part, ["event", "pid", "z"], root=root)
        vh = h.pid.to_numpy(np.int64) >> 52
        H.append(h[(vh >= GUN_FIRST) & (vh < GUN_FIRST + N_ISO + N_DENSE)])
    return pd.concat(P, ignore_index=True), pd.concat(H, ignore_index=True)


def load_outputs(set_name: str, wp: str, grids: dict, sealed: str | None = None, expect: str | None = None,
                 root: str | None = None, split: dict | None = None) -> tuple[dict, dict]:
    """({condition: (event, particle_id, dm)}, {condition: source record}) of a set at a working point (module
    docstring, "Inputs"). grids: {condition: hnl.overlay.tracks grid directory}."""
    from hnl.overlay import blind as B
    from hnl.overlay import tracks as TR
    split = split or SH.load_split(SPLIT_H)
    k = SET_INDEX[set_name]
    if set_name not in ("hval", "htest", "pilot"):
        raise KeyError(f"H1 is computed on hval, htest or the pilot, not {set_name!r}")
    if "1gev" not in grids:
        raise ValueError("H1 needs the outputs in C-1GeV")
    outs, src = {}, {}
    if set_name in ("hval", "pilot"):
        if "full" in grids:
            raise PermissionError(f"{set_name} in C-full is read from its sealed directory (hnl.overlay.blind), "
                                  "never from a grid")
        if not sealed:
            raise ValueError(f"{set_name} in C-full needs --sealed (its sealed directory)")
        if set_name == "pilot" and not expect:
            raise ValueError("the pilot in C-full needs --expect (the manifest sha256 of its scratch sealed directory)")
        rows = B.sealed_per_target(sealed, expect, wp, root=root, split=split)
        outs["full"] = rows[["event", "particle_id", "dm"]].astype({"event": np.int64, "particle_id": np.int64,
                                                                    "dm": bool}).reset_index(drop=True)
        with open(os.path.join(sealed, "blind.json")) as fh:
            rec = json.load(fh)
        if rec["set"] != set_name or int(rec["k"]) != k:
            raise ValueError(f"{sealed}: a sealed directory of {rec['set']}, not {set_name}")
        src["full"] = {"sealed": os.path.realpath(sealed), "manifest_sha256": B.manifest_sha256(sealed),
                       "placeholder": bool(rec["scores"].get("placeholder"))}
    else:
        if sealed:
            raise ValueError("H-test has no sealed directory")
        if "full" not in grids:
            raise ValueError("H1 needs the outputs in C-full")
    for cond, d in grids.items():
        if set_name == "pilot":                           # placeholder grids of the pilot only
            rec = TR.grid_record(d, set_name, cond, split)
        else:                                             # the check shared with hnl.overlay.v1 (closure grids)
            rec = TR.grid_provenance(d, set_name, cond, root, None, split)
        outs[cond] = HS.read_per_target(d, wp)
        src[cond] = {"grid": os.path.realpath(d), "placeholder": bool(rec["scores_provenance"]["placeholder"]),
                     "scores": rec["scores"], "git": rec.get("git")}
    return outs, src


def decisive_out(set_name: str, wp: str) -> str:
    """The one output path of a decisive run of a set (hval, htest) at a working point (decision D3, crash rule)."""
    if set_name not in DECISIVE:
        raise ValueError(f"{set_name} has no decisive output")
    if wp not in WP_NAMES:
        raise ValueError(f"no working point {wp!r} (one of {WP_NAMES})")
    return os.path.join(RESULTS_DIR, f"h1_{set_name}_{wp}_full-1gev-hs_v1.json")


def record_prefix(set_name: str) -> str:
    """The repo path prefix of the records of decision D7 that a decisive run of the set may read: on H-val the H-val
    records results/pileup_hp/h_t05_* (hnl.overlay.blind.HT05_RECORDS), on H-test the H-T1 records
    results/pileup_hp/h_t1_* (hnl.overlay.blind.HT1_RECORDS)."""
    from hnl.overlay import blind as B
    prefixes = {"hval": B.HT05_RECORDS, "htest": B.HT1_RECORDS}
    if set_name not in prefixes:
        raise PermissionError(f"{set_name} reads no record of decision D7 (hval and htest only)")
    return prefixes[set_name]


def restriction_record(path: str, repo: str | None = None, set_name: str | None = None) -> dict:
    """The committed record that gives a decisive run its inputs besides the outputs (written by the code of decision
    D7, hnl.overlay.d7record): a tracked file under results/pileup_hp/, for set_name hval an H-val record h_t05_* and
    for htest an H-T1 record h_t1_* (`record_prefix`), committed unchanged (as the guard checks files), JSON with
    restriction (null, or the regions of P_P kept by the fallback of section 9.3), flags (the failed flag gates, a
    list), lost_share (null, or the navigator-loss share of section 7, in [0, 1]) and optionally descriptive (bool;
    `record_values`). Returns them with the record's repo path and sha256; PermissionError or ValueError otherwise."""
    from hnl.overlay import blind as B
    repo = os.path.realpath(repo or REPO)
    real = os.path.realpath(path)
    rel = os.path.relpath(real, repo)
    if rel.startswith("..") or not rel.startswith(RECORD_DIR):
        raise PermissionError(f"{path}: the restriction record of a decisive run is a tracked file under "
                              f"{RECORD_DIR} of the repository")
    if set_name is not None and not rel.startswith(record_prefix(set_name)):
        raise PermissionError(f"{rel}: a decisive run on {set_name} reads a record {record_prefix(set_name)}* only "
                              "(decision D7)")
    if not B.git_committed(real, repo):
        raise PermissionError(f"{rel} is not committed as is: a decisive run reads its restriction, flags and lost "
                              "share from a committed record only (decision D7)")
    with open(real) as fh:
        d = json.load(fh)
    return {"path": rel, "sha256": B.sha256_file(real), **record_values(d, rel)}


def record_values(d, rel: str = "the record") -> dict:
    """The four inputs of a record of decision D7 with their value checks (`restriction_record`); ValueError
    otherwise."""
    missing = [k for k in RECORD_KEYS if not isinstance(d, dict) or k not in d]
    if missing:
        raise ValueError(f"{rel}: no {', '.join(missing)}")
    r, flags, ls, desc = d["restriction"], d["flags"], d["lost_share"], d.get("descriptive", False)
    if r is not None and (not isinstance(r, list) or not r or not set(r) <= set(REGIONS) or len(set(r)) != len(r)):
        raise ValueError(f"{rel}: restriction {r!r} is not null or a list of regions of {REGIONS}")
    if not isinstance(flags, list) or not all(isinstance(f, str) and f for f in flags):
        raise ValueError(f"{rel}: flags {flags!r} is not a list of gate names")
    if ls is not None and (isinstance(ls, bool) or not isinstance(ls, (int, float)) or not 0 <= ls <= 1):
        raise ValueError(f"{rel}: lost_share {ls!r} is not null or a share in [0, 1]")
    if not isinstance(desc, bool):
        raise ValueError(f"{rel}: descriptive {desc!r} is not true or false")
    return {"restriction": r, "flags": flags, "lost_share": None if ls is None else float(ls), "descriptive": desc}


def placeholder_outputs(g: pd.DataFrame, conditions=("full", "1gev", "hs"), seed: int = 20261014) -> dict:
    """Code test only: truth-made (event, particle_id, dm) tables of the gun targets in each condition, dm drawn with
    fixed probabilities from numpy default_rng([seed, condition index]). No model is run, no model output is read."""
    t = g[g.target]
    prob = {"full": 0.80, "1gev": 0.84, "hs": 0.86}
    out = {}
    for i, c in enumerate(conditions):
        rng = np.random.default_rng([seed, i])
        out[c] = pd.DataFrame({"event": t.event.to_numpy(np.int64), "particle_id": t.particle_id.to_numpy(np.int64),
                               "dm": rng.random(len(t)) < prob[c]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True, choices=["hval", "htest", "pilot"])
    r.add_argument("--root", default=None, help="overlay tables (default: the production overlay root; the pilot "
                   "may name a scratch root)")
    r.add_argument("--sealed", default=None, help="hval, pilot: the sealed C-full directory")
    r.add_argument("--expect", default=None, help="pilot: the manifest sha256 of its scratch sealed directory "
                   "(required); hval: optional, it must agree with the line of RESULTS.md, which is always used")
    r.add_argument("--grid-full", default=None, help="htest only")
    r.add_argument("--grid-1gev", required=True)
    r.add_argument("--grid-hs", default=None, help="required (every set reads C-hs)")
    r.add_argument("--wp", default=None, help="working point (default WP-A; on the pilot WP-R-P3)")
    # default None: on hval and htest any of these four given at all, even empty or zero, is refused
    r.add_argument("--restriction", default=None, help="pilot only: regions of P_P, e.g. central,fwd+ (section 9.3)")
    r.add_argument("--flags", default=None, help="pilot only: failed flag gates, comma separated")
    r.add_argument("--descriptive", action="store_const", const=True, default=None, help="pilot only")
    r.add_argument("--lost-share", type=float, default=None, help="pilot only")
    r.add_argument("--restriction-record", default=None, help="hval, htest (required): the committed record of the "
                   "restriction, flags, lost share and descriptive state (results/pileup_hp/, decision D7)")
    r.add_argument("--out-json", required=True, help="hval, htest: the fixed path of `out-path` only")
    o = sub.add_parser("out-path")
    o.add_argument("--set", required=True, choices=list(DECISIVE))
    o.add_argument("--wp", default="WP-A")
    d = sub.add_parser("dry-run")
    d.add_argument("--set", default="pilot", choices=["pilot"])
    d.add_argument("--root", required=True)
    d.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "out-path":
        print(decisive_out(a.set, a.wp))
        return
    t0 = time.time()
    events = set_events(a.set)
    once = None
    rr = None
    if a.cmd == "run":
        from hnl.overlay import blind as B
        from hnl.overlay import grant as G
        grids = {c: d for c, d in (("full", a.grid_full), ("1gev", a.grid_1gev), ("hs", a.grid_hs)) if d}
        need = ("full", "1gev", "hs") if a.set == "htest" else ("1gev", "hs")
        missing = [c for c in need if c not in grids]
        if missing:                                       # every condition or nothing, before any other check
            ap.error(f"run --set {a.set} reads every condition: --grid-{' and --grid-'.join(missing)} missing")
        if a.set != "htest" and "full" in grids:
            raise PermissionError(f"{a.set} in C-full is read from its sealed directory (hnl.overlay.blind), never "
                                  "from a grid")
        conds = list(CONDITIONS)
        for c in conds:                                   # a grant for every condition the run reads
            G.check(events, "dry" if a.set == "pilot" else "eval", c, out=a.out_json)
        wp = a.wp or ("WP-R-P3" if a.set == "pilot" else "WP-A")
        if a.set in DECISIVE:                             # fixed output, committed record, launch, written once
            if os.path.realpath(a.out_json) != os.path.realpath(decisive_out(a.set, wp)):
                raise PermissionError(f"{a.set} at {wp} writes {decisive_out(a.set, wp)} only, not {a.out_json} "
                                      "(decision D3: one output path per working point)")
            free = [f"--{n}" for n, v in (("restriction", a.restriction), ("flags", a.flags),
                                          ("lost-share", a.lost_share), ("descriptive", a.descriptive))
                    if v is not None]                        # given at all (argparse default None), whatever value
            if free:
                raise PermissionError(f"{a.set}: {', '.join(free)} given on the command line; a decisive run reads "
                                      "them from its committed --restriction-record only (decision D7)")
            if not a.restriction_record:
                raise PermissionError(f"{a.set}: --restriction-record (a committed record under {RECORD_DIR}, "
                                      "written by the code of decision D7) is required")
            rr = restriction_record(a.restriction_record, set_name=a.set)
            runs = B.runs_root()
            bad = B.check_launch(a.set, os.environ.get("HNL_H_LAUNCH"), runs)
            if bad:
                raise PermissionError(f"{a.set}: refused by its launch record: {bad[:5]}")
            with open(B.launch_path(a.set, runs)) as fh:         # the record is one the set's launch pinned
                pins = json.load(fh).get("pins") or {}
            if pins.get(rr["path"]) != rr["sha256"]:
                what = ("an H-T1 record that R15 names" if a.set == "htest" else
                        f"the H-val record of decision D7, {B.HT05_RECORDS}*, committed before the launch")
                raise PermissionError(f"{rr['path']} is not pinned by the {'H-test' if a.set == 'htest' else 'H-val'} "
                                      f"launch record (it must be {what})")
            once = B.consumed_path(a.set, wp, runs)
            B.once_refusal(a.out_json, once, {"set": a.set, "wp": wp, "conditions": conds})
        root = B.overlay_root(SET_INDEX[a.set], a.root) or DERIVED_OVERLAY
    else:
        root = a.root
    parts, hits = load_truth(root, events)
    g = gun_targets(parts, hits)
    src = None
    if a.cmd == "dry-run":
        outs = placeholder_outputs(g)
        label = "DRY RUN: truth-made placeholder outputs; no model output was read"
        res = report(g, outs, events)
    else:
        outs, src = load_outputs(a.set, wp, grids, a.sealed, a.expect, a.root)
        if a.set == "pilot":
            label = (f"DRY RUN (truth-made placeholder outputs; code test, no result): pilot at {wp} "
                     "(builders without a learned component)")
        elif a.set == "hval":
            label = f"H-val, frozen Canto-L (TrackML) at {wp}: the val estimate of H1 (no claim)"
        else:
            label = f"H-test, frozen Canto-L (TrackML) at {wp}" + (": claim H1" if wp == "WP-A" else ": read-out")
        if rr is not None:                                # decisive runs: the committed record only
            res = report(g, outs, events, tuple(rr["restriction"]) if rr["restriction"] else None,
                         tuple(rr["flags"]), rr["descriptive"], rr["lost_share"])
        else:
            res = report(g, outs, events, tuple(a.restriction.split(",")) if a.restriction else None,
                         tuple(x for x in (a.flags or "").split(",") if x), bool(a.descriptive), a.lost_share)
    out = {"label": label, "set": a.set, "root": root, "events": len(events), "sources": src,
           "restriction_record": rr, **res, "seconds": round(time.time() - t0, 1)}
    text = json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    if once:
        B.write_once(a.out_json, text, once, {"set": a.set, "wp": wp, "conditions": conds, "git": B._git_head(),
                                              "launch": os.environ.get("HNL_H_LAUNCH"),
                                              "restriction_record": {k: rr[k] for k in ("path", "sha256")}})
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
        with open(a.out_json, "w") as fh:
            fh.write(text)
    print(json.dumps({"label": label, "h1": {k: out["h1"][k] for k in ("n_pp", "delta", "sigma", "label")}},
                     default=str))


if __name__ == "__main__":
    main()
