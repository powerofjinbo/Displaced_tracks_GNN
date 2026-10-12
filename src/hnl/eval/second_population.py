"""Feedback round 2, task F (interim answer to supervisor point 3): a second displaced population, V0 daughters.

Point 3: "testing on a second displaced topology shows that one tracker can serve many signals". Until a sample with
another long-lived-particle signal exists, the frozen trackers are measured on a second, physically different displaced
population that the samples already contain: the charged daughters of K0S and Lambda decays (V0 daughters); electrons
from photon conversions are reported as a separate group.

Populations (truth; target cuts of docs/definitions.md section 4: table pT > 1 GeV, |eta| < 4, >= 5 space points;
double-majority matching, section 5):
  v0      targets whose parent has |PDG| = 310 or 3122 (particle-table column parent_pid = the parent's signed PDG code,
          derived column parent_pdg); sub-groups k0s (310) and lambda (3122, either sign).
  conv    targets that are e+- from a photon conversion in the detector simulation: generation >= 1 and simulation
          process code 2 (ActsFatras ProcessType: 0 undefined, 1 decay, 2 photon conversion, 3 bremsstrahlung,
          4 nuclear interaction; in these samples code 2 gives only e+- and code 3 only photons, codes 1 and 4 never
          occur). parent_pid is 0 for every simulated secondary, so |parent_pid| = 22 does not select conversions.
The census (`census`, every event of every sample) shows where the populations exist: in the ttbar sample K0S and
Lambda are final-state generator particles (not decayed, no hits, every one leaves the simulated volume), so ttbar has
no V0 daughters; the HNL samples contain them (generator decays, generation 0, at their own secondary vertex). In
hnl15_100 parent_pid is 0 for every particle, so its V0 daughters cannot be labelled with this definition and the
external set is not used for them.

Displacement from hits (the project's code): |d0| of the truth-hit helix fit of hnl.truth.fit_events
(hnl.truth.report.truth_d0: fixed radius for muons, free curvature for every other particle, docs/definitions.md
section 3); start radius r_start = radius of the particle's innermost hit (smallest hit r). Bins: r_start [0, 50),
[50, 100), [100, 200), [200, 300), >= 300 mm; |d0| the preregistered bins; pooled r_start >= 100 mm.

Models (frozen working points; nothing is chosen here). Val: the per-seed working points of
results/strong_baselines_v1_val_entries.csv (Canto-L; acorn (HNL) and acorn (ttbar) with connected components, acorn's
CCandWalk plain and score-guided, walkthrough + merging, MWS-layer + merging; Canto-L + MWS-layer + merging) and of
results/ranking/ranking_val_v3_entries.csv (Canto-H, Canto-B; acorn (ttbar+HNL) with MWS-layer + merging and connected
components). Test (post-freeze, descriptive): the same per-seed configurations on the stored test outputs
(results/ranking/spec_test_v1.json, results/strong_baselines_v1_spec_test.json; acorn (ttbar+HNL): the frozen
RUN/eval_test where it holds the cut, otherwise the task-E outputs under runs/feedback_v2/strong_baselines_armiii when
they exist with their provenance); every test configuration is checked against the val working point of the same seed.
The per-particle tables of every evaluation hold every target of every event, so no track candidate is rebuilt.
Round 3 (task F2): task A2's val-best configuration of each acorn arm (MWS-layer + learned merging for all nine runs,
with A2's merge model of each run) is added: val from A2's per-seed choices (results/strong_baselines_v3_val_entries.csv;
the learned-merging row must equal the arm's val-best row) on the united val grid RUN/trackbuild_val_fb3_all, test from
A2's one-time test outputs named in A2's val-fixed spec (results/strong_baselines_v3_spec_test.json, sha256 checked;
provenance and merge-model sha256 checked). The rows are appended with --add (val and test) to a copy of the round-2
read-out, and A2's primary efficiencies and differences are recomputed as identity checks.

Estimator (hnl.eval.bootstrap_tables, as round 1): per seed the efficiency pooled over the events of a set; value = mean
of the three seeds; seed sd (ddof 1); boot sd = event-bootstrap sd of the seed mean with the fixed replicate tables of
round 1, read only (val: runs/feedback_v1/bootstrap_v1/tables/val_b1000, B = 1000, seed 20261005; test: the
preregistered tables derived/bootstrap/v1, B = 500, seed 12345), so every difference is paired over the events;
combined = sqrt(boot sd^2 + seed sd^2 / 3). Differences Canto-L minus each other row: hnl.eval.bootstrap.linear_combination
(paired bootstrap of the difference of seed means plus both seed terms); for the same trained networks (Canto-L with
another builder) also the seed term paired by seed. Event sets are separate: ttbar (ttbar_val / ttbar_test) and HNL
(hnl_joint_val / hnl_joint_test = hnl10_100 + hnl15_200, resampled together by event index: the two samples share their
underlying event, so most of their V0 daughters are the same particles), and for conversions hnl15_100_external on test.

    python -m hnl.eval.second_population census --out DIR
    python -m hnl.eval.second_population checks --out DIR
    python -m hnl.eval.second_population val --out DIR
    python -m hnl.eval.second_population test --out DIR             (once: refuses to overwrite)
    python -m hnl.eval.second_population test --out DIR --add       (only models whose test outputs were missing)
    python -m hnl.eval.second_population val --out DIR --add        (only models missing in the val read-out)
    python -m hnl.eval.second_population report --dir DIR --results results --name second_population_v1 [--notes F]
        [--title T]
Jobs: slurm/feedback/second_population.sbatch (DIR = runs/feedback_v2/second_population_v1 on /ceph; round 3:
runs/feedback_v3/second_population_v2, a copy of the round-2 read-out with the A2 rows appended).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS
from hnl.eval import bootstrap_tables as BT
from hnl.eval import test_stats as TS
from hnl.eval.particles import CODE_SAMPLE, split_event_id

REPO = BT.REPO
DATA = BT.DATA
DERIVED = os.path.join(DATA, "derived")
EVENTS = os.path.join(DERIVED, "events", "v1")
FITS = os.path.join(DERIVED, "truth_fits", "v1")
FB2 = os.path.join(DATA, "runs", "feedback_v2", "strong_baselines_armiii")   # task E (acorn (ttbar+HNL) on test)
FB2_SPEC = "results/strong_baselines_v2_armiii_spec_test.json"   # task E's final val-fixed test spec
VAL_TABLES = os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}")
SB_VAL = "results/strong_baselines_v1_val_entries.csv"
SB_TEST = "results/strong_baselines_v1_spec_test.json"
V3_ENTRIES = "results/ranking/ranking_val_v3_entries.csv"
TEST_SPEC = "results/ranking/spec_test_v1.json"
SELECTION = "results/final_selection_v1.json"
# round 3 (task F2): the val-best acorn configurations under Canto's full builder menu (task A2), MWS-layer + learned
# merging for all nine acorn runs; per-seed val choices, the val-fixed test spec and A2's tables (identity checks)
A2_VAL = "results/strong_baselines_v3_val_entries.csv"
A2_TEST = "results/strong_baselines_v3_spec_test.json"
A2_TEST_SHA256 = "3039efbcf960b1d79b0cc8a6913d2cc21cab3b21e68bb7c0de957a27e2781296"   # DECISIONS "Task A2", 15:32 UTC
A2_TABLE = "results/strong_baselines_v3.csv"
A2_DIFF = "results/strong_baselines_v3_differences.csv"
WP = "results/working_points_v1.json"
WP_SHA256 = "186ac8dd9c8652f8a20dd2402eb7e6254d86789f4d60bdb8468371d743888d16"   # results/final_selection_v1.json

SAMPLES = ("ttbar", "hnl10_100", "hnl15_200", "hnl15_100")
SAMPLE_CODE = {v: k for k, v in CODE_SAMPLE.items()}
K0S, LAMBDA, PHOTON, ELECTRON = 310, 3122, 22, 11
CONVERSION, BREMSSTRAHLUNG = 2, 3          # ActsFatras ProcessType
R_EDGES = (0.0, 50.0, 100.0, 200.0, 300.0, np.inf)
R_POOLED = 100.0
D0_EDGES = BT.D0_EDGES
R_BINS = tuple(BT.bin_name("r", lo, hi) for lo, hi in zip(R_EDGES[:-1], R_EDGES[1:]))
D0_BINS = BT.D0_BINS
QUANTITIES = ("all", "r_ge100") + R_BINS + D0_BINS
POPULATIONS = ("v0", "k0s", "lambda", "conv")
POP_LABEL = {"v0": "V0 daughters (K0S and Lambda)", "k0s": "K0S daughters", "lambda": "Lambda daughters",
             "conv": "photon-conversion e+-"}
Q_LABEL = {"all": "all", "r_ge100": "r_start >= 100 mm", "r_0_50": "[0, 50)", "r_50_100": "[50, 100)",
           "r_100_200": "[100, 200)", "r_200_300": "[200, 300)", "r_300_inf": ">= 300",
           **{b: "[" + b[3:].replace("_inf", ", inf").replace("_", ", ") + ")" for b in D0_BINS}}
Q_LABEL["d0_300_inf"] = ">= 300"
# boundary of the simulated volume: 84-90 % of all particles end at r = 1028.5 mm or |z| = 3001 mm (brief section 2.3)
R_WORLD, Z_WORLD = 1028.0, 3000.0
PCOLS = ["event", "particle_id", "pdg", "q", "pt", "eta", "process", "parent_pdg", "vprim", "vsec", "particle", "gen",
         "sub", "n_hits", "n_pixel_hits", "r_first", "is_target", "category"]

# --------------------------------------------------------------------------------------------------------------------
# populations


def population_masks(t: pd.DataFrame) -> dict[str, np.ndarray]:
    """Population membership of the rows of a particle table (targets only; see the module docstring)."""
    tgt = t.is_target.to_numpy(dtype=bool)
    par = np.abs(t.parent_pdg.to_numpy(dtype=np.int64))
    k0s = tgt & (par == K0S)
    lam = tgt & (par == LAMBDA)
    conv = (tgt & (t.process.to_numpy(dtype=np.int64) == CONVERSION) & (t.gen.to_numpy(dtype=np.int64) >= 1)
            & (np.abs(t.pdg.to_numpy(dtype=np.int64)) == ELECTRON))
    return {"v0": k0s | lam, "k0s": k0s, "lambda": lam, "conv": conv}


def quantity_masks(r_start, d0) -> dict[str, np.ndarray]:
    """Bins of the start radius (innermost-hit radius) and of |d0|, and the pooled r_start >= 100 mm."""
    r = np.asarray(r_start, dtype=np.float64)
    ad0 = np.abs(np.asarray(d0, dtype=np.float64))
    m = {"all": np.ones(len(r), dtype=bool), "r_ge100": r >= R_POOLED}
    for name, lo, hi in zip(R_BINS, R_EDGES[:-1], R_EDGES[1:]):
        m[name] = (r >= lo) & (r < hi)
    for name, lo, hi in zip(D0_BINS, D0_EDGES[:-1], D0_EDGES[1:]):
        m[name] = (ad0 >= lo) & (ad0 < hi)
    return m


def innermost_radius(hits: pd.DataFrame) -> pd.DataFrame:
    """(event, particle_id, r_start): the smallest hit radius of each particle (hits: event, r, pid)."""
    g = hits.groupby(["event", "pid"], sort=False).r.min().reset_index()
    return g.rename(columns={"pid": "particle_id", "r": "r_start"}).astype({"r_start": np.float64})


def _read(table: str, sample: str, events, columns) -> pd.DataFrame:
    from hnl.data.convert import read_table
    return read_table(EVENTS, table, sample, events=events, columns=columns)


def population_table(event_ids) -> pd.DataFrame:
    """Every target of the populations in the given acorn event ids, with r_start (innermost hit), r_first (chain head),
    d0 (definitions section 3) and the fixed-radius / free fits' d0 (for the check of the bins)."""
    from hnl.truth.report import truth_d0
    codes, idx = split_event_id(np.unique(np.asarray(event_ids, dtype=np.int64)))
    out = []
    for c in np.unique(codes):
        sample = CODE_SAMPLE[int(c)]
        ev = np.sort(idx[codes == c])
        p = _read("particles", sample, ev, PCOLS)
        p = p[p.is_target.astype(bool)]
        m = population_masks(p)
        p = p[m["v0"] | m["conv"]].copy()
        f = pd.read_parquet(os.path.join(FITS, f"particles_{sample}.parquet"),
                            columns=["event", "particle_id", "pdg", "fix_d0", "free_d0"],
                            filters=[("event", "in", [int(e) for e in ev])])
        f["d0"] = truth_d0(f)
        p = p.merge(f.drop(columns="pdg"), on=["event", "particle_id"], how="left", validate="one_to_one")
        h = _read("hits", sample, ev, ["event", "r", "pid"])
        h = h[h.pid.isin(p.particle_id.unique())]
        p = p.merge(innermost_radius(h), on=["event", "particle_id"], how="left", validate="one_to_one")
        p["sample"] = sample
        p["event_id"] = np.int64(c) * 100_000_000 + p.event.to_numpy(dtype=np.int64)
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    bad = p.d0.isna() | p.r_start.isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} population targets without a truth fit or without hits")
    return p.reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------
# census and hand checks


def _split_names(sample: str, events: np.ndarray) -> np.ndarray:
    from hnl.data.splits import read_split
    out = np.full(len(events), "", dtype=object)
    for sp in (("external",) if sample == "hnl15_100" else ("train", "val", "test")):
        out[np.isin(events, read_split(sample, sp))] = sp
    return out


def census_counts(p: pd.DataFrame) -> dict:
    """Counts of one sample / split (particles table rows with the census columns)."""
    apdg = np.abs(p.pdg.to_numpy(dtype=np.int64))
    par = np.abs(p.parent_pdg.to_numpy(dtype=np.int64))
    proc = p.process.to_numpy(dtype=np.int64)
    gen = p.gen.to_numpy(dtype=np.int64)
    tgt = p.is_target.to_numpy(dtype=bool)
    hits = p.n_hits.to_numpy(dtype=np.int64)
    with np.errstate(invalid="ignore"):
        r_end = np.hypot(p.vx.to_numpy(dtype=float), p.vy.to_numpy(dtype=float))
        world = (~np.isfinite(r_end)) | (r_end >= R_WORLD) | (np.abs(p.vz.to_numpy(dtype=float)) >= Z_WORLD)
    pm = population_masks(p)
    out = {"events": int(p.event.nunique()), "particles": int(len(p)),
           "parent_pid_nonzero": int((par != 0).sum())}
    for name, code in (("k0s", K0S), ("lambda", LAMBDA)):
        s = apdg == code
        out[f"{name}_final_state"] = int(s.sum())
        out[f"{name}_final_state_with_hits"] = int((s & (hits > 0)).sum())
        out[f"{name}_final_state_leaving_volume"] = int((s & world).sum())
        d = par == code
        out[f"{name}_daughters"] = int(d.sum())
        out[f"{name}_daughters_with_hits"] = int((d & (hits > 0)).sum())
        out[f"{name}_daughter_targets"] = int((d & tgt).sum())
        out[f"{name}_daughter_targets_gen0"] = int((d & tgt & (gen == 0)).sum())
    out["v0_daughter_targets"] = int(pm["v0"].sum())
    v0d = np.isin(par, (K0S, LAMBDA))
    out["v0_daughters_at_vsec0"] = int((v0d & (p.vsec.to_numpy(dtype=np.int64) == 0)).sum())
    conv_all = (proc == CONVERSION) & (gen >= 1)
    out["conv_secondaries"] = int(conv_all.sum())
    out["conv_secondaries_pt0"] = int((conv_all & (p.pt.to_numpy(dtype=float) == 0)).sum())
    out["conv_secondaries_not_electron"] = int((conv_all & (apdg != ELECTRON)).sum())
    out["conv_targets"] = int(pm["conv"].sum())
    out["brems_secondaries"] = int(((proc == BREMSSTRAHLUNG) & (gen >= 1)).sum())
    out["brems_secondaries_not_photon"] = int(((proc == BREMSSTRAHLUNG) & (gen >= 1) & (apdg != PHOTON)).sum())
    out["process_1_or_4"] = int(np.isin(proc, (1, 4)).sum())
    out["gen_ge1_parent_nonzero"] = int(((gen >= 1) & (par != 0)).sum())
    out["parent22"] = int((par == PHOTON).sum())
    out["parent22_targets"] = int(((par == PHOTON) & tgt).sum())
    out["targets"] = int(tgt.sum())
    return out


ARMIII_HNL_TRAIN = {"hnl10_100": lambda e: e < 5000, "hnl15_200": lambda e: e >= 5000}   # DECISIONS 2026-10-02


def census() -> pd.DataFrame:
    """Counts per sample and split over every event; split "train_ttbar+HNL" = the HNL training events of
    acorn (ttbar+HNL) (hnl10_100 train index < 5000, hnl15_200 train index >= 5000)."""
    cols = ["event", "pdg", "parent_pdg", "process", "gen", "vsec", "is_target", "n_hits", "pt", "vx", "vy", "vz"]
    rows = []
    for sample in SAMPLES:
        p = _read("particles", sample, None, cols)
        ev = p.event.to_numpy()
        sp = _split_names(sample, np.unique(ev))
        split_of = pd.Series(sp, index=np.unique(ev))
        p["split"] = split_of.reindex(ev).to_numpy()
        for split, g in p.groupby("split", sort=False):
            rows.append({"sample": sample, "split": split, **census_counts(g)})
        if sample in ARMIII_HNL_TRAIN:
            g = p[(p.split == "train") & ARMIII_HNL_TRAIN[sample](p.event.to_numpy())]
            rows.append({"sample": sample, "split": "train_ttbar+HNL", **census_counts(g)})
        rows.append({"sample": sample, "split": "all", **census_counts(p)})
        print(f"[census] {sample}: {len(p)} particles", flush=True)
    return pd.DataFrame(rows)


def hand_checks() -> str:
    """Event-by-event checks of the population definition, printed for the facts (a few events by hand)."""
    from hnl.data.splits import read_split
    lines = []
    say = lines.append
    # (a) ttbar: K0S / Lambda are final-state particles, nothing has a V0 parent
    ev = read_split("ttbar", "val")
    p = _read("particles", "ttbar", ev, PCOLS + ["vx", "vy", "vz", "px", "py", "pz"])
    par = p.parent_pdg.abs()
    say(f"(a) ttbar val, {len(ev)} events, {len(p)} particle rows")
    for name, code in (("K0S", K0S), ("Lambda", LAMBDA)):
        s = p[p.pdg.abs() == code]
        r_end = np.hypot(s.vx, s.vy)
        world = (~np.isfinite(r_end)) | (r_end >= R_WORLD) | (s.vz.abs() >= Z_WORLD)
        say(f"    {name} (|PDG| {code}) final-state rows {len(s)}, with hits {(s.n_hits > 0).sum()}, ending at the "
            f"boundary of the simulated volume {int(world.sum())}; rows with |parent_pid| = {code}: {(par == code).sum()}")
    pi = p[(p.pdg.abs() == 211) & (p.gen == 0)]
    top = pi.parent_pdg.abs().value_counts().head(6)
    say(f"    parents of the {len(pi)} generator-level charged pions (|PDG|, rows): "
        + ", ".join(f"{k} {v}" for k, v in top.items()))
    e0 = int(ev[0])
    s0 = p[(p.event == e0) & p.pdg.abs().isin([K0S, LAMBDA])].head(4)
    for _, x in s0.iterrows():
        say(f"    event {e0}: PDG {x.pdg} (parent {x.parent_pdg}), pT {x.pt:.3f} GeV, {x.n_hits} hits, end point r = "
            f"{np.hypot(x.vx, x.vy):.1f} mm, z = {x.vz:.1f} mm")
    # (b) HNL val: V0 decays event by event
    hit_cols = ["event", "row", "r", "phi", "z", "volume", "layer", "pid"]
    for sample in ("hnl10_100", "hnl15_200"):
        evs = read_split(sample, "val")[:3]
        p = _read("particles", sample, evs, PCOLS)
        h = _read("hits", sample, evs, hit_cols)
        f = pd.read_parquet(os.path.join(FITS, f"particles_{sample}.parquet"),
                            columns=["event", "particle_id", "fix_d0", "free_d0"],
                            filters=[("event", "in", [int(e) for e in evs])])
        rmin = innermost_radius(h[["event", "r", "pid"]])
        p = p.merge(f, on=["event", "particle_id"], how="left").merge(rmin, on=["event", "particle_id"], how="left")
        v = p[p.parent_pdg.abs().isin([K0S, LAMBDA])]
        say(f"(b) {sample} val events {list(map(int, evs))}: {len(v)} particles with a K0S / Lambda parent, "
            f"{int(v.is_target.sum())} targets; K0S / Lambda final-state rows: "
            f"{int(p.pdg.abs().isin([K0S, LAMBDA]).sum())}")
        for (e, vs), g in v.groupby(["event", "vsec"]):
            g = g.sort_values("pdg")
            desc = "; ".join(f"PDG {x.pdg} pT {x.pt:.2f} {x.n_hits} hits r_start "
                             + ("-" if np.isnan(x.r_start) else f"{x.r_start:.1f}")
                             + (" target" if x.is_target else "") for _, x in g.iterrows())
            say(f"    event {e} vertex vsec {vs} (gen {sorted(set(g.gen))}), parent {int(g.parent_pdg.iloc[0])}: {desc}")
        if sample == "hnl10_100":
            pairs = v[v.is_target].groupby(["event", "vsec"]).filter(lambda g: len(g) == 2 and g.is_target.all())
            for (e, vs), g in list(pairs.groupby(["event", "vsec"]))[:2]:
                for _, x in g.iterrows():
                    hh = h[(h.event == e) & (h.pid == x.particle_id)].sort_values("r")
                    say(f"    hits of event {e} PDG {x.pdg} (parent {x.parent_pdg}): r_start {x.r_start:.1f} mm, "
                        f"chain head r_first {x.r_first:.1f} mm, d0 free fit {x.free_d0:.2f} mm, fixed-R fit "
                        f"{x.fix_d0:.2f} mm; (r, z, volume, layer) = "
                        + ", ".join(f"({a:.1f}, {b:.1f}, {int(c)}, {int(d)})"
                                    for a, b, c, d in zip(hh.r, hh.z, hh.volume, hh.layer)))
    # (c) conversions: e+ e- pairs of one photon, same start radius
    ev = read_split("ttbar", "val")[:2]
    p = _read("particles", "ttbar", ev, PCOLS)
    h = _read("hits", "ttbar", ev, ["event", "r", "pid"])
    p = p.merge(innermost_radius(h), on=["event", "particle_id"], how="left")
    c = p[(p.process == CONVERSION) & (p.gen >= 1)]
    key = ["event", "vprim", "vsec", "particle", "gen"]
    grp = c.groupby(key).pdg.agg(lambda x: tuple(sorted(x)))
    say(f"(c) ttbar val events {list(map(int, ev))}: {len(c)} conversion secondaries (process 2, gen >= 1), all e+-: "
        f"{bool(c.pdg.abs().eq(ELECTRON).all())}; groups by (vprim, vsec, particle, gen): "
        + ", ".join(f"{k} x{v}" for k, v in grp.value_counts().items()) + f"; targets {int(c.is_target.sum())}")
    tg = c[c.is_target]
    for _, x in tg.head(4).iterrows():
        mates = c[(c[key] == x[key]).all(axis=1) & (c.particle_id != x.particle_id)]
        say(f"    target PDG {x.pdg}, pT {x.pt:.2f} GeV (final state), {x.n_hits} hits, r_start {x.r_start:.1f} mm; "
            f"partner(s) from the same photon: "
            + (", ".join(f"PDG {m.pdg} {m.n_hits} hits r_start "
                         + ("-" if np.isnan(m.r_start) else f"{m.r_start:.1f}") for _, m in mates.iterrows()) or "none"))
    return "\n".join(lines)


def population_checks(pop: pd.DataFrame) -> dict:
    """Agreement of the innermost-hit radius with the chain head and of the free / fixed-radius fits' |d0| bins."""
    out = {}
    pm = population_masks(pop)
    for name in ("v0", "conv"):
        t = pop[pm[name]]
        if t.empty:
            continue
        same_r = np.isclose(t.r_start, t.r_first, atol=1e-3)
        rb = np.digitize(t.r_start, R_EDGES[1:-1])
        rbf = np.digitize(t.r_first, R_EDGES[1:-1])
        db = np.digitize(np.abs(t.free_d0), D0_EDGES[1:-1])
        dbx = np.digitize(np.abs(t.fix_d0), D0_EDGES[1:-1])
        out[name] = {"n": int(len(t)), "r_start_equals_chain_head": float(same_r.mean()),
                     "same_r_bin_as_chain_head": float((rb == rbf).mean()),
                     "same_d0_bin_free_vs_fixed_radius_fit": float((db == dbx).mean()),
                     "median_abs_d0": float(np.median(np.abs(t.d0))),
                     "median_r_start": float(np.median(t.r_start)),
                     "median_abs_d0_over_r_start": float(np.median(np.abs(t.d0) / t.r_start))}
    return out


# --------------------------------------------------------------------------------------------------------------------
# replicate tables (read only)


def load_tables(directory: str, sets: dict, set_of_code: dict, label: str) -> BT.Resampler:
    """The fixed replicate tables of round 1, read only (never created here); units checked against the splits."""
    from hnl.data.splits import read_split
    units, w = {}, {}
    for name, parts in sets.items():
        path = os.path.join(directory, f"{name}.npz")
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing: the replicate tables of round 1 are required (not created here)")
        z = np.load(path)
        u = np.unique(np.concatenate([read_split(s, sp) for s, sp in parts]))
        if not np.array_equal(z["units"], u):
            raise SystemExit(f"{path}: units differ from the split lists")
        units[name], w[name] = z["units"], z["w"]
    return BT.Resampler(units, w, set_of_code, label)


def val_resampler() -> BT.Resampler:
    rs = load_tables(VAL_TABLES, BT.VAL_SETS, BT.VAL_CODE, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    if rs.b != BT.B_VAL:
        raise SystemExit(f"val tables with B = {rs.b}")
    return rs


def test_resampler() -> BT.Resampler:
    rs = load_tables(BS.DEFAULT_DIR, BS.SETS, TS.SET_OF_CODE, f"preregistered test tables (B = {BS.B}, seed {BS.SEED})")
    if rs.b != BS.B:
        raise SystemExit(f"test tables with B = {rs.b}")
    return rs


EVENT_SETS = {  # split -> {event set: sample codes}; hnl15_100_external only for the conversion group; the two halves
    # of the joint HNL set on the joint resampling units (CSV only; their underlying events are shared)
    "val": {"ttbar_val": (1,), "hnl_joint_val": (2, 4), "hnl10_100_val": (2,), "hnl15_200_val": (4,)},
    "test": {"ttbar_test": (1,), "hnl_joint_test": (2, 4), "hnl15_100_external": (3,), "hnl10_100_test": (2,),
             "hnl15_200_test": (4,)},
}
MAIN_SETS = {"val": ("ttbar_val", "hnl_joint_val"), "test": ("ttbar_test", "hnl_joint_test", "hnl15_100_external")}
SET_LABEL = {"ttbar_val": "ttbar val", "hnl_joint_val": "HNL val (hnl10_100 + hnl15_200)",
             "ttbar_test": "ttbar test", "hnl_joint_test": "HNL test (hnl10_100 + hnl15_200)",
             "hnl15_100_external": "hnl15_100 external", "hnl10_100_val": "hnl10_100 val",
             "hnl15_200_val": "hnl15_200 val", "hnl10_100_test": "hnl10_100 test", "hnl15_200_test": "hnl15_200 test"}
UNDEFINED = {("hnl15_100_external", "v0"), ("hnl15_100_external", "k0s"), ("hnl15_100_external", "lambda")}


# --------------------------------------------------------------------------------------------------------------------
# models


def _repo(p: str) -> str:
    return BT._repo(p)


def _a2_label(group: str) -> str:
    """Task A2's label of the learned-merging family row of an acorn arm (results/strong_baselines_v3*)."""
    return f"{group} + MWS-layer + learned merging (Canto-L's builder)"


def _a2_best(group: str) -> str:
    """Task A2's label of the val-best row of an acorn arm (the per-seed choice over the full menu)."""
    return f"{group} + val-best builder of the full menu (chosen per seed)"


CANTO_SAME = "Canto-L + MWS-layer + merging (same builder as the headline baseline; not the frozen builder)"
V3_CANTO_H = "Canto-H (full budget): hand-cut helix union, walkthrough + merging (rule R6)"
V3_CANTO_B = "Canto-B (full budget): base graph, learned merging (rule R6)"
V3_CANTO_L = "Canto-L (full budget): learned triplet union, learned merging (rule R6)"
# (key, label, short, group, val file, label in the val file, test source, label in the test source)
MODELS = (
    ("canto_l", "Canto-L (frozen: learned merging)", "Canto-L", "Canto", SB_VAL, "Canto-L (frozen: learned merging)",
     TEST_SPEC, V3_CANTO_L),
    ("canto_h", "Canto-H (frozen: walkthrough + merging)", "Canto-H", "Canto", V3_ENTRIES, V3_CANTO_H,
     TEST_SPEC, V3_CANTO_H),
    ("canto_b", "Canto-B (frozen: learned merging)", "Canto-B", "Canto", V3_ENTRIES, V3_CANTO_B, TEST_SPEC, V3_CANTO_B),
    ("canto_l_mwsm", "Canto-L + MWS-layer + merging (same builder; not the frozen builder)", "Canto-L MWS+M", "Canto",
     SB_VAL, CANTO_SAME, SB_TEST, CANTO_SAME),
    ("hnl_walkm", "acorn (HNL) + walkthrough + merging (val-best acorn (HNL) configuration)", "(HNL) walk+M",
     "acorn (HNL)", SB_VAL, "acorn (HNL) + walkthrough + merging (Canto builder family)",
     SB_TEST, "acorn (HNL) + walkthrough + merging (Canto builder family)"),
    ("hnl_mwsm", "acorn (HNL) + MWS-layer + merging", "(HNL) MWS+M", "acorn (HNL)", SB_VAL,
     "acorn (HNL) + MWS-layer + merging (Canto builder)", SB_TEST, "acorn (HNL) + MWS-layer + merging (Canto builder)"),
    ("hnl_acwalk", "acorn (HNL) + acorn's CCandWalk", "(HNL) CCandWalk", "acorn (HNL)", SB_VAL,
     "acorn (HNL) + acorn's CC + walkthrough (CCandWalk)", SB_TEST, "acorn (HNL) + acorn's CC + walkthrough (CCandWalk)"),
    ("hnl_acwalkg", "acorn (HNL) + acorn's CCandWalk, score-guided", "(HNL) CCandWalk-g", "acorn (HNL)", SB_VAL,
     "acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk)",
     SB_TEST, "acorn (HNL) + acorn's CC + walkthrough, score-guided (CCandWalk)"),
    ("hnl_ccbest", "acorn (HNL) + CC, best cut <= FR*", "(HNL) CC*", "acorn (HNL)", SB_VAL,
     "acorn (HNL) + CC, best cut <= FR*", TEST_SPEC, "acorn IGNN, HNL-trained + CC (best cut <= FR*)"),
    ("hnl_cc", "acorn (HNL) + CC, preregistered cut", "(HNL) CC", "acorn (HNL)", SB_VAL,
     "acorn (HNL) + CC, preregistered cut (claims' reference)", TEST_SPEC,
     "acorn IGNN, HNL-trained + CC (prereg. cut rule)"),
    ("ttbar_mwsm", "acorn (ttbar) + MWS-layer + merging", "(ttbar) MWS+M", "acorn (ttbar)", SB_VAL,
     "acorn (ttbar) + MWS-layer + merging (Canto builder)", SB_TEST,
     "acorn (ttbar) + MWS-layer + merging (Canto builder)"),
    ("ttbar_walkm", "acorn (ttbar) + walkthrough + merging", "(ttbar) walk+M", "acorn (ttbar)", SB_VAL,
     "acorn (ttbar) + walkthrough + merging (Canto builder family)", SB_TEST,
     "acorn (ttbar) + walkthrough + merging (Canto builder family)"),
    ("ttbar_acwalk", "acorn (ttbar) + acorn's CCandWalk", "(ttbar) CCandWalk", "acorn (ttbar)", SB_VAL,
     "acorn (ttbar) + acorn's CC + walkthrough (CCandWalk)", SB_TEST,
     "acorn (ttbar) + acorn's CC + walkthrough (CCandWalk)"),
    ("ttbar_acwalkg", "acorn (ttbar) + acorn's CCandWalk, score-guided", "(ttbar) CCandWalk-g", "acorn (ttbar)", SB_VAL,
     "acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk)", SB_TEST,
     "acorn (ttbar) + acorn's CC + walkthrough, score-guided (CCandWalk)"),
    ("ttbar_cc", "acorn (ttbar) + CC (preregistered cut = best cut)", "(ttbar) CC", "acorn (ttbar)", SB_VAL,
     "acorn (ttbar) + CC, preregistered cut (= best cut)", TEST_SPEC,
     "acorn IGNN, ttbar-trained + CC (prereg. cut rule)"),
    ("mix_mwsm", "acorn (ttbar+HNL) + MWS-layer + merging", "(ttbar+HNL) MWS+M", "acorn (ttbar+HNL)", V3_ENTRIES,
     "acorn IGNN, ttbar+HNL-trained + Canto builder (MWS-layer + merging)", "armiii", "mwsm"),
    ("mix_ccbest", "acorn (ttbar+HNL) + CC, best cut <= FR*", "(ttbar+HNL) CC*", "acorn (ttbar+HNL)", V3_ENTRIES,
     "acorn IGNN, ttbar+HNL-trained + CC (best cut <= FR*)", "armiii", "cc"),
    ("mix_cc", "acorn (ttbar+HNL) + CC, preregistered cut", "(ttbar+HNL) CC", "acorn (ttbar+HNL)", V3_ENTRIES,
     "acorn IGNN, ttbar+HNL-trained + CC (prereg. cut rule)", "armiii", "cc"),
    # round 3, task F2: task A2's val-best configuration of each acorn arm (learned merging for every seed)
    ("hnl_lm", "acorn (HNL) + MWS-layer + learned merging (val-best acorn (HNL) configuration, full menu)",
     "(HNL) LM", "acorn (HNL)", A2_VAL, _a2_label("acorn (HNL)"), A2_TEST, _a2_label("acorn (HNL)")),
    ("ttbar_lm", "acorn (ttbar) + MWS-layer + learned merging (val-best acorn (ttbar) configuration, full menu)",
     "(ttbar) LM", "acorn (ttbar)", A2_VAL, _a2_label("acorn (ttbar)"), A2_TEST, _a2_label("acorn (ttbar)")),
    ("mix_lm", "acorn (ttbar+HNL) + MWS-layer + learned merging (val-best acorn (ttbar+HNL) configuration, full menu)",
     "(ttbar+HNL) LM", "acorn (ttbar+HNL)", A2_VAL, _a2_label("acorn (ttbar+HNL)"), A2_TEST,
     _a2_label("acorn (ttbar+HNL)")),
)
NEW = "canto_l"
SAME_NETWORKS = {("canto_l", "canto_l_mwsm")}
LM_KEYS = ("hnl_lm", "ttbar_lm", "mix_lm")
GROUP_ORDER = ("Canto", "acorn (HNL)", "acorn (ttbar)", "acorn (ttbar+HNL)")


def _seed_of(run: str) -> int:
    return int(re.search(r"_s(\d+)$", os.path.normpath(run)).group(1))


def _parse_val_wp(wp: str) -> tuple[str, str]:
    """('cc', cut) or (family, params) of a val working point string."""
    m = re.match(r"^CC (\d+\.\d+)", wp)
    if m:
        return "cc", m.group(1)
    m = re.match(r"^(\S+) (\S+) \((\S+)\)$", wp)
    if m is None:
        raise ValueError(f"working point {wp!r}")
    return m.group(1), m.group(2)


def val_entries() -> list[dict]:
    """Per model the per-seed (run, working point) of the committed val files, seeds 0, 1, 2 in order."""
    files = {f: pd.read_csv(_repo(f)) for f in (SB_VAL, V3_ENTRIES, A2_VAL)}
    out = []
    for key, label, short, group, vf, vlab, *_ in MODELS:
        e = files[vf][files[vf].label == vlab]
        if len(e) != 3:
            raise SystemExit(f"{vf}: {len(e)} entries for {vlab!r}")
        e = e.assign(seed=[_seed_of(r) for r in e.run]).sort_values("seed")
        if list(e.seed) != [0, 1, 2]:
            raise SystemExit(f"{vf}: seeds {list(e.seed)} for {vlab!r}")
        if vf == A2_VAL:   # the learned-merging row must be task A2's val-best row of the arm, seed by seed
            b = files[vf][files[vf].label == _a2_best(group)]
            b = b.assign(seed=[_seed_of(r) for r in b.run]).sort_values("seed")
            if list(zip(b.run, b.working_point)) != list(zip(e.run, e.working_point)):
                raise SystemExit(f"{vf}: {vlab!r} is not the val-best configuration of {group} in every seed")
        out.append({"key": key, "label": label, "short": short, "group": group, "source": vf,
                    "entries": [{"run": r.run, "wp": r.working_point} for _, r in e.iterrows()]})
    return out


A2_RUNS = os.path.join(DATA, "runs", "feedback_v3", "strong_baselines_a2")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def a2_val_merge_model(run: str, wp: str) -> dict:
    """Provenance of an A2 learned-merging val working point: the configuration lies in the run's united grid
    RUN/trackbuild_val_fb3_all, which holds RUN/trackbuild_val_fb3_lmerge, made with A2's merge model of this run;
    that model file is unchanged (sha256 of its training provenance) and is the model of A2's test spec."""
    fam, params = _parse_val_wp(wp)
    if fam != "mwslmerge" or not wp.endswith("(trackbuild_val_fb3_all)"):
        raise SystemExit(f"{run}: A2 learned-merging working point {wp!r}")
    merged = json.load(open(os.path.join(run, "trackbuild_val_fb3_all", "merged_from.json")))["merged_from"]
    part = os.path.join(run, "trackbuild_val_fb3_lmerge")
    if os.path.normpath(part) not in [os.path.normpath(p) for p in merged]:
        raise SystemExit(f"{run}/trackbuild_val_fb3_all does not hold {part}")
    cfg = [c for c in json.load(open(os.path.join(part, "configs.json")))
           if c[0] == fam and ",".join(str(x) for x in c[1:]) == params]
    if len(cfg) != 1:
        raise SystemExit(f"{part}: configuration {fam} {params} found {len(cfg)} times")
    model = os.path.join(A2_RUNS, os.path.basename(os.path.normpath(run)), "lmerge", "model.joblib")
    gp = json.load(open(os.path.join(part, "provenance.json")))
    tp = json.load(open(os.path.join(os.path.dirname(model), "provenance.json")))
    sha = _sha256(model)
    if (os.path.normpath(gp["merge_model"]) != os.path.normpath(model) or sha != tp["model_sha256"]
            or os.path.normpath(tp["run"]) != os.path.normpath(run)):
        raise SystemExit(f"{part}: merge model {gp['merge_model']} is not {model} (sha256 {sha})")
    return {"merge_model": model, "merge_model_sha256": sha, "val_grid_part": part, "val_grid_job": gp.get("slurm_job")}


def _check_selection(run: str, config: str) -> None:
    sel = json.load(open(_repo(SELECTION)))
    cfgs = json.load(open(os.path.join(run, "trackbuild_test", "configs.json")))
    if ",".join(str(x) for x in cfgs[0]) != config:
        raise SystemExit(f"{run}/trackbuild_test: configuration 0 {cfgs[0]} is not the frozen {config}")
    if json.load(open(os.path.join(run, "final_selection_used.json"))) != sel:
        raise SystemExit(f"{run}: the test job's selection copy differs from {SELECTION}")


def _cut_in(d: str, cut: float) -> bool:
    for f in ("per_event.parquet", "per_particle.parquet"):
        c = pd.read_parquet(os.path.join(d, f), columns=["cut"]).cut.astype(np.float64).unique()
        if not np.isclose(c, cut, atol=1e-6).any():
            return False
    return True


def _in_fb2_spec(run: str, d: str, config: str | None = None, cut: float | None = None) -> bool:
    """Whether task E's final test spec names this output (same run and directory; same configuration or cut)."""
    spec = json.load(open(_repo(FB2_SPEC)))
    for m in spec["models"]:
        for e in m["entries"]:
            sub = e.get("sub", "")
            sub = sub if os.path.isabs(sub) else os.path.join(e["run"], sub)
            if os.path.normpath(e["run"]) != os.path.normpath(run) or os.path.normpath(sub) != os.path.normpath(d):
                continue
            if config is not None and e.get("config") == config:
                return True
            if cut is not None and e.get("config") is None and abs(float(e["cut"]) - cut) < 1e-6:
                return True
    return False


def _armiii_test(kind: str, run: str, seed: int, fam: str, params: str) -> tuple[str, float, str] | str | None:
    """Test source of an acorn (ttbar+HNL) seed: (directory, cut or configuration index, note); None if the output does
    not exist (yet), a message if it exists but does not match the val working point. Task E's outputs are read only
    after task E's final test spec exists (its rule: no test output is read before that spec is written) and only if
    that spec names them with the same configuration."""
    if kind == "cc":
        cut = float(params)
        d = os.path.join(run, "eval_test")
        if _cut_in(d, cut):
            return d, cut, f"CC {params} (frozen RUN/eval_test)"
        d = os.path.join(FB2, f"armiii_s{seed}", "eval_test_bestcut")
        f = os.path.join(d, "provenance.json")
        if not os.path.exists(_repo(FB2_SPEC)) or not os.path.exists(f):
            return None
        if not _in_fb2_spec(run, d, cut=cut):
            return f"{d}: not in task E's final spec {FB2_SPEC} at cut {cut}"
        p = json.load(open(f))
        ok = (os.path.normpath(p.get("run", "")) == os.path.normpath(run) and abs(float(p["cut_best_val"]) - cut) < 1e-6
              and bool(p.get("identity_check_s_star")) and all(p["identity_check_s_star"].values()) and _cut_in(d, cut))
        if not ok:
            return f"{f}: not run {run} at cut {cut}, or its s* identity check failed"
        return d, cut, f"CC {params} (task E post-freeze CC on the stored test scores)"
    d = os.path.join(FB2, f"armiii_s{seed}", f"{kind}_test")
    f = os.path.join(d, "provenance.json")
    if not os.path.exists(_repo(FB2_SPEC)) or not os.path.exists(f):
        return None
    if not _in_fb2_spec(run, d, config=f"{fam},{params}"):
        return f"{d}: not in task E's final spec {FB2_SPEC} with configuration {fam},{params}"
    p = json.load(open(f))
    cfgs = json.load(open(os.path.join(d, "configs.json")))
    want = f"{fam},{params}"
    if (os.path.normpath(p["run"]) != os.path.normpath(run) or p["config"] != want
            or ",".join(str(x) for x in cfgs[0]) != want):
        return f"{d}: provenance {p.get('run')} / {p.get('config')} / {cfgs[0]} is not {run} / {want}"
    return d, 0, f"{fam} {params} (task E test output, configuration 0)"


def test_entries(val: list[dict]) -> tuple[list[dict], list[str]]:
    """Per model the per-seed test source (directory, cut / configuration index), each checked against the val working
    point of the same seed; models whose test outputs do not exist (yet) are skipped and listed."""
    spec1 = {m["label"]: m for m in json.load(open(_repo(TEST_SPEC)))}
    sbt = {m["label"]: m for m in json.load(open(_repo(SB_TEST)))["models"]}
    a2t = {m["label"]: m for m in json.load(open(_repo(A2_TEST)))["models"]}
    if hashlib.sha256(open(_repo(A2_TEST), "rb").read()).hexdigest() != A2_TEST_SHA256:
        raise SystemExit(f"{A2_TEST}: sha256 differs from the val-fixed spec of task A2")
    wpj = json.load(open(_repo(WP)))
    if hashlib.sha256(open(_repo(WP), "rb").read()).hexdigest() != WP_SHA256:
        raise SystemExit(f"{WP}: sha256 differs from the one recorded in {SELECTION}")
    vals = {m["key"]: m for m in val}
    out, skipped = [], []
    for key, label, short, group, vf, vlab, tsrc, tlab in MODELS:
        v = vals[key]
        ents = []
        for seed, ve in enumerate(v["entries"]):
            fam, params = _parse_val_wp(ve["wp"])
            run = ve["run"]
            if tsrc == TEST_SPEC:
                te = spec1[tlab]["entries"][seed]
                if os.path.normpath(te["run"]) != os.path.normpath(run):
                    raise SystemExit(f"{tlab} seed {seed}: test run {te['run']} is not the val run {run}")
                if te["sub"] == "trackbuild_test":
                    _check_selection(run, f"{fam},{params}")
                    ents.append({"run": run, "dir": os.path.join(run, "trackbuild_test"), "cut": 0,
                                 "wp": f"{fam} {params} (trackbuild_test, configuration 0)"})
                else:
                    if fam != "cc" or abs(float(te["cut"]) - float(params)) > 1e-9:
                        raise SystemExit(f"{tlab} seed {seed}: test cut {te['cut']} is not the val cut {params}")
                    d = te["sub"] if os.path.isabs(te["sub"]) else os.path.join(run, te["sub"])
                    ents.append({"run": run, "dir": d, "cut": float(te["cut"]), "wp": te["wp"]})
            elif tsrc == SB_TEST:
                te = sbt[tlab]["entries"][seed]
                if os.path.normpath(te["run"]) != os.path.normpath(run) or te["config"] != f"{fam},{params}":
                    raise SystemExit(f"{tlab} seed {seed}: test {te['run']} {te['config']} is not val {run} {fam},{params}")
                prov = json.load(open(os.path.join(te["sub"], "provenance.json")))
                cfgs = json.load(open(os.path.join(te["sub"], "configs.json")))
                if (os.path.normpath(prov["run"]) != os.path.normpath(run) or prov["config"] != te["config"]
                        or ",".join(str(x) for x in cfgs[0]) != te["config"]):
                    raise SystemExit(f"{te['sub']}: provenance does not match {run} {te['config']}")
                ents.append({"run": run, "dir": te["sub"], "cut": 0, "wp": te["wp"]})
            elif tsrc == A2_TEST:   # task A2's test output of the val-best configuration (made once, read only)
                te = a2t[tlab]["entries"][seed]
                be = a2t[_a2_best(group)]["entries"][seed]
                want = f"{fam},{params}"
                if (os.path.normpath(te["run"]) != os.path.normpath(run) or te["config"] != want
                        or te["val_wp"] != ve["wp"] or (te["sub"], te["config"]) != (be["sub"], be["config"])):
                    raise SystemExit(f"{tlab} seed {seed}: test {te['run']} {te['config']} ({te['val_wp']}) is not the "
                                     f"val-best {run} {ve['wp']}")
                prov = json.load(open(os.path.join(te["sub"], "provenance.json")))
                cfgs = json.load(open(os.path.join(te["sub"], "configs.json")))
                if (os.path.normpath(prov["run"]) != os.path.normpath(run) or prov["config"] != want
                        or len(cfgs) != 1 or ",".join(str(x) for x in cfgs[0]) != want
                        or prov.get("spec_sha256") != A2_TEST_SHA256
                        or prov.get("merge_model_sha256") != te["merge_model_sha256"]
                        or os.path.normpath(prov.get("merge_model", "")) != os.path.normpath(te["merge_model"])):
                    raise SystemExit(f"{te['sub']}: provenance does not match {run} {want} and A2's merge model")
                ents.append({"run": run, "dir": te["sub"], "cut": 0, "wp": te["wp"], "merge_model": te["merge_model"],
                             "merge_model_sha256": te["merge_model_sha256"], "test_job": prov.get("slurm_job")})
            else:   # acorn (ttbar+HNL): frozen eval_test or the task-E outputs
                if fam == "cc" and key == "mix_cc":
                    want = float(wpj["cuts_at_FR_star"]["iii"][str(seed)])
                    if abs(want - float(params)) > 1e-9:
                        raise SystemExit(f"acorn (ttbar+HNL) seed {seed}: val cut {params} is not the preregistered {want}")
                r = _armiii_test(tlab, run, seed, fam, params)
                if r is None or isinstance(r, str):
                    why = f"seed {seed}: " + ("no test output (task E)" if r is None else r)
                    ents = None
                    break
                ents.append({"run": run, "dir": r[0], "cut": r[1], "wp": r[2]})
        if ents is None:
            skipped.append(f"{label} ({why})")
            continue
        out.append({**{k: v[k] for k in ("key", "label", "short", "group")}, "entries": ents})
    return out, skipped


# --------------------------------------------------------------------------------------------------------------------
# counting


def matched_flags(pop: pd.DataFrame, pp: pd.DataFrame, where: str) -> np.ndarray:
    """dm_matched of every population row (pp: event_id, particle_id, dm_matched at one working point); every
    population particle must appear exactly once."""
    t = pop[["event_id", "particle_id"]].merge(pp[["event_id", "particle_id", "dm_matched"]],
                                               on=["event_id", "particle_id"], how="left", validate="one_to_one")
    miss = t.dm_matched.isna()
    if miss.any():
        raise SystemExit(f"{where}: {int(miss.sum())} of {len(t)} population particles missing in the per-particle table")
    return t.dm_matched.to_numpy(dtype=float)


def seed_counts(rs: BT.Resampler, pop: pd.DataFrame, ok: np.ndarray, split: str) -> dict:
    """{(event set, population, quantity): (point, replicates, k, n)} of one seed."""
    code = split_event_id(pop.event_id.to_numpy())[0]
    pm = population_masks(pop)
    qm = quantity_masks(pop.r_start, pop.d0)
    ev = pop.event_id.to_numpy()
    out = {}
    for s, codes in EVENT_SETS[split].items():
        sm = np.isin(code, codes)
        for pname in POPULATIONS:
            if (s, pname) in UNDEFINED:
                continue
            for q in QUANTITIES:
                m = sm & pm[pname] & qm[q]
                sums = rs.unit_sums(ev[m], ok[m], np.ones(int(m.sum()))) if m.any() else {}
                if not sums:
                    out[(s, pname, q)] = (np.nan, np.full(rs.b, np.nan), 0.0, 0.0)
                else:
                    out[(s, pname, q)] = rs.ratio(sums)
    return out


class Model:
    def __init__(self, key: str, label: str, short: str, group: str, runs: list[str], wps: list[str], seeds: list[dict]):
        self.key, self.label, self.short, self.group, self.runs, self.wps = key, label, short, group, runs, wps
        self.seeds = seeds

    def pr(self, k) -> tuple[list, list]:
        return [s[k][0] for s in self.seeds], [s[k][1] for s in self.seeds]

    def kn(self, k) -> tuple[list, list]:
        return [s[k][2] for s in self.seeds], [s[k][3] for s in self.seeds]


def summary(m: Model, k) -> dict:
    p, r = m.pr(k)
    ks, ns = m.kn(k)
    ns = np.asarray(ns, dtype=float)
    if not np.all(ns == ns[0]):
        raise SystemExit(f"{m.label} {k}: population sizes differ between seeds {ns}")
    n = float(ns[0])
    out = {"n_per_seed": int(n), "k_pooled": int(round(float(np.sum(ks)))), "n_pooled": int(round(float(ns.sum()))),
           "per_seed": ";".join(f"{x:.10g}" for x in p), "n_seeds": len(p)}
    if n == 0:
        return {**out, "value": np.nan, "seed_sd": np.nan, "boot_sd": np.nan, "combined": np.nan}
    d = BS.seed_mean(p, r)
    rm = np.mean(np.stack(r), axis=0)
    lo, hi = np.nanpercentile(rm, BT.PCT) if np.isfinite(rm).any() else (np.nan, np.nan)
    c_lo, c_hi = BT.cp(d["mean"] * n, n)
    return {**out, "value": d["mean"], "seed_sd": d["s_seeds"], "boot_sd": d["sigma_boot"], "combined": d["sigma"],
            "boot68_lo": float(lo), "boot68_hi": float(hi), "cp68_lo": c_lo, "cp68_hi": c_hi}


def difference(new: Model, base: Model, k) -> dict:
    pn, rn = new.pr(k)
    pb, rb = base.pr(k)
    if new.kn(k)[1][0] == 0:
        return {"value": np.nan}
    d = BS.linear_combination([(1.0, pn, rn), (-1.0, pb, rb)])
    rep = np.mean(np.stack(rn), axis=0) - np.mean(np.stack(rb), axis=0)
    lo, hi = np.nanpercentile(rep, BT.PCT) if np.isfinite(rep).any() else (np.nan, np.nan)
    out = {"value": d["value"], "boot_sd": d["sigma_boot"], "seed_sd": d["sigma_seeds"], "combined": d["sigma"],
           "z": d["z"], "boot68_lo": float(lo), "boot68_hi": float(hi)}
    if (new.key, base.key) in SAME_NETWORKS:
        ds = np.asarray(pn) - np.asarray(pb)
        st = float(np.std(ds, ddof=1) / np.sqrt(len(ds)))
        out["seed_term_paired"] = st
        out["combined_paired"] = float(np.sqrt(d["sigma_boot"] ** 2 + st ** 2))
        out["per_seed_difference"] = ";".join(f"{x:.10g}" for x in ds)
    return out


def evaluate(split: str, models: list[dict], rs: BT.Resampler, pop: pd.DataFrame, expected_events: np.ndarray,
             info: pd.DataFrame | None = None):
    """Model objects of one split from the per-particle tables at the fixed working points. With `info`
    (hnl.eval.bootstrap_tables.info_table of the split's events) each model also carries round 1's quantities
    (primary efficiency etc., hnl.eval.bootstrap_tables.counts) from the same tables, for the identity checks."""
    out = []
    for m in models:
        seeds, r1 = [], []
        for e in m["entries"]:
            if split == "val":
                pp, pe, d = BT.val_tables(e["run"], e["wp"])
            else:
                rt = TS.RunTest(e["run"], e["cut"], sub=e["dir"])
                pp, pe, d = rt.pp, rt.pe, e["dir"]
            ev = np.unique(pe.event_id.to_numpy())
            if not np.array_equal(ev, expected_events) or len(pe) != len(ev):
                raise SystemExit(f"{d} at {e['wp']}: {len(ev)} events ({len(pe)} rows), expected {len(expected_events)}")
            if pp.duplicated(["event_id", "particle_id"]).any():
                raise SystemExit(f"{d} at {e['wp']}: duplicate particles")
            ok = matched_flags(pop, pp, f"{d} at {e['wp']}")
            seeds.append(seed_counts(rs, pop, ok, split))
            if info is not None:
                r1.append(BT.counts(rs, pp, pe, info))
            print(f"[{split}] {m['short']} {e['run']} {e['wp']}: {int(ok.sum())}/{len(ok)} matched", flush=True)
        mod = Model(m["key"], m["label"], m["short"], m["group"], [e["run"] for e in m["entries"]],
                    [e["wp"] for e in m["entries"]], seeds)
        mod.r1 = BT.Model(m["label"], m["short"], r1, rs, mod.wps, mod.runs, m["group"]) if r1 else None
        out.append(mod)
    return out


BOOT_REF = "results/bootstrap_v1.csv"
SB_TABLE = "results/strong_baselines_v1.csv"
DIFF_REF = {"val": "results/strong_baselines_v1_val_differences.csv", "test": "results/strong_baselines_v1_differences.csv"}
BOOT_SHORT = {"canto_l": "Canto-L", "canto_h": "Canto-H", "canto_b": "Canto-B", "canto_l_mwsm": "Canto-L MWS+M",
              "hnl_mwsm": "acorn (HNL) + MWS+M", "ttbar_mwsm": "acorn (ttbar) + MWS+M",
              "mix_mwsm": "acorn (ttbar+HNL) + MWS+M", "hnl_ccbest": "acorn (HNL) CC*", "hnl_cc": "acorn (HNL) CC",
              "ttbar_cc": "acorn (ttbar) CC", "mix_ccbest": "acorn (ttbar+HNL) CC*", "mix_cc": "acorn (ttbar+HNL) CC"}
BOOT_TABLES = {"val": ("val_ranking_v3", "val_same_builder"), "test": ("test_release_card",)}


def identity_checks(split: str, models: list[Model], tol: float = 1e-12, tol_boot: float = 1e-10) -> list[tuple]:
    """Round 1's primary efficiency of every model (value, seed sd, event-bootstrap sd, combined; results/bootstrap_v1.csv
    and results/strong_baselines_v1.csv) and the differences Canto-L minus each round-1 baseline
    (results/strong_baselines_v1*_differences.csv), recomputed here from the same tables and replicate tables.
    results/bootstrap_v1.csv holds 10 significant digits (tolerance tol_boot); the other files full precision (tol)."""
    boot = pd.read_csv(_repo(BOOT_REF))
    boot = boot[(boot.scope == split) & (boot.quantity == "eff_primary") & boot.table.isin(BOOT_TABLES[split])]
    sb = pd.read_csv(_repo(SB_TABLE))
    sb = sb[sb.set == split]
    dref = pd.read_csv(_repo(DIFF_REF[split]))
    dref = dref[dref.quantity == "primary"]
    vlab = {k: vl for k, _, _, _, vf, vl, *_ in MODELS if vf == SB_VAL}
    checks = []
    for m in models:
        s = BT.summary(m.r1, "eff_primary")
        r = boot[boot.short == BOOT_SHORT.get(m.key, "-")]
        if len(r) == 1:
            r = r.iloc[0]
            for a, b in (("value", "value"), ("seed_sd", "seed_sd"), ("boot_sd", "boot_sd"), ("combined", "combined")):
                checks.append((f"{split} {m.short} primary {a} (bootstrap_v1)", s[a], float(r[b]),
                               abs(s[a] - float(r[b])) <= tol_boot))
        r = sb[sb.label == vlab.get(m.key, "-")]
        if len(r) == 1:
            r = r.iloc[0]
            for a, b in (("value", "eff_primary"), ("seed_sd", "eff_primary_sd")):
                checks.append((f"{split} {m.short} primary {a} (strong_baselines_v1)", s[a], float(r[b]),
                               abs(s[a] - float(r[b])) <= tol))
    new = next(m for m in models if m.key == NEW)
    for m in models:
        r = dref[(dref.new == vlab["canto_l"]) & (dref.baseline == vlab.get(m.key, "-"))]
        if m.key == NEW or len(r) != 1:
            continue
        r = r.iloc[0]
        d = BT.difference(new.r1, m.r1, "eff_primary")
        for a, b in (("value", "difference"), ("boot_sd", "sigma_boot"), ("seed_sd", "sigma_seeds"),
                     ("combined", "sigma")):
            checks.append((f"{split} Canto-L - {m.short} primary {a} (strong_baselines_v1 differences)", d[a],
                           float(r[b]), abs(d[a] - float(r[b])) <= tol))
    return checks + a2_identity_checks(split, models, tol)


def a2_identity_checks(split: str, models: list[Model], tol: float = 1e-12) -> list[tuple]:
    """Task A2's primary efficiency of its val-best acorn rows and of Canto-L (results/strong_baselines_v3.csv: value,
    seed sd; test also the event-bootstrap sd and the total) and the differences Canto-L minus each of them
    (results/strong_baselines_v3_differences.csv: both values, both seed sds, both totals, the difference, its
    bootstrap, seed and total sigma), recomputed through this module's loading and replicate tables."""
    lab = {k: vl for k, _, _, _, vf, vl, *_ in MODELS if vf == A2_VAL}
    if not any(m.key in lab for m in models):
        return []
    tab = pd.read_csv(_repo(A2_TABLE))
    tab = tab[tab.set == split]
    dif = pd.read_csv(_repo(A2_DIFF))
    dif = dif[(dif.set == split) & (dif.block == "Canto-L minus acorn") & (dif.quantity == "primary")]
    new = next(m for m in models if m.key == NEW)
    new_label = "Canto-L (frozen: learned merging)"
    cols = [("value", "eff_primary"), ("seed_sd", "eff_primary_sd")]
    if split == "test":
        cols += [("boot_sd", "primary_sigma_boot"), ("combined", "primary_sigma_total")]
    checks = []
    for m in [new] + [m for m in models if m.key in lab]:
        s = BT.summary(m.r1, "eff_primary")
        r = tab[tab.label == (new_label if m.key == NEW else lab[m.key])]
        if len(r) != 1:
            raise SystemExit(f"{A2_TABLE}: {len(r)} {split} rows for {m.label!r}")
        r = r.iloc[0]
        for a, b in cols:
            checks.append((f"{split} {m.short} primary {a} (strong_baselines_v3)", s[a], float(r[b]),
                           abs(s[a] - float(r[b])) <= tol))
    sn = BT.summary(new.r1, "eff_primary")
    for m in models:
        if m.key not in lab:
            continue
        r = dif[(dif.new == new_label) & (dif.baseline == lab[m.key])]
        if len(r) != 1:
            raise SystemExit(f"{A2_DIFF}: {len(r)} {split} rows for Canto-L minus {lab[m.key]!r}")
        r = r.iloc[0]
        d = BT.difference(new.r1, m.r1, "eff_primary")
        sb = BT.summary(m.r1, "eff_primary")
        pairs = [(d["value"], "difference"), (d["boot_sd"], "sigma_boot"), (d["seed_sd"], "sigma_seeds"),
                 (d["combined"], "sigma"), (sn["value"], "value_new"), (sn["combined"], "sigma_new"),
                 (sn["seed_sd"], "s_seeds_new"), (sb["value"], "value_base"), (sb["combined"], "sigma_base"),
                 (sb["seed_sd"], "s_seeds_base")]
        for x, b in pairs:
            checks.append((f"{split} Canto-L - {m.short} primary {b} (strong_baselines_v3 differences)", x,
                           float(r[b]), abs(x - float(r[b])) <= tol))
    return checks


def rows(split: str, models: list[Model]) -> pd.DataFrame:
    new = next(m for m in models if m.key == NEW)
    keys = sorted(new.seeds[0], key=lambda k: (list(EVENT_SETS[split]).index(k[0]), POPULATIONS.index(k[1]),
                                               QUANTITIES.index(k[2])))
    out = []
    for k in keys:
        s, pname, q = k
        for m in models:
            out.append({"split": split, "kind": "efficiency", "event_set": s, "population": pname, "quantity": q,
                        "model": m.label, "short": m.short, "group": m.group, "base": "",
                        "working_points": " | ".join(m.wps), **summary(m, k)})
        for m in models:
            if m.key == NEW:
                continue
            d = difference(new, m, k)
            if np.isnan(d["value"]):
                continue
            out.append({"split": split, "kind": "difference", "event_set": s, "population": pname, "quantity": q,
                        "model": new.label, "short": f"Canto-L - {m.short}", "group": m.group, "base": m.label,
                        "n_per_seed": summary(new, k)["n_per_seed"], "n_seeds": 3, **d})
    df = pd.DataFrame(out)
    df.insert(df.columns.get_loc("n_seeds") + 1, "B", models[0].seeds[0][keys[0]][1].shape[0])
    return df


# --------------------------------------------------------------------------------------------------------------------
# commands


def expected_events(split: str) -> np.ndarray:
    from hnl.data.splits import read_split
    pairs = (BT.VAL_SETS if split == "val" else BS.SETS)
    return np.sort(np.concatenate([np.int64(SAMPLE_CODE[s]) * 100_000_000 + read_split(s, sp).astype(np.int64)
                                   for parts in pairs.values() for s, sp in parts]))


def run_split(split: str, out: str, add: bool = False) -> None:
    """val, or the one-time test read-out. add: evaluate only the models that are not in the existing read-out of the
    split (test: acorn (ttbar+HNL) rows from task E; val and test: task A2's val-best acorn rows, round 3 task F2) and
    append their rows; Canto-L is recomputed for the differences and must equal its rows; every other row stays as
    written."""
    t0 = time.time()
    path = os.path.join(out, f"{split}_rows.csv")
    meta_path = os.path.join(out, f"{split}_meta.json")
    old = None
    if add:
        if not os.path.exists(path):
            raise SystemExit(f"--add appends to a read-out that exists ({path} missing)")
        old = json.load(open(meta_path))
    elif split == "test" and os.path.exists(path):
        raise SystemExit(f"{path} exists: the test read-out is done once (post-freeze, descriptive); --add "
                         f"evaluates only models that were missing")
    os.makedirs(out, exist_ok=True)
    events = expected_events(split)
    pop = population_table(events)
    info = BT.info_table(events)
    val = val_entries()
    if split == "val":
        models, skipped = val, []
        rs = val_resampler()
        for m in models:   # A2's learned-merging rows: the val grid part and merge model behind each working point
            if m["key"] in LM_KEYS:
                m["entries"] = [{**e, **a2_val_merge_model(e["run"], e["wp"])} for e in m["entries"]]
    else:
        models, skipped = test_entries(val)
        rs = test_resampler()
        for m in models:   # the same merge model on val and test
            if m["key"] in LM_KEYS:
                for e, ve in zip(m["entries"], next(v for v in val if v["key"] == m["key"])["entries"]):
                    vm = a2_val_merge_model(ve["run"], ve["wp"])
                    if (os.path.normpath(vm["merge_model"]) != os.path.normpath(e["merge_model"])
                            or vm["merge_model_sha256"] != e["merge_model_sha256"]):
                        raise SystemExit(f"{e['dir']}: merge model differs from the val grid's {vm['merge_model']}")
    if old is not None:
        done = {m["key"] for m in old["models"]}
        new_models = [m for m in models if m["key"] not in done]
        if not new_models:
            print(f"nothing to add: skipped models still missing: {skipped}")
            return
        models = [m for m in models if m["key"] == NEW] + new_models
    ms = evaluate(split, models, rs, pop, events, info)
    checks = identity_checks(split, ms)
    bad = [c for c in checks if not c[3]]
    for c in bad:
        print("IDENTITY CHECK FAILED:", c)
    print(f"[{split}] identity checks: {len(checks)}, failed {len(bad)}", flush=True)
    df = rows(split, ms)
    if old is not None:
        prev = pd.read_csv(path, float_precision="round_trip")
        # Canto-L is recomputed only for the differences: its rows must equal the first read-out exactly
        key = ["event_set", "population", "quantity"]
        a = df[(df.kind == "efficiency") & (df.model == ms[0].label)].set_index(key)[["value", "boot_sd", "n_per_seed"]]
        b = prev[(prev.kind == "efficiency") & (prev.model == ms[0].label)].set_index(key)[["value", "boot_sd", "n_per_seed"]]
        a, b = a.sort_index(), b.sort_index()
        if not a.index.equals(b.index) or not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=0, atol=1e-12,
                                                          equal_nan=True):
            raise SystemExit(f"Canto-L recomputed in --add differs from the first {split} read-out: inputs changed")
        df = df[~((df.kind == "efficiency") & (df.model == ms[0].label))]
        extra = [c for c in df.columns if c not in prev.columns]
        if extra:
            raise SystemExit(f"--add: columns {extra} are not in {path}")
        old["models"] += [{"key": m["key"], "label": m["label"], "entries": m["entries"]} for m in new_models]
        old["skipped_models"] = skipped
        old.setdefault("additions", []).append({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                                "slurm_job": os.environ.get("SLURM_JOB_ID"),
                                                "models": [m["key"] for m in new_models],
                                                "identity_checks": len(checks), "identity_failed": len(bad)})
        # the earlier rows are kept byte for byte (no CSV round trip): the new rows are appended to the file's text
        text = open(path).read()
        if not text.endswith("\n"):
            raise SystemExit(f"{path}: no final newline")
        with open(path + ".tmp", "w") as f:
            f.write(text)
            df.reindex(columns=prev.columns).to_csv(f, index=False, header=False)
        if pd.read_csv(path + ".tmp", float_precision="round_trip").shape[0] != len(prev) + len(df):
            raise SystemExit(f"{path}.tmp: row count after the append")
        os.replace(path + ".tmp", path)
        json.dump(old, open(meta_path, "w"), indent=1, default=str)
        print(json.dumps({"added": [m["key"] for m in new_models], "skipped_models": skipped}))
        return
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    pop_cols = ["event_id", "sample", "particle_id", "pdg", "parent_pdg", "process", "gen", "pt", "eta", "n_hits",
                "n_pixel_hits", "r_first", "r_start", "d0", "free_d0", "fix_d0"]
    pop[pop_cols].to_parquet(os.path.join(out, f"{split}_population.parquet"), index=False)
    meta = {"split": split, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "n_events": int(len(events)), "replicates": rs.label,
            "B": rs.b, "models": [{"key": m["key"], "label": m["label"], "entries": m["entries"]} for m in models],
            "skipped_models": skipped, "population_checks": population_checks(pop),
            "identity_checks": {"n": len(checks), "failed": [list(map(str, c)) for c in bad]},
            "population_counts": {s: {p: int((population_masks(pop)[p]
                                               & np.isin(split_event_id(pop.event_id.to_numpy())[0], c)).sum())
                                      for p in POPULATIONS} for s, c in EVENT_SETS[split].items()},
            "elapsed_s": round(time.time() - t0, 1)}
    json.dump(meta, open(meta_path, "w"), indent=1, default=str)
    print(json.dumps({k: meta[k] for k in ("split", "n_events", "skipped_models", "population_counts", "elapsed_s")}))


# --------------------------------------------------------------------------------------------------------------------
# report


def _f(x, nd=4) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _cell(r: pd.Series) -> str:
    """value +- combined (k/n pooled over the three seeds: each particle counted once per seed)."""
    if r.n_per_seed == 0:
        return "-"
    return f"{_f(r.value)} +- {_f(r.combined)} ({int(r.k_pooled)}/{int(r.n_pooled)})"


def _dcell(r: pd.Series) -> str:
    return f"{100 * r.value:+.2f} +- {100 * r.combined:.2f} (z {r.z:.1f})"


def _order(labels) -> list[str]:
    """Models in table order: by group (Canto, acorn (HNL), acorn (ttbar), acorn (ttbar+HNL)), within a group task A2's
    val-best row first, then the order of MODELS; labels not in MODELS keep their order at the end."""
    pos = {lab: (GROUP_ORDER.index(g), k not in LM_KEYS, i) for i, (k, lab, _, g, *_) in enumerate(MODELS)}
    labels = list(dict.fromkeys(labels))
    return sorted(labels, key=lambda x: pos.get(x, (len(GROUP_ORDER), True, labels.index(x))))


def _table(df: pd.DataFrame, s: str, pop: str, qs, header: list[str], disp=lambda x: x, models=None) -> list[str]:
    e = df[(df.kind == "efficiency") & (df.event_set == s) & (df.population == pop)]
    if models is not None:
        e = e[e.model.isin(models)]
    if e.empty or e[e.quantity == "all"].n_per_seed.iloc[0] == 0:
        return ["(no particle of this population in this set)", ""]
    n = e[e.model == e.model.iloc[0]].set_index("quantity").n_per_seed
    lines = ["| model | " + " | ".join(f"{h} ({int(n[q])})" for h, q in zip(header, qs)) + " |",
             "|---|" + "---|" * len(header)]
    for model in _order(e.model.unique()):
        g = e[e.model == model].set_index("quantity")
        lines.append(f"| {disp(model)} | " + " | ".join(_cell(g.loc[q]) for q in qs) + " |")
    return lines + [""]


def _dtable(df: pd.DataFrame, s: str, pop: str, qs, header: list[str], disp=lambda x: x, bases=None) -> list[str]:
    d = df[(df.kind == "difference") & (df.event_set == s) & (df.population == pop)]
    if bases is not None:
        d = d[d.base.isin(bases)]
    if d.empty:
        return []
    lines = ["| Canto-L minus | " + " | ".join(header) + " |", "|---|" + "---|" * len(header)]
    for base in _order(d.base.unique()):
        g = d[d.base == base].set_index("quantity")
        cells = []
        for q in qs:
            if q not in g.index:
                cells.append("-")
                continue
            r = g.loc[q]
            c = _dcell(r)
            if isinstance(r.get("seed_term_paired"), float) and not np.isnan(r.get("seed_term_paired")):
                c += f"; seed term paired by seed: +- {100 * r.combined_paired:.2f}"
            cells.append(c)
        lines.append(f"| {disp(base)} | " + " | ".join(cells) + " |")
    return lines + [""]


def _census_tables(cen: pd.DataFrame) -> list[str]:
    def c(r, a, b=None, d=None):
        x = f"{int(r[a]):,}"
        if b is not None:
            x += f" ({int(r[b]):,}" + (f" / {int(r[d]):,}" if d else "") + ")"
        return x
    L = ["| sample | split | events | K0S final state (with hits / leaving the volume) | Lambda final state (with hits "
         "/ leaving the volume) | K0S daughters (with hits) | Lambda daughters (with hits) | V0-daughter targets "
         "(K0S / Lambda) | conversion e+- (targets) | \\|parent_pid\\| = 22 (targets) | rows with parent_pid != 0 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in cen[cen.split != "all"].iterrows():
        L.append(f"| {r['sample']} | {r.split} | {int(r.events):,} | "
                 f"{c(r, 'k0s_final_state', 'k0s_final_state_with_hits', 'k0s_final_state_leaving_volume')} | "
                 f"{c(r, 'lambda_final_state', 'lambda_final_state_with_hits', 'lambda_final_state_leaving_volume')} | "
                 f"{c(r, 'k0s_daughters', 'k0s_daughters_with_hits')} | {c(r, 'lambda_daughters', 'lambda_daughters_with_hits')} | "
                 f"{int(r.v0_daughter_targets):,} ({int(r.k0s_daughter_targets):,} / {int(r.lambda_daughter_targets):,}) | "
                 f"{c(r, 'conv_secondaries', 'conv_targets')} | {c(r, 'parent22', 'parent22_targets')} | "
                 f"{int(r.parent_pid_nonzero):,} of {int(r.particles):,} |")
    a = cen[cen.split == "all"]
    L += ["", "Process codes (every event of every sample): conversion secondaries that are not e+- "
          f"{int(a.conv_secondaries_not_electron.sum())}; bremsstrahlung secondaries that are not photons "
          f"{int(a.brems_secondaries_not_photon.sum())}; particles with process code 1 (decay) or 4 (nuclear "
          f"interaction) {int(a.process_1_or_4.sum())}; simulated secondaries (generation >= 1) with a non-zero "
          f"parent_pid {int(a.gen_ge1_parent_nonzero.sum())}; V0-daughter targets that are not generation 0 "
          f"{int((a.k0s_daughter_targets - a.k0s_daughter_targets_gen0 + a.lambda_daughter_targets - a.lambda_daughter_targets_gen0).sum())}; "
          f"V0 daughters at the primary vertex (vsec 0) {int(a.v0_daughters_at_vsec0.sum())}; conversion e+- with "
          f"table pT = 0 (final state) {int(a.conv_secondaries_pt0.sum())} of {int(a.conv_secondaries.sum())} "
          f"({100 * a.conv_secondaries_pt0.sum() / a.conv_secondaries.sum():.1f} %). Split train_ttbar+HNL: the HNL "
          f"training events of acorn (ttbar+HNL) (hnl10_100 train index < 5000, hnl15_200 train index >= 5000).",
          ""]
    return L


def _checks_table(metas: dict) -> list[str]:
    L = ["| split | population | targets | r_start = chain head | same r_start bin as the chain head | same \\|d0\\| bin, "
         "free vs fixed-radius fit | median \\|d0\\| [mm] | median r_start [mm] | median \\|d0\\| / r_start |",
         "|---|---|---|---|---|---|---|---|---|"]
    for split, meta in metas.items():
        for pop, c in meta["population_checks"].items():
            L.append(f"| {split} | {POP_LABEL[pop]} | {c['n']} | {100 * c['r_start_equals_chain_head']:.1f} % | "
                     f"{100 * c['same_r_bin_as_chain_head']:.1f} % | {100 * c['same_d0_bin_free_vs_fixed_radius_fit']:.1f} % | "
                     f"{c['median_abs_d0']:.2f} | {c['median_r_start']:.1f} | {c['median_abs_d0_over_r_start']:.3f} |")
    return L + [""]


ROUND1_BEST = "acorn (HNL) + walkthrough + merging (val-best acorn (HNL) configuration)"
TITLE_V1 = "A second displaced population: V0 daughters (feedback round 2, task F)"


def _lm_section(df: pd.DataFrame, metas: dict, sec: int, disp) -> list[str]:
    """Canto-L and task A2's val-best acorn rows (learned merging), efficiencies and differences in every start-radius
    bin and pooled: V0 daughters on the HNL events, conversion e+- on the ttbar events."""
    lm = [lab for k, lab, *_ in MODELS if k in LM_KEYS]
    canto = next(lab for k, lab, *_ in MODELS if k == NEW)
    rq = ["all", "r_ge100", *R_BINS]
    rh = ["all", "r_start >= 100", *[f"r_start {Q_LABEL[q]}" for q in R_BINS]]
    L = [f"## {sec}. Canto-L against the val-best acorn configurations (task A2: MWS-layer + learned merging)", "",
         "Efficiencies (value +- combined (k/n pooled over the seeds); column headers: particles per seed) and "
         "differences Canto-L minus each val-best acorn row in points (paired event bootstrap + both seed terms; "
         "z = difference / combined), in every start-radius bin and pooled. The same rows are in the full tables of the "
         "later sections.", ""]
    sub = 1
    for split in ("val", "test"):
        if split not in metas:
            continue
        sdf = df[df.split == split]
        tag = "val" if split == "val" else "test, post-freeze descriptive"
        for pop, s in (("v0", f"hnl_joint_{split}"), ("conv", f"ttbar_{split}")):
            n = metas[split]["population_counts"][s][pop]
            L += [f"### {sec}.{sub} {POP_LABEL[pop]}, {SET_LABEL[s]} ({tag}; {n} targets per seed)", ""]
            sub += 1
            L += _table(sdf, s, pop, rq, rh, disp, models=[canto] + lm)
            L += ["Canto-L minus (points):", ""] + _dtable(sdf, s, pop, rq, rh, disp, bases=lm)
    return L


def report(directory: str, results: str, name: str, notes: str | None = None, title: str | None = None) -> None:
    cen = pd.read_csv(os.path.join(directory, "census.csv"))
    checks = open(os.path.join(directory, "checks.txt")).read()
    parts, metas = [], {}
    for split in ("val", "test"):
        f = os.path.join(directory, f"{split}_rows.csv")
        if os.path.exists(f):
            parts.append(pd.read_csv(f))
            metas[split] = json.load(open(os.path.join(directory, f"{split}_meta.json")))
    df = pd.concat(parts, ignore_index=True)
    jobs = ", ".join(f"{k}: Slurm job {v.get('slurm_job')}" + "".join(f" + {a['slurm_job']}" for a in v.get("additions", []))
                     for k, v in metas.items())
    lm_keys = [m["key"] for meta in metas.values() for m in meta["models"] if m["key"] in LM_KEYS]
    relabel = {ROUND1_BEST: "acorn (HNL) + walkthrough + merging (round 1's val-best acorn (HNL) configuration)"}
    disp = (lambda x: relabel.get(x, x)) if lm_keys else (lambda x: x)
    os.makedirs(results, exist_ok=True)
    # the repository CSV: every row of the main sets; the two halves of the joint HNL sets for all and r_start >= 100
    # only; working points in the markdown (section 1.4) instead of on every row; full tables stay in `directory`
    main = df.event_set.isin([s for v in MAIN_SETS.values() for s in v])
    out = df[(main | df.quantity.isin(["all", "r_ge100"])) & (df.n_per_seed > 0)].drop(columns=["working_points"])
    short = dict(zip(df.model, df.short))
    out = out.assign(base=out.base.map(lambda b: short.get(b, b) if isinstance(b, str) and b else ""))
    out.to_csv(os.path.join(results, f"{name}.csv"), index=False, float_format="%.10g")
    cen.to_csv(os.path.join(results, f"{name}_census.csv"), index=False)
    L = [f"# {title or TITLE_V1}", "",
         "Generated by `python -m hnl.eval.second_population report` from "
         f"{directory} ({jobs}); code src/hnl/eval/second_population.py, test tests/test_second_population.py. "
         "Validation numbers are validation; test numbers are post-freeze descriptive summaries at configurations fixed "
         "on val (nothing chosen on test; results/claims_test_v1.json unchanged). Within this simulation (ACTS generic "
         "detector, no pileup).", ""]
    if lm_keys:
        L += ["Rows added in round 3 (task F2): task A2's val-best configuration of each acorn arm, MWS-layer + learned "
              "merging for every seed, with A2's merge model of each run (results/strong_baselines_v3_val_entries.csv, "
              "results/strong_baselines_v3_spec_test.json; docs/feedback/round3_A2_decisions.md). Val from A2's united "
              "val grids (RUN/trackbuild_val_fb3_all, learned-merging part RUN/trackbuild_val_fb3_lmerge), test from "
              "A2's one-time test outputs (runs/feedback_v3/strong_baselines_a2/arm<A>_s<S>/lmerge_test); no track "
              "candidate was rebuilt and no test job was run. The rows were appended with `--add` to a copy of the round-2 "
              "read-out; every earlier row is unchanged. In the tables below the round-2 row "
              f"'{ROUND1_BEST}' is shown as '{disp(ROUND1_BEST)}' (the CSV keeps the round-2 label): under A2's full "
              "builder menu the val-best acorn (HNL) configuration is learned merging.", ""]
    L += ["Cells: value +- combined, value = mean of the three seeds' efficiencies (each pooled over the events of the set), "
          "combined = sqrt(boot sd^2 + seed sd^2 / 3) with the event bootstrap of round 1 (val B = 1000, test the "
          "preregistered B = 500 tables); (k/n) pooled over the three seeds (each particle counted once per seed). "
          "Column headers give the particles per seed. Differences in points, paired over the events, with both seed "
          "terms (z = difference / combined). The HNL sets pool hnl10_100 and hnl15_200, which share their underlying "
          "events: most V0 daughters appear in both samples, the bootstrap resamples both samples' event i together, so "
          "quote the combined uncertainty, not a binomial interval on the pooled count. Every column (seed sd, boot sd, "
          "68 % Clopper-Pearson and bootstrap intervals, per-seed values, the hnl10_100 and hnl15_200 halves) is in the CSV. Where k = n in every seed the bootstrap sd is 0: quote the 68 % "
          "Clopper-Pearson interval of the CSV there.",
          ""]
    if notes and os.path.exists(notes):
        L += [open(notes).read().strip(), ""]
    L += ["## 1. Populations", "", "### 1.1 Census (every event of every sample)", ""] + _census_tables(cen)
    L += ["### 1.2 Checks by hand (a few events)", "", "```", checks.strip(), "```", "",
          "### 1.3 Population checks and identity checks", ""] + _checks_table(metas)
    L += ["### 1.4 Models and working points (per seed 0 / 1 / 2; fixed on val, applied unchanged on test)", "",
          "| split | model | working points |", "|---|---|---|"]
    for split, meta in metas.items():
        for m in meta["models"]:
            L.append(f"| {split} | {disp(m['label'])} | " + " / ".join(e["wp"] for e in m["entries"]) + " |")
    L += [""]
    if lm_keys:
        L += ["Merge models of the learned-merging rows (task A2; sha256 checked against the model's training "
              "provenance on val and against the provenance of A2's test output; the same model on val and test):", "",
              "| split | model | run | merge model sha256 | Slurm job of the val grid / test output |", "|---|---|---|---|---|"]
        for split, meta in metas.items():
            for m in meta["models"]:
                if m["key"] not in LM_KEYS:
                    continue
                for e in m["entries"]:
                    job = e.get("val_grid_job") if split == "val" else e.get("test_job")
                    L.append(f"| {split} | {m['label'].split(' + ')[0]} | {os.path.basename(e['run'])} | "
                             f"{e['merge_model_sha256'][:16]}... | {job} |")
        L += [""]
    for split, meta in metas.items():
        ic = meta.get("identity_checks", {})
        adds = meta.get("additions", [])
        L += [f"Identity checks ({split}): round 1's primary efficiencies (value, seed sd, event-bootstrap sd, combined) "
              f"and the differences Canto-L minus each round-1 baseline, recomputed through this module's loading and "
              f"replicate tables: {ic.get('n', 0)} checks, {len(ic.get('failed', []))} failed (tolerance 1e-12; 1e-10 "
              f"against results/bootstrap_v1.csv, which holds 10 significant digits)"
              + "".join(f"; addition {a['utc'][:16]} UTC (job {a['slurm_job']}"
                        + (", rows " + ", ".join(a["models"]) if any(k in LM_KEYS for k in a["models"]) else "")
                        + f"): {a['identity_checks']} checks, {a['identity_failed']} failed" for a in adds) + ".", ""]
    if lm_keys:
        L += ["The additions of round 3 also recompute task A2's primary efficiency of Canto-L and of each val-best "
              "acorn row (results/strong_baselines_v3.csv: value and seed sd; test also the event-bootstrap sd and the "
              "total) and the differences Canto-L minus each of them (results/strong_baselines_v3_differences.csv: the "
              "difference, its bootstrap, seed and total sigma, and both rows' value, seed sd and total), tolerance "
              "1e-12.", ""]
    rq = ["all", "r_ge100", *R_BINS]
    rh = ["all", "r_start >= 100", *[f"r_start {Q_LABEL[q]}" for q in R_BINS]]
    dq = list(D0_BINS)
    dh = [f"\\|d0\\| {Q_LABEL[q]}" for q in D0_BINS]
    sec = 2
    if lm_keys:
        L += _lm_section(df, metas, sec, disp)
        sec += 1
    for split in ("val", "test"):
        if split not in metas:
            continue
        sdf = df[df.split == split]
        stitle = "Validation" if split == "val" else "Test (post-freeze, descriptive; nothing chosen on test)"
        L += [f"## {sec}. {stitle}", ""]
        if metas[split]["skipped_models"]:
            L += ["Rows without test outputs at the time of the read-out (not evaluated): "
                  + "; ".join(metas[split]["skipped_models"]), ""]
        sub = 1
        for pop in ("v0", "conv"):
            for s in MAIN_SETS[split]:
                if (s, pop) in UNDEFINED:
                    continue
                n = metas[split]["population_counts"][s][pop]
                L += [f"### {sec}.{sub} {POP_LABEL[pop]}, {SET_LABEL[s]} ({n} targets per seed)", ""]
                sub += 1
                if n == 0:
                    L += ["No particle of this population exists in these events (section 1.1).", ""]
                    continue
                L += ["Start radius r_start (innermost hit) [mm]:", ""] + _table(sdf, s, pop, rq, rh, disp)
                L += ["\\|d0\\| (truth-hit helix fit) [mm]:", ""] + _table(sdf, s, pop, dq, dh, disp)
                L += ["Differences [points]:", ""] + _dtable(sdf, s, pop, ["all", "r_ge100"], ["all", "r_start >= 100"],
                                                             disp)
                if pop == "v0":
                    for sp in ("k0s", "lambda"):
                        L += [f"{POP_LABEL[sp]}:", ""] + _table(sdf, s, sp, ["all", "r_ge100"], ["all", "r_start >= 100"],
                                                                disp)
        sec += 1
    open(os.path.join(results, f"{name}.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {os.path.join(results, name)}.md / .csv / _census.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("census", "checks", "val", "test"):
        a = sp.add_parser(c)
        a.add_argument("--out", required=True)
        if c in ("val", "test"):
            a.add_argument("--add", action="store_true", help="append only models missing in the existing read-out")
    r = sp.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--results", default=os.path.join(REPO, "results"))
    r.add_argument("--name", default="second_population_v1")
    r.add_argument("--notes", default=None)
    r.add_argument("--title", default=None, help="markdown title (default: the round-2 title)")
    a = ap.parse_args(argv)
    if a.cmd == "census":
        os.makedirs(a.out, exist_ok=True)
        c = census()
        c.to_csv(os.path.join(a.out, "census.csv"), index=False)
        pd.set_option("display.width", 250)
        print(c.to_string())
    elif a.cmd == "checks":
        os.makedirs(a.out, exist_ok=True)
        t = hand_checks()
        open(os.path.join(a.out, "checks.txt"), "w").write(t + "\n")
        print(t)
    elif a.cmd in ("val", "test"):
        run_split(a.cmd, a.out, add=getattr(a, "add", False))
    else:
        report(a.dir, a.results, a.name, a.notes, a.title)


if __name__ == "__main__":
    main()
