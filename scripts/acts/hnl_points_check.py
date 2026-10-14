"""Truth-only checks of the produced HNL points (docs/hnl_points_spec.md section 5). No model is run.

scan (one sample, sbatch):
  * shards: hnl.data.shards.check (contiguous ranges from 0, one member per event, Zenodo keys and dtypes);
  * SHA-256 of every shard and truth sidecar, written as <sample dir>/SHA256SUMS (sha256sum format, paths relative to
    hnl_lrt/generated/, so `cd generated && sha256sum -c <dir>/SHA256SUMS` verifies) and copied to the repository;
  * every event read through the tar reader (hnl.data.shards.iter_events_from, check=True) with the format invariants
    of hnl.data.scans.event_checks (as addendum G step T1) and of scripts/acts/check_reader.py (hid a permutation,
    every pid in the table, ids exact in float64, edges within one particle and between different modules, one chain
    per particle, E = hits - particles with hits), and the corruption rule of docs/definitions.md section 1 (a hit
    more than 50 mm outside its module's padded envelope, Zenodo envelopes of derived/scans/v1);
  * per event: W muons (|parent| = 24, |PDG| = 13) and N daughters (|parent| = 50) in the particle table, and the
    generator truth of the sidecar (W muon and two N daughters recorded, next() calls, proper lifetime);
  * the label-exchange test of scripts/acts/sample_stats.py (hnl.eval.dense_jet_check) on events 0-499, read through
    the tar reader.
summary (all samples, after scan and the census of scripts/acts/hnl_census.py shards):
  * results/hnl_points_truth_v1.md (+ .json): checks, lifetime, populations, r_v and |d0| tables per point;
  * splits/hnl_<m>_<ctau>_all.txt: every kept event.

    python scripts/acts/hnl_points_check.py scan --sample hnl5_50 --out DIR [--workers 8]
    python scripts/acts/hnl_points_check.py swapref --zenodo hnl10_100 --split val --out DIR    (Zenodo reference)
    python scripts/acts/hnl_points_check.py summary --scan DIR --census DIR
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hnl.data import shards  # noqa: E402
from hnl.data.npz import COL, GENERATED_ROOT, particle_ids  # noqa: E402
from hnl.data.scans import CORRUPT_EXCESS_MM, event_checks  # noqa: E402
from hnl.eval.dense_jet_check import chain_orders  # noqa: E402

FORMAT_TRUE = ("hid_is_perm", "pid_in_table", "particle_ids_unique", "edges_eq_hits_minus_particles",
               "edge_rows_in_range")
FORMAT_ZERO = ("n_edges_diff_pid", "n_edges_same_module", "n_bad_geoid_bits", "n_unexpected_volume", "n_nonfinite_x",
               "n_particles_not_chain")
REPO_SHA = os.path.join(REPO, "results", "hnl_points_truth_v1")
RV_EDGES = [0, 10, 30, 100, 200, 300, np.inf]
D0_EDGES = [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf]
SWAP_EVENTS = 500
# reference (in-stack) point of each new point (spec section 1) and the planned expectation (spec section 4)
REFERENCE = {"hnl5_50": "hnl10_100_ref", "hnl20_200": "hnl10_100_ref", "hnl15_500": "hnl15_200_ref"}
EXPECTED_PRIMARY_3000 = {"hnl5_50": 1635, "hnl20_200": 2350, "hnl15_500": 1136, "hnl10_100_ref": 1880,
                         "hnl15_200_ref": 1970}
ZENODO_PRIMARY_TEST = {"hnl10_100": 603, "hnl15_200": 549}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reader_invariants(ev: dict) -> None:
    """The asserts of scripts/acts/check_reader.py (one)."""
    n = len(ev["x"])
    assert np.array_equal(np.sort(ev["hid"]), np.arange(n)), "hid"
    ids = particle_ids(ev["particles"])
    assert np.isin(ev["pid"], ids).all(), "pid not in table"
    e = ev["layerless_true_edges"]
    assert (ev["pid"][e[0]] == ev["pid"][e[1]]).all(), "edge across particles"
    assert (ev["layers"][e[0]] != ev["layers"][e[1]]).all(), "edge within a module"
    chain_orders(ev["pid"], e[0], e[1])
    assert e.shape[1] == n - len(np.unique(ev["pid"])), "edge count"


def _scan_shard(args):
    directory, first, last = args
    from hnl.eval.addendum_g_t1 import envelope_excess
    rows = []
    for i, ev in shards.iter_events_from(directory, range(first, last + 1), check=True):
        err = ""
        try:
            reader_invariants(ev)
        except AssertionError as exc:
            err = str(exc) or "assert"
        st, _ = event_checks(ev)
        mx, nbad, nunseen = envelope_excess(ev)
        fmt_ok = all(bool(st[k]) for k in FORMAT_TRUE) and all(int(st[k]) == 0 for k in FORMAT_ZERO)
        P = ev["particles"]
        parent = P[:, COL["parent_pid"]].astype(np.int64)
        pdg = P[:, COL["particle_type"]].astype(np.int64)
        ids = particle_ids(P)
        rows.append({
            "event": i, "n_hits": st["n_hits"], "n_particles": st["n_particles"],
            "n_particles_with_hits": st["n_particles_with_hits"], "reader_ok": err == "", "reader_error": err,
            "format_ok": fmt_ok, "env_max_excess_mm": mx, "n_env_bad": nbad, "n_env_unseen_module": nunseen,
            "corrupted": mx > CORRUPT_EXCESS_MM,
            "n_w_table": int(((np.abs(parent) == 24) & (np.abs(pdg) == 13)).sum()),
            "n_d_table": int((np.abs(parent) == 50).sum()),
            "n_d_table_muons": int(((np.abs(parent) == 50) & (np.abs(pdg) == 13)).sum()),
            "n_gen_rows_labeled": int((parent != 0).sum()),
            "ids": ids,
        })
    return rows


def _swap_one(args):
    """sample_stats.one on an event read through the tar reader (or a Zenodo file path)."""
    import sample_stats
    kind, where, i = args
    if kind == "tar":
        sample_stats.load = lambda _path: shards.load_event_from(where, i, check=True)
        return sample_stats.one(f"{where}::{i}")
    return sample_stats.one(os.path.join(where, f"{i}.npz"))


def swap_test(kind: str, where: str, events, workers: int) -> dict:
    import sample_stats
    with cf.ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_swap_one, [(kind, where, int(i)) for i in events], chunksize=8))
    tot = sample_stats.merge(rows)
    keep = ("swap_same_n", "swap_same_x", "swap_ctrl_n", "swap_ctrl_x", "chains", "chains_R_monotone",
            "particles_repeated_module", "edges_same_module", "n_events", "edges")
    out = {k: tot[k] for k in keep}
    sep = tot["edges"]["sep"]
    hi = [k for k in range(len(sep) - 1) if sep[k] >= 2]
    for kind_ in ("same", "ctrl"):
        n = int(np.sum(np.array(tot[f"swap_{kind_}_n"])[hi]))
        x = int(np.sum(np.array(tot[f"swap_{kind_}_x"])[hi]))
        out[f"{kind_}_gt2mm"] = {"n": n, "x": x, "f": x / n if n else None}
    out["chains_monotone_frac"] = tot["chains_R_monotone"]["sum"] / tot["chains"]["sum"]
    return out


def scan(sample: str, out: str, workers: int):
    s = shards.POINTS[sample]
    d = shards.sample_dir(s)
    tdir = d + "_truth"
    os.makedirs(out, exist_ok=True)
    res = {"sample": sample, "dir": d, "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    res["shards_check"] = shards.check(d)
    sh = shards.list_shards(d)
    sides = sorted(glob.glob(os.path.join(tdir, "truth_*.npz")))
    res["n_shards"], res["n_sidecars"] = len(sh), len(sides)
    res["shard_bytes"] = int(sum(os.path.getsize(p) for _, _, p in sh))
    res["sidecar_bytes"] = int(sum(os.path.getsize(p) for p in sides))
    # checksums (paths relative to generated/)
    files = [p for _, _, p in sh] + sides
    with cf.ThreadPoolExecutor(4) as ex:
        sums = list(ex.map(sha256_file, files))
    lines = [f"{h}  {os.path.relpath(p, GENERATED_ROOT)}\n" for h, p in zip(sums, files)]
    sha_path = os.path.join(d, "SHA256SUMS")
    if os.path.exists(sha_path):
        if open(sha_path).read() != "".join(lines):
            raise SystemExit(f"{sha_path} exists and differs")
    else:
        tmp = sha_path + ".part"
        with open(tmp, "w") as f:
            f.writelines(lines)
        os.replace(tmp, sha_path)
    os.makedirs(REPO_SHA, exist_ok=True)
    with open(os.path.join(REPO_SHA, f"{s.dirname}.sha256"), "w") as f:
        f.writelines(lines)
    res["sha256sums"] = sha_path
    # every event through the tar reader
    with cf.ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_scan_shard, [(d, a, b) for a, b, _ in sh]))
    rows = [r for p in parts for r in p]
    # generator truth of the sidecars
    G = []
    for f in sides:
        with np.load(f, allow_pickle=False) as z:
            G.append(pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith("g_")}))
    G = pd.concat(G, ignore_index=True)
    ids_by_event = {r["event"]: set(r.pop("ids").tolist()) for r in rows}
    E = pd.DataFrame(rows).sort_values("event").reset_index(drop=True)
    G = G.sort_values("event").reset_index(drop=True)
    if not np.array_equal(E.event.to_numpy(), G.event.to_numpy()):
        raise SystemExit("events of the shards and of the sidecars differ")
    E["gen_w_recorded"] = G.mu_w_id.to_numpy() != 0
    E["gen_two_daughters"] = (G.d1_id.to_numpy() != 0) & (G.d2_id.to_numpy() != 0) & \
        (G.n_visible_daughters.to_numpy() == 2)
    E["gen_w_in_table"] = [int(w) in ids_by_event[e] for e, w in zip(G.event, G.mu_w_id.astype(np.int64))]
    E["gen_d_in_table"] = [int(int(a) in ids_by_event[e]) + int(int(b) in ids_by_event[e])
                           for e, a, b in zip(G.event, G.d1_id.astype(np.int64), G.d2_id.astype(np.int64))]
    E["n_tries"] = G.n_tries.to_numpy()
    E["n_tau"] = G.n_tau.to_numpy()
    E["rv_true"] = np.hypot(G.n_dec_x, G.n_dec_y).to_numpy()
    E["dropped"] = ~E.reader_ok | ~E.format_ok | E.corrupted
    E.to_parquet(os.path.join(out, f"{sample}_events.parquet"), index=False)
    res["events"] = int(len(E))
    res["reader_failures"] = E.loc[~E.reader_ok, ["event", "reader_error"]].to_dict("records")
    res["format_failures"] = int((~E.format_ok).sum())
    res["corrupted"] = E.loc[E.corrupted, "event"].tolist()
    res["env_max_excess_mm"] = float(E.env_max_excess_mm.max())
    res["n_env_unseen_module_hits"] = int(E.n_env_unseen_module.sum())
    res["dropped"] = E.loc[E.dropped, "event"].tolist()
    res["events_one_w_table"] = int((E.n_w_table == 1).sum())
    res["events_by_n_daughters_table"] = {str(k): int(v) for k, v in E.n_d_table.value_counts().sort_index().items()}
    res["daughters_not_muons"] = int((E.n_d_table - E.n_d_table_muons).sum())
    res["events_gen_w_recorded"] = int(E.gen_w_recorded.sum())
    res["events_gen_two_daughters"] = int(E.gen_two_daughters.sum())
    res["events_gen_w_in_table"] = int(E.gen_w_in_table.sum())
    res["events_gen_daughters_in_table"] = {str(k): int(v) for k, v in E.gen_d_in_table.value_counts().sort_index().items()}
    res["events_n_tries_gt1"] = int((E.n_tries > 1).sum())
    res["hits_per_event_mean"] = float(E.n_hits.mean())
    tau = E.n_tau.to_numpy(float)
    res["lifetime"] = {"ctau0": s.hnl_ctau_mm, "mean": float(tau.mean()),
                       "se": float(tau.std(ddof=1) / np.sqrt(len(tau))),
                       "ratio": float(tau.mean() / s.hnl_ctau_mm),
                       "ratio_se": float(tau.std(ddof=1) / np.sqrt(len(tau)) / s.hnl_ctau_mm)}
    # label-exchange test on events 0-499 through the tar reader
    res["swap"] = swap_test("tar", d, range(SWAP_EVENTS), workers)
    with open(os.path.join(out, f"{sample}_scan.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: res[k] for k in ("sample", "events", "format_failures", "corrupted", "events_one_w_table",
                                          "events_gen_two_daughters", "lifetime")}, default=str))
    print("swap same > 2 mm", res["swap"]["same_gt2mm"], "control", res["swap"]["ctrl_gt2mm"])


def swapref(sample: str, split: str, out: str, workers: int):
    from hnl.data.npz import SAMPLES, UNPACKED
    from hnl.data.splits import read_split
    if split == "test":
        raise SystemExit("the test split is not read")
    ev = read_split(sample, split)[:SWAP_EVENTS]
    res = swap_test("file", os.path.join(UNPACKED, SAMPLES[sample].dirname), ev, workers)
    res["sample"], res["split"], res["events_used"] = sample, split, f"first {len(ev)} of the {split} list"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"zenodo_{sample}_{split}_swap.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(sample, split, res["same_gt2mm"], res["ctrl_gt2mm"])


# ------------------------------------------------------------------------------------------------- summary
def _fmt(x, nd=1):
    return f"{x:,.{nd}f}"


def summary(scan_dir: str, census_dir: str):
    from hnl_closure import derived
    order = ["hnl10_100_ref", "hnl5_50", "hnl20_200", "hnl15_200_ref", "hnl15_500"]
    R = {}
    for key in order:
        s = shards.POINTS[key]
        sc = json.load(open(os.path.join(scan_dir, f"{key}_scan.json")))
        E = pd.read_parquet(os.path.join(census_dir, f"{key}_events.parquet"))
        S = pd.read_parquet(os.path.join(census_dir, f"{key}_signal.parquet"))
        drop = set(sc["dropped"])
        E = E[~E.event.isin(drop)]
        S = S[~S.event.isin(drop)]
        E, S = derived(E, S, s.hnl_mass_gev)
        d = S[S.role == "d"]
        w = S[S.role == "w"]
        t = d[d.is_target]
        pr = d[d.primary]
        rv_tab = []
        for k in range(len(RV_EDGES) - 1):
            m = (t.rv >= RV_EDGES[k]) & (t.rv < RV_EDGES[k + 1])
            rv_tab.append({"lo": RV_EDGES[k], "hi": RV_EDGES[k + 1], "targets": int(m.sum()),
                           "primary": int((m & t.primary).sum())})
        m = t.rv.isna()
        rv_tab.append({"lo": "no vertex", "hi": "", "targets": int(m.sum()), "primary": int((m & t.primary).sum())})
        d0_tab = []
        a = t.d0.abs()
        for k in range(len(D0_EDGES) - 1):
            mm = (a >= D0_EDGES[k]) & (a < D0_EDGES[k + 1])
            d0_tab.append({"lo": D0_EDGES[k], "hi": D0_EDGES[k + 1], "targets": int(mm.sum()),
                           "primary": int((mm & t.primary).sum())})
        rvt = E.rv_true
        R[key] = {
            "dir": s.dirname, "mass": s.hnl_mass_gev, "ctau0": s.hnl_ctau_mm, "scan": sc,
            "events_kept": int(len(E)),
            "daughter_targets": int(len(t)), "daughter_targets_per_event": float(len(t) / len(E)),
            "primary": int(len(pr)), "primary_per_event": float(len(pr) / len(E)),
            "primary_se_per_event": float(E.n_d_primary.std(ddof=1) / np.sqrt(len(E))),
            "primary_expected": EXPECTED_PRIMARY_3000[key],
            "primary_rv_ge_300": int((pr.rv >= 300).sum()), "primary_absd0_ge_100": int((pr.d0.abs() >= 100).sum()),
            "primary_ge3_pixel": int((pr.n_pixel_hits >= 3).sum()),
            "w_targets": int(w.is_target.sum()), "w_ge5_hits": int((w.n_hits >= 5).sum()),
            "rv_table": rv_tab, "d0_table": d0_tab,
            "rv_true_fractions": {"<100": float((rvt < 100).mean()), "100-300": float(((rvt >= 100) & (rvt < 300)).mean()),
                                  "300-700": float(((rvt >= 300) & (rvt < 700)).mean()), ">1000": float((rvt > 1000).mean())},
            "median_open_angle": float(E.open_angle.median()),
            "median_absd0_daughter_targets": float(a.median()),
            "fitted_rv_minus_true_median_mm": float(np.nanmedian(E.rv - E.rv_true)),
        }
        kept = sorted(set(E.event.tolist()))
        path = os.path.join(REPO, "splits", f"hnl_{s.dirname[4:-3]}_all.txt")
        with open(path, "w") as f:
            f.write(f"# {s.dirname}: all {len(kept)} kept events (docs/hnl_points_spec.md section 5; evaluation only, "
                    f"no split); dropped: {sorted(drop)}\n")
            f.writelines(f"{i}\n" for i in kept)
        R[key]["split_file"] = os.path.relpath(path, REPO)
    ref = {}
    for f in sorted(glob.glob(os.path.join(scan_dir, "zenodo_*_swap.json"))):
        z = json.load(open(f))
        ref[f"{z['sample']} {z['split']}"] = z
    out = {"points": R, "zenodo_swap_reference": ref,
           "git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    with open(os.path.join(REPO, "results", "hnl_points_truth_v1.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_md(out)


def write_md(out: dict):
    R = out["points"]
    L = ["# HNL points: truth summary of the produced samples", "",
         "Produced per `docs/hnl_points_spec.md` (sections 4-6, changes of section 11); checks of its section 5 by "
         "`scripts/acts/hnl_points_check.py` (scan per sample, then summary over the census of "
         "`scripts/acts/hnl_census.py shards`). Truth only: no model was run on these events and nothing was chosen on "
         "them. All five samples are evaluation samples (spec section 7).", ""]
    L += ["## Samples and checks", "",
          "| sample | m_N / c tau0 | events | shards / sidecars | size (GB) | reader and format invariants | corrupted "
          "(largest envelope excess) | one W muon in the table | generator: W muon and two N daughters | next() > 1 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in R.items():
        sc = r["scan"]
        L.append(f"| `{k}` (`{r['dir']}`) | {r['mass']:g} GeV / {r['ctau0']:g} mm | {sc['events']:,} | "
                 f"{sc['n_shards']} / {sc['n_sidecars']} | {(sc['shard_bytes'] + sc['sidecar_bytes']) / 1e9:.2f} | "
                 f"{sc['events'] - len(sc['reader_failures']) - sc['format_failures']:,} of {sc['events']:,} pass | "
                 f"{len(sc['corrupted'])} ({sc['env_max_excess_mm']:.1f} mm) | {sc['events_one_w_table']:,} | "
                 f"{sc['events_gen_two_daughters']:,} | {sc['events_n_tries_gt1']} |")
    L += ["", "Daughters in the particle table per event (0 / 1 / 2) and lifetime of the generator truth (each sample's "
          "own mean; the card's c tau is fixed in spec section 10):", "",
          "| sample | events with 0 / 1 / 2 N daughters in the table | daughters that are not muons | mean proper lifetime "
          "(mm) | / c tau0 |", "|---|---|---|---|---|"]
    for k, r in R.items():
        sc = r["scan"]
        nd = sc["events_by_n_daughters_table"]
        lt = sc["lifetime"]
        L.append(f"| `{k}` | {nd.get('0', 0)} / {nd.get('1', 0)} / {nd.get('2', 0)} | {sc['daughters_not_muons']} | "
                 f"{lt['mean']:.1f} +- {lt['se']:.1f} | {lt['ratio']:.3f} +- {lt['ratio_se']:.3f} |")
    L += ["", "Label check (same-module label-exchange test of `hnl.eval.dense_jet_check` through "
          "`scripts/acts/sample_stats.py`, events 0-499 of each sample read through the tar reader; flagged pairs "
          "with transverse separation > 2 mm):", "",
          "| sample | same module | different modules of one layer (control) | chains monotone in sqrt(r^2+z^2) |",
          "|---|---|---|---|"]
    for k, r in R.items():
        sw = r["scan"]["swap"]
        a, b = sw["same_gt2mm"], sw["ctrl_gt2mm"]
        L.append(f"| `{k}` | {100 * a['f']:.2f} % ({a['x']:,} / {a['n']:,}) | {100 * b['f']:.2f} % ({b['x']:,} / "
                 f"{b['n']:,}) | {100 * sw['chains_monotone_frac']:.1f} % |")
    for lab, z in out["zenodo_swap_reference"].items():
        a, b = z["same_gt2mm"], z["ctrl_gt2mm"]
        L.append(f"| Zenodo {lab} (reference, {z['events_used']}) | {100 * a['f']:.2f} % ({a['x']:,} / {a['n']:,}) | "
                 f"{100 * b['f']:.2f} % ({b['x']:,} / {b['n']:,}) | {100 * z['chains_monotone_frac']:.1f} % |")
    L += ["", "## Populations", "",
          "Targets: pT > 1 GeV, |eta| < 4, >= 5 hits; primary population: N-daughter targets with fitted |d0| >= 10 mm "
          "or fitted r_v >= 100 mm (`docs/definitions.md` sections 3-4; truth fits of `hnl.truth.fit_events`, as in "
          "the closure census). Expected: spec section 4. Zenodo test sets for scale: 603 (hnl10_100), 549 "
          "(hnl15_200).", "",
          "| sample | kept events | N-daughter targets | primary population | per event | expected (spec) | of which "
          "r_v >= 300 mm | of which abs(d0) >= 100 mm | with >= 3 pixel hits | W-muon targets |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in R.items():
        L.append(f"| `{k}` | {r['events_kept']:,} | {r['daughter_targets']:,} | {r['primary']:,} | "
                 f"{r['primary_per_event']:.3f} +- {r['primary_se_per_event']:.3f} | {r['primary_expected']:,} | "
                 f"{r['primary_rv_ge_300']:,} | {r['primary_absd0_ge_100']:,} | {r['primary_ge3_pixel']:,} | "
                 f"{r['w_targets']:,} |")
    keys = list(R)
    L += ["", "N-daughter targets (primary in parentheses) by fitted r_v (mm):", "",
          "| r_v | " + " | ".join(f"`{k}`" for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for i, row in enumerate(R[keys[0]]["rv_table"]):
        lab = row["lo"] if row["lo"] == "no vertex" else (f"[{row['lo']:g}, {row['hi']:g})" if np.isfinite(row["hi"])
                                                           else f">= {row['lo']:g}")
        L.append(f"| {lab} | " + " | ".join(f"{R[k]['rv_table'][i]['targets']:,} ({R[k]['rv_table'][i]['primary']:,})"
                                             for k in keys) + " |")
    L += ["", "N-daughter targets (primary in parentheses) by fitted |d0| (mm):", "",
          "| abs(d0) | " + " | ".join(f"`{k}`" for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for i, row in enumerate(R[keys[0]]["d0_table"]):
        lab = f"[{row['lo']:g}, {row['hi']:g})" if np.isfinite(row["hi"]) else f">= {row['lo']:g}"
        L.append(f"| {lab} | " + " | ".join(f"{R[k]['d0_table'][i]['targets']:,} ({R[k]['d0_table'][i]['primary']:,})"
                                             for k in keys) + " |")
    L += ["", "Generator truth of the N (all kept events): decay-radius fractions, median opening angle of the pair "
          "(end-state momenta) and median |d0| of the daughter targets:", "",
          "| sample | r_v < 100 mm | 100-300 mm | 300-700 mm | > 1 m | median opening angle (rad) | median abs(d0) of "
          "daughter targets (mm) |", "|---|---|---|---|---|---|---|"]
    for k, r in R.items():
        f = r["rv_true_fractions"]
        L.append(f"| `{k}` | {f['<100']:.3f} | {f['100-300']:.3f} | {f['300-700']:.3f} | {f['>1000']:.3f} | "
                 f"{r['median_open_angle']:.3f} | {r['median_absd0_daughter_targets']:.1f} |")
    L += ["", "## Statistical reach", "",
          "Binomial standard error of an efficiency on the primary population (events are independent; the two "
          "daughters of an event can be correlated, so these are lower bounds), and of the difference between a new "
          "point and its in-stack reference (independent samples). A comparison of two models on the same events is "
          "paired and has a smaller error.", "",
          "| sample | primary population | SE at eff 0.90 (points) | SE at eff 0.80 | reference | SE of the difference at "
          "0.90 / 0.80 |", "|---|---|---|---|---|---|"]
    for k, r in R.items():
        n = r["primary"]
        se9, se8 = 100 * np.sqrt(0.09 / n), 100 * np.sqrt(0.16 / n)
        ref = REFERENCE.get(k)
        if ref:
            m = R[ref]["primary"]
            d9, d8 = 100 * np.sqrt(0.09 / n + 0.09 / m), 100 * np.sqrt(0.16 / n + 0.16 / m)
            L.append(f"| `{k}` | {n:,} | {se9:.2f} | {se8:.2f} | `{ref}` | {d9:.2f} / {d8:.2f} |")
        else:
            L.append(f"| `{k}` | {n:,} | {se9:.2f} | {se8:.2f} | - | - |")
    L += ["", "## Files", ""]
    for k, r in R.items():
        L.append(f"- `{k}`: `{r['scan']['sha256sums']}` (copy: `results/hnl_points_truth_v1/{r['dir']}.sha256`); "
                 f"kept events: `{r['split_file']}`.")
    L.append("")
    with open(os.path.join(REPO, "results", "hnl_points_truth_v1.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--sample", required=True, choices=sorted(shards.POINTS))
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("swapref")
    p.add_argument("--zenodo", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("summary")
    p.add_argument("--scan", required=True)
    p.add_argument("--census", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        scan(a.sample, a.out, a.workers)
    elif a.cmd == "swapref":
        swapref(a.zenodo, a.split, a.out, a.workers)
    else:
        summary(a.scan, a.census)


if __name__ == "__main__":
    main()
