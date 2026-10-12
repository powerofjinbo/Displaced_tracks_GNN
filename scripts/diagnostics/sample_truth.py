"""Truth of the samples (paper draft 7, Fig. 2 and Table I): distributions and shares of the targets, from the
truth tables only (no model output is read).

Populations (target = pT > 1 GeV, |eta| < 4, >= 5 hits; the is_target column of derived/events/v1/particles):
  - ttbar_train_targets: every target of the 7999 ttbar training events (splits/ttbar_train.txt), all categories;
  - <sample>_daughters: the HNL-daughter targets (category "hnl") of every event of each HNL sample (the events of
    the split files: training, validation and test of hnl10_100 and hnl15_200, the external hnl15_100);
  - w_muons_hnl: the W-muon targets (category "w") of every event of the three HNL samples.
|d0| and r_v come from derived/truth_lookup/v1 (the robust-fit |d0| and the radius of the fitted daughter-pair
vertex, as in the evaluation); r_first is the transverse radius of the first hit of the particle's chain.

Checks (the script stops if one fails): targets per event over every event of each sample against NUMBERS D36
(ttbar 61.8) and D23 (HNL daughters 0.932 / 0.793 / 1.125); the primary population (HNL-daughter targets with
|d0| >= 10 mm or r_v >= 100 mm) per validation and test set against the n_per_seed column of
results/per_sample_v1.csv (692 / 639 validation, 603 / 549 / 3861 test, 5013 pooled).

Writes results/sample_truth_v1.md (counts, shares, medians, checks; one quoted value per line) and
results/sample_truth_v1_hist.csv (the binned arrays of Fig. 2: |d0| and pT fractions per bin over all targets of the
population, and the fraction with r_first > r on a 1 mm grid).

    sbatch ... --wrap "source env/activate.sh; python scripts/diagnostics/sample_truth.py"
"""
import glob
import os
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results")
SPLITS = os.path.join(REPO, "splits")
DERIVED = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived")
OUT_MD = os.path.join(RES, "sample_truth_v1.md")
OUT_CSV = os.path.join(RES, "sample_truth_v1_hist.csv")

SAMPLES = ("ttbar", "hnl10_100", "hnl15_200", "hnl15_100")
HNL = ("hnl10_100", "hnl15_200", "hnl15_100")
SPLIT_NAMES = {"ttbar": ("train", "val", "test"), "hnl10_100": ("train", "val", "test"),
               "hnl15_200": ("train", "val", "test"), "hnl15_100": ("external",)}
NAME = {"ttbar_train_targets": "ttbar targets, 7999 training events",
        "hnl10_100_daughters": "HNL-daughter targets, 10 GeV, 100 mm (all events)",
        "hnl15_200_daughters": "HNL-daughter targets, 15 GeV, 200 mm (all events)",
        "hnl15_100_daughters": "HNL-daughter targets, 15 GeV, 100 mm, external (all events)",
        "w_muons_hnl": "W-muon targets of the three HNL samples (all events)"}
D0_EDGES = np.logspace(-3, 3, 37)            # mm, 6 bins per decade
PT_EDGES = np.logspace(0, 2.5, 21)           # GeV, 8 bins per decade (targets have pT > 1 GeV)
R_GRID = np.arange(0.0, 1101.0, 1.0)         # mm, survival fraction N(r_first > r) / N
D0_BINS = [(0, 1), (1, 10), (10, 30), (30, 100), (100, 300), (300, np.inf)]
# Reference values (NUMBERS.md rows); the script only compares, it never adjusts.
REF_PER_EVENT = {"ttbar": ("61.8", 1), "hnl10_100": ("0.932", 3), "hnl15_200": ("0.793", 3),
                 "hnl15_100": ("1.125", 3)}
PRIMARY_SETS = {("val", "hnl10_100"): "hnl10_100_val", ("val", "hnl15_200"): "hnl15_200_val",
                ("test", "hnl10_100"): "hnl10_100_test", ("test", "hnl15_200"): "hnl15_200_test",
                ("external", "hnl15_100"): "hnl15_100_external"}


def read_split(sample, name):
    return np.loadtxt(os.path.join(SPLITS, f"{sample}_{name}.txt"), dtype=np.int64, comments="#", ndmin=1)


def load(sample):
    files = sorted(glob.glob(os.path.join(DERIVED, "events", "v1", "particles", sample, "shard_*.parquet")))
    cols = ["event", "particle_id", "category", "is_target", "pt", "n_hits", "r_first"]
    p = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    n_all_events = int(p.event.nunique())
    p = p[p.is_target].copy()
    t = pd.read_parquet(os.path.join(DERIVED, "truth_lookup", "v1", f"{sample}.parquet"),
                        columns=["event", "particle_id", "absd0", "rv"])
    p = p.merge(t, on=["event", "particle_id"], how="left", validate="one_to_one")
    split = np.full(len(p), "", dtype=object)
    events = []
    for name in SPLIT_NAMES[sample]:
        ev = read_split(sample, name)
        events.append(ev)
        split[np.isin(p.event.to_numpy(), ev)] = name
    p["split"] = split
    p["sample"] = sample
    return p, np.unique(np.concatenate(events)), n_all_events, len(files)


def frac_hist(x, edges, n):
    c, _ = np.histogram(x[np.isfinite(x)], bins=edges)
    return c, c / n


def summary(x: pd.DataFrame) -> dict:
    d0 = x.absd0.to_numpy(float)
    r1 = x.r_first.to_numpy(float)
    n = len(x)
    out = {"n": n, "d0_nan": int(np.isnan(d0).sum()),
           "n_d0_ge10": int((d0 >= 10).sum()), "n_r1_gt100": int((r1 > 100).sum()),
           "n_r1_gt200": int((r1 > 200).sum()),
           "med_pt": float(np.median(x.pt)), "med_d0": float(np.nanmedian(d0)), "med_r1": float(np.median(r1)),
           "med_hits": float(np.median(x.n_hits)), "d0_max": float(np.nanmax(d0)),
           "d0_under": int((d0 < D0_EDGES[0]).sum()), "d0_over": int((d0 >= D0_EDGES[-1]).sum()),
           "pt_over": int((x.pt.to_numpy(float) >= PT_EDGES[-1]).sum()), "r1_max": float(np.max(r1))}
    for lo, hi in D0_BINS:
        out[f"n_d0_{lo}_{hi}"] = int(((d0 >= lo) & (d0 < hi)).sum())
    return out


def pct(k, n, dec):
    return f"{100.0 * k / n:.{dec}f}"


def main():
    t0 = time.time()
    tabs, events, n_all = {}, {}, {}
    for s in SAMPLES:
        p, ev, na, nf = load(s)
        tabs[s], events[s], n_all[s] = p, ev, na
        print(f"{s}: {nf} shards, {na} events in the particle table, {len(ev)} in the split files, "
              f"{len(p)} targets ({time.time() - t0:.0f} s)", flush=True)

    # Targets per event over every event of each sample (the events of the split files, after the drop).
    per_event, lines_chk, problems = {}, [], []
    for s in SAMPLES:
        p = tabs[s]
        p_ev = p[p.event.isin(events[s])]
        k = len(p_ev) if s == "ttbar" else int((p_ev.category == "hnl").sum())
        per_event[s] = (k, len(events[s]), k / len(events[s]))
        ref, dec = REF_PER_EVENT[s]
        got = f"{k / len(events[s]):.{dec}f}"
        ok = got == ref
        problems += [] if ok else [f"targets per event {s}: {got} != {ref}"]
        lines_chk.append(f"- targets per event, {s}: {got} ({k} / {len(events[s])} events); reference {ref} "
                         f"({'D36' if s == 'ttbar' else 'D23'}): {'equal' if ok else 'DIFFERENT'}")
        if n_all[s] != len(events[s]):
            lines_chk.append(f"  (the particle table of {s} has {n_all[s]} events; the extra events are the "
                             f"dropped ones and are not counted)")

    # Primary population per validation / test set against results/per_sample_v1.csv.
    ps = pd.read_csv(os.path.join(RES, "per_sample_v1.csv"))
    prim = {}
    for (name, s), set_name in PRIMARY_SETS.items():
        p = tabs[s]
        disp = ((p.absd0 >= 10) | (p.rv >= 100)).fillna(False).to_numpy(bool)
        m = (p.split == name).to_numpy() & (p.category == "hnl").to_numpy() & disp
        k = int(m.sum())
        ref = ps.loc[ps["set"] == set_name, "n_per_seed"].unique()
        ok = len(ref) == 1 and int(ref[0]) == k
        prim[set_name] = k
        problems += [] if ok else [f"primary {set_name}: {k} != per_sample_v1.csv {list(ref)}"]
        lines_chk.append(f"- primary population, {set_name}: {k}; results/per_sample_v1.csv n_per_seed "
                         f"{', '.join(str(int(r)) for r in ref)}: {'equal' if ok else 'DIFFERENT'}")
    pooled = prim["hnl10_100_test"] + prim["hnl15_200_test"] + prim["hnl15_100_external"]
    ref = ps.loc[ps["set"] == "hnl_test_pooled", "n_per_seed"].unique()
    ok = len(ref) == 1 and int(ref[0]) == pooled
    problems += [] if ok else [f"primary pooled test: {pooled} != {list(ref)}"]
    lines_chk.append(f"- primary population, pooled test sets: {pooled}; results/per_sample_v1.csv hnl_test_pooled "
                     f"n_per_seed {', '.join(str(int(r)) for r in ref)}: {'equal' if ok else 'DIFFERENT'}")

    # Populations of Fig. 2.
    tt = tabs["ttbar"]
    pops = {"ttbar_train_targets": tt[tt.split == "train"]}
    for s in HNL:
        p = tabs[s]
        pops[f"{s}_daughters"] = p[(p.split != "") & (p.category == "hnl")]
    pops["w_muons_hnl"] = pd.concat([tabs[s][(tabs[s].split != "") & (tabs[s].category == "w")] for s in HNL])
    n_train = len(np.unique(tt.event[tt.split == "train"]))
    n_train_split = len(read_split("ttbar", "train"))
    stats = {k: summary(v) for k, v in pops.items()}

    rows = []
    for key, x in pops.items():
        n = len(x)
        c, f = frac_hist(x.absd0.to_numpy(float), D0_EDGES, n)
        rows += [dict(panel="absd0", series=key, x_lo=lo, x_hi=hi, count=int(ci), n_total=n, value=fi)
                 for lo, hi, ci, fi in zip(D0_EDGES[:-1], D0_EDGES[1:], c, f)]
        c, f = frac_hist(x.pt.to_numpy(float), PT_EDGES, n)
        rows += [dict(panel="pt", series=key, x_lo=lo, x_hi=hi, count=int(ci), n_total=n, value=fi)
                 for lo, hi, ci, fi in zip(PT_EDGES[:-1], PT_EDGES[1:], c, f)]
        r1 = np.sort(x.r_first.to_numpy(float))
        above = n - np.searchsorted(r1, R_GRID, side="right")          # N(r_first > r)
        rows += [dict(panel="r_first_survival", series=key, x_lo=r, x_hi=r, count=int(a), n_total=n, value=a / n)
                 for r, a in zip(R_GRID, above)]
    hist = pd.DataFrame(rows)
    hist.to_csv(OUT_CSV, index=False, float_format="%.8g")

    # The quoted shares.
    st = stats["ttbar_train_targets"]
    z = {s: stats[f"{s}_daughters"] for s in HNL}
    zp = {s: 100.0 * z[s]["n_r1_gt100"] / z[s]["n"] for s in HNL}
    zr = [f"{v:.0f}" for v in (min(zp.values()), max(zp.values()))]

    L = []
    L.append("# Truth of the samples (sample_truth_v1)")
    L.append("")
    L.append("Written by scripts/diagnostics/sample_truth.py (truth tables only: derived/events/v1/particles, "
             "derived/truth_lookup/v1, splits/). Target = pT > 1 GeV, |eta| < 4, >= 5 hits. r_first = transverse "
             "radius of the first hit of the particle's chain; \"first hit beyond 100 mm\" means r_first > 100 mm. "
             "Shares are over every target of the population (targets without a |d0| fit count as not >= 10 mm). "
             "Fig. 2 (paper/figures/fig15_samples.pdf) plots results/sample_truth_v1_hist.csv.")
    L.append("")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')}; Slurm job {os.environ.get('SLURM_JOB_ID', 'none')}.")
    L.append("")
    L.append("## Quoted values (one per line)")
    L.append("")
    L.append(f"- SA01 ttbar training targets with |d0| >= 10 mm: {pct(st['n_d0_ge10'], st['n'], 1)} % "
             f"({st['n_d0_ge10']} of {st['n']})")
    L.append(f"- SA02 ttbar training targets with first hit beyond 100 mm: {pct(st['n_r1_gt100'], st['n'], 1)} % "
             f"({st['n_r1_gt100']} of {st['n']})")
    L.append("- SA03 HNL-daughter targets with first hit beyond 100 mm, all events: "
             f"{zr[0]}-{zr[1]} % (10 GeV 100 mm / 15 GeV 200 mm / 15 GeV 100 mm ext.: "
             f"{' / '.join(f'{zp[s]:.0f}' for s in HNL)} %, "
             f"{' / '.join(str(z[s]['n_r1_gt100']) for s in HNL)} of {' / '.join(str(z[s]['n']) for s in HNL)})")
    L.append(f"- SA04 targets in Fig. 2: ttbar training events {st['n']} ({n_train} events); HNL-daughter targets, "
             f"all events, 10 GeV 100 mm / 15 GeV 200 mm / 15 GeV 100 mm ext.: "
             f"{' / '.join(str(z[s]['n']) for s in HNL)}; W-muon targets of the HNL samples, all events: "
             f"{stats['w_muons_hnl']['n']}")
    L.append("- SA05 primary population per seed, test, 10 GeV 100 mm / 15 GeV 200 mm / 15 GeV 100 mm ext.: "
             f"{prim['hnl10_100_test']} / {prim['hnl15_200_test']} / {prim['hnl15_100_external']} "
             f"(pooled {pooled})")
    L.append("")
    L.append("## Checks")
    L.append("")
    L += lines_chk
    L.append("- primary population per seed, validation, 10 GeV 100 mm / 15 GeV 200 mm: "
             f"{prim['hnl10_100_val']} / {prim['hnl15_200_val']} (NUMBERS D31)")
    L.append(f"- ttbar training events with at least one target: {n_train}; splits/ttbar_train.txt: {n_train_split}")
    L.append(f"- result: {'all checks passed' if not problems else 'FAILED: ' + '; '.join(problems)}")
    L.append("")
    L.append("## Per population")
    L.append("")
    L.append("| population | targets | \\|d0\\| >= 10 mm (%) | first hit > 100 mm (%) | first hit > 200 mm (%) | "
             "median pT (GeV) | median \\|d0\\| (mm) | median r_first (mm) | median hits | no \\|d0\\| fit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, s in stats.items():
        L.append(f"| {NAME[k]} | {s['n']} | {pct(s['n_d0_ge10'], s['n'], 2)} | {pct(s['n_r1_gt100'], s['n'], 2)} | "
                 f"{pct(s['n_r1_gt200'], s['n'], 2)} | {s['med_pt']:.2f} | {s['med_d0']:.3g} | {s['med_r1']:.1f} | "
                 f"{s['med_hits']:.0f} | {s['d0_nan']} |")
    L.append("")
    L.append("Targets per |d0| bin (mm):")
    L.append("")
    L.append("| population | " + " | ".join(f"[{lo:g}, {hi:g})" for lo, hi in D0_BINS) + " | max \\|d0\\| (mm) |")
    L.append("|---|" + "---|" * (len(D0_BINS) + 1))
    for k, s in stats.items():
        L.append(f"| {NAME[k]} | " + " | ".join(str(s[f'n_d0_{lo}_{hi}']) for lo, hi in D0_BINS)
                 + f" | {s['d0_max']:.0f} |")
    L.append("")
    L.append("Outside the plotted ranges of Fig. 2 (counted in the denominators, not drawn): "
             + "; ".join(f"{k}: |d0| < 1e-3 mm {s['d0_under']}, |d0| >= 1e3 mm {s['d0_over']}, "
                         f"pT >= {PT_EDGES[-1]:.0f} GeV {s['pt_over']}, max r_first {s['r1_max']:.0f} mm"
                         for k, s in stats.items()) + ".")
    L.append("")
    L.append(f"Histogram file: results/sample_truth_v1_hist.csv ({len(hist)} rows; panels absd0 and pt: fraction of "
             "the population's targets per bin, value = count / n_total; panel r_first_survival: fraction with "
             "r_first > x_lo on a 1 mm grid).")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {OUT_MD} and {OUT_CSV} ({time.time() - t0:.0f} s)")
    if problems:
        raise SystemExit("checks failed: " + "; ".join(problems))


if __name__ == "__main__":
    main()
