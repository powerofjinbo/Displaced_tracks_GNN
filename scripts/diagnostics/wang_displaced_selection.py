"""Which HNL daughters make up the displaced denominator of Wang et al. (arXiv:2203.08800, Fig. 3)? Truth only: no model
output is read, and no efficiency is computed here.

The paper defines the displaced category only by origin ("displaced tracks from the HNL process") and the
reconstructable particles as pT > 1 GeV, |eta| < 4, >= 5 space points (docs/paper_facts.md C7, C13). Its Fig. 3 gives
the matched and total counts per bin (binomial error bars), 2,882 displaced particles in the |d0| panel (two bins drawn
at efficiency 1 have no count), about half of our 5,623 HNL-daughter targets on the same 4,999 events
(docs/paper_facts.md, "Reproducing the paper's denominators"). This script compares candidate selections of our
daughter targets on hnl15_100 (the external sample) with the paper's per-bin totals:

  candidates   all daughter targets; mu+ / mu-; the higher- / lower-pT daughter of the event (ranked among both
               daughters, then required to be a target); the highest-pT daughter target of each event (one per event
               with a target); the daughter with the same / the opposite charge as the W muon; the first / second
               daughter in barcode order (particle index);
  panels       pT (the paper's nine bins, 1-28 GeV); "|d0|" with two readings, our truth-fit |d0| and our two-track
               vertex radius r_v; "z0" as truth-fit z0 and as vertex z; "production vertex radius" as r_v and as the 3D
               distance sqrt(r_v^2 + z_v^2);
  prompt check the W muons with >= 5 hits (4,744, the paper's prompt denominator) in the z0 and radius panels, with
               truth-fit z0 and |z0| (their production point is on the beam line), to test the readings of the axes.

Score of a candidate: chi2 = sum over the bins with a known paper total of (n_ours - n_paper)^2 / (n_ours + n_paper),
per panel and reading, and the total count against 2,882 (+ the two uncounted bins). Daughters without r_v (no fitted
pair vertex) are counted in their own column and left out of the vertex-variable panels; the paper's totals in those
panels are then compared after scaling by the fraction of the candidate's daughters that have a vertex (reported).

Writes results/wang_displaced_selection_v1.csv (counts per candidate, panel, reading and bin, with the paper's totals)
and results/wang_displaced_selection_v1.md (summary and the chi2 table).

    sbatch ... --wrap "source env/activate.sh; python scripts/diagnostics/wang_displaced_selection.py"
"""
import os
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results")
DERIVED = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived")
SAMPLE = "hnl15_100"
VERSION = os.environ.get("WDS_VERSION", "v1")
OUT_CSV = os.path.join(RES, f"wang_displaced_selection_{VERSION}.csv")
OUT_MD = os.path.join(RES, f"wang_displaced_selection_{VERSION}.md")

# Fig. 3 totals per bin (docs/paper_facts.md, "Values derived from the figures"); None = drawn at 1.0, total unknown;
# 0 = empty bin.
PAPER_DISP = {
    "pt": ([1, 4, 7, 10, 13, 16, 19, 22, 25, 28], [613, 501, 431, 375, 273, 244, 158, 101, 90]),
    "d0": (list(range(0, 751, 50)), [1084, 645, 358, 268, 186, 147, 93, 41, 34, 26, None, 0, None, 0, 0]),
    "z0": (list(range(-2000, 2001, 200)), [0, None, 19, 20, 42, 78, 109, 155, 266, 757, 756, 257, 153, 97, 76, 56, 27,
                                          8, 11, 0]),
    "radius": (list(range(0, 301, 25)), [107, 204, 218, 193, 177, 136, 128, 112, 103, 95, 118, 75]),
}
PAPER_DISP_TOTAL = 2882           # |d0| panel without the two bins drawn at 1.0
PAPER_PROMPT = {
    "pt": ([1, 4, 7, 10, 13, 16, 19, 22, 25, 28], [None, None, 131, 191, 214, 266, 315, 367, 435]),
    "z0": ([-200, 0, 200], [2390, 2354]),
    "radius": ([0, 25, 50, 75, 100, 125, 150], [1656, 1308, 940, 548, 183, 77]),
}
PAPER_PROMPT_TOTAL = 4744
READINGS = {"pt": {"pt": "pt"}, "d0": {"truth |d0|": "absd0", "r_v": "rv"}, "z0": {"truth z0": "z0", "vertex z": "zv"},
            "radius": {"r_v": "rv", "3D vertex distance": "r3"}}
PROMPT_READINGS = {"pt": {"pt": "pt"}, "z0": {"truth z0": "z0"},
                   "radius": {"|z0|": "absz0", "sqrt(d0^2 + z0^2)": "r3"}}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    import glob
    cols = ["event", "particle_id", "pdg", "q", "pt", "eta", "particle", "n_hits", "category", "is_target"]
    p = pd.concat([pd.read_parquet(f, columns=cols)
                   for f in sorted(glob.glob(os.path.join(DERIVED, "events", "v1", "particles", SAMPLE, "*.parquet")))],
                  ignore_index=True)
    from hnl.data.splits import read_split
    ev = set(int(x) for x in read_split(SAMPLE, "external"))
    p = p[p.event.isin(ev) & p.category.isin(["hnl", "w"])].copy()
    f = pd.read_parquet(os.path.join(DERIVED, "truth_fits", "v1", f"particles_{SAMPLE}.parquet"),
                        columns=["event", "particle_id", "pdg", "fix_d0", "free_d0", "fix_z0", "free_z0"])
    f["d0"] = np.where(f.pdg.abs() == 13, f.fix_d0, f.free_d0)
    f["z0"] = np.where(f.pdg.abs() == 13, f.fix_z0, f.free_z0)
    p = p.merge(f[["event", "particle_id", "d0", "z0"]], on=["event", "particle_id"], how="left")
    v = pd.read_parquet(os.path.join(DERIVED, "truth_fits", "v1", f"vertices_{SAMPLE}.parquet"))
    v = v[v.ok]
    vv = pd.concat([v[["event", k, "rv", "vz"]].rename(columns={k: "particle_id"}) for k in ("pid_a", "pid_b")])
    p = p.merge(vv.rename(columns={"vz": "zv"}), on=["event", "particle_id"], how="left")
    p["absd0"] = p.d0.abs()
    p["absz0"] = p.z0.abs()
    d = p[p.category == "hnl"].copy()
    w = p[p.category == "w"].copy()
    d["r3"] = np.hypot(d.rv, d.zv)
    w["r3"] = np.hypot(w.d0, w.z0)
    return d, w


def candidates(d: pd.DataFrame, w: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = d.copy()
    d["rank_pt"] = d.groupby("event").pt.rank(ascending=False, method="first")
    d["rank_rec"] = d.groupby("event").particle.rank(method="first")
    wq = w.drop_duplicates("event").set_index("event").q
    d["wq"] = d.event.map(wq)
    t = d[d.is_target]
    lead_target = t.loc[t.groupby("event").pt.idxmax()]
    out = {}
    if VERSION != "v1":
        # v2: the reconstructable definition varied (the paper's prompt denominator, 4,744, applies only the >= 5
        # space-point requirement); each variant for the four charge-defined halves and all daughters
        h5 = d.n_hits >= 5
        variants = {"pT > 1, |eta| < 4, >= 5 hits": d.is_target, ">= 5 hits only": h5,
                    ">= 5 hits, pT > 1": h5 & (d.pt > 1.0), ">= 5 hits, |eta| < 4": h5 & (d.eta.abs() < 4.0)}
        for vn, vm in variants.items():
            x = d[vm]
            out[f"[{vn}] all daughters"] = x
            out[f"[{vn}] mu+"] = x[x.q > 0]
            out[f"[{vn}] mu-"] = x[x.q < 0]
            out[f"[{vn}] same charge as W"] = x[x.q == x.wq]
            out[f"[{vn}] opposite charge to W"] = x[x.q == -x.wq]
        return out
    return {
        "all daughter targets": t,
        "mu+ only": t[t.q > 0],
        "mu- only": t[t.q < 0],
        "higher-pT daughter (target)": t[t.rank_pt == 1],
        "lower-pT daughter (target)": t[t.rank_pt == 2],
        "highest-pT target per event": lead_target,
        "same charge as the W muon": t[t.q == t.wq],
        "opposite charge to the W muon": t[(t.q == -t.wq)],
        "first daughter in barcode order": t[t.rank_rec == 1],
        "second daughter in barcode order": t[t.rank_rec == 2],
    }


def md(df: pd.DataFrame, index: bool = False) -> str:
    """A pipe table (no tabulate in the env)."""
    if index:
        df = df.reset_index()
    cols = [" / ".join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in r) + " |")
    return "\n".join(out)


def histo(x: pd.Series, edges) -> np.ndarray:
    x = x.dropna().to_numpy(float)
    return np.histogram(x, bins=np.asarray(edges, float))[0]


def compare(name: str, sel: pd.DataFrame, paper: dict, readings: dict, kind: str) -> list[dict]:
    rows = []
    for panel, (edges, totals) in paper.items():
        for reading, col in readings[panel].items():
            has = sel[col].notna()
            ours = histo(sel[col], edges)
            frac = float(has.mean()) if len(sel) else np.nan
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                rows.append({"population": kind, "candidate": name, "panel": panel, "reading": reading,
                             "lo": lo, "hi": hi, "ours": int(ours[i]),
                             "paper": (np.nan if totals[i] is None else totals[i]), "frac_with_value": frac,
                             "n_candidate": len(sel), "n_without_value": int((~has).sum())})
    return rows


def chi2(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (pop, cand, panel, reading), g in df.groupby(["population", "candidate", "panel", "reading"], sort=False):
        k = g.paper.notna()
        scale = 1.0
        if panel != "pt" and g.frac_with_value.iloc[0] < 1:
            scale = float(g.frac_with_value.iloc[0])          # the paper's totals include particles we cannot place
        n, m = g.ours[k].to_numpy(float), g.paper[k].to_numpy(float) * scale
        c = float(np.sum((n - m) ** 2 / np.maximum(n + m, 1.0)))
        out.append({"population": pop, "candidate": cand, "panel": panel, "reading": reading, "bins": int(k.sum()),
                    "chi2": round(c, 1), "ours_in_bins": int(n.sum()), "paper_in_bins_scaled": round(float(m.sum()), 1),
                    "scale": round(scale, 4), "n_candidate": int(g.n_candidate.iloc[0]),
                    "n_without_value": int(g.n_without_value.iloc[0])})
    return pd.DataFrame(out)


def main() -> None:
    t0 = time.time()
    for f in (OUT_CSV, OUT_MD):
        if os.path.exists(f):
            raise SystemExit(f"{f} exists")
    d, w = load()
    wp = w[w.n_hits >= 5]
    if len(wp) != PAPER_PROMPT_TOTAL:
        raise SystemExit(f"W muons with >= 5 hits: {len(wp)}, expected {PAPER_PROMPT_TOTAL}")
    n_targets = int(d.is_target.sum())
    if n_targets != 5623:
        raise SystemExit(f"daughter targets: {n_targets}, expected 5623 (docs/paper_facts.md)")
    rows = compare("W muons, >= 5 hits", wp, PAPER_PROMPT, PROMPT_READINGS, "prompt")
    cands = candidates(d, w)
    for name, sel in cands.items():
        rows += compare(name, sel, PAPER_DISP, READINGS, "displaced")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    c = chi2(df)
    tot = pd.DataFrame([{"candidate": k, "n": len(v), "events": v.event.nunique(),
                         "n_minus_2882": len(v) - PAPER_DISP_TOTAL} for k, v in cands.items()])
    piv = c[c.population == "displaced"].pivot_table(index="candidate", columns=["panel", "reading"], values="chi2",
                                                     sort=False)
    lines = ["# Wang et al.'s displaced denominator: candidate selections of our daughter targets (truth only)", "",
             f"Sample hnl15_100 (external, {d.event.nunique()} events with a daughter in the table; 4,999 events); "
             f"daughter targets {n_targets}; W muons with >= 5 hits {len(wp)} (paper 4,744). No model output read. "
             f"Script scripts/diagnostics/wang_displaced_selection.py, job {os.environ.get('SLURM_JOB_ID', 'local')}, "
             f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.", "",
             "## Totals (paper: 2,882 in the |d0| panel plus two bins drawn at 1.0 with unknown totals)", "",
             md(tot), "",
             "## chi2 per panel and reading (bins with a known paper total; vertex panels scaled by the fraction "
             "of the candidate's daughters with a fitted vertex)", "", md(piv, index=True), "",
             "## Prompt check (axis readings)", "",
             md(c[c.population == "prompt"]), "",
             "## All chi2 rows", "", md(c), ""]
    open(OUT_MD, "w").write("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
