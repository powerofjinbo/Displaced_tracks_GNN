"""Tables of results/generated_nav_losses_v1.md from the scans of hnl.truth.nav_losses (truth only; no model output).

Writes results/generated_nav_losses_v1.json and prints the markdown tables (the note's prose is written by hand
around them).

    python -m hnl.truth.nav_losses_report [--tables <file.md>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from hnl.eval.addendum_g_t1 import d0_exact
from hnl.truth.nav_losses import (COV_MARGIN_A_MM, COV_MARGIN_B_MM, END_MARGIN_MM, GEN_DIRS, GENERATED,
                                  MARGIN_B_MM, MARGIN_MM, NEAR_ARC_MM, OUT_DATA, OUT_JSON, PMIN, PT_MIN_B, R_WORLD,
                                  Z_WORLD, allowed_events, layer_table)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SPECIES = {11: "e", 13: "mu", 211: "pi", 321: "K", 2212: "p"}
PT_GUN = [0.5, 1, 2, 5, 10, 30, np.inf]
ETA_GUN = [0, 1, 2, 3, np.inf]
RV_GUN = [0, 10, 100, 200, 300, 450, 650, 800]
PT_PROMPT = [0, 0.3, 0.5, 1, 2, 5, np.inf]
ETA_PROMPT = [0, 1, 2, 2.5, 3, np.inf]
CAUSES = ("outside", "world", "loop", "below_pmin", "inside")


def _frac(k: int, n: int) -> dict:
    p = k / n if n else float("nan")
    return {"k": int(k), "n": int(n), "pct": round(100 * p, 3) if n else None,
            "se_pct": round(100 * np.sqrt(p * (1 - p) / n), 3) if n else None}


def _fmt(f: dict, se: bool = False) -> str:
    if not f["n"]:
        return "-"
    s = f"{f['pct']:.2f} %"
    if se:
        s += f" +- {f['se_pct']:.2f}"
    return s + f" ({f['k']:,}/{f['n']:,})"


def _bins(x: np.ndarray, edges) -> pd.Categorical:
    labels = [f"{a:g}-{b:g}" if np.isfinite(b) else f">= {a:g}" for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(x, edges, right=False, labels=labels)


def load() -> dict[str, pd.DataFrame]:
    out = {}
    for s in ("generic_iso", "generic_dense", "ttbar"):
        df = pd.read_parquet(os.path.join(OUT_DATA, f"{s}.parquet"))
        for c in ("ok_b", "in_table", "by_loop_a", "final_hit_a", "final_hit_b"):
            if c in df:
                df[c] = df[c].fillna(False).astype(bool)
        out[s] = df
    return out


def gun_d0(sample: str) -> pd.DataFrame:
    """Exact |d0| of every gun particle of the train and val events (gun_truth sidecars; other events dropped)."""
    allowed = allowed_events(sample)
    rows = []
    for f in sorted(glob.glob(os.path.join(GENERATED, GEN_DIRS[sample] + "_truth", "gun_truth_*.npz"))):
        with np.load(f) as z:
            keep = np.isin(z["event"], list(allowed))
            rows.append(pd.DataFrame({"event": z["event"][keep], "particle_id": z["particle_id"][keep].astype(np.int64),
                                      "d0": d0_exact(z["vx"][keep], z["vy"][keep], z["px"][keep], z["py"][keep],
                                                     z["q"][keep].astype(float))}))
    return pd.concat(rows, ignore_index=True)


def add_flags(d: pd.DataFrame, has_a: bool) -> pd.DataFrame:
    d = d.copy()
    d["species"] = np.abs(d.pdg).map(SPECIES).fillna("other")
    d["elig_b"] = d.ok_b & (d.n_rob_b >= 1)
    d["prem_b"] = d.elig_b & (d.n_after_b >= 1)
    d["prem2_b"] = d.elig_b & (d.n_after_b >= 2)
    d["hole_b"] = d.ok_b & (d.n_holes_b >= 1)
    if has_a:
        d["elig_a"] = d.in_table & (d.n_rob_a >= 1)
        d["prem_a"] = d.elig_a & (d.n_after_a >= 1)
        d["prem2_a"] = d.elig_a & (d.n_after_a >= 2)
        d["hole_a"] = d.in_table & (d.n_holes_a >= 1)
    # hits the particle would have had without the missing crossings (method B where it applies, else A)
    if has_a:
        miss = np.where(d.ok_b, d.n_holes_b.clip(lower=0) + d.n_acc_after_b.clip(lower=0),
                        d.n_holes_a.clip(lower=0) + d.n_acc_after_a.clip(lower=0))
        miss_end = np.where(d.ok_b, d.n_acc_after_b.clip(lower=0), d.n_acc_after_a.clip(lower=0))
    else:
        miss = np.where(d.ok_b, d.n_holes_b.clip(lower=0) + d.n_acc_after_b.clip(lower=0), 0)
        miss_end = np.where(d.ok_b, d.n_acc_after_b.clip(lower=0), 0)
    d["would_hits"] = d.n_hits + miss
    d["would_hits_end"] = d.n_hits + miss_end
    kin = (d.pt_table > 1.0) & (np.abs(d.eta_table) < 4.0)
    d["target"] = kin & (d.n_hits >= 5)
    d["target_would"] = kin & (d.would_hits >= 5)
    d["target_would_end"] = kin & (d.would_hits_end >= 5)
    return d


def breakdown(d: pd.DataFrame, by: pd.Series, cols: dict[str, tuple[str, str]]) -> list[dict]:
    """cols: name -> (flag column, denominator column)."""
    rows = []
    for key, g in d.groupby(by, observed=True):
        r = {"bin": str(key)}
        for name, (flag, den) in cols.items():
            gg = g[g[den]]
            r[name] = _frac(int(gg[flag].sum()), len(gg))
        rows.append(r)
    return rows


def md_table(rows: list[dict], cols: list[str], head: list[str], se: bool = False) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join([r["bin"]] + [_fmt(r[c], se) if isinstance(r[c], dict) else str(r[c])
                                                   for c in cols]) + " |")
    return "\n".join(out)


def cause_table(d: pd.DataFrame, flag: str) -> dict:
    p = d[d[flag]]
    return {c: int((p.end_cat == c).sum()) for c in CAUSES} | {"total": len(p)}


def gun_section(name: str, d: pd.DataFrame, d0: pd.DataFrame, tables: list[str]) -> dict:
    d = d.merge(d0, on=["event", "particle_id"], how="left", validate="one_to_one")
    d = add_flags(d, True)
    res: dict = {}
    sim = d[d.in_table]
    res["counts"] = {"gun_particles": len(d), "events": int(d.event.nunique()),
                     "below_pmin_at_start": int((d.p0 < PMIN).sum()),
                     "p0_ge_pmin_not_in_table": int(((d.p0 >= PMIN) & ~d.in_table).sum()),
                     "simulated": len(sim), "with_hits": int((sim.n_hits > 0).sum()),
                     "method_b_valid": int(sim.ok_b.sum())}
    res["end_categories"] = {c: _frac(int((sim.end_cat == c).sum()), len(sim)) for c in CAUSES}
    res["end_categories_with_hits"] = {c: _frac(int(((sim.end_cat == c) & (sim.n_hits > 0)).sum()),
                                                int((sim.n_hits > 0).sum())) for c in CAUSES}
    res["premature"] = {
        "b_ge1": _frac(int(d.prem_b.sum()), int(d.elig_b.sum())),
        "b_ge2": _frac(int(d.prem2_b.sum()), int(d.elig_b.sum())),
        "a_ge1": _frac(int(d.prem_a.sum()), int(d.elig_a.sum())),
        "a_ge2": _frac(int(d.prem2_a.sum()), int(d.elig_a.sum())),
        "causes_b_ge1": cause_table(d, "prem_b"), "causes_a_ge1": cause_table(d, "prem_a"),
        "causes_b_ge2": cause_table(d, "prem2_b"),
        "zero_hits_b": int((d.prem_b & (d.n_hits == 0)).sum()),
    }
    ab = d[d.elig_a & d.elig_b]
    res["a_vs_b"] = {"both": int((ab.prem_a & ab.prem_b).sum()), "a_only": int((ab.prem_a & ~ab.prem_b).sum()),
                     "b_only": int((~ab.prem_a & ab.prem_b).sum()), "neither": int((~ab.prem_a & ~ab.prem_b).sum())}
    bv = d[d.ok_b]
    res["holes"] = {"per_crossing_b": _frac(int(bv.n_holes_b.sum()), int(bv.n_before_b.sum())),
                    "per_crossing_a": _frac(int(sim.n_holes_a.sum()), int(sim.n_before_a.sum())),
                    "particles_with_hole_b": _frac(int(bv.hole_b.sum()), len(bv)),
                    "first_hole_rank_b": {int(k): int(v) for k, v in
                                          bv[bv.hole_b].first_hole_rank_b.value_counts().sort_index().items()}}
    res["unmatched_hits"] = {"b": _frac(int((bv.n_unm_b > 0).sum()), len(bv)),
                             "a": _frac(int((sim.n_unm_a > 0).sum()), len(sim))}
    cols = {"prem_b": ("prem_b", "elig_b"), "outside": ("lost_out", "elig_b"), "prem_a": ("prem_a", "elig_a"),
            "hole_b": ("hole_b", "ok_b")}
    d["lost_out"] = d.prem_b & (d.end_cat == "outside")
    by = {"species": d.species, "pT [GeV]": _bins(d.pt0.to_numpy(), PT_GUN),
          "|eta|": _bins(np.abs(d.eta0.to_numpy()), ETA_GUN), "r_v [mm]": _bins(d.r_v.to_numpy(), RV_GUN)}
    res["breakdown"] = {}
    for label, b in by.items():
        rows = breakdown(d, b, cols)
        res["breakdown"][label] = rows
        tables.append(f"\n#### {name}: by {label}\n\n" + md_table(
            rows, ["prem_b", "outside", "prem_a", "hole_b"],
            [label, "premature end (B)", "of which flew out", "premature end (A)", "mid-track hole (B)"]))
    # target status
    t = d[d.in_table]
    pg = t.r_v.ge(100) | t.d0.ge(10)
    res["targets"] = {}
    for lab, m in (("all gun", np.ones(len(t), bool)), ("P_G-like (|d0| >= 10 mm or r_v >= 100 mm)", pg.to_numpy())):
        g = t[m]
        res["targets"][lab] = {
            "targets": int(g.target.sum()),
            "would_be_targets": int(g.target_would.sum()),
            "lost_by_premature_end": int((g.target_would_end & ~g.target).sum()),
            "lost_by_holes_only": int((g.target_would & ~g.target_would_end & ~g.target).sum()),
            "targets_with_missing_hits": int((g.target & (g.would_hits > g.n_hits)).sum()),
            "targets_with_premature_end": int((g.target & g.prem_b).sum()),
            "lost_fraction_of_would_be": _frac(int((g.target_would & ~g.target).sum()), int(g.target_would.sum())),
        }
    res["targets_by_split"] = {sp: {"targets": int(g.target.sum()), "would_be": int(g.target_would.sum()),
                                    "P_G_like_targets": int((g.target & (g.r_v.ge(100) | g.d0.ge(10))).sum()),
                                    "P_G_like_would_be": int((g.target_would & (g.r_v.ge(100) | g.d0.ge(10))).sum())}
                               for sp, g in t.groupby("split")}
    o = sim[sim.end_cat == "outside"]
    res["outside"] = {"n": len(o), "turn_within_0.05_of_pi": _frac(int((np.abs(o.turn - np.pi) < 0.05).sum()), len(o)),
                      "zero_hits": _frac(int((o.n_hits == 0).sum()), len(o)),
                      "mean_hits": round(float(o.n_hits.mean()), 3) if len(o) else None,
                      "mean_hits_others_with_hits": round(float(sim[(sim.end_cat != "outside") &
                                                                   (sim.n_hits > 0)].n_hits.mean()), 3),
                      "no_robust_crossing_after_last_hit": _frac(int((o.ok_b & (o.n_after_b == 0)).sum()),
                                                                 int(o.ok_b.sum())),
                      "mean_missed_crossings_in_acceptance": round(float(o[o.ok_b].n_acc_after_b.mean()), 3)}
    res["first_missed_layer_b"] = {int(k): int(v) for k, v in
                                   d[d.prem_b].first_missed_key_b.value_counts().head(12).items()}
    res["first_hole_layer_b"] = {int(k): int(v) for k, v in
                                 d[d.hole_b].first_hole_key_b.value_counts().head(12).items()}
    w = d[d.prem_b & (d.end_cat == "world")]
    res["first_missed_layer_b_world_end"] = {int(k): int(v) for k, v in
                                             w.first_missed_key_b.value_counts().head(8).items()}
    # first holes at the first robust crossing after the vertex, in a barrel layer: vertex radius minus layer radius
    L = layer_table().set_index("key")
    h0 = d[d.hole_b & (d.first_hole_rank_b == 0)]
    Rl = L.R.reindex(h0.first_hole_key_b.astype(int)).to_numpy()
    dr = (h0.r0.to_numpy() - Rl)[np.isfinite(Rl)]
    res["first_hole_rank0_barrel"] = {
        "n": int(len(dr)), "vertex_r_minus_layer_r_mm_quantiles_10_50_90": [round(float(x), 1) for x in
                                                                          np.quantile(dr, [0.1, 0.5, 0.9])],
        "within_5mm": _frac(int((np.abs(dr) < 5).sum()), len(dr))} if len(dr) else {}
    el = sim[sim.species == "e"]
    ne = sim[sim.species != "e"]
    res["below_pmin_share"] = {"electrons": _frac(int((el.end_cat == "below_pmin").sum()), len(el)),
                               "other_species": _frac(int((ne.end_cat == "below_pmin").sum()), len(ne))}
    # material expectation: premature ends with p_end < pMin, by species, and the initial momenta of hadrons there
    bp = d[d.prem_b & (d.end_cat == "below_pmin")]
    res["below_pmin_by_species"] = {s: int(v) for s, v in bp.species.value_counts().items()}
    hb = d[(d.end_cat == "below_pmin") & d.in_table & (d.species != "e")]
    res["hadron_mu_below_pmin_p0_quantiles"] = [round(float(x), 3) for x in
                                                np.quantile(hb.p0, [0.5, 0.9, 0.99])] if len(hb) else []
    return res


def prompt_section(name: str, d: pd.DataFrame, census: pd.DataFrame, has_a: bool, tables: list[str]) -> dict:
    d = add_flags(d, has_a)
    d["species"] = np.abs(d.pdg_table).map(SPECIES).fillna("other")
    res: dict = {"particles_with_hits": len(d), "events": int(census.event.nunique()),
                 "method_b_valid": int(d.ok_b.sum())}
    res["census"] = {c: _frac(int((census.end_cat == c).sum()), len(census)) for c in
                     ("outside", "world", "below_pmin", "inside")}
    res["census_outside_by_hits"] = {"with_hits": int(((census.end_cat == "outside") & (census.n_hits > 0)).sum()),
                                     "without_hits": int(((census.end_cat == "outside") & (census.n_hits == 0)).sum())}
    res["end_categories"] = {c: _frac(int((d.end_cat == c).sum()), len(d)) for c in CAUSES}
    res["premature"] = {"b_ge1": _frac(int(d.prem_b.sum()), int(d.elig_b.sum())),
                        "b_ge2": _frac(int(d.prem2_b.sum()), int(d.elig_b.sum())),
                        "causes_b_ge1": cause_table(d, "prem_b"), "causes_b_ge2": cause_table(d, "prem2_b")}
    if has_a:
        res["premature"]["a_ge1"] = _frac(int(d.prem_a.sum()), int(d.elig_a.sum()))
        res["premature"]["a_ge2"] = _frac(int(d.prem2_a.sum()), int(d.elig_a.sum()))
        ab = d[d.elig_a & d.elig_b]
        res["a_vs_b"] = {"both": int((ab.prem_a & ab.prem_b).sum()), "a_only": int((ab.prem_a & ~ab.prem_b).sum()),
                         "b_only": int((~ab.prem_a & ab.prem_b).sum()),
                         "neither": int((~ab.prem_a & ~ab.prem_b).sum())}
    bv = d[d.ok_b]
    res["holes"] = {"per_crossing_b": _frac(int(bv.n_holes_b.sum()), int(bv.n_before_b.sum())),
                    "per_crossing_b_near_end": _frac(int(bv.n_holes_near_b.sum()), int(bv.n_near_b.sum())),
                    "particles_with_hole_b": _frac(int(bv.hole_b.sum()), len(bv))}
    hp = bv[(bv.pt_table > 2) & (np.abs(bv.eta_table) < 1)]
    res["holes"]["per_crossing_b_pt_gt2_eta_lt1"] = _frac(int(hp.n_holes_b.sum()), int(hp.n_before_b.sum()))
    res["unmatched_hits_b"] = _frac(int((bv.n_unm_b > 0).sum()), len(bv))
    res["vertex_miss_b_mm_quantiles"] = [round(float(x), 3) for x in np.nanquantile(bv.vtx_miss_b, [0.5, 0.9, 0.99])]
    z = bv.groupby("event").z0_b.transform("median")
    res["z0_b_minus_event_median_mm_quantiles"] = [round(float(x), 3) for x in
                                                   np.nanquantile(np.abs(bv.z0_b - z), [0.5, 0.9, 0.99])]
    cols = {"prem_b": ("prem_b", "elig_b"), "prem2_b": ("prem2_b", "elig_b"), "hole_b": ("hole_b", "ok_b")}
    by = {"pT [GeV]": _bins(d.pt_table.to_numpy(), PT_PROMPT), "|eta|": _bins(np.abs(d.eta_table.to_numpy()),
                                                                                ETA_PROMPT),
          "species": d.species}
    res["breakdown"] = {}
    for label, b in by.items():
        rows = breakdown(d, b, cols)
        res["breakdown"][label] = rows
        tables.append(f"\n#### {name}: by {label}\n\n" + md_table(
            rows, ["prem_b", "prem2_b", "hole_b"],
            [label, "premature end, >= 1 (B)", ">= 2 (B)", "mid-track hole (B)"]))
    t = d
    res["targets"] = {"targets": int(t.target.sum()), "would_be_targets": int(t.target_would.sum()),
                      "lost_by_premature_end": int((t.target_would_end & ~t.target).sum()),
                      "lost_by_holes_only": int((t.target_would & ~t.target_would_end & ~t.target).sum()),
                      "targets_with_missing_hits": int((t.target & (t.would_hits > t.n_hits)).sum()),
                      "lost_fraction_of_would_be": _frac(int((t.target_would & ~t.target).sum()),
                                                         int(t.target_would.sum()))}
    res["first_missed_layer_b"] = {int(k): int(v) for k, v in
                                   d[d.prem_b].first_missed_key_b.value_counts().head(12).items()}
    tg = d[d.target]
    res["targets_premature"] = _frac(int(tg.prem_b.sum()), int(tg.elig_b.sum()))
    hi = d[d.elig_b & (d.pt_table > 1.0)]
    res["pt_gt_1"] = {"premature_b": _frac(int(hi.prem_b.sum()), len(hi)),
                      "first_missed_layer_b": {int(k): int(v) for k, v in
                                               hi[hi.prem_b].first_missed_key_b.value_counts().head(5).items()},
                      "holes_per_crossing_b": _frac(int(hi.n_holes_b.sum()), int(hi.n_before_b.sum()))}
    lp = d[d.end_cat == "loop"]
    res["loop_ended"] = {"n": len(lp), "premature_b": _frac(int(lp.prem_b.sum()), int(lp.elig_b.sum()))}
    return res


def layer_rates(path: str) -> list[dict]:
    lay = pd.read_csv(path)
    out = []
    for (kind, m), g in lay.groupby(["kind", "method"]):
        for k, r in g.set_index("key").iterrows():
            if r.before:
                out.append(dict(kind=kind, method=m, key=int(k), before=int(r.before), holes=int(r.holes),
                                after=int(r.after), hole_pct=round(100 * r.holes / r.before, 3)))
    return out


def _row(label: str, vals: list) -> str:
    return "| " + " | ".join([label] + [v if isinstance(v, str) else (_fmt(v) if isinstance(v, dict) else f"{v:,}")
                                        for v in vals]) + " |"


def summary_tables(res: dict) -> str:
    gi, gd = res["generic_iso"]["gun"], res["generic_dense"]["gun"]
    pg, pz = res["prompt_generated"], res["prompt_zenodo"]
    out = ["\n#### Gun particles: how they end\n", "| | generic_iso | generic_dense |", "|---|---|---|"]
    for k, lab in (("gun_particles", "gun particles (train + val)"),
                   ("below_pmin_at_start", "p < 0.5 GeV, not simulated"),
                   ("p0_ge_pmin_not_in_table", "p >= 0.5 GeV but missing from the table"),
                   ("simulated", "simulated"), ("with_hits", "simulated, with hits")):
        out.append(_row(lab, [gi["counts"][k], gd["counts"][k]]))
    for c in CAUSES:
        out.append(_row(f"end: {c}", [gi["end_categories"][c], gd["end_categories"][c]]))
    out.append(_row("outside: mean hits (others with hits)",
                    [f"{gi['outside']['mean_hits']} ({gi['outside']['mean_hits_others_with_hits']})",
                     f"{gd['outside']['mean_hits']} ({gd['outside']['mean_hits_others_with_hits']})"]))
    out.append(_row("outside: without any hit", [gi["outside"]["zero_hits"], gd["outside"]["zero_hits"]]))
    out.append(_row("outside: transverse turn within 0.05 rad of pi",
                    [gi["outside"]["turn_within_0.05_of_pi"], gd["outside"]["turn_within_0.05_of_pi"]]))
    out.append(_row("outside: no robust crossing after the last hit",
                    [gi["outside"]["no_robust_crossing_after_last_hit"],
                     gd["outside"]["no_robust_crossing_after_last_hit"]]))
    out += ["\n#### Gun particles: premature ends\n", "| | generic_iso | generic_dense |", "|---|---|---|"]
    for k, lab in (("b_ge1", "B, >= 1 crossing after the last hit"), ("b_ge2", "B, >= 2"),
                   ("a_ge1", "A, >= 1"), ("a_ge2", "A, >= 2")):
        out.append(_row(lab, [gi["premature"][k], gd["premature"][k]]))
    for m in ("b", "a"):
        for c in CAUSES:
            out.append(_row(f"{m.upper()} >= 1, end {c}", [gi["premature"][f"causes_{m}_ge1"][c],
                                                          gd["premature"][f"causes_{m}_ge1"][c]]))
    out.append(_row("A and B agree (both / A only / B only / neither)",
                    [" / ".join(str(gi["a_vs_b"][k]) for k in ("both", "a_only", "b_only", "neither")),
                     " / ".join(str(gd["a_vs_b"][k]) for k in ("both", "a_only", "b_only", "neither"))]))
    out += ["\n#### Holes before the last hit (B)\n",
            "| | gun, iso | gun, dense | prompt, our ttbar | prompt, Zenodo |", "|---|---|---|---|---|"]
    out.append(_row("per robust crossing", [gi["holes"]["per_crossing_b"], gd["holes"]["per_crossing_b"],
                                            pg["holes"]["per_crossing_b"], pz["holes"]["per_crossing_b"]]))
    out.append(_row("particles with >= 1 hole", [gi["holes"]["particles_with_hole_b"],
                                                 gd["holes"]["particles_with_hole_b"],
                                                 pg["holes"]["particles_with_hole_b"],
                                                 pz["holes"]["particles_with_hole_b"]]))
    sub = {7: "pixel disks", 9: "pixel disks", 8: "pixel barrel", 12: "short-strip disks", 14: "short-strip disks",
           13: "short-strip barrel", 16: "long-strip disks", 18: "long-strip disks", 17: "long-strip barrel"}
    lr = {s: pd.DataFrame(res["layer_rates"][s]) for s in res["layer_rates"]}
    cols = [("generic_iso", "gun"), ("generic_dense", "gun"), (("generic_iso", "generic_dense"), "prompt"),
            ("ttbar", "prompt")]
    rates = []
    for smp, kind in cols:
        x = pd.concat([lr[t] for t in (smp if isinstance(smp, tuple) else (smp,))])
        x = x[(x.kind == kind) & (x.method == "b")].assign(sub=lambda t: (t.key // 100).map(sub))
        rates.append(x.groupby("sub")[["before", "holes"]].sum())
    out += ["\n#### Holes per robust crossing by subsystem (B)\n",
            "| | gun, iso | gun, dense | prompt, our ttbar | prompt, Zenodo |", "|---|---|---|---|---|"]
    for name in ("pixel barrel", "pixel disks", "short-strip barrel", "short-strip disks", "long-strip barrel",
                 "long-strip disks"):
        out.append(_row(name, [_frac(int(r.holes.get(name, 0)), int(r.before.get(name, 0))) for r in rates]))
    out += ["\n#### Targets (pT > 1 GeV, |eta| < 4, >= 5 hits)\n",
            "| | iso, all gun | iso, P_G-like | dense, all gun | dense, P_G-like |", "|---|---|---|---|---|"]
    keys = list(gi["targets"])
    for k, lab in (("targets", "targets"), ("would_be_targets", "targets if no crossing were missed"),
                   ("lost_by_premature_end", "lost: premature end"), ("lost_by_holes_only", "lost: holes only"),
                   ("lost_fraction_of_would_be", "lost, share of would-be targets"),
                   ("targets_with_missing_hits", "targets that miss >= 1 hit"),
                   ("targets_with_premature_end", "targets with a premature end")):
        out.append(_row(lab, [gi["targets"][keys[0]][k], gi["targets"][keys[1]][k], gd["targets"][keys[0]][k],
                              gd["targets"][keys[1]][k]]))
    out += ["\n#### Prompt particles: our ttbar (generic_iso + generic_dense) and the Zenodo ttbar\n",
            "| | our ttbar | Zenodo ttbar |", "|---|---|---|"]
    out.append(_row("events", [pg["events"], pz["events"]]))
    out.append(_row("charged prompt particles with hits", [pg["particles_with_hits"], pz["particles_with_hits"]]))
    for c in CAUSES:
        out.append(_row(f"end: {c}", [pg["end_categories"][c], pz["end_categories"][c]]))
    for k, lab in (("b_ge1", "premature end, B >= 1"), ("b_ge2", "premature end, B >= 2")):
        out.append(_row(lab, [pg["premature"][k], pz["premature"][k]]))
    for c in CAUSES:
        out.append(_row(f"B >= 1, end {c}", [pg["premature"]["causes_b_ge1"][c], pz["premature"]["causes_b_ge1"][c]]))
    out.append(_row("loop-ended particles with a premature end", [pg["loop_ended"]["premature_b"],
                                                                  pz["loop_ended"]["premature_b"]]))
    out.append(_row("pT > 1 GeV: premature end", [pg["pt_gt_1"]["premature_b"], pz["pt_gt_1"]["premature_b"]]))
    out.append(_row("pT > 1 GeV: holes per crossing", [pg["pt_gt_1"]["holes_per_crossing_b"],
                                                       pz["pt_gt_1"]["holes_per_crossing_b"]]))
    out.append(_row("targets with a premature end", [pg["targets_premature"], pz["targets_premature"]]))
    out.append(_row("all charged prompt particles in the table: end outside",
                    [pg["census"]["outside"], pz["census"]["outside"]]))
    out.append(_row("unmatched hit (B), particles", [pg["unmatched_hits_b"], pz["unmatched_hits_b"]]))
    out.append(_row("|z0 - event median| of the B start, quantiles 50/90/99 % [mm]",
                    [str(pg["z0_b_minus_event_median_mm_quantiles"]), str(pz["z0_b_minus_event_median_mm_quantiles"])]))
    return "\n".join(out)


def disk_coverage(n: int = 200_000, seed: int = 1) -> dict:
    """Share of each disk's annulus [r_lo + MARGIN_MM, r_hi - MARGIN_MM] that lies on a module (depth >= 0 for a line
    along z), from uniform random points."""
    from hnl.truth.nav_losses import modules
    M = modules()
    rng = np.random.default_rng(seed)
    out = {}
    for _, r in layer_table().iterrows():
        if r.barrel:
            continue
        lo, hi = r.rlo + MARGIN_MM, r.rhi - MARGIN_MM
        rad = np.sqrt(rng.uniform(lo ** 2, hi ** 2, n))
        ph = rng.uniform(-np.pi, np.pi, n)
        P = np.stack([rad * np.cos(ph), rad * np.sin(ph), np.full(n, r.Z)], 1)
        D = np.tile([0.0, 0.0, 1.0], (n, 1))
        out[int(r.key)] = round(float((M.depth(int(r.key), P, D) >= 0).mean()), 4)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=None)
    a = ap.parse_args(argv)
    D = load()
    tables: list[str] = []
    res: dict = {"settings": dict(R_world_mm=R_WORLD, Z_world_mm=Z_WORLD, pmin_GeV=PMIN, margin_a_mm=MARGIN_MM,
                                  margin_b_mm=MARGIN_B_MM, cov_margin_a_mm=COV_MARGIN_A_MM,
                                  cov_margin_b_mm=COV_MARGIN_B_MM, end_margin_mm=END_MARGIN_MM,
                                  near_arc_mm=NEAR_ARC_MM, pt_min_b_GeV=PT_MIN_B)}
    metas = {}
    for s in ("generic_iso", "generic_dense", "ttbar"):
        with open(os.path.join(OUT_DATA, f"{s}_meta.json")) as f:
            metas[s] = json.load(f)
    res["scans"] = metas
    import hashlib
    res["code_sha256"] = {f: hashlib.sha256(open(os.path.join(REPO_DIR, f), "rb").read()).hexdigest()
                          for f in ("src/hnl/truth/nav_losses.py", "slurm/acts/nav_losses.sbatch")}
    L = layer_table()
    res["disk_coverage"] = disk_coverage()
    res["layers"] = L.drop(columns=["gaps"]).assign(gaps=[g for g in L.gaps]).to_dict(orient="records")
    for s in ("generic_iso", "generic_dense"):
        d = D[s]
        res[s] = {"gun": gun_section(s + " gun", d[d.kind == "gun"].copy(), gun_d0(s), tables)}
    gen_prompt = pd.concat([D[s][D[s].kind == "prompt"] for s in ("generic_iso", "generic_dense")], ignore_index=True)
    gen_census = pd.concat([D[s][D[s].kind == "census"] for s in ("generic_iso", "generic_dense")],
                           ignore_index=True)
    gen_census["event"] = gen_census["sample"] + ":" + gen_census.event.astype(str)
    gen_prompt["event"] = gen_prompt["sample"] + ":" + gen_prompt.event.astype(str)
    res["prompt_generated"] = prompt_section("generated ttbar prompt", gen_prompt, gen_census, True, tables)
    z = D["ttbar"]
    res["prompt_zenodo"] = prompt_section("Zenodo ttbar prompt", z[z.kind == "prompt"].copy(),
                                          z[z.kind == "census"], False, tables)
    res["layer_rates"] = {s: layer_rates(os.path.join(OUT_DATA, f"{s}_layers.csv"))
                          for s in ("generic_iso", "generic_dense", "ttbar")}
    tables.insert(0, summary_tables(res))
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", OUT_JSON)
    if a.tables:
        with open(a.tables, "w") as f:
            f.write("\n".join(tables) + "\n")
        print("wrote", a.tables)


if __name__ == "__main__":
    main()
