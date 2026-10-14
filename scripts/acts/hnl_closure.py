"""Closure of the HNL generator settings against the Zenodo HNL samples, on truth only (docs/hnl_points_spec.md;
results/hnl_points_closure_v1.md). Inputs are the censuses of scripts/acts/hnl_census.py: the Zenodo train and val
events of hnl10_100 and hnl15_200, the full-simulation pilot of each point (tar shards), and generator-level variants
of the N decay and the beam energy. No model is run.

Tests (pilot against Zenodo, per point): means (z = difference / combined standard error), fractions (binomial
standard errors), shapes (two-sample Kolmogorov-Smirnov for continuous quantities, chi-square on the histogram for
hit counts). A quantity closes when p >= 0.01; the closure holds when no p is below 0.001 and the number of p below
0.01 is compatible with chance (Poisson mean 0.01 x number of tests).

    python scripts/acts/hnl_closure.py --census DIR --out results/hnl_points_closure_v1 [--pilot TAG] [--genvar DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

POINTS = {"hnl10_100": (10.0, 100.0), "hnl15_200": (15.0, 200.0)}
# Differences with a known cause outside the generator settings: the hit-label exchange of the Zenodo files puts hits
# of other particles into about 16 % of the W-muon chains (scripts/acts/prompt_fit_check.py: rms of the fixed-radius
# fit, 84 % quantile 1.2 mm in Zenodo against 0.044 mm here), which widens the fitted |d0| and moves the chain head.
# Reported, but not counted in the closure.
KNOWN = {("W muon", "|d0| of targets [mm]"), ("W muon", "first-hit radius of targets [mm]")}
C_MM_PER_NS = 299.792458


def mean_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(ddof=1) / np.sqrt(len(a)), b.std(ddof=1) / np.sqrt(len(b))
    z = (ma - mb) / np.hypot(sa, sb) if np.hypot(sa, sb) > 0 else 0.0
    return {"pilot": ma, "pilot_err": sa, "zenodo": mb, "zenodo_err": sb, "n_pilot": len(a), "n_zenodo": len(b),
            "z": z, "p": 2 * stats.norm.sf(abs(z)), "test": "mean"}


def frac_test(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    pa, pb = a.mean(), b.mean()
    pool = (a.sum() + b.sum()) / (len(a) + len(b))
    se = np.sqrt(pool * (1 - pool) * (1 / len(a) + 1 / len(b)))
    z = (pa - pb) / se if se > 0 else 0.0
    return {"pilot": pa, "pilot_err": np.sqrt(pa * (1 - pa) / len(a)), "zenodo": pb,
            "zenodo_err": np.sqrt(pb * (1 - pb) / len(b)), "n_pilot": len(a), "n_zenodo": len(b), "z": z,
            "p": 2 * stats.norm.sf(abs(z)), "test": "fraction"}


def ks_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    r = stats.ks_2samp(a, b)
    return {"pilot": np.median(a), "pilot_err": np.nan, "zenodo": np.median(b), "zenodo_err": np.nan,
            "n_pilot": len(a), "n_zenodo": len(b), "z": r.statistic, "p": r.pvalue, "test": "KS (median shown)"}


def chi2_test(a, b, edges):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ha, hb = np.histogram(a, edges)[0], np.histogram(b, edges)[0]
    keep = (ha + hb) > 0
    t = np.vstack([ha[keep], hb[keep]])
    # merge sparse bins from the top until every expected count is >= 5
    while t.shape[1] > 2:
        exp = t.sum(0, keepdims=True) * t.sum(1, keepdims=True) / t.sum()
        if exp.min() >= 5:
            break
        j = int(np.argmin(exp.min(0)))
        k = j - 1 if j == t.shape[1] - 1 else j + 1
        t[:, min(j, k)] += t[:, max(j, k)]
        t = np.delete(t, max(j, k), axis=1)
    c2, p, dof, _ = stats.chi2_contingency(t)
    return {"pilot": a.mean(), "pilot_err": a.std(ddof=1) / np.sqrt(len(a)), "zenodo": b.mean(),
            "zenodo_err": b.std(ddof=1) / np.sqrt(len(b)), "n_pilot": len(a), "n_zenodo": len(b),
            "z": c2 / max(dof, 1), "p": p, "test": f"chi2 ({dof} dof; means shown)"}


def load(prefix):
    return (pd.read_parquet(prefix + "_events.parquet"), pd.read_parquet(prefix + "_signal.parquet"),
            json.load(open(prefix + "_ue.json")))


def derived(E: pd.DataFrame, S: pd.DataFrame, mass: float):
    """Columns shared by the Zenodo and pilot censuses."""
    S = S.copy()
    S["is_target"] = (S.pt > 1.0) & (S.eta.abs() < 4.0) & (S.n_hits >= 5)
    rv = E.set_index("event").rv
    S["rv"] = S.event.map(rv)
    S["primary"] = S.is_target & ((S.d0.abs() >= 10) | (S.rv >= 100))
    w = S[S.role == "w"].drop_duplicates("event").set_index("event")
    S["w_q"] = S.event.map(w.q)
    S["opp_w"] = np.sign(S.q) != np.sign(S.w_q)
    E = E.copy()
    E["m_ratio"] = E.m_mumu / mass
    d = S[S.role == "d"]
    E["n_d_target"] = E.event.map(d.groupby("event").is_target.sum()).fillna(0)
    E["n_d_primary"] = E.event.map(d.groupby("event").primary.sum()).fillna(0)
    E["n_sign_eq_w"] = (E.n_sign == E.w_sign) & (E.n_d > 0) & (E.n_w > 0)
    wm = S[(S.role == "w") & (S.r_end > 1000) & (S.r_end < 1100)]
    path = np.hypot(wm.r_end, wm.z_end)
    E["w_t0"] = E.event.map(pd.Series((wm.vt_end - path / C_MM_PER_NS).to_numpy(), index=wm.event.to_numpy()))
    return E, S


def zenodo_facts(E: pd.DataFrame, S: pd.DataFrame, ue: dict, mass: float) -> dict:
    """What the Zenodo train + val truth says about the generator (task 1 of docs/hnl_points_spec.md)."""
    d = S[S.role == "d"]
    w = S[S.role == "w"]
    two = E[E.n_d == 2]
    wm = w[(w.r_end > 1000) & (w.r_end < 1100)]
    t0 = wm.vt_end - np.hypot(wm.r_end, wm.z_end) / C_MM_PER_NS
    rend = d.groupby("event").r_end.max()
    E2 = two.assign(maxrend=two.event.map(rend))
    out = {
        "events": len(E),
        "events_with_one_w_muon": int((E.n_w == 1).sum()),
        "w_muon_at_vertex_1_alone": int(E.w_alone_vsec1.sum()),
        "w_plus_events": int((E.w_sign > 0).sum()),
        "events_by_number_of_n_daughters": {str(k): int(v) for k, v in E.n_d.value_counts().items()},
        "n_daughters_muons": int((d.pdg.abs() == 13).sum()), "n_daughters_total": len(d),
        "two_daughter_events_opposite_charge": int(two.d_opposite.sum()),
        "two_daughter_events_same_vertex": int(two.d_same_vsec.sum()),
        "events_with_other_particle_at_n_vertex": int((E.n_other_at_dvsec > 0).sum()),
        "n_sign_equals_w_sign": int(((E.n_sign == E.w_sign) & (E.n_d > 0)).sum()),
        "events_with_n_daughters": int((E.n_d > 0).sum()),
        "rows_k0s_per_event": float(E.n_rows_k0s.mean()), "rows_lambda_per_event": float(E.n_rows_lambda.mean()),
        "rows_k0l_per_event": float(E.n_rows_k0l.mean()), "parent_k0s_per_event": float(E.n_parent_k0s.mean()),
        "parent_lambda_per_event": float(E.n_parent_lambda.mean()),
        "secondaries_with_nonzero_parent": int(E.n_sec_parent_nonzero.sum()),
        "events_with_gen0_particle_besides_w_muon_at_vertex_1": int((E.n_other_at_wvsec > 0).sum()),
        "photons_with_parent_24": int(E.w_photons_parent24.sum()),
        "photon_parent_codes_top": dict(list(ue["parent_photon"].items())[:8]),
        "w_muon_t0_sigma_ns": float(t0.std()), "w_muon_t0_n": len(t0),
        "hits_per_event": float(E.n_hits.mean()), "rows_per_event": float(E.n_rows.mean()),
        "particles_with_hits_per_event": float(E.n_phits.mean()),
        "m_mumu_over_mN_mean_end_state": float((two.m_mumu / mass).mean()),
        "pairs_m_mumu_above_mN": int((two.m_mumu > mass).sum()),
        "pairs_m_above_mN_share_if_a_daughter_ends_beyond_1100mm": float((E2[E2.maxrend > 1100].m_mumu > mass).mean()),
        "pairs_m_above_mN_share_otherwise": float((E2[E2.maxrend <= 1100].m_mumu > mass).mean()),
        "daughter_p_ratio_opposite_over_same_charge_to_w": list(asym(S.assign(
            opp_w=np.sign(S.q) != np.sign(S.event.map(w.drop_duplicates("event").set_index("event").q))))),
        "w_muon_eta_abs_gt_4": float((w.eta.abs() > 4).mean()),
        "w_muon_pt_quantiles_50_90_95_99": np.quantile(w.pt, [0.5, 0.9, 0.95, 0.99]).round(2).tolist(),
    }
    return out


def compare_point(Ez, Sz, Ep, Sp) -> list[dict]:
    rows = []

    def add(group, name, r):
        rows.append({"group": group, "quantity": name, **r})

    hit_edges = np.arange(0, 41) - 0.5
    for q in ("n_hits", "n_rows", "n_phits", "n_gen0_phits", "n_sec_phits"):
        add("event", q, mean_test(Ep[q], Ez[q]))
        add("event", q, ks_test(Ep[q], Ez[q]))
    for q in ("n_rows_k0s", "n_rows_lambda", "n_rows_k0l", "n_parent_k0s", "n_parent_lambda"):
        add("event", q, mean_test(Ep[q], Ez[q]))
    add("event", "exactly one W muon", frac_test(Ep.n_w == 1, Ez.n_w == 1))
    add("event", "W+ (prompt mu+)", frac_test(Ep.w_sign > 0, Ez.w_sign > 0))
    add("event", "two N daughters in the table", frac_test(Ep.n_d == 2, Ez.n_d == 2))
    add("event", "N daughters all muons", frac_test(Ep.d_all_muons, Ez.d_all_muons))
    add("event", "N daughters opposite charge", frac_test(Ep.d_opposite, Ez.d_opposite))
    add("event", "N sign = W sign", frac_test(Ep.n_sign_eq_w, Ez.n_sign_eq_w))
    add("event", "W muon alone at vertex 1", frac_test(Ep.w_alone_vsec1, Ez.w_alone_vsec1))
    add("event", "other particle at the N vertex", frac_test(Ep.n_other_at_dvsec > 0, Ez.n_other_at_dvsec > 0))
    add("event", "fitted N vertex (both daughters fitted)", frac_test(np.isfinite(Ep.rv), np.isfinite(Ez.rv)))
    add("event", "r_v (fitted) [mm]", ks_test(Ep.rv, Ez.rv))
    add("event", "r_v (fitted) [mm]", mean_test(Ep.rv, Ez.rv))
    add("event", "r_v >= 100 mm among fitted", frac_test(Ep.rv[np.isfinite(Ep.rv)] >= 100, Ez.rv[np.isfinite(Ez.rv)] >= 100))
    add("event", "m(mu mu) / m_N", ks_test(Ep.m_ratio, Ez.m_ratio))
    add("event", "m(mu mu) / m_N", mean_test(Ep.m_ratio, Ez.m_ratio))
    add("event", "opening angle [rad]", ks_test(Ep.open_angle, Ez.open_angle))
    add("event", "Delta R of the pair", ks_test(Ep.dR, Ez.dR))
    add("event", "pair pT [GeV]", ks_test(Ep.pair_pt, Ez.pair_pt))
    add("event", "daughter targets per event", mean_test(Ep.n_d_target, Ez.n_d_target))
    add("event", "primary daughters per event", mean_test(Ep.n_d_primary, Ez.n_d_primary))
    for role, label in (("w", "W muon"), ("d", "N daughter")):
        a, b = Sp[Sp.role == role], Sz[Sz.role == role]
        for q, nm in (("pt", "pT [GeV]"), ("eta", "eta"), ("p", "p [GeV]")):
            add(label, nm, ks_test(a[q], b[q]))
        add(label, "pT [GeV]", mean_test(a.pt, b.pt))
        add(label, "hits", chi2_test(a.n_hits, b.n_hits, hit_edges))
        add(label, "pixel hits", chi2_test(a.n_pixel_hits, b.n_pixel_hits, hit_edges))
        add(label, ">= 5 hits", frac_test(a.n_hits >= 5, b.n_hits >= 5))
        add(label, "target (pT > 1, |eta| < 4, >= 5 hits)", frac_test(a.is_target, b.is_target))
        at, bt = a[a.is_target], b[b.is_target]
        add(label, "|d0| of targets [mm]", ks_test(at.d0.abs(), bt.d0.abs()))
        add(label, "first-hit radius of targets [mm]", ks_test(at.r_first, bt.r_first))
        add(label, ">= 3 pixel hits among targets", frac_test(at.n_pixel_hits >= 3, bt.n_pixel_hits >= 3))
    a, b = Sp[Sp.role == "d"], Sz[Sz.role == "d"]
    add("N daughter", "|d0| >= 10 mm among targets", frac_test(a[a.is_target].d0.abs() >= 10, b[b.is_target].d0.abs() >= 10))
    add("N daughter", "primary among targets", frac_test(a[a.is_target].primary, b[b.is_target].primary))
    for side, nm in ((True, "opposite"), (False, "same")):
        add("N daughter", f"p of the daughter with charge {nm} to the W muon [GeV]",
            mean_test(a[a.opp_w == side].p, b[b.opp_w == side].p))
    return rows


def asym(S):
    d = S[S.role == "d"]
    po, ps = d[d.opp_w].p, d[~d.opp_w].p
    r = po.mean() / ps.mean()
    # error of a ratio of means
    e = r * np.hypot(po.std(ddof=1) / np.sqrt(len(po)) / po.mean(), ps.std(ddof=1) / np.sqrt(len(ps)) / ps.mean())
    return r, e


# generator-level variants (directories p<m>_<ctau>_<tag> of slurm/acts/produce_hnl.sbatch SIMULATE=0); Pythia's
# meMode 22 weight is (P.p1)(p2.p3): "mu+ paired with N" is the lepton-number-conserving V-A decay of N, "mu-
# paired with N" the lepton-number-violating one, "Majorana" the 50/50 sum
GEN_LABELS = {
    "d": "phase space, 13 TeV",
    "a": "meMode 22, nu paired with N, 13 TeV",
    "e": "meMode 22, nu paired with N, 14 TeV",
    "c": "V-A, mu+ paired with N (lepton number conserved only), 13 TeV",
    "b": "V-A, mu- paired with N (lepton number violated only), 13 TeV",
    "g": "Majorana V-A, 13 TeV",
    "h": "Majorana V-A, 14 TeV",
    "t2": "Majorana V-A, 14 TeV, no ME correction of the first ISR emission",
    "t4": "Majorana V-A, 14 TeV, A14 tune",
    "t5": "Majorana V-A, 13 TeV, A14 tune",
    "t3": "Majorana V-A, 13 TeV, ISR up to the factorization scale",
    "u1": "Majorana V-A, 13 TeV, A14 tune, ISR up to the factorization scale",
    "t1": "Majorana V-A, 14 TeV, ISR up to the factorization scale (the final card)",
    "F": "final card, second sample",
    "Facts": "final card, ACTS v15 Pythia8Generator",
}


def gen_variants(genvar: str, Ez, Sz, point: str, mass: float) -> list[dict]:
    """N-decay matrix element and beam energy at generator level. Compared with Zenodo only through quantities that the
    2 T field leaves unchanged between production and the end state stored in the Zenodo tables (|p|, pT, eta): the
    charge asymmetry of the daughter momenta, daughter p and eta, W-muon eta and pT, the W+ fraction (daughters with
    p > 0.5 GeV, the simulation threshold). m(mu mu)/m_N is shown at production (generator momenta) for reference
    only: the Zenodo tables hold end-state momenta, whose azimuth has turned in the field."""
    out = []
    zr, ze = asym(Sz)
    zw = Sz[Sz.role == "w"]
    for tag in GEN_LABELS:
        d = os.path.join(genvar, f"p{int(mass)}_{int(POINTS[point][1])}_{tag}")
        ev = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(d, "gen_*_events.parquet")))]
        sg = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(d, "gen_*_signal.parquet")))]
        if not ev:
            continue
        E = pd.concat(ev, ignore_index=True)
        S = pd.concat(sg, ignore_index=True) if sg and sum(len(x) for x in sg) else pd.DataFrame()
        row = {"point": point, "variant": tag, "label": GEN_LABELS[tag], "events": len(E), "n_gen_mean": E.n_gen.mean(),
               "n_ch_p05_mean": E.n_ch_p05.mean(), "mu_vsec1": (E.n_mu_vsec1 == 1).mean(),
               "one_rule_pair": (E.n_rule_pairs == 1).mean()}
        if len(S):
            S = S.copy()
            w = S[S.role == "w"].drop_duplicates(["dir", "event"]).set_index(["dir", "event"])
            S["w_q"] = [w.q.get((a, b), np.nan) for a, b in zip(S.dir, S.event)]
            S["opp_w"] = np.sign(S.q) != np.sign(S.w_q)
            dd = S[(S.role == "d") & (S.p > 0.5)]
            both = dd.groupby(["dir", "event"]).size()
            keep = set(both[both == 2].index)
            E2 = E[[(a, b) in keep for a, b in zip(E.dir, E.event)]]
            m = E2.m_mumu / mass
            r, re = asym(dd.assign(role="d"))
            ww = S[S.role == "w"]
            row.update({
                "m_ratio_mean": m.mean(), "m_ratio_err": m.std(ddof=1) / np.sqrt(len(m)),
                "d_p_ks_p": stats.ks_2samp(dd.p, Sz[Sz.role == "d"].p).pvalue,
                "d_eta_ks_p": stats.ks_2samp(dd.eta.dropna(), Sz[Sz.role == "d"].eta.dropna()).pvalue,
                "d_p_mean": dd.p.mean(), "d_p_mean_zenodo": Sz[Sz.role == "d"].p.mean(),
                "p_ratio_opp_same": r, "p_ratio_err": re, "p_ratio_zenodo": zr, "p_ratio_zenodo_err": ze,
                "w_plus": (ww.q > 0).mean(), "w_plus_zenodo": (zw.q > 0).mean(),
                "w_eta_ks_p": stats.ks_2samp(ww.eta.dropna(), zw.eta.dropna()).pvalue,
                "w_eta_gt4": float((ww.eta.abs() > 4).mean()), "w_eta_gt4_zenodo": float((zw.eta.abs() > 4).mean()),
                "w_pt_ks_p": stats.ks_2samp(ww.pt, zw.pt).pvalue,
                "d_pt_ks_p": stats.ks_2samp(dd.pt, Sz[Sz.role == "d"].pt).pvalue,
            })
        out.append(row)
    return out


def driver_vs_acts(genvar: str, point: str, mass: float, drv: str, acts: str) -> list[dict]:
    """The stand-alone Pythia8 driver (hnl_pythia.cc) against the ACTS v15 Pythia8Generator, same card, generator
    level: barcode structure and kinematics (roles in the ACTS output from the barcode rule, hnl_census.py gen)."""
    def load_dir(tag):
        fs = sorted(glob.glob(os.path.join(genvar, f"p{int(mass)}_{int(POINTS[point][1])}_{tag}", "gen_*_events.parquet")))
        E = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        S = pd.concat([pd.read_parquet(f.replace("_events", "_signal")) for f in fs], ignore_index=True)
        return E, S
    Ea, Sa = load_dir(drv)
    Eb, Sb = load_dir(acts)
    rows = []
    for q in ("n_gen", "n_gen_p05", "n_ch_p05", "n_vsec", "vsec_max", "n_k0s", "n_lambda"):
        r = mean_test(Ea[q], Eb[q])
        rows.append({"point": point, "quantity": f"{q} per event", **r})
    for q, f in (("W muon alone at vertex 1", lambda E: E.n_mu_vsec1 == 1),
                 ("exactly one two-muon vertex (the brief's rule)", lambda E: E.n_rule_pairs == 1)):
        rows.append({"point": point, "quantity": q, **frac_test(f(Ea), f(Eb))})
    for role, nm in (("w", "W muon"), ("d", "N daughter")):
        for q in ("pt", "eta", "p", "r_prod"):
            rows.append({"point": point, "quantity": f"{nm} {q}",
                         **ks_test(Sa[Sa.role == role][q], Sb[Sb.role == role][q])})
    return rows


VARIANT_KEYS = (("event", "n_rows", "mean"), ("event", "m(mu mu) / m_N", "mean"), ("W muon", "pT [GeV]", "KS (median shown)"),
                ("N daughter", "pT [GeV]", "KS (median shown)"), ("W muon", ">= 5 hits", "fraction"),
                ("N daughter", ">= 3 pixel hits among targets", "fraction"))


def variant_table(census: str, tags: list[str], labels: dict) -> list[str]:
    """One line per pilot variant and point: failing tests and the quantities that separate the variants."""
    head = ("| variant | point | events | p < 0.01 | p < 0.001 | rows / event (z) | m(mu mu)/m_N (z) | W-muon pT KS p "
            "| daughter pT KS p | W muon >= 5 hits (z) | daughter targets >= 3 pixel hits (z) |")
    lines = [head, "|" + "---|" * 11]
    for t in tags:
        f = os.path.join(census, f"closure_{t}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        summ = json.load(open(os.path.join(census, f"closure_{t}_summary.json")))
        for point in POINTS:
            d = df[df.point == point].set_index(["group", "quantity", "test"])
            v = [d.loc[k] for k in VARIANT_KEYS]
            s = summ[point]
            lines.append(f"| {labels.get(t, t)} | {point} | {s['pilot_events']:,} | {s['p_below_0.01']} | "
                         f"{s['p_below_0.001']} | {v[0].z:+.1f} | {v[1].z:+.1f} | {v[2].p:.2g} | {v[3].p:.2g} | "
                         f"{v[4].z:+.1f} | {v[5].z:+.1f} |")
    return lines


def fmt(x, nd=3):
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "-"
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:.{nd}f}" if abs(x) < 100 else f"{x:.1f}"
    return str(x)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--pilot", default="pilot")
    ap.add_argument("--genvar")
    ap.add_argument("--driver-acts", help="TAG_DRIVER,TAG_ACTS: generator-level runs of the same card to compare")
    ap.add_argument("--variants", help="TAG=label;TAG=label...: summary line per pilot variant (closure_<TAG>.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    allrows, genrows, summary = [], [], {}
    for point, (mass, ctau) in POINTS.items():
        Ez, Sz, Uz = load(os.path.join(a.census, f"zenodo_{point}"))
        Ez, Sz = derived(Ez, Sz, mass)
        zfacts = zenodo_facts(Ez, Sz, Uz, mass)
        pfx = os.path.join(a.census, f"{a.pilot}_{point}")
        if os.path.exists(pfx + "_events.parquet"):
            Ep, Sp, _ = load(pfx)
            Ep, Sp = derived(Ep, Sp, mass)
            rows = compare_point(Ez, Sz, Ep, Sp)
            for r in rows:
                r["point"] = point
            allrows += rows
            for r in rows:
                r["known"] = (r["group"], r["quantity"]) in KNOWN
            p = np.array([r["p"] for r in rows if not r["known"]])
            summary[point] = {"tests": len(p), "p_below_0.01": int((p < 0.01).sum()),
                              "p_below_0.001": int((p < 0.001).sum()), "expected_below_0.01": 0.01 * len(p),
                              "known_difference_tests": int(sum(r["known"] for r in rows)),
                              "pilot_events": len(Ep), "zenodo_events": len(Ez),
                              "w_t0_sigma_ns_zenodo": float(np.nanstd(Ez.w_t0)),
                              "w_t0_sigma_ns_pilot": float(np.nanstd(Ep.w_t0))}
            if "rv_true" in Ep:
                f = Ep[np.isfinite(Ep.rv)]
                summary[point]["rv_fit_minus_true_median_mm"] = float(np.median(f.rv - f.rv_true))
                summary[point]["rv_fit_minus_true_mad_mm"] = float(stats.median_abs_deviation(f.rv - f.rv_true))
                dd = Sp[(Sp.role == "d") & Sp.is_target]
                summary[point]["d0_fit_minus_true_median_mm"] = float(np.median(dd.d0.abs() - dd.d0_true))
                summary[point]["d0_fit_minus_true_mad_mm"] = float(stats.median_abs_deviation(dd.d0.abs() - dd.d0_true))
                L = Ep.L_true / Ep.betagamma
                summary[point]["proper_decay_length_mean_mm"] = float(L.mean())
                summary[point]["proper_decay_length_err_mm"] = float(L.std(ddof=1) / np.sqrt(len(L)))
                summary[point]["n_tries_gt1"] = int((Ep.n_tries > 1).sum())
        else:
            summary[point] = {"zenodo_events": len(Ez), "w_t0_sigma_ns_zenodo": float(np.nanstd(Ez.w_t0))}
        summary[point]["zenodo_truth"] = zfacts
        if a.genvar:
            genrows += gen_variants(a.genvar, Ez, Sz, point, mass)
        if a.genvar and a.driver_acts:
            da = a.driver_acts.split(",")
            dvrows = driver_vs_acts(a.genvar, point, mass, da[0], da[1])
            summary.setdefault("driver_vs_acts", {})[point] = {
                "tests": len(dvrows), "p_below_0.01": int(sum(r["p"] < 0.01 for r in dvrows))}
            allrows_dv = summary.setdefault("_dv", [])
            allrows_dv += dvrows
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if allrows:
        pd.DataFrame(allrows).to_csv(a.out + ".csv", index=False)
    if genrows:
        pd.DataFrame(genrows).to_csv(a.out + "_genvar.csv", index=False)
    dv = summary.pop("_dv", [])
    if dv:
        pd.DataFrame(dv).to_csv(a.out + "_driver_vs_acts.csv", index=False)
    with open(a.out + "_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    # markdown tables (pasted into the results file)
    lines = []
    if allrows:
        df = pd.DataFrame(allrows)
        for point in POINTS:
            d = df[df.point == point]
            if not len(d):
                continue
            s = summary[point]
            lines += [f"### {point}: pilot ({s['pilot_events']:,} events) against Zenodo train + val "
                      f"({s['zenodo_events']:,} events)", "",
                      "| group | quantity | test | pilot | Zenodo | z or statistic | p |", "|---|---|---|---|---|---|---|"]
            for r in d.itertuples():
                pe = f" +- {fmt(r.pilot_err)}" if np.isfinite(r.pilot_err) else ""
                ze = f" +- {fmt(r.zenodo_err)}" if np.isfinite(r.zenodo_err) else ""
                flag = " (known)" if r.known else (" **" if r.p < 0.01 else "")
                lines.append(f"| {r.group} | {r.quantity} | {r.test} | {fmt(r.pilot)}{pe} | {fmt(r.zenodo)}{ze} | "
                             f"{fmt(r.z, 2)} | {r.p:.3g}{flag} |")
            lines += ["", f"Tests {s['tests']} (without the {s['known_difference_tests']} marked known), p < 0.01: "
                      f"{s['p_below_0.01']} (chance: {s['expected_below_0.01']:.1f}), p < 0.001: {s['p_below_0.001']}.",
                      ""]
    if genrows:
        g = pd.DataFrame(genrows)
        lines += ["### Generator-level variants (N decay matrix element, beam energy, initial-state shower)", "",
                  "| point | variant | events | m(mu mu)/m_N at production | p(opp)/p(same) | Zenodo | mean daughter p "
                  "[GeV] | Zenodo | daughter p KS p | daughter pT KS p | daughter eta KS p | W+ | Zenodo "
                  "| W-muon eta KS p | W-muon pT KS p |", "|" + "---|" * 15]
        for r in g.itertuples():
            lines.append(f"| {r.point} | {r.label} | {r.events:,} | {fmt(r.m_ratio_mean)} +- {fmt(r.m_ratio_err)} | "
                         f"{fmt(r.p_ratio_opp_same)} +- {fmt(r.p_ratio_err)} | {fmt(r.p_ratio_zenodo)} +- "
                         f"{fmt(r.p_ratio_zenodo_err)} | {fmt(r.d_p_mean, 1)} | {fmt(r.d_p_mean_zenodo, 1)} | "
                         f"{r.d_p_ks_p:.3g} | {r.d_pt_ks_p:.3g} | {r.d_eta_ks_p:.3g} | {fmt(r.w_plus)} | "
                         f"{fmt(r.w_plus_zenodo)} | {r.w_eta_ks_p:.3g} | {r.w_pt_ks_p:.3g} |")
    if dv:
        lines += ["", "### Driver (hnl_pythia.cc) against the ACTS v15 Pythia8Generator, same card, generator level", "",
                  "| point | quantity | test | driver | ACTS generator | z or statistic | p |", "|---|---|---|---|---|---|---|"]
        for r in dv:
            pe = f" +- {fmt(r['pilot_err'])}" if np.isfinite(r["pilot_err"]) else ""
            ze = f" +- {fmt(r['zenodo_err'])}" if np.isfinite(r["zenodo_err"]) else ""
            lines.append(f"| {r['point']} | {r['quantity']} | {r['test']} | {fmt(r['pilot'])}{pe} | {fmt(r['zenodo'])}{ze} | "
                         f"{fmt(r['z'], 2)} | {r['p']:.3g} |")
    if a.variants:
        lab = dict(x.split("=", 1) for x in a.variants.split(";"))
        lines += ["", "### Full-simulation pilot variants against Zenodo train + val", ""] + \
            variant_table(a.census, list(lab), lab)
    with open(a.out + "_tables.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
