"""Track metrics of Canto-L on TrackML (docs/trackml_plan.md section 4): ours (docs/definitions.md sections 5-6) and the
published definitions P0-P4 reproduced exactly as the plan fixes them, as poolable per-event counts.

Per event: pid (particle id per hit, 0 = noise), labels (track label per hit, -1 = no track; builder tracks have >= 2
hits), the event's particles table (hnl.trackml.convert: particle_id, pt, eta, n_hits, is_target, ...), and for P0 the
hit ids and score weights. In a truth-filtered condition (nonoise, 1gev) the arrays hold the kept hits only; whole
particles or noise hits are removed, so |P| (hits of the particle in the arrays) equals the table's n_hits for every kept
particle. Matching (strict majorities): DM if |T n P| > |T| / 2 and |T n P| > |P| / 2; SM if |T n P| > |T| / 2. |T|
counts every hit of the track, noise included (except variant J', which drops noise hits first). Noise is never a
particle: a track whose majority is noise matches nothing.

Ours (plan 4.1): efficiency = DM-matched targets / targets (target: pT > 1 GeV, |eta| < 4, >= 5 hits; populations by the
table flags); fake rate = tracks with |T| >= 5 that DM-match no particle / tracks with |T| >= 5; duplicate rate = sum over
targets of max(k - 1, 0) / targets, k = SM-matched tracks with |T| >= 5.
P0 TrackML score (Amrouche et al. arXiv:1904.06778v3 p. 10; trackml-library score_event): every hit gets a track id;
unassigned hits get unique singleton ids. Needs the stored weights (val, test, T2, dev, calibration events).
P1 Ju et al., EPJC 81 (2021) 876, p. 5: selected = |eta| < 4 and pT > 0.5 GeV (headline) or > 0.1 GeV ("without any
fiducial pT cut"), over every particles-table row; reconstructable = selected and >= 5 hits; eps_phys = DM-matched
selected / selected; eps_tech = DM-matched reconstructable / reconstructable; purity J = tracks DM-matched to a selected
particle / tracks (all builder tracks, >= 2 hits; variant: tracks with >= 5 hits); J' (their eval_reco_trkx.py with
min_hits = 4): noise hits and the hits of particles with n_hits <= 4 dropped before matching, every track id with a kept
hit counted (so the eps_phys numerator holds reconstructable particles only).
P2 Liu et al., J. Phys. Conf. Ser. 3206 (2026) 012084, p. 5: reconstructable = > 5 hits and pT >= 1 GeV (also with
|eta| < 4); Eff = DM-matched reconstructable / reconstructable; FR_Liu = 1 - tracks DM-matched to any particle / tracks
(all builder tracks).
P3 / P4 Chan et al. 2025 (arXiv:2506.03415v1 p. 4) / Calafiura et al. 2024 (arXiv:2407.13925v1 p. 5): candidates =
tracks with >= 3 hits; efficiency = targets with at least one SM-matched candidate / targets; duplication = sum over
particles of max(k - 1, 0) / candidates (k = SM-matched candidates); fake = candidates SM-matched to no particle /
candidates. Condition C-full (P3) and C-1gev (P4).

    python -m hnl.trackml.evaluate references        (the published numbers of plan section 4.3, as JSON)
    python -m hnl.trackml.evaluate report --grid DIR --condition full --out DIR   (bootstrap of every configuration of a
        hnl.trackml.trackbuild grid and the comparison rows with the references of that condition, plan 4.3-4.4)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from hnl.overlay import grant as GRANT

JP_MIN_HITS = 4   # variant J': their "reconstructable" is nhits > min_hits; 4 = "at least five spacepoints" (plan 4.2)

# published numbers (plan section 4.3; value, quoted uncertainty or None)
REFERENCES = {
    "P1": {"source": "Ju et al., Eur. Phys. J. C 81 (2021) 876, p. 5", "condition": "nonoise (and full)",
           "eps_tech_pt05": (0.976, 0.003), "eps_phys_pt05": (0.887, 0.003), "eps_tech_pt01": (0.913, 0.002),
           "eps_phys_pt01": (0.672, 0.001), "purity": (0.583, 0.006), "trackml_score": (0.877, 0.005),
           "trackml_score_trained_with_noise": (0.914, 0.006)},
    "P2": {"source": "Liu et al., J. Phys.: Conf. Ser. 3206 (2026) 012084, p. 5",
           "full": {"EC-GNN": (0.9635, 0.5558), "BC-HGNN": (0.9786, 0.3671), "Truth-CC": (0.9775, 0.5767)},
           "1gev": {"BC-HGNN": (0.9855, 0.0123), "Truth-CC": (0.9872, 0.0587)}, "columns": ("Eff", "FR_Liu")},
    "P3": {"source": "Chan, Wang, Calafiura, EPJ Web Conf. 337 (2025) 01121 (arXiv:2506.03415v1 p. 4)",
           "condition": "full", "efficiency": (0.9587, 0.0003), "duplication": (0.0276, 0.0002),
           "fake": (0.0019, 0.0001)},
    "P4": {"source": "Calafiura et al., arXiv:2407.13925v1, Table 1 p. 6", "condition": "1gev",
           "EC": {"efficiency": (0.9898, 0.0009), "duplication": (0.0421, 0.0011), "fake": (0.0012, 0.0)},
           "EggNet": {"efficiency": (0.9956, 0.0003), "duplication": (0.0129, 0.0002), "fake": (0.0006, 0.0)}},
    "P0": {"source": "Amrouche et al., arXiv:1904.06778v3 section 2.4", "kaggle_best_context_only": 0.92182},
}


def particle_sets(parts: pd.DataFrame) -> dict[str, np.ndarray]:
    """Particle ids of every population (plan sections 3.4 and 4.2) from a particles table."""
    from hnl.trackml.convert import reference_masks
    m = reference_masks(parts)
    pid = parts.particle_id.to_numpy()
    out = {k: pid[np.asarray(v, dtype=bool)] for k, v in m.items()}
    for col in ("is_prompt_target", "is_displaced_target", "is_far_displaced_target", "is_hard_scatter",
                "is_target_pix3"):
        if col in parts:
            flag = parts[col].to_numpy(bool)
            if col == "is_hard_scatter":
                flag = flag & parts.is_target.to_numpy(bool)
                col = "is_hard_scatter_target"
            out[col.removeprefix("is_")] = pid[flag]
    if "category" in parts:
        t = parts.is_target.to_numpy(bool) & parts.is_displaced_target.to_numpy(bool)
        for c in ("K_S", "Lambda", "other"):
            out[f"displaced_target_{c}"] = pid[t & (parts.category.to_numpy() == c)]
    return out


def _match(pid: np.ndarray, labels: np.ndarray):
    particles, pidx = np.unique(np.asarray(pid), return_inverse=True)
    P = len(particles)
    n_particle = np.bincount(pidx, minlength=P)
    on = np.asarray(labels) >= 0
    t, p = np.asarray(labels)[on], pidx[on]
    if len(t) == 0:
        e = np.zeros(0, dtype=np.int64)
        return particles, n_particle, e, e, e, e, e
    tu, tinv = np.unique(t, return_inverse=True)
    n_track = np.bincount(tinv)
    key, n_shared = np.unique(tinv.astype(np.int64) * P + p, return_counts=True)
    return particles, n_particle, tu, n_track, key // P, key % P, n_shared


def event_counts(pid, labels, parts: pd.DataFrame, hit_id=None, weight=None) -> tuple[dict, pd.DataFrame]:
    """Poolable counts of every metric for one event, and per-target rows (particle_id, dm, sm_any, k_sm_ge5)."""
    pid = np.asarray(pid, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    sets = particle_sets(parts)
    particles, n_particle, tu, n_track, pt, pp, n_shared = _match(pid, labels)
    P, T = len(particles), len(tu)
    real = particles[pp] != 0 if len(pp) else np.zeros(0, dtype=bool)
    dm = (2 * n_shared > n_track[pt]) & (2 * n_shared > n_particle[pp]) & real
    sm = (2 * n_shared > n_track[pt]) & real
    dm_p = np.zeros(P, dtype=bool)
    dm_p[pp[dm]] = True
    trk_dm_pid = np.full(T, 0, dtype=np.int64)
    trk_dm_pid[pt[dm]] = particles[pp[dm]]
    ge5, ge3 = n_track >= 5, n_track >= 3
    k_sm5 = np.bincount(pp[sm & ge5[pt]], minlength=P)
    k_sm3 = np.bincount(pp[sm & ge3[pt]], minlength=P)
    trk_sm3 = np.zeros(T, dtype=bool)
    trk_sm3[pt[sm & ge3[pt]]] = True

    def in_set(name):
        return np.isin(particles, sets[name])

    c = {"n_tracks": int(T), "n_tracks_ge5": int(ge5.sum()), "n_fake_ge5": int((ge5 & (trk_dm_pid == 0)).sum()),
         "n_tracks_dm_any": int((trk_dm_pid != 0).sum()), "n_cand_ge3": int(ge3.sum()),
         "n_cand_ge3_unmatched": int((ge3 & ~trk_sm3).sum()), "dup_sum_ge3_all": int(np.maximum(k_sm3 - 1, 0).sum())}
    for name in sets:
        ids = sets[name]
        m = in_set(name)
        c[f"n_{name}"] = int(len(ids))                       # denominators over the particles table
        c[f"m_{name}"] = int(dm_p[m].sum())                  # DM-matched among them (particles with hits)
    tgt = in_set("target")
    c["dup_sum_targets"] = int(np.maximum(k_sm5[tgt] - 1, 0).sum())
    c["m_target_sm3"] = int((k_sm3[tgt] > 0).sum())
    for nm in ("P1_selected_pt05", "P1_selected_pt01"):
        sel = np.isin(trk_dm_pid, sets[nm]) & (trk_dm_pid != 0)
        c[f"tracks_dm_{nm}"] = int(sel.sum())
        c[f"tracks_ge5_dm_{nm}"] = int((sel & ge5).sum())
    # J' (eval_reco_trkx.py, min_hits = JP_MIN_HITS): noise hits and the hits of particles with n_hits <= min_hits are
    # dropped before matching
    nh = pd.Series(parts.n_hits.to_numpy(), index=parts.particle_id.to_numpy())
    keep = (pid != 0) & (nh.reindex(pid).fillna(0).to_numpy() > JP_MIN_HITS)
    particles2, n_particle2, tu2, n_track2, pt2, pp2, sh2 = _match(pid[keep], labels[keep])
    dm2 = (2 * sh2 > n_track2[pt2]) & (2 * sh2 > n_particle2[pp2])
    trk2 = np.full(len(tu2), 0, dtype=np.int64)
    trk2[pt2[dm2]] = particles2[pp2[dm2]]
    dmp2 = np.zeros(len(particles2), dtype=bool)
    dmp2[pp2[dm2]] = True
    c["n_tracks_jprime"] = int(len(tu2))
    for nm in ("P1_selected_pt05", "P1_selected_pt01", "P1_reconstructable_pt05", "P1_reconstructable_pt01"):
        c[f"mJp_{nm}"] = int(dmp2[np.isin(particles2, sets[nm])].sum())
    for nm in ("P1_selected_pt05", "P1_selected_pt01"):
        c[f"tracks_jprime_dm_{nm}"] = int((np.isin(trk2, sets[nm]) & (trk2 != 0)).sum())
    if hit_id is not None and weight is not None and np.isfinite(np.asarray(weight, dtype=np.float64)).all():
        c["trackml_score"] = trackml_score(hit_id, pid, weight, labels)
        c["n_score_events"] = 1
    rows = pd.DataFrame({"particle_id": particles[tgt], "dm": dm_p[tgt], "sm_any": k_sm3[tgt] > 0,
                         "k_sm_ge5": k_sm5[tgt].astype(np.int16)})
    return c, rows


def trackml_score(hit_id, pid, weight, labels) -> float:
    """TrackML score of one event (trackml-library score_event); unassigned hits get unique singleton track ids."""
    from trackml.score import score_event
    labels = np.asarray(labels, dtype=np.int64)
    tid = labels.copy()
    free = tid < 0
    tid[free] = labels.max(initial=-1) + 1 + np.arange(int(free.sum()))
    truth = pd.DataFrame({"hit_id": np.asarray(hit_id, dtype=np.int64), "particle_id": np.asarray(pid, dtype=np.int64),
                          "weight": np.asarray(weight, dtype=np.float64)})
    sub = pd.DataFrame({"hit_id": truth.hit_id, "track_id": tid})
    return float(score_event(truth, sub))


def pooled(counts: pd.DataFrame) -> dict:
    """Metric values from summed per-event counts (rows = events, possibly resampled)."""
    s = counts.sum(numeric_only=True)
    r = lambda a, b: float(s[a] / s[b]) if s.get(b, 0) else np.nan
    out = {
        "ours_eff_target": r("m_target", "n_target"),
        "ours_eff_prompt_target": r("m_prompt_target", "n_prompt_target"),
        "ours_eff_displaced_target": r("m_displaced_target", "n_displaced_target"),
        "ours_eff_far_displaced_target": r("m_far_displaced_target", "n_far_displaced_target"),
        "ours_eff_hard_scatter_target": r("m_hard_scatter_target", "n_hard_scatter_target"),
        "ours_fake_rate": r("n_fake_ge5", "n_tracks_ge5"), "ours_duplicate_rate": r("dup_sum_targets", "n_target"),
        "P1_eps_phys_pt05": r("m_P1_selected_pt05", "n_P1_selected_pt05"),
        "P1_eps_tech_pt05": r("m_P1_reconstructable_pt05", "n_P1_reconstructable_pt05"),
        "P1_eps_phys_pt01": r("m_P1_selected_pt01", "n_P1_selected_pt01"),
        "P1_eps_tech_pt01": r("m_P1_reconstructable_pt01", "n_P1_reconstructable_pt01"),
        "P1_purity_J_pt05": r("tracks_dm_P1_selected_pt05", "n_tracks"),
        "P1_purity_J_pt01": r("tracks_dm_P1_selected_pt01", "n_tracks"),
        "P1_purity_J_ge5_pt05": r("tracks_ge5_dm_P1_selected_pt05", "n_tracks_ge5"),
        "P1_purity_J_ge5_pt01": r("tracks_ge5_dm_P1_selected_pt01", "n_tracks_ge5"),
        "P1_Jprime_eps_tech_pt05": r("mJp_P1_reconstructable_pt05", "n_P1_reconstructable_pt05"),
        "P1_Jprime_eps_phys_pt05": r("mJp_P1_selected_pt05", "n_P1_selected_pt05"),
        "P1_Jprime_purity_pt05": r("tracks_jprime_dm_P1_selected_pt05", "n_tracks_jprime"),
        "P1_Jprime_purity_pt01": r("tracks_jprime_dm_P1_selected_pt01", "n_tracks_jprime"),
        "P2_eff": r("m_P2_reconstructable", "n_P2_reconstructable"),
        "P2_eff_eta4": r("m_P2_reconstructable_eta4", "n_P2_reconstructable_eta4"),
        "P2_fr_liu": 1.0 - r("n_tracks_dm_any", "n_tracks") if s.get("n_tracks", 0) else np.nan,
        "P3_efficiency": r("m_target_sm3", "n_target"), "P3_duplication": r("dup_sum_ge3_all", "n_cand_ge3"),
        "P3_fake": r("n_cand_ge3_unmatched", "n_cand_ge3"),
    }
    if "trackml_score" in counts:
        sc = counts.trackml_score.dropna()
        out["P0_trackml_score"] = float(sc.mean()) if len(sc) else np.nan
    return out


def bootstrap(counts: pd.DataFrame, B: int = 500, seed: int = 12345) -> pd.DataFrame:
    """Event bootstrap: B resamples of the event rows (one fixed list of indices for the seed, plan section 4.1);
    returns per metric the pooled value, the bootstrap standard deviation and the 16 / 84 % quantiles."""
    rng = np.random.default_rng(seed)
    n = len(counts)
    idx = rng.integers(0, n, size=(B, n))
    vals = pd.DataFrame([pooled(counts.iloc[i]) for i in idx])
    base = pooled(counts)
    return pd.DataFrame({"value": pd.Series(base), "boot_std": vals.std(), "q16": vals.quantile(0.16),
                         "q84": vals.quantile(0.84)})


def wording(delta: float, s_ours: float, s_pub: float | None) -> str:
    """Plan section 4.4: higher / lower beyond 2 sigma_Delta, else compatible."""
    sd = float(np.hypot(s_ours, s_pub or 0.0))
    if delta > 2 * sd:
        return "higher"
    if delta < -2 * sd:
        return "lower"
    return "compatible"


# (our metric, reference, reference key, condition) for the comparison table of plan section 4.3
COMPARISONS = [
    ("P1_eps_tech_pt05", "P1", "eps_tech_pt05", ("nonoise", "full")),
    ("P1_eps_phys_pt05", "P1", "eps_phys_pt05", ("nonoise", "full")),
    ("P1_eps_tech_pt01", "P1", "eps_tech_pt01", ("nonoise", "full")),
    ("P1_eps_phys_pt01", "P1", "eps_phys_pt01", ("nonoise", "full")),
    ("P1_purity_J_pt01", "P1", "purity", ("nonoise", "full")),
    ("P1_Jprime_purity_pt01", "P1", "purity", ("nonoise", "full")),
    ("P0_trackml_score", "P1", "trackml_score", ("nonoise", "full")),
    ("P0_trackml_score", "P1", "trackml_score_trained_with_noise", ("nonoise", "full")),
    ("P2_eff", "P2", ("full", "EC-GNN", 0), ("full",)),
    ("P2_fr_liu", "P2", ("full", "EC-GNN", 1), ("full",)),
    ("P2_eff", "P2", ("full", "BC-HGNN", 0), ("full",)),
    ("P2_fr_liu", "P2", ("full", "BC-HGNN", 1), ("full",)),
    ("P2_eff", "P2", ("1gev", "BC-HGNN", 0), ("1gev",)),
    ("P2_fr_liu", "P2", ("1gev", "BC-HGNN", 1), ("1gev",)),
    ("P3_efficiency", "P3", "efficiency", ("full",)),
    ("P3_duplication", "P3", "duplication", ("full",)),
    ("P3_fake", "P3", "fake", ("full",)),
    ("P3_efficiency", "P4", ("EC", "efficiency"), ("1gev",)),
    ("P3_duplication", "P4", ("EC", "duplication"), ("1gev",)),
    ("P3_fake", "P4", ("EC", "fake"), ("1gev",)),
    ("P3_efficiency", "P4", ("EggNet", "efficiency"), ("1gev",)),
    ("P3_duplication", "P4", ("EggNet", "duplication"), ("1gev",)),
    ("P3_fake", "P4", ("EggNet", "fake"), ("1gev",)),
]


def _ref_value(ref: str, key):
    d = REFERENCES[ref]
    if ref == "P2":
        cond, model, col = key
        v = d[cond][model][col]
        return v, None                                   # Liu et al. quote no uncertainties
    if isinstance(key, tuple):
        return d[key[0]][key[1]]
    return d[key]


def report(grid: str, condition: str, out: str, B: int = 500, seed: int = 12345) -> pd.DataFrame:
    import os
    counts = pd.read_parquet(os.path.join(grid, "per_event.parquet"))
    GRANT.check(counts.event.unique(), "eval", condition, out=out)   # overlay ids only under a grant (addendum H)
    cfgs = json.load(open(os.path.join(grid, "configs.json")))
    names_path = os.path.join(grid, "config_names.json")
    names = json.load(open(names_path)) if os.path.exists(names_path) else [None] * len(cfgs)
    rows = []
    for i, g in counts.groupby("config"):
        g = g.sort_values("event")             # the same resampled event lists for every configuration
        bt = bootstrap(g.drop(columns=["event", "config"]), B, seed)
        for metric, r in bt.iterrows():
            rows.append({"config": int(i), "name": names[int(i)], "family": cfgs[int(i)][0],
                         "params": ",".join(f"{x:g}" for x in cfgs[int(i)][1:]), "condition": condition,
                         "metric": metric, "value": r.value, "boot_std": r.boot_std, "q16": r.q16, "q84": r.q84,
                         "n_events": len(g)})
    tab = pd.DataFrame(rows)
    comp = []
    for metric, ref, key, conds in COMPARISONS:
        if condition not in conds:
            continue
        pub, s_pub = _ref_value(ref, key)
        for _, r in tab[tab.metric == metric].iterrows():
            delta = r.value - pub
            comp.append({"config": r.config, "name": r["name"], "condition": condition, "metric": metric,
                         "reference": ref, "reference_key": str(key), "published": pub, "published_err": s_pub,
                         "ours": r.value, "ours_err": r.boot_std, "delta": delta,
                         "wording": wording(delta, r.boot_std, s_pub)})
    os.makedirs(out, exist_ok=True)
    tab.to_csv(os.path.join(out, f"metrics_{condition}.csv"), index=False)
    pd.DataFrame(comp).to_csv(os.path.join(out, f"comparison_{condition}.csv"), index=False)
    return pd.DataFrame(comp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("references")
    r = sub.add_parser("report")
    r.add_argument("--grid", required=True)
    r.add_argument("--condition", default="full")
    r.add_argument("--out", required=True)
    r.add_argument("--B", type=int, default=500)
    r.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args(argv)
    if a.cmd == "references":
        print(json.dumps(REFERENCES, indent=1))
    else:
        c = report(a.grid, a.condition, a.out, a.B, a.seed)
        pd.set_option("display.width", 220)
        print(c.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
