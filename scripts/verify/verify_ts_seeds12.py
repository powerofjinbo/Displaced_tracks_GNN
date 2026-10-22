"""Independent recomputation of the R19 / R20 read-outs (ledger R21) from the stored per-particle / per-event tables.

Imports nothing from hnl.*. Recomputed here from the raw truth columns (docs/definitions.md sections 3-5,
docs/preregistration.md section 6, docs/preregistration_new_sample.md section 4, docs/addendum_g_t2_readouts.md
section 4):
  - HNL test sets: targets (pT > 1 GeV, |eta| < 4, >= 5 hits; the hits counted here from the hit table), categories
    (labeled samples: |parent| = 50 / |parent| = 24 and |PDG| = 13; hnl15_100: the vsec rule of definitions.md
    section 4, reimplemented), |d0| (muons fixed-R fit, others free fit; stored truth fits) and r_v (stored pair
    vertices with ok). Primary = HNL-daughter targets with |d0| >= 10 mm or r_v >= 100 mm.
  - Efficiency = matched targets / targets, with the stored dm_matched flag of the track builder output (the hit-level
    track labels are not stored, so the double-majority matching itself is not redone). Fake rate = sum n_fake /
    sum n_tracks_ge5 of the stored per-event table.
  - Generated sets: P_G (gun: vsec >= 1000, gen 0, sub 0), P_H (generator: gen 0, sub 0; |d0| >= 10 or r_prod >= 100),
    P_HNL fitted (N daughter targets, |d0_fit| >= 10 or rv_fit >= 100) from the T1 / T2 truth columns.
  - Uncertainties: the stored bootstrap replicate tables (derived/bootstrap/v1, generated_v1; units = event index),
    ratio of sums per replicate, sigma_boot = sd of the replicates of the seed mean, sigma^2 = sigma_boot^2 + s^2 / 3.
  - Selection: the builder rule (DECISIONS.md 2026-10-04 20:18 UTC) re-applied to the stored val tables of each run's
    mwslmerge configurations, and the configurations of the stored test outputs compared with the committed
    selection file.
Writes a JSON of every recomputed number; `compare` writes the text summary against the committed results files.
Usage: python -I verify_ts_seeds12.py compute OUT.json ; python -I verify_ts_seeds12.py compare OUT.json SUMMARY.txt
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

DATA = "/ceph/submit/data/user/j/jinboz1/hnl_lrt"
REPO = "/work/submit/jinboz1/hnl/hnl-displaced"
EXTRA = "/work/submit/jinboz1/hnl/runs_extra/ts_seeds12"
DER = f"{DATA}/derived"
FR_STAR = 0.02936834277897644
CODE = {"ttbar": 1, "hnl10_100": 2, "hnl15_100": 3, "hnl15_200": 4}
HNL_SETS = {"hnl10_100_test": ("hnl10_100", "hnl10_100_test"), "hnl15_200_test": ("hnl15_200", "hnl15_200_test"),
            "hnl15_100_external": ("hnl15_100", "hnl15_100_external")}
D0_EDGES = [0, 0.1, 1, 3, 10, 30, 100, 300, np.inf]
RV_EDGES = [0, 10, 30, 100, 200, 300, np.inf]
MODELS = {"ii": "Canto-L (HNL)", "iii": "Canto-L (ttbar+HNL)"}
GMODEL = {"ii": "canto_l_hnl", "iii": "canto_l_tthnl"}
POINTS = {"hnl5_50": (10, "hnl_5_50"), "hnl20_200": (11, "hnl_20_200"), "hnl15_500": (12, "hnl_15_500"),
          "hnl10_100_ref": (13, "hnl_10_100"), "hnl15_200_ref": (14, "hnl_15_200")}


def split(name):
    return np.array(sorted(int(x) for x in open(f"{REPO}/splits/{name}.txt") if x.strip() and not x.startswith("#")))


def shards(table, sample, events, columns):
    d = f"{DER}/events/v1/{table}/{sample}"
    ev = set(int(e) for e in events)
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".parquet"):
            continue
        first = int(f[6:11])
        if not any(first <= e < first + 500 for e in ev):
            continue
        out.append(pd.read_parquet(f"{d}/{f}", columns=columns, filters=[("event", "in", sorted(ev))]))
    return pd.concat(out, ignore_index=True)


def hnl15_100_rule(p):
    """definitions.md section 4: HNL daughters = the opposite-charge gen-0 muon pair alone (as gen-0 particles) at one
    vsec > 1, smallest such vsec if several; W muon = the only gen-0 muon at vsec = 1."""
    g0 = p[p["gen"] == 0]
    hnl, w = set(), set()
    for e, ge in g0.groupby("event"):
        cand = []
        for v, gv in ge[ge.vsec > 1].groupby("vsec"):
            if len(gv) == 2 and (gv.pdg.abs() == 13).all() and gv.q.sum() == 0 and (gv.q != 0).all():
                cand.append(v)
        if cand:
            v = min(cand)
            hnl.update((e, int(x)) for x in ge[ge.vsec == v].particle_id)
        mu1 = ge[(ge.vsec == 1) & (ge.pdg.abs() == 13)]
        if len(mu1) == 1:
            w.add((e, int(mu1.particle_id.iloc[0])))
    key = list(zip(p.event.astype(int), p.particle_id.astype(int)))
    return np.array([k in hnl for k in key]), np.array([k in w for k in key])


def truth_hnl(sample, events):
    p = shards("particles", sample, events, ["event", "particle_id", "pdg", "q", "pt", "eta", "parent_pdg", "vsec",
                                              "gen", "n_hits", "category", "is_target"])
    h = shards("hits", sample, events, ["event", "pid"])
    nh = h.groupby(["event", "pid"]).size().rename("n_hits_mine").reset_index().rename(columns={"pid": "particle_id"})
    p = p.merge(nh, on=["event", "particle_id"], how="left")
    p["n_hits_mine"] = p.n_hits_mine.fillna(0).astype(int)
    chk = {"n_hits_mismatch": int((p.n_hits_mine != p.n_hits).sum())}
    p["tgt"] = (p.pt > 1.0) & (p.eta.abs() < 4.0) & (p.n_hits_mine >= 5)
    chk["target_mismatch"] = int((p.tgt != p.is_target.astype(bool)).sum())
    if sample == "hnl15_100":
        is_h, is_w = hnl15_100_rule(p)
    else:
        is_h = (p.parent_pdg.abs() == 50).to_numpy()
        is_w = ((p.parent_pdg.abs() == 24) & (p.pdg.abs() == 13)).to_numpy()
    cat = np.where(is_h, "hnl", np.where(is_w, "w", "other"))
    chk["category_mismatch"] = int((cat != p.category.to_numpy()).sum())
    p["cat"] = cat
    f = pd.read_parquet(f"{DER}/truth_fits/v1/particles_{sample}.parquet",
                        columns=["event", "particle_id", "pdg", "fix_d0", "free_d0"])
    f = f[f.event.isin(events)]
    f["d0"] = np.where(f.pdg.abs() == 13, f.fix_d0, f.free_d0)
    p = p.merge(f[["event", "particle_id", "d0"]], on=["event", "particle_id"], how="left")
    vp = f"{DER}/truth_fits/v1/vertices_{sample}.parquet"
    p["rv"] = np.nan
    if os.path.exists(vp):
        v = pd.read_parquet(vp)
        v = v[v.ok & v.event.isin(events)]
        rv = pd.concat([v[["event", "pid_a", "rv"]].rename(columns={"pid_a": "particle_id"}),
                        v[["event", "pid_b", "rv"]].rename(columns={"pid_b": "particle_id"})])
        p = p.drop(columns="rv").merge(rv, on=["event", "particle_id"], how="left")
    p["event_id"] = np.int64(CODE[sample]) * 100_000_000 + p.event.astype(np.int64)
    p = p[p.tgt].copy()
    p["primary"] = (p.cat == "hnl") & ((p.d0.abs() >= 10.0) | (p.rv >= 100.0))
    return p, chk


class Tables:
    def __init__(self, path):
        z = np.load(path)
        self.units = np.asarray(z["units"], dtype=np.int64)
        self.w = np.asarray(z["w"], dtype=np.float64)

    def sums(self, idx, vals):
        pos = np.searchsorted(self.units, idx)
        assert np.all(pos < len(self.units)) and np.all(self.units[pos] == idx), "event outside table"
        return np.bincount(pos, weights=np.asarray(vals, dtype=float), minlength=len(self.units))


def ratio(parts):
    """parts: list of (Tables, event index array, num, den). Returns point, replicates, k, n."""
    k = n = 0.0
    rk = rn = 0.0
    for t, idx, num, den in parts:
        a, b = t.sums(idx, num), t.sums(idx, den)
        k += a.sum()
        n += b.sum()
        rk = rk + t.w @ a
        rn = rn + t.w @ b
    return k / n, rk / rn, k, n


def three(points, reps):
    m = float(np.mean(points))
    s = float(np.std(points, ddof=1))
    rm = np.mean(np.vstack(reps), axis=0)
    sb = float(np.std(rm, ddof=1))
    return {"per_seed": [float(x) for x in points], "mean": m, "seed_sd": s, "boot_sd": sb,
            "boot_sd_ddof0": float(np.std(rm, ddof=0)), "sigma": float(np.sqrt(sb ** 2 + s ** 2 / 3))}


def hnl_run(a, s):
    if s == 0:
        return f"{DATA}/runs/stage1_final_learned_ts/l1tl/arm{a}_s0"
    return f"{EXTRA}/l1tl/arm{a}_s{s}"


def compute_hnl_test(out):
    ev = {k: split(v[1]) for k, v in HNL_SETS.items()}
    ev["ttbar_test"] = split("ttbar_test")
    truth, chks = [], {}
    for name, (sample, _) in list(HNL_SETS.items()) + [("ttbar_test", ("ttbar", None))]:
        t, c = truth_hnl(sample, ev[name])
        t["set"] = name
        truth.append(t)
        chks[name] = c
    T = pd.concat(truth, ignore_index=True)
    out["hnl_truth_checks"] = chks
    out["hnl_truth_counts"] = {"primary_pooled": int(T.primary.sum()),
                               "primary_per_set": {k: int(T[T.set == k].primary.sum()) for k in HNL_SETS},
                               "hnl_targets": int(((T.cat == "hnl") & (T.set != "ttbar_test")).sum()),
                               "w_targets": int(((T.cat == "w") & (T.set != "ttbar_test")).sum()),
                               "ttbar_prompt": int(((T.set == "ttbar_test") & (T.d0.abs() < 1.0)).sum())}
    tb = {"joint": Tables(f"{DER}/bootstrap/v1/hnl_joint_test.npz"),
          "ext": Tables(f"{DER}/bootstrap/v1/hnl15_100_external.npz"),
          "ttbar": Tables(f"{DER}/bootstrap/v1/ttbar_test.npz")}
    tab_of_code = {2: "joint", 4: "joint", 3: "ext", 1: "ttbar"}

    def parts_for(df, num, den, codes):
        code = (df.event_id.to_numpy() // 100_000_000)
        idx = df.event_id.to_numpy() % 100_000_000
        res = []
        for c in codes:
            m = code == c
            if m.any():
                res.append((tb[tab_of_code[c]], idx[m], np.asarray(num)[m], np.asarray(den)[m]))
        return res

    hnl_mask = T.set != "ttbar_test"
    pops = {"eff_primary": T.primary & hnl_mask, "eff_hnl_targets": (T.cat == "hnl") & hnl_mask,
            "eff_w_muons": (T.cat == "w") & hnl_mask,
            "eff_ttbar_prompt": (T.set == "ttbar_test") & (T.d0.abs() < 1.0)}
    hd = (T.cat == "hnl") & hnl_mask
    d0a = T.d0.abs()
    for lo, hi in zip(D0_EDGES[:-1], D0_EDGES[1:]):
        nm = f"d0_{lo:g}_{'inf' if hi == np.inf else f'{hi:g}'}"
        pops[nm] = hd & (d0a >= lo) & (d0a < hi)
    for lo, hi in zip(RV_EDGES[:-1], RV_EDGES[1:]):
        nm = f"rv_{lo:g}_{'inf' if hi == np.inf else f'{hi:g}'}"
        pops[nm] = hd & (T.rv >= lo) & (T.rv < hi)
    pops["rv_none"] = hd & T.rv.isna()
    for k in HNL_SETS:
        pops[f"primary@{k}"] = T.primary & (T.set == k)

    res, sel = {}, {}
    for a in MODELS:
        per = {q: ([], []) for q in list(pops) + ["fake_rate_hnl"]}
        kn = {q: [] for q in per}
        for s in (0, 1, 2):
            run = hnl_run(a, s)
            d = f"{run}/trackbuild_test"
            cfgs = json.load(open(f"{d}/configs.json"))
            sel[f"{a}_s{s}"] = {"trackbuild_test_configs": cfgs,
                                "llpblind_configs": json.load(open(f"{run}/trackbuild_test_llpblind/configs.json"))}
            pp = pd.read_parquet(f"{d}/per_particle.parquet")
            pe = pd.read_parquet(f"{d}/per_event.parquet")
            assert len(cfgs) == 1 and set(pp.config.unique()) == {0} and set(pe.config.unique()) == {0}
            m = T[["event_id", "particle_id"]].merge(pp[["event_id", "particle_id", "dm_matched"]],
                                                     on=["event_id", "particle_id"], how="left")
            miss = int(m.dm_matched.isna().sum())
            sel[f"{a}_s{s}"]["targets_missing_in_stored_table"] = miss
            match = m.dm_matched.fillna(False).astype(bool).to_numpy()
            for q, mask in pops.items():
                mk = mask.to_numpy()
                df = T[mk]
                codes = (1,) if q == "eff_ttbar_prompt" else (2, 3, 4)
                pt, rp, k, n = ratio(parts_for(df, match[mk].astype(float), np.ones(mk.sum()), codes))
                per[q][0].append(pt)
                per[q][1].append(rp)
                kn[q].append((k, n))
            pe = pe[(pe.event_id // 100_000_000).isin([2, 3, 4])]
            pt, rp, k, n = ratio(parts_for(pe, pe.n_fake.to_numpy(float), pe.n_tracks_ge5.to_numpy(float), (2, 3, 4)))
            per["fake_rate_hnl"][0].append(pt)
            per["fake_rate_hnl"][1].append(rp)
            kn["fake_rate_hnl"].append((k, n))
            sel[f"{a}_s{s}"]["n_hnl_events_in_per_event"] = int(len(pe))
        res[a] = {}
        for q in per:
            r = three(per[q][0], per[q][1])
            r["per_seed_boot_sd"] = [float(np.std(x, ddof=1)) for x in per[q][1]]
            r["k_n"] = kn[q]
            r["_reps"] = per[q][1]
            res[a][q] = r
    # paired difference (ttbar+HNL minus HNL), seeds unpaired (independent trainings)
    diffs = {}
    for q in ("eff_primary", "d0_100_300", "d0_300_inf", "eff_hnl_targets", "eff_w_muons", "eff_ttbar_prompt",
              "fake_rate_hnl"):
        A, B = res["iii"][q], res["ii"][q]
        rd = np.mean(np.vstack(A["_reps"]), axis=0) - np.mean(np.vstack(B["_reps"]), axis=0)
        sb = float(np.std(rd, ddof=1))
        ss = float(np.sqrt(A["seed_sd"] ** 2 / 3 + B["seed_sd"] ** 2 / 3))
        diffs[q] = {"value": A["mean"] - B["mean"], "boot": sb, "seeds": ss, "sigma": float(np.hypot(sb, ss))}
    for a in res:
        for q in res[a]:
            res[a][q].pop("_reps")
    out["hnl_test"] = res
    out["hnl_test_paired_iii_minus_ii"] = diffs
    out["hnl_test_selection_used"] = sel


def gen_truth():
    """{sample: (truth frame with event_id, pop masks)}."""
    out = {}
    iso_ev = split("generic_iso_union_test")
    t = pd.concat([pd.read_parquet(f"{DER}/generated_v1/t1/truth_{s}.parquet") for s in
                   ("generic_iso", "generic_iso_x1")], ignore_index=True)
    t = t[t.event.isin(iso_ev)].copy()
    tgt = (t.pt > 1.0) & (t.eta.abs() < 4.0) & (t.n_hits >= 5)
    gun = (t.vsec >= 1000) & (t["gen"] == 0) & (t["sub"] == 0)
    pg = tgt & gun & ((t.d0.abs() >= 10.0) | (t.r_v >= 100.0))
    t["event_id"] = np.int64(5) * 100_000_000 + t.event.astype(np.int64)
    chk = {"target_mismatch": int((tgt != t.is_target.astype(bool)).sum()),
           "gun_mismatch": int((gun != t.gun.astype(bool)).sum()),
           "pop_mismatch": int((pg != t.in_pop.astype(bool)).sum()), "d0_negative": int((t.d0 < 0).sum()),
           "events": int(len(iso_ev))}
    out["generic_iso"] = (t, {"P_G": pg.to_numpy()}, chk, "generic_iso_union_test", iso_ev)
    for s, code in (("hss15_100", 7), ("hss40_300", 8)):
        ev = split(f"{s}_all")
        t = pd.read_parquet(f"{DER}/generated_v1/t1/truth_{s}.parquet")
        t = t[t.event.isin(ev)].copy()
        tgt = (t.pt > 1.0) & (t.eta.abs() < 4.0) & (t.n_hits >= 5)
        gen = (t["gen"] == 0) & (t["sub"] == 0)
        ph = tgt & gen & ((t.d0.abs() >= 10.0) | (t.r_prod >= 100.0))
        t["event_id"] = np.int64(code) * 100_000_000 + t.event.astype(np.int64)
        chk = {"target_mismatch": int((tgt != t.is_target.astype(bool)).sum()),
               "pop_mismatch": int((ph != t.in_pop.astype(bool)).sum()), "d0_negative": int((t.d0 < 0).sum()),
               "events": int(len(ev))}
        out[s] = (t, {"P_H": ph.to_numpy()}, chk, f"{s}_all", ev)
    for s, (code, stem) in POINTS.items():
        ev = split(f"{stem}_all")
        t = pd.read_parquet(f"{DER}/generated_v1/t2/truth_{s}.parquet")
        t = t[t.event.isin(ev)].copy()
        tgt = (t.pt > 1.0) & (t.eta.abs() < 4.0) & (t.n_hits >= 5)
        ph = tgt & (t.category == "hnl") & ((t.d0_fit.abs() >= 10.0) | (t.rv_fit >= 100.0))
        t["event_id"] = np.int64(code) * 100_000_000 + t.event.astype(np.int64)
        chk = {"target_mismatch": int((tgt != t.is_target.astype(bool)).sum()),
               "pop_mismatch": int((ph != t.in_pop.astype(bool)).sum()), "events": int(len(ev))}
        out[s] = (t, {"P_HNL (fitted |d0|, r_v)": ph.to_numpy()}, chk, f"{s}_all", ev)
    return out


def compute_generated(out):
    G = gen_truth()
    out["gen_truth_checks"] = {s: dict(G[s][2], **{p: int(m.sum()) for p, m in G[s][1].items()}) for s in G}
    setdir = {"generic_iso": "test", "hss15_100": "hss", "hss40_300": "hss"}
    res, prov = {}, {}
    for a, gm in GMODEL.items():
        res[a] = {}
        cache = {}
        for s in (0, 1, 2):
            for sd in ("test", "hss", "hnl"):
                d = f"{EXTRA}/addendum_g/eval/{gm}_s{s}/{sd}"
                cache[(s, sd)] = (pd.read_parquet(f"{d}/per_particle.parquet",
                                                  columns=["event_id", "particle_id", "dm_matched"]),
                                  pd.read_parquet(f"{d}/per_event.parquet"))
                pv = json.load(open(f"{d}/provenance.json"))
                prov[f"{gm}_s{s}_{sd}"] = sorted({(json.dumps(b.get("builder")), str(b["settings"].get("cut_e")),
                                                   str(b["settings"].get("config"))) for b in pv["blocks"]})
        for samp, (t, pops, _, tabname, ev) in G.items():
            tab = Tables(f"{DER}/bootstrap/generated_v1/{tabname}.npz")
            sd = setdir.get(samp, "hnl")
            code = int(t.event_id.iloc[0] // 100_000_000)
            for pname, mask in list(pops.items()) + [("track fake rate", None)]:
                pts, reps, kns, miss = [], [], [], []
                for s in (0, 1, 2):
                    pp, pe = cache[(s, sd)]
                    if mask is None:
                        e = pe[pe.event_id // 100_000_000 == code]
                        assert len(e) == len(ev), (samp, len(e), len(ev))
                        r = ratio([(tab, (e.event_id % 100_000_000).to_numpy(), e.n_fake.to_numpy(float),
                                    e.n_tracks_ge5.to_numpy(float))])
                    else:
                        df = t[mask][["event_id", "particle_id"]]
                        m = df.merge(pp, on=["event_id", "particle_id"], how="left")
                        miss.append(int(m.dm_matched.isna().sum()))
                        r = ratio([(tab, (m.event_id % 100_000_000).to_numpy(),
                                    m.dm_matched.fillna(False).astype(float).to_numpy(), np.ones(len(m)))])
                    pts.append(r[0])
                    reps.append(r[1])
                    kns.append((r[2], r[3]))
                x = three(pts, reps)
                x["k_n"] = kns
                x["missing"] = miss
                res[a][f"{samp}|{pname}"] = x
    out["generated"] = res
    out["generated_provenance_builders"] = prov


def val_selection(out):
    """Builder rule on val (DECISIONS.md 2026-10-04 20:18 UTC) over the mwslmerge configurations of each run."""
    from collections import OrderedDict
    ev = {k: split(k) for k in ("hnl10_100_val", "hnl15_200_val")}
    T = []
    for name, sample in (("hnl10_100_val", "hnl10_100"), ("hnl15_200_val", "hnl15_200")):
        t, _ = truth_hnl(sample, ev[name])
        T.append(t[t.primary][["event_id", "particle_id"]])
    T = pd.concat(T, ignore_index=True)
    hnl_ids = set((np.int64(2) * 100_000_000 + ev["hnl10_100_val"]).tolist()) | \
        set((np.int64(4) * 100_000_000 + ev["hnl15_200_val"]).tolist())
    res = {}
    for a in MODELS:
        for s in (0, 1, 2):
            run = f"{DATA}/runs/stage1_final_learned_ts/l1tl/arm{a}_s{s}"
            rows = OrderedDict()
            for g in ("trackbuild_val_ext", "trackbuild_val_lmerge", "trackbuild_val"):
                cf = json.load(open(f"{run}/{g}/configs.json"))
                idx = [i for i, c in enumerate(cf) if c[0] == "mwslmerge"]
                if not idx:
                    continue
                pp = pd.read_parquet(f"{run}/{g}/per_particle.parquet", filters=[("config", "in", idx)])
                pe = pd.read_parquet(f"{run}/{g}/per_event.parquet", filters=[("config", "in", idx)])
                pe = pe[pe.event_id.isin(hnl_ids)]
                for i in idx:
                    key = f"mwslmerge,{cf[i][1]:g}"
                    if key in rows:
                        continue
                    p = T.merge(pp[pp.config == i][["event_id", "particle_id", "dm_matched"]],
                                on=["event_id", "particle_id"], how="left")
                    e = pe[pe.config == i]
                    rows[key] = {"grid": g, "eff": float(p.dm_matched.fillna(False).mean()), "n": int(len(p)),
                                 "missing": int(p.dm_matched.isna().sum()),
                                 "fr": float(e.n_fake.sum() / e.n_tracks_ge5.sum()), "n_ev": int(len(e))}
            ok = {k: v for k, v in rows.items() if v["fr"] <= FR_STAR}
            emax = max(v["eff"] for v in ok.values())
            n = next(iter(ok.values()))["n"]
            se = np.sqrt(emax * (1 - emax) / n)
            tied = {k: v for k, v in ok.items() if v["eff"] >= emax - se - 1e-12}
            win = sorted(tied.items(), key=lambda kv: (kv[1]["fr"], -kv[1]["eff"]))[0][0]
            res[f"{a}_s{s}"] = {"rows": rows, "e_max": emax, "se": se, "tied": sorted(tied), "chosen": win}
    out["val_selection"] = res


def compute(path):
    out = {"git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    compute_hnl_test(out)
    val_selection(out)
    compute_generated(out)
    json.dump(out, open(path, "w"), indent=1, default=float)


# ----------------------------------------------------------------------------------------------------------- compare
def compare(path, txt):
    R = json.load(open(path))
    L, tally = [], {"agree": 0, "disagree": 0}

    def cmp(label, mine, rep, dec):
        """agree if equal when both are rounded to `dec` decimals (percent points unless stated)."""
        ok = round(mine, dec) == round(rep, dec) or abs(mine - rep) < 0.5 * 10 ** -dec + 1e-9
        tally["agree" if ok else "disagree"] += 1
        L.append(f"  {'OK  ' if ok else 'DIFF'} {label}: mine {mine:.{dec + 2}f} reported {rep:.{dec + 2}f}")

    L.append("Independent recomputation of the R19 / R20 read-outs (ledger R21); scripts/verify/verify_ts_seeds12.py")
    L.append(f"compute: git {R['git']}, Slurm job {R['slurm_job']}; no import of hnl.*; stored tables only")
    L.append("")
    L.append("== Truth re-derivation (HNL test sets) against the stored derived tables")
    for k, v in R["hnl_truth_checks"].items():
        L.append(f"  {k}: {v}")
    L.append(f"  counts: {R['hnl_truth_counts']}")

    # --- 1. bootstrap_ts_test_3seed_v1.csv (pooled test rows)
    L.append("")
    L.append("== Pooled HNL test, results/bootstrap_ts_test_3seed_v1.csv (test_release_card rows); percent, 2 decimals")
    rows = [r for r in csv.DictReader(open(f"{REPO}/results/bootstrap_ts_test_3seed_v1.csv"))
            if r["scope"] == "test" and r["table"] == "test_release_card"]
    for a, lab in MODELS.items():
        for r in rows:
            if not r["model"].startswith(lab + ":"):
                continue
            q = r["quantity"]
            m = R["hnl_test"][a].get(q)
            if m is None:
                L.append(f"  ---- {lab} {q}: not recomputed")
                continue
            for i, ps in enumerate(r["per_seed"].split(";")):
                cmp(f"{lab} {q} s{i}", 100 * m["per_seed"][i], 100 * float(ps), 2)
            cmp(f"{lab} {q} mean", 100 * m["mean"], 100 * float(r["value"]), 2)
            cmp(f"{lab} {q} seed sd", 100 * m["seed_sd"], 100 * float(r["seed_sd"]), 2)
            cmp(f"{lab} {q} boot sd (Eq. 2)", 100 * m["boot_sd"], 100 * float(r["boot_sd"]), 2)
            cmp(f"{lab} {q} sigma (Eq. 2)", 100 * m["sigma"], 100 * float(r["combined"]), 2)
            kp = sum(k for k, _ in m["k_n"])
            np_ = sum(n for _, n in m["k_n"])
            ok = int(round(kp)) == int(float(r["k_pooled"])) and int(round(np_)) == int(float(r["n_pooled"]))
            tally["agree" if ok else "disagree"] += 1
            L.append(f"  {'OK  ' if ok else 'DIFF'} {lab} {q} k/n pooled: mine {kp:.0f}/{np_:.0f} reported "
                     f"{r['k_pooled']}/{r['n_pooled']}")
    # --- 2. training_sample_test_seeds12_v1.csv per sample primary
    L.append("")
    L.append("== Primary per test set, results/training_sample_test_seeds12_v1.csv (per_sample block); percent")
    rows = [r for r in csv.DictReader(open(f"{REPO}/results/training_sample_test_seeds12_v1.csv"))
            if r["block"] == "per_sample" and r["quantity"] == "eff_primary" and r["parent"] in MODELS.values()]
    inv = {v: k for k, v in MODELS.items()}
    for r in rows:
        a = inv[r["parent"]]
        if r["set"] not in HNL_SETS:
            continue
        m = R["hnl_test"][a][f"primary@{r['set']}"]
        if r["seed"] in ("0", "1", "2"):
            s = int(r["seed"])
            cmp(f"{r['parent']} {r['set']} s{s} value", 100 * m["per_seed"][s], 100 * float(r["value"]), 2)
            cmp(f"{r['parent']} {r['set']} s{s} boot sd", 100 * m["per_seed_boot_sd"][s], 100 * float(r["sigma"]), 2)
            k, n = m["k_n"][s]
            ok = (int(k), int(n)) == (int(float(r["k"])), int(float(r["n"])))
            tally["agree" if ok else "disagree"] += 1
            L.append(f"  {'OK  ' if ok else 'DIFF'} {r['parent']} {r['set']} s{s} k/n: mine {k:.0f}/{n:.0f} "
                     f"reported {r['k']}/{r['n']}")
        else:
            cmp(f"{r['parent']} {r['set']} 3-seed mean", 100 * m["mean"], 100 * float(r["value"]), 2)
            cmp(f"{r['parent']} {r['set']} 3-seed sigma", 100 * m["sigma"], 100 * float(r["sigma"]), 2)
            cmp(f"{r['parent']} {r['set']} 3-seed seed sd", 100 * m["seed_sd"], 100 * float(r["seed_sd"]), 2)
    # --- 3. paired
    L.append("")
    L.append("== Paired difference Canto-L (ttbar+HNL) minus Canto-L (HNL), results/training_sample_paired_test_3seed_v1"
             ".csv; points, 2 decimals")
    qmap = {"primary": "eff_primary", "hnl_d0_100_300": "d0_100_300", "hnl_d0_ge300": "d0_300_inf",
            "hnl_targets": "eff_hnl_targets", "w_muons": "eff_w_muons", "ttbar_prompt": "eff_ttbar_prompt",
            "fake_rate": "fake_rate_hnl"}
    for r in csv.DictReader(open(f"{REPO}/results/training_sample_paired_test_3seed_v1.csv")):
        if r["label"] != "Canto-L, trained on ttbar+HNL minus trained on HNL" or r["quantity"] not in qmap:
            continue
        m = R["hnl_test_paired_iii_minus_ii"][qmap[r["quantity"]]]
        for f, g in (("value", "value"), ("boot", "sigma_boot"), ("seeds", "sigma_seeds"), ("sigma", "sigma")):
            cmp(f"paired {r['quantity']} {f}", 100 * m[f], 100 * float(r[g]), 2)
    # --- 4. generated
    L.append("")
    L.append("== Generated samples, results/addendum_g_canto_hnl_seeds12_v1.csv (main / points rows); percent, 2 decimals")
    L.append(f"  truth checks: {json.dumps(R['gen_truth_checks'])}")
    rows = list(csv.DictReader(open(f"{REPO}/results/addendum_g_canto_hnl_seeds12_v1.csv")))
    want = {"P_G", "P_H", "P_HNL (fitted |d0|, r_v)", "track fake rate"}
    ginv = {v: k for k, v in GMODEL.items()}
    seen = set()
    for r in rows:
        if r["table"] != "main" or r["model"] not in ginv or r["population"] not in want:
            continue
        key = f"{r['sample']}|{r['population']}"
        m = R["generated"][ginv[r["model"]]].get(key)
        if m is None or (r["model"], key) in seen:
            continue
        seen.add((r["model"], key))
        lab = f"{MODELS[ginv[r['model']]]} {key}"
        for i in range(3):
            cmp(f"{lab} s{i}", 100 * m["per_seed"][i], 100 * float(r[f"eff_s{i}"]), 2)
        cmp(f"{lab} mean", 100 * m["mean"], 100 * float(r["mean"]), 2)
        cmp(f"{lab} sigma_boot", 100 * m["boot_sd"], 100 * float(r["sigma_boot"]), 2)
        cmp(f"{lab} s_seeds", 100 * m["seed_sd"], 100 * float(r["s_seeds"]), 2)
        cmp(f"{lab} sigma (Eq. 2)", 100 * m["sigma"], 100 * float(r["sigma"]), 2)
        if r["population"] != "track fake rate" and r["n"]:
            nn = m["k_n"][0][1]
            ok = int(nn) == int(float(r["n"]))
            tally["agree" if ok else "disagree"] += 1
            L.append(f"  {'OK  ' if ok else 'DIFF'} {lab} n: mine {nn:.0f} reported {float(r['n']):.0f}"
                     f"; missing in stored tables {m['missing']}")
    # --- 5. selection
    L.append("")
    L.append("== Selection: committed results/training_sample_test_selection_seeds12_v1.json (1c8cbcf) vs stored test "
             "outputs vs builder rule re-applied on val")
    S = json.loads(subprocess.run(["git", "-C", REPO, "show", "1c8cbcf:results/training_sample_test_selection_seeds12"
                                   "_v1.json"], capture_output=True, text=True).stdout)
    S0 = json.load(open(f"{REPO}/results/training_sample_test_selection_v1.json"))
    pipe = {"ii": "l1tl_ii", "iii": "l1tl_iii"}
    for a in MODELS:
        for s in (0, 1, 2):
            src = S if s else S0
            e = src["pipelines"][pipe[a]]["seeds"][str(s)]
            used = R["hnl_test_selection_used"][f"{a}_s{s}"]
            tcfg = ",".join(f"{x:g}" if isinstance(x, float) else str(x) for x in used["trackbuild_test_configs"][0])
            lcfg = ",".join(f"{x:g}" if isinstance(x, float) else str(x) for x in used["llpblind_configs"][0])
            v = R["val_selection"][f"{a}_s{s}"]
            ok1 = tcfg == e["config"]
            ok2 = lcfg == e["llp_blind_config"]
            ok3 = v["chosen"] == e["config"]
            for ok in (ok1, ok2, ok3):
                tally["agree" if ok else "disagree"] += 1
            L.append(f"  {MODELS[a]} s{s}: file FR* {e['config']} / LLP-blind {e['llp_blind_config']} / cut_e "
                     f"{e['cut_e']}; stored test FR* {tcfg} [{'OK' if ok1 else 'DIFF'}], stored LLP-blind {lcfg} "
                     f"[{'OK' if ok2 else 'DIFF'}]; builder rule on val -> {v['chosen']} [{'OK' if ok3 else 'DIFF'}] "
                     f"(e_max {100 * v['e_max']:.2f} %, SE {100 * v['se']:.2f}, tied {v['tied']}); "
                     f"targets missing in stored test table: {used['targets_missing_in_stored_table']}")
            rows_ = ", ".join(f"{k} {100 * x['eff']:.2f}/{100 * x['fr']:.2f}" for k, x in v["rows"].items())
            L.append(f"      val mwslmerge eff/fake (%): {rows_}")
    L.append("  generated-sample eval provenance (builder, cut_e, config) per run and set:")
    for k, v in R["generated_provenance_builders"].items():
        L.append(f"    {k}: {v}")
    L.append("")
    L.append(f"TOTAL: {tally['agree']} agree, {tally['disagree']} disagree")
    L.insert(2, f"SUMMARY: {tally['agree']} agree, {tally['disagree']} disagree")
    open(txt, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        compute(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
