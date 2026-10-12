"""Value-level checks of the throughput-phase tables (hnl.trackml.throughput): TRAINING events only (dev included).

Test discipline (plan section 2.3 and the task rules of 2026-10-05): no value of a val or test event is read here;
`events_or_refuse` refuses them and every table read goes through it. Val and test events get only the structural
checks of `python -m hnl.trackml.throughput verify` (schema, row counts, the converter's pass/fail checks). The
accuracy-phase comparison reads train_1 training events only (never its val or test events, i.e. never T2).

  raw       csv members fetched by HTTP range requests at the manifest offsets (md5 checked) against the stored
            tables, for selected dev and training events: hnl.trackml.check_convert.check_event unchanged (every value
            and hit-particle link, ranks, true edges against the feasibility construction, weights, truth table of
            calibration events), plus particle_type against the ptype side table; the field sign from the truth
            momenta (check_convert.sense_from_truth)
  all-train every stored training event: ranks 0 .. n-1 per particle, weights stored exactly for the calibration
            events, NaN otherwise; truth rows = non-noise hits of calibration events; ptype aligned with particles;
            per-event counts and distributions (hits, noise, particles, targets, displaced targets by category,
            hard-scatter vertex z, primary d0, score weights of secondaries, PDG composition of the mass-hypothesis
            categories), against the plan's numbers for events 21000-21004 and against the accuracy-phase train_1
            training events

    python -m hnl.trackml.check_tp [--root DIR] [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.trackml import check_convert as CC
from hnl.trackml import convert as C
from hnl.trackml import throughput as T

PLAN_21000_21004 = {"hits": 114.5e3, "noise_hits": 17.7e3, "particles_ge1": 9.8e3, "targets": 1169}
CHUNK = 50


def events_or_refuse(events, split: dict) -> list[int]:
    ev = sorted(int(e) for e in events)
    bad = [e for e in ev if e in split["val"] or e in split["test"]]
    if bad:
        raise SystemExit(f"refusing val / test events {bad[:10]} (value checks are for training events only)")
    if not set(ev) <= split["train"]:
        raise SystemExit("refusing events outside the training set")
    return ev


def stored_train(root: str, split: dict) -> dict[int, str]:
    idx = C.event_index(root, "hits")
    return {e: a for e, a in sorted(idx.items()) if e in split["train"]}


def pick_raw_events(root: str, split: dict) -> list[int]:
    """dev 5 smallest, the first calib event outside dev, the first training event outside calib, one in the next
    shard of part01 if any, and the first and last stored training events of training_part02."""
    idx = stored_train(root, split)
    tr = sorted(idx)
    dev = sorted(e for e in tr if e in split["dev"])[:5]
    cal = [e for e in tr if e in split["calib"] and e not in split["dev"]][:1]
    plain = [e for e in tr if e not in split["calib"]]
    p1 = [e for e in plain if idx[e] == "training_part01"]
    p2 = [e for e in tr if idx[e] == "training_part02"]
    out = dev + cal + p1[:1]
    later = [e for e in p1 if e // 500 != p1[0] // 500] if p1 else []
    out += later[:1] + (p2[:1] + p2[-1:] if p2 else [])
    return sorted(set(out))


def check_raw(root: str, split: dict, events: list[int]) -> dict:
    events = events_or_refuse(events, split)
    idx = C.event_index(root, "hits")
    srcs: dict[str, T.TarMembers] = {}
    res = {"events": events, "per_event": []}
    for ev in events:
        part = idx[ev]
        zf = srcs.setdefault(part, T.TarMembers(part))
        names = zf.names[ev]
        r = CC.check_event(zf, names, ev, root, ev in split["calib"] or ev in split["val"] or ev in split["test"],
                           ev in split["calib"])
        p_raw = pd.read_csv(zf.open(names["particles"]))
        pt = C.read_table("ptype", [ev], root=root)
        ok = (len(pt) == len(p_raw) and np.array_equal(pt.particle_id.to_numpy(), p_raw.particle_id.to_numpy())
              and np.array_equal(pt.particle_type.to_numpy().astype(np.int64), p_raw.particle_type.to_numpy()))
        if not ok:
            r["problems"].append("ptype != csv particle_type")
        r["archive"] = part
        res["per_event"].append(r)
        print(json.dumps(r, default=str), flush=True)
    first = events[0]
    sense, frac = CC.sense_from_truth(srcs[idx[first]], srcs[idx[first]].names[first])
    res["field_sense"] = {"event": first, "sense_pos_from_truth": sense, "fraction_ccw_positive": round(frac, 4),
                          "converter_SENSE_POS": C.SENSE_POS, "agree": sense == C.SENSE_POS}
    res["range_requests"] = {p: {"requests": z.nreq, "bytes": z.nbytes} for p, z in srcs.items()}
    print(json.dumps(res["field_sense"]), flush=True)
    return res


def _per_event(h: pd.DataFrame, p: pd.DataFrame, calib: set, pt: pd.DataFrame | None, problems: list) -> pd.DataFrame:
    """Per-event counts of a chunk of training events, and consistency checks appended to problems."""
    rows = []
    for e, he in h.groupby("event", sort=True):
        pe = p[p.event == e]
        nz = he.pid.to_numpy() != 0
        # ranks 0 .. n-1 per particle
        s = he.loc[nz, ["pid", "rank"]].sort_values(["pid", "rank"])
        pos = s.groupby("pid").cumcount().to_numpy()
        if not np.array_equal(s["rank"].to_numpy().astype(np.int64), pos):
            problems.append(f"event {e}: ranks are not 0..n-1")
        if not (he["rank"].to_numpy()[~nz] == C.NOISE_RANK).all():
            problems.append(f"event {e}: noise rank")
        cnt = pd.Series(he.pid.to_numpy()[nz]).value_counts().reindex(pe.particle_id.to_numpy()).fillna(0)
        if not np.array_equal(cnt.to_numpy().astype(np.int64), pe.n_hits.to_numpy().astype(np.int64)):
            problems.append(f"event {e}: n_hits != stored hit count")
        w = he.weight.to_numpy()
        if e in calib:
            if np.isnan(w).any():
                problems.append(f"event {e}: calibration event without weights")
        elif not np.isnan(w).all():
            problems.append(f"event {e}: weights stored for a non-weight training event")
        if pt is not None:
            pte = pt[pt.event == e]
            if not np.array_equal(pte.particle_id.to_numpy(), pe.particle_id.to_numpy()):
                problems.append(f"event {e}: ptype rows not aligned with particles")
        p5 = pe[pe.n_hits >= 5]
        hs = pe[pe.vprim == 1]
        prim_t = pe[pe.is_target & (pe.category == "primary")]
        row = {"event": int(e), "hits": len(he), "noise_hits": int((~nz).sum()), "particle_rows": len(pe),
               "particles_ge1": int((pe.n_hits >= 1).sum()), "particles_ge5": len(p5),
               "targets": int(pe.is_target.sum()), "prompt_targets": int(pe.is_prompt_target.sum()),
               "displaced_targets": int(pe.is_displaced_target.sum()),
               "far_displaced_targets": int(pe.is_far_displaced_target.sum()),
               "soft_displaced": int(pe.is_soft_displaced.sum()),
               "hard_scatter_targets": int((pe.is_target & pe.is_hard_scatter).sum()),
               "hs_vz_median": float(hs.vz.median()) if len(hs) else np.nan,
               "primary_target_absd0_median": float(prim_t.d0.abs().median()) if len(prim_t) else np.nan,
               "max_hits_per_particle": int(pe.n_hits.max()) if len(pe) else 0}
        for c in C.CATEGORIES:
            row[f"ge5_absd0_ge10_{c}"] = int(((p5.d0.abs() >= 10) & (p5.category == c)).sum())
        if e in calib:
            pid_rv = pe.set_index("particle_id").rv
            rv = pid_rv.reindex(he.pid.to_numpy()[nz]).to_numpy()
            wz = w[nz]
            sec = rv > 1.0
            row["weight_sum"] = float(np.nansum(w))
            row["noise_weight_max"] = float(np.nanmax(w[~nz])) if (~nz).any() else 0.0
            row["frac_secondary_hits_weighted"] = float((wz[sec] > 0).mean()) if sec.any() else np.nan
            row["frac_primary_hits_weighted"] = float((wz[~sec] > 0).mean()) if (~sec).any() else np.nan
            nh = pe.set_index("particle_id").n_hits.reindex(he.pid.to_numpy()[nz]).to_numpy()
            row["frac_weighted_hits_lt4"] = float((wz[nh < 4] > 0).mean()) if (nh < 4).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def check_all_train(root: str, split: dict) -> dict:
    """Every stored training event of the throughput tables (chunks of CHUNK events)."""
    idx = stored_train(root, split)
    events = events_or_refuse(list(idx), split)
    problems: list[str] = []
    frames, cat_pdg = [], []
    hcols = ["event", "pid", "rank", "weight", "volume"]
    for k in range(0, len(events), CHUNK):
        part = events_or_refuse(events[k:k + CHUNK], split)
        h = C.read_table("hits", part, hcols, root=root)
        p = C.read_table("particles", part, root=root)
        pt = C.read_table("ptype", part, root=root)
        frames.append(_per_event(h, p, split["calib"], pt, problems))
        m = p.merge(pt, on=["event", "particle_id"], how="left")
        m = m[(m.n_hits >= 5) & (m.category != "primary")]
        cat_pdg.append(m.groupby(["category", "particle_type"]).size().rename("n").reset_index())
        cal = [e for e in part if e in split["calib"]]
        if cal:
            tt = C.read_table("truth", cal, root=root)
            hh = h[h.event.isin(cal)]
            for e in cal:
                rows_t = tt.row.to_numpy()[tt.event.to_numpy() == e]
                pid = hh.pid.to_numpy()[hh.event.to_numpy() == e]
                if not np.array_equal(rows_t, np.flatnonzero(pid != 0)):
                    problems.append(f"event {e}: truth rows != non-noise hits")
        print(f"[all-train] {min(k + CHUNK, len(events))}/{len(events)} events", flush=True)
    ev = pd.concat(frames, ignore_index=True)
    pdg = pd.concat(cat_pdg).groupby(["category", "particle_type"]).n.sum().reset_index()
    comp = {}
    for c, g in pdg.groupby("category"):
        g = g.assign(apdg=g.particle_type.abs()).groupby("apdg").n.sum().sort_values(ascending=False)
        comp[c] = {"n": int(g.sum()), "top_abs_pdg": {str(int(a)): round(float(v / g.sum()), 4)
                                                      for a, v in g.head(4).items()}}
    first5 = ev[ev.event.isin(sorted(ev.event)[:5])]
    summ = ev.drop(columns=["event"]).describe(percentiles=[0.5]).T[["count", "mean", "std", "min", "50%", "max"]]
    return {"n_events": int(len(ev)), "problems": problems[:50], "n_problems": len(problems),
            "per_event_summary": summ.round(4).to_dict(orient="index"),
            "first5_events": sorted(int(e) for e in first5.event),
            "first5_means": first5[["hits", "noise_hits", "particles_ge1", "targets"]].mean().round(1).to_dict(),
            "plan_21000_21004": PLAN_21000_21004,
            "hs_vz_std_mm": round(float(ev.hs_vz_median.std()), 2),
            "category_pdg_composition_ge5_secondaries": comp}


def accuracy_train_reference(split_v1: str = C.SPLIT, max_events: int = 300) -> dict:
    """The same per-event counts on accuracy-phase train_1 TRAINING events (derived/v1; never val or test, i.e. never
    T2): the first max_events training ids."""
    with open(split_v1) as fh:
        s = json.load(fh)
    held = set(s["val"]) | set(s["test"])
    tr = sorted(e for e in s["train_train1"] if e not in held)[:max_events]
    calib = C.load_split(split_v1)["calib"]
    frames, problems = [], []
    for k in range(0, len(tr), CHUNK):
        part = tr[k:k + CHUNK]
        assert not set(part) & held
        h = C.read_table("hits", part, ["event", "pid", "rank", "weight", "volume"])
        p = C.read_table("particles", part)
        frames.append(_per_event(h, p, calib, None, problems))
    ev = pd.concat(frames, ignore_index=True)
    summ = ev.drop(columns=["event"]).describe(percentiles=[0.5]).T[["count", "mean", "std", "min", "50%", "max"]]
    return {"n_events": int(len(ev)), "events": [tr[0], tr[-1]], "n_problems": len(problems),
            "per_event_summary": summ.round(4).to_dict(orient="index"),
            "hs_vz_std_mm": round(float(ev.hs_vz_median.std()), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=T.DERIVED_TP)
    ap.add_argument("--split", default=T.SPLIT_TP)
    ap.add_argument("--events", default=None, help="raw-check events (default: pick_raw_events)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    split = T.load_tp_split(a.split)
    events = [int(x) for x in a.events.split(",")] if a.events else pick_raw_events(a.root, split)
    res = {"raw": check_raw(a.root, split, events)}
    res["all_train"] = check_all_train(a.root, split)
    print(json.dumps({k: v for k, v in res["all_train"].items() if k != "per_event_summary"}, indent=1, default=str),
          flush=True)
    print(pd.DataFrame(res["all_train"]["per_event_summary"]).T.to_string(), flush=True)
    res["accuracy_train_reference"] = accuracy_train_reference()
    print(pd.DataFrame(res["accuracy_train_reference"]["per_event_summary"]).T.to_string(), flush=True)
    print(f"accuracy-phase hard-scatter vz std {res['accuracy_train_reference']['hs_vz_std_mm']} mm; "
          f"throughput {res['all_train']['hs_vz_std_mm']} mm", flush=True)
    res["disk_bytes"] = CC.disk_usage(a.root)
    n_prob = (sum(len(r["problems"]) for r in res["raw"]["per_event"]) + res["all_train"]["n_problems"]
              + (0 if res["raw"]["field_sense"]["agree"] else 1))
    res["n_problems"] = n_prob
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    print(f"[check_tp] raw events {len(events)}, training events {res['all_train']['n_events']}, {n_prob} problems",
          flush=True)
    if n_prob:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
