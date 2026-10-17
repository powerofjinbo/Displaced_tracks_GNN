"""C1 confirmation on the H-val gun hits (docs/preregistration_pileup.md section 9.2), truth only: no model is run and
no model output is read. Reference: the C1 calibration hits (hnl.overlay.c1.trackml_hits of the first 40 calibration
events, hnl.overlay.c1.calib_events); test: the stored gun hits of the set with their true point and direction from
the gun-only CSV truth member (hnl.overlay.truth_checks.gun_hit_features, column is_pp_target).

Part 1, matched bins: hnl.overlay.c1.compare(ref, gun, matched=True) at abf1b0e: the C1 rules in the bins with at least
  200 hits on each side (common support). It fails if a counted bin fails (span0 and span1 10 %, pixel charge x cos
  5 %, MAD(du) and pixel MAD(dv) 20 %) or if the mean u bias over the counted bins of a volume differs by more than
  1 um. The effective-thickness fits of compare() are reported and decide nothing here (the draft's part 1 names the
  rules "in the bins that count"). With no counted bin, part 1 is "not informative": reported, neither a pass nor a
  fail, and it does not block; with one or two counted bins it is reported as "few bins".
Part 2, pooled per volume: hnl.overlay.truth_checks.at_test_incidence at 2e48c42: per volume and C1 quantity, the
  reference reweighted to the gun's joint distribution of both incidence angles (cells of the C1 bins of the binning
  variable times the bins of the other angle), the gun statistic on the common cells. A (volume, quantity) counts when
  at least MIN_HITS = 200 gun hits lie on its common support. It then passes when the relative difference lies within
  the C1 tolerance (span0, span1 10 %, charge x cos 5 %, MAD(du), MAD(dv) 20 %); the u bias fails only when
  |diff| > max(1 um, 2 sigma_diff), sigma_diff = sqrt(sigma_gun^2 + sigma_ref^2) the standard error of the difference
  of the two means: sigma_gun the bootstrap standard deviation of the gun mean that at_test_incidence reports (B = 200,
  default_rng(1)), sigma_ref the standard error of the reweighted reference mean with its weights held fixed,
  sqrt(sum w^2 (x - mean)^2) / sum w (`ref_mean_se`). At the 200-hit floor the standard error of the gun's mean u
  bias is about the size of the 1 um tolerance (pilot volume 13: 204 hits, sigma_gun 1.16 um). Every row reports its
  rule, its limit and, for the mean statistics (spans, charge x cos, u bias), sigma_ref and sigma_diff.
Part 3, support: per volume and quantity, the share of gun hits and of P_P-target hits outside the common support
  (C1 does not validate them).
Outcome: "fail" if a counted bin or a volume's mean u bias fails part 1 or a counted (volume, quantity) fails part 2;
  "not informative" if neither part counts anything; "pass" otherwise. Only "fail" blocks (`blocks`). A
  (volume, quantity) below 200 gun hits on the common support is "not informative": it is listed, decides nothing, and
  for the pixel volumes the same comparison runs on the H-test gun hits at H-T1 and is reported beside H1
  (`not_informative_pixel`, the list for that rerun).

    python -m hnl.overlay.c1_confirm run --set hval --work DIR --root ROOT --out-json J [--ref-events 40]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.overlay import c1 as C1
from hnl.overlay import digitise as D
from hnl.overlay import truth_checks as TC
from hnl.overlay import writer as W

MIN_HITS = C1.MIN_HITS                       # 200
BIAS_TOL_UM = C1.BIAS_TOL_MM * 1e3           # 1 um
BIAS_N_SIGMA = 2.0                           # u bias: |diff| <= max(1 um, 2 sigma_diff)
FEW = 3                                      # fewer counted bins than this: "few bins"
REL_TOL = {q: tol for q, (_, tol) in C1.RULES.items()}
NOT_INFORMATIVE = "not informative"


def _outcome(n_counted: int, failed: bool) -> str:
    if failed:
        return "fail"
    return "pass" if n_counted else NOT_INFORMATIVE


def _pass_of(outcome: str):
    return {"pass": True, "fail": False}.get(outcome)


def part1(ref: pd.DataFrame, gun: pd.DataFrame) -> dict:
    """The C1 rules at matched incidence in the bins that count (module docstring)."""
    cmp = C1.compare(ref, gun, matched=True)
    failed = []
    bias = {}
    for v, r in cmp["volumes"].items():
        for q, bins in r["bins"].items():
            failed += [{"volume": int(v), "quantity": q, "lo": b["lo"], "hi": b["hi"], "rel_diff": b["rel_diff"]}
                       for b in bins if b["counted"] and not b["pass"]]
        if "bias_u" in r:
            bias[int(v)] = r["bias_u"]
    bias_failed = [v for v, b in bias.items() if not b["pass"]]
    n = int(cmp["n_bins_counted"])
    outcome = _outcome(n + len(bias), bool(cmp["n_bins_failed"] or bias_failed))
    return {"n_bins_counted": n, "n_bins_failed": int(cmp["n_bins_failed"]),
            "failed_bins": failed, "bias_u": bias, "bias_failed_volumes": bias_failed,
            "counted_per_volume": {int(v): {q: s.get("counted", 0) for q, s in r["summary"].items()}
                                   for v, r in cmp["volumes"].items()},
            "t_eff_reported": {int(v): r["t_eff"] for v, r in cmp["volumes"].items()},
            "outcome": outcome, "few_bins": bool(0 < n < FEW), "pass": _pass_of(outcome)}


def _quantities(volume: int) -> list[str]:
    return [q for q, (_, _, _, pix) in C1.QUANTITIES.items() if not pix or volume in D.PIXEL_VOLUMES]


def support(ref: pd.DataFrame, gun: pd.DataFrame, pp=None) -> pd.DataFrame:
    """Per volume and quantity: gun hits and their share outside the common support (cells of
    hnl.overlay.truth_checks._cells populated by both samples), for all gun hits and for the P_P-target hits (pp: bool
    per gun hit; default the column is_pp_target)."""
    pp = (gun.is_pp_target.to_numpy(bool) if "is_pp_target" in gun else np.zeros(len(gun), bool)) if pp is None \
        else np.asarray(pp, bool)
    rows = []
    for v in C1.VOLUMES:
        rsel, tsel = (ref.volume == v).to_numpy(), (gun.volume == v).to_numpy()
        rv, tv, ppv = ref[rsel], gun[tsel], pp[tsel]
        for q in _quantities(v):
            if len(tv) and len(rv):
                cr, ct = TC._cells(rv, q, v), TC._cells(tv, q, v)
                m = int(max(cr.max(), ct.max())) + 1
                common = (np.bincount(cr, minlength=m) > 0) & (np.bincount(ct, minlength=m) > 0)
                on = common[ct]
            else:
                on = np.zeros(len(tv), bool)
            rows.append({"volume": int(v), "quantity": q, "gun_hits": int(len(tv)), "on_support": int(on.sum()),
                         "share_outside": float(1 - on.mean()) if len(tv) else None,
                         "pp_hits": int(ppv.sum()),
                         "pp_share_outside": float(1 - on[ppv].mean()) if ppv.any() else None})
    return pd.DataFrame(rows)


def ref_mean_se(ref: pd.DataFrame, gun: pd.DataFrame, volume: int, q: str) -> dict | None:
    """The reference mean of a mean-type quantity in one volume, reweighted to the gun's joint incidence distribution
    with the weights of hnl.overlay.truth_checks.at_test_incidence (cells of `_cells`, weight n_gun / n_ref on the
    common cells, 0 elsewhere), and its standard error with the weights held fixed: sqrt(sum w^2 (x - mean)^2) / sum w,
    in the unit of at_test_incidence (um for du and dv). None for the MAD quantities or without a common cell."""
    col, stat, _, _ = C1.QUANTITIES[q]
    if stat != "mean":
        return None
    rv, tv = ref[(ref.volume == volume).to_numpy()], gun[(gun.volume == volume).to_numpy()]
    if not len(rv) or not len(tv):
        return None
    cr, ct = TC._cells(rv, q, volume), TC._cells(tv, q, volume)
    m = int(max(cr.max(), ct.max())) + 1
    nr, nt = np.bincount(cr, minlength=m), np.bincount(ct, minlength=m)
    common = (nr > 0) & (nt > 0)
    w = np.where(common[cr], nt[cr] / np.maximum(nr[cr], 1), 0.0)
    if w.sum() <= 0:
        return None
    x = rv[col].to_numpy(np.float64)
    mean = float(np.average(x, weights=w))
    se = float(np.sqrt(np.sum(w ** 2 * (x - mean) ** 2)) / w.sum())
    scale = 1e3 if col in ("du", "dv") else 1.0
    return {"mean": mean * scale, "se": se * scale, "n_eff": float(w.sum() ** 2 / np.sum(w ** 2))}


def bias_limit(sigma_diff) -> float:
    """The u-bias limit [um]: max(1 um, 2 sigma_diff); 1 um when sigma_diff is not finite."""
    s = float(sigma_diff) if sigma_diff is not None else float("nan")
    return max(BIAS_TOL_UM, BIAS_N_SIGMA * s) if np.isfinite(s) else BIAS_TOL_UM


def part2(ref: pd.DataFrame, gun: pd.DataFrame, n_boot: int = TC.N_BOOT) -> pd.DataFrame:
    """Pooled per volume (module docstring): one row per volume and quantity with the reweighted reference, the gun
    statistic on the common support, the standard errors of the mean statistics, the count rule, the rule and its
    limit, and the outcome."""
    got = {(r["volume"], r["quantity"]): r for r in TC.at_test_incidence(ref, gun, n_boot=n_boot)}
    rows = []
    for v in C1.VOLUMES:
        for q in _quantities(v):
            r = got.get((v, q))
            n = int(r["n_test"]) if r else 0
            se = ref_mean_se(ref, gun, v, q) if r else None
            if se is not None and not np.isclose(se["mean"], r["ref"], rtol=1e-9, atol=1e-9):
                raise RuntimeError(f"volume {v} {q}: reference mean {se['mean']} differs from at_test_incidence "
                                   f"{r['ref']}")
            sig_ref = se["se"] if se else None
            sig_diff = float(np.hypot(r["sigma_test"], sig_ref)) if se else None
            row = {"volume": int(v), "pixel": v in D.PIXEL_VOLUMES, "quantity": q, "n_gun_on_support": n,
                   "coverage": r["coverage"] if r else None, "ref": r["ref"] if r else None,
                   "gun": r["test"] if r else None, "diff": r["diff"] if r else None,
                   "rel_diff": r["rel_diff"] if r else None, "sigma_gun": r["sigma_test"] if r else None,
                   "sigma_ref": sig_ref, "sigma_diff": sig_diff, "n_eff_ref": se["n_eff"] if se else None,
                   "unit": r["unit"] if r else None, "counted": n >= MIN_HITS}
            if q == "bias_u":
                lim = bias_limit(sig_diff)
                row.update(rule=f"|diff| <= max({BIAS_TOL_UM:g} um, {BIAS_N_SIGMA:g} sigma_diff)", limit=lim,
                           tolerance=f"{lim:.3g} um")
                ok = bool(abs(r["diff"]) <= lim) if row["counted"] else None
            else:
                row.update(rule=f"|rel_diff| <= {REL_TOL[q]:g}", limit=REL_TOL[q],
                           tolerance=f"{REL_TOL[q]:g} relative")
                ok = bool(abs(r["rel_diff"]) <= REL_TOL[q]) if row["counted"] else None
            row.update(passed=ok, outcome=NOT_INFORMATIVE if ok is None else ("pass" if ok else "fail"))
            rows.append(row)
    return pd.DataFrame(rows)


def confirm(ref: pd.DataFrame, gun: pd.DataFrame, n_boot: int = TC.N_BOOT) -> dict:
    """The three parts and the outcome (module docstring)."""
    p1 = part1(ref, gun)
    p2 = part2(ref, gun, n_boot)
    sup = support(ref, gun)
    failed2 = p2[p2.outcome == "fail"]
    ni = p2[~p2.counted]
    out2 = _outcome(int(p2.counted.sum()), bool(len(failed2)))
    vols = {}
    for v, g in p2.groupby("volume"):
        vols[int(v)] = {"counted": int(g.counted.sum()), "quantities": int(len(g)),
                        "failed": sorted(g[g.outcome == "fail"].quantity.tolist()),
                        "not_informative": sorted(g[~g.counted].quantity.tolist()),
                        "n_gun_hits": int((gun.volume == v).sum())}
    if "fail" in (p1["outcome"], out2):
        outcome = "fail"
    elif p1["outcome"] == out2 == NOT_INFORMATIVE:
        outcome = NOT_INFORMATIVE
    else:
        outcome = "pass"
    return {"part1_matched_bins": p1, "part2_pooled": p2.to_dict("records"), "part2_outcome": out2,
            "part3_support": sup.to_dict("records"),
            "volumes": vols, "part2_failed": failed2[["volume", "quantity"]].to_dict("records"),
            "not_informative": ni[["volume", "quantity", "n_gun_on_support"]].to_dict("records"),
            "not_informative_pixel": ni[ni.pixel][["volume", "quantity", "n_gun_on_support"]].to_dict("records"),
            "rules": {"min_gun_hits_on_support": MIN_HITS, "rel_tol": REL_TOL, "bias_tol_um": BIAS_TOL_UM,
                      "bias_rule": f"|diff| <= max({BIAS_TOL_UM:g} um, {BIAS_N_SIGMA:g} sigma_diff), sigma_diff = "
                                   "sqrt(sigma_gun^2 + sigma_ref^2)",
                      "not_informative": "a part with nothing counted is neither a pass nor a fail and does not "
                                         "block"},
            "outcome": outcome, "pass": _pass_of(outcome), "blocks": outcome == "fail"}


def gun_context(set_name: str, work: str, root: str) -> dict:
    """What hnl.overlay.truth_checks.gun_hit_features reads (sidecar events, stored rows, gun-only CSV members), without
    the host tables: the comparison needs no host row, so it runs on H-test gun hits at H-T1 as on H-val."""
    with open(os.path.join(root, f"sidecar_{set_name}.json")) as fh:
        events = {int(s["overlay_event"]): s for s in json.load(fh)["events"]}
    return {"set": set_name, "events": events, "stored": {e: W.host_tables(e, root) for e in events},
            "csv": TC.csv_members(os.path.join(work, f"gun_csv_{set_name}.tar"))}


def cmd_run(a) -> dict:
    t0 = time.time()
    ctx = gun_context(a.set, a.work, a.root)
    mods = D.Modules.load()
    gun = TC.gun_hit_features(ctx, mods)
    ev = C1.calib_events(a.ref_events)
    ref = C1.trackml_hits(ev, mods)
    out = {"set": a.set, "root": a.root, "work": a.work, "events": sorted(ctx["events"]),
           "reference": {"events": ev, "n_hits": int(len(ref))}, "n_gun_hits": int(len(gun)),
           "n_pp_target_hits": int(gun.is_pp_target.sum()), **confirm(ref, gun)}
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    p1 = out["part1_matched_bins"]
    print(f"[c1 confirmation] {out['outcome']}: part 1 {p1['outcome']} ({p1['n_bins_counted']} counted bins, "
          f"{p1['n_bins_failed']} failed); part 2 {out['part2_outcome']} (failed {out['part2_failed']}); not "
          f"informative {len(out['not_informative'])} ({out['seconds']} s)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--root", required=True)
    r.add_argument("--out-json", required=True)
    r.add_argument("--ref-events", type=int, default=40)
    cmd_run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
