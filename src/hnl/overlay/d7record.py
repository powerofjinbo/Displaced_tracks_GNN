"""The records of decision D7 of addendum H (docs/preregistration_pileup.md sections 6, 7, 9.3 and 10, step 4).

A decisive run of the H1 code (`python -m hnl.overlay.h1 run --set hval|htest`, slurm/overlay/h1.sbatch) takes four
inputs besides the model outputs from a committed record only (--restriction-record; hnl.overlay.h1.restriction_record):
the restriction of P_P to detector regions (the forward fallback of section 9.3, decision D12), the failed flag gates,
the navigator-loss share of section 7 and whether H is descriptive (section 9.3). This module writes those records.
It reads truth-level check outputs and gate outputs, as the JSON files their modules write, and the meta files of the
stored sets. It reads no stored model output of a gun target, no sealed directory and no output of the H1 code.

Records (one per set, at a fixed path, written once and never over an existing file):
  H-val   results/pileup_hp/h_t05_d7_v1.json  H-T0.5 step 4, committed together with the version record
          configs/trackml/overlay_version_h_v1.json; `bash slurm/overlay/guard_h.sh --launch hval` needs it committed
          and pins it (hnl.overlay.blind.HT05_RECORDS).
  H-test  results/pileup_hp/h_t1_d7_v1.json   H-T1 step 5, one of the H-T1 records; ledger entry R15 names it and the
          H-test launch pins it (hnl.overlay.blind.HT1_RECORDS).

Inputs of the H-val record. Every one is required; the record keeps the path and sha256 of each.
  --truth-checks    hnl.overlay.truth_checks run --set hval    per-set truth checks of H-val            decisive
  --closure-checks  hnl.overlay.closure_checks run            per-set truth checks of the closure set   decisive
  --c1-confirm      hnl.overlay.c1_confirm run --set hval     C1 confirmation                           decisive
  --closure-gate    hnl.overlay.closure_gate run              closure truth gate                        flag
  --v1              hnl.overlay.v1 run, with --closure-1gev   V1 (a), V1 (c) and V1-Delta (a)           decisive
                                                              V1 (b) and V1 (d)                         flag
  --v2              hnl.overlay.blind v2                      V2 (host numbers only, blinded)           flag
  --gun-content     hnl.overlay.gun_content run --set hval    navigator losses of H-val (lost share)
  --passes          the overlay_hash of every version made at H-T0.5, in the order made. Each version is one pass
                    and there are at most three (section 9.3). The first is the version of the addendum's commit
                    (ADDENDUM_OVERLAY_HASH), and the last is the version the gate outputs carry.
Optional: --c1-flag-decision F, the logged decision that carries named failed rows of the C1 confirmation as a flag
(configs/trackml/overlay_h_c1_flag_v1.json; DECISIONS 2026-10-16), a path in the repository (relative to it or
absolute); see the derivation. When it is given, the record keeps its repo path and sha256 too (c1_flag_decision).
Every output must be of its set and hold every event of it, on the production overlay root (`DERIVED_OVERLAY`), and
every overlay_hash it carries must be the last pass, as must the overlay_hash of the meta files of H-val and closure.
V1 must come from the production closure grids (no placeholder) in C-full and C-1GeV, and V2 from the sealed H-val
directory whose manifest sha256 RESULTS.md records at HEAD (decision D4), made from the H-val meta file as it is now.
A version record on disk must carry the same overlay_hash (and the same restriction and descriptive state, where it
records them).

Derivation of the H-val record (`derive`; section 9.3). It is mechanical, and a case it does not settle is refused.
  - A per-set truth check that fails: refused. The set does not go to the model; the cause is found, fixed and logged.
  - restriction: the forward fallback applies when every other decisive gate passes or is not informative, and V1 (a)
    passes in all (|Delta| <= 1.0 point) but fails in one or two of the three regions (|Delta| <= 1.5 points in each).
    P_P then keeps the regions that pass, in the order of hnl.overlay.h1.REGIONS. Otherwise restriction is null.
  - A decisive gate that fails where the fallback does not apply: on the third pass H is descriptive (descriptive true,
    restriction null). Before the third pass the record is refused: section 9.3 allows a change, and a failure traced
    to Fatras, the v15 material or the geometry (rule 3 of the changes) needs a logged decision that this module does
    not take.
  - The C1 confirmation carried as a flag (--c1-flag-decision F). F is committed as is, strict JSON with exactly the
    keys decision, first_overlay_hash (the first pass), flag, part1_bias_u_volumes (volumes) and part2_failed
    ([{volume, quantity}]). When the C1 confirmation fails, on any pass, and every failed row is one F names (part 1: no
    failed bin, and every volume of bias_failed_volumes in part1_bias_u_volumes; part 2: every row of part2_failed in
    F's part2_failed), C1 counts as no failed decisive gate and F's flag is the first flag. Any other failed row: as
    above. A C1 confirmation that passes or is not informative leaves F unused. derivation.c1_flag records the use,
    F's path and sha256 and the rows.
  - flags: the flag gates that fail, in the order closure truth gate, V1 (b), V1 (d), V2, after the flag of F when it
    is used. A C1 confirmation or a closure truth gate that counts nothing is "not informative": it neither passes nor
    fails, and it is no flag.
  - lost_share: the share of lost gun particles among the generator-level P_P-like particles of H-val
    (nav_losses.pp_like_lost_share of hnl.overlay.gun_content).

The H-test record (`htest_record`). Its restriction, flags, descriptive state and passes are those of the committed
H-val record: the restriction is committed before H-T1, and the flag gates are those of the committed version. Its
lost share is the navigator-loss share of H-test. Inputs: the committed H-val record, the committed version record,
the H-test truth checks (every check passes, the version check against the committed version record, the hosts from
derived/tp03_v1) and the H-test navigator losses; both outputs hold every evaluated H-test host
(hnl.overlay.split_h.htest_hosts) and carry the committed overlay_hash, as does the H-test meta file.

Also refused: a record file that exists already, a launch record of the set (it pins the records), for H-test a ledger
entry R15 (it names the H-T1 records), an input under the runs root's sealed directories, an output of the H1 code given
as an input, and an input that is not the JSON output of its module. The command line also refuses to write while
this module, hnl.overlay.h1 or hnl.overlay.blind differs from HEAD.

Schema (both records; JSON):
  record ("addendum H decision D7"), schema (1), set ("hval" or "htest"), step,
  restriction (null or a list of regions of hnl.overlay.h1.REGIONS), flags (list of gate names; first F's flag if used),
  lost_share (a share in [0, 1]), descriptive (bool)                      the four inputs hnl.overlay.h1 reads
  overlay_hash, passes (list of overlay_hash), derivation (how the four were found), gates (per gate: kind, outcome
  and the numbers that decide it), nav_losses, inputs (path and sha256 of each input), metas, version_record,
  hval_record (H-test only), code (module sha256 and git HEAD), written_utc, rules.

    python -m hnl.overlay.d7record hval --passes OH[,OH,...] --truth-checks J --closure-checks J --c1-confirm J \\
        --closure-gate J --v1 J --v2 J --gun-content J [--c1-flag-decision F]
    python -m hnl.overlay.d7record htest --truth-checks J --gun-content J
    python -m hnl.overlay.d7record path --set hval|htest              (prints the fixed record path)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys

from hnl.overlay import blind as B
from hnl.overlay import h1 as H

REPO = B.REPO
DERIVED_OVERLAY = B.DERIVED_OVERLAY          # the production overlay root (= hnl.overlay.writer.DERIVED_OVERLAY)
ADDENDUM_OVERLAY_HASH = "3e23b3630f6b7ec54e0b3872e6ab25909fa4cd4f0f2c810ac1b9a434263df1f4"   # section 2, pass 1
MAX_PASSES = 3
SCHEMA = 1
RECORD_PATHS = {"hval": B.HT05_RECORDS + "d7_v1.json",                # results/pileup_hp/h_t05_d7_v1.json
                "htest": B.HT1_RECORDS + "d7_v1.json"}                # results/pileup_hp/h_t1_d7_v1.json
STEPS = {"hval": "H-T0.5 step 4", "htest": "H-T1 step 5"}
REGIONS = H.REGIONS                                   # central, fwd+, fwd- (= hnl.overlay.closure.REGIONS)
TRUTH_KEYS = ("format", "weights", "containment", "production", "version")             # hnl.overlay.truth_checks
CLOSURE_KEYS = ("population", "ids", "format", "containment", "converter_view", "production", "version")
OUTCOMES = ("pass", "fail", "not informative")
DECISIVE = ("per-set truth checks, H-val", "per-set truth checks, closure", "C1 confirmation", "V1 (a)", "V1 (c)",
            "V1-Delta (a)")
FLAG_GATES = ("closure truth gate", "V1 (b)", "V1 (d)", "V2")
C1_FLAG_KEYS = ("decision", "first_overlay_hash", "flag", "part1_bias_u_volumes", "part2_failed")   # --c1-flag-decision
V1_LABEL = "closure set, frozen Canto-L (TrackML) at WP-A"           # the label of `hnl.overlay.v1 run`
V1_A_ALL, V1_A_REGION = 1.0, 1.5                                     # = hnl.overlay.v1.RULES["a"]
RULES = ("docs/preregistration_pileup.md sections 6, 7, 9.3 and 10 (decision D7): the P_P restriction (forward "
         "fallback, decision D12), the failed flag gates, the navigator-loss share and the descriptive state of the "
         "decisive H1 runs; derived mechanically from the gate outputs of the committed version (hnl.overlay.d7record)")


class Refused(Exception):
    """The record is not written: an input is missing, inconsistent or of another kind, or section 9.3 does not settle
    the case mechanically."""


# ------------------------------------------------------------------------------------------------ inputs
def _real(p: str) -> str:
    return os.path.realpath(p)


def _same(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _real(a) == _real(b)


def load_input(name: str, path: str, runs: str | None = None) -> tuple[dict, dict]:
    """(the JSON output, {path, sha256}). Refused: a missing file, a file that is no JSON object, a path under the
    sealed directories of the runs root (blinding) or an output of the H1 code."""
    if not path:
        raise Refused(f"--{name} is required")
    runs = runs or B.RUNS_H
    if B._under(path, os.path.join(runs, "sealed")):
        raise Refused(f"--{name} {path}: nothing under {os.path.join(runs, 'sealed')} is read (blinding on H-val)")
    if os.path.basename(path).startswith("h1_"):
        raise Refused(f"--{name} {path}: an output of the H1 code is no input of the record")
    if not os.path.isfile(path):
        raise Refused(f"--{name} {path}: no such file")
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        raise Refused(f"--{name} {path}: not JSON ({e})") from e
    if not isinstance(d, dict):
        raise Refused(f"--{name} {path}: not a JSON object")
    return d, {"path": _real(path), "sha256": B.sha256_file(path)}


def _need(d: dict, name: str, *keys):
    """d[k0][k1]...; Refused if absent."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise Refused(f"{name}: no {'.'.join(str(x) for x in keys)} (not the output of its module)")
        cur = cur[k]
    return cur


def _check_set(d: dict, name: str, set_name: str, ids: list[int], root: str, events_listed: bool) -> None:
    if d.get("set") != set_name:
        raise Refused(f"{name}: an output of set {d.get('set')!r}, not {set_name}")
    ev = _need(d, name, "events")
    if events_listed:
        if not isinstance(ev, list) or sorted(int(e) for e in ev) != sorted(ids):
            raise Refused(f"{name}: its events are not every event of {set_name} ({len(ids)} overlay ids)")
    elif ev != len(ids):
        raise Refused(f"{name}: {ev} events, not the {len(ids)} of {set_name}")
    if not _same(d.get("root"), root):
        raise Refused(f"{name}: root {d.get('root')}, not the overlay root {root}")


def _outcome(name: str, value) -> str:
    if value not in OUTCOMES:
        raise Refused(f"{name}: outcome {value!r} is none of {OUTCOMES}")
    return value


def truth_gate(d: dict, name: str, set_name: str, ids: list[int], root: str, keys=TRUTH_KEYS) -> dict:
    """Per-set truth checks (hnl.overlay.truth_checks or hnl.overlay.closure_checks): pass when every check passes."""
    _check_set(d, name, set_name, ids, root, events_listed=True)
    p = _need(d, name, "pass")
    if not isinstance(p, dict) or not set(keys) <= set(p) or not all(isinstance(v, bool) for v in p.values()):
        raise Refused(f"{name}: its pass record does not hold the checks {', '.join(keys)}")
    failed = sorted(k for k, v in p.items() if not v)
    return {"kind": "decisive", "outcome": "fail" if failed else "pass", "failed_checks": failed,
            "overlay_hash": _need(d, name, "version", "overlay_hash")}


def c1_gate(d: dict, set_name: str, ids: list[int], root: str) -> dict:
    name = "C1 confirmation"
    _check_set(d, name, set_name, ids, root, events_listed=True)
    out = _outcome(name, _need(d, name, "outcome"))
    if d.get("pass") is not {"pass": True, "fail": False}.get(out):
        raise Refused(f"{name}: pass {d.get('pass')!r} does not agree with its outcome {out!r}")
    return {"kind": "decisive", "outcome": out, "part1": _need(d, name, "part1_matched_bins", "outcome"),
            "part2": _need(d, name, "part2_outcome"), "part2_failed": d.get("part2_failed")}


def c1_failed_rows(d: dict) -> dict:
    """The failed rows of a C1 confirmation that fails (hnl.overlay.c1_confirm): part 1 failed bins and volumes whose
    mean u bias fails, part 2 failed (volume, quantity) rows. Refused if they do not agree with its outcomes."""
    name = "C1 confirmation"
    bins = _need(d, name, "part1_matched_bins", "failed_bins")
    vols = _need(d, name, "part1_matched_bins", "bias_failed_volumes")
    n_bins = _need(d, name, "part1_matched_bins", "n_bins_failed")
    p2 = _need(d, name, "part2_failed")
    if not all(isinstance(x, list) for x in (bins, vols, p2)) or type(n_bins) is not int or n_bins != len(bins):
        raise Refused(f"{name}: failed_bins, bias_failed_volumes and part2_failed are not lists of the failed rows "
                      f"(n_bins_failed {n_bins!r})")
    p1_fail, p2_fail = bool(bins or vols), bool(p2)
    if ((d["part1_matched_bins"].get("outcome") == "fail") != p1_fail or (d.get("part2_outcome") == "fail") != p2_fail
            or (d.get("outcome") == "fail") != (p1_fail or p2_fail)):
        raise Refused(f"{name}: its failed rows do not agree with its outcomes")
    return {"part1_failed_bins": bins, "part1_bias_u_volumes": vols, "part2_failed": p2}


def _c1_row(r) -> bool:
    return (isinstance(r, dict) and set(r) == {"volume", "quantity"} and type(r["volume"]) is int
            and isinstance(r["quantity"], str) and bool(r["quantity"]))


def _unique_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"a key appears twice: {sorted({k for k in keys if keys.count(k) > 1})}")
    return dict(pairs)


def load_c1_flag_decision(path: str, repo: str, passes: list[str]) -> dict:
    """The logged decision of --c1-flag-decision (module docstring): a file of the repository committed as is, strict
    JSON with exactly C1_FLAG_KEYS, its first_overlay_hash the first pass. Its fields, repo path and sha256."""
    name = f"--c1-flag-decision {path}"
    p = path if os.path.isabs(path) else os.path.join(repo, path)
    rel = os.path.relpath(_real(p), _real(repo))
    if rel == ".." or rel.startswith("../") or not os.path.isfile(p):
        raise Refused(f"{name}: no file of the repository {repo}")
    if not B.git_committed(os.path.join(repo, rel), repo):
        raise Refused(f"{name}: not committed as is (the decision is a committed input)")
    try:
        with open(p) as fh:
            d = json.load(fh, object_pairs_hook=_unique_keys)
    except (OSError, ValueError) as e:
        raise Refused(f"{name}: not strict JSON ({e})") from e
    if not isinstance(d, dict) or set(d) != set(C1_FLAG_KEYS):
        raise Refused(f"{name}: its keys are not exactly {', '.join(C1_FLAG_KEYS)}")
    oh, vols, rows = d["first_overlay_hash"], d["part1_bias_u_volumes"], d["part2_failed"]
    text = lambda x: isinstance(x, str) and bool(x.strip())                     # noqa: E731
    ok = {"decision": text(d["decision"]), "flag": text(d["flag"]),
          "first_overlay_hash": isinstance(oh, str) and bool(B.HEX64.match(oh)),
          "part1_bias_u_volumes": isinstance(vols, list) and all(type(v) is int for v in vols)
          and len(set(vols)) == len(vols),
          "part2_failed": isinstance(rows, list) and all(_c1_row(r) for r in rows)
          and len({(r["volume"], r["quantity"]) for r in rows}) == len(rows)}
    bad = [k for k in C1_FLAG_KEYS if not ok[k]]
    if bad:
        raise Refused(f"{name}: {', '.join(bad)} not of the schema (decision and flag text, first_overlay_hash 64 hex, "
                      "distinct int volumes, distinct {volume: int, quantity: str} rows)")
    if not vols and not rows:
        raise Refused(f"{name}: names no row")
    if d["flag"] in DECISIVE + FLAG_GATES:
        raise Refused(f"{name}: its flag {d['flag']!r} is the name of a gate")
    if d["first_overlay_hash"] != passes[0]:
        raise Refused(f"{name}: first_overlay_hash {d['first_overlay_hash'][:12]}... is not the first pass "
                      f"{passes[0][:12]}...")
    return {"path": rel, "sha256": B.sha256_file(p), **d}


def c1_flag_use(outcome: str, dec: dict, rows: dict | None) -> dict:
    """Whether the decision dec carries the C1 confirmation of this outcome as a flag (module docstring); rows: its
    failed rows (`c1_failed_rows`) when it fails."""
    out = {"decision": {k: dec[k] for k in ("path", "sha256", "decision", "first_overlay_hash")}, "flag": dec["flag"]}
    if outcome != "fail":
        return {"used": False, **out, "how": f"the C1 confirmation outcome is {outcome!r}: the decision is not used"}
    vols, named = set(dec["part1_bias_u_volumes"]), {(r["volume"], r["quantity"]) for r in dec["part2_failed"]}
    unc = [{"part": 1, "failed_bin": b} for b in rows["part1_failed_bins"]]
    unc += [{"part": 1, "bias_u_volume": v} for v in rows["part1_bias_u_volumes"]
            if type(v) is not int or v not in vols]
    unc += [{"part": 2, "row": r} for r in rows["part2_failed"]
            if not _c1_row(r) or (r["volume"], r["quantity"]) not in named]
    if unc:
        return {"used": False, **out, "failed_rows": rows, "uncovered_rows": unc,
                "how": "a failed C1 row is not one the decision names: section 9.3 as written"}
    return {"used": True, **out, "covered_rows": rows,
            "how": "every failed C1 row is one the decision names: the C1 confirmation is carried as a flag, not as "
                   "a failed decisive gate"}


def closure_gate(d: dict, ids: list[int], root: str) -> dict:
    name = "closure truth gate"
    _check_set(d, name, "closure", ids, root, events_listed=False)
    out = _outcome(name, _need(d, name, "outcome"))
    return {"kind": "flag", "outcome": out, "counted": _need(d, name, "counted"), "failed_cells": d.get("failed")}


def _rule_pass(d: dict, name: str, *keys, kind: str) -> bool:
    rule = _need(d, name, *keys, "rule")
    if rule.get("kind") != kind or not isinstance(rule.get("pass"), bool):
        raise Refused(f"{name}: {'.'.join(keys)} has no {kind} rule with a pass")
    return rule["pass"]


def v1_gates(d: dict, ids: list[int], root: str, dry: bool = False) -> dict:
    """V1 (a)-(d) and V1-Delta (a) of `hnl.overlay.v1 run` on the closure grids in C-full and C-1GeV."""
    name = "V1"
    if not dry and d.get("label") != V1_LABEL:
        raise Refused(f"{name}: label {d.get('label')!r}, not the closure run's ({V1_LABEL!r})")
    if _need(d, name, "events") != len(ids):
        raise Refused(f"{name}: {d.get('events')} events, not the {len(ids)} of closure")
    if not _same(d.get("root"), root):
        raise Refused(f"{name}: root {d.get('root')}, not the overlay root {root}")
    for c in ("full", "1gev"):
        g = _need(d, name, "grids", c)
        if g.get("set") != "closure" or g.get("condition") != c or g.get("n_events") != len(ids):
            raise Refused(f"{name}: the {c} grid is not the closure grid of every event in {c}")
        if not dry and _need(g, f"{name} grid {c}", "scores_provenance", "placeholder") is not False:
            raise Refused(f"{name}: the {c} grid holds placeholder scores")
    _need(d, name, "v1_delta")
    a = _need(d, name, "v1", "a")
    rule = _need(a, "V1 (a)", "rule")
    a_pass = _rule_pass(d, name, "v1", "a", kind="decisive")
    lim = rule.get("limit")
    delta = a.get("delta")
    finite = isinstance(delta, (int, float)) and math.isfinite(delta)
    all_pass = bool(lim is not None and finite and a.get("pairs") and abs(delta) <= lim)
    regions = rule.get("regions_pass")
    if lim is not None and (not isinstance(regions, dict) or set(regions) != set(REGIONS)):
        raise Refused(f"{name}: V1 (a) has no pass for each of the regions {REGIONS}")
    if lim is not None and (lim != V1_A_ALL or rule.get("limit_region") != V1_A_REGION):
        raise Refused(f"{name}: V1 (a) limits {lim} / {rule.get('limit_region')}, not {V1_A_ALL} / {V1_A_REGION}")
    if lim is not None and a_pass != (all_pass and all(regions.values())):
        raise Refused(f"{name}: V1 (a) pass {a_pass} does not agree with its parts")
    c_pass = _rule_pass(d, name, "v1", "c", kind="decisive")
    vd_pass = _rule_pass(d, name, "v1_delta", "a", kind="decisive")
    b_pass = _rule_pass(d, name, "v1", "b", kind="flag")
    d_pass = _rule_pass(d, name, "v1", "d", kind="flag")
    if d.get("decisive_pass") is not (a_pass and c_pass and vd_pass):
        raise Refused(f"{name}: decisive_pass does not agree with V1 (a), V1 (c) and V1-Delta (a)")
    reg = {r: {"pairs": (a.get("regions") or {}).get(r, {}).get("pairs"),
               "delta": (a.get("regions") or {}).get(r, {}).get("delta"), "pass": (regions or {}).get(r)}
           for r in REGIONS}
    oc = lambda ok: "pass" if ok else "fail"                                       # noqa: E731
    return {"V1 (a)": {"kind": "decisive", "outcome": oc(a_pass), "pairs": a.get("pairs"), "delta": delta,
                       "limit": lim, "passes_in_all": all_pass, "limit_region": rule.get("limit_region"),
                       "regions": reg},
            "V1 (c)": {"kind": "decisive", "outcome": oc(c_pass), **_brief(d["v1"]["c"])},
            "V1-Delta (a)": {"kind": "decisive", "outcome": oc(vd_pass),
                             **{k: d["v1_delta"]["a"].get(k) for k in ("pairs", "difference", "sigma_paired")},
                             "limit": d["v1_delta"]["a"]["rule"].get("limit")},
            "V1 (b)": {"kind": "flag", "outcome": oc(b_pass), **_brief(d["v1"]["b"])},
            "V1 (d)": {"kind": "flag", "outcome": oc(d_pass), **_brief(d["v1"]["d"])}}


def _brief(p: dict) -> dict:
    return {"pairs": p.get("pairs"), "delta": p.get("delta"), "sigma_paired": p.get("sigma_paired"),
            "limit": (p.get("rule") or {}).get("limit")}


def v2_gate(d: dict, ids: list[int], expect_manifest: str | None, meta_sha: str | None, dry: bool = False) -> dict:
    """V2 of `hnl.overlay.blind v2` (host numbers only), decided at WP-A."""
    name = "V2"
    if d.get("set") != "hval" and not dry:
        raise Refused(f"{name}: an output of set {d.get('set')!r}, not hval")
    if not dry and str(d.get("title", "")).startswith("DRY RUN"):
        raise Refused(f"{name}: a dry run on placeholder scores")
    if _need(d, name, "events") != len(ids):
        raise Refused(f"{name}: {d.get('events')} events, not the {len(ids)} of the set")
    gate = _need(d, name, "gate")
    if not dry and gate.get("config") != "WP-A":
        raise Refused(f"{name}: decided at {gate.get('config')!r}, not WP-A")
    if not isinstance(gate.get("pass"), bool):
        raise Refused(f"{name}: its gate has no pass")
    man = _need(d, name, "manifest_sha256")
    if expect_manifest is not None and man != expect_manifest:
        raise Refused(f"{name}: made from a sealed directory with manifest {str(man)[:12]}..., not the one RESULTS.md "
                      f"records ({expect_manifest[:12]}...)")
    if meta_sha is not None and _need(d, name, "overlay_meta", "sha256") != meta_sha:
        raise Refused(f"{name}: made from another H-val meta file than the one stored now")
    return {"kind": "flag", "outcome": "pass" if gate["pass"] else "fail", "config": gate.get("config"),
            "manifest_sha256": man, "overlay_hash": _need(d, name, "overlay_meta", "overlay_hash"),
            **{k: v for k, v in (_v2_row(d) or {}).items()}}


def _v2_row(d: dict) -> dict | None:
    rows = d.get("configs") or []
    if not rows:
        return None
    r = rows[0]
    return {k: r.get(k) for k in ("delta_prompt_eff_points", "delta_fake_rate_points", "delta_trackml_score",
                                  "within")}


def nav_losses(d: dict, set_name: str, ids: list[int], root: str) -> dict:
    """The navigator losses of a gun set (hnl.overlay.gun_content) and the lost share of section 7."""
    name = "gun content"
    _check_set(d, name, set_name, ids, root, events_listed=False)
    nl = _need(d, name, "nav_losses")
    share = nl.get("pp_like_lost_share")
    pp, lost = nl.get("pp_like"), nl.get("pp_like_lost")
    if not isinstance(pp, int) or pp <= 0 or not isinstance(lost, int) or share is None:
        raise Refused(f"{name}: no navigator-loss share (P_P-like particles {pp})")
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share <= 1 \
            or abs(share - lost / pp) > 1e-12:
        raise Refused(f"{name}: lost share {share!r} is not {lost} / {pp}")
    return {"gun_particles": nl.get("gun_particles"), "lost": nl.get("lost"), "pp_like": pp, "pp_like_lost": lost,
            "pp_like_lost_share": float(share)}


def meta_record(root: str, set_name: str) -> dict:
    p = os.path.join(root, f"meta_{set_name}.json")
    if not os.path.isfile(p):
        raise Refused(f"no meta file {p}")
    with open(p) as fh:
        m = json.load(fh)
    return {"path": _real(p), "sha256": B.sha256_file(p), "overlay_hash": m.get("overlay_hash")}


def check_passes(passes: list[str], oh: str) -> list[str]:
    """The pass history: 1-3 distinct overlay hashes (64 hex), the first the addendum's version, the last oh."""
    if not passes or len(passes) > MAX_PASSES:
        raise Refused(f"--passes names {len(passes)} versions; section 9.3 allows one to {MAX_PASSES} passes")
    if any(not B.HEX64.match(p) for p in passes):
        raise Refused("--passes: every pass is an overlay_hash of 64 hex digits")
    if len(set(passes)) != len(passes):
        raise Refused("--passes names a version twice (each change gives a new overlay_hash)")
    if passes[0] != ADDENDUM_OVERLAY_HASH:
        raise Refused(f"--passes: the first pass is the version of the addendum's commit, "
                      f"{ADDENDUM_OVERLAY_HASH[:12]}..., not {passes[0][:12]}...")
    if passes[-1] != oh:
        raise Refused(f"--passes: the last pass {passes[-1][:12]}... is not the version the gate outputs carry "
                      f"({oh[:12]}...)")
    return list(passes)


def version_record(repo: str, oh: str, committed: bool, state: dict | None = None) -> dict | None:
    """The version record (configs/trackml/overlay_version_h_v1.json) of the repository: its overlay_hash must be oh,
    and a restriction or descriptive state it records must agree. committed: it must exist and be committed as is."""
    p = os.path.join(repo, os.path.relpath(B.VERSION_RECORD, B.REPO))
    if not os.path.exists(p):
        if committed:
            raise Refused(f"{os.path.relpath(p, repo)} is not committed (H-T0.5 step 4)")
        return None
    is_committed = B.git_committed(p, repo)
    if committed and not is_committed:
        raise Refused(f"{os.path.relpath(p, repo)} is not committed as is")
    with open(p) as fh:
        v = json.load(fh)
    if v.get("overlay_hash") != oh:
        raise Refused(f"the version record holds overlay_hash {str(v.get('overlay_hash'))[:12]}..., not {oh[:12]}...")
    for k in ("restriction", "descriptive"):
        if state is not None and k in v and v[k] != state[k]:
            raise Refused(f"the version record holds {k} {v[k]!r}, the record gives {state[k]!r}")
    return {"path": os.path.relpath(p, repo), "sha256": B.sha256_file(p), "committed": is_committed,
            "overlay_hash": oh}


# ------------------------------------------------------------------------------------------------ derivation
def derive(gates: dict, n_passes: int, c1_flag: dict | None = None) -> dict:
    """Section 9.3 on the gate table of the version (module docstring): the restriction, the flags and the descriptive
    state, with how they were found. Refused where the rules do not settle the case. c1_flag: {decision, rows} of
    --c1-flag-decision (`load_c1_flag_decision`; rows `c1_failed_rows` when the C1 confirmation fails, else None)."""
    for g in DECISIVE + FLAG_GATES:
        if g not in gates:
            raise Refused(f"no outcome of {g}")
    truth_failed = [g for g in DECISIVE[:2] if gates[g]["outcome"] != "pass"]
    if truth_failed:
        raise Refused(f"{', '.join(truth_failed)} failed: the set does not go to the model (section 9.3); the cause is "
                      "found, fixed and logged, and every gate runs again")
    failed = [g for g in DECISIVE if gates[g]["outcome"] == "fail"]
    c1 = None
    if c1_flag is not None:
        c1 = c1_flag_use(gates["C1 confirmation"]["outcome"], c1_flag["decision"], c1_flag["rows"])
        if c1["used"]:
            failed.remove("C1 confirmation")
    a = gates["V1 (a)"]
    regions_failed = [r for r in REGIONS if not a["regions"][r]["pass"]]
    fallback = (failed == ["V1 (a)"] and a["passes_in_all"] and 1 <= len(regions_failed) <= 2)
    restriction, descriptive = None, False
    if fallback:
        restriction = [r for r in REGIONS if r not in regions_failed]
        how = (f"forward fallback (decision D12): V1 (a) passes in all but fails in {', '.join(regions_failed)}, and "
               f"every other decisive gate passes or is not informative; P_P keeps {', '.join(restriction)}")
    elif failed:
        if n_passes < MAX_PASSES:
            unc = ""
            if c1 is not None and "uncovered_rows" in c1:
                unc = f" (C1 rows that the logged decision does not name: {c1['uncovered_rows']})"
            raise Refused(f"decisive gate(s) {', '.join(failed)} failed on pass {n_passes} of {MAX_PASSES} and the "
                          "fallback does not apply: section 9.3 allows a change, and a failure traced to Fatras, the "
                          f"v15 material or the geometry needs a logged decision; no record is written{unc}")
        descriptive = True
        how = (f"descriptive: decisive gate(s) {', '.join(failed)} still fail after the third pass and the fallback "
               "does not apply (section 9.3)")
    else:
        how = "every decisive gate passes or is not informative: no restriction, not descriptive"
    flags = [g for g in FLAG_GATES if gates[g]["outcome"] == "fail"]
    if c1 is not None and c1["used"]:
        flags = [c1["flag"]] + flags
        how += (f"; the C1 confirmation fails only in rows that the logged decision names ({c1['decision']['path']}) "
                f"and is carried as the flag {c1['flag']!r}, not as a failed decisive gate")
    deriv = {"decisive_failed": failed, "fallback": fallback, "regions_failed_v1a": regions_failed,
             "not_informative": [g for g in DECISIVE + FLAG_GATES if gates[g]["outcome"] == "not informative"],
             "passes": n_passes, "how": how}
    if c1 is not None:
        deriv["c1_flag"] = c1
    return {"restriction": restriction, "flags": flags, "descriptive": descriptive, "derivation": deriv}


# ------------------------------------------------------------------------------------------------ records
def _clean(x):
    """JSON-safe: non-finite floats become null (the record is strict JSON)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):
        return _clean(x.item())
    return x


def _git_head(repo: str) -> str | None:
    p = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() or None


def _code(repo: str) -> dict:
    f = os.path.abspath(__file__)
    return {"module": "hnl.overlay.d7record", "sha256": B.sha256_file(f), "git": _git_head(B.REPO),
            "repo_git": _git_head(repo)}


def hval_record(inputs: dict, passes: list[str], repo: str | None = None, root: str | None = None,
                split: dict | None = None, runs: str | None = None, c1_flag_decision: str | None = None) -> dict:
    """The H-val record (module docstring) from the paths of its inputs {truth_checks, closure_checks, c1_confirm,
    closure_gate, v1, v2, gun_content} and, if given, of the C1 flag decision (--c1-flag-decision)."""
    repo = repo or REPO
    root = root or DERIVED_OVERLAY
    split = split or B.load_split()
    need = ("truth_checks", "closure_checks", "c1_confirm", "closure_gate", "v1", "v2", "gun_content")
    missing = [k for k in need if not inputs.get(k)]
    if missing:
        raise Refused(f"the H-val record needs {', '.join('--' + k.replace('_', '-') for k in missing)}")
    loaded = {k: load_input(k.replace("_", "-"), inputs[k], runs) for k in need}
    d = {k: v[0] for k, v in loaded.items()}
    hv, cl = B.set_ids("hval", split), B.set_ids("closure", split)
    gates = {"per-set truth checks, H-val": truth_gate(d["truth_checks"], "truth checks", "hval", hv, root),
             "per-set truth checks, closure": truth_gate(d["closure_checks"], "closure checks", "closure", cl, root,
                                                         CLOSURE_KEYS),
             "C1 confirmation": c1_gate(d["c1_confirm"], "hval", hv, root),
             "closure truth gate": closure_gate(d["closure_gate"], cl, root)}
    if not _same(d["c1_confirm"].get("work"), d["truth_checks"].get("work")):
        raise Refused("C1 confirmation: another work directory than the H-val truth checks")
    if not _same(d["closure_gate"].get("work"), d["closure_checks"].get("work")):
        raise Refused("closure truth gate: another work directory than the closure checks")
    gates.update(v1_gates(d["v1"], cl, root))
    metas = {s: meta_record(root, s) for s in ("hval", "closure")}
    try:
        expect = B.results_expect(repo)
    except LookupError as e:
        raise Refused(f"V2: {e} (decision D4: the sealed H-val manifest is recorded before V2 is read)") from e
    gates["V2"] = v2_gate(d["v2"], hv, expect, metas["hval"]["sha256"])
    nl = nav_losses(d["gun_content"], "hval", hv, root)
    if not _same(d["gun_content"].get("work"), d["truth_checks"].get("work")):
        raise Refused("gun content: another work directory than the H-val truth checks")
    oh = gates["per-set truth checks, H-val"]["overlay_hash"]
    carried = {"truth checks": oh, "closure checks": d["closure_checks"].get("overlay_hash"),
               "closure checks version": gates["per-set truth checks, closure"]["overlay_hash"],
               "V2": gates["V2"]["overlay_hash"], "meta_hval.json": metas["hval"]["overlay_hash"],
               "meta_closure.json": metas["closure"]["overlay_hash"]}
    other = {k: v for k, v in carried.items() if v != oh}
    if not isinstance(oh, str) or not B.HEX64.match(oh) or other:
        raise Refused(f"the inputs carry different overlay hashes: {carried}")
    passes = check_passes(passes, oh)
    ins = {k: v[1] for k, v in loaded.items()}
    c1f = None
    if c1_flag_decision is not None:
        dec = load_c1_flag_decision(c1_flag_decision, repo, passes)
        c1f = {"decision": dec, "rows": c1_failed_rows(d["c1_confirm"])
               if gates["C1 confirmation"]["outcome"] == "fail" else None}
        ins["c1_flag_decision"] = {"path": dec["path"], "sha256": dec["sha256"]}
    st = derive(gates, len(passes), c1f)
    vr = version_record(repo, oh, committed=False, state=st)
    return _clean({"record": "addendum H decision D7", "schema": SCHEMA, "set": "hval", "step": STEPS["hval"],
                   "restriction": st["restriction"], "flags": st["flags"], "lost_share": nl["pp_like_lost_share"],
                   "descriptive": st["descriptive"], "overlay_hash": oh, "passes": passes,
                   "derivation": st["derivation"], "gates": gates, "nav_losses": nl,
                   "inputs": ins, "metas": metas, "version_record": vr,
                   "hval_record": None, "rules": RULES})


def read_hval_record(repo: str) -> dict:
    """The committed H-val record of the repository, checked as hnl.overlay.h1 reads it and against the schema."""
    rel = RECORD_PATHS["hval"]
    p = os.path.join(repo, rel)
    if not os.path.exists(p) or not B.git_committed(p, repo):
        raise Refused(f"{rel} is not committed as is (H-T0.5 step 4)")
    try:
        rr = H.restriction_record(p, repo, "hval")
    except (PermissionError, ValueError) as e:
        raise Refused(f"{rel}: {e}") from e
    with open(p) as fh:
        d = json.load(fh)
    if d.get("record") != "addendum H decision D7" or d.get("schema") != SCHEMA or d.get("set") != "hval":
        raise Refused(f"{rel}: not an H-val record of this module (schema {SCHEMA})")
    if not isinstance(d.get("passes"), list) or not d["passes"] or d["passes"][-1] != d.get("overlay_hash"):
        raise Refused(f"{rel}: no pass history ending at its overlay_hash")
    return {**rr, "overlay_hash": d["overlay_hash"], "passes": d["passes"]}


def htest_record(inputs: dict, repo: str | None = None, root: str | None = None, split: dict | None = None,
                 runs: str | None = None, host_root: str | None = None) -> dict:
    """The H-test record (module docstring) from the paths of its inputs {truth_checks, gun_content}."""
    from hnl.overlay import hosts as HR
    repo = repo or REPO
    root = root or DERIVED_OVERLAY
    split = split or B.load_split()
    host_root = host_root or HR.DERIVED_TP03
    need = ("truth_checks", "gun_content")
    missing = [k for k in need if not inputs.get(k)]
    if missing:
        raise Refused(f"the H-test record needs {', '.join('--' + k.replace('_', '-') for k in missing)}")
    hrec = read_hval_record(repo)
    oh = hrec["overlay_hash"]
    vr = version_record(repo, oh, committed=True, state=hrec)
    try:
        ids = B.set_ids("htest", split)
    except (KeyError, ValueError) as e:
        raise Refused(f"no H-test list: {e}") from e
    loaded = {k: load_input(k.replace("_", "-"), inputs[k], runs) for k in need}
    d = {k: v[0] for k, v in loaded.items()}
    tg = truth_gate(d["truth_checks"], "H-test truth checks", "htest", ids, root)
    if tg["outcome"] != "pass":
        raise Refused(f"H-test truth checks failed ({', '.join(tg['failed_checks'])}): H-T1 stops (section 9.4)")
    if _need(d["truth_checks"], "H-test truth checks", "version", "committed_version") is not True:
        raise Refused("H-test truth checks: the version check did not compare with the committed version record")
    if not _same(_need(d["truth_checks"], "H-test truth checks", "format", "host_root"), host_root):
        raise Refused(f"H-test truth checks: hosts not read from {host_root}")
    nl = nav_losses(d["gun_content"], "htest", ids, root)
    if not _same(d["gun_content"].get("work"), d["truth_checks"].get("work")):
        raise Refused("gun content: another work directory than the H-test truth checks")
    meta = meta_record(root, "htest")
    carried = {"truth checks": tg["overlay_hash"], "meta_htest.json": meta["overlay_hash"]}
    other = {k: v for k, v in carried.items() if v != oh}
    if other:
        raise Refused(f"the H-test inputs carry another overlay_hash than the committed {oh[:12]}...: {other}")
    hv_path = os.path.join(repo, RECORD_PATHS["hval"])
    return _clean({"record": "addendum H decision D7", "schema": SCHEMA, "set": "htest", "step": STEPS["htest"],
                   "restriction": hrec["restriction"], "flags": hrec["flags"],
                   "lost_share": nl["pp_like_lost_share"], "descriptive": hrec["descriptive"], "overlay_hash": oh,
                   "passes": hrec["passes"],
                   "derivation": {"how": "restriction, flags, descriptive state and passes of the committed H-val "
                                         "record; the navigator-loss share of H-test",
                                  "htest_hosts": len(ids)},
                   "gates": {"per-set truth checks, H-test": tg}, "nav_losses": nl,
                   "inputs": {k: v[1] for k, v in loaded.items()}, "metas": {"htest": meta}, "version_record": vr,
                   "hval_record": {"path": RECORD_PATHS["hval"], "sha256": B.sha256_file(hv_path)}, "rules": RULES})


def check_writable(set_name: str, repo: str, runs: str | None = None) -> str:
    """The fixed path of the set's record, refused if it exists, if the set was launched, or (H-test) if ledger entry
    R15 exists."""
    if set_name not in RECORD_PATHS:
        raise Refused(f"no record of decision D7 for set {set_name!r}")
    path = os.path.join(repo, RECORD_PATHS[set_name])
    if os.path.lexists(path):
        raise Refused(f"{RECORD_PATHS[set_name]} exists: the record is written once")
    runs = runs or B.runs_root(repo)
    if os.path.exists(B.launch_path(set_name, runs)):
        raise Refused(f"set {set_name} was launched ({B.launch_path(set_name, runs)}), which pins its records")
    if set_name == "htest":
        led = os.path.join(repo, os.path.relpath(B.LEDGER, B.REPO))
        if os.path.exists(led) and B.ledger_entry(led, "R15"):
            raise Refused("ledger entry R15 exists: it names the H-T1 records, which are written before it")
    return path


def write_record(rec: dict, repo: str, runs: str | None = None) -> str:
    """Writes the record to its fixed path, exclusively, after the value checks that hnl.overlay.h1 makes when it reads
    a record (hnl.overlay.h1.record_values)."""
    path = check_writable(rec["set"], repo, runs)
    try:
        H.record_values(rec, RECORD_PATHS[rec["set"]])
    except ValueError as e:
        raise Refused(str(e)) from e
    rec = {**rec, "code": _code(repo), "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    text = json.dumps(rec, indent=1, allow_nan=False) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    return path


def code_clean(repo: str = REPO) -> list[str]:
    """Modules of the record that differ from HEAD (the command line writes with committed code only)."""
    out = []
    for m in ("d7record.py", "h1.py", "blind.py"):
        p = os.path.join(repo, "src", "hnl", "overlay", m)
        if not B.git_committed(p, repo):
            out.append(os.path.relpath(p, repo))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("hval")
    v.add_argument("--passes", required=True, help="overlay_hash of every version made at H-T0.5, in order, comma "
                                                   "separated (the last is the version of the gate outputs)")
    for k in ("truth-checks", "closure-checks", "c1-confirm", "closure-gate", "v1", "v2", "gun-content"):
        v.add_argument(f"--{k}", required=True)
    v.add_argument("--c1-flag-decision", default=None, help="the committed decision that carries named failed rows of "
                   "the C1 confirmation as a flag (configs/trackml/overlay_h_c1_flag_v1.json; module docstring)")
    t = sub.add_parser("htest")
    for k in ("truth-checks", "gun-content"):
        t.add_argument(f"--{k}", required=True)
    p = sub.add_parser("path")
    p.add_argument("--set", required=True, choices=sorted(RECORD_PATHS))
    a = ap.parse_args(argv)
    if a.cmd == "path":
        print(RECORD_PATHS[a.set])
        return
    try:
        dirty = code_clean(REPO)
        if dirty:
            raise Refused(f"uncommitted code: {', '.join(dirty)} (the record is written by committed code only)")
        check_writable(a.cmd, REPO)
        if a.cmd == "hval":
            inputs = {k: getattr(a, k) for k in ("truth_checks", "closure_checks", "c1_confirm", "closure_gate", "v1",
                                                 "v2", "gun_content")}
            rec = hval_record(inputs, [x for x in a.passes.split(",") if x], REPO,
                              c1_flag_decision=a.c1_flag_decision)
        else:
            rec = htest_record({"truth_checks": a.truth_checks, "gun_content": a.gun_content}, REPO)
        path = write_record(rec, REPO)
    except Refused as e:
        print(f"d7record: refused: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps({"wrote": os.path.relpath(path, REPO), **{k: rec[k] for k in (
        "restriction", "flags", "lost_share", "descriptive", "overlay_hash")}}))


if __name__ == "__main__":
    main()
