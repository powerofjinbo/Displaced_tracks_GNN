"""The code of addendum H step H-T1 (truth only; docs/preregistration_pileup.md sections 3, 9.4 and 10). No model reads
an H-test event, and no command of this module reads a model output.

  split-tp03     after pass 1 (`python -m hnl.trackml.throughput scan --part training_part03`): the converter split
                 file configs/trackml/split_tp03_v1.json. hnl.trackml.throughput convert stores score weights for the
                 calib, val and test events of its split file and refuses a part outside it; throughput.py and
                 convert.py are in the provenance manifest, so they stay as they are. This file puts every complete
                 part03 event in `test` (weights stored, no truth kinematics) and leaves every other set empty. It is a
                 converter input only: the roles of part03 events are those of split_h_v1.json (H-test hosts and
                 spares; no model reads a spare).
  verify-tp03    hnl.trackml.throughput.verify_part on derived/tp03_v1 with that split file. Its two role checks for
                 parts other than part01 ("dev / val / test events outside training_part01", "events outside train")
                 hold for the throughput split of parts 01 and 02 and are expected here; every other problem stops.
  eligible       the eligible part03 ids: complete events with status ok in dq_training_part03.tsv (the converter's
                 data-quality rule). More than 1 % of the complete events failing stops H-T1 (the converter publishes
                 nothing then either) and goes to the user.
  lists          the hash rule (hnl.overlay.split_h.htest_lists) on the eligible ids, with the Fatras event numbers.
  screen         the per-event checks of section 9.4 on the hosts in hash order, before any gun is simulated: no
                 hard-scatter vertex by hnl.overlay.gun.hard_scatter_vertex; host weights that do not sum to one within
                 1e-6 in the host's own stored rows; host particles that break a precondition of the merge
                 (hnl.overlay.writer.check_event on the host side: a repeated or non-positive particle id, a vertex
                 field above 1000, ptype rows other than the particles rows, n_hits other than the hit count, hits of
                 particles without a row). Each failed host gets one drop record {host, reason, replacement}, the
                 replacement being the next spare in hash order that has not joined the list; a replacement that fails
                 is dropped in turn. Writes a record; split_h_v1.json is not touched.
  write-split    writes htest, htest_spares and htest_dropped of a screen record into split_h_v1.json and checks them
                 with hnl.overlay.split_h.fatras_order and htest_hosts. Run at H-T1 only, after the screen.
  earlier-read   the "No earlier read" check of section 9.4: the part03 ids are disjoint from every id in
                 split_tp_v1.json and split_v1.json, and no part03 event has the same hits as a tp_v1 event (sha256 of
                 the hits' x, y, z (float32 bytes), volume, layer and module, rows sorted, per event).
  fatras-check   after the H-test overlay (section 9.4: "any Fatras failure other than a logged navigator loss" stops
                 H-T1): every ERROR or FATAL line of the set's fatras.log must carry the navigator loss of the patched
                 build ("Interpolation out of bounds", MagneticFieldError); counts the failed particles.
  summary        the H-T1 summary record results/pileup_hp/h_t1_summary_v1.json from the outputs of the steps:
                 checksums, drops, twin fractions, navigator losses, the P_P and P_D counts of the stored H-test
                 overlay (hnl.overlay.h1.gun_targets, truth), the long-strip sensitivity, the per-set checks and the
                 earlier-read check. Refuses an existing record, an input whose pass flag is false, and a part03
                 conversion whose converter code differs from the provenance manifest (section 3).

    python -m hnl.overlay.ht1 split-tp03 [--out configs/trackml/split_tp03_v1.json]
    python -m hnl.overlay.ht1 verify-tp03 --out FILE.json
    python -m hnl.overlay.ht1 eligible --out FILE.txt
    python -m hnl.overlay.ht1 lists --eligible FILE.txt --out FILE.json
    python -m hnl.overlay.ht1 screen --lists FILE.json --out FILE.json [--root derived/tp03_v1]
    python -m hnl.overlay.ht1 write-split --screen FILE.json
    python -m hnl.overlay.ht1 earlier-read --out FILE.json
    python -m hnl.overlay.ht1 fatras-check --work DIR --out FILE.json
    python -m hnl.overlay.ht1 summary --inputs FILE.json --out results/pileup_hp/h_t1_summary_v1.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os

import numpy as np
import pandas as pd

from hnl.overlay import split_h as SH
from hnl.trackml.split import EXT, REPO

PART = "training_part03"
SPLIT_TP03 = os.path.join(REPO, "configs", "trackml", "split_tp03_v1.json")
SPLIT_TP = os.path.join(REPO, "configs", "trackml", "split_tp_v1.json")
SPLIT_V1 = os.path.join(REPO, "configs", "trackml", "split_v1.json")
DERIVED_TP03 = os.path.join(EXT, "derived", "tp03_v1")
DERIVED_TP = os.path.join(EXT, "derived", "tp_v1")
MAX_FAIL_FRAC = 0.01
WEIGHT_TOL = 1e-6
EXPECTED_ROLE_PROBLEMS = (f"{PART}: dev / val / test events outside training_part01", f"{PART}: events outside train")
SUMMARY = os.path.join(REPO, "results", "pileup_hp", "h_t1_summary_v1.json")


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write_json(path: str, obj, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        raise SystemExit(f"{path} exists: refusing to overwrite it")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------------------------------------ G1: converter split
def split_tp03(manifest_dir: str | None = None) -> dict:
    """The converter split file of part03 (module docstring) from the pass-1 manifest."""
    from hnl.trackml import throughput as T
    manifest_dir = manifest_dir or T.MANIFEST
    summ = T.read_summary(PART, manifest_dir)
    if not (summ["size_match"] and summ["md5_match"]):
        raise RuntimeError(f"{PART}: pass 1 did not verify size and md5 ({summ.get('md5_streamed')})")
    man = T.read_manifest(PART, manifest_dir)
    complete = T.complete_events(man)
    tsv, _ = T.manifest_paths(PART, manifest_dir)
    size_z, md5_z = T.TP_ARCHIVES[PART + ".tar"]
    return {
        "version": "tp03_v1", "phase": "throughput", "record": "10.5281/zenodo.4730157",
        "purpose": ("converter input of addendum H step H-T1 only (hnl.overlay.ht1 split-tp03): "
                    "hnl.trackml.throughput convert stores score weights for the calib, val and test events of its "
                    "split file, so every complete training_part03 event is listed as test and every other set is "
                    "empty. The roles of these events are those of configs/trackml/split_h_v1.json (H-test hosts "
                    "and spares); no model of this project reads a spare, and no event here is a test event of "
                    "the frozen model's split (configs/trackml/split_tp_v1.json)."),
        "parts": [PART],
        "archives": {PART: {"bytes": size_z, "md5": md5_z, "n_events": len(complete),
                            "first_event": int(min(complete)), "last_event": int(max(complete)),
                            "incomplete_events": summ.get("incomplete_events", []),
                            "manifest_tsv": os.path.basename(tsv), "manifest_tsv_sha256": sha256_file(tsv)}},
        "dev": [], "val": [], "valmon": [], "train": [], "calib": [], "gbdt": [], "merge_train": [],
        "test": [int(e) for e in complete],
        "sha256": {"test": SH._sha(complete)},
    }


def write_split_tp03(out: str = SPLIT_TP03, manifest_dir: str | None = None) -> dict:
    from hnl.trackml.throughput import load_tp_split
    rec = split_tp03(manifest_dir)
    _write_json(out, rec)
    s = load_tp_split(out)                               # the converter's own reader accepts it
    if s["parts"] != [PART] or len(s["test"]) != len(rec["test"]):
        raise RuntimeError(f"{out}: not read back as written")
    return rec


# ------------------------------------------------------------------------------------------------ G1: verification
def verify_tp03(root: str = DERIVED_TP03, split_path: str = SPLIT_TP03) -> dict:
    from hnl.trackml import throughput as T
    r = T.verify_part(PART, root=root, split_path=split_path)
    other = [p for p in r["problems"] if p not in EXPECTED_ROLE_PROBLEMS]
    r["expected_role_problems"] = [p for p in r["problems"] if p in EXPECTED_ROLE_PROBLEMS]
    r["problems"] = other
    r["pass"] = not other
    return r


# ------------------------------------------------------------------------------------------------ G2: eligible ids
def eligible(root: str = DERIVED_TP03) -> dict:
    """Eligible ids: complete events with status ok in the converter's data-quality table, all stored."""
    with open(os.path.join(root, f"meta_{PART}.json")) as fh:
        meta = json.load(fh)
    dq = pd.read_csv(os.path.join(root, f"dq_{PART}.tsv"), sep="\t")
    n_complete = int(meta["n_events_in_archive"])
    if len(dq) != n_complete:
        raise RuntimeError(f"dq_{PART}.tsv holds {len(dq)} events, the archive {n_complete} complete events")
    ok = sorted(int(e) for e in dq.loc[dq.status == "ok", "event"])
    failed = dq[dq.status != "ok"]
    stored = sorted({int(e) for sh in meta["shards"].values() for e in sh["hits"]["events"]})
    if stored != ok:
        raise RuntimeError("the stored events differ from the events with status ok")
    share = len(failed) / n_complete if n_complete else 1.0
    rec = {"part": PART, "n_complete": n_complete, "n_eligible": len(ok), "n_failed": int(len(failed)),
           "failed": [{"event": int(e), "status": s, "problems": p} for e, s, p in
                      zip(failed.event, failed.status, failed.problems.fillna(""))],
           "failed_share": share, "max_fail_frac": MAX_FAIL_FRAC, "stop": share > MAX_FAIL_FRAC,
           "sha256_eligible": SH._sha(ok), "eligible": ok}
    return rec


# ------------------------------------------------------------------------------------------------ G2: lists
def lists(eligible_ids) -> dict:
    r = SH.htest_lists(eligible_ids)
    order = SH.hash_order(eligible_ids)
    rank = {h: i for i, h in enumerate(order)}
    return {**r, "n_eligible": len(order), "sha256_htest": SH._sha(r["htest"]),
            "fatras_rank_max_htest": max(rank[h] for h in r["htest"]) if r["htest"] else None,
            "spares_within_seed_check": int(sum(1 for h in r["spares_hash_order"] if rank[h] < SH.N_EVENT_NEW))}


# ------------------------------------------------------------------------------------------------ G3: screen
def host_problems(host: int, root: str = DERIVED_TP03) -> list[str]:
    """The per-event checks of section 9.4 on one host's stored tables (module docstring)."""
    from hnl.overlay import gun as GUN
    from hnl.overlay import writer as W
    probs = []
    t = W.host_tables(int(host), root)
    h = t["hits"].to_pandas()
    p = t["particles"].to_pandas()
    y = t["ptype"].to_pandas()
    try:
        GUN.hard_scatter_vertex(p[["particle_id", "vx", "vy", "vz"]])
    except ValueError as e:
        probs.append(f"no hard-scatter vertex ({e})")
    w = h.weight.to_numpy(np.float64)
    s = float(np.nansum(w))
    if not np.isfinite(w).all() or abs(s - 1.0) > WEIGHT_TOL:
        probs.append(f"host weight sum {s:.9f} != 1 within {WEIGHT_TOL}")
    pid = p.particle_id.to_numpy().astype(np.int64)
    if pd.Series(pid).duplicated().any():
        probs.append("repeated particle_id")
    if (pid <= 0).any():
        probs.append("particle_id <= 0 (as int64)")
    if ((pid >> 52) > 1000).any():
        probs.append("vertex field above 1000")
    if not np.array_equal(y.particle_id.to_numpy(), p.particle_id.to_numpy()):
        probs.append("ptype rows differ from particles rows")
    hp = h.pid.to_numpy(np.int64)
    cnt = pd.Series(hp[hp != 0]).value_counts()
    if not np.array_equal(cnt.reindex(pid).fillna(0).to_numpy(np.int64), p.n_hits.to_numpy(np.int64)):
        probs.append("n_hits differs from the hit count")
    if not np.isin(hp[hp != 0], pid).all():
        probs.append("hits of particles without a row")
    return probs


def screen(lists_rec: dict, root: str = DERIVED_TP03, check=host_problems) -> dict:
    """Drop records over the hash order (module docstring): the 600 hosts first, then each replacement as it joins."""
    htest = [int(h) for h in lists_rec["htest"]]
    spares = [int(h) for h in lists_rec["spares_hash_order"]]
    order = SH.hash_order([*htest, *spares])
    rank = {h: i for i, h in enumerate(order)}
    checked: dict[int, list[str]] = {}
    for h in sorted(htest, key=lambda x: rank[x]):
        checked[h] = check(h, root)
    cur, joined, drops = set(htest), set(htest), []
    queue = [h for h in sorted(htest, key=lambda x: rank[x]) if checked[h]]
    nxt_i = 0
    while queue:
        h = queue.pop(0)
        while nxt_i < len(spares) and spares[nxt_i] in joined:
            nxt_i += 1
        if nxt_i >= len(spares):
            raise RuntimeError("no spare left to replace a dropped host")
        r = spares[nxt_i]
        if rank[r] >= SH.N_EVENT_NEW:
            raise RuntimeError(f"spare {r} has Fatras rank {rank[r]}, beyond the seed check (0-{SH.N_EVENT_NEW - 1})")
        drops.append({"host": h, "reason": "; ".join(checked[h]), "replacement": r})
        cur.remove(h)
        cur.add(r)
        joined.add(r)
        checked[r] = check(r, root)
        if checked[r]:
            queue.append(r)
    split_like = {"htest": htest, "htest_spares": spares, "htest_dropped": drops}
    hosts = SH.htest_hosts(split_like)                   # the rule of split_h_v1.json, checked here already
    return {"htest": htest, "htest_spares": spares, "htest_dropped": drops, "evaluated_hosts": hosts,
            "n_checked": len(checked), "n_dropped": len(drops), "sha256_evaluated": SH._sha(hosts),
            "fatras_rank_max": max(rank[h] for h in hosts), "root": root, "written_utc": utc()}


# ------------------------------------------------------------------------------------------------ G4: split file
def write_split(screen_rec: dict, path: str = SH.OUT_SPLIT) -> dict:
    with open(path) as fh:
        split = json.load(fh)
    if any(split.get(k) is not None for k in ("htest", "htest_spares", "htest_dropped")):
        raise SystemExit(f"{path} holds H-test lists already: refusing to overwrite them")
    split["htest"] = sorted(int(h) for h in screen_rec["htest"])
    split["htest_spares"] = [int(h) for h in screen_rec["htest_spares"]]
    split["htest_dropped"] = [{"host": int(d["host"]), "reason": d["reason"], "replacement": int(d["replacement"])}
                              for d in screen_rec["htest_dropped"]]
    SH.fatras_order("htest", split)
    hosts = SH.htest_hosts(split)
    if hosts != screen_rec["evaluated_hosts"]:
        raise RuntimeError("the evaluated hosts differ from the screen record")
    split["sha256"]["htest"] = SH._sha(hosts)
    with open(path + ".tmp", "w") as fh:
        json.dump(split, fh, indent=1)
        fh.write("\n")
    os.replace(path + ".tmp", path)
    return {"path": path, "sha256": sha256_file(path), "htest_sha256": SH.htest_sha256(split)}


# ------------------------------------------------------------------------------------------------ G5: earlier read
def _all_ids(obj) -> set[int]:
    out: set[int] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _all_ids(v)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, int) for x in obj):
            out |= {int(x) for x in obj}
        else:
            for v in obj:
                out |= _all_ids(v)
    return out


def hits_digest(h: pd.DataFrame) -> str:
    """sha256 of one event's hits: x, y, z as float32 bytes with volume, layer and module, rows sorted."""
    a = np.empty(len(h), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("v", "<u2"), ("l", "<u2"),
                                ("m", "<u4")])
    a["x"], a["y"], a["z"] = (h[c].to_numpy(np.float32) for c in ("x", "y", "z"))
    a["v"], a["l"], a["m"] = (h[c].to_numpy() for c in ("volume", "layer", "module"))
    a = np.sort(a, order=["x", "y", "z", "v", "l", "m"])
    return hashlib.sha256(a.tobytes()).hexdigest()


def event_digests(events, root: str, chunk: int = 50) -> dict[int, str]:
    from hnl.trackml.convert import read_table
    out = {}
    events = sorted(int(e) for e in events)
    for i in range(0, len(events), chunk):
        h = read_table("hits", events[i:i + chunk], ["event", "x", "y", "z", "volume", "layer", "module"], root=root)
        for e, g in h.groupby("event"):
            out[int(e)] = hits_digest(g)
    missing = sorted(set(events) - set(out))
    if missing:
        raise KeyError(f"events without hits under {root}: {missing[:5]}")
    return out


def stored_events(root: str, part: str) -> list[int]:
    with open(os.path.join(root, f"meta_{part}.json")) as fh:
        meta = json.load(fh)
    return sorted({int(e) for sh in meta["shards"].values() for e in sh["hits"]["events"]})


def earlier_read(root03: str = DERIVED_TP03, root_tp: str = DERIVED_TP) -> dict:
    with open(SPLIT_TP) as fh:
        tp = json.load(fh)
    with open(SPLIT_V1) as fh:
        v1 = json.load(fh)
    p03 = stored_events(root03, PART)
    ids_tp, ids_v1 = _all_ids({k: v for k, v in tp.items() if k != "archives"}), _all_ids(v1)
    tp_events = sorted({*stored_events(root_tp, "training_part01"), *stored_events(root_tp, "training_part02")})
    d03, dtp = event_digests(p03, root03), event_digests(tp_events, root_tp)
    same = sorted(e for e, d in d03.items() if d in set(dtp.values()))
    rec = {"part03_events": len(p03), "tp_v1_events": len(tp_events),
           "ids_in_split_tp_v1": sorted(set(p03) & ids_tp), "ids_in_split_v1": sorted(set(p03) & ids_v1),
           "n_ids_split_tp_v1": len(ids_tp), "n_ids_split_v1": len(ids_v1),
           "same_hits_as_tp_v1": same, "repeated_digests_part03": int(len(d03) - len(set(d03.values()))),
           "digest": "sha256 of x, y, z (float32), volume, layer, module per hit, rows sorted",
           "split_tp_v1_sha256": sha256_file(SPLIT_TP), "split_v1_sha256": sha256_file(SPLIT_V1)}
    rec["pass"] = not rec["ids_in_split_tp_v1"] and not rec["ids_in_split_v1"] and not same
    return rec


# ------------------------------------------------------------------------------------------------ C2: Fatras log
NAVIGATOR_LOSS = "Interpolation out of bounds"


def fatras_check(work: str) -> dict:
    """Section 9.4: the Fatras failures of the set's log are logged navigator losses only (module docstring)."""
    path = os.path.join(work, "gun", "fatras_out", "fatras.log")
    with open(path) as fh:
        lines = fh.read().splitlines()
    err = [x for x in lines if " ERROR " in x or " FATAL " in x]
    other = [x for x in err if NAVIGATOR_LOSS not in x]
    failed = [x for x in lines if "failed to simulate" in x]
    return {"log": path, "sha256": sha256_file(path), "error_lines": len(err), "navigator_loss_lines": len(err) - len(other),
            "other_error_lines": len(other), "other_examples": other[:10], "failed_particles": len(failed),
            "failed_navigator_loss": sum(NAVIGATOR_LOSS in x for x in failed), "pass": not other}


PASS_FLAGS = {"verify_tp03": ("pass", True), "earlier_read": ("pass", True), "eligible": ("stop", False),
              "fatras": ("pass", True)}
# step outputs committed as their own H-T1 records (results/pileup_hp/h_t1_truth_*): the summary keeps their path,
# sha256 and these fields only
def _n(x) -> int | None:
    return len(x) if isinstance(x, (list, dict)) else (int(x) if x is not None else None)


COMPACT = {
    "truth_checks": lambda d: {"pass": d.get("pass"), "events": _n(d.get("events")),
                               "host_root": d.get("format", {}).get("host_root"),
                               "version": {k: d.get("version", {}).get(k) for k in
                                           ("overlay_hash", "base_seed", "split_base_seed", "fatras_events_checked",
                                            "committed_version", "pass")}},
    "gun_content": lambda d: {"events": _n(d.get("events")),
                              "nav_losses": {k: v for k, v in d.get("nav_losses", {}).items()
                                             if not isinstance(v, (dict, list))}},
    "c1_confirm": lambda d: {k: d.get(k) for k in ("part2_outcome", "part2_failed", "not_informative",
                                                    "not_informative_pixel", "n_gun_hits", "n_pp_target_hits")},
}


def check_inputs(rec: dict) -> list[str]:
    """Problems of the step outputs gathered by the summary: a pass flag that is false, a failed long-strip rerun, a
    per-set truth check that does not pass, or a part03 conversion by converter code other than the manifest's."""
    bad = [f"{k}: {flag} is {rec[k].get(flag)!r}, not {want!r}" for k, (flag, want) in PASS_FLAGS.items()
           if k in rec and rec[k].get(flag) is not want]
    for k in PASS_FLAGS:
        if k not in rec:
            bad.append(f"no {k} input")
    if "longstrip" in rec and not rec["longstrip"].get("check_nominal_rerun", {}).get("pass"):
        bad.append("longstrip: the nominal rerun does not reproduce the stored set")
    if "truth_checks" in rec:
        p = rec["truth_checks"].get("pass")
        if not isinstance(p, dict) or not p or not all(v is True for v in p.values()):
            bad.append(f"truth_checks: a per-set truth check does not pass ({p})")
    else:
        bad.append("no truth_checks input")
    return bad


def converter_against_manifest(meta_path: str, version_path: str = os.path.join(REPO, "configs", "trackml",
                                                                                "overlay_version_h_v1.json")) -> dict:
    with open(meta_path) as fh:
        used = json.load(fh)["code_sha256"]
    with open(version_path) as fh:
        man = json.load(fh)["manifest"]["code_sha256"]
    rows = {f: {"conversion": used.get(f), "manifest": man.get(f"src/hnl/trackml/{f}")}
            for f in ("convert.py", "throughput.py")}
    return {"files": rows, "pass": all(r["conversion"] == r["manifest"] is not None for r in rows.values())}


# ------------------------------------------------------------------------------------------------ G8: summary
def yields(set_name: str = "htest", root: str | None = None) -> dict:
    """P_P, P_D and gun-target counts of a stored overlay set from truth (hnl.overlay.h1.gun_targets), with regions."""
    from hnl.overlay import blind as B
    from hnl.overlay.h1 import gun_targets
    from hnl.trackml.convert import read_table
    root = root or B.DERIVED_OVERLAY
    ids = B.set_ids(set_name)
    p = read_table("particles", ids, ["event", "particle_id", "pt", "eta", "d0", "rv", "n_hits", "is_target",
                                      "r_first"], root=root)
    h = read_table("hits", ids, ["event", "pid", "z"], root=root)
    h = h[B.is_gun(h.pid.to_numpy(np.int64))]
    g = gun_targets(p, h)
    pp = g[g.pp.astype(bool)]
    return {"events": len(ids), "gun_particles": int(len(g)), "targets": int(g.target.sum()), "P_P": int(len(pp)),
            "P_D": int(g.pd.sum()), "P_P_by_region": {str(k): int(v) for k, v in pp.region.value_counts().items()},
            "P_P_with_soft_sibling": int(pp.soft_sibling.sum()),
            "underpowered": bool(len(pp) < 1000)}


def summary(inputs: dict, out: str = SUMMARY) -> dict:
    """The H-T1 summary record from the outputs named in `inputs` (paths): verify_tp03, eligible, screen,
    earlier_read, fatras, truth_checks, gun_content, c1_confirm, twins, longstrip; it refuses failed inputs and adds the
    yields and the checksums."""
    from hnl.overlay import blind as B
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: refusing to overwrite it")
    rec = {"record": "addendum H H-T1 summary", "schema": 1, "set": "htest", "inputs": {}}
    for k, path in inputs.items():
        with open(path) as fh:
            rec[k] = json.load(fh)
        rec["inputs"][k] = {"path": os.path.abspath(path), "sha256": sha256_file(path)}
    for k in ("eligible",):
        if k in rec and "eligible" in rec[k]:
            rec[k] = {kk: vv for kk, vv in rec[k].items() if kk != "eligible"}
    bad = check_inputs(rec)
    for k, f in COMPACT.items():
        if k in rec:
            rec[k] = f(rec[k])
    rec["converter_manifest"] = converter_against_manifest(os.path.join(DERIVED_TP03, f"meta_{PART}.json"))
    if not rec["converter_manifest"]["pass"]:
        bad.append(f"the part03 conversion's converter code differs from the manifest: {rec['converter_manifest']}")
    if bad:
        raise SystemExit(f"summary refused: {bad}")
    rec["yields"] = yields("htest")
    root = B.DERIVED_OVERLAY
    with open(os.path.join(root, "meta_htest.json")) as fh:
        meta = json.load(fh)
    rec["checksums"] = {
        "tp03_v1_shards": json.load(open(os.path.join(DERIVED_TP03, f"meta_{PART}.json")))["shards"],
        "overlay_htest_shards": meta.get("shards"), "meta_htest_sha256": sha256_file(os.path.join(root, "meta_htest.json")),
        "sidecar_htest_sha256": sha256_file(os.path.join(root, "sidecar_htest.json")),
        "gun_truth_htest_sha256": sha256_file(os.path.join(root, "gun_truth_htest.parquet")),
        "split_h_v1_sha256": sha256_file(SH.OUT_SPLIT), "split_tp03_v1_sha256": sha256_file(SPLIT_TP03),
        "htest_sha256": SH.htest_sha256(SH.load_split())}
    rec["overlay_hash"] = meta.get("overlay_hash")
    rec["written_utc"] = utc()
    rec["git"] = os.popen(f"git -C {REPO} rev-parse HEAD").read().strip()
    _write_json(out, rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split-tp03")
    s.add_argument("--out", default=SPLIT_TP03)
    v = sub.add_parser("verify-tp03")
    v.add_argument("--out", required=True)
    e = sub.add_parser("eligible")
    e.add_argument("--out", required=True)
    li = sub.add_parser("lists")
    li.add_argument("--eligible", required=True)
    li.add_argument("--out", required=True)
    sc = sub.add_parser("screen")
    sc.add_argument("--lists", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--root", default=DERIVED_TP03)
    ws = sub.add_parser("write-split")
    ws.add_argument("--screen", required=True)
    er = sub.add_parser("earlier-read")
    er.add_argument("--out", required=True)
    fc = sub.add_parser("fatras-check")
    fc.add_argument("--work", required=True)
    fc.add_argument("--out", required=True)
    su = sub.add_parser("summary")
    su.add_argument("--inputs", required=True, help="JSON {name: path} of the step outputs")
    su.add_argument("--out", default=SUMMARY)
    a = ap.parse_args(argv)
    if a.cmd == "split-tp03":
        r = write_split_tp03(a.out)
        print(json.dumps({"out": a.out, "n_events": len(r["test"]), "sha256": sha256_file(a.out)}))
    elif a.cmd == "verify-tp03":
        r = verify_tp03()
        _write_json(a.out, r)
        print(json.dumps({k: r[k] for k in ("events_in_archive", "events_stored", "events_dropped", "problems",
                                            "expected_role_problems", "pass")}))
        if not r["pass"]:
            raise SystemExit("verify-tp03: problems (see the record)")
    elif a.cmd == "eligible":
        r = eligible()
        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} exists")
        with open(a.out, "w") as fh:
            fh.write("\n".join(str(i) for i in r["eligible"]) + "\n")
        _write_json(a.out + ".json", {k: v for k, v in r.items() if k != "eligible"})
        print(json.dumps({k: r[k] for k in ("n_complete", "n_eligible", "n_failed", "failed_share", "stop")}))
        if r["stop"]:
            raise SystemExit(f"more than {MAX_FAIL_FRAC:.0%} of the part03 events fail the data-quality rule: "
                             "H-T1 stops and the user decides")
    elif a.cmd == "lists":
        with open(a.eligible) as fh:
            ids = [int(x) for x in fh.read().split()]
        r = lists(ids)
        _write_json(a.out, r)
        print(json.dumps({k: r[k] for k in ("n_eligible", "sha256_htest", "fatras_rank_max_htest",
                                            "spares_within_seed_check")}))
    elif a.cmd == "screen":
        with open(a.lists) as fh:
            r = screen(json.load(fh), a.root)
        _write_json(a.out, r)
        print(json.dumps({k: r[k] for k in ("n_checked", "n_dropped", "sha256_evaluated", "fatras_rank_max")}))
    elif a.cmd == "write-split":
        with open(a.screen) as fh:
            print(json.dumps(write_split(json.load(fh))))
    elif a.cmd == "earlier-read":
        r = earlier_read()
        _write_json(a.out, r)
        print(json.dumps({k: r[k] for k in ("part03_events", "tp_v1_events", "ids_in_split_tp_v1",
                                            "ids_in_split_v1", "same_hits_as_tp_v1", "pass")}))
        if not r["pass"]:
            raise SystemExit("earlier-read: a part03 event was read before (see the record): H-T1 stops")
    elif a.cmd == "fatras-check":
        r = fatras_check(a.work)
        _write_json(a.out, r)
        print(json.dumps({k: r[k] for k in ("error_lines", "navigator_loss_lines", "other_error_lines",
                                            "failed_particles", "pass")}))
        if not r["pass"]:
            raise SystemExit("fatras-check: a Fatras failure other than a logged navigator loss (section 9.4): H-T1 stops")
    else:
        with open(a.inputs) as fh:
            r = summary(json.load(fh), a.out)
        print(json.dumps({"out": a.out, "yields": r["yields"]}))


if __name__ == "__main__":
    main()
