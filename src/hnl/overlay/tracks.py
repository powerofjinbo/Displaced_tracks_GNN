"""Track building and evaluation of overlay events in the conditions of addendum H (docs/preregistration_pileup.md
sections 5, 7 and 10): the frozen builders of hnl.trackml.trackbuild, unchanged, on scored edges of overlay events in
C-full, C-1GeV or C-hs (hnl.overlay.conditions.with_condition, the only place C-hs is defined), with the frozen
hnl.trackml.evaluate.event_counts. This module runs no graph or edge-classifier network: it builds tracks from edge
scores it is given (WP-A and WP-S with the frozen learned merge model of selection_v1.json, WP-R-P3 and the pilot's dry
configurations without a learned component) and writes the per-event counts and per-target rows in the layout of
`hnl.trackml.trackbuild grid` (per_event.parquet, per_target.parquet, timing.parquet, configs.json, config_names.json,
grid.json), which hnl.overlay.hstats.read_per_target and hnl.overlay.h1 read.

Rules (checked here and, for jobs, by slurm/overlay/guard_h.sh first):
- H-val in C-full is evaluated by hnl.overlay.blind alone (sealed directory): refused here.
- H-val in C-1GeV and C-hs: only after the overlay version record (configs/trackml/overlay_version_h_v1.json) is
  committed (hnl.overlay.blind.unblinded).
- closure (k = 1): C-full and C-1GeV only (hnl.overlay.conditions refuses C-hs on closure ids as well).
- pilot (k = 9): truth-made placeholder scores only (hnl.overlay.blind.placeholder_scores, `dry-scores` below), the
  builders without a learned component (hnl.overlay.blind.DRY_CONFIGS; mwslmerge refused), outputs never under the
  production runs root; the tables may come from a scratch root.
- every other set: scores written by the frozen classifier (sha256 of selection_v1.json in their sidecars), tables
  from the production overlay root.
- grants (hnl.overlay.grant): `grid` runs on overlay ids only under a grant of the condition, of kind eval (dry on the
  pilot), whose out holds the grid directory; on H-val and H-test also only while the set's launch record accepts
  HNL_H_LAUNCH with every pinned file unchanged (hnl.overlay.blind.check_launch), so H-test grids need the launch
  token and R15 that the guard checks before it grants them.
Grids that a decisive read-out takes (`grid_provenance`, used by hnl.overlay.v1 for the closure set and by
hnl.overlay.h1 on H-val and H-test): `grid_record` passes for the set and condition (every event of the set, no
placeholder), the scores of every block carry the frozen classifier's sha256 of selection_v1.json and were made in the
same condition, the learned merge model is the one selection_v1.json names, the grid's events are those of the tables
read and it was built on those tables (the production overlay root).
Configurations: by default the working points that section 7 reports in every condition, WP-A (primary), WP-S and
WP-R-P3 (`H_WPS`, from configs/trackml/selection_v1.json); on the pilot the dry configurations. The TrackML score is
computed in C-full only (as hnl.trackml.trackbuild grid: its normalisation changes in the truth-filtered conditions).

    python -m hnl.overlay.tracks dry-scores --set pilot --cond 1gev --out DIR [--root ROOT]     (code tests only)
    python -m hnl.overlay.tracks grid --set SET --cond full|1gev|hs --scores DIR --out DIR [--root ROOT]
        [--block K --n-blocks N] [--workers W]
    python -m hnl.overlay.tracks merge --out DIR          (pools the blocks DIR/blocks/b<K>of<N> into DIR)
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.overlay import blind as B
from hnl.overlay import grant as G

H_WPS = ("WP-A", "WP-S", "WP-R-P3")
GRID_FILES = ("per_event", "per_target", "timing")


def _fmt(x: float) -> str:
    s = repr(float(x))
    return s[:-2] if s.endswith(".0") else s


def wp_configs(selection: str = B.SELECTION) -> list[dict]:
    """The working points of `H_WPS` as {name, family, params} (params comma-joined as in selection_v1.json)."""
    from hnl.overlay.hstats import wp_params
    out = []
    for wp in H_WPS:
        fam, *par = wp_params(wp, selection)
        out.append({"name": wp, "family": fam, "params": ",".join(_fmt(p) for p in par)})
    return out


def check_request(set_name: str, cond: str, out: str, root: str | None = None, split: dict | None = None,
                  repo: str = B.REPO) -> tuple[int, str | None]:
    """(set index, table root for hnl.trackml.events.iter_events) of an allowed request; raises otherwise."""
    split = split or B.load_split()
    k = B.set_index(set_name, split)
    if cond not in B.CONDITIONS_H:
        raise ValueError(f"condition {cond!r} not in {B.CONDITIONS_H}")
    if k == 0 and cond == "full":
        raise PermissionError("H-val in C-full is evaluated by hnl.overlay.blind only (the sealed directory)")
    if k == 0 and not B.unblinded(repo):
        raise PermissionError("H-val in C-1GeV and C-hs only after configs/trackml/overlay_version_h_v1.json "
                              "(overlay_hash) is committed")
    if k == 1 and cond == "hs":
        raise PermissionError("the closure set (k = 1) runs in C-full and C-1GeV only")
    if k == 9 and B._under(out, B.RUNS_H):
        raise PermissionError("placeholder outputs never go under the production runs root")
    return k, B.overlay_root(k, root)


def _grid_chunk(args):
    from hnl.overlay.conditions import with_condition
    from hnl.trackml import events as E
    from hnl.trackml import trackbuild as TB
    from hnl.trackml.evaluate import event_counts
    events, scores, cfgs, cond, merge_model_path, root = args
    model = None
    if merge_model_path:
        import joblib
        model = joblib.load(merge_model_path)
    sc_all = TB.read_scores(scores, events, cond)
    counts, parts, timing = [], [], []
    for ev0 in E.iter_events(events, root=root):
        ev = with_condition(ev0, cond)
        a, b, sc = sc_all[ev.event]
        if len(a) and max(int(np.max(a)), int(np.max(b))) >= ev.n:
            raise ValueError(f"event {ev.event}: scored edges index beyond the {ev.n} hits of condition {cond}")
        st = TB.EventState(ev, a, b, sc, model)
        hid = ev.full_rows + 1
        for i, cfg in enumerate(cfgs):
            t0 = time.time()
            lab = TB.build(cfg, st)
            secs = time.time() - t0
            c, rows = event_counts(ev.pid, lab, ev.particles, hid, ev.weight if cond == "full" else None)
            counts.append({"event": ev.event, "config": i, **c})
            rows.insert(0, "config", np.int16(i))
            rows.insert(0, "event", np.int32(ev.event))
            parts.append(rows)
            timing.append({"event": ev.event, "config": i, "family": cfg[0], "seconds": round(secs, 2)})
    return pd.DataFrame(counts), pd.concat(parts, ignore_index=True), pd.DataFrame(timing)


def _write(d: str, tabs: dict, configs: list[dict], rec: dict) -> None:
    os.makedirs(d, exist_ok=True)
    for n, t in tabs.items():
        t.to_parquet(os.path.join(d, n + ".parquet"), index=False)
    with open(os.path.join(d, "configs.json"), "w") as fh:
        json.dump([[c["family"], *[float(x) for x in str(c["params"]).split(",") if x != ""]] for c in configs], fh)
    with open(os.path.join(d, "config_names.json"), "w") as fh:
        json.dump([c["name"] for c in configs], fh)
    with open(os.path.join(d, "grid.json"), "w") as fh:
        json.dump(rec, fh, indent=1)


def grid(set_name: str, cond: str, scores: str, out: str, configs: list[dict] | None = None,
         root: str | None = None, block: int = 0, n_blocks: int = 1, workers: int = 1, split: dict | None = None,
         selection: str = B.SELECTION, repo: str = B.REPO) -> dict:
    """The frozen builders on the scored edges of one set (or one block of it) in one condition (module docstring)."""
    import concurrent.futures as cf
    split = split or B.load_split()
    k, root = check_request(set_name, cond, out, root, split, repo)
    events = B.set_ids(set_name, split)
    per = -(-len(events) // n_blocks)
    events = events[block * per:(block + 1) * per]
    if not events:
        raise SystemExit(f"block {block} of {n_blocks} is empty")
    G.check(events, "dry" if k == 9 else "eval", cond, out=out)
    if k in (0, 2):                                   # the launch, with every pinned file unchanged (decision D3)
        bad = B.check_launch(set_name, os.environ.get("HNL_H_LAUNCH"), B.runs_root(repo), repo)
        if bad:
            raise PermissionError(f"{set_name}: refused by its launch record: {bad[:5]}")
    prov = B.scores_provenance(scores, k, events, selection, cond)
    if configs is None:
        configs = B.DRY_CONFIGS if k == 9 else wp_configs(selection)
    cfgs = [B.config_tuple(c) for c in configs]
    merge_model = None
    if any(c[0] == "mwslmerge" for c in cfgs):
        if k == 9:
            raise PermissionError("no learned component runs on a pilot event (mwslmerge refused on k = 9)")
        with open(selection) as fh:
            mm = json.load(fh)["inputs"]["merge_model"]
        if B.sha256_file(mm["path"]) != mm["sha256"]:
            raise ValueError(f"{mm['path']}: sha256 differs from selection_v1.json")
        merge_model = mm["path"]
    d = out if n_blocks == 1 else os.path.join(out, "blocks", B.block_name(block, n_blocks))
    if os.path.exists(os.path.join(d, "grid.json")):
        raise SystemExit(f"{d} holds a grid already: refusing to overwrite it")
    t0 = time.time()
    chunks = [(events[i:i + 1], scores, cfgs, cond, merge_model, root) for i in range(len(events))]
    with cf.ProcessPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as ex:
        res = list(ex.map(_grid_chunk, chunks))
    tabs = {n: pd.concat([r[i] for r in res], ignore_index=True) for i, n in enumerate(GRID_FILES)}
    rec = {"set": set_name, "k": k, "condition": cond, "events": [int(e) for e in events], "n_events": len(events),
           "block": block, "n_blocks": n_blocks, "root": root or B.DERIVED_OVERLAY, "scores": os.path.abspath(scores),
           "scores_provenance": prov, "configs": configs, "merge_model": merge_model,
           "builders": "hnl.trackml.trackbuild.build (frozen), hnl.trackml.evaluate.event_counts",
           "condition_code": "hnl.overlay.conditions.with_condition", "elapsed_s": round(time.time() - t0, 1),
           "git": B._git_head(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    _write(d, tabs, configs, rec)
    return rec


def merge(out: str, split: dict | None = None) -> dict:
    """Pools the blocks out/blocks/b<K>of<N>: one set, condition and configuration list, every event once."""
    split = split or B.load_split()
    dirs = sorted(glob.glob(os.path.join(out, "blocks", "b*of*")))
    if not dirs:
        raise SystemExit(f"{out}: no blocks")
    recs = [json.load(open(os.path.join(d, "grid.json"))) for d in dirs]
    keys = {json.dumps([r["set"], r["condition"], r["configs"], r["root"], r["scores"]], sort_keys=True) for r in recs}
    if len(keys) != 1 or len({r["n_blocks"] for r in recs}) != 1 or len(dirs) != recs[0]["n_blocks"]:
        raise SystemExit(f"{out}: blocks differ in set, condition, configurations, root or scores, or are missing")
    tabs = {n: pd.concat([pd.read_parquet(os.path.join(d, n + ".parquet")) for d in dirs], ignore_index=True)
            for n in GRID_FILES}
    want = B.set_ids(recs[0]["set"], split)
    per = tabs["per_event"].groupby("config").event.agg(lambda e: sorted(int(x) for x in e))
    if len(per) != len(recs[0]["configs"]) or any(v != want for v in per):
        raise SystemExit(f"{out}: blocks do not hold every event of {recs[0]['set']} once for every configuration")
    if os.path.exists(os.path.join(out, "grid.json")):
        raise SystemExit(f"{out} holds a merged grid already")
    rec = {**{k: recs[0][k] for k in ("set", "k", "condition", "root", "scores", "configs", "merge_model", "builders",
                                       "condition_code")},
           "events": want, "n_events": len(want), "block": None, "n_blocks": recs[0]["n_blocks"],
           "scores_provenance": {"blocks": [r["scores_provenance"] for r in recs],
                                 "placeholder": all(r["scores_provenance"]["placeholder"] for r in recs),
                                 "ckpt_sha256": recs[0]["scores_provenance"]["ckpt_sha256"],
                                 "condition": recs[0]["condition"]},
           "elapsed_s": [r["elapsed_s"] for r in recs], "slurm_job": [r["slurm_job"] for r in recs],
           "git": B._git_head(), "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    _write(out, tabs, recs[0]["configs"], rec)
    return rec


def grid_record(d: str, set_name: str, cond: str, split: dict | None = None) -> dict:
    """The record of a complete grid of this module for one set and condition (raises if it is of another set or
    condition, incomplete, or, for the pilot, not made from placeholders, or otherwise from placeholders)."""
    split = split or B.load_split()
    with open(os.path.join(d, "grid.json")) as fh:
        rec = json.load(fh)
    k = B.set_index(set_name, split)
    if rec.get("set") != set_name or rec.get("condition") != cond or int(rec.get("k", -1)) != k:
        raise ValueError(f"{d}: a grid of {rec.get('set')} in {rec.get('condition')}, not {set_name} in {cond}")
    if sorted(int(e) for e in rec["events"]) != B.set_ids(set_name, split):
        raise ValueError(f"{d}: the grid does not hold every event of {set_name}")
    if bool(rec["scores_provenance"]["placeholder"]) != (k == 9):
        raise ValueError(f"{d}: placeholder scores are for the pilot set only, and the pilot takes nothing else")
    return rec


def grid_provenance(d: str, set_name: str, cond: str, root: str | None = None, events=None,
                    split: dict | None = None, selection: str = B.SELECTION) -> dict:
    """The record of a grid that a decisive read-out may take (module docstring; hnl.overlay.v1 for the closure set,
    hnl.overlay.h1 on H-val and H-test), after every check; raises otherwise. root: the overlay tables the read-out
    reads (the production overlay root; k = 0, 1, 2 are never read from elsewhere); events: their events, if given."""
    split = split or B.load_split()
    k = B.set_index(set_name, split)
    if k == 9:
        raise PermissionError("the pilot's grids are placeholders: grid_record, not grid_provenance")
    root = B.overlay_root(k, root) or B.DERIVED_OVERLAY       # raises unless root is the production overlay root
    rec = grid_record(d, set_name, cond, split)               # set, condition, every event of the set, no placeholder
    with open(selection) as fh:
        inputs = json.load(fh)["inputs"]
    want = inputs["classifier_checkpoint"]["sha256"]
    prov = rec["scores_provenance"]
    for q in [prov, *prov.get("blocks", [])]:
        if q.get("placeholder") or q.get("ckpt_sha256") != want or q.get("condition", cond) != cond:
            raise ValueError(f"{d}: scores not of the frozen classifier in {cond} (sha256 "
                             f"{str(q.get('ckpt_sha256'))[:12]}..., placeholder {q.get('placeholder')}, condition "
                             f"{q.get('condition')})")
    if any(c.get("family") == "mwslmerge" for c in rec.get("configs", [])) and \
            rec.get("merge_model") != inputs["merge_model"]["path"]:
        raise ValueError(f"{d}: merge model {rec.get('merge_model')} is not the one selection_v1.json names")
    if events is not None and sorted(int(e) for e in rec["events"]) != sorted(int(e) for e in events):
        raise ValueError(f"{d}: the grid's events are not those of the tables read at {root}")
    if os.path.realpath(rec.get("root") or "") != os.path.realpath(root):
        raise ValueError(f"{d}: built on the tables at {rec.get('root')}, not {root}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dry-scores")
    s.add_argument("--set", default="pilot")
    s.add_argument("--cond", required=True, choices=B.CONDITIONS_H)
    s.add_argument("--out", required=True)
    s.add_argument("--root", default=None)
    s.add_argument("--sigma", type=float, default=2.0)
    g = sub.add_parser("grid")
    g.add_argument("--set", required=True)
    g.add_argument("--cond", required=True, choices=B.CONDITIONS_H)
    g.add_argument("--scores", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--root", default=None)
    g.add_argument("--configs", default=None, help="JSON list of {name, family, params} (default: WP-A, WP-S and "
                   "WP-R-P3; on the pilot the dry configurations)")
    g.add_argument("--block", type=int, default=0)
    g.add_argument("--n-blocks", type=int, default=1)
    g.add_argument("--workers", type=int, default=1)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "dry-scores":
        print(json.dumps(B.placeholder_scores(a.set, a.out, a.cond, a.sigma, root=a.root)))
    elif a.cmd == "grid":
        r = grid(a.set, a.cond, a.scores, a.out, json.loads(a.configs) if a.configs else None, a.root, a.block,
                 a.n_blocks, a.workers)
        print(json.dumps({k: r[k] for k in ("set", "condition", "n_events", "block", "n_blocks", "elapsed_s")}))
    else:
        r = merge(a.out)
        print(json.dumps({k: r[k] for k in ("set", "condition", "n_events", "n_blocks")}))


if __name__ == "__main__":
    main()
