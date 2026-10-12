"""Post-freeze track labels of the frozen TrackML configurations (DECISIONS 2026-10-11, the user's decision).

The one-time test stored only pooled counts per event and configuration, under our metric definitions and our
readings of the published ones. Recomputing Canto-L's numbers under the published references' own evaluation code
(EggNet, HierarchicalGNN, Exa.TrkX) needs the tracks themselves, so this rebuilds them with the frozen pipeline: the
same scores ($RUN/scores/<set>[_<cond>], written by the test chain), the same frozen configurations
($RUN/test/frozen_<cond>.json), the same builders (hnl.trackml.trackbuild.build) and merge model. Nothing is chosen
here. Per block it writes:
  labels_b<K>.parquet     event, row (hit row in the full event), pid, lab_<i> (track id per hit, configuration i)
  particles_b<K>.parquet  the events' particle tables (event column added)
  counts_b<K>.parquet     hnl.trackml.evaluate.event_counts per event and configuration: must equal the test
                          chain's per_event.parquet up to the merging builders' node-dependent last digits
                          (DECISIONS 2026-10-11, unsplit vs block-merged val grid)

    python -m hnl.trackml.labels rebuild --events test --condition full --scores DIR --configs FROZEN.json \
        --merge-model MODEL --out DIR --block K [--block-size 10] --allow-test
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd


def cmd_rebuild(a, cfg) -> None:
    from hnl.trackml import config as CFG
    from hnl.trackml import events as E
    from hnl.trackml import trackbuild as TB
    from hnl.trackml.evaluate import event_counts
    cfgs, names = TB.explicit_configs(a.configs)
    events = CFG.event_set(a.events, cfg, allow_test=a.allow_test)
    if a.block is not None:
        events = events[a.block * a.block_size:(a.block + 1) * a.block_size]
        if not events:
            raise SystemExit(f"block {a.block} of size {a.block_size} is empty")
    model = None
    if any(c[0] == "mwslmerge" for c in cfgs):
        import joblib
        model = joblib.load(a.merge_model)
    sc_all = TB.read_scores(a.scores, events, a.condition)
    labs, parts, counts = [], [], []
    t0 = time.time()
    for ev in E.iter_events(events):
        ev = E.with_condition(ev, a.condition)
        aa, bb, sc = sc_all[ev.event]
        st = TB.EventState(ev, aa, bb, sc, model)
        hid = ev.full_rows + 1
        tab = {"event": np.full(ev.n, ev.event, np.int32), "row": ev.full_rows.astype(np.int32),
               "pid": ev.pid.astype(np.int64)}
        for i, c in enumerate(cfgs):
            lab = np.asarray(TB.build(c, st))
            tab[f"lab_{i}"] = lab.astype(np.int64)
            cnt, _rows = event_counts(ev.pid, lab, ev.particles, hid, None if a.condition == "1gev" else ev.weight)
            counts.append({"event": ev.event, "config": i, **cnt})
        labs.append(pd.DataFrame(tab))
        p = ev.particles.copy()
        if "event" not in p.columns:
            p.insert(0, "event", np.int32(ev.event))
        parts.append(p)
    os.makedirs(a.out, exist_ok=True)
    tag = f"b{a.block}" if a.block is not None else "all"
    for name, df in (("labels", pd.concat(labs, ignore_index=True)), ("particles", pd.concat(parts, ignore_index=True)),
                     ("counts", pd.DataFrame(counts))):
        tmp = os.path.join(a.out, f"{name}_{tag}.parquet.tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, os.path.join(a.out, f"{name}_{tag}.parquet"))
    with open(os.path.join(a.out, f"meta_{tag}.json"), "w") as fh:
        json.dump({"events": [int(e) for e in events], "condition": a.condition, "scores": a.scores,
                   "configs": a.configs, "config_names": names, "merge_model": a.merge_model,
                   "seconds": round(time.time() - t0, 1), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                   "host": os.uname().nodename}, fh, indent=1)
    print(json.dumps({"block": a.block, "events": len(events), "configs": len(cfgs),
                      "seconds": round(time.time() - t0, 1)}), flush=True)


def main(argv=None):
    from hnl.trackml import config as CFG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebuild")
    r.add_argument("--events", required=True)
    r.add_argument("--condition", default="full", choices=["full", "nonoise", "1gev"])
    r.add_argument("--scores", required=True)
    r.add_argument("--configs", required=True, help="frozen configurations (JSON), e.g. $RUN/test/frozen_full.json")
    r.add_argument("--merge-model", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--block", type=int, default=None)
    r.add_argument("--block-size", type=int, default=10)
    r.add_argument("--allow-test", action="store_true")
    a = ap.parse_args(argv)
    cfg = CFG.load(a.config)
    {"rebuild": cmd_rebuild}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
