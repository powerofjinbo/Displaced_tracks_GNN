"""Attachment oracle (Stage 1 C diagnostic, val only): what could graph-free attachment recover at best?

For a first pass (hnl.trackbuild.attach.first_pass), every particle of a population that the first pass loses is
classified by where its missing hits are (the hits not on its best track): unassigned (no track), on a small track
(< 3 hits: an attach candidate), on another particle's track (owned), on another fragment of itself; and two oracle
decodes are matched: (1) truth attachment within attach's candidate window (every candidate pair whose hit belongs to
the track's majority particle is applied), (2) the same without any window (every free hit of a lost particle goes to
its best track). The gap between (1) and the learned attachment is the model's; the gap between (2) and (1) is the
window's; the rest needs a different building step (owned hits, merges).

    python -m hnl.eval.attach_oracle --graphs DIR --first-pass mwslayermerge --out OUT.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from hnl.eval import matching as M


def _event(path, kind):
    from hnl.eval.trackbuild_val import load_scored
    from hnl.geometry import resolution as RS
    from hnl.trackbuild import attach as AT
    eid, pid, r, phi, z, R, a, b, sc = load_scored(path)
    geo = RS.hit_geometry([int(eid)])[int(eid)]
    lab = AT.first_pass(len(pid), a, b, sc, R, (r, phi, z), kind, geo["layer_id"], geo)
    rows, X = AT.candidates(lab, r, phi, z, a, b, sc, geo=geo)
    fm = M.FastMatch(pid)
    m0 = fm.match(lab)
    # oracle 1: truth attachment inside the window
    lab1 = lab.copy()
    if len(rows):
        y = AT.labels_for(rows, lab, pid)
        for (t, h), ok in zip(rows, y):
            if ok:
                lab1[h] = t
    m1 = fm.match(lab1)
    # oracle 2: every free hit (no track or a track < 3 hits) of a particle goes to its particle's best track
    lab2 = lab.copy()
    trk, cnt = np.unique(lab[lab >= 0], return_counts=True)
    small = set(trk[cnt < 3].tolist())
    free = (lab < 0) | np.isin(lab, list(small))
    best = {}
    on = (lab >= 0) & ~free
    for p_ in np.unique(pid[on]):
        t_, c_ = np.unique(lab[on & (pid == p_)], return_counts=True)
        best[int(p_)] = int(t_[np.argmax(c_)])
    for h in np.flatnonzero(free & (pid != 0)):
        t_ = best.get(int(pid[h]))
        if t_ is not None:
            lab2[h] = t_
    m2 = fm.match(lab2)
    rows_out = []
    for k, p_ in enumerate(fm.particles):
        if p_ == 0:
            continue
        hits = np.flatnonzero(pid == p_)
        on_t = lab[hits]
        tb = np.unique(on_t[on_t >= 0], return_counts=True)
        bt = tb[0][np.argmax(tb[1])] if len(tb[0]) else -2
        miss = hits[on_t != bt]
        other_owner = np.array([lab[h] >= 0 and not free[h] and (pid[lab == lab[h]] == p_).mean() <= 0.5
                                for h in miss], dtype=bool) if len(miss) else np.zeros(0, dtype=bool)
        rows_out.append({"event_id": int(eid), "particle_id": int(p_), "n_hits": len(hits), "found": bool(m0["dm_matched"][k]),
                         "oracle_window": bool(m1["dm_matched"][k]), "oracle_free": bool(m2["dm_matched"][k]),
                         "miss_unassigned": int((lab[miss] < 0).sum()), "miss_small_track": int(np.isin(lab[miss], list(small)).sum()),
                         "miss_owned": int(other_owner.sum()),
                         "miss_own_fragment": int(len(miss) - (lab[miss] < 0).sum() - np.isin(lab[miss], list(small)).sum()
                                                  - other_owner.sum())})
    return pd.DataFrame(rows_out)


def _chunk(args):
    paths, kind = args
    return pd.concat([_event(p, kind) for p in paths], ignore_index=True)


def main(argv=None):
    import concurrent.futures as cf
    from hnl.eval.baseline_val import populations
    from hnl.eval.particles import particle_table
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--first-pass", default="mwslayermerge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--population", default="primary")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args(argv)
    paths = [p for p in sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz"))) if os.path.basename(p)[5] in "24"]
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        d = pd.concat(list(ex.map(_chunk, [(paths[k:k + 10], a.first_pass) for k in range(0, len(paths), 10)])),
                      ignore_index=True)
    info = particle_table(np.unique(d.event_id))
    keys = info.loc[populations(info)[a.population], ["event_id", "particle_id"]]
    d = d.merge(keys, on=["event_id", "particle_id"], how="inner")
    lost = d[~d.found]
    out = {"first_pass": a.first_pass, "population": a.population, "n": int(len(d)),
           "eff_first_pass": float(d.found.mean()), "eff_oracle_window": float(d.oracle_window.mean()),
           "eff_oracle_free": float(d.oracle_free.mean()), "n_lost": int(len(lost)),
           "lost_missing_hits": {k: int(lost[k].sum()) for k in ("miss_unassigned", "miss_small_track", "miss_owned",
                                                                   "miss_own_fragment")}}
    d.to_parquet(os.path.splitext(a.out)[0] + ".parquet", index=False)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
