"""Control of the hit-label exchange test on another production: TrackML (truth only, no model output).

tp:  throughput-phase validation events from our tables (hnl.trackml.events), the RANSAC test of label_swap_ransac.py
     (pT >= PTMIN, >= 6 hits, transverse circle, tolerance 1 mm); module = (volume, layer, module); noise hits are
     other hits of the module but never a particle's.
acc: accuracy-phase validation events read from train_1.zip directly (hits, truth, particles CSV): the same RANSAC
     test, and the distance of every pixel hit (volumes 7, 8, 9) from the true crossing (tx, ty, tz) of the particle it
     is labelled with, on modules shared with another particle and on modules the particle crosses alone.

    python scripts/diagnostics/label_swap_trackml.py --part tp|acc --out DIR [--limit N] [--ptmin 2]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_swap_ransac import TOL, ransac  # noqa: E402

ZIP = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/train_1.zip"
PIXEL = (7, 8, 9)


def ransac_counts(x, y, pid, mod, pt, ptmin):
    """Counts of the RANSAC test for one event (as label_swap_ransac.work, summary level)."""
    up, cnt = np.unique(pid[pid != 0], return_counts=True)
    nmod = pd.Series(mod).map(pd.Series(mod).value_counts()).to_numpy()
    c = dict(clean=0, hits_unshared=0, out_unshared=0, hits_shared=0, out_shared=0, out_shared_expl=0)
    for p, n in zip(up, cnt):
        if n < 6 or pt.get(p, 0) < ptmin:
            continue
        h = np.flatnonzero(pid == p)
        cen, R, res = ransac(x[h], y[h], seed=int(p % 100000))
        if not ((res < TOL).sum() >= 5 and (res < TOL).mean() >= 0.6):
            continue
        c["clean"] += 1
        for k, hh in enumerate(h):
            same = np.flatnonzero((mod == mod[hh]) & (pid != p) & (pid != 0))
            if not len(same):
                c["hits_unshared"] += 1
                c["out_unshared"] += int(res[k] >= TOL)
                continue
            c["hits_shared"] += 1
            if res[k] >= TOL:
                c["out_shared"] += 1
                rr = np.abs(np.hypot(x[same] - cen[0], y[same] - cen[1]) - R)
                c["out_shared_expl"] += int(rr.min() < TOL)
    return c


def work_tp(task):
    from hnl.trackml.events import iter_events
    ev = next(iter_events([task["event"]]))
    mod = ev.volume * 10 ** 8 + ev.layer * 10 ** 5 + ev.module
    pt = dict(zip(ev.particles.particle_id.to_numpy(), ev.particles.pt.to_numpy()))
    return ransac_counts(ev.x.astype(float), ev.y.astype(float), ev.pid, mod, pt, task["ptmin"])


def read_csv(z, e, kind):
    return pd.read_csv(io.BytesIO(z.read(f"train_1/event{e:09d}-{kind}.csv")))


def work_acc(task):
    z = zipfile.ZipFile(ZIP)
    e = task["event"]
    h, t, p = read_csv(z, e, "hits"), read_csv(z, e, "truth"), read_csv(z, e, "particles")
    h = h.merge(t, on="hit_id", how="left")
    pid = h.particle_id.to_numpy(np.int64)
    mod = (h.volume_id.to_numpy(np.int64) * 10 ** 8 + h.layer_id.to_numpy(np.int64) * 10 ** 5
           + h.module_id.to_numpy(np.int64))
    pt = dict(zip(p.particle_id.to_numpy(np.int64), np.hypot(p.px.to_numpy(), p.py.to_numpy())))
    x, y = h.x.to_numpy(float), h.y.to_numpy(float)
    c = ransac_counts(x, y, pid, mod, pt, task["ptmin"])
    # truth-position distance of pixel hits, shared vs unshared module
    d = np.sqrt((h.x - h.tx) ** 2 + (h.y - h.ty) ** 2 + (h.z - h.tz) ** 2).to_numpy()
    pix = np.isin(h.volume_id.to_numpy(), PIXEL) & (pid != 0)
    m = pd.DataFrame({"mod": mod, "pid": pid})
    npart = m[m.pid != 0].groupby("mod").pid.nunique()
    shared = pd.Series(mod).map(npart).fillna(0).to_numpy() > 1
    c.update(pix_unshared=int((pix & ~shared).sum()), pix_unshared_far=int((pix & ~shared & (d > 1)).sum()),
             pix_shared=int((pix & shared).sum()), pix_shared_far=int((pix & shared & (d > 1)).sum()),
             pix_d_median_shared=float(np.median(d[pix & shared])) if (pix & shared).any() else float("nan"))
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("tp", "acc"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    split = "split_tp_v1.json" if a.part == "tp" else "split_v1.json"
    evs = sorted(int(e) for e in json.load(open(os.path.join(repo, "configs", "trackml", split)))["val"])
    if a.limit:
        evs = evs[:a.limit]
    fn = work_tp if a.part == "tp" else work_acc
    tasks = [{"event": e, "ptmin": a.ptmin} for e in evs]
    tot = {}
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for c in ex.map(fn, tasks):
            for k, v in c.items():
                if k == "pix_d_median_shared":
                    tot.setdefault(k, []).append(v)
                else:
                    tot[k] = tot.get(k, 0) + v
    if "pix_d_median_shared" in tot:
        tot["pix_d_median_shared"] = float(np.nanmedian(tot["pix_d_median_shared"]))
    tot.update(part=a.part, events=len(evs), ptmin=a.ptmin)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"trackml_{a.part}_summary.json"), "w") as f:
        json.dump(tot, f, indent=1)
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
