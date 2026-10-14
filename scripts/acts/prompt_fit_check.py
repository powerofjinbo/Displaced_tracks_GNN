"""Fitted |d0| and first-hit radius of prompt high-pT particles, Zenodo against our ACTS v15 simulation (truth only;
docs/hnl_points_spec.md section 8). Separates a detector-level difference of the two simulations from the generator
settings of the HNL closure: the same comparison on ttbar events, where no HNL setting enters.

Prompt high-pT particles: generator particles (barcode generation 0) with table pT > 5 GeV, |eta| < 2.5, >= 7 hits and
>= 3 pixel hits and fitted |d0| < 1 mm (the Zenodo files have no production vertex); with --w-muons only the W muons
of the labeled HNL samples. Fits: hnl.truth.fit_events.fit_event; fit_rms is the rms transverse residual of the
fixed-radius fit (large when hits of other particles sit in the particle's chain).

    python scripts/acts/prompt_fit_check.py --zenodo-sample ttbar --zenodo-split train --n 1000 \
        --ours /work/submit/jinboz1/hnl/acts/validation/ttbar_v15_measured --out OUT.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from hnl.data.npz import load_event  # noqa: E402
from hnl.data.splits import read_split  # noqa: E402
from hnl.truth.fit_events import fit_event  # noqa: E402
from hnl.truth.report import truth_d0  # noqa: E402


def one(args):
    kind, ref = args
    if kind == "zenodo":
        sample, i = ref
        ev = load_event(sample, int(i))
    elif isinstance(ref, tuple):
        from hnl.data.shards import load_event_from
        ev = load_event_from(ref[0], int(ref[1]))
    else:
        with np.load(ref, allow_pickle=False) as f:
            ev = {k: f[k] for k in f.files}
    df, res = fit_event(ev, -1)
    df["d0"] = truth_d0(df)
    sel = (df.gen == 0) & (df.pt > 5) & (df.eta.abs() < 2.5) & (df.n_hits >= 7) & (df.n_pixel_hits >= 3)
    sel &= df.d0.abs() < 1.0
    if WMUON_ONLY:
        sel &= df.label_w
    d = df[sel]
    return d[["pdg", "pt", "eta", "d0", "r_first", "fix_rms"]].assign(src=kind)


WMUON_ONLY = False


def main(argv=None):
    global WMUON_ONLY
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo-sample", default="ttbar")
    ap.add_argument("--zenodo-split", default="train")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ours", required=True, help="directory of npz files or of tar shards (hnl.data.shards)")
    ap.add_argument("--w-muons", action="store_true", help="only the W muons (labeled HNL samples)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args(argv)
    WMUON_ONLY = a.w_muons
    zev = read_split(a.zenodo_sample, a.zenodo_split)[: a.n]
    if glob.glob(os.path.join(a.ours, "events_*.tar")):
        from hnl.data.shards import event_indices_from
        ours = [(a.ours, int(i)) for i in event_indices_from(a.ours)[: a.n]]
    else:
        ours = sorted(glob.glob(os.path.join(a.ours, "*.npz")))[: a.n]
    jobs = [("zenodo", (a.zenodo_sample, i)) for i in zev] + [("ours", f) for f in ours]
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        D = pd.concat(list(ex.map(one, jobs, chunksize=20)), ignore_index=True)
    z, o = D[D.src == "zenodo"], D[D.src == "ours"]
    out = {"n_zenodo": len(z), "n_ours": len(o)}
    for q, f in (("abs_d0", lambda x: x.d0.abs()), ("r_first", lambda x: x.r_first), ("fit_rms", lambda x: x.fix_rms)):
        a1, b1 = f(z), f(o)
        out[q] = {"zenodo_quantiles_16_50_84": np.quantile(a1, [0.16, 0.5, 0.84]).round(4).tolist(),
                  "ours_quantiles_16_50_84": np.quantile(b1, [0.16, 0.5, 0.84]).round(4).tolist(),
                  "ks_p": float(stats.ks_2samp(a1, b1).pvalue)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
