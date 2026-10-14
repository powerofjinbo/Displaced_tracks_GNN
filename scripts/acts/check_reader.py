"""Read every produced event with the project's reader (hnl.data.npz.load_event, check=True) and check the format
invariants of docs/brief.md section 2.3 on truth only: hid a permutation, every pid in the particle table, ids exact
in float64, true edges forming one chain per particle with edges between different modules, E = hits - particles
with hits. No model is run.

    python scripts/acts/check_reader.py [--workers N]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from hnl.data import npz  # noqa: E402
from hnl.eval.dense_jet_check import chain_orders  # noqa: E402

GEN = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/generated"
NEW = {"generic_iso": ("generic_iso_v1", 8000), "generic_dense": ("generic_dense_v1", 8000),
       "hss15_100": ("hss15_100_v1", 1500), "hss40_300": ("hss40_300_v1", 1500)}
for k, (d, n) in NEW.items():          # registered here only, for the reader's own checks
    npz.SAMPLES[k] = npz.Sample(k, d, n, "generic", False)


def one(args):
    s, i = args
    ev = npz.load_event(s, i, root=GEN, check=True)
    n = len(ev["x"])
    assert np.array_equal(np.sort(ev["hid"]), np.arange(n)), "hid"
    ids = npz.particle_ids(ev["particles"])
    assert np.isin(ev["pid"], ids).all(), "pid not in table"
    e = ev["layerless_true_edges"]
    assert (ev["pid"][e[0]] == ev["pid"][e[1]]).all(), "edge across particles"
    assert (ev["layers"][e[0]] != ev["layers"][e[1]]).all(), "edge within a module"
    chain_orders(ev["pid"], e[0], e[1])
    assert e.shape[1] == n - len(np.unique(ev["pid"])), "edge count"
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for s, (d, n) in NEW.items():
        with cf.ProcessPoolExecutor(a.workers) as ex:
            hits = list(ex.map(one, [(s, i) for i in range(n)], chunksize=50))
        print(f"{d}: {len(hits)} events read with hnl.data.npz.load_event; all invariants hold; hits per event "
              f"{np.mean(hits):.0f} (min {min(hits)}, max {max(hits)})")


if __name__ == "__main__":
    main()
