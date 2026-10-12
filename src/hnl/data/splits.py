"""Train / val / test splits (brief §5c), written to splits/*.txt.

  * Dropped events (docs/dropped_events.tsv) are removed first.
  * ttbar: 80 / 10 / 10 by event index.
  * 10 GeV 100 mm and 15 GeV 200 mm: ONE joint 80 / 10 / 10 split by event index (event i of both samples shares its
    underlying event; the §2.6 overlap scan found no other overlapping pair, so no further extension is needed).
  * 15 GeV 100 mm: held out entirely as the external test set.
Assignment: numpy.random.default_rng(seed).permutation(N) of the index range 0..N-1 (before dropping), the first
80 % of the permutation -> train, next 10 % -> val, rest -> test; dropped events are then removed from their own
sample only (the partner event of a dropped event stays in its split).

    python -m hnl.data.splits [--seed 0]          (writes splits/; refuses to overwrite different content)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from hnl.data.npz import SAMPLES

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SPLITS_DIR = os.path.join(REPO, "splits")
FRACTIONS = (0.8, 0.1, 0.1)
JOINT = ("hnl10_100", "hnl15_200")
EXTERNAL = "hnl15_100"


def dropped_events(repo: str = REPO) -> dict[str, set[int]]:
    df = pd.read_csv(os.path.join(repo, "docs", "dropped_events.tsv"), sep="\t", comment="#")
    out = {s: set() for s in SAMPLES}
    for s, e in zip(df["sample"], df["event"]):
        out[s].add(int(e))
    return out


def assign(n: int, seed: int) -> dict[str, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(FRACTIONS[0] * n))
    n_val = int(round(FRACTIONS[1] * n))
    return {"train": np.sort(perm[:n_train]), "val": np.sort(perm[n_train:n_train + n_val]),
            "test": np.sort(perm[n_train + n_val:])}


def make_splits(seed: int = 0, repo: str = REPO) -> dict[str, dict[str, np.ndarray]]:
    drop = dropped_events(repo)
    out: dict[str, dict[str, np.ndarray]] = {}
    base = assign(SAMPLES["ttbar"].n_events, seed)
    out["ttbar"] = {k: np.setdiff1d(v, sorted(drop["ttbar"])) for k, v in base.items()}
    n = SAMPLES[JOINT[0]].n_events
    assert all(SAMPLES[s].n_events == n for s in JOINT)
    joint = assign(n, seed + 1)          # a different stream from ttbar's, so the two splits are unrelated
    for s in JOINT:
        out[s] = {k: np.setdiff1d(v, sorted(drop[s])) for k, v in joint.items()}
    keep = np.setdiff1d(np.arange(SAMPLES[EXTERNAL].n_events), sorted(drop[EXTERNAL]))
    out[EXTERNAL] = {"external": keep}
    return out


def write_splits(splits: dict, seed: int, out_dir: str = SPLITS_DIR) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for s, parts in splits.items():
        for name, idx in parts.items():
            path = os.path.join(out_dir, f"{s}_{name}.txt")
            how = (f"joint split of {' + '.join(JOINT)} by event index, permutation seed {seed + 1}" if s in JOINT
                   else f"permutation seed {seed}" if s != EXTERNAL else "all kept events (external test set)")
            header = (f"# {SAMPLES[s].dirname} {name}: {len(idx)} events; hnl.data.splits --seed {seed}: {how}; "
                      f"dropped events removed (docs/dropped_events.tsv)\n")
            body = header + "".join(f"{i}\n" for i in idx)
            if os.path.exists(path):
                if open(path).read() != body:
                    raise RuntimeError(f"{path} exists with different content; refusing to overwrite")
                continue
            with open(path, "w") as f:
                f.write(body)
            written.append(path)
    return written


def read_split(sample: str, name: str, split_dir: str = SPLITS_DIR) -> np.ndarray:
    path = os.path.join(split_dir, f"{sample}_{name}.txt")
    return np.loadtxt(path, dtype=np.int64, comments="#", ndmin=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sp = make_splits(a.seed)
    for p in write_splits(sp, a.seed):
        print("wrote", p)
    for s, parts in sp.items():
        print(s, {k: len(v) for k, v in parts.items()})


if __name__ == "__main__":
    main()
