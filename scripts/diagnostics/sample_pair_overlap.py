"""Positive control for scripts/diagnostics/external_overlap.py: the same shared-collision count (generation-0
underlying-event particles, category "other", table pT > 0.5 GeV, coarse key pdg / pT to 0.01 GeV / eta / phi to 0.01)
between every pair of our four samples, reported per pair of splits, and for events with the same index in both
samples. Truth only: no model output is read. Writes results/sample_pair_overlap_v1.txt.

    sbatch ... --wrap "source env/activate.sh; python scripts/diagnostics/sample_pair_overlap.py"
"""
import glob
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_overlap import DERIVED, REPO, SAMPLES, combine, split_of  # noqa: E402

OUT = os.path.join(REPO, "results", "sample_pair_overlap_v1.txt")


def ue_table(s: str) -> pd.DataFrame:
    sp = split_of(s)
    pf = sorted(glob.glob(os.path.join(DERIVED, "events", "v1", "particles", s, "*.parquet")))
    p = pd.concat([pd.read_parquet(f, columns=["event", "pdg", "px", "py", "pt", "eta", "gen", "category"])
                   for f in pf], ignore_index=True)
    u = p[(p.gen == 0) & (p.category == "other") & (p.pt > 0.5) & p.event.isin(sp)]
    phi = np.arctan2(u.py.to_numpy(), u.px.to_numpy())
    return pd.DataFrame({"event": u.event.to_numpy(), "split": u.event.map(sp).to_numpy(),
                         "k": combine(u.pdg.to_numpy(), np.round(u.pt.to_numpy() * 100),
                                      np.round(u.eta.to_numpy() * 100), np.round(phi * 100)).view(np.int64)})


def main() -> None:
    t0 = time.time()
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists")
    U = {s: ue_table(s) for s in SAMPLES}
    lines = ["# Shared underlying-event particles between our samples (positive control of external_overlap_v1)", "",
             f"Job {os.environ.get('SLURM_JOB_ID', 'local')}, {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; "
             "truth only (scripts/diagnostics/sample_pair_overlap.py). Per event of sample A, the largest number of "
             "coarse-key particles shared with one event of sample B.", ""]
    for a, b in itertools.combinations(SAMPLES, 2):
        m = U[a].merge(U[b], on="k", suffixes=("_a", "_b"))
        c = m.groupby(["event_a", "event_b"]).size().rename("n").reset_index()
        best = c.groupby("event_a").n.max().reindex(U[a].event.unique(), fill_value=0)
        same = c[c.event_a == c.event_b]
        big = c[c.n >= 10]
        sa = U[a].drop_duplicates("event").set_index("event").split
        sb = U[b].drop_duplicates("event").set_index("event").split
        big = big.assign(split_a=big.event_a.map(sa), split_b=big.event_b.map(sb))
        q = np.percentile(best, [50, 90, 99, 100])
        lines.append(f"## {a} vs {b}")
        lines.append(f"- largest shared per {a} event: median {q[0]:.0f}, 90% {q[1]:.0f}, 99% {q[2]:.0f}, max {q[3]:.0f}")
        lines.append(f"- pairs with >= 10 shared: {len(big)}; of them with the same event index: "
                     f"{int((big.event_a == big.event_b).sum())}; by split pair: "
                     f"{big.groupby(['split_a', 'split_b']).size().to_dict()}")
        lines.append(f"- same-index pairs with any shared particle: {len(same)}; median shared among them "
                     f"{same.n.median() if len(same) else 0:.0f}")
        lines.append("")
    lines.append(f"Elapsed {time.time() - t0:.0f} s.")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
