"""hnl.eval.oblique_val with a cut per run: graph- and classifier-stage losses of the primary-population HNL daughters on
val versus |d0| / r_first, with the classifier stage taken at each run's own edge-score cut instead of the fixed
s0 = 0.2 of oblique_val. For classifiers whose scores are calibrated differently (the secondary-particle upweighting
test, docs/secondary_upweight_plan.md) the cut is the one at a common edge false-positive rate (RUN/
edge_tpr_estar_efpr.json "cut"; EFPR 0.0122 there). Population, classes and outputs as hnl.eval.oblique_val.

    python -m hnl.eval.oblique_at_cut --run LABEL=SCORES_DIR:ARM:CUT [--run ...] --out results --name NAME [--workers N]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import pandas as pd

from hnl.eval import oblique_val as OV


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR:ARM:CUT")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--tmp", default=os.environ.get("LOCAL", tempfile.gettempdir()))
    a = ap.parse_args(argv)
    t0 = time.time()
    pop = OV.population()
    parts, cuts = [], {}
    for spec in a.run:
        lab, rest = spec.split("=", 1)
        run, arm, cut = rest.rsplit(":", 2)
        cuts[lab] = float(cut)
        df = OV.run_one(lab, run, arm, pop, a.workers, a.tmp, float(cut))
        if len(df) != len(pop):
            raise SystemExit(f"{lab}: {len(df)} classified, {len(pop)} in the population")
        df["cut"] = float(cut)
        parts.append(df.merge(pop, on=["event_id", "particle_id"], how="left"))
        print(f"{lab}: cut {cut}, {len(df)} particles, {time.time() - t0:.0f} s", flush=True)
    parts = pd.concat(parts, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    parts.to_csv(os.path.join(a.out, f"{a.name}_particles.csv"), index=False)
    summ = OV.summarize(parts)
    summ.to_csv(os.path.join(a.out, f"{a.name}_summary.csv"), index=False)
    meta = {"slurm_job": os.environ.get("SLURM_JOB_ID", "local"), "host": os.uname().nodename, "cuts": cuts,
            "population": int(len(pop)), "runs": a.run, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(a.out, f"{a.name}_meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
