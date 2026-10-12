"""GPU-hour ledger from Slurm accounting (Slurm here does not record GPUs: GPU-h = elapsed x 1 GPU per job).

Collects the job ids recorded by slurm/baseline/orchestrate.sh and slurm/stage1/orchestrate_s1.sh
(<base>/_orchestrator/*.jobs) and any extra ids, and prints one row per job (stage, arm, seed, state, elapsed, GPU-h)
plus totals per stage. Since 2026-10-03 CPU jobs may also run in the GPU partition (on non-A30 nodes), so a job counts
as a GPU job by its type, not by its partition: baseline stages ml_train / ml_infer / gnn_train / gnn_infer /
test_eval and the Stage-1 variant jobs; CPU steps (eval, ceiling_ml, the Stage-1 track-building step *_tb) count 0.
Extra ids (--jobs) are GPU jobs.

    python -m hnl.train.ledger --base RUNS/baseline_v1 [--jobs 123,456]
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess

import pandas as pd

GPU_PARTITIONS = ("submit-gpu", "submit-gpu-express")
GPU_STAGES = ("ml_train", "ml_infer", "gnn_train", "gnn_infer", "test_eval")


def sacct(job_ids: list[str]) -> pd.DataFrame:
    if not job_ids:
        return pd.DataFrame(columns=["JobID", "JobName", "Partition", "State", "ElapsedRaw", "NodeList"])
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(job_ids), "-o",
                          "JobID,JobName,Partition,State,ElapsedRaw,NodeList"], capture_output=True, text=True,
                         check=True).stdout
    rows = [line.split("|") for line in out.strip().splitlines() if line]
    df = pd.DataFrame(rows, columns=["JobID", "JobName", "Partition", "State", "ElapsedRaw", "NodeList"])
    df["ElapsedRaw"] = df.ElapsedRaw.astype(int)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="comma list of orchestrated run roots (baseline_v1, stage1_v1)")
    ap.add_argument("--jobs", default="")
    a = ap.parse_args(argv)
    ids, meta, gpu = [], {}, {}
    for base in a.base.split(","):
        stage1 = "stage1" in os.path.basename(base.rstrip("/"))
        for f in sorted(glob.glob(os.path.join(base, "_orchestrator", "*.jobs"))):
            stem = os.path.basename(f)[:-5]  # baseline: <arm>_s<seed>_<stage>; stage 1: <name>_<arm>_s<seed>[_tb]
            if stage1:
                is_tb = stem.endswith("_tb")
                name, arm, seed = (stem[:-3] if is_tb else stem).rsplit("_", 2)
                stage, is_gpu = (f"s1:{name}:trackbuild", False) if is_tb else (f"s1:{name}", True)
            else:
                arm, seed, stage = stem.split("_", 2)
                is_gpu = stage in GPU_STAGES
            for line in open(f):
                if "submitted" in line:
                    j = line.split()[0]
                    ids.append(j)
                    meta[j] = (arm, seed, stage)
                    gpu[j] = is_gpu
    extra = [j for j in a.jobs.split(",") if j]
    ids += extra
    gpu.update({j: True for j in extra})
    df = sacct(ids)
    df["arm"] = [meta.get(j, ("", "", ""))[0] for j in df.JobID]
    df["seed"] = [meta.get(j, ("", "", ""))[1] for j in df.JobID]
    df["stage"] = [meta.get(j, ("", "", df.JobName[i]))[2] for i, j in enumerate(df.JobID)]
    df["gpu_h"] = df.ElapsedRaw / 3600.0 * df.Partition.isin(GPU_PARTITIONS) * df.JobID.map(gpu).fillna(True).astype(bool)
    pd.set_option("display.width", 200)
    print(df[["JobID", "stage", "arm", "seed", "Partition", "State", "ElapsedRaw", "gpu_h"]].to_string(index=False))
    print(df.groupby("stage").gpu_h.sum().round(3).to_string())
    print(f"total GPU-h {df.gpu_h.sum():.3f}")


if __name__ == "__main__":
    main()
