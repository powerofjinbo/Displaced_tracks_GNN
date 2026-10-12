#!/bin/bash
# Feedback round 3, task A2 (val, diagnostic only): time and peak memory of acorn's CCandWalk (hnl.trackbuild.acorn_walk,
# fast path) per configuration on a few val events of a run, each configuration in its own process with a memory cap
# and a time limit, to find which walk-add values are computable. Nothing is selected from this output.
# Usage: env -u OMP_NUM_THREADS sbatch -J fb3_acwalk_probe slurm/feedback/fb3_acwalk_probe.sbatch RUN_DIR "fam,c,m,a;..." N_EVENTS OUT_JSON
#SBATCH -J fb3_acwalk_probe
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
RUN=$1; CFGS=$2; N=$3; OUT=$4
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/s"
trap 'rm -rf "$LOCAL"' EXIT
tar -tf "$RUN/gnn_scores/valset.tar" | sort | grep '^event1' | head -"$N" > "$LOCAL/list.txt"
tar -xf "$RUN/gnn_scores/valset.tar" -C "$LOCAL/s" -T "$LOCAL/list.txt"
python -I - "$LOCAL/s" "$CFGS" "$OUT" <<'PY'
import glob, json, os, resource, subprocess, sys, time
d, cfgs, out = sys.argv[1:]
res = []
code = """
import resource, sys, time
from hnl.trackbuild.acorn_walk import AcornCCWalk
p, fam, c, m, a = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:6])
t0 = time.time(); st = AcornCCWalk.from_path(p); lab = st.labels(c, m, a, guided=fam == 'acwalkg')
print(time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""
for p in sorted(glob.glob(os.path.join(d, "*.pyg.gz"))):
    for cf in cfgs.split(";"):
        f = cf.split(",")
        def lim():
            resource.setrlimit(resource.RLIMIT_AS, (24 << 30, 24 << 30))
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code, p, *f], capture_output=True, text=True, timeout=900,
                               preexec_fn=lim)
            ok = r.returncode == 0
            line = r.stdout.strip().splitlines()[-1] if ok and r.stdout.strip() else ""
            sec, kb = (float(line.split()[0]), int(line.split()[1])) if line else (None, None)
            res.append({"graph": os.path.basename(p), "config": cf, "ok": ok, "seconds": sec, "maxrss_gb": kb / 2**20 if kb else None,
                        "err": "" if ok else r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else f"rc {r.returncode}"})
        except subprocess.TimeoutExpired:
            res.append({"graph": os.path.basename(p), "config": cf, "ok": False, "seconds": None, "maxrss_gb": None, "err": "timeout 900 s"})
        print(json.dumps(res[-1]), flush=True)
json.dump(res, open(out, "w"), indent=1)
PY
