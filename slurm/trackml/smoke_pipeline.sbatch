#!/bin/bash
# End-to-end CPU dry run of the TrackML stage code (no GPU, no training) on accuracy-phase dev events with the
# ttbar-trained Canto-L embedding and support GBDT standing in for the TrackML-trained ones: S2 radii -> valmon-style
# scan (reduced grid) -> choice -> S3 union additions -> event store -> oracle-noise scores (instead of a classifier)
# -> builder grid (both metric families) -> working points -> report. Everything under a scratch run root on /work
# (a copy of configs/trackml/pipeline_v1.yaml with the accuracy split and a reduced grid); nothing is a result.
# Usage: [EVENTS=1000,1001] sbatch slurm/trackml/smoke_pipeline.sbatch
#SBATCH -J tml_dryrun
#SBATCH -p submit
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
EVENTS=${EVENTS:-1000,1001}
W=/work/submit/jinboz1/hnl/tmp/trackml_smoke/dryrun_$SLURM_JOB_ID
RUNS=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs
ML=$RUNS/baseline_v1/armi_s0/ml/artifacts/best-epoch=20-f1=0.356576.ckpt
GBDT=$RUNS/baseline_v1/armi_s0/learned_triplet_filter_v2/model.joblib
mkdir -p "$W/run/graph" "$W/run/scores"
export OMP_NUM_THREADS=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
python -I - "$W" <<'EOF'
import sys, yaml
w = sys.argv[1]
c = yaml.safe_load(open("configs/trackml/pipeline_v1.yaml"))
c["data"]["split"] = "configs/trackml/split_v1.json"
c["run_root"] = w + "/run"
c["graph"].update({"densities": [12], "radii_events": 2, "rw_factors": [1.5], "ks": [1, 5], "taus": [0.005, 0.5],
                   "union_max_pairs_per_hit": 100.0})
c["builders"]["families"] = ["cc", "mwslayer", "mwslayermerge", "walkmergeX"]
yaml.safe_dump(c, open(w + "/pipeline_dryrun.yaml", "w"), sort_keys=False)
EOF
export HNL_TML_CONFIG=$W/pipeline_dryrun.yaml
echo "git $(git rev-parse HEAD); host $(hostname); job $SLURM_JOB_ID; cpus $(nproc); events $EVENTS; scratch $W"
G=python
step() { echo "[dryrun] $(date -u +%T) $*"; }
step radii
$G -m hnl.trackml.graphs radii --events "$EVENTS" --ml-ckpt "$ML" --out "$W/run/graph/radii.json" --device cpu
step scan
$G -m hnl.trackml.graphs scan --events "${EVENTS%%,*}" --ml-ckpt "$ML" --radii "$W/run/graph/radii.json" \
    --ltf-model "$GBDT" --out "$W/run/graph/scan" --device cpu
step choose
$G -m hnl.trackml.graphs choose --scan "$W/run/graph/scan" --out "$W/run/graph/graph_choice.json"
step additions
$G -m hnl.trackml.graphs additions --events "$EVENTS" --graph-choice "$W/run/graph/graph_choice.json" \
    --out "$W/run/additions/dev" --device cpu
step stage
$G -m hnl.trackml.graphs stage --events "$EVENTS" --graph-choice "$W/run/graph/graph_choice.json" \
    --additions "$W/run/additions/dev" --dest "$W/store/valset" --device cpu
step score
$G -m hnl.trackml.infer score --store "$W/store/valset" --oracle 1.0 --out "$W/run/scores/dev" --device cpu
step grid
$G -m hnl.trackml.trackbuild grid --scores "$W/run/scores/dev" --events "$EVENTS" --out "$W/run/grid/full" \
    --builders cc,mwslayer,mwslayermerge --workers 2 --chunk 1
step select
$G -m hnl.trackml.trackbuild select --grid "$W/run/grid/full" --out "$W/run/selection_dryrun.json"
step report
$G -m hnl.trackml.evaluate report --grid "$W/run/grid/full" --condition full --out "$W/run/report" --B 50
step done
du -sh "$W"
