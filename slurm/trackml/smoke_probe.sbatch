#!/bin/bash
# CPU smoke test and sizing of the sectorised Canto-L pipeline on TrackML dev events (hnl.trackml.probe; no trained
# TrackML network, no GPU, no training): chunked learned-support exactness on a real event, graph sizes of the
# ttbar-trained (and an untrained) embedding at the plan's scan densities, the learned-support union, sector tables for
# the inference layout / G2 fallback / training windows, sector consistency checks, and the classifier's peak memory per
# inference sector and per training window (forward / forward + backward, untrained weights).
# Scratch (event-store files) on /work (shared between nodes); the small result tables are copied to
# /ceph/.../external/trackml/runs/smoke_scaffold_v1/<job>/.
# Steps: support (chunked support exactness), graphs (sizes; GRAPH_ARGS adds probe options, e.g. the plan-density store
# "--densities 12,16.5 --union-at '' --store-graph base --store-density 16.5"), memory (CPU peak per sector / window),
# score (oracle-noise scores of the stored dev events, plumbing only), timing (builder grid runtime per configuration on
# one full event and the exactness of the fast fragment merging against the original), report (bootstrap and comparison
# code on that grid; SCORES / G3 / TIMING_CONFIGS='[["cc",0.5],...]' choose the score directory, the output and explicit
# configurations), gridsel (the val-grid code path on these scores: grid with cc and MWS-layer, working points,
# report), mergecheck (fast vs original fragment merging on one full event), lmtiming (the learned-merging family with the ttbar-trained Canto-B merge model, runtime only),
# g4cpu (the torch port of the learned support on the CPU device against the numpy reference on a
# 30 deg wedge of a real event). W=<dir> reuses the scratch of an earlier job (its store/ and scores/).
# Usage: [STEPS="support graphs memory"] [EVENTS=1000,1001] [W=dir] sbatch slurm/trackml/smoke_probe.sbatch
#SBATCH -J tml_smoke
#SBATCH -p submit
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
STEPS=${STEPS:-support graphs memory}
EVENTS=${EVENTS:-1000,1001}
W=${W:-/work/submit/jinboz1/hnl/tmp/trackml_smoke/$SLURM_JOB_ID}
OUT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/smoke_scaffold_v1/$SLURM_JOB_ID
mkdir -p "$W" "$OUT"
export OMP_NUM_THREADS=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
TTBAR_ML=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/baseline_v1/armi_s0/ml/artifacts/best-epoch=20-f1=0.356576.ckpt
TTBAR_GBDT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/baseline_v1/armi_s0/learned_triplet_filter_v2/model.joblib
CANTO_B_MERGE=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_final/l1t/armi_s0/lmerge/model.joblib
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm configs | wc -l); host $(hostname); job $SLURM_JOB_ID; cpus $(nproc); steps $STEPS; events $EVENTS"
if [ -n "${TIMING_CONFIGS:-}" ]; then CFGARG=(--configs "$TIMING_CONFIGS")
else CFGARG=(--builders cc,walkmergeX,mwslayer,mwslayermergeX,mws2passmergeX); fi
copy_out() { cp "$W"/*.csv "$W"/*.json "$OUT"/ 2>/dev/null || true; for d in g3 g3lm g3wp report grid; do [ -d "$W/$d" ] && cp -r "$W/$d" "$OUT"/; done; true; }
trap copy_out EXIT
for step in $STEPS; do
    t0=$(date +%s)
    case $step in
        support) python -m hnl.trackml.probe check-support --event "${EVENTS%%,*}" --out "$W" --radii 0.09,0.105 ;;
        graphs)  eval python -m hnl.trackml.probe graphs --events "$EVENTS" --out "$W" ${GRAPH_ARGS:---random} ;;
        memory)  python -m hnl.trackml.probe memory --store "$W/store" --out "$W" --max-events 1 ;;
        score)   python -m hnl.trackml.infer score --store "$W/store/valset" --oracle 1.0 --out "$W/${SCORES:-scores}" \
                     --device cpu ;;
        timing)  python -m hnl.trackml.trackbuild timing --scores "$W/${SCORES:-scores}" --events "${EVENTS%%,*}" \
                     --max-events 1 --out "$W/${G3:-g3}" --merge-model "$CANTO_B_MERGE" "${CFGARG[@]}" ;;
        mergecheck) python -m hnl.trackml.trackbuild merge-check --scores "$W/${SCORES:-scores}" --events "${EVENTS%%,*}" \
                     --out "$W/${G3:-g3}" ;;
        report)  python -m hnl.trackml.evaluate report --grid "$W/${G3:-g3}" --condition full --out "$W/report" --B 50 ;;
        gridsel) python -m hnl.trackml.trackbuild grid --scores "$W/${SCORES:-scores}" --events "$EVENTS" --out "$W/grid" \
                     --builders cc,mwslayer --workers 2 --chunk 1 && \
                 python -m hnl.trackml.trackbuild select --grid "$W/grid" --out "$W/grid/selection_smoke.json" && \
                 python -m hnl.trackml.evaluate report --grid "$W/grid" --condition full --out "$W/grid/report" --B 50 ;;
        lmtiming) python -m hnl.trackml.trackbuild timing --scores "$W/${SCORES:-scores}" --events "${EVENTS%%,*}" --max-events 1 \
                     --out "$W/g3lm" --builders mwslmerge --merge-model "$CANTO_B_MERGE" ;;
        g4cpu)   python -m hnl.trackml.support_gpu compare --events "${EVENTS%%,*}" --ml-ckpt "$TTBAR_ML" \
                     --r-wide 0.1026 --model "$TTBAR_GBDT" --ops 0.005:5,0.05:3,0.3:1 --out "$W" --device cpu \
                     --wedge-deg 30 --max-triplets 5e5 ;;
        *) echo "unknown step $step"; exit 1 ;;
    esac
    echo "[smoke] step $step done in $(( $(date +%s) - t0 )) s"
done
du -sh "$W"
