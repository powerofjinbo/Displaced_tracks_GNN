#!/bin/bash
# Gate G3 work on the sequential helix fragment merging (hnl.trackml.seqmerge, exact pre-filter; plan section 6.7):
# unit tests, environment facts, dev-event inputs (oracle-noise scores of the scaffold's timing test: the stored
# plan-density graphs of events 1000-1001 and the same recipe for 1002-1004), the identity check against the unchanged
# unfiltered pass on full events, the timing grid, and the HNL val identity check through the HNL builder code.
# Scratch only (W); nothing is a result.
# Steps (STEPS, in order):
#   tests      pytest of src/hnl/trackml/tests/test_seqmerge.py and test_trackbuild.py
#   env        versions, CPU, row-sum padding dependence                                      -> $W/env
#   inputs     graphs (ttbar embedding, base graph at 16.5 pairs/hit) + oracle scores for 1002-1004; 1000-1001 copied
#              from the scaffold's timing test (plan_density_v1/scores_v2)                       -> $W/inputs, $W/scores
#   quick      the filtered pass alone on event $EVENT (bound statistics, time)                  -> $W/ref
#   reference  filtered vs unfiltered on event $EVENT (REF_CONFIG="" = merge_check's input)       -> $W/ref
#   refpair    the same for the cut sets in REF_CUT_LIST (default "H C") on one input           -> $W/ref
#   timing     every walkmergeX / mwslayermergeX / mws2passmergeX configuration on $EVENTS        -> $W/timing
#   profile    per configuration: candidates, kept pairs, sampled acceptance and fit cost (no _merge) -> $W/profile
#   options    exact pass vs tighter windows, any-order pass only, chunked _merge on $OPT_EVENT        -> $W/options
#   anycheck   the any-order pass (exact pre-fit) against all its candidates to _merge, H and C     -> $W/anycheck
#   hnl        HNL val events of $HNL_RUN through hnl.eval.trackbuild_val.evaluate_event           -> $W/hnl
# Usage: STEPS="tests env" sbatch slurm/trackml/g3_seqmerge.sbatch
#        STEPS=reference REF_CONFIG=mwslayermerge,0.2 REF_TAG=mwsl02 sbatch --mem=64G slurm/trackml/g3_seqmerge.sbatch
#SBATCH -J tml_g3seq
#SBATCH -p submit
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -uo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
export OMP_NUM_THREADS=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
W=${W:-/work/submit/jinboz1/hnl/tmp/trackml_g3}
STEPS=${STEPS:-tests env}
EVENT=${EVENT:-1000}
EVENTS=${EVENTS:-1000,1001,1002,1003,1004}
SMOKE=/work/submit/jinboz1/hnl/tmp/trackml_smoke/plan_density_v1
HNL_RUN=${HNL_RUN:-/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_final/l1t/armi_s0}
mkdir -p "$W"
echo "git $(git rev-parse HEAD 2>/dev/null); dirty code: $(git status --porcelain -- src slurm 2>/dev/null | wc -l); host $(hostname); job $SLURM_JOB_ID; cpus $(nproc); mem ${SLURM_MEM_PER_NODE:-?}; steps $STEPS; scratch $W"
rc_all=0
for step in $STEPS; do
    t0=$(date +%s)
    echo "[g3] $(date -u +%T) step $step"
    case $step in
        tests)
            python -m pytest -q -x -p no:cacheprovider src/hnl/trackml/tests/test_seqmerge.py \
                src/hnl/trackml/tests/test_trackbuild.py ;;
        env)
            python -m hnl.trackml.g3check env --out "$W/env" ;;
        inputs)
            mkdir -p "$W/scores" "$W/inputs"
            cp -n "$SMOKE/scores_v2/scores_01000.parquet" "$SMOKE/scores_v2/scores_01000.parquet.json" "$W/scores/"
            python -m hnl.trackml.probe graphs --events 1002,1003,1004 --out "$W/inputs" --densities 12,16.5 \
                --union-at '' --store-graph base --store-density 16.5 && \
            python -m hnl.trackml.infer score --store "$W/inputs/store/valset" --oracle 1.0 --out "$W/inputs/scores" \
                --device cpu && \
            cp "$W"/inputs/scores/scores_01002.parquet* "$W/scores/" ;;
        quick)
            python -m hnl.trackml.g3check reference --scores "$W/scores" --event "$EVENT" --out "$W/ref" \
                --tag "${REF_TAG:-direct}" ${REF_CONFIG:+--config "$REF_CONFIG"} --skip-original \
                --cuts "${REF_CUTS:-H}" --method "${REF_METHOD:-prefit}" ;;
        reference)
            python -m hnl.trackml.g3check reference --scores "$W/scores" --event "$EVENT" --out "$W/ref" \
                --tag "${REF_TAG:-direct}" ${REF_CONFIG:+--config "$REF_CONFIG"} ${UNPINNED:+--unpinned} \
                --cuts "${REF_CUTS:-H}" --method "${REF_METHOD:-prefit}" ;;
        refpair)
            for c in ${REF_CUT_LIST:-H C}; do
                python -m hnl.trackml.g3check reference --scores "$W/scores" --event "$EVENT" --out "$W/ref" \
                    --tag "${REF_TAG:-direct}_$c" ${REF_CONFIG:+--config "$REF_CONFIG"} ${UNPINNED:+--unpinned} \
                    --cuts "$c" --method "${REF_METHOD:-prefit}" || false
            done ;;
        timing)
            python -m hnl.trackml.g3check timing --scores "$W/scores" --events "$EVENTS" --out "$W/timing${TIMING_TAG:-}" \
                --workers "${WORKERS:-5}" ${CHECK_BUILD:+--check-build} ${BUILDERS:+--builders "$BUILDERS"} \
                ${TIMING_CONFIGS:+--configs "$TIMING_CONFIGS"} --cut-sets "${CUT_SETS:-H,C}" --n-proc "${N_PROC:-1}" ;;
        timing_h)
            python -m hnl.trackml.g3check timing --scores "$W/scores" --events "$EVENTS" --out "$W/timing_fullH" \
                --workers 4 --n-proc 4 --cut-sets H ;;
        anycheck)
            python -m hnl.trackml.g3check anycheck --scores "$W/scores" --event "${ANY_EVENT:-1001}" --out "$W/anycheck" \
                --n-proc "${N_PROC:-12}" ;;
        profile)
            python -m hnl.trackml.g3check profile --scores "$W/scores" --events "$EVENTS" --out "$W/profile" \
                --workers "${WORKERS:-5}" ${BUILDERS:+--builders "$BUILDERS"} ;;
        options)
            python -m hnl.trackml.g3check options --scores "$W/scores" --event "${OPT_EVENT:-1001}" --out "$W/options" \
                --configs "${OPT_CONFIGS:-walkmerge,0.1,0.5,1.0;mwslayermerge,0.2;mws2passmerge,0.2,0.05}" \
                ${OPT_WINDOWS:+--windows "$OPT_WINDOWS"} ${OPT_CHUNKS:+--chunks "$OPT_CHUNKS"} ;;
        hnl)
            L=/tmp/$USER/g3_$SLURM_JOB_ID
            mkdir -p "$L/val"
            mapfile -t members < <(tar -tf "$HNL_RUN/gnn_scores/valset.tar" | grep 'pyg.gz$' | head -n "${HNL_EVENTS:-5}")
            tar -xf "$HNL_RUN/gnn_scores/valset.tar" -C "$L/val" "${members[@]}"
            python -m hnl.trackml.g3check hnl --graphs "$L/val" --out "$W/hnl" --max-events "${HNL_EVENTS:-5}"
            rc_h=$?
            rm -rf "$L"
            [ $rc_h -eq 0 ] ;;
        *) echo "unknown step $step"; false ;;
    esac
    rc=$?
    [ $rc -ne 0 ] && rc_all=$rc
    echo "[g3] step $step rc $rc in $(( $(date +%s) - t0 )) s"
done
exit $rc_all
