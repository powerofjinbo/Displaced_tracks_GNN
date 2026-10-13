#!/bin/bash
# ONE-TIME test evaluation split by event, final step: checks that the blocks of slurm/stage1/final_test_shard.sbatch
# cover every scored test graph once, concatenates them in block order into RUN/trackbuild_test (FR* config) and
# RUN/trackbuild_test_llpblind (LLP-blind config), then runs the rest of slurm/stage1/final_test.sbatch unchanged (the
# classifier at its e* cut and the per-event dD counts) and marks the run done. Refuses to run twice.
# Usage: sbatch slurm/stage1/final_test_merge.sbatch SELECTION_JSON PIPELINE SEED N_FR N_LLP
#SBATCH -J s1_ft_merge
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
SEL=$1; P=$2; S=$3; NFR=$4; NLLP=$5
git ls-files --error-unmatch "$SEL" > /dev/null 2>&1 && git diff --quiet HEAD -- "$SEL" \
    || { echo "$SEL is not committed as is: refusing the test evaluation"; exit 1; }
read -r RUN CFG LLP CUTE < <(python -I -c "
import json, sys
s = json.load(open(sys.argv[1]))
d = s['pipelines'][sys.argv[2]]['seeds'][sys.argv[3]]
print(d['run'], d['config'], d['llp_blind_config'], f\"{d['cut_e']:.4f}\")" "$SEL" "$P" "$S")
[ -e "$RUN/final_test.done" ] && { echo "test evaluation already done for $RUN: refusing to repeat it"; exit 1; }
LOCAL=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL/t"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID"
echo "selection $SEL ($(sha256sum "$SEL" | cut -c1-16)); pipeline $P seed $S; run $RUN; FR* config $CFG; LLP-blind $LLP; e* cut $CUTE"
tar -tf "$RUN/gnn_scores/testset.tar" | grep '\.pyg\.gz$' | LC_ALL=C sort > "$LOCAL/all.txt"
python -I - "$RUN" "$LOCAL/all.txt" "$CFG" "$LLP" "$NFR" "$NLLP" <<'EOF'
import json, os, sys
import pandas as pd
run, all_txt, cfg, llp, nfr, nllp = sys.argv[1:]
everything = open(all_txt).read().split()
for which, cfg_s, n, dest in (("fr", cfg, int(nfr), "trackbuild_test"), ("llp", llp, int(nllp), "trackbuild_test_llpblind")):
    dirs = [os.path.join(run, "trackbuild_test_shards", which, f"k{k:03d}_of_{n:03d}") for k in range(n)]
    missing = [d for d in dirs if not os.path.exists(os.path.join(d, "shard.done"))]
    if missing:
        sys.exit(f"{which}: blocks not done: {missing}")
    members = [m for d in dirs for m in open(os.path.join(d, "members.txt")).read().split()]
    if members != everything:
        sys.exit(f"{which}: the blocks do not cover the {len(everything)} test graphs once, in order")
    want = [[cfg_s.split(",")[0], *[float(x) for x in cfg_s.split(",")[1:]]]]
    for d in dirs:
        if json.load(open(os.path.join(d, "configs.json"))) != want:
            sys.exit(f"{d}: configuration differs from {want}")
    out = os.path.join(run, dest)
    os.makedirs(out, exist_ok=True)
    for t in ("per_particle", "per_event"):
        pd.concat([pd.read_parquet(os.path.join(d, f"{t}.parquet")) for d in dirs],
                  ignore_index=True).to_parquet(os.path.join(out, f"{t}.parquet"), index=False)
    json.dump(want, open(os.path.join(out, "configs.json"), "w"))
    print(json.dumps({"set": dest, "blocks": n, "graphs": len(members)}))
EOF
tar -xf "$RUN/gnn_scores/testset.tar" -C "$LOCAL/t"
echo "$(ls "$LOCAL/t" | wc -l) test graphs"
python -m hnl.eval.evaluate_gnn --graphs "$LOCAL/t" --out "$RUN/eval_test" --cuts "$CUTE"
python -m hnl.eval.edge_tpr --graphs "$LOCAL/t" --out "$RUN/edge_tpr_test.csv" --cut "$CUTE" \
    --per-event "$RUN/edge_tpr_test_events.parquet"
cp "$SEL" "$RUN/final_selection_used.json"
touch "$RUN/final_test.done"
