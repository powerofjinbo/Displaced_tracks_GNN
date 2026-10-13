#!/bin/bash
# ONE-TIME test evaluation split by event: track building of one block of the scored test graphs with one of the two
# committed builder configurations (fr: FR* config, llp: LLP-blind config) of the selection file, as in
# slurm/stage1/final_test.sbatch. hnl.eval.trackbuild_val builds each event on its own, so the blocks (contiguous in
# the sorted file order that trackbuild_val uses), concatenated in order by slurm/stage1/final_test_merge.sbatch, give
# the tables of one final_test.sbatch job. Short jobs that fit a busy cluster. Refuses to run once the merge is done.
# Usage: sbatch --array=0-$((N-1)) slurm/stage1/final_test_shard.sbatch SELECTION_JSON PIPELINE SEED fr|llp N
#SBATCH -J s1_ft_shard
#SBATCH -p submit,submit-gpu
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 8
#SBATCH --mem=10G
#SBATCH -t 04:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%A_%a.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
SEL=$1; P=$2; S=$3; WHICH=$4; N=$5; K=$SLURM_ARRAY_TASK_ID
git ls-files --error-unmatch "$SEL" > /dev/null 2>&1 && git diff --quiet HEAD -- "$SEL" \
    || { echo "$SEL is not committed as is: refusing the test evaluation"; exit 1; }
read -r RUN CFG LLP MERGE < <(python -I -c "
import json, sys
s = json.load(open(sys.argv[1]))
d = s['pipelines'][sys.argv[2]]['seeds'][sys.argv[3]]
print(d['run'], d['config'], d['llp_blind_config'], d['merge_model'])" "$SEL" "$P" "$S")
[ -e "$RUN/final_test.done" ] && { echo "test evaluation already done for $RUN: refusing to repeat it"; exit 1; }
case "$WHICH" in fr) C=$CFG ;; llp) C=$LLP ;; *) echo "fr or llp, not $WHICH"; exit 1 ;; esac
OUT=$RUN/trackbuild_test_shards/$WHICH/$(printf 'k%03d_of_%03d' "$K" "$N")
[ -e "$OUT/shard.done" ] && { echo "$OUT done already"; exit 0; }
LOCAL=/tmp/$USER/${SLURM_JOB_ID}
mkdir -p "$LOCAL/t"
trap 'rm -rf "$LOCAL"' EXIT
echo "git $(git rev-parse HEAD); dirty code: $(git status --porcelain -- src slurm | wc -l); host $(hostname); job $SLURM_JOB_ID"
echo "selection $SEL ($(sha256sum "$SEL" | cut -c1-16)); pipeline $P seed $S; run $RUN; $WHICH config $C; block $K of $N"
tar -tf "$RUN/gnn_scores/testset.tar" | grep '\.pyg\.gz$' | LC_ALL=C sort > "$LOCAL/all.txt"
T=$(wc -l < "$LOCAL/all.txt"); B=$(( (T + N - 1) / N ))
sed -n "$((K * B + 1)),$(((K + 1) * B))p" "$LOCAL/all.txt" > "$LOCAL/members.txt"
[ -s "$LOCAL/members.txt" ] || { echo "empty block"; exit 1; }
tar -xf "$RUN/gnn_scores/testset.tar" -C "$LOCAL/t" -T "$LOCAL/members.txt"
echo "$(ls "$LOCAL/t" | wc -l) of $T test graphs"
python -m hnl.eval.trackbuild_val --graphs "$LOCAL/t" --out "$OUT" --config "$C" --merge-model "$MERGE"
cp "$LOCAL/members.txt" "$OUT/members.txt"
touch "$OUT/shard.done"
