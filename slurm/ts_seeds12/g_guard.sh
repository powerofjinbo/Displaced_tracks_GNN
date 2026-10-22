#!/bin/bash
# Seeds 1 and 2 of Canto-L (HNL) and Canto-L (ttbar+HNL) on the generated samples (ledger R20): a model may read the
# events of a set (test, hss, hnl) only when the T2 code, these scripts and the set's records and lists are committed
# unchanged and ledger entry R20 is committed and names every input of the set (as slurm/addendum_g/guard.sh for R14).
# Usage: bash slurm/ts_seeds12/g_guard.sh SET
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
SET=$1
CODE="src/hnl/eval/addendum_g.py src/hnl/eval/addendum_g_summary.py src/hnl/eval/addendum_g_seeds12.py
      src/hnl/eval/evaluate_gnn.py src/hnl/eval/trackbuild_val.py src/hnl/data/acorn_store.py src/hnl/data/convert.py
      src/hnl/data/shards.py results/training_sample_test_selection_seeds12_v1.json
      results/addendum_g_models_seeds12_v1.json slurm/ts_seeds12/g_env.sh slurm/ts_seeds12/g_guard.sh
      slurm/ts_seeds12/g_chain.sh slurm/ts_seeds12/g_acorn.sbatch slurm/ts_seeds12/g_union.sbatch
      slurm/ts_seeds12/g_score.sbatch slurm/ts_seeds12/g_eval.sbatch slurm/ts_seeds12/g_cpu.sbatch"
case $SET in
    test) FILES="results/addendum_g_t1_v1.md results/addendum_g_t1_x1_v1.md splits/generic_iso_test.txt
                 splits/generic_iso_x1_test.txt splits/generic_iso_union_test.txt splits/generic_dense_test.txt"
          WORDS="generic_iso_union_test generic_iso_x1 generic_dense" ;;
    hss)  FILES="results/addendum_g_t1_v1.md splits/hss15_100_all.txt splits/hss40_300_all.txt"
          WORDS="hss15_100 hss40_300" ;;
    hnl)  FILES="results/hnl_points_truth_v1.md splits/hnl_5_50_all.txt splits/hnl_20_200_all.txt
                 splits/hnl_15_500_all.txt splits/hnl_10_100_all.txt splits/hnl_15_200_all.txt"
          WORDS="hnl5_50 hnl20_200 hnl15_500 hnl10_100_ref hnl15_200_ref" ;;
    *)    echo "g_guard: no rule for set '$SET'"; exit 1 ;;
esac
for f in $CODE $FILES docs/feedback/test_use_ledger.md; do
    git ls-files --error-unmatch "$f" > /dev/null 2>&1 && git diff --quiet HEAD -- "$f" \
        || { echo "g_guard: $f is not committed as is"; exit 1; }
done
ENTRY=$(awk '/^### R20\. /{p=1; print; next} p && /^#{2,3} /{exit} p' docs/feedback/test_use_ledger.md)
[ -n "$ENTRY" ] || { echo "g_guard: no ledger entry R20"; exit 1; }
for w in $WORDS; do
    grep -qF -- "$w" <<< "$ENTRY" || { echo "g_guard: ledger entry R20 does not name $w (set $SET)"; exit 1; }
done
echo "g_guard: $SET allowed (ledger R20 committed at $(git log -1 --format=%h -- docs/feedback/test_use_ledger.md))"
