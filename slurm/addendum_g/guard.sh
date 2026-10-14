#!/bin/bash
# Addendum G: a model may read the events of a set other than val (test: the generic_iso union test split and
# generic_dense test; hss: both h -> s s samples; hnl: the five HNL points) only when
#   - the set's T1 records and split lists, the addendum G T2 code and this guard are committed unchanged, and
#   - the ledger entry R14 is committed (docs/feedback/test_use_ledger.md) and names every input of the set.
# Exits non-zero otherwise. Called by every job that runs a model or a builder on a set, and by chain.sh before it
# submits anything. Usage: bash slurm/addendum_g/guard.sh SET
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
SET=$1
CODE="src/hnl/eval/addendum_g.py src/hnl/eval/addendum_g_summary.py src/hnl/eval/addendum_g_readouts.py
      src/hnl/eval/addendum_g_truth.py src/hnl/eval/evaluate_gnn.py src/hnl/eval/trackbuild_val.py
      src/hnl/data/acorn_store.py src/hnl/data/convert.py src/hnl/data/shards.py
      slurm/addendum_g/guard.sh slurm/addendum_g/chain.sh slurm/addendum_g/acorn_infer.sbatch
      slurm/addendum_g/canto_union.sbatch slurm/addendum_g/canto_score.sbatch slurm/addendum_g/evaluate.sbatch
      docs/addendum_g_t2_readouts.md results/addendum_g_zenodo_rv_shape_v1.csv"
case $SET in
    test) FILES="results/addendum_g_t1_v1.md results/addendum_g_t1_x1_v1.md splits/generic_iso_test.txt
                 splits/generic_iso_x1_test.txt splits/generic_iso_union_test.txt splits/generic_dense_test.txt"
          WORDS="generic_iso_union_test generic_iso_x1 generic_dense" ;;
    hss)  FILES="results/addendum_g_t1_v1.md splits/hss15_100_all.txt splits/hss40_300_all.txt"
          WORDS="hss15_100 hss40_300" ;;
    hnl)  FILES="results/hnl_points_truth_v1.md splits/hnl_5_50_all.txt splits/hnl_20_200_all.txt
                 splits/hnl_15_500_all.txt splits/hnl_10_100_all.txt splits/hnl_15_200_all.txt"
          WORDS="hnl5_50 hnl20_200 hnl15_500 hnl10_100_ref hnl15_200_ref" ;;
    *)    echo "guard: no rule for set '$SET'"; exit 1 ;;
esac
for f in $CODE $FILES docs/feedback/test_use_ledger.md; do
    git ls-files --error-unmatch "$f" > /dev/null 2>&1 && git diff --quiet HEAD -- "$f" \
        || { echo "guard: $f is not committed as is"; exit 1; }
done
ENTRY=$(awk '/^### R14\. /{p=1; print; next} p && /^#{2,3} /{exit} p' docs/feedback/test_use_ledger.md)
[ -n "$ENTRY" ] || { echo "guard: no ledger entry R14"; exit 1; }
for w in $WORDS; do
    grep -qF -- "$w" <<< "$ENTRY" || { echo "guard: ledger entry R14 does not name $w (set $SET)"; exit 1; }
done
echo "guard: $SET allowed (ledger R14 committed at $(git log -1 --format=%h -- docs/feedback/test_use_ledger.md))"
