#!/bin/bash
# Addendum G step T4: may a job read the events of SET at PHASE? val: always (code tests; no claim). test:
#   pre   (stages without a metric: acorn (ttbar+G) graphs and edge scores, Canto-L (ttbar+G) union graphs) only when
#         the T4 code, this guard, the T1 records and split lists and the ledger are committed unchanged, and ledger
#         entry R16 is committed and names the test inputs and both retrained models;
#   post  (Canto-L scores, track building, matching, every metric) only when, in addition, the selection file
#         results/addendum_g_selection_v1.json is committed unchanged.
# Exits non-zero otherwise. Called by chain.sh before it submits anything and by every T4 job.
# Usage: bash slurm/addendum_g_t4/guard.sh pre|post val|test
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
PHASE=$1; SET=$2
case $PHASE in pre|post) ;; *) echo "guard: unknown phase '$PHASE'"; exit 1 ;; esac
[ "$SET" = val ] && { echo "guard: val allowed ($PHASE; code test, no claim)"; exit 0; }
[ "$SET" = test ] || { echo "guard: no rule for set '$SET'"; exit 1; }
CODE="src/hnl/eval/addendum_g_t4.py src/hnl/eval/addendum_g.py src/hnl/eval/addendum_g_summary.py
      src/hnl/eval/bootstrap.py src/hnl/eval/evaluate_gnn.py src/hnl/eval/trackbuild_val.py src/hnl/data/acorn_store.py
      src/hnl/data/convert.py src/hnl/data/shards.py src/hnl/graphs/helix_union.py
      src/hnl/graphs/learned_triplet_filter.py src/hnl/train/acorn_runner.py
      slurm/addendum_g_t4/guard.sh slurm/addendum_g_t4/chain.sh slurm/addendum_g_t4/acorn_infer.sbatch
      slurm/addendum_g_t4/canto_union.sbatch slurm/addendum_g_t4/canto_score.sbatch slurm/addendum_g_t4/evaluate.sbatch
      slurm/addendum_g_t4/cpu.sbatch"
FILES="results/addendum_g_t1_v1.md results/addendum_g_t1_x1_v1.md splits/generic_iso_test.txt
       splits/generic_iso_x1_test.txt splits/generic_iso_union_test.txt splits/generic_dense_test.txt
       docs/feedback/test_use_ledger.md"
[ "$PHASE" = post ] && FILES="$FILES results/addendum_g_selection_v1.json"
for f in $CODE $FILES; do
    git ls-files --error-unmatch "$f" > /dev/null 2>&1 && git diff --quiet HEAD -- "$f" \
        || { echo "guard: $f is not committed as is"; exit 1; }
done
ENTRY=$(awk '/^### R16\. /{p=1; print; next} p && /^#{2,3} /{exit} p' docs/feedback/test_use_ledger.md)
[ -n "$ENTRY" ] || { echo "guard: no ledger entry R16"; exit 1; }
for w in generic_iso_union_test generic_iso_x1 generic_dense "acorn (ttbar+G)" "Canto-L (ttbar+G)"; do
    grep -qF -- "$w" <<< "$ENTRY" || { echo "guard: ledger entry R16 does not name $w"; exit 1; }
done
echo "guard: test allowed for $PHASE (ledger R16 at $(git log -1 --format=%h -- docs/feedback/test_use_ledger.md)$([ "$PHASE" = post ] && echo ", selection file at $(git log -1 --format=%h -- results/addendum_g_selection_v1.json)"))"
