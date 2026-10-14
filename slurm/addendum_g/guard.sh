#!/bin/bash
# Addendum G: a model may read the test / hss events only after the T1 commit and the ledger entry R14 are committed
# (docs/feedback/test_use_ledger.md). Exits non-zero otherwise. Usage: bash slurm/addendum_g/guard.sh SET
set -euo pipefail
cd /work/submit/jinboz1/hnl/hnl-displaced
for f in results/addendum_g_t1_v1.md docs/feedback/test_use_ledger.md; do
    git ls-files --error-unmatch "$f" > /dev/null 2>&1 && git diff --quiet HEAD -- "$f" \
        || { echo "guard: $f is not committed as is"; exit 1; }
done
grep -q "^### R14\. " docs/feedback/test_use_ledger.md || { echo "guard: no ledger entry R14"; exit 1; }
echo "guard: $1 allowed (T1 and ledger R14 committed at $(git log -1 --format=%h -- docs/feedback/test_use_ledger.md))"
