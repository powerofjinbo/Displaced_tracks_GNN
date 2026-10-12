#!/bin/bash
# Submit one job when this user has a free slot (default bound 64 GPU / 64 CPU jobs queued), polling every POLL
# seconds; prints the job id. KIND=gpu (default, limit MAX_GPU) or KIND=cpu (limit MAX_CPU; CPU = any partition
# without "gpu" in its name). Counts and submits under the project's submission lock (see slurm/baseline/orchestrate.sh);
# exits without submitting if the lock cannot be opened.
# Usage: [KIND=cpu] bash slurm/submit_when_slot.sh [sbatch args...] SCRIPT [script args...]
KIND=${KIND:-gpu}
POLL=${POLL:-120}
# the site sets no per-user limit on queued jobs (2026-10-03); 64 is only a sanity bound
if [ "$KIND" = cpu ]; then MAX=${MAX_CPU:-64}; else MAX=${MAX_GPU:-64}; fi
SUBMIT_LOCK=${SUBMIT_LOCK:-/work/submit/jinboz1/hnl/locks/hnl_submit.lock}
exec 9>> "$SUBMIT_LOCK" || { echo "cannot open the submission lock $SUBMIT_LOCK" >&2; exit 1; }
while true; do
    flock 9 || { sleep "$POLL"; continue; }
    if q=$(squeue -h -u "$USER" -o "%P" 2>/dev/null); then
        n=$(awk -v kind="$KIND" 'NF { if ((kind == "gpu") == ($0 ~ /gpu/)) n++ } END { print n + 0 }' <<< "$q")
        [ "$n" -lt "$MAX" ] && break
    fi
    flock -u 9
    sleep "$POLL"
done
sbatch --parsable "$@"
