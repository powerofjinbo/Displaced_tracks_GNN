#!/bin/bash
# Wait until Slurm job $1 reaches a terminal state (polling every POLL seconds) and print that state.
J=$1
POLL=${POLL:-30}
until s=$(sacct -n -X -j "$J" -o State 2>/dev/null | head -1 | tr -d ' '); case "$s" in
    COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*) true;; *) false;; esac
do sleep "$POLL"; done
echo "$s"
