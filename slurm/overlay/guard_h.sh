#!/bin/bash
# Addendum H (docs/preregistration_pileup.md section 10, "Blinding on H-val" and H-T2): every job that runs the frozen
# model on overlay events (ids 50,000,000 + 1,000,000 k + host), or evaluates its outputs on them, calls this guard
# first with --grant, stops if it exits non-zero, and exports the grant it prints. Every entry point that runs a network
# or a learned component, or evaluates outputs, on overlay ids refuses them without a matching grant
# (hnl.overlay.grant.check: kind, condition, ids, code state, Slurm job, out directory, launch token). A grant binds the
# code an overlay job runs, not the commit (decision E1, narrowed in the fourth code round;
# hnl.overlay.grant.bound_paths): the git blob hashes at HEAD of every file of src/hnl/overlay, src/hnl/trackml and
# slurm/overlay, of every other src/hnl module in their static import closure and of the configs files they name, and
# the blob hash of the addendum at HEAD (none while it is uncommitted, as for the pilot kinds dry and plumb). A commit
# of paper, docs, results, DECISIONS or of code outside the bound files (src/hnl/eval outside the closure,
# slurm/addendum_g*) leaves running jobs valid; a commit to a bound file (or of the addendum), an uncommitted change to
# one, or an untracked file that python would import in place of a bound module, stops them at their next check. The
# guard and the grants stop accidental runs; a deliberate bypass through direct Python calls (or a SLURM_JOB_ID set by
# hand) stays possible, and every job log records its grant.
#   always   no uncommitted change to a tracked file under src, slurm or configs and no untracked .py file under src
#            (decision D2); the files of CODE committed; every id is an overlay id of the set's hosts (hnl.overlay.blind
#            check-ids; for H-test the evaluated hosts of hnl.overlay.split_h.htest_hosts); for every kind but dry and
#            plumb the addendum is committed unchanged too
#   pilot    (k = 9) kinds dry (truth-made placeholder outputs, which hnl.overlay.blind and hnl.overlay.tracks check)
#            and plumb (decision D6: the frozen model's stages on the pilot as a plumbing test, run twice: the GPU jobs
#            and then the scores again in CPU jobs; no number of any kind is computed from their outputs), both with
#            --out outside the production runs root and tables
#   closure  (k = 1) model and eval in C-full and C-1GeV (V1, and V1-Delta as preregistered); never C-hs
#   hval     (k = 0) until configs/trackml/overlay_version_h_v1.json (overlay_hash) is committed: model in C-full with
#            outputs in the sealed directory RUNS/sealed/hval_full (refused once it is sealed), and kind blind
#            (hnl.overlay.blind, the only evaluation of k = 0) on that directory. After the commit, and only if the H-val
#            meta carries the committed overlay_hash: model in C-1GeV and C-hs and the step-5 evaluation (eval, any
#            condition), under the launch token of `--launch hval` and only while every file pinned at the launch is
#            unchanged; the launch needs the H-val record of decision D7 committed (results/pileup_hp/h_t05_*, the P_P
#            restriction, flags, lost share and descriptive state of the step-5 runs) and pins it; the step-5 outputs
#            are written once (hnl.overlay.blind.write_once). The C-full model run is not repeated.
#   htest    (k = 2) model and eval in C-full, C-1GeV and C-hs, under the launch token of `--launch htest` and only
#            while every pinned file is unchanged; the launch needs ledger entry R15 in docs/feedback/test_use_ledger.md
#            committed and naming H-test and, each with at least 8 hex digits of its sha256, the H-test id list
#            (hnl.overlay.split_h.htest_sha256), overlay_hash, split_h_v1.json, the version record, the addendum, the
#            H-T1 records and the H1 code (hnl.overlay.blind ledger-r15, decision D3); every repo file R15 names is
#            committed unchanged; only the ids of the evaluated hosts pass (never a spare that replaces no dropped host);
#            H1 at a working point is written once
# Kinds (hnl.overlay.grant): model (a network or learned component of the frozen chain: graph, event store, edge
# scores), plumb (the same, pilot only), blind (hnl.overlay.blind evaluate / merge / seal / v2), eval (any other
# evaluation of model outputs: track building, matching, per-target tables, V1, the H1 code), dry (placeholder code
# tests on k = 9).
# Grants: with --grant the guard writes a grant (hnl.overlay.grant write) under RUNS/grants/ (production sets) or under
# the --out directory's parent (pilot) and prints exactly one line on stdout, HNL_H_GRANT=<path>; every other message
# goes to stderr:   G=$(bash slurm/overlay/guard_h.sh ... --grant) || exit 1; export "$G"
# Once: `bash slurm/overlay/guard_h.sh --launch SET` (login node, after every check of SET passes) writes
# RUNS/launch/SET.json exactly once (hnl.overlay.blind write-launch: the token and the sha256 of every file the decisive
# run depends on: CODE, the addendum, the record, every tracked file of slurm/overlay, the configs, the import closure
# of the entry points, for H-test every file R15 names and the R15 text, for H-val the manifest line of RESULTS.md and
# the H-val records of decision D7, results/pileup_hp/h_t05_*) and
# prints HNL_H_LAUNCH=<token> on stdout. `--launch hval` verifies the sealed directory against the manifest sha256 that
# RESULTS.md records (one line 'Addendum H sealed H-val manifest sha256: <64 hex>', decision D4). A second launch is
# refused.
# Usage: bash slurm/overlay/guard_h.sh --set SET --cond full|1gev|hs --kind model|plumb|blind|eval|dry
#                                      --ids ID,ID,...|@FILE [--out DIR] [--grant]
#        bash slurm/overlay/guard_h.sh --launch hval|htest
set -euo pipefail
exec 3>&1 1>&2                  # fd 3: the one line HNL_H_GRANT=... or HNL_H_LAUNCH=...; every message to stderr
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
PROD=/work/submit/jinboz1/hnl/hnl-displaced
RUNS_PROD=/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/runs/overlay_h_v1      # = hnl.overlay.blind.RUNS_H
DERIVED_PROD=/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/derived/overlay_v1
if [ "$REPO" = "$PROD" ]; then            # the production repository: fixed roots
    RUNS=$RUNS_PROD; DERIVED=$DERIVED_PROD
else                                      # a copy of the repository (tests): roots may be redirected
    RUNS=${HNL_H_RUNS:-$RUNS_PROD}; DERIVED=${HNL_H_DERIVED:-$DERIVED_PROD}
fi
cd "$REPO"
die() { echo "guard_h: $*"; exit 1; }
committed() { git ls-files --error-unmatch "$1" > /dev/null 2>&1 && git diff --quiet HEAD -- "$1"; }
need() { local f; for f in "$@"; do committed "$f" || die "$f is not committed as is"; done; }
py() { python -I -m hnl.overlay.blind "$@"; }
SPLIT=configs/trackml/split_h_v1.json
RECORD=configs/trackml/overlay_version_h_v1.json
LEDGER=docs/feedback/test_use_ledger.md
ADDENDUM=docs/preregistration_pileup.md
HT05=results/pileup_hp/h_t05_          # = hnl.overlay.blind.HT05_RECORDS: the H-val records of decision D7
CODE="src/hnl/overlay/blind.py src/hnl/overlay/grant.py slurm/overlay/guard_h.sh slurm/overlay/blind.sbatch
      slurm/overlay/h1.sbatch src/hnl/overlay/split_h.py src/hnl/overlay/conditions.py src/hnl/overlay/tracks.py
      src/hnl/overlay/h1.py src/hnl/overlay/hstats.py src/hnl/overlay/v1.py src/hnl/trackml/evaluate.py
      src/hnl/trackml/trackbuild.py src/hnl/trackml/infer.py src/hnl/trackml/events.py $SPLIT
      configs/trackml/selection_v1.json"

clean_tree() {     # decision D2: nothing uncommitted under src, slurm or configs, no untracked module under src
    local d u
    d=$(git --no-optional-locks status --porcelain --untracked-files=no -- src slurm configs)
    [ -z "$d" ] || die "uncommitted changes under src, slurm or configs: $(echo "$d" | head -5 | tr '\n' ';')"
    u=$(git ls-files --others -- 'src/*.py')
    [ -z "$u" ] || die "untracked python files under src: $(echo "$u" | head -5 | tr '\n' ' ')"
}
record_ok() {      # the overlay version record committed, with an overlay_hash equal to the set's stored one
    committed "$RECORD" || return 1
    py check-version --set "$1" --record "$RECORD" --meta "$DERIVED/meta_$1.json"
}
htest_ok() {       # H-T2 rule: R15 with every sha256 prefix it must carry, and every repo file it names, committed
    need "$LEDGER" "$RECORD"
    local named f
    named=$(py ledger-r15 --ledger "$LEDGER" --split "$SPLIT" --record "$RECORD" --repo "$REPO") \
        || die "ledger entry R15 does not satisfy the H-T2 rule (see above)"
    for f in $named; do committed "$f" || die "$f (named in R15) is not committed as is"; done
}
LAUNCH=""
token_ok() {       # the launch record of the set: HNL_H_LAUNCH carries its token and every pinned file is unchanged
    py check-launch --set "$1" --runs "$RUNS" --repo "$REPO" --token "${HNL_H_LAUNCH:-}" \
        || die "set $1: refused by its launch record $RUNS/launch/$1.json (see above)"
    LAUNCH=$HNL_H_LAUNCH
}

if [ "${1:-}" = --launch ]; then
    SET=${2:-}
    case $SET in hval|htest) ;; *) die "--launch takes hval or htest" ;; esac
    clean_tree
    # shellcheck disable=SC2086
    need $CODE "$ADDENDUM"
    case $SET in
        hval)  record_ok hval || die "no committed overlay_hash record matching the H-val meta ($RECORD)"
               py verify --sealed "$RUNS/sealed/hval_full" --results --repo "$REPO" \
                   || die "the sealed H-val directory $RUNS/sealed/hval_full does not verify against the manifest" \
                          "sha256 that RESULTS.md records"
               D7=$(git ls-files -- "$HT05*")
               [ -n "$D7" ] || die "no H-val record of decision D7 (${HT05}*) is committed: it gives the step-5 runs" \
                                   "their P_P restriction, flags, lost share and descriptive state"
               # shellcheck disable=SC2086
               need $D7 ;;
        htest) htest_ok
               record_ok htest || die "the H-test meta does not carry the committed overlay_hash" ;;
    esac
    TOK=$(py write-launch --set "$SET" --runs "$RUNS" --repo "$REPO" --files "$CODE $ADDENDUM $RECORD") \
        || die "set $SET: no launch record written"
    echo "guard_h: launched $SET ($RUNS/launch/$SET.json, git $(git rev-parse --short HEAD))"
    echo "HNL_H_LAUNCH=$TOK" >&3
    exit 0
fi

SET=""; COND=""; KIND=""; IDS=""; OUT=""; GRANT=0
while [ $# -gt 0 ]; do
    case $1 in
        --set)   SET=$2; shift 2 ;;
        --cond)  COND=$2; shift 2 ;;
        --kind)  KIND=$2; shift 2 ;;
        --ids)   IDS=$2; shift 2 ;;
        --out)   OUT=$2; shift 2 ;;
        --grant) GRANT=1; shift ;;
        *)       die "unknown argument $1" ;;
    esac
done
[ -n "$SET" ] && [ -n "$COND" ] && [ -n "$KIND" ] && [ -n "$IDS" ] || die "--set, --cond, --kind and --ids are required"
case $COND in full|1gev|hs) ;; *) die "no condition '$COND'" ;; esac
case $KIND in model|plumb|blind|eval|dry) ;; *) die "no kind '$KIND'" ;; esac
clean_tree
# shellcheck disable=SC2086
need $CODE
case $KIND in dry|plumb) ;; *) need "$ADDENDUM" ;; esac
py check-ids --set "$SET" --ids "$IDS" --split "$SPLIT" || die "ids outside set $SET"
under() { local p r; p=$(realpath -m "$1"); r=$(realpath -m "$2"); [ "$p" = "$r" ] || [[ "$p" == "$r"/* ]]; }
SEALED=$RUNS/sealed/${SET}_full
case $SET in
    pilot)
        case $KIND in dry|plumb) ;; *) die "on the pilot set (k = 9) kinds dry and plumb only: no evaluation of model" \
                                          "outputs" ;; esac
        [ -n "$OUT" ] || die "kind $KIND needs --out"
        if under "$OUT" "$RUNS_PROD" || under "$OUT" "$RUNS" || under "$OUT" "$DERIVED_PROD"; then
            die "pilot outputs never go under the production runs root or the overlay tables"
        fi ;;
    closure)
        case $KIND in model|eval) ;; *) die "closure (k = 1) takes kinds model and eval" ;; esac
        case $COND in full|1gev) ;; *) die "closure (k = 1) runs in C-full and C-1GeV only" ;; esac ;;
    hval)
        case $KIND in dry|plumb) die "kinds dry and plumb are for the pilot set" ;; esac
        if committed "$RECORD"; then
            record_ok hval || die "the H-val meta does not carry the overlay_hash of the committed $RECORD"
            case $KIND:$COND in
                blind:full) under "${OUT:?--out (the sealed directory)}" "$SEALED" || die "kind blind works in $SEALED" ;;
                model:full) die "the C-full model run on H-val is not repeated after the overlay_hash commit" ;;
                model:*|eval:*) token_ok hval ;;
                *) die "H-val after the overlay_hash commit: model in C-1GeV / C-hs, eval, or blind in C-full" ;;
            esac
        else
            case $KIND:$COND in
                model:full) under "${OUT:?--out (the sealed directory)}" "$SEALED" \
                                || die "until the overlay_hash commit, H-val model outputs go to $SEALED only"
                            [ -e "$SEALED/MANIFEST.sha256" ] && die "$SEALED is sealed: no further model output" ;;
                blind:full) under "${OUT:?--out (the sealed directory)}" "$SEALED" || die "kind blind works in $SEALED" ;;
                *) die "until $RECORD (overlay_hash) is committed, H-val (k = 0) takes the C-full model run and hnl.overlay.blind only" ;;
            esac
        fi ;;
    htest)
        case $KIND in model|eval) ;; *) die "H-test (k = 2) takes kinds model and eval" ;; esac
        htest_ok
        record_ok htest || die "the H-test meta does not carry the overlay_hash of the committed $RECORD"
        token_ok htest ;;
    *)  die "no rule for set '$SET'" ;;
esac
echo "guard_h: $SET $COND $KIND allowed (git $(git rev-parse --short HEAD))"
if [ "$GRANT" = 1 ]; then
    if [ "$SET" = pilot ]; then GDIR=$(dirname "$(realpath -m "$OUT")")/grants; else GDIR=$RUNS/grants; fi
    GP=$(python -I -m hnl.overlay.grant write --set "$SET" --cond "$COND" --kind "$KIND" --ids "$IDS" --dir "$GDIR" \
         ${OUT:+--out "$OUT"} ${LAUNCH:+--launch "$LAUNCH"} --repo "$REPO") || die "no grant written"
    echo "guard_h: grant $GP"
    echo "HNL_H_GRANT=$GP" >&3
fi
