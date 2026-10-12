#!/bin/bash
# Apply the acorn patch series (patches/acorn/NNNN-*.patch) to the pinned acorn clone at $HNL_ENV/src/acorn.
# Adapted from the main project's env/apply_patches.sh (gnn-displaced repo dc603528).
#
#   bash env/apply_acorn_patches.sh            git apply --check, apply any missing patches (idempotent), then --verify
#   bash env/apply_acorn_patches.sh --verify   check only: HEAD == pin and working tree == pin + all patches
#   bash env/apply_acorn_patches.sh --revert   remove the patches again (back to the pinned tree)
#
# acorn itself is never edited by hand: bug fixes are new patches here, new methods live in src/hnl.
# A patch that is neither applied nor cleanly applicable stops the script; never force it.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/env/env.sh" || exit 1
ACORN=${ACORN_SRC:-$HNL_ENV/src/acorn}
PIN=99d153418981cba9018e70b476fc7ea0f17bacd4
shopt -s nullglob
PATCHES=("$REPO"/patches/acorn/[0-9][0-9][0-9][0-9]-*.patch)
[ ${#PATCHES[@]} -gt 0 ] || { echo "no patches found in $REPO/patches/acorn" >&2; exit 1; }
(cd "$REPO/patches/acorn" && sha256sum --quiet -c SHA256SUMS) || { echo "patch checksums differ from SHA256SUMS" >&2; exit 1; }

git_acorn() { git -C "$ACORN" "$@"; }

ensure_checkout() {
    [ -e "$ACORN/.git" ] || { echo "$ACORN is not a git checkout (run env/make_env.sh --stage deps)" >&2; exit 1; }
    local head
    head=$(git_acorn rev-parse HEAD)
    if [ "$head" != "$PIN" ]; then
        echo "$ACORN is at $head, expected pin $PIN" >&2
        exit 1
    fi
}

is_applied() { git_acorn apply --reverse --check "$1" 2> /dev/null; }
can_apply() { git_acorn apply --check "$1" 2> /dev/null; }

verify() {
    ensure_checkout
    local tmp expected actual
    tmp=$(mktemp -d)
    # expected tree: the pinned commit plus every patch, built in a scratch index
    GIT_INDEX_FILE=$tmp/expected git_acorn read-tree "$PIN"
    for p in "${PATCHES[@]}"; do
        GIT_INDEX_FILE=$tmp/expected git_acorn apply --cached "$p"
    done
    expected=$(GIT_INDEX_FILE=$tmp/expected git_acorn write-tree)
    # actual tree: the working tree, including untracked files that are not ignored
    GIT_INDEX_FILE=$tmp/actual git_acorn read-tree "$PIN"
    GIT_INDEX_FILE=$tmp/actual git_acorn add -A .
    actual=$(GIT_INDEX_FILE=$tmp/actual git_acorn write-tree)
    rm -rf "$tmp"
    if [ "$expected" != "$actual" ]; then
        echo "FAIL: $ACORN differs from pin + ${#PATCHES[@]} patches:" >&2
        git_acorn diff --stat "$expected" "$actual" >&2
        return 1
    fi
    echo "OK: acorn == ${PIN:0:12} + ${#PATCHES[@]} patches (tree $actual)"
}

apply_all() {
    ensure_checkout
    # --check the whole series on a scratch index first, so a conflict changes nothing
    local tmp
    tmp=$(mktemp -d)
    GIT_INDEX_FILE=$tmp/check git_acorn read-tree "$PIN"
    for p in "${PATCHES[@]}"; do
        if ! is_applied "$p"; then
            GIT_INDEX_FILE=$tmp/check git_acorn apply --cached --check "$p" \
                || { rm -rf "$tmp"; echo "git apply --check failed: $(basename "$p")" >&2; exit 1; }
            GIT_INDEX_FILE=$tmp/check git_acorn apply --cached "$p"
        fi
    done
    rm -rf "$tmp"
    for p in "${PATCHES[@]}"; do
        name=$(basename "$p")
        if is_applied "$p"; then
            echo "already applied  $name"
        elif can_apply "$p"; then
            git_acorn apply "$p"
            echo "applied          $name"
        else
            echo "CONFLICT         $name (neither applied nor applicable)" >&2
            git_acorn apply --check "$p" || true
            exit 1
        fi
    done
    verify
}

revert_all() {
    ensure_checkout
    for ((i = ${#PATCHES[@]} - 1; i >= 0; i--)); do
        p=${PATCHES[$i]}
        name=$(basename "$p")
        if is_applied "$p"; then
            git_acorn apply --reverse "$p"
            echo "reverted         $name"
        else
            echo "not applied      $name"
        fi
    done
}

case "${1:-apply}" in
    apply) apply_all ;;
    --verify) verify ;;
    --revert) revert_all ;;
    *) echo "usage: $0 [--verify|--revert]" >&2; exit 2 ;;
esac
