#!/bin/bash
# Unpack the four checksum-verified HNL/ttbar tarballs (brief 3.2): member counts by name, unpack, byte totals,
# then make the unpacked tree read-only. Also records whether this CPU node reaches the internet.
#SBATCH -J unpack
#SBATCH -p submit
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh || exit 1
RAW=$HNL_DATA/raw
OUT=$HNL_DATA/unpacked
echo "== start $(date -u +%FT%TZ) on $(hostname), job ${SLURM_JOB_ID:-none}"
for u in https://pypi.org/simple/ https://download.pytorch.org/whl/cu126/ https://github.com; do
    printf 'internet %-40s %s\n' "$u" "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$u" || echo FAIL)"
done

# expected values from brief 2.2: .npz member count and the sum of .npz file sizes in bytes
declare -A EXP_N=([ttbar_nopileup_10K]=10000 [WmuHNL10GeV100mm_nopileup_10K]=10000
                  [WmuHNL15GeV100mm_nopileup_5K]=5000 [WmuHNL15GeV200mm_nopileup_10K]=10000)
declare -A EXP_B=([ttbar_nopileup_10K]=1663507124 [WmuHNL10GeV100mm_nopileup_10K]=1164020140
                  [WmuHNL15GeV100mm_nopileup_5K]=584559520 [WmuHNL15GeV200mm_nopileup_10K]=1163838496)
NAMES="ttbar_nopileup_10K WmuHNL10GeV100mm_nopileup_10K WmuHNL15GeV100mm_nopileup_5K WmuHNL15GeV200mm_nopileup_10K"

# 1) listing checks, before anything is written
fail=0
for name in $NAMES; do
    f=$RAW/$name.tar.gz
    tar tzf "$f" > "$TMPDIR/members_$name.txt"
    n_npz=$(grep -c '\.npz$' "$TMPDIR/members_$name.txt" || true)
    n_all=$(wc -l < "$TMPDIR/members_$name.txt")
    n_out=$(grep -vc "^$name/" "$TMPDIR/members_$name.txt" || true)       # members outside <name>/
    n_dot=$(grep -c '\.\./' "$TMPDIR/members_$name.txt" || true)           # path traversal
    n_bad=$(grep '\.npz$' "$TMPDIR/members_$name.txt" | sed "s#^$name/##" | grep -vcE '^[0-9]+\.npz$' || true)
    echo "list $name: npz=$n_npz (expected ${EXP_N[$name]}) all_members=$n_all outside_dir=$n_out dotdot=$n_dot nonnumeric_npz=$n_bad"
    grep -v '\.npz$' "$TMPDIR/members_$name.txt" | sed 's/^/    non-npz member: /' | head -5
    # every index 0..N-1 present exactly once
    missing=$(comm -3 <(seq 0 $((EXP_N[$name] - 1)) | sort) \
                      <(grep '\.npz$' "$TMPDIR/members_$name.txt" | sed "s#^$name/##; s#\.npz\$##" | sort) | wc -l)
    echo "    index set 0..$((EXP_N[$name] - 1)) symmetric difference: $missing"
    if [ "$n_npz" != "${EXP_N[$name]}" ] || [ "$n_out" != 0 ] || [ "$n_dot" != 0 ] || [ "$n_bad" != 0 ] || [ "$missing" != 0 ]; then
        echo "    LISTING CHECK FAILED for $name"; fail=1
    fi
done
[ "$fail" = 0 ] || { echo "listing checks failed; nothing unpacked"; exit 1; }

# 2) unpack (refuse to touch an existing directory)
for name in $NAMES; do
    if [ -e "$OUT/$name" ]; then echo "$OUT/$name exists; refusing to overwrite"; exit 1; fi
done
for name in $NAMES; do
    t0=$(date +%s)
    tar xzf "$RAW/$name.tar.gz" -C "$OUT" --no-same-owner
    echo "unpacked $name in $(( $(date +%s) - t0 )) s"
done

# 3) verification: file count and byte sum of .npz files (never du: CephFS directory sizes differ)
for name in $NAMES; do
    read -r n b < <(find "$OUT/$name" -type f -name '*.npz' -printf '%s\n' | awk '{s+=$1} END{print NR, s+0}')
    n_other=$(find "$OUT/$name" -mindepth 1 ! -name '*.npz' | wc -l)
    ok=OK; { [ "$n" = "${EXP_N[$name]}" ] && [ "$b" = "${EXP_B[$name]}" ] && [ "$n_other" = 0 ]; } || ok=MISMATCH
    echo "verify $name: files=$n (expected ${EXP_N[$name]}) bytes=$b (expected ${EXP_B[$name]}) other_entries=$n_other -> $ok"
    [ "$ok" = OK ] || fail=1
done
[ "$fail" = 0 ] || { echo "verification failed; unpacked tree left writable for inspection"; exit 1; }

# 4) read-only from now on
chmod -R a-w "$OUT"
echo "chmod -R a-w $OUT done"
echo "ceph rfiles under $HNL_DATA: $(getfattr --only-values -n ceph.dir.rfiles "$HNL_DATA" 2>/dev/null)"
echo "== end $(date -u +%FT%TZ) ALL OK"
