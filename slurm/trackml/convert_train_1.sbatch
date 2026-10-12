#!/bin/bash
# TrackML train_1 (accuracy phase, Zenodo 10.5281/zenodo.4730167) -> compact tables (docs/trackml_plan.md 6.0).
#   1. size and md5 of train_1.zip against the Zenodo listing; writes train_1.zip.md5 when they match
#   2. hnl.trackml.convert convert: every event, read in place from the zip (nothing unpacked), into
#      external/trackml/derived/v1/{hits,particles,truth}/train_1/shard_<first>.parquet (+ meta, dq)
#   3. hnl.trackml.convert verify: structural check of the shards against the meta (sha256, row groups, events)
#   4. hnl.trackml.check_convert: raw csv vs tables on dev / training events, the 100 dev events against the
#      feasibility study, field sign, disk usage (no test event is read)
#   sbatch slurm/trackml/convert_train_1.sbatch       (download first: slurm/trackml/download_train_1.sh)
#   sbatch --export=ALL,OVERWRITE=1 slurm/trackml/convert_train_1.sbatch    (redo an existing conversion)
#SBATCH -J tml_convert
#SBATCH -p submit
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
EXT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml
ZIP=$EXT/train_1.zip
SIZE=15941486907
MD5=c1aa5e944b2a07942e8f6c3613497478          # Zenodo API listing of record 4730167
NP=$(nproc)
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm configs | wc -l); host $(hostname); job ${SLURM_JOB_ID:-none}; cpus $NP; $(date -u +%FT%TZ)"
echo "code sha256: $(sha256sum src/hnl/trackml/convert.py src/hnl/trackml/check_convert.py | awk '{print $1}' | tr '\n' ' ')"

got=$(stat -c %s "$ZIP")
[ "$got" = "$SIZE" ] || { echo "size $got != $SIZE"; exit 1; }
if [ -e "$ZIP.md5" ]; then
    echo "md5 already verified: $(cat "$ZIP.md5")"
else
    t0=$(date +%s)
    m=$(md5sum "$ZIP" | awk '{print $1}')
    echo "md5 $m ($(( $(date +%s) - t0 )) s)"
    [ "$m" = "$MD5" ] || { echo "md5 MISMATCH: $m != $MD5"; exit 1; }
    echo "$m  train_1.zip" > "$ZIP.md5"
fi
date -u +%FT%TZ
OMP_NUM_THREADS=1 python -I -m hnl.trackml.convert convert --source "$ZIP" --archive train_1 --nproc "$NP" ${OVERWRITE:+--overwrite}
date -u +%FT%TZ
OMP_NUM_THREADS=4 python -I -m hnl.trackml.convert verify --archive train_1
date -u +%FT%TZ
OMP_NUM_THREADS=4 python -I -m hnl.trackml.check_convert --out "$EXT/derived/v1/check_train_1.json"
date -u +%FT%TZ
du -sb "$EXT/derived/v1"/* "$EXT"
