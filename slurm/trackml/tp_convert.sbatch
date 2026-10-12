#!/bin/bash
# TrackML throughput phase (Zenodo 4730157), pass 2 (docs/trackml_plan.md sections 2.1, 6.0): training_part01 and
# training_part02 streamed again and converted to compact tables (hnl.trackml.throughput convert: every member's md5
# against pass 1, the archive md5 against Zenodo before anything is published, at most 1 % failed events) under
# external/trackml/derived/tp_v1/; then
#   * structural verification of both parts (hnl.trackml.throughput verify: sha256, schema, row groups, stored events,
#     csv row counts; no physics value read, so it covers val and test events),
#   * value-level checks on TRAINING events only (hnl.trackml.check_tp: raw csv by range requests against the tables,
#     field sign, every training event's ranks / weights / truth rows / distributions, accuracy-phase training events
#     for comparison). Val and test values are never read or printed.
# Needs configs/trackml/split_tp_v1.json and the manifest of slurm/trackml/tp_stream.sbatch.
#   env -u OMP_NUM_THREADS sbatch slurm/trackml/tp_convert.sbatch
#   env -u OMP_NUM_THREADS sbatch --export=ALL,OVERWRITE=1 slurm/trackml/tp_convert.sbatch   (redo the shards)
#SBATCH -J tml_tp_convert
#SBATCH -p submit
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
EXT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml
ROOT=$EXT/derived/tp_v1
NP=$(env -u OMP_NUM_THREADS nproc)
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm configs | wc -l); host $(hostname); job ${SLURM_JOB_ID:-none}; cpus $NP; $(date -u +%FT%TZ)"
echo "code sha256: $(sha256sum src/hnl/trackml/throughput.py src/hnl/trackml/convert.py src/hnl/trackml/check_tp.py | awk '{print $1}' | tr '\n' ' ')"
echo "split: $(sha256sum configs/trackml/split_tp_v1.json)"
for a in ceph.dir.rbytes ceph.quota.max_bytes; do
    echo "$a /ceph/submit/data/user/j/jinboz1: $(getfattr --only-values -n $a /ceph/submit/data/user/j/jinboz1)"
done
echo "ceph.dir.rbytes $EXT: $(getfattr --only-values -n ceph.dir.rbytes "$EXT")"

for p in training_part01 training_part02; do
    OMP_NUM_THREADS=1 python -I -m hnl.trackml.throughput convert --part "$p" --nproc "$NP" ${OVERWRITE:+--overwrite}
    date -u +%FT%TZ
done
for p in training_part01 training_part02; do
    OMP_NUM_THREADS=4 python -I -m hnl.trackml.throughput verify --part "$p" --out "$ROOT/verify_$p.json"
done
date -u +%FT%TZ
OMP_NUM_THREADS=4 python -I -m hnl.trackml.check_tp --out "$ROOT/check_tp.json"
date -u +%FT%TZ
du -sb "$ROOT"/* "$EXT"
for a in ceph.dir.rbytes ceph.quota.max_bytes; do
    echo "$a /ceph/submit/data/user/j/jinboz1: $(getfattr --only-values -n $a /ceph/submit/data/user/j/jinboz1)"
done
echo "ceph.dir.rbytes $EXT: $(getfattr --only-values -n ceph.dir.rbytes "$EXT")"
