#!/bin/bash
# TrackML throughput phase (Zenodo 4730157, D1 / D2 of docs/trackml_plan.md section 1.4), pass 1:
#   1. unit tests of the throughput code (hnl.trackml.throughput, the split, hnl.trackml.calibrate);
#   2. training_part01 and training_part02 streamed over HTTP, nothing stored but the member lists:
#      external/trackml/manifest/<part>.members.tsv (every member: event, kind, offsets, size, md5) and
#      manifest_tp_summary.json (size and md5 of each part against the Zenodo listing). A mismatch stops the job: no
#      split, no conversion from that part;
#   3. configs/trackml/split_tp_v1.json made from the manifest (hnl.trackml.split make-tp, plan section 2.4), then made
#      a second time into a scratch file and compared byte for byte.
#   env -u OMP_NUM_THREADS sbatch slurm/trackml/tp_stream.sbatch
#SBATCH -J tml_tp_stream
#SBATCH -p submit
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
source /work/submit/jinboz1/hnl/hnl-displaced/env/activate.sh || exit 1
cd "$HNL_REPO"
EXT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml
SCR=/work/submit/jinboz1/hnl/tmp/trackml_tp
mkdir -p "$SCR"
echo "git $(git rev-parse HEAD); uncommitted code: $(git status --porcelain -- src slurm configs | wc -l); host $(hostname); job ${SLURM_JOB_ID:-none}; cpus $(nproc); $(date -u +%FT%TZ)"
echo "code sha256: $(sha256sum src/hnl/trackml/throughput.py src/hnl/trackml/split.py src/hnl/trackml/convert.py | awk '{print $1}' | tr '\n' ' ')"
for a in ceph.dir.rbytes ceph.quota.max_bytes; do
    echo "$a /ceph/submit/data/user/j/jinboz1: $(getfattr --only-values -n $a /ceph/submit/data/user/j/jinboz1)"
done
echo "ceph.dir.rbytes $EXT: $(getfattr --only-values -n ceph.dir.rbytes "$EXT")"
python -I -c "from hnl.trackml.convert import storage_check; print('storage', storage_check(5e6))"

OMP_NUM_THREADS=2 python -I -m pytest -q -p no:cacheprovider src/hnl/trackml/tests/test_throughput.py \
    src/hnl/trackml/tests/test_calibrate.py
date -u +%FT%TZ

for p in training_part01 training_part02; do
    python -I -m hnl.trackml.throughput scan --part "$p"
    date -u +%FT%TZ
done

python -I -m hnl.trackml.split make-tp --out configs/trackml/split_tp_v1.json
python -I -m hnl.trackml.split make-tp --out "$SCR/split_tp_v1.regen_${SLURM_JOB_ID:-local}.json"
sha256sum configs/trackml/split_tp_v1.json "$SCR/split_tp_v1.regen_${SLURM_JOB_ID:-local}.json"
cmp configs/trackml/split_tp_v1.json "$SCR/split_tp_v1.regen_${SLURM_JOB_ID:-local}.json" \
    && echo "split_tp_v1.json: the two generations are byte-identical"
ls -l "$EXT/manifest"
date -u +%FT%TZ
