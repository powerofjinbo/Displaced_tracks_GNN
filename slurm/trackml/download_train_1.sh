#!/bin/bash
# Download the TrackML accuracy-phase archive train_1.zip once from Zenodo (record 4730167, DOI
# 10.5281/zenodo.4730167, public, no login) into the TrackML data directory. Resumable (curl -C -): rerun after an
# interruption. The file is written as train_1.zip.part and renamed to train_1.zip only when its size matches the
# Zenodo listing; the md5 is checked by the conversion job (slurm/trackml/convert_train_1.sbatch) before any use and
# written next to the zip (train_1.zip.md5) when it matches.
# Runs on the login node (network transfer, no computation):
#   setsid nohup bash slurm/trackml/download_train_1.sh > /work/submit/jinboz1/hnl/logs/trackml_download_train_1.log 2>&1 &
set -euo pipefail
EXT=/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml
URL=https://zenodo.org/api/records/4730167/files/train_1.zip/content
SIZE=15941486907                      # Zenodo API listing, 2026-10-05 UTC
DST=$EXT/train_1.zip
cd "$EXT"
if [ -e "$DST" ]; then echo "$DST exists; nothing to do"; exit 0; fi
# storage guard: the user area must keep >= 15 GB free after the download (docs/trackml_plan.md section 7)
used=$(getfattr --only-values -n ceph.dir.rbytes /ceph/submit/data/user/j/jinboz1)
quota=$(getfattr --only-values -n ceph.quota.max_bytes /ceph/submit/data/user/j/jinboz1)
have=0; [ -e "$DST.part" ] && have=$(stat -c %s "$DST.part")
need=$((SIZE - have))
if [ $((quota - used - need)) -lt 15000000000 ]; then
    echo "storage guard: used $used of $quota bytes, need $need more; stopping"; exit 2
fi
echo "start $(date -u +%FT%TZ) host $(hostname) have $have bytes"
nice -n 10 curl -L --fail --retry 20 --retry-delay 15 --retry-all-errors -C - -sS -o "$DST.part" \
    -w "curl: %{http_code} %{size_download} bytes in %{time_total} s (%{speed_download} B/s)\n" "$URL"
got=$(stat -c %s "$DST.part")
if [ "$got" != "$SIZE" ]; then echo "size $got != $SIZE; keep $DST.part and rerun to resume"; exit 3; fi
mv "$DST.part" "$DST"
echo "done $(date -u +%FT%TZ): $DST $got bytes"
