#!/bin/bash
# Fetch the TrackML-1GeV quickstart events that Liu et al. released with HierarchicalGNN (README "Usage";
# portal.nersc.gov/cfs/m3443/ryanliu/TrackML1GeV/events.tar.gz, 459,017,664 bytes, last modified 2022-11-08) for the
# closure test of our port of their evaluation code (hnl.trackml.port_closure). Third-party data, regenerable: kept
# under $HNL_TMP, not in the repository or on /ceph.
#   sbatch slurm/trackml/port_closure_fetch.sbatch
#SBATCH -J port_closure_fetch
#SBATCH -p submit-gpu-express
#SBATCH --exclude=submit20-1,submit21-1,submit22-1,submit23-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -euo pipefail
DEST=${DEST:-/work/submit/jinboz1/hnl/tmp/port_closure/liu_1gev}
URL=https://portal.nersc.gov/cfs/m3443/ryanliu/TrackML1GeV/events.tar.gz
mkdir -p "$DEST"
cd "$DEST"
echo "host $(hostname)"
if [ ! -s events.tar.gz ] || [ "$(stat -c %s events.tar.gz)" != 459017664 ]; then
    curl -sSfL --retry 3 -o events.tar.gz "$URL"
fi
test "$(stat -c %s events.tar.gz)" = 459017664
sha256sum events.tar.gz | tee events.tar.gz.sha256
if [ ! -f unpacked.done ]; then
    rm -rf ev && mkdir ev
    tar -xzf events.tar.gz -C ev
    touch unpacked.done
fi
find ev -type f | sed 's|^ev/||' | sort > files.txt
wc -l files.txt
head -5 files.txt
du -sh ev
