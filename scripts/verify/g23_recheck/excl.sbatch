#!/bin/bash
#SBATCH --job-name=g23_recheck_excl
#SBATCH --partition=submit
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/work/submit/jinboz1/hnl/logs/g23_recheck/excl_%j.out
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
# Independent recomputation of G2/G3 (addendum G T4), as written (EXCL=-1) and without iso event 1592 (EXCL=1592).
source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh
cd /work/submit/jinboz1/hnl/hnl-displaced
for e in -1 1592; do echo "=== EXCL=$e"; EXCL=$e python -I scripts/verify/g23_recheck/recheck_excl.py; done
