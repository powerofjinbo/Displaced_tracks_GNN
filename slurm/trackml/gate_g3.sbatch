#!/bin/bash
# Gate G3 (plan section 6.7): builder runtime on builders.gate_g3.events dev events scored by the first-epoch
# checkpoint (score.sbatch dev, CKPT=epoch0 TAG=epoch0): every grid family except the learned merging (no model yet),
# with both merge-cut sets of the hand-cut merging families (builders.merge_cut_sets: H, C), at most 3 min per event
# and configuration on 16 CPUs; plus the exactness of the fast fragment merging against the original
# hnl.trackbuild.merge_fragments on one full event (CHECK_ORIGINAL=0 skips it).
# Both merging passes go through hnl.trackml.seqmerge.merge_exact (exact pre-fit of the candidates, the sequential pass
# after an exact pre-filter; _merge's acceptance loop; the unchanged _merge on the merges): same labels as the
# unfiltered passes. The pre-fits and the any-order candidate search run on HNL_TML_PREFIT_PROCS processes (default
# here the 16 CPUs, one build at a time; the val grid uses 1 per build). The unfiltered sequential original that
# merge_check compares with refits ~3e6 pairs on a TrackML event (~20 min for the any-order original, ~60 min and
# ~25-30 GB for the sequential one), hence 64 GB and the long limit. Diagnostics: hnl.trackml.g3check,
# slurm/trackml/g3_seqmerge.sbatch.
# Usage: [SEED=0] sbatch slurm/trackml/gate_g3.sbatch
#SBATCH -J tml_g3
#SBATCH -p submit
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
S=$(scores_dir dev full epoch0)
require "$S.done"
mkdir -p "$RUN/gates"
export HNL_TML_PREFIT_PROCS=${PREFIT_PROCS:-16}
python -m hnl.trackml.trackbuild timing --scores "$S" --events dev --out "$RUN/gates" \
    $( [ "${CHECK_ORIGINAL:-1}" = 1 ] && echo --check-original )
