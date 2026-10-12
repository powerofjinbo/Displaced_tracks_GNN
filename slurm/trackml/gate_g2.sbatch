#!/bin/bash
# Gate G2 (plan section 6.3) on the first-epoch checkpoint: stitched sector scores against one full-event forward on
# CPU for graph classifier.gate_g2.events val events (at most 0.1 % of edges with |delta score| > 0.01, and the target DM
# efficiency of walkthrough + merging, MWS-layer + merging and CC 0.5 equal within 0.02 points). If it fails, rerun with
# SECTORS=g2_fallback (+-45 deg halos) and report. The event store is built on CPU here (cKDTree base graph); the
# additions of the X1 run were made on the GPU (frnn_cuda), hence --allow-backend-change: pairs at distance ~ r* may
# differ from the training graphs, and both sides of the G2 comparison use the same staged graph.
# Usage: [SEED=0] [SECTORS=inference|g2_fallback] sbatch slurm/trackml/gate_g2.sbatch
#SBATCH -J tml_g2
#SBATCH -p submit
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73   # GTX 1080 nodes: no AVX2 (SIGILL on acorn/polars imports)
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
C=$(first_epoch_ckpt)
N=$(cfg classifier.gate_g2.events)
EV=$(python -I -c "import sys; print(','.join(sys.argv[1].split(',')[:int(sys.argv[2])]))" "$(events val)" "$N")
python -m hnl.trackml.graphs stage --events "$EV" --graph-choice "$RUN/graph/graph_choice.json" \
    --additions "$(adds_dir val)" --dest "$LOCAL/store/valset" --device cpu --allow-backend-change
mkdir -p "$RUN/gates"
python -m hnl.trackml.infer g2 --store "$LOCAL/store/valset" --ckpt "$C" --out "$RUN/gates" --sectors "${SECTORS:-inference}"
