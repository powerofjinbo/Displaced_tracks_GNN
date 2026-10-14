# Sourced by slurm/addendum_g_t3/*.sbatch (addendum G step T3, docs/addendum_g_t3_plan.md): arm g, seed $1. The
# Stage-0 helpers of slurm/baseline/_common.sh (environment, node-local scratch, train_stage / best_ckpt / run_fwd) with
# TAG g1, so RUN = runs/baseline_g1/armg_s<S> unless RUN_OVERRIDE is set; the generated samples registered
# (HNL_GENERATED=1); RG stages arm g's event lists (hnl.train.arm_g).
export HNL_GENERATED=1
export TAG=${TAG:-g1}
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh g "$1"
RG="python -m hnl.train.arm_g"
G3="python -m hnl.eval.addendum_g_t3"
