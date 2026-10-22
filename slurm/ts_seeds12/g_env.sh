# Sourced by the slurm/ts_seeds12/g_* scripts: the addendum-G T2 code (hnl.eval.addendum_g) for seeds 1 and 2 of
# Canto-L (HNL) and Canto-L (ttbar+HNL) (ledger R20), with its output root on /work (ADDG_OUT; the hnl_lrt file
# budget) and three seeds of both models (ADDG_TS_SEEDS). In ADDG_OUT, acorn_ii_s1, acorn_ii_s2 and eval/<model>_s<S>
# of the R14 runs are links to runs/addendum_g_v1 (read only); everything else there is written by these jobs.
export HNL_GENERATED=1
export ADDG_OUT=/work/submit/jinboz1/hnl/runs_extra/ts_seeds12/addendum_g
export ADDG_TS_SEEDS=0,1,2
export ADDG_CLEANUP_LOG=/work/submit/jinboz1/hnl/runs_extra/ts_seeds12/cleanup_log.tsv
