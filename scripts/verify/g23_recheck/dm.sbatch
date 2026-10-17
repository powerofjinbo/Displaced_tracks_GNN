#!/bin/bash
#SBATCH -J g23_dmcheck
#SBATCH -p submit
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
#SBATCH --exclude=submit00,submit01,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,submit71,submit72,submit73
cd /tmp/claude-26036/-work-submit-jinboz1/1be031d8-ec01-427d-9202-8adf1f8f92bc/scratchpad/g23_recheck
A=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs
python -I /tmp/claude-26036/-work-submit-jinboz1/1be031d8-ec01-427d-9202-8adf1f8f92bc/scratchpad/g23_recheck/dm.py $A/addendum_g_t4_v1/eval/acorn_g_s{0,1,2}/test $A/addendum_g_t4_v1/eval/canto_g_s{0,1,2}/test $A/addendum_g_v1/eval/canto_l_ttbar_s{0,1,2}/test
