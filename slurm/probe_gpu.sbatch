#!/bin/bash
# Site probe of one A30 GPU node (brief 4.2): driver, CUDA, CPUs/memory, local disks, internet. Read-only.
#SBATCH -J probe_gpu
#SBATCH -p submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
set -u
source /work/submit/jinboz1/hnl/hnl-displaced/env/env.sh || exit 1
echo "== start $(date -u +%FT%TZ) on $(hostname)"
echo "== slurm"; env | grep -E '^SLURM_(JOB_ID|JOB_PARTITION|JOB_NODELIST|CPUS_PER_TASK|CPUS_ON_NODE|MEM_PER_NODE|JOB_GPUS|STEP_GPUS|GPUS_ON_NODE)=' | sort
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
scontrol show job "$SLURM_JOB_ID" | grep -E 'SubmitTime|StartTime|TRES|NumCPUs|MinMemory'
echo "== nvidia-smi"; nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,mig.mode.current,pci.bus_id --format=csv
echo "== cuda dirs"; ls -d /usr/local/cuda* 2>&1; readlink -f /usr/local/cuda
/usr/local/cuda/bin/nvcc --version 2>&1 | tail -2
echo "== cpu/mem"; nproc; free -g; lscpu | grep -E '^Model name|^CPU\(s\)|^Thread|^Socket|^NUMA node\(s\)'
echo "== cgroup"; cg=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup); cat /sys/fs/cgroup${cg}/memory.max /sys/fs/cgroup${cg}/cpuset.cpus.effective 2>&1
echo "== disks"; df -h /tmp /work/submit/jinboz1 /ceph/submit/data/user/j/jinboz1 2>&1; ls -ld /scratch /scratch/submit 2>&1
echo "== internet"
for u in https://pypi.org/simple/ https://download.pytorch.org/whl/cu126/ https://data.pyg.org/whl/ https://github.com https://gitlab.cern.ch https://conda.anaconda.org/conda-forge/; do
    printf '%-44s %s\n' "$u" "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$u" || echo FAIL)"
done
echo "== toolchain"; ldd --version | head -1; gcc --version | head -1; which apptainer
echo "== cvmfs"; ls /cvmfs 2>&1 | head
echo "== end $(date -u +%FT%TZ)"
