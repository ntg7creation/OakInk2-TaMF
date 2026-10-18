#!/usr/bin/env bash
# CPU selection/GPU sampling resources and dependencies are supplied by the controller.
set -euo pipefail
state_file=${1:?Expected absolute inference state.json}
conda_base=${2:?Expected existing Conda base}
conda_env=${3:?Expected Conda environment}
controller=${4:?Expected inference controller path}
phase=${5:?Expected select or sample}
source "$conda_base/etc/profile.d/conda.sh"
conda activate "$conda_env"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
if [[ "$phase" == select ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi
# One process owns the allocation and receives B:USR1/TERM directly. Sampling
# never spawns the original sampler's worker pool or any detached child.
exec python -u "$controller" worker --state "$state_file" --phase "$phase"
