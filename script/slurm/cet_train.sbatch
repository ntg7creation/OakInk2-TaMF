#!/usr/bin/env bash
# Resources are passed explicitly by cet_jobs.py, from the working server setup.
set -euo pipefail
state_file=${1:?Expected absolute workflow state.json path}
conda_base=${2:?Expected existing Conda base directory}
conda_env=${3:?Expected existing Conda environment name}
controller=${4:?Expected absolute controller script path}
source "$conda_base/etc/profile.d/conda.sh"
conda activate "$conda_env"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 40000))
# exec retains the batch PID: B:USR1 goes to the controller, which forwards it
# to the original trainer's existing epoch-boundary shutdown handler.
exec python -u "$controller" worker --state "$state_file"
