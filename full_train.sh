#!/usr/bin/env bash
#SBATCH --job-name=tamf-full-cet
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=smoke_test/logs/full-%j.log

set -euo pipefail

PROJECT="$HOME/Desktop/GPU_access_folder/OakInk2-TaMF"
cd "$PROJECT"

# Called from slurm-login: submit the job, then monitor its log.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p smoke_test/logs
    JOB_ID=$(sbatch --parsable "$0")
    LOG_FILE="smoke_test/logs/full-${JOB_ID}.log"
    echo "Submitted full-training job ${JOB_ID}"
    echo "Live log: ${LOG_FILE}"
    echo "Ctrl+C stops monitoring only; the Slurm job keeps running."

    while [[ ! -f "$LOG_FILE" ]]; do
        if [[ -z "$(squeue -h -j "$JOB_ID")" ]]; then
            echo "Job ${JOB_ID} has left the queue before its log appeared. Check: sacct -j ${JOB_ID}"
            exit 1
        fi
        sleep 2
    done
    tail -n 50 -F "$LOG_FILE"
    exit 0
fi

# Called inside Slurm: run full training independently of the SSH session.
source /opt/sw/miniforge3/etc/profile.d/conda.sh
conda activate oakink2-tamf
DATA="$HOME/Desktop/GPU_access_folder/OakInk2-data"

for path in \
    "$DATA/anno_preview" \
    "asset/split/train.txt" \
    "config/cet_smoke.yml"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: Missing training prerequisite: $path" >&2
        exit 1
    fi
done

printf 'Job ID: %s\nNode: %s\nPython: %s\nDataset: %s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$(which python)" "$DATA"
nvidia-smi

# From-scratch model, full train split, CET weight 0.1 (experimental).
# Checkpoints are saved at the end of every completed epoch.
python -u -m oakink2_tamf.launch.train \
    --cfg config/data_reverse_segment.yml \
    --cfg config/obj_embedding.yml \
    --cfg config/obj_pointcloud.yml \
    --cfg config/arch_mdm_l.yml \
    --cfg config/loss_param.yml \
    --cfg config/cet_smoke.yml \
    --data.data_prefix "$DATA" \
    --train.process_range '?(file:asset/split/train.txt)' \
    --train.batch_size 1 \
    --train.num_epoch 400 \
    --train.record_freq 1 \
    --runtime.device_id 0 \
    --runtime.num_worker 2 \
    --val.val_freq -1 \
    --test.test_freq -1 \
    --exp_id "full_cet_${SLURM_JOB_ID}" \
    --commit
