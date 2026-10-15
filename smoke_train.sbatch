#!/bin/bash
#SBATCH --job-name=tamf-smoke
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=smoke_test/logs/slurm-%j.out

set -euo pipefail

cd ~/Desktop/GPU_access_folder/OakInk2-TaMF

source /opt/sw/miniforge3/etc/profile.d/conda.sh
conda activate oakink2-tamf

DATA="$HOME/Desktop/GPU_access_folder/OakInk2-data"

timeout --signal=INT --kill-after=20s 600s \
python -u -m oakink2_tamf.launch.train \
  --cfg config/obj_embedding.yml \
  --cfg config/obj_pointcloud.yml \
  --cfg config/arch_mdm_l.yml \
  --cfg config/loss_param.yml \
  --cfg config/cet_smoke.yml \
  --data.data_prefix "$DATA" \
  --train.process_range '?(file:smoke_test/train_three.txt)' \
  --val.process_range '?(file:smoke_test/train_three.txt)' \
  --test.process_range '?(file:smoke_test/train_three.txt)' \
  --train.batch_size 1 \
  --train.num_epoch 1 \
  --runtime.device_id 0 \
  --runtime.num_worker 2 \
  --val.val_freq -1 \
  --test.test_freq -1 \
  --exp_id "smoke_raw_${SLURM_JOB_ID}"
