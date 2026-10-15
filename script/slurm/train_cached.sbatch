#!/usr/bin/env bash
# No partition, wall-time or GRES assumptions: provide confirmed sbatch options.
set -euo pipefail
: "${REPO_ROOT:?Set the absolute checkout path}"
: "${DATA_ROOT:?Set the OakInk2 dataset root}"
: "${CACHE_ROOT:?Set the persistent cache directory}"
: "${EXPERIMENT_ID:?Set a new experiment ID, or the original ID for resume}"
: "${EXPERIMENT_CONFIG:?Set the complete reviewed experiment YAML path}"
cd "$REPO_ROOT"
if [[ -n "${CONDA_BASE:-}" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${TAMF_CONDA_ENV:-oakink2-tamf}"
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((15000 + ${SLURM_JOB_ID:-0} % 40000))}"
args=(--cfg "$EXPERIMENT_CONFIG" --data.data_prefix "$DATA_ROOT"
      --train.cache_dict_filepath "$CACHE_ROOT/train.pkl" --train.require_verified_cache
      --train.record_freq 1 --runtime.num_worker 0 --runtime.device_id 0
      --val.val_freq -1 --test.test_freq -1 --exp_id "$EXPERIMENT_ID" --commit)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    args+=(--train.resume_checkpoint "$RESUME_CHECKPOINT")
fi
if [[ -n "${SESSION_MAX_SECONDS:-}" ]]; then
    args+=(--runtime.max_seconds "$SESSION_MAX_SECONDS")
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}; MASTER_PORT=$MASTER_PORT"
echo "Resume=${RESUME_CHECKPOINT:-none}; epoch-boundary checkpoints only"
training_pid=""
stop_signal_count=0
forward_stop() {
    stop_signal_count=$((stop_signal_count + 1))
    if [[ -n "$training_pid" ]]; then
        kill -USR1 "$training_pid" 2>/dev/null || true
    fi
}
trap forward_stop USR1
"${TAMF_PYTHON:-python}" -u -m oakink2_tamf.launch.train "${args[@]}" &
training_pid=$!
# wait can return early when the shell receives USR1. Keep waiting for Python,
# preserving its final exit status instead of treating the signal as success.
set +e
while true; do
    before_signal_count=$stop_signal_count
    wait "$training_pid"
    result=$?
    if [[ "$before_signal_count" -eq "$stop_signal_count" ]]; then
        break
    fi
done
exit "$result"
