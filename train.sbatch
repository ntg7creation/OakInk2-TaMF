#!/usr/bin/env bash
#SBATCH --job-name=cet-tamf_cet_001_retry1
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue
#SBATCH --output=common/cet_jobs/tamf_cet_001_retry1/logs/allocation-%j.log

# Continuation only: require an existing complete training-state checkpoint.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-"$HOME/Desktop/GPU_access_folder/OakInk2-TaMF"}
EXPERIMENT_ID=${EXPERIMENT_ID:-tamf_cet_001_retry1}
TARGET_EPOCHS=${TARGET_EPOCHS:-30}
DATA_ROOT=${DATA_ROOT:-"$(dirname -- "$REPO_ROOT")/OakInk2-data"}
TRAIN_CACHE=${TRAIN_CACHE:-"$REPO_ROOT/common/preprocessed_cache_v1/train.pkl"}
CONDA_BASE=${CONDA_BASE:-/opt/sw/miniforge3}
TAMF_PYTHON=${TAMF_PYTHON:-python}
[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid experiment ID: '$EXPERIMENT_ID'" >&2; exit 2; }
[[ "$TARGET_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "TARGET_EPOCHS must be positive" >&2; exit 2; }
cd "$REPO_ROOT" || { echo "Repository directory unavailable: $REPO_ROOT" >&2; exit 1; }
experiment_dir="$REPO_ROOT/common/train/$EXPERIMENT_ID"
[[ -d "$experiment_dir" ]] || { echo "Experiment '$EXPERIMENT_ID' does not exist: $experiment_dir. Check EXPERIMENT_ID; continuation cannot create an experiment." >&2; exit 1; }
[[ -f "$TRAIN_CACHE" ]] || { echo "Existing training cache missing: $TRAIN_CACHE. Restore the existing cache; continuation never rebuilds it." >&2; exit 1; }
[[ -f "$TRAIN_CACHE.manifest.json" ]] || { echo "Training cache manifest missing: $TRAIN_CACHE.manifest.json. Restore the matching manifest for the existing cache." >&2; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TAMF_CONDA_ENV:-oakink2-tamf}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((15000 + ${SLURM_JOB_ID:-0} % 40000))
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Protect the experiment even if this script is submitted outside the launcher.
mkdir -p "common/cet_jobs/$EXPERIMENT_ID"
exec 9>"common/cet_jobs/$EXPERIMENT_ID/training.lock"
flock -n 9 || { echo "Another continuation is training this experiment" >&2; exit 1; }

# Command substitution preserves selection failures as a nonzero batch exit.
# Resolve at RUN TIME, after the predecessor has saved its newest checkpoint.
selection=$("$TAMF_PYTHON" - "$experiment_dir" <<'PY'
import json
import sys
from pathlib import Path
from oakink2_tamf.dataset.cache_util import manifest_path
from oakink2_tamf.util.training_state import resolve_checkpoint

experiment = Path(sys.argv[1])
try:
    checkpoint = resolve_checkpoint("latest", experiment).resolve()
    # resolve_checkpoint already verified this immutable file's size and checksum.
    metadata = json.loads(manifest_path(checkpoint).read_text(encoding="utf-8"))
    epoch = metadata["completed_epoch"]
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Invalid completed epoch in checkpoint manifest")
except (OSError, ValueError, KeyError) as exc:
    sys.exit(f"Cannot continue experiment '{experiment.name}': no usable verified training-state "
             f"checkpoint in {experiment / 'save'}: {exc}. Expected state_epoch_*.pt with matching "
             ".manifest.json; check EXPERIMENT_ID or submit_train_chain.sh --experiment ID.")
print(checkpoint)
print(epoch + 1)
PY
)
checkpoint=${selection%$'\n'*}
next_epoch=${selection##*$'\n'}
echo "Job ${SLURM_JOB_ID:-manual}: checkpoint=$checkpoint; next_epoch=$next_epoch; target=$TARGET_EPOCHS"
if (( next_epoch >= TARGET_EPOCHS )); then
    echo "Epoch target already reached; no training needed."
    exit 0
fi

# exec preserves Python's exit status and delivers batch USR1 directly to the
# existing epoch-boundary stop handler. No model/training code is modified.
exec "$TAMF_PYTHON" -u -m oakink2_tamf.launch.train \
    --cfg config/shenkar_generator_full.yml --cfg config/cet_experiment.yml \
    --data.data_prefix "$DATA_ROOT" --train.cache_dict_filepath "$TRAIN_CACHE" \
    --train.process_range '?(file:asset/split/train.txt)' \
    --train.resume_checkpoint "$checkpoint" \
    --train.num_epoch "$TARGET_EPOCHS" --train.record_freq 1 \
    --runtime.num_worker 0 --runtime.device_id 0 --runtime.max_seconds 13800 \
    --val.val_freq -1 --test.test_freq -1 --exp_id "$EXPERIMENT_ID" --commit
