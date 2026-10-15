#!/usr/bin/env bash
# Resource requests are intentionally supplied to sbatch after site validation.
set -euo pipefail
: "${REPO_ROOT:?Set the absolute checkout path}"
: "${DATA_ROOT:?Set the OakInk2 dataset root}"
: "${CACHE_ROOT:?Set a persistent cache directory}"
SPLIT="${SPLIT:-train}"
case "$SPLIT" in train|val|test) ;; *) echo "SPLIT must be train, val or test" >&2; exit 2;; esac
cd "$REPO_ROOT"
if [[ -n "${CONDA_BASE:-}" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${TAMF_CONDA_ENV:-oakink2-tamf}"
fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
"${TAMF_PYTHON:-python}" -u -m script.save_cache_dict \
    --data.data_prefix "$DATA_ROOT" \
    --data.process_range "?(file:$REPO_ROOT/asset/split/$SPLIT.txt)" \
    --data.split_name "$SPLIT" \
    --cache.output_filepath "$CACHE_ROOT/$SPLIT.pkl" \
    --cache.resume --commit
