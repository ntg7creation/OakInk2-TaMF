#!/usr/bin/env bash
# Usage: bash submit_train_chain.sh SESSIONS [--experiment ID] [--after JOB_ID] [--dry-run]
set -euo pipefail
usage="Usage: bash submit_train_chain.sh SESSIONS [--experiment ID] [--after JOB_ID] [--dry-run]"
sessions=${1:-}
[[ "$sessions" =~ ^[1-9][0-9]*$ ]] || { echo "$usage" >&2; exit 2; }
shift
previous=""
dry_run=0
inherited_experiment=${EXPERIMENT_ID:-}
# Only the command line can intentionally override this continuation default.
experiment_id=tamf_cet_001_retry1
while (( $# )); do
    case "$1" in
        --experiment)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "--experiment requires an experiment ID" >&2; exit 2; }
            experiment_id=$2; shift 2 ;;
        --after)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo "--after requires a numeric job ID" >&2; exit 2; }
            previous=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
export REPO_ROOT=${REPO_ROOT:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"}
export EXPERIMENT_ID="$experiment_id"
export TARGET_EPOCHS=${TARGET_EPOCHS:-30}
[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid experiment ID: '$EXPERIMENT_ID'; use letters, digits, underscores or hyphens" >&2; exit 2; }
[[ "$TARGET_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "TARGET_EPOCHS must be positive" >&2; exit 2; }
if [[ -n "$inherited_experiment" && "$inherited_experiment" != "$EXPERIMENT_ID" ]]; then
    echo "Ignoring inherited EXPERIMENT_ID='$inherited_experiment'; selected '$EXPERIMENT_ID'. Use --experiment ID to select an experiment." >&2
fi
cd "$REPO_ROOT" || { echo "Repository directory unavailable: $REPO_ROOT" >&2; exit 1; }
export REPO_ROOT="$PWD"
export TRAIN_CACHE=${TRAIN_CACHE:-"$REPO_ROOT/common/preprocessed_cache_v1/train.pkl"}
TAMF_PYTHON=${TAMF_PYTHON:-python}
experiment_dir="$REPO_ROOT/common/train/$EXPERIMENT_ID"
[[ -d "$experiment_dir" ]] || { echo "Experiment '$EXPERIMENT_ID' does not exist: $experiment_dir. Check --experiment ID; continuation cannot create an experiment." >&2; exit 1; }
[[ -f "$TRAIN_CACHE" ]] || { echo "Existing training cache missing: $TRAIN_CACHE. Restore the existing cache; this launcher never rebuilds it." >&2; exit 1; }
[[ -f "$TRAIN_CACHE.manifest.json" ]] || { echo "Training cache manifest missing: $TRAIN_CACHE.manifest.json. Restore the matching manifest for the existing cache." >&2; exit 1; }

# Read-only preflight, before any scheduler calls or submission directories.
# This checkpoint is informational; train.sbatch selects again at RUN TIME.
if ! checkpoint=$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$TAMF_PYTHON" - "$experiment_dir" <<'PY'
import sys
from pathlib import Path
from oakink2_tamf.util.training_state import resolve_checkpoint

experiment = Path(sys.argv[1])
try:
    checkpoint = resolve_checkpoint("latest", experiment).resolve()
except (OSError, ValueError, KeyError) as exc:
    sys.exit(f"Experiment '{experiment.name}' has no complete, verified training-state checkpoint "
             f"in {experiment / 'save'}: {exc}. Expected state_epoch_*.pt with matching .manifest.json; "
             "check --experiment ID.")
print(checkpoint)
PY
); then
    echo "Checkpoint preflight failed for '$EXPERIMENT_ID'; no jobs submitted. Run in the activated oakink2-tamf environment or set TAMF_PYTHON to its Python executable." >&2
    exit 1
fi
printf 'Experiment ID: %s\nVerified checkpoint (preflight): %s\nCache path: %s\nRepository path: %s\nTarget epoch count: %s\nNumber of sessions: %s\n' \
    "$EXPERIMENT_ID" "$checkpoint" "$TRAIN_CACHE" "$REPO_ROOT" "$TARGET_EPOCHS" "$sessions"
job_name="cet-$EXPERIMENT_ID"
log_dir="$REPO_ROOT/common/cet_jobs/$EXPERIMENT_ID/logs"

if (( ! dry_run )); then
    mkdir -p "$log_dir"
    # Serialize launcher invocations so a second chain sees the first one's IDs.
    exec 8>"$REPO_ROOT/common/cet_jobs/$EXPERIMENT_ID/submission.lock"
    flock -n 8 || { echo "Another launcher is submitting this experiment" >&2; exit 1; }
fi
if [[ -z "$previous" ]]; then
    # Attach to the last existing session, including the current manual run.
    previous=$(squeue --noheader --user="${USER:-$(id -un)}" --name="$job_name" --format=%A | sort -n | tail -n 1 | tr -d '[:space:]')
fi
[[ -z "$previous" || "$previous" =~ ^[0-9]+$ ]] || { echo "Invalid predecessor job ID: $previous" >&2; exit 2; }

for (( session=1; session<=sessions; session++ )); do
    dependency=singleton
    if [[ -n "$previous" ]]; then dependency="afterok:$previous,singleton"; fi
    args=(--parsable --export=ALL --job-name="$job_name" --chdir="$REPO_ROOT"
          --output="$log_dir/allocation-%j.log" --kill-on-invalid-dep=yes
          --dependency="$dependency" "$REPO_ROOT/train.sbatch")
    if (( dry_run )); then
        printf 'DRY RUN: sbatch'; printf ' %q' "${args[@]}"; printf '\n'
        previous="DRY_RUN_JOB_$session"
    else
        # set -e stops submission immediately on a scheduler rejection. Already
        # submitted job IDs are printed, never silently cancelled or resubmitted.
        response=$(sbatch "${args[@]}")
        job_id=${response%%;*}
        [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response; inspect squeue before retrying" >&2; exit 1; }
        echo "Session $session/$sessions: job $job_id, dependency=$dependency"
        previous=$job_id
    fi
done
