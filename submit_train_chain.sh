#!/usr/bin/env bash
# Usage: bash submit_train_chain.sh SESSIONS [--after JOB_ID] [--dry-run]
set -euo pipefail
sessions=${1:-}
[[ "$sessions" =~ ^[1-9][0-9]*$ ]] || { echo "Usage: bash submit_train_chain.sh SESSIONS [--after JOB_ID] [--dry-run]" >&2; exit 2; }
shift
previous=""
dry_run=0
while (( $# )); do
    case "$1" in
        --after) previous=${2:?--after requires a job ID}; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
export REPO_ROOT=${REPO_ROOT:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"}
export EXPERIMENT_ID=${EXPERIMENT_ID:-tamf_cet_001_retry1}
export TARGET_EPOCHS=${TARGET_EPOCHS:-30}
[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid experiment ID" >&2; exit 2; }
[[ "$TARGET_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "TARGET_EPOCHS must be positive" >&2; exit 2; }
cd "$REPO_ROOT"
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
