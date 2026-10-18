# Detached, resumable CET-model inference

Run from Shenkar in `oakink2-tamf`. Training, losses, architecture, the original
sampler and existing outputs/weights are untouched.

## Prepare/reuse the cache separately

The default source is **`common/preprocessed_cache_v1/test.pkl`**, with its
matching `.manifest.json`. There is no fallback to `train.pkl`. Cache split,
sequence membership, preprocessing identity, size and SHA-256 are checked.
Use `--split train` explicitly to use the existing training cache instead.

If test preprocessing is missing, submit the existing resumable **CPU** tool:

```bash
cd /home/ela.nitai/Desktop/GPU_access_folder/OakInk2-TaMF
conda activate /home/ela.nitai/.conda/envs/oakink2-tamf
export REPO_ROOT="$PWD"
export DATA_ROOT=/home/ela.nitai/Desktop/GPU_access_folder/OakInk2-data
export CACHE_ROOT="$PWD/common/preprocessed_cache_v1"
export SPLIT=test
export CONDA_BASE=/opt/sw/miniforge3
export TAMF_CONDA_ENV=/home/ela.nitai/.conda/envs/oakink2-tamf
mkdir -p common/inference_jobs/cache_logs
sbatch --partition=gpu --cpus-per-task=2 --mem=16G --time=04:00:00 \
  --output="$PWD/common/inference_jobs/cache_logs/test-%j.log" \
  --export=ALL script/slurm/preprocess.sbatch
```

No GPU is requested; override the CPU partition if required by site policy.
Rerun after interruption: `--cache.resume` reuses completed sequence work and
compatible final caches. Wait for successful publication. For a sequence-only
smoke cache, use the existing
`python -m script.save_cache_dict` with a small `--data.process_range` split file,
`--data.split_name test`, a separate `--cache.output_filepath`, and
`--cache.resume --commit`, inside a CPU allocation.

## Start with a bounded smoke run

```bash
python script/slurm/cet_inference_jobs.py start \
  --experiment tamf_cet_epoch11_smoke \
  --max-motions 3 --strides-per-motion 2 --seed 23 \
  --walltime 00:20:00
```

Add `--dry-run` to preview without submitting. This example schedules at most
six samples. Source parsing runs once in CPU selection; a small cache reduces
that preparation cost too.
Use `--cache /path/to/test.pkl` and, when needed, `--split-file /path/to/test.txt`
to reuse a smaller compatible cache.

Normal research submission, after the smoke succeeds:

```bash
python script/slurm/cet_inference_jobs.py start
```

Defaults: **`tamf_cet_epoch11`**, checkpoint
**`common/train/tamf_cet_001_retry1/save/model_0011.pt`**, test split, two temporal
variants per motion, seed 0. `start` submits immediately unless `--dry-run` is
present. CPU selection is followed by a dependent GPU job; both survive SSH
disconnects.

Overrides include `--checkpoint`, `--model-config`, `--output-dir`, `--split`,
`--cache`, `--split-file`, `--seed`, `--max-motions`, `--strides-per-motion all`,
`--partition`, `--selection-partition`, `--gres`, `--mem`, `--cpus`, `--walltime`,
`--selection-walltime`, `--conda-base`, `--conda-env`, and `--max-allocations`.
GPU defaults are **gpu / shard:1 / 16G / 2 CPUs / 4 hours**, with device **cuda:0**.
CPU selection defaults to 16G/2 CPUs/30 minutes on the same partition, **no GRES**.
Settings are persisted. Inherited shell `EXPERIMENT_ID` values are ignored.

## What two variants means

A motion is `(sequence, primitive occurrence ID, hand side)`. The toolkit suffixes
each annotated interval (`scoop:00`, `scoop:01`, etc.); action labels alone are
never grouping keys. Preprocessing adapts its initial 12-frame gap to the
16–160-frame limits and emits interleaved offsets, identified by saved frame IDs.

The default selects the first and last available offsets for each motion; other
positive limits spread ranks evenly, and a limit of one selects the middle.
`all` selects every source cache entry in original order when no motion limit is
specified. This reproduces the original **sample population**, not its unseeded
multi-worker random-number sequence. No diffusion timesteps, sampling equations,
training striding, or dataset processing are changed.

With `--max-motions`, motion selection round-robins action categories with seeded
within-category ranks. Sampling covers the first selected variant of each motion
before generating second variants. Each selected cached condition keeps its own
text, shapes, object trajectories, side, frame IDs and mask. One diffusion draw
is generated per variant, with a stable per-sample seed.

For fair comparisons, reuse the exact plan with different weights:

```bash
python script/slurm/cet_inference_jobs.py start \
  --experiment original_same_selection \
  --checkpoint common/train/main__remastered/save/model_0399.pt \
  --selection-manifest common/inference_jobs/tamf_cet_epoch11/output/selection/selection.json
```

Reuse requires the same cache/split; omit stride/max-motion/seed overrides.
IDs and seeds stay identical. Resume rejects changed checkpoints/configuration;
different weights require a new experiment.

## Monitor, stop, resume

```bash
python script/slurm/cet_inference_jobs.py status --experiment tamf_cet_epoch11
tail -F common/inference_jobs/tamf_cet_epoch11/logs/inference.log
python script/slurm/cet_inference_jobs.py stop --experiment tamf_cet_epoch11
python script/slurm/cet_inference_jobs.py resume --experiment tamf_cet_epoch11
```

`status` reads Slurm and validated commits independently: pending/running/
completed/failed/interrupted, motion/sample counts, elapsed time, throughput and
ETA. No model is initialized. Stopping `tail` does not stop Slurm.

`stop` marks the run stopped, cancels pending jobs and sends running batches
USR1. The current sample finishes before stopping. `stop --force` cancels the
allocation; an unfinished sample is retried from its saved seed on resume.
USR1 is requested 120 seconds before the Slurm limit. The sampler also stops
starting new samples before its time budget expires. One `afterok` successor is
submitted only after successful per-sample progress, without a stop marker, and
within the allocation cap (100 GPU allocations by default). No-progress/hard
termination/failure requires explicit resume; failures never create a retry storm.

Locks/singleton dependencies prevent competing workers. Ambiguous submissions
fail closed; review `status`/`squeue` before `resume --clear-submission-intent`.
Extend a cap with `resume --max-allocations 200`. Corrupt commits are preserved
and block resume for investigation.

## Outputs and downstream handoff

```text
common/inference_jobs/<experiment>/
  config.json, state.json, progress.json
  logs/inference.log, logs/allocation-<job>.log
  output/selection/{selection.json,cache.pkl,cache.pkl.manifest.json}
  output/motions/<ORIGINAL_CACHE_INDEX:06d>.npy
  output/completed/<ORIGINAL_CACHE_INDEX:06d>/{motion.npy,receipt.json}
  output/manifest.json  # completed view at the last clean allocation boundary
```

The immutable plan preserves source-cache hash, original index, subset position,
motion key, frames, offset/rank/gap, variant slot and seed. Filenames retain
**original cache indices**. Samples/receipts commit by atomic directory rename;
resume validates them and repairs missing flat links. Incomplete staging files
are excluded. The batch `exec`s one signal-aware process, without a child pool.

The stock refiner deduplicates by motion key, so **do not pass both offsets in
one refinement tree**. Export completed samples into per-variant snapshots:

```bash
python script/slurm/cet_inference_jobs.py export --experiment tamf_cet_epoch11
```

This creates `common/inference_jobs/tamf_cet_epoch11/handoff/variant_00/` and
`variant_01/`, each with `motions/`, a matching `cache.pkl`/sidecar and a
generation-pinned `manifest.json`. It also works on completed portions of a
stopped run. Existing handoffs are protected; use `export --output-dir NEW_PATH`
for a later snapshot. Run large exports in a CPU allocation if needed.

Example for variant 00, inside the usual GPU allocation:

```bash
H="$PWD/common/inference_jobs/tamf_cet_epoch11/handoff/variant_00"
python -m oakink2_tamf.launch.sample_refine \
  --cfg config/arch_refine.yml --exp_id tamf_cet_epoch11_v00 \
  --data.process_range '?(file:asset/split/test.txt)' \
  --data.cache_dict_filepath "$H/cache.pkl" \
  --debug.pose_repr_sample_dir "$H/motions" \
  --debug.model_weight_filepath common/train/refine__remastered/save/model_0399.pt \
  --debug.sample_save_offset test/tamf_cet_epoch11_v00 --commit

python Scripts/evaluate.py \
  --motion-dir "$H/motions" --test-cache "$H/cache.pkl" \
  --pairing-manifest "$H/manifest.json" \
  --refined-dir common/sample_refine/tamf_cet_epoch11_v00/sample/test/tamf_cet_epoch11_v00 \
  --output-dir common/evaluation/tamf_cet_epoch11_v00
```

Repeat with variant 01 and distinct refinement/evaluation directories. The
refiner weights and `Scripts/evaluate.py` must be available on the server; no
new metric implementation is included here. For G-only CET, pass the handoff
motion/cache/manifest paths to `Scripts/evaluate.py --stage G --metrics cet`.

## Validation and memory limits

```bash
python -m unittest discover -s tests -p test_cet_inference.py -v
```

CPU selection loads the source once; GPU sampling loads only the GT subset,
without worker replication or unused meshes/point clouds. A 4 GiB source-pickle
cap protects the 16G workflow; larger sources need a smaller sequence cache.
RSS is logged. Full-cache peak RAM and CPU partition policy need server checks.

Validation passed: 26 inference/controller tests and 7 existing cache tests.
Slurm calls were mocked. A real local GPU canary completed two full 1,000-step
samples with original TaMF weights: one initial sample, then a fresh-process
resume generating exactly one more while preserving the first commit. Two
other planned samples stayed pending, and a partial handoff was exported.
Host high-water RAM was approximately 3 GB for this small subset. The report is
`tmp/selected_inference_smoke_cc1c04fddd/smoke_report.json`. The epoch-11 CET
checkpoint exists only on Shenkar; run its bounded server canary before full use.
