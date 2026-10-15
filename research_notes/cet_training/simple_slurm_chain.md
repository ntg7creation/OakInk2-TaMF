# Simple four-hour continuation chain

This is the minimal continuation workflow for **`tamf_cet_001_retry1`**. It uses
the existing server checkpoints at `common/train/tamf_cet_001_retry1/save/`.
Checkpoint files stay on Shenkar; neither script needs local access to them.
Use this workflow instead of the older `cet_jobs.py` launcher, whose original
defaults requested 4.5 hours and two DataLoader workers.

`train.sbatch` requests the confirmed `gpu` partition, `shard:1`, two CPUs,
32 GB RAM, and **04:00:00**, matching the `msc` QoS limit. Python gets **13800
seconds** (3 h 50 min). The original predictive epoch-boundary stop/checkpoint
logic is unchanged. Zero DataLoader workers matches the current retry run and
avoids spawning full-dataset copies.

The batch script activates the existing `/opt/sw/miniforge3` Conda installation
and `oakink2-tamf` environment. It uses both `shenkar_generator_full.yml` and
`cet_experiment.yml`, keeping batch 4, the original losses, and CET weight 0.05.
The cache override is the existing `common/preprocessed_cache_v1/train.pkl`.
No cache is generated. No model, loss, dataset, or training source is changed.

At **execution time**, each job calls the existing
`oakink2_tamf.util.training_state.resolve_checkpoint` (`training_state.py:80`).
It selects a `state_epoch_*.pt` file whose manifest, size, and SHA-256 verify,
skipping incomplete/corrupt newer files. The original trainer receives that
specific path through `--train.resume_checkpoint`, validates compatibility, and
restores model/optimizer/scheduler/counters/RNG state (`training_state.py:136`).
It resumes at the checkpoint's `next_epoch`; it does not use legacy model-only
exports. There is no fresh-start fallback: missing complete checkpoints fail
the job, protecting this existing experiment.

The launcher submits **N additional sessions** immediately, chained using
`afterok`. The first session automatically depends on the highest job ID already
queued/running under your user with name `cet-tamf_cet_001_retry1`, including the
current manual training job. Alternatively, specify `--after JOB_ID`. A
`singleton` dependency and experiment locks prevent overlapping continuations.
Slurm schedules each GPU allocation after its predecessor succeeds; no terminal
or compute-node `sbatch` loop is needed. Nonzero exits stop the chain, and
`--kill-on-invalid-dep=yes` cancels jobs whose required predecessor fails.
These dependency semantics follow [Slurm's sbatch documentation](https://slurm.schedmd.com/sbatch.html#OPT_dependency).

The total epoch target defaults to **30**, retaining the current experiment's
target. Jobs that start after the target is complete exit successfully without
training. Since this is a fixed prequeued chain, those extra jobs can still
receive brief allocations. Choose a reasonable session count; increase it later
if needed. One session can complete a varying number of epochs.

After synchronizing the two new root scripts and `.gitattributes` to the server,
run from the checkout (no chmod required):

```bash
cd ~/Desktop/GPU_access_folder/OakInk2-TaMF

# Preview three additional sessions; reads squeue, submits nothing.
bash submit_train_chain.sh 3 --dry-run

# Submit three additional sessions, automatically following the current job.
bash submit_train_chain.sh 3

# Or explicitly follow a particular currently active job.
bash submit_train_chain.sh 3 --after JOB_ID
```

If no job is currently active, the first continuation is eligible immediately
and must find a complete checkpoint already saved on the server. An explicit
old job ID may no longer exist in Slurm accounting; use automatic detection if
it has already finished. The user-facing output prints every submitted ID.
If submission is rejected partway through, inspect the printed IDs and queue
before submitting more; the accepted jobs remain queued.

Monitoring and stopping:

```bash
squeue -u "$USER" -o '%.18i %.32j %.12T %.10M %R'
tail -F common/cet_jobs/tamf_cet_001_retry1/logs/allocation-JOB_ID.log
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS

# Cancel the QUEUED continuation jobs using the IDs printed by the launcher.
scancel QUEUED_JOB_ID_1 QUEUED_JOB_ID_2

# Optional: ask a running new train.sbatch job to stop after its current epoch.
scancel --batch --signal=USR1 RUNNING_JOB_ID
```

Cancel queued successors before requesting a graceful stop; a successful
epoch-boundary exit otherwise permits its `afterok` successor to run. A hard
timeout or kill preserves earlier complete checkpoints, but any unfinished
epoch must be repeated. After diagnosing a failure, `bash submit_train_chain.sh
3` starts a new continuation chain from the latest complete checkpoint.

To continue beyond 30 complete epochs without resetting the model:

```bash
TARGET_EPOCHS=100 bash submit_train_chain.sh 3
```

Each submitted job inherits the target at submission. Keep source/configs,
cache, and worker count unchanged when resuming, because the existing trainer
checks their signature. Paths can be overridden with `REPO_ROOT`, `DATA_ROOT`,
`TRAIN_CACHE`, `CONDA_BASE`, or `TAMF_CONDA_ENV`; `EXPERIMENT_ID` selects another
existing compatible experiment. The launcher exports these values to Slurm.

Local verification uses fake `sbatch`/`squeue` and small synthetic complete
checkpoints. Actual server checkpoint restoration, scheduler dependencies, and
four-hour epoch shutdown must still be observed on Shenkar. No real jobs are
submitted by the tests.
