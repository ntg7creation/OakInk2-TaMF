# CET training integration

Implementation date: 2026-10-15. This adds **CET only** to the G-stage auxiliary
loss callback. It changes neither the 99D representation nor the generator,
dataset, diffusion algorithm, inference, or checkpoint/resume behavior. The
existing `Scripts/evaluate_cet.py` and `tests/test_cet.py` remain unchanged.

## Objective and conventions

`src/oakink2_tamf/model/loss/cet.py` implements anatomy-aligned signed XYZ
Z-bend, using the same fixed, side-specific `AxisLayerFK.TMPL_R_p_a` templates
as the evaluator. Rotation 0 is global wrist orientation and is excluded;
rotations 1–15 are already parent-relative. The loss decodes the first two
**rows** of each articulated rotation from `pose_repr[...,9:99]`.

For local rotation `R` and anatomical basis `A`, it computes
`Q = A.T @ R @ A` and `theta = atan2(-Q[0,1], Q[0,0])`.

For each motion, the detached CPU alignment minimizes cumulative cost
`mean_j(abs(theta_pred[a,j] - theta_GT[b,j]))`, within the default 10% normalized
time window. It preserves the evaluator's endpoint constraints and tie order.
The selected path is then scored again on the **original Torch prediction**:

\[
\ell_i=\frac{1}{15|P_i|}\sum_{(a,b)\in P_i}\sum_{j=1}^{15}
|\theta^{pred}_{a,j}-\theta^{GT}_{b,j}|,\qquad
L_{CET}=\frac1B\sum_i\ell_i.
\]

The path minimizes cumulative error, **not** path-average error; division by
path length happens afterward, matching the evaluator. Training units are
radians; evaluator scores in degrees differ by `180/pi`.

`L_total = L_diffusion + L_extra_original + cet_weight * L_CET`.
Original auxiliary terms still **sum** across the local batch. CET deliberately
uses an equal-motion **batch mean**. No original coefficient or reduction changes.

Gradients flow through selected L1 differences, `atan2`, anatomical rotations,
6D decoding, and predicted x0. GT and path-selection copies are detached. Hard
path switches, L1 equality points, and angle branches are nonsmooth; the path is
not differentiated. There is no framewise fallback, soft-DTW, or velocity term.

## Configuration and logging

The registered options under `train.loss` are:

| Key | Default | Meaning |
|---|---|---|
| `cet_weight` | `0.0` | Nonnegative CET coefficient. Zero skips initialization, decoding, DTW, and objective addition. |
| `cet_dtw_rho` | `0.10` | Allowed normalized-time separation; range `[0,1]`. |
| `cet_profile` | `false` | Measure CPU DTW and synchronized CET forward wall time. |

The existing training logger automatically records:

- `extra_loss/cet`: raw batch-mean radians.
- `extra_loss/cet_weighted`: its contribution to the objective.
- `extra_loss/cet_pred_branch_crossings`: number of adjacent predicted joint-angle jumps greater than pi.
- `extra_loss/cet_pred_branch_crossing_rate`: count divided by valid adjacent joint comparisons.
- `extra_loss/cet_aligned_error_over_pi_rate`: fraction of selected joint pairs whose principal-angle absolute error exceeds pi.
- `extra_loss/cet_non_circular_excess_rad`: equal-motion mean excess over circular error, **diagnostic only**.
- `extra_loss/cet_path_length_mean`.
- With profiling: `extra_loss/cet_cpu_dtw_seconds` and `extra_loss/cet_forward_seconds`.

The original `extra_loss` total includes weighted CET when enabled. Disabled mode
logs raw/weighted CET as zero and retains the original objective graph.

`config/cet_smoke.yml` is an **opt-in smoke-only overlay**, with weight 0.1 and
profiling enabled. It does not alter the original loss YAML. The research-run
weight is still undecided. Enabled CET uses the evaluator's existing anatomical
dependency `deprecation==2.1.0` (`Scripts/requirements-cet.txt`); disabled CET does
not import the anatomical-axis module or construct extra MANO templates.

## Validity and branch policy

The loss requires matching `[B,T,99]` prediction/GT tensors, `[B,T]` binary prefix
masks, `B` hand-side strings, and `B` valid frame-ID lists. Only valid articulated
values enter decoding. Padding cannot create undefined angle gradients.

Zero/collinear 6D axes, nonfinite valid articulation, or undefined bend/gimbal
angles fail explicitly. They are never silently skipped or assigned zero error.
GT branch crossings are rejected as in the evaluator. Predicted crossings are
permitted and recorded because early model outputs can cross the principal
angle boundary. The actual loss remains **non-circular L1**. The aligned-error
and circular-excess diagnostics help identify boundary artifacts even when no
temporal crossing occurs, for example constant +3.1 versus -3.1 radians.

Frame IDs may strictly increase or decrease. Descending IDs arise from existing
reverse augmentation; normalize along sequence order without sorting. On the
evaluator's accepted forward-time domain, the mathematical conventions agree.

## Callback input verification

The actual data path was inspected and exercised by a unit test:

1. `InteractionSegmentData.__getitem__` returns `mask`, `hand_side`, and `frame_id`.
2. `interaction_segment_collate` stacks masks and retains side/frame IDs as lists.
3. `map_copy_select_to` keeps unselected fields; it does **not** discard frame IDs.
4. `GaussianDiffusion.training_losses` passes this same batch to `loss_callback`.
5. `InteractionSegmentExtraLoss.forward` reads the three fields directly when CET is enabled.

No dataset, collator, or launcher change is necessary. A callback test uses the
real diffusion implementation and a tiny test denoiser to check metadata and
gradient propagation. This is separate from a full TaMF generator smoke test.

## Lightweight validation

In the existing TaMF environment, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -p 'test_cet*.py' -v
```

The new tests cover evaluator angle/path/score parity, both hands, actual MANO
anatomical templates, root/translation invariance, known errors, real DTW time
alignment, variable valid lengths, padding, reversed time, numerical failures,
branch diagnostics, gradient accumulation, float64 gradcheck, GT detachment,
CUDA parity/backward, configuration, and callback integration.

Validation completed locally on 2026-10-15 in the existing WSL environment
(`torch 1.13.0+cu116`, `numpy 1.26.4`): **34 tests passed, zero failed, zero
skipped**—22 new training-loss tests plus 12 unchanged evaluator tests. This
includes a 160-frame forward/backward case and the small CUDA gradient/parity
test. The smoke harness's `--help` and input-contract unit test also passed.
Existing manotorch/chumpy deprecation/resource warnings were emitted; they were
not test failures. No full generator forward/optimizer smoke run, real-data
training benchmark, or reverse-diffusion inference was executed. Those remain
for the GPU server after data-path confirmation.

The disabled-loss regression compares all five original auxiliary formulas and
gradients on a small two-hand fixture. A separate direct comparison against the
original committed implementation at root revision
`4cb9c461b92c032715374bd252c7179c70a14bde` also found **bitwise-identical** losses,
components, and gradients for both missing `cet_weight` and explicit zero;
maximum gradient difference was `0.0`.

## Prepared GPU smoke and overhead comparison — not executed

`script/debug/smoke_train_cet.py` is a bounded, explicit harness. It reads an
existing training cache, selects samples **in memory**, and verifies the callback
contract. It never writes caches or checkpoints and never samples motion.

After confirming server paths, input-only inspection is available:

```bash
python script/debug/smoke_train_cet.py \
  --cache /CONFIRMED/PATH/train.pkl \
  --data-prefix /CONFIRMED/PATH/OakInk2/data \
  --sample-indices 0 1 --check-inputs-only
```

Only when the training smoke test is authorized, run:

```bash
python script/debug/smoke_train_cet.py \
  --cache /CONFIRMED/PATH/train.pkl \
  --data-prefix /CONFIRMED/PATH/OakInk2/data \
  --sample-indices 0 1 --steps 3 --warmup-steps 1 \
  --output common/cet_smoke/NEW_RUN_ID
```

Additional path flags cover MANO, object embeddings/point clouds, and existing
model/loss configs. Use the original checkpoint architecture config; the harness
defaults to `arch_mdm_l.yml`. It initializes a fresh model, not pretrained TaMF
weights, and resets the same non-CLIP state and random seed before each arm.

The harness compares original-loss weight-zero versus enabled CET, using the
same resident batch and controlled timesteps 0, 500, and 999. Each arm has three
measured optimizer steps by default. Warmups are optional additional steps.
It checks finite losses/gradients and parameter updates, writes TensorBoard logs
and `results.json`, and reports median step overhead and peak CUDA allocation.
Step timing excludes data loading and logging but includes numerical checks.

By default the harness turns detailed CET profiling **off**, even though the
smoke YAML enables it, to avoid adding synchronization to only the enabled arm.
Use `--profile-cet` for attribution of CPU-DTW versus CET-forward time, and label
that measurement as instrumented. Full-forward timing includes CUDA
synchronization/device transfer; CPU-DTW timing excludes those and backward.
Three-step timings are noisy and establish feasibility, not benchmark-grade
throughput or improved motion quality.

The harness caps measured steps at 10, warmups at 2, and samples at 4, and requires
a new/empty output directory. It is intentionally not a long-training launcher.
Checkpoint/resume support remains deferred.
