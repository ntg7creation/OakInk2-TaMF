# 07 — Research modification plan

[Atlas home](README.md) · [Previous: evaluation](06_evaluation.md)

Everything in this document is a **proposal**, except explicitly cited current-code facts. No architectures, loss functions, launchers, configurations, or tests were changed. The objective is to preserve a reproducible original default while making two independent research contributions measurable.

## A. Finger-curl supervision

### Representation-based starting point

The output already contains the information needed for articulation supervision: `pose_repr[...,3:99]` reshapes to 16 6D rotations; root is index 0, indices 1–15 are **already parent-relative** MANO finger rotations (`InteractionSegmentExtraLoss.recover_mano_from_pose_repr`, `interaction_segment_extra_loss.py:67–75`; MANO hierarchy at `thirdparty/manotorch/manotorch/manolayer.py:164–198`). There is no need to infer local rotations from world-coordinate joints or subtract a parent's local rotation again.

The simplest safe integration is a new term in `InteractionSegmentExtraLoss.forward` at `src/oakink2_tamf/model/interaction_segment_extra_loss.py:109–112`, added to the weighted scalar at 183–188 and dictionary at 189–196. This automatically receives the predicted **clean x0** and GT hand at every sampled diffusion step (`gaussian_diffusion.py:1181–1182`). It requires neither a new G output head nor a diffusion change.

### Formulations worth comparing

| Candidate | Construction from existing representation | Benefits | Limitations / what must be decided |
|---|---|---|---|
| Local rotation chordal error | Convert 15 predicted/GT 6D vectors to matrices; compare squared Frobenius distance | Torch-only, inexpensive, root/translation independent; no MANO skinning or Euler singularity | Supervises twist and spread as well as bend; call it articulation loss, not exclusively curl |
| Local rotation geodesic error | Compare relative rotation angle of `R_pred^T R_GT` per nonroot joint | Interpretable angular articulation discrepancy; root/translation independent | `acos` near identity/π needs careful numerical treatment; still not curl-specific |
| Anatomy-aligned signed bend | Use side-specific fixed `AxisLayerFK` template bases; form `A_j^T R_j A_j`, compare bend component | Most directly matches the existing CET descriptor; distinguishes flexion direction | Euler singularities, angle wrap, thumb convention, and stable gradients need explicit treatment; do not copy evaluator's NumPy/reject behavior into training |
| Bone-angle/cosine loss from decoded joints | Reuse predicted/GT SNAP joints; compare normalized adjacent-bone dot products along verified finger chains | Rotation/translation invariant; reuses already decoded differentiable joints; cosine avoids `acos` | Unsigned cosine cannot distinguish opposite bend directions; MCP wrist→base vector is not automatically an anatomical flexion axis; cannot observe bone twist; thumb deserves separate design |
| Anatomical signed bone-angle loss | Derive local palm/parent frames and use differentiable dot/cross/atan2 features | More joint-geometric interpretation, potentially signed | More frame construction, degenerate axes, and shape dependence; greater implementation/testing scope |
| Curl displacement/velocity term | Compare local bend or descriptor changes across neighboring valid frames | Targets movement dynamics in addition to pose | Must use real frame intervals or a clearly documented processed-frame clock; cannot replace a pose term; reverse augmentation changes temporal direction |

Do not finalize the mathematical objective before choosing whether the research target is **curl angle**, **all local articulation**, or **temporal curl dynamics**. Those are distinct. Merely taking the norm of a local axis-angle vector measures rotation magnitude, including spread/twist; it does not isolate flexion.

A candidate family for further evaluation is

\[
L_{\mathrm{curl}}=\mathcal R_b\left[
\frac{\sum_{f,j=1}^{15}m_{bf}\,w_j\,
\rho\big(c_j(\hat R_{bfj}),c_j(R_{bfj})\big)}
{(\sum_f m_{bf})(\sum_j w_j)}\right].
\]

Here `c_j` is the **to-be-chosen** articulation descriptor, `rho` its discrepancy, and `R_b` a declared sum or mean over the local batch. This is a design template, not a finalized curl formula. For signed angles, a periodic sin/cos discrepancy can avoid a ±π jump, but it changes the scale/meaning from absolute degrees and still needs singularity handling in the descriptor.

**Recommended first formulation to prototype after documentation review:** a Torch anatomy-aligned bend descriptor using the already-local rotations and fixed side-specific bases, with a coefficient-zero baseline and synthetic validation before training. Compare it to a local-rotation chordal control. This aligns the research target with the existing CET convention while separating bend specificity from generic articulation improvements. If anatomy gradients are unstable, retain the matrix control rather than hiding numerical failures with blanket `nan_to_num`.

### Concrete integration steps

1. Introduce a small independent Torch helper, proposed `src/oakink2_tamf/model/loss/finger_curl.py`, that accepts predicted/GT `[B,T,16,3,3]` (or per-example equivalents), side information, and valid masks. Keep the descriptor and reduction separate for testing. Exclude invalid padded frames before singular normalization/angle operations, rather than relying on multiplying their final errors by zero. Register fixed anatomical bases as buffers; do not rebuild a MANO template for every batch.
2. Gate computation on a new positive `train.loss.coef_curl_loss`, registered in `launch/param/loss.py:15`. Default **0.0**, so loading an existing config retains the same objective and return behavior unless a deliberate new logging key is chosen.
3. In `InteractionSegmentExtraLoss.__init__:19–55`, store the coefficient/enable flag. In `forward:109–112`, obtain predicted/GT matrices in Torch and ignore root. Reuse `joints_pred/gt` for the joint-angle alternative. Avoid `.numpy()`, `.detach()`, `float(loss)`, or `no_grad` on predictions.
4. Add the term to the scalar/dictionary at 183–196. The generic `train.summary_extra_loss:292–305` already logs arbitrary dictionary entries. Leave diffusion and every current loss accumulator unchanged.
5. Put a nonzero coefficient/descriptor choice in a **new experiment YAML**, with explicit registration for each configurable key. Keep `config/loss_param.yml` as the original baseline.
6. Declare the reduction and coefficient scale. Matching the existing G extra-loss per-example **sum** minimizes behavioral scope. If using a batch mean for curl, keep that choice explicit and calibrate lambda accordingly; do not change the five original reductions.

Target behavior is `L_total=L_TaMF+lambda_curl L_curl`. A zero coefficient must be an actual disabled path, avoiding numerical/compute changes from extra conversion. The helper need not consume object data at all, so this contribution can be tested independently of trajectory masking.

### Verification before a training experiment

Future tests should establish: identical poses have zero loss; perturbing one finger changes the intended component; changing only root rotation/translation does not change curl error; both hand sides and thumb chains match the documented convention; valid-length normalization ignores padding; gradients are finite/nonzero for a nontrivial articulation error; finite-difference/gradcheck agrees away from singularities; near-degenerate 6D and Euler branch cases have an explicit policy. A zero-lambda regression should compare total scalar and gradients with the original callback on the same small fixture. These tests were **not** created or run for the documentation task.

Hard-DTW CET is an evaluation metric. Initially keep training curl aligned to GT frames, because G is also supervised by aligned hand/object motion. Soft-DTW or another differentiable alignment could be a later ablation; it can tolerate incorrect timing and needs a separate coefficient/compute analysis. Do not directly call `Scripts/evaluate_cet.py` from training.

## B. Progressive reduction of GT conditioning

### Minimum clean interface

Current G and the extra-loss callback read one batch; both access its `obj_traj` (`interaction_segment_mdm.py:162`; `interaction_segment_extra_loss.py:104,150–154`). Therefore the first interface rule is **separate a model-conditioning view from the authoritative supervision batch**.

A minimal proposal is a shared helper, e.g. `src/oakink2_tamf/model/conditioning.py`, called **after collate/device transfer** in training and before the sampling loop in inference. It returns a nested `batch["conditioning"]` with conditioned fields and availability tensors. G reads that view when the new feature is enabled; the loss callback keeps using original `batch["pose_repr"]`, `shape`, `obj_traj`, and geometry. A shallow dictionary copy is fine only if contained original tensors are never mutated; any edit requires a cloned/new tensor.

This avoids changing cache files, dataset target semantics, or the diffusion callback signature. It also avoids collator changes for dynamically generated policies. If masks later become stored per-sample dataset fields, explicitly register their stacking/padding behavior in `collate.py`; unknown keys currently raise (`collate.py:57`). Never silently fall back to full GT trajectory when a reduced-mode field is missing—validate the conditioning contract.

Proposed fields:

| Field | Shape / purpose |
|---|---|
| `obj_exists` | `[B,N]` Boolean real object-slot validity, distinct from pose availability |
| `traj_observed` | `[B,N,T]` Boolean; subset of real objects × valid frames |
| `obj_traj` in conditioning view | `[B,N,T,9]`; observed values plus finite placeholders defined by policy |
| `geometry_available` | `[B,N]` Boolean; known/hidden static embeddings |
| `obj_embedding` in conditioning view | `[B,N,768]`; known/static or null placeholders |
| Optional other availability | Explicit text/shape/side flags if those signals become part of the experiment |
| Original batch fields | Unmodified full GT supervision/scene and metadata |

`mask[B,T]` remains the valid-duration loss mask. Observation absence does **not** remove that frame from the training target: G must still predict the whole valid hand motion.

### Proposed config policies

These key names are suggestions requiring registration, not existing options:

```yaml
conditioning:
  enabled: false                 # exact historical default path
  trajectory_mode: full          # full / random / sparse / initial / none
  trajectory_keep_probability: 1.0
  keyframe_stride: null
  include_initial_frame: true
  object_selection: all          # define any subset by semantic or explicit ID policy
  geometry_mode: full            # full / dropped / random
  pooling_mode: legacy            # separately version a validity-aware alternative
  policy_seed: 0
```

Register these through a shared config function consumed by both `launch/train.py:60` and `launch/sample.py:54`. The existing registry does not honor arbitrary YAML fields just because they exist. Record resolved settings and mask-selection provenance with outputs/checkpoints.

| Regime | Observation semantics to specify |
|---|---|
| Full baseline | Every real object's pose is observed on every valid frame; original pooling path unchanged |
| Random frames | Decide per-object versus shared-across-object frame sampling and whether the initial frame is always retained |
| Sparse keyframes | Choose observation positions from processed frame indices or physical frame times; use only observations for any interpolation |
| Initial poses | Pose observed at valid frame 0 for each retained object; future unknown, not automatically static |
| No trajectory | No observed dynamic pose; static object set/geometry can remain; fully absent feature has explicit null behavior |
| Reduced geometry | Decide object subset and availability independently of trajectory observations; removal of the G embedding is not removal of R point-cloud geometry |
| Minimal conditions | Declare exactly retained text/side/chosen shape/duration; do not accidentally retain dataset identity/shape/scene information through another branch |

Do not conflate “first object” with “initial pose.” If a single object is kept, lexicographic order is a deterministic choice but not a semantic definition of primary object. A valid experiment must state the selection rule.

### Feature handling and baseline compatibility

Extend `ObjectInputProcess.forward` (`interaction_segment_mdm.py:241–248`) to accept validity/observation information. A proposed new path can replace an unobserved object's **projected feature** by a learned null vector and add an availability embedding or observed-fraction feature. Pool only real objects under a new validity-aware mode. Fully unobserved frames need a well-defined null output and safe denominator. If more than one object is retained, consider whether an observed-count feature is needed; averaging alone can hide how much evidence exists.

Geometry requires its own null/availability handling around `ObjectEmbedProcess.forward:258–263`. For removing text/shape, use explicit gates/nulls in the prefix branches, accounting for projection biases. Hand side and reconstruction shape must still be defined for valid MANO decoding even if they are no longer dataset-derived conditions.

**Preserve the legacy default explicitly.** Switching to real-object masked means changes fully observed batches whenever original slot padding existed. Correcting shape pooling similarly changes prefixes. Version those corrections as separate controls, or retain the exact legacy path when `conditioning.enabled=false`; otherwise the “full” baseline and the mask experiment differ in more than availability.

The 99D hand output and 1000-step diffusion algorithm need not change for the initial fixed-grid masking experiments. Learned availability features/nulls are small architecture additions. Simply zeroing input values avoids parameter changes but cannot communicate missingness reliably; it should be labeled a crude corruption probe, not the final missing-observation design.

### Keep training and inference consistent

- Apply the same transformation helper, field names, null definition, object-order policy, and time grid in both launchers and any in-memory extraction helper.
- Draw training masks per example/step with controlled randomness. For repeated sample evaluation, derive deterministic masks from a pinned motion ID/policy seed and record the observed indices. Set them once before `p_sample_loop`; do not draw a new mask each denoising step.
- Keep full GT scene data for the original losses and evaluation. Add tests that the original distance-loss scene tensors do not change when only the conditioning view changes.
- Give inference requested shape/duration/side without needing clean hand pose values; initially the stock cache can still supply metadata for paired research evaluation.
- For a mixed-regime model, choose a documented training distribution over policies. Test-only omissions are a distinct out-of-distribution robustness study, not a replacement for matched training.

### R-stage policy

For initial work, evaluate G-only and keep R outside the contribution claim. A full-condition R can be reported as a separately labeled post-processing condition. For an end-to-end reduced-condition model, audit R's pooled trajectory **and** its hand–object distance features (`segment_refine_model.py:190–203,223–232`). Filling missing poses into distance computations silently introduces fictitious geometry. R needs its own observed-geometry/null-feature design and matched training; changing only the G batch is insufficient.

## Proposed change matrix

| Proposed change | Source files affected | Functions affected | Implementation complexity | Risks | Testing strategy |
|---|---|---|---|---|---|
| New Torch curl descriptor | New `model/loss/finger_curl.py`; optional reuse of vendored anatomical basis | Proposed descriptor/reduction helpers; existing `rot6d_to_rotmat` | Low for matrix control; medium for signed anatomy bend | Wrong joint order, double local-relative conversion, thumb axes, singular gradients | Synthetic finger rotations, root/translation invariance, both sides, gradient checks |
| Add independently weighted curl term | `model/interaction_segment_extra_loss.py`; `launch/param/loss.py`; new experiment YAML | `__init__`, `forward`, `reg_loss_param` | Low | Batch reduction/lambda scale; accidentally altering original terms | Zero-lambda scalar/gradient equivalence; known-error fixture; existing loss dictionary retained |
| Shared conditioning config/view | New conditioning module/param registration; `launch/train.py`, `launch/sample.py` | `reg_entry`, `run`, `sample_worker`; proposed view builder | Medium | In-place GT mutation, unregistered keys, full-condition fallback leakage | Target/scene immutability; default equivalence; config roundtrip |
| Trajectory observation masks/nulls | `model/interaction_segment_mdm.py`; shared conditioning helper | `forward`, `ObjectInputProcess.forward` | Medium | Zero interpreted as pose; denominator zero; checkpoint compatibility | Full/none/random/initial/sparse fixtures; finite outputs; absent data cannot affect features |
| Geometry availability/nulls | Same model/helper; config registration | `ObjectEmbedProcess.forward`, prefix construction | Low–medium | Projection bias and hidden geometry from other stages | Vary hidden embedding values with fixed mask; output invariant to hidden values |
| Optional validity-aware pooling | G object processors; shared conditioning metadata | Both object processor forwards | Medium | Changes fully observed baseline and batching behavior | Padding invariance for new mode; exact legacy behavior retained under old mode |
| Dynamic mask fields in Dataset, if needed later | `dataset/interaction_segment.py`, `dataset/collate.py` | `__getitem__`, collate/`SegmentCollate` | Low–medium; avoid initially | Cache schema drift, mask/object ordering | Variable N and L fixtures; all fields align; no cache rewrite required |
| Target-free requested-shape sampling | `launch/sample.py`; optional extraction helper | `sample_worker`, `extract_refined_sample` | Low | Losing valid-length/frame identity; changing random stream | Same shape/seed/conditions produce same baseline output; metadata manifest |
| G-only original metric input adapter | New research evaluator/adapter; existing metric formulas left versioned | Proposed G decode/read path | Medium | Stage/sample mismatch; silent population differences | Pinned cache/IDs/frames; G=GT fixture; per-motion coverage records |
| CET pairing/generalization | Existing `Scripts/evaluate_cet.py`, preparation utilities, separate tests | `inspect_batch`, `run`, proposed full-cache/G-only mode | Medium | Partial-batch coupling, historical provenance, changing descriptor convention | Same selected population across stages; explicit missing statuses; retain old mode |
| Correct deterministic FID / audit SIV | Metric scripts only in a **separate** future change | FID `main`; SIV `solid_intersection_volume`; `process_sdf` | Small code change, medium scientific audit | Historical comparability; units/grid/mesh assumptions | Deterministic identical-feature case; translated analytic meshes; version and recompute baseline |
| Reduced-condition R, later | `segment_refine_model.py`, R trainer/sampler/loss interfaces | `forward`, `multi_object_h2o_dist`, R batch preparation | High relative to initial G work | Reintroducing unavailable GT through geometry features | Features independent of hidden poses; matched R training; compare G-only and G+R |

All `model/`, `launch/`, and `dataset/` entries above are under `src/oakink2_tamf/` unless explicitly labeled `Scripts/` or new. Complexity is a planning estimate, not a measured implementation effort.

## Fair experimental comparison

Pin code/submodule versions, exact caches/splits, sample IDs, MANO/object assets, architecture/checkpoint initialization, reverse augmentation, optimizer/scheduler, local batch size/world size, number of updates, and diffusion steps. Match **updates and processed examples**, not only nominal epoch count if augmentation/policy changes populations. Preserve all original loss coefficients/reductions in the first experiments.

Use the same held-out motions, observed keyframes/masks, per-motion random seeds, and sampling settings across models. Report G-only and G+R independently. Full GT object motion can remain the common evaluator's scene even when hidden from the generator: that is evaluation reference data, not inference conditioning. Declare the remaining information budget (text, shape, side, geometry, object selection, duration) explicitly for each regime.

Use one versioned deterministic metric implementation across all compared models; metric repairs require a recomputed baseline. Report CET with coverage, per-finger errors and time-warp amount/unwarped error, alongside CR/SIV/PSKL-J/FID. Preserve per-motion outputs/IDs and multiple-seed variation; silently skipping failed motions can make a model appear better. For CET's current partial batch, state the partial population and both-stage eligibility restriction.

Recommended sequence: establish the pinned baseline/manifest and source-derived invariance fixtures; add default-off curl plumbing and validate gradients; train the curl-only ablation; add conditioning-view separation; add one observation regime at a time with matched training; evaluate the combined model only after measuring each contribution independently. R changes and soft temporal training alignment can follow later.
