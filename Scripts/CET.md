# Standalone Curl Error with Time Alignment

`evaluate_cet.py` evaluates the available prepared partial batch, with R versus
GT as the default. It imports neither generation, refinement, the existing four
metric entry points, nor viewer/notebook code. Those files are unchanged by this
implementation. All metric work is on CPU. MANO is invoked only to construct
two fixed anatomical templates, once per run; no per-motion mesh reconstruction
is needed. Original motion indices remain the indices from the generation cache.

## Commands

In the existing WSL TaMF environment:

```bash
cd /mnt/d/repos/oakink/OakInk2-TaMF
# One-time extra dependency of manotorch.axislayer:
python -m pip install -r Scripts/requirements-cet.txt

# R versus GT, rho=0.10; creates a fresh timestamped output directory:
python Scripts/evaluate_cet.py --stage R

# Optional G versus GT, using the same valid R/G paired set:
python Scripts/evaluate_cet.py --stage G

# Best way to compare G and R on exactly the same input snapshot:
python Scripts/evaluate_cet.py --stage both
```

Explicit current paths (defaults already match these):

```bash
python Scripts/evaluate_cet.py --stage R --rho 0.10 \
  --manifest common/partial_batch/test_one__0399/manifest.json \
  --cache common/partial_batch/test_one__0399/cache.pkl \
  --generated-root common/sample/main/sample/test_one/arch_mdm_l__0399 \
  --refined-root common/sample_refine/partial_test_one/sample/test_one/arch_mdm_l__0399 \
  --mano-root asset/mano_v1_2
```

Read-only inventory and the lightweight checks:

```bash
python Scripts/evaluate_cet.py --inventory-only
python -m unittest discover -s tests -p test_cet.py -v
python Scripts/evaluate_cet.py --motion-index 0 --stage both
```

`--motion-index` accepts the original **representative** cache index, not a new
CSV row number or an index of a discarded duplicate. `--output-dir` may specify
a new or empty directory. Existing results are never overwritten. The evaluator
reports individual missing/invalid pairs and continues over eligible motions.
Fatal cache/manifest drift stops the run; no valid results returns exit code 2.
An inventory-only run never calculates angles or CET.

## Input matching and validity

The source inspection covered `compute_score_{cr,siv,psklj,fid}.py`,
`Scripts/partial_batch.py`, `InteractionSegmentData`, the generated-sample
adaptor, refinement's save schema, and the vendored MANO/axis utilities.
The prepared cache lists are loaded directly, as in `InteractionSegmentData`.
GT rotations come from `interaction_segment_pose_list`; this is equivalent to
the official conversion to TaMF pose representation and back. Prediction pose
is decoded from G's `.npy` or R's `refine_pose_repr`.

The preparation manifest associates each subset-cache position with an original
`sample_id`, the exact G filename, its checksum, the motion info key, valid length,
and original frame IDs. Matching uses those records and validates cache hashes;
it never infers correspondence from filesystem order. Original filenames must
agree with manifest IDs. Duplicate `(sequence, primitive, hand_side)` entries
use the first cache occurrence, matching the official evaluation loops. That
entry's original ID is the representative G ID. No different duplicate is tried
if the representative fails. Every discarded duplicate is recorded with its
original ID and reason.

R is located by `<sequence with / replaced by ++>/<primitive>/<side>/save_dict.pkl`.
Its info key, hand side, length, frame IDs and valid pose are checked. Both G and
R must be present and valid for either comparison stage. Both are also checked
for angle-extraction failures even in a single-stage run, so singular G cannot
silently lead to a different successful R set. `--stage both` shares one
selection and input snapshot; separate runs must compare their provenance hashes
to establish identical inputs. No G substitution for missing R is implemented.

Only `[:valid_length]` enters decoding, extraction, metrics or descriptors.
Padding, including nonfinite padding, is excluded. Valid-frame nonfinite values,
nonpositive lengths, malformed matrices, zero/collinear 6D axes and ambiguous
anatomical decompositions produce explicit invalid statuses. METRIC failures
never become zero-valued curl. A motion failure row has blank numeric fields.
Input paths, checksums and reasons are retained even for failed rows.

TaMF stores translation plus 16 rotations, with each 6D vector holding the first
two **rows** of a rotation matrix. Gram-Schmidt matches
`dev_fn.transform.rotation_np.rot6d_to_rotmat_np`, with explicit checks before
normalization instead of zero-vector clamping.

## Anatomical angle convention

Representation ID: `manotorch_AxisLayerFK_template_XYZ_signed_bend_Z_v1`.
This is a signed, anatomy-aligned local-rotation bend component. It is not an
axis-angle magnitude or a 3D angle between reconstructed bones.

The native `AxisLayerFK` creates a flat-hand, zero-shape MANO template for each
side. Its `AxisAdaptiveLayer` defines back/twist, up/spread and left/bend axes;
`AnatomyConstraintLossEE` explicitly identifies XYZ components 0, 1 and 2 as
twist, spread and bend. We retain only component 2. All motions and G/R/GT use
the same fixed per-side template; the exact template basis and model/source
checksums are saved. Shape-dependent changes to the template are not introduced.

If `A_j` is `AxisLayerFK.TMPL_R_p_a[j]` and `L_j` is the parent-relative MANO
rotation, the library's template-relative anatomical rotation simplifies to:

```text
Q_j = transpose(A_j) @ L_j @ A_j
bend_j = atan2(-Q_j[0,1], Q_j[0,0])
```

This is exactly the Z component of the library's `XYZ` convention. Parent axes
cancel algebraically in its rest-corrected relative rotation. Root index 0 and
translation never enter this formula. Tests compare the result to native
`AxisLayerFK.forward` and `compose` on actual left and right MANO models.

For MCP joints, sideways spread is the anatomy-aligned Y component and is not
included in bend. Pure spread, or twist with spread but zero bend, gives zero
curl in the declared decomposition. This convention measures rotation in the
library's anatomical template, rather than a motion-dependent palm-plane angle.

The thumb joints are **CMC, MCP, IP** at MANO indices 13, 14, 15. The library
labels its last thumb joint `pip`; the output uses the anatomical name `IP`.
Thumb axes use the library's special `[1,1,1]` up-vector construction (other
fingers use `[0,1,0]`) and the same signed Z bend component. Thumb opposition,
twist and spread are excluded as separate components. This is a reproducible
MANO thumb convention, not a validated clinical measurement of opposition.
Signs follow right-handed rotation about the side-specific library bend axis;
no ad hoc left-hand sign flip or absolute-value conversion is applied.

Exact 15-joint output order:

| MANO indices | Finger | Joint labels | SNAP 21-joint indices |
|---|---|---|---|
| 1,2,3 | index | MCP,PIP,DIP | 5,6,7 |
| 4,5,6 | middle | MCP,PIP,DIP | 9,10,11 |
| 7,8,9 | little | MCP,PIP,DIP | 17,18,19 |
| 10,11,12 | ring | MCP,PIP,DIP | 13,14,15 |
| 13,14,15 | thumb | CMC,MCP,IP | 1,2,3 |

MANO parent indices including root are
`[0,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14]`.
The SNAP ordering is verified from `ManoLayer.skinning`, not assumed from a
generic 21-joint skeleton. No geometry fallback is used or silently selected.

Limitations: XYZ bend is ambiguous at spread gimbal lock, and signed angles
have a +/-pi branch. Gimbal singularities (`cos(spread) <= 1e-6`) and within-clip
bend changes exceeding pi are reported as invalid. There is no smoothing,
unwrapping, zero replacement or alternate extraction for those motions. The
fixed library template provides a reproducible anatomical model convention;
it does not establish clinical anatomical accuracy for every hand shape.

## CET and timing

Frame cost is `sum_j(w_j * abs(pred[t,j] - GT[s,j])) / sum_j(w_j)` in radians;
initial weights are 15 ones. One shared DTW path uses all 15 joints together.
Allowed steps are `(1,1), (1,0), (0,1)` with first and last frame pairs included.
Only pairs satisfying `abs(u[t]-v[s]) <= rho` are permitted (roundoff tolerance
`1e-12`). Default rho is 0.10. Dynamic programming minimizes **cumulative** cost,
backtracks, and then divides by path length. It does not minimize average path
cost. Ties prefer diagonal, then advancing prediction, then advancing GT.

```text
CET_degrees = (180/pi) * cumulative_cost_radians / path_length
W = mean(abs(u[path[:,0]] - v[path[:,1]]))
```

Saved frame IDs are the dataset's original mocap coordinates. `FPS_MOCAP=120`
is read from the repository's toolkit metadata, with its path/hash and the
dataset code hash recorded. Actual strides are counted from the current cache;
the evaluator does not assume that every cached sequence uses the constructor's
default `target_fps=10`. Normalized time is `(frame_id-first)/(last-first)`.
`SegmentSlice.from_gap` adapts the stride when a primitive would otherwise be
shorter than the minimum or longer than the maximum segment length. The current
cache therefore has several saved strides (1 through 26 among observed values),
not a universal 10 Hz rate. The evaluated example's stride is 16 mocap frames,
equivalent to 7.5 Hz. Both comparison sides keep that exact sampling.
For regular sampling this equals frame index divided by `T-1`. A singleton's
time is 0; singleton/singleton can align, while singleton/multiframe requires
rho=1 to include the latter's last endpoint. Disconnected narrow bands return
`infeasible_path`. The core function supports unequal sequence lengths; this
prepared pipeline rejects mismatched R metadata rather than inventing lengths.

G, R and GT inherit identical saved frame IDs and valid lengths. CET uses their
original sampling with no resampling or filtering. The unwarped MAE diagnostic
uses linear interpolation on the sorted union of both normalized time grids,
then averages absolute joint errors over that grid with the same weights.
For the current paired inputs these grids coincide, so the diagnostic is simply
the original-frame angular MAE. Diagnostic interpolation arrays are saved and
are never passed to CET.

## Curl descriptors and aggregation

GT and prediction are separate. For each joint over valid frames we save mean
signed flexion, mean absolute flexion, maximum absolute flexion, range
`max(angle)-min(angle)`, and total angular variation `sum(abs(diff(angle)))`.
All are in degrees. Duration is the elapsed first-to-last saved frame-ID span
divided by the verified mocap clock. For positive duration we also save total
variation/duration in degrees/second; it is omitted for single-frame inputs.
Total variation describes movement/activity and is never called energy.

Each finger descriptor is the arithmetic mean of its three **joint statistics**;
the hand descriptor is the arithmetic mean of all 15 joint statistics. Thus a
finger's maximum descriptor means mean of the three individual joint maxima,
not the maximum over the whole finger and not the maximum of a pre-averaged
timeline. Range and variation also summarize individual joint statistics.
Mean signed values can cancel and are retained alongside absolute magnitude.
Per-finger CET similarly averages the three per-joint errors on the single
shared path. Dataset aggregation weights successful unique motions equally,
with population standard deviation (`ddof=0`). No frame-weighted summary or
arbitrary high/low category is produced.

## Output schema

Default directory: `common/partial_batch/test_one__0399/cet/<UTC timestamp>/`.

| File | Contents |
|---|---|
| `motions.csv` | Flat motion-stage rows, original/subset/representative IDs, key components, description, source paths, valid lengths, status/reason, CET, W, path length, cumulative cost, unwarped MAE, per-joint/finger CET and separate GT/prediction descriptors |
| `motions.json` | Same rows with nested descriptor maps and named metric vectors; lossless structured companion to CSV |
| `arrays/000000_R.npz` etc. | Original valid curl arrays and numeric reconstruction data for each successful motion-stage |
| `summary.json` | Equal-motion statistics, inventory, failures/skips with reasons, duplicate original IDs, complete pairing manifest, paths/checksums, command/version, rho/weights, mappings, anatomical bases, units, timing/sampling policies and partial-batch scope |

NPZ arrays are readable with `allow_pickle=False`. Fields:

| Field | Shape / meaning |
|---|---|
| `prediction_curl_degrees`, `GT_curl_degrees` | `[T_pred,15]`, `[T_GT,15]`, original valid frames |
| `joint_names`, `finger_names` | Exact Unicode name/order vectors, 15 and 5 entries |
| `prediction_frame_ids`, `GT_frame_ids` | Original integer frame coordinates |
| `prediction_normalized_time`, `GT_normalized_time` | Unitless normalized saved frame coordinates |
| `prediction_time_seconds`, `GT_time_seconds` | Elapsed time from first frame, using recorded mocap clock |
| `dtw_frame_pairs` | `[L,2]` zero-based prediction/GT **valid-array offsets**, endpoints included |
| `aligned_abs_error_degrees` | `[L,15]`, every shared-path joint error |
| `diagnostic_common_normalized_time` | Union time grid for unwarped diagnostic only |
| `diagnostic_prediction_degrees`, `diagnostic_GT_degrees` | `[N_grid,15]` diagnostic interpolated arrays |
| `joint_weights` | `[15]` equal initial weights |
| `anatomy_basis` | `[16,3,3]` exact side-specific template basis |
| `cumulative_cost_radians` | DP optimum before path normalization |
| `original_motion_index`, `stage`, `angle_convention`, `rho`, `frame_id_clock_hz` | Scalar metadata |

To reproduce CET without MANO:

```python
import numpy as np
z = np.load('arrays/000000_R.npz', allow_pickle=False)
p = z['dtw_frame_pairs']
e = abs(z['prediction_curl_degrees'][p[:, 0]] - z['GT_curl_degrees'][p[:, 1]])
per_joint = e.mean(axis=0)
cet = per_joint @ z['joint_weights'] / z['joint_weights'].sum()
per_finger = per_joint.reshape(5, 3).mean(axis=1)
W = abs(z['prediction_normalized_time'][p[:, 0]] - z['GT_normalized_time'][p[:, 1]]).mean()
```

Failed motions remain table rows with reasons and no invented numeric arrays.
Fatal invalid preparation metadata stops before producing a misleading partial
score. The generation-time cache checksum was not recorded by the original
pipeline; the preparation checksums only pin cache provenance from preparation
onward. This existing provenance limitation is also in every run summary.

## Validation performed

Only read-only inventory, lightweight tests and original motion 0 for G/R were
run during implementation. No generation, refinement, full-batch CET, four-metric
evaluation, viewer edit or notebook creation was run. Unit tests cover identity,
known constant offset, constrained timing shift, one path for all fingers,
joint/finger aggregation, cumulative-versus-average objective, irregular time
coordinates, singleton/infeasible paths, invalid input, padding, descriptor
activity, NPZ reproduction, original-index pairing/deduplication, missing R,
mismatched metadata, anatomy-aligned spread separation, and actual left/right
MANO equivalence and rigid-transform invariance.

The local TaMF environment lacked `deprecation`, needed by the native anatomy
helper. For validation it was staged under `.cache/cet-test-deps`; the existing
TaMF environment was not changed. Install the supplied requirements file once
before running the regular commands above. Existing MANO/chumpy emits legacy
deprecation/resource warnings; no upstream files were altered.

Validation result: **12 tests passed**. Current inventory: 1,285 generated
segments, 91 unique paired motions, 1,194 discarded duplicates, zero missing
pairs and zero pose/metadata-invalid pairs. Anatomy extraction was checked on
motion 0 only, not all 91 motions. Original cache size is 13,317 segments.

Example output record excerpt from the actual single-motion check:

```json
{
  "original_motion_index": 0,
  "subset_cache_index": 0,
  "representative_G_index": 0,
  "primitive": "scoop:00",
  "hand_side": "rh",
  "stage": "R",
  "status": "success",
  "angle_convention": "manotorch_AxisLayerFK_template_XYZ_signed_bend_Z_v1",
  "rho": 0.1,
  "valid_length_GT": 155,
  "valid_length_prediction": 155,
  "duration_seconds": 20.533333333333335,
  "cet_degrees": 6.60707827757564,
  "per_finger_degrees": [4.107016917904761, 5.6596871721990185, 10.048699839752755, 10.002524024952242, 3.21746343306943],
  "unwarped_mae_degrees": 6.651898458327521,
  "W": 0.0023726273726273725,
  "path_length": 156,
  "arrays_path": "arrays/000000_R.npz"
}
```

The complete records (including every descriptor, joint error, key and path)
are in `tmp/cet_single_motion_final/motions.json`, with NPZ arrays and
`summary.json` beside it. The corresponding G CET for this one motion is
6.410447053678238 degrees; these are example values, not dataset averages.
