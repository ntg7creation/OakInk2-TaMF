# 06 — Evaluation, motion pairing, and CET

[Atlas home](README.md) · [Previous: conditioning](05_inference_and_conditioning.md) · [Next: research plan](07_research_modification_plan.md)

## Shared population and saved motion schema

All four original evaluators construct `InteractionSegmentData` from a cache with reverse augmentation disabled. They iterate cache order and deduplicate `info=(sequence,primitive,hand_side)` **before checking file existence**. Only the first temporal-offset slice per info is considered; missing records are silently skipped and later duplicates are not tried. Examples: CR at `script/compute_score/compute_score_cr.py:213–231`, SIV at `compute_score_siv.py:219–237`, PSKL-J at `compute_score_psklj.py:215–234`, FID at `compute_score_fid.py:273–291`.

The standard input is:

```text
<sample_refine_filepath>/
  <sequence with / replaced by ++>/<primitive>/<lh or rh>/save_dict.pkl
```

`src/oakink2_tamf/launch/sample_refine.py:282–304` saves:

| Key | Meaning / default shape |
|---|---|
| `info`, `process_key`, `hand_side` | Motion identity and side |
| `refine_pose_repr` | `[160,99]` refined hand translation/local rotations |
| `verts` | `[160,778,3]` translated MANO vertices |
| `joints` | `[160,21,3]` translated SNAP-order joints |
| `faces` | Closed hand mesh face indices |
| `obj_list` | Selected object IDs |
| `len`, `frame_id` | Valid prefix length and original selected frame IDs |

The original metric loops do not validate saved identity, frame IDs, or cache provenance against these fields. They trust path lookup and shape compatibility. G's numeric `.npy` files contain `[160,99]` but no IDs; matching them requires the exact generation cache order (`sample.py:235–237`). `GeneratedPoseReprSampleAdaptor` checks the total file count, then attaches sorted loaded files by list position, not by an explicit identity match (`pose_repr_sample.py:26–48`). Sparse/missing files and multiple directories therefore need a manifest, not just a count check.

## Metric overview

| Metric | Actual motion/scene input | Selection and aggregation | Per-motion information retained by original script? | G-only applicability |
|---|---|---|---|---|
| CR | GT reconstructed hand vertices; saved predicted vertices; canonical sampled object points transformed by GT object poses | All valid frames of first-info motions with existing records; mean binary contact over concatenated frames | Flat GT/predicted per-frame minimum distances saved, **no motion IDs/boundaries** | Geometry measure applies to G; stock reader expects saved geometry records |
| SIV | GT and saved predicted hand vertices, closed hand faces; object meshes/SDF grids placed by GT poses | Frame offsets `0,20,40,...<L`; mean over concatenated selected frames | Flat frame volume arrays saved, **no IDs/boundaries** | Applies to G meshes; stock reader expects refinement schema |
| PSKL-J | GT reconstructed and saved predicted 21 XYZ joints, extended to padded T with last valid pose | FFT of second differences; sum power over motions; two directional KL scores | Only aggregate scores/shapes printed; spectra/motion scores not saved | Applies to decoded G joints; stock reader expects saved joints |
| FID | GT and saved predicted 99D hand motion, shared GT shape/side/object embedding/trajectory; action encoder checkpoint | One learned feature per motion, mean/covariance Fréchet distance | Aggregate/debug values printed; feature matrix/statistics not saved | G 99D motion can substitute; stock reader expects `refine_pose_repr` |

The metrics themselves do not mathematically require R. Their supplied input adapters/paths do. A future G evaluator can decode G `.npy` motions with the same shape/side or feed the G pose to the FID encoder. Keep both stages labeled; do not compare G-only improvements to an R-stage baseline without disclosing the stage difference.

## CR — Contact Ratio

`transf_merge_obj_pointcloud` (`compute_score_cr.py:123–137`) decodes `[N,T,9]` object poses, transforms `[N,P,3]` points, and merges all objects into `[T,N·P,3]`. `contact_min_cdist` (140–149) computes **all pairwise** distances between hand vertices and object points, then takes a single minimum per frame:

\[
d_f=\min_{v,p}\|V_{fv}-O_{fp}\|_2,\qquad
CR=\frac1{\sum_b L_b}\sum_{b,f<L_b}\mathbf1[d_{bf}<0.005].
\]

GT and prediction are evaluated separately against the same object point sets/poses, with valid-prefix trimming at lines 268–275. Contact means **at least one pair** is closer than the threshold. It does not measure contact area, which finger touches, penetration, or agreement with the GT contact state. High CR can coexist with incorrect grasp/articulation.

Lines 276–289 concatenate distances, print GT/predicted ratios, and save `tmp/compute_score/contact_ratio/{gt_contact_dist,refined_contact_dist}.npy`. Aggregation weights long motions more because each frame contributes equally. If input lengths/order and missing-file selection are pinned, frame boundaries can be reconstructed from a manifest, but the original arrays do not carry that mapping.

The threshold is in raw coordinate units. The conventional 5 mm interpretation assumes the motions/objects use meters; this evaluator does not check or convert dataset units.

## SIV — Solid Intersection Volume

`solid_intersection_volume` (`compute_score_siv.py:128–155`) approximates how much object interior lies inside the closed hand. `process_sdf` (`src/dev_fn/util/sdf_util.py:59–99`) builds a `resolution=100` grid around a 1.2-expanded canonical mesh bounding box and keeps points with positive SDF (“inside” for the chosen library convention).

For each object and selected frame:

\[
\mathrm{SIV}_f=10^6\sum_o
\left(\prod_{d=1}^{3}\Delta_{od}\right)
\sum_{p\in\mathrm{interiorGrid}(o)}\mathbf1[T_{of}p\text{ inside closed hand}_f].
\]

`check_mesh_contains` tests grid points against the hand mesh. The scale factor is exactly `10**6` (148), conventionally m³→cm³ for meter-scale inputs. This is a grid approximation, not exact mesh intersection. The objects are summed independently, so overlap between objects can be counted more than once.

`compute_score_siv.py:275–306` trims valid prefixes, caches one SDF per object, samples **every twentieth processed frame** starting at zero, concatenates values, prints mean GT/predicted volumes, and saves `tmp/compute_score/solid_intersection_volume/{gt,refined}.npy`. Every twentieth frame is not a fixed physical interval because dataset stride can vary. Motions with more selected frames contribute more weight.

### Source-level concerns requiring a separate metric audit

- **Apparent double center addition:** `process_sdf` already stores `point=query_point+obj_mesh_center` (`sdf_util.py:81–96`). The evaluator then adds `mesh_center` again to those points (`compute_score_siv.py:141–145`). The composed code therefore applies a double offset when the canonical center is nonzero. Its numerical impact on existing reported results was not evaluated.
- **Grid volume versus sample spacing:** `tick_unit=extent_expanded/resolution` (`sdf_util.py:70`) but the grid is made with `linspace(..., resolution)` including endpoints (73–75), whose adjacent spacing is `extent_expanded/(resolution-1)`. The estimator uses the former for cell volume. This is the implemented estimator, not an exact quadrature claim.
- `process_sdf` centers `obj_mesh.vertices` in place (67). This mutates the in-memory mesh object used by the evaluator; it does not write the dataset mesh file.

Do not silently correct these while claiming exact reproduction of original metric values. A corrected metric should have a separate version, controlled fixtures, and baseline recomputation.

## PSKL-J — Joint Power-Spectrum KL divergence

`compute_score_psklj.py:249–274` reconstructs GT joints with MANO and obtains predicted joints from the record. It sets padded frames to the last valid pose for **both** sequences (270–271). It does not simply trim to `L`.

At default T=160, it then computes `[158,21,3]` second differences and full complex FFT power (`280–299`):

\[
a_{bfj d}=J_{b,f+2,jd}-2J_{b,f+1,jd}+J_{bfjd},\qquad
S_{bkjd}=|\operatorname{FFT}_f(a_{bfjd})_k|^2.
\]

There is no division by the physical frame interval squared, root-relative conversion, global rotation normalization, or Welch spectrum in executed code. The commented Welch call is not active. Root movement and all 21 world-coordinate joints contribute.

Power is **summed** over motions, regularized by `1e-8`, and normalized across frequency independently for each joint/axis (`305–310`):

\[
p_{kjd}=\frac{\sum_b S^{GT}_{bkjd}+10^{-8}}{\sum_{k'}(\sum_b S^{GT}_{bk'jd}+10^{-8})},
\quad q\text{ analogously for predictions.}
\]

Lines 314–317 implement two numbers:

\[
PSKL_{GT\to pred}=\frac1{21}\sum_{k,j,d}p_{kjd}\log\frac{p_{kjd}}{q_{kjd}},\qquad
PSKL_{pred\to GT}=\frac1{21}\sum_{k,j,d}q_{kjd}\log\frac{q_{kjd}}{p_{kjd}}.
\]

The divisor is `dataset_mean_ps.shape[1]`, **21 joints**, not 63 joint-coordinate features. It does not average the two directions. This is a population spectrum comparison, not a frame-aligned reconstruction error or a DTW score. Motion amplitudes affect the aggregate power before normalization. Last-pose extension also introduces a stopping boundary into short sequences.

Only arrays in memory, debug shapes, and final scores are present; this script saves no per-motion scores or spectra. Different physical sampling strides are not corrected by the implementation.

## FID — Fréchet distance of TaMF encoder features

FID here uses **`SegmentEncoder`**, not an image Inception network (`compute_score_fid.py:248–268`). The encoder is trained for action recognition (`segment_encoder_loss.py:10–26`, `launch/train_encoder.py:453–480`). `ActionRecognitionAdapter` derives one of 69 labels from the prefix before `:` in `info[1]` (`dataset/action_adapter.py:12–32`).

`SegmentEncoder.forward` (`model/segment_encoder.py:75–111`) combines:

- Side, temporal-mean shape, and pooled object embedding prefixes.
- Projected hand representation and pooled object trajectory frame features.
- A zero-buffer classification token appended **after** the motion sequence.

Its last transformer token is `encoding[1,B,D]`; this is the FID feature. With the supplied `config/arch_encoder.yml`, D=64. No temporal key-padding mask is used. The saved predicted representation is zeroed after valid length before encoding (`compute_score_fid.py:330–331`). The GT cache already has zero tails.

Both sides of the comparison reuse **GT object motion/embedding, shape, side, and duration**; only the hand representation is replaced (`compute_score_fid.py:306–349`). Thus FID measures a learned motion-and-context distribution. It is not a hand-articulation-only metric and it does not explicitly score text fidelity.

One feature vector per selected motion yields matrices `[M,D]` (`351–352`). Mean and covariance are `np.mean(...,axis=0)` and `np.cov(...,rowvar=False)` (`198–202`). The implemented distance is:

\[
FID=\|\mu_{GT}-\mu_{pred}\|_2^2+
\operatorname{tr}(\Sigma_{GT}+\Sigma_{pred}-2\sqrt{\Sigma_{GT}\Sigma_{pred}}).
\]

`calculate_frechet_distance:142–195` handles a nonfinite matrix square root by diagonal epsilon regularization and accepts small imaginary components. The script prints the result; it does not save features, statistics, or per-motion contributions. FID is a set statistic, so an “individual motion FID” is not defined by this code.

### FID implementation risks

1. **No `segment_encoder.eval()` call appears in `compute_score_fid.py`.** `torch.no_grad()` at 319/345 prevents autograd, but does not disable dropout in the newly initialized model. Its positional/transformer dropout remains active. The script also does not set a seed. Repeated scores can vary even for identical inputs; the magnitude was not measured here.
2. Encoder `output_dim` is stored but its classification head is built with `self.input_feats` (99) at `segment_encoder.py:33`, 73. The action vocabulary is 69. Cross entropy can train a 99-logit head with 69 used labels, but this is not the expected output_dim wiring. FID uses the pre-head `encoding`, so that mismatch does not itself change the feature shape.
3. Encoder registry defaults are latent 256/eight layers; `arch_encoder.yml` specifies 64/two layers. Supply the checkpoint's matching architecture. `strict=False` does not make size mismatches safe or guarantee a fully restored feature extractor.
4. FID registers the cache under **`debug.cache_dict_filepath`**, whereas the other three use `data.cache_dict_filepath` (`compute_score_fid.py:96–104`,225). Metric defaults also contain stale `mocap_meta/process_range/test.txt` paths. A cache is authoritative for sample population, but explicit paths are still preferable.

## Where an independent CET evaluator gets identical motion data

For an evaluator matched to original metrics:

1. Pin the exact cache and original numeric sample IDs with hashes/manifest.
2. Iterate the **first info occurrence** in cache order. Match only records available to the metric under comparison; record every skip instead of silently changing denominators.
3. GT local rotations are directly available in `interaction_segment_pose_list[index][:L]`, shape `[L,16,3,3]`. This avoids a redundant rotation→6D→rotation→MANO round trip.
4. G prediction comes from `<original_sample_id:06d>.npy`, decode `[:L,3:99]` to `[L,16,3,3]`.
5. R prediction comes from the same info-key `save_dict.pkl["refine_pose_repr"]`, decode its valid prefix. Validate its `info`, side, length, and frame IDs.
6. If using joint-angle descriptors instead, reconstruct G/GT with matching side/betas and use the same SNAP joint chains as saved R `joints`. Do not mix 16-rotation ordering and 21-joint ordering.
7. Keep only valid frames for CET. Carry original frame IDs/time coordinates; do not inherit PSKL-J's last-frame extension or SIV's every-20-frame subset unless designing a separately labeled variant.

These locations are independent of diffusion or refinement execution. A saved-motion evaluation script should not import a launcher or rerun a network simply to obtain hand articulation.

## CET already present in this working tree

`Scripts/evaluate_cet.py` and `tests/test_cet.py` predate this investigation and are untracked local research additions. They were inspected, not run or changed. The current implementation already provides:

- Strict valid-prefix 99D decoding with zero/collinear 6D rejection (`decode_pose:80–92`).
- Fixed side-specific anatomical bases from `AxisLayerFK.TMPL_R_p_a` (`load_anatomical_bases:117–138`). With local rotation `L_j` and anatomical basis `A_j`, it forms `Q_j=A_j^T L_j A_j` for joints 1–15, extracts signed XYZ Z-bend as `atan2(-Q[0,1],Q[0,0])`, and rejects gimbal singularities/branch crossings (`curl_from_local_rotations:95–114`). Root is omitted, so changing global wrist orientation does not change the descriptor.
- Normalized times from original frame IDs (`normalized_time:141–147`), weighted 15-joint absolute-angle cost and constrained DTW (`constrained_dtw:163–225`), default time band `rho=0.10`. It minimizes **cumulative** cost, then divides by the selected path length; it does not optimize the minimum path-average cost directly.
- Manifest/hash/identity/length/frame checks and first-info deduplication (`inspect_batch:255–352`), G/R/both stage selection, saved per-motion DTW paths/errors, CSV, and aggregate mean/median/std (`359–397`,489–512).

The current evaluator is tied to a prepared **partial batch**. Its pairing policy requires valid G **and** R for either stage, including angle-extraction compatibility, so it is not automatically a full-cache G-only evaluator (`run:422`,452–464; `main:518–525`). Existing preparation hashes pin data after preparation but do not prove a historical generation-time cache checksum. Preserve that limitation when reporting current outputs.

Its convention is an anatomical Euler bend component, not a raw rotation magnitude or generic 3D bone angle. The manotorch anatomy code identifies XYZ components as twist/spread/bend (`thirdparty/manotorch/manotorch/anatomy_loss.py:87–89`). The quality and anatomical adequacy of this descriptor, especially thumb conventions and singular cases, remain research decisions. Existing tests cover extraction, root-rotation invariance, DTW recurrence, and pairing by source inspection; this atlas does not claim those tests were executed.

## CET evaluation versus differentiable curl training

| Aspect | CET evaluation | Additional curl loss |
|---|---|---|
| Input | Final saved G/R motion and GT valid sequence | Predicted x0 at a sampled diffusion timestep and GT target |
| Time alignment | Hard constrained DTW can accommodate timing shifts | Start with aligned-frame articulation supervision; soft alignment is a separate formulation |
| Execution | NumPy/CPU is fine; failure statuses can reject invalid examples | Must stay in Torch with gradients; numerical handling must avoid detached graph paths |
| Global invariance | Omit root and use local articulation/anatomical bases | Same representation principle; verify root/translation invariance explicitly |
| Output | Report degrees, per-joint/per-finger errors, paths, coverage, population summary | Scalar weighted by lambda_curl in the optimizer objective |
| Can existing CET be called directly in training? | No need for autograd | **No**: NumPy conversions, argmin/backtracking, and rejection policies are not a differentiable training path |

Sharing a documented joint/basis convention is useful. Sharing the entire hard-DTW evaluator with the training callback would be incorrect. No new CET or loss implementation was added in this task.
