# Evaluate saved TaMF motions

`Scripts/evaluate.py` runs evaluation separately from training, generation and
refinement. It never loads a generator/refiner checkpoint or launches those
networks. `--experiment` is only a report label. Existing results are read-only;
every run requires a **new** output directory.

## Existing implementations reused

| Existing code | Reuse / finding |
| --- | --- |
| `script/compute_score/compute_score_{cr,siv,psklj,fid}.py` | Invoked directly as separate official metric processes. Their calculations, sampling, padding, thresholds and deduplication are preserved. Optional structured exports, device selection and a FID seed were added; historical defaults remain supported. |
| `Scripts/evaluate_cet.py` | Same pose decoding, MANO anatomical bases, constrained DTW and detailed curl exports. Added opt-in stage-only pairing so G CET can run without R. Its historical CLI still defaults to requiring both. |
| `Scripts/partial_batch.py`, `Scripts/PARTIAL_BATCH.md` | Existing batch preparation and manifest contract. The new command accepts these manifests and preserves original indices when reading a subset cache. |
| Sibling `../Scripts/run_original_tamf_evaluation.py` | Existing standalone official-metric wrapper, inspected for CLI routing and isolation requirements. It is outside this repository and is not a deployment dependency of the new command. |
| `Scripts/prepare_original_tamf_analysis.py`, `Scripts/tamf_result_plots.py`, `Scripts/visualize_original_tamf_results.ipynb` | Existing saved-result analysis/visualization. They remain compatible with historical bundles. The legacy notebook does not automatically load the new result schema. |

The previous tools already evaluated saved motions, but were tied to a specific
partial batch, split across official/CET commands, and did not retain PSKL-J
spectra or FID features. The new script handles pairing and orchestration, rather
than implementing new metric formulas.

## Commands on Shenkar

Activate the existing TaMF environment, then run from the repository. Replace
the input placeholders with paths to the completed inference artifacts:

```bash
cd /home/ela.nitai/Desktop/GPU_access_folder/OakInk2-TaMF
conda activate /home/ela.nitai/.conda/envs/oakink2-tamf

python Scripts/evaluate.py \
  --motion-dir /path/to/generated_motions \
  --test-cache /path/to/exact_generation_test_cache.pkl \
  --refined-dir /path/to/refined_motions \
  --output-dir /path/to/new_evaluation_results \
  --trust-cache-order \
  --experiment tamf_cet_001_retry1_epoch0011
```

This runs **all five metrics on R-stage outputs**. It does not need
`common/train/tamf_cet_001_retry1/save/model_0011.pt`. It does need the separate
fixed **FID reference encoder**, which is not the generator's training checkpoint.

Use `--pairing-manifest /path/to/manifest.json` instead of `--trust-cache-order`
when a compatible manifest exists. Both source-cache and prepared subset-cache
hashes are recognized. Manifests exported in `inputs/manifest.json` can also be
reused with the original supplied cache and the same explicit selection.

Add `--allow-subset` if the directory intentionally omits expected numeric files.
Missing IDs are then recorded as excluded; missing required R records still fail.
For a bounded check in an existing full folder, repeat `--sample-id ORIGINAL_ID`.
The selection does not renumber files. FID needs at least two unique motions.

Append `--dry-run` to validate selected inputs and assets and print commands,
without creating output files or running metrics. On Slurm, run the ordinary
command inside your existing allocation; it does not submit jobs itself.

For raw G outputs without refinement, the supported metric is CET:

```bash
python Scripts/evaluate.py \
  --motion-dir /path/to/generated_motions \
  --test-cache /path/to/exact_generation_test_cache.pkl \
  --output-dir /path/to/new_G_cet_results \
  --stage G --metrics cet --trust-cache-order --allow-subset
```

Use `--metrics cr siv psklj fid cet` to select metrics explicitly. `--stage R` is
the default. Official metrics require existing refined outputs; the command
rejects requests to treat raw G poses as R results and never refines them.

Other overrides: `--data-root` (default repository `data` symlink), `--test-split`
(default `asset/split/test.txt`), `--mano-root`, `--obj-embedding-dir`,
`--obj-pointcloud-dir`, `--fid-config`, `--fid-encoder`, `--device` (default
`cuda:0`), `--seed` (FID, default 0), and `--rho` (CET, default 0.10).

## Inputs and matching

```text
generated_motions/
  000000.npy
  000001.npy
  ...
refined_motions/
  <sequence with / replaced by ++>/<primitive>/<lh or rh>/save_dict.pkl
```

Each `.npy` is the generator's finite numeric **[160,99]** pose representation:
three translations plus 16 local rotations in TaMF's row-based 6D convention.
The filename is the **index in the cache used for generation**, as written by
`oakink2_tamf.launch.sample`; it is not the index in a newly filtered cache.
In particular, setting `process_range=test_one` does not filter a loaded cache.
Sparse files must retain their original names. Nested or malformed filenames,
unknown IDs and duplicate manifest IDs are rejected.

The explicit test cache supplies GT rotations, translations, hand shapes, side,
valid lengths, original frame IDs, descriptions and object trajectories. Selected
sequences must belong to `--test-split`. A prepared subset manifest maps original
sample IDs to cache positions; positional renumbering is never inferred.

As in the official evaluators, only the **first selected cache occurrence** of
each `(sequence, primitive, hand side)` is scored. Later temporal-offset slices
are listed in `duplicate_segments`; they are not independent official metric
rows. Selecting a different offset requires matching refined frame IDs, or it
fails. The report distinguishes available files, selected segments, unique
motions, missing IDs, unselected IDs and unselected refined paths.

R records must have matching `info`, `process_key`, side, length, frame IDs and
object IDs, plus finite `refine_pose_repr [160,99]`, `verts [160,778,3]` and
`joints [160,21,3]`. This matches the existing refinement save schema. Missing
R records fail before any metric runs. Inputs are fingerprinted and checked
again between metrics; failures return a nonzero exit status and are reported.

**Provenance limitation:** `.npy` files do not embed identity or cache hashes.
The evaluator cannot prove historical pairing from shape/index alone, nor prove
that an R tree was produced from this particular G run just from its metadata.
`--trust-cache-order` is an explicit user assertion, never a silent fallback.
An older preparation manifest pins its saved snapshot, not necessarily the
generation-time cache. Preserve the exact cache and motion/manifest snapshot
for future comparisons; do not rebuild or reorder the cache.

## Output files

| File | Contents |
| --- | --- |
| `summary.json`, `summary.csv` | Experiment/stage/scope, selected metrics, success/failure and aggregate scores; JSON also records commands, implementation hashes, MANO/FID reference hashes. |
| `motions.json`, `motions.csv` | Original sample ID, sequence, primitive, hand, valid length, per-motion GT/predicted CR/SIV, CET scalar and status, and array mappings. JSON also retains all 15 joint and five finger CET errors. No per-motion FID or population PSKL-J is invented. |
| `inputs/manifest.json` | Cache/file hashes, original-ID-to-source-cache-position mapping, selected representatives, discarded temporal slices, coverage and provenance limitations. |
| `inputs/cache.pkl` | Separate GT selection for the official metric readers; originals remain untouched. Only trusted local pickle files should be used. |
| `metrics/{cr,siv}/results.json`, `arrays.npz` | Per-motion scores and frame boundaries, plus lossless GT/predicted frame distances (CR, metres) or volumes (SIV, cm³). CR uses `<0.005`; SIV keeps offsets `0,20,40,...`. Summary values pool frames, rather than averaging motion means. |
| `metrics/psklj/results.json`, `arrays.npz` | Both population KL directions, each motion's acceleration power spectra, and the normalized pooled distributions. Array row indices map through the manifest. |
| `metrics/fid/results.json`, `arrays.npz` | Aggregate FID, seed and encoder mode; both feature matrices, means and covariances. The row mapping identifies each motion's features, not a per-motion FID. |
| `metrics/cet/{motions.json,motions.csv,summary.json}` | Existing detailed CET schema, anatomical provenance, per-motion/joint/finger scores and failure reasons. |
| `metrics/cet/arrays/<ID>_<stage>.npz` | Original curl sequences, 15 joint names, DTW frame pairs, all aligned absolute curl errors, time coordinates, bases and weights. Recompute/rank errors without rerunning MANO. |
| `logs/<metric>_{stdout,stderr}.txt` | Official process logs. Historical raw CR/SIV files are contained under this run's `work/<metric>/tmp/`; previous repository `tmp/compute_score` results are not touched. |

All NPZ files load with `np.load(path, allow_pickle=False)`. All five metrics
use the same selected unique motion set. CET failures are preserved and fail
the run rather than silently changing its denominator. Exit code 0 means every
requested metric succeeded; 2 means an input or metric failure. Successful
metric outputs remain available when another metric fails.

## Dependencies and preserved limitations

- Official metrics use the existing TaMF environment, dataset/toolkit metadata,
  MANO/manotorch, canonical object meshes, sampled object point clouds and object
  embeddings. SIV additionally needs `pysdf` and the compiled `libmesh` containment
  extension. The evaluator installs nothing and never rebuilds datasets/caches.
- FID uses `config/arch_encoder.yml` and
  `common/train/encoder__fid_1/save/model_0399.pt` by default. Keep this reference
  encoder fixed across experiments. Both populations contain the same paired GT
  contexts; subsets have different statistics from a full test benchmark.
- The official FID script leaves its encoder in training mode, so dropout is
  active. This implementation preserves that historical behavior, supplies a
  fixed seed through the new command, and records both seed and mode. Exact
  numerical repeatability still depends on the same environment/device. Changing
  to `eval()` would be a separately reviewed metric-protocol change.
- The official SIV grid, centering behavior and coordinate units are preserved.
  Source-level caveats are documented in
  `research_notes/tamf_codebase_atlas/06_evaluation.md`; no mathematical fixes
  were folded into this evaluation wrapper.
- PSKL-J is a **population** spectrum comparison in the official code. Saved
  per-motion spectra allow further analysis; they are not individual PSKL-J
  scores. CET rejects undefined anatomical/DTW cases and reports them.

## Local validation

```bash
python -m unittest discover -s tests -p test_saved_evaluation.py -v
python -m unittest discover -s tests -p test_cet.py -v
```

On 2026-10-18 all five real metrics completed on CPU for saved original-model
motions **1179 and 1263**, each 16 valid frames, using the existing prepared
cache/manifest and refined tree. This was a two-motion smoke, not a benchmark
of the new CET-trained model. Artifacts are in
`tmp/saved_evaluation_smoke_20261018_01`. No full evaluation, inference, training,
package installation, server access or old-result overwrite was performed.
