# Prepared partial-generation evaluation

Only source inspection, CPU file/cache validation, and preparation were performed.
No generation, refinement, evaluation, GPU job, package installation, or download was run.

## Current coverage

| Input / selection | Count |
|---|---:|
| Original test-cache segments | 13,317 |
| Generated files, IDs 000000 through 001284 | 1,285 |
| Successfully matched and validated generated IDs | 1,285 |
| Unmatched generated IDs | 0 |
| Official unique `(sequence, primitive, hand side)` keys | 91 |
| Duplicate segment entries skipped by official code | 1,194 |
| Original-cache segments outside the generated batch | 12,032 |
| Refined outputs currently available in the dedicated output tree | 0 |
| Missing unique refined outputs | 91 |
| Evaluated motions / metrics run during preparation | 0 |

After successful refinement, expect **91** `save_dict.pkl` files, rather than 1,285.
If all 91 are available, CR aggregates **9,387 valid frames**, SIV aggregates
**502 frame samples**, and PSKL-J/FID compare **91 paired motions**. If fewer
refined outputs exist, the official evaluators skip missing files on both sides.

All matched cache entries have finite hand rotation matrices `(160,16,3,3)`,
translations `(160,3)`, shapes `(160,10)`, and object transforms `(160,4,4)`.
Valid lengths are 16–160, frame-ID lists match those lengths and increase,
hand sides agree with info keys, and FID action labels are supported.
All 30 referenced objects have valid finite 768-dimensional embeddings,
8192-point clouds, descriptions, and one readable nonempty mesh at the official
loader location. MANO left/right files are available. Seventeen meshes are
non-watertight; this geometry was preserved. The official SIV code uses its
existing robust SDF path; no SDF computation or mesh repair was performed here.
Per-object details are in `common/partial_batch/test_one__0399/validation.json`.

`downloads/encoder__fid_1.tar.xz` is complete (1,538,188 bytes). Its checkpoint
was deserialized on CPU, all tensor values were finite, and the 64-dimensional
encoder agrees with `config/arch_encoder.yml`. The existing archive's exact
checkpoint bytes were placed at `common/train/encoder__fid_1/save/model_0399.pt`
(1,899,691 bytes). No further assets were downloaded.

## Why these cache entries match

`src/oakink2_tamf/launch/sample.py` partitions dataset indices across workers,
gets `all_dataset[sample_id]`, and saves `{sample_id:06d}.npy`. Saved
`common/sample/main/opt.yml` records one worker on GPU 0, the original test
cache, and the `test_one/arch_mdm_l__0399` output offset. In
`InteractionSegmentData.__init__`, a supplied cache selects `load_cache`
instead of `load_dataset`; `load_cache` assigns the stored lists directly.
There is **no `process_range` filtering of cached entries**. Thus these
filenames map to original cache indices 0–1284 despite `test_one.txt`.

**Remaining provenance assumption:** the original cache's order and contents
have not changed since generation. No contemporaneous cache checksum exists
in the saved settings. The preparation manifest records a checksum now and
the runner rejects subsequent cache or generated-file changes.

The generated-sample adaptor pairs sorted filenames with positional dataset
entries and asserts equal file/dataset counts. The separate subset cache uses
those exact parsed original indices in sorted-filename order, keeping all
duplicates and restricting the object list to referenced objects. The original
cache and all generated files remain unchanged. The adaptor's assertion is
retained. Refinement's G input directory now has a CLI override.

Refinement and all four metric scripts select `cuda:0`; no other official
metric source was changed. Each official loop deduplicates by info key and
checks `save_dict.pkl` existence before adding either comparison side.
CR retains its `<0.005` contact threshold and valid-length frames. SIV retains
its `range(0, avai_len, 20)` frames, SDF settings and aggregation. PSKL-J retains
last-valid-frame padding, acceleration FFT, normalization and both KL directions.
FID retains its encoder, valid-length zero padding and population statistics.

The existing sibling `../Scripts/run_original_tamf_evaluation.py` accepts
`--partial-batch-manifest`. It verifies hashes, refined metadata, official
deduplication, and paired coverage; reports generated/matched/refined/missing/
evaluated counts in JSON, CSV and Markdown; and checks official output counts.
Frame counts and motion counts are kept distinct. Extra refined files are
reported and ignored. Zero-pair and invalid-input batches are blocked. Successful
results are reused only for the same prepared batch and refined-file snapshot.
Scores are labeled **partial-batch results**, not full-test-set reproduction.

## Manual WSL commands

Run these yourself in your existing activated TaMF environment. The first
command below is an optional CPU-only consistency check; it does not launch jobs.

```bash
cd /mnt/d/repos/oakink/OakInk2-TaMF
python Scripts/partial_batch.py --check
```

Refine the existing generated files:

```bash
python -m oakink2_tamf.launch.sample_refine \
  --cfg config/arch_refine.yml \
  --exp_id partial_test_one \
  --data.process_range '?(file:./asset/split/test_one.txt)' \
  --data.cache_dict_filepath common/partial_batch/test_one__0399/cache.pkl \
  --debug.pose_repr_sample_dir common/sample/main/sample/test_one/arch_mdm_l__0399 \
  --debug.model_weight_filepath common/train/refine__remastered/save/model_0399.pt \
  --debug.sample_save_offset test_one/arch_mdm_l__0399 \
  --commit
```

Then run the four official metrics through the existing runner:

```bash
python ../Scripts/run_original_tamf_evaluation.py \
  --tamf-root "$PWD" \
  --cache-dict-filepath common/partial_batch/test_one__0399/cache.pkl \
  --partial-batch-manifest common/partial_batch/test_one__0399/manifest.json \
  --process-range '?(file:./asset/split/test_one.txt)' \
  --sample-refine-filepath common/sample_refine/partial_test_one/sample/test_one/arch_mdm_l__0399 \
  --python "$(command -v python)" \
  --metrics all \
  --output-dir common/partial_batch/test_one__0399/results \
  --logs-dir common/partial_batch/test_one__0399/logs
```

The runner never runs refinement or generation. It uses the official FID encoder
path by default. Refinement runs in one process on GPU index 0; the runner invokes
the metrics sequentially. No generation command is part of this workflow.

## Paths relative to the TaMF repository

- G input: `common/sample/main/sample/test_one/arch_mdm_l__0399/*.npy`
- Original GT cache: `common/save_cache_dict/main/cache/test.pkl` (preserved)
- Prepared GT subset: `common/partial_batch/test_one__0399/cache.pkl`
- Provenance / checks: `common/partial_batch/test_one__0399/{manifest,validation,coverage}.json`
- R checkpoint: `common/train/refine__remastered/save/model_0399.pt`
- R output: `common/sample_refine/partial_test_one/sample/test_one/arch_mdm_l__0399/<sequence with / replaced by ++>/<primitive>/<hand side>/save_dict.pkl`
- FID encoder: `common/train/encoder__fid_1/save/model_0399.pt`
- Runner reports: `common/partial_batch/test_one__0399/results/{preflight.json,original_tamf_metrics.json,original_tamf_metrics.csv,summary.md}`
- Runner logs: `common/partial_batch/test_one__0399/logs/<run ID>/*_{stdout,stderr}.txt`
- Official CR arrays: `tmp/compute_score/contact_ratio/{gt_contact_dist,refined_contact_dist}.npy`
- Official SIV arrays: `tmp/compute_score/solid_intersection_volume/{gt,refined}.npy`

No missing input blocker remains for this prepared batch. Refined outputs are
pending your manual run; runtime dependencies/GPU execution were not tested.
Wrapper/preparation checks passed for syntax, duplicate/missing/extraneous
pairing, invalid refined metadata, cache drift, metric command construction,
and exact refinement CLI registration without invoking any job entry point.
