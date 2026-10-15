# Saved evaluation analysis and notebook viewer

Run preparation in the existing **WSL TaMF environment**, from the repository:

```bash
python Scripts/prepare_original_tamf_analysis.py
```

This reads existing evaluation JSON, logs, manifest, caches, refined metadata and
CR/SIV arrays. It verifies the official loop ordering, first-occurrence deduplication,
frame boundaries, provenance and frame-weighted score reconstruction. It never runs
generation, refinement, the four evaluators, the FID encoder or package installation.
Existing evaluation artifacts are read-only.

The script saves a separate portable bundle:

| File | Content |
| --- | --- |
| `common/analysis/test_one__0399/analysis.json` | Official scores, raw accepted CR/SIV arrays, per-motion values and differences, original indices, metadata, frame IDs/sample offsets, distributions, FID terms, runtimes, validation and plot explanations |
| `common/analysis/test_one__0399/motions.csv` | One row per verified unique motion, including original cache index, sequence, primitive, hand, refined path, frame weights, array boundaries and CR/SIV values with units |
| `common/analysis/test_one__0399/plots/*.png` | Figures rendered from the same validated snapshot, with consistent GT/refined colors |

Then open `Scripts/visualize_original_tamf_results.ipynb` in **Windows VS Code** and
run its cells using your existing Python notebook kernel. The notebook loads JSON
and displays PNGs and tables. It needs only the standard library and IPython, which
the notebook kernel already provides. It does **not** require Matplotlib, NumPy,
Linux paths, cache loading or evaluator dependencies. A WSL notebook kernel is not
required. Its `BUNDLE_DIR` setting selects the prepared bundle.

Preparation requires NumPy; PNG rendering requires Matplotlib in the WSL environment.
Neither is installed automatically. `--no-plots` exports values only if you do not
want previews; the notebook will explain that no figures were exported. If the input
artifacts change, rerun the preparation script to refresh the snapshot.

Optional paths:

```bash
python Scripts/prepare_original_tamf_analysis.py \
  --batch-dir common/partial_batch/test_one__0399 \
  --output-dir common/analysis/test_one__0399
```

CR remains the valid-frame fraction with distance **<0.005 m**. JSON/CSV motion CR
uses a ratio; difference columns explicitly identify percentage points. Saved CR
distance arrays remain metres, while plots convert to millimetres. SIV means use
official offsets `0,20,40,…` and already-converted **cm³**. Weighted motion recovery
must reproduce the saved pooled scores; motion means are never averaged equally to
reconstruct official results.

Unsupported views remain explicit: no PSKL-J spectra or FID feature matrices were
saved, and both official metrics compare populations rather than providing individual
motion scores. The prepared JSON records the provenance limitations inherited from
the historical run, along with current export hashes.

Validation mismatches produce explicit skipped views and exit status `2`; successful
preparation exits `0`. Output paths overlapping existing input/evaluation directories
are rejected. The analysis is a snapshot, not a new evaluation run.
