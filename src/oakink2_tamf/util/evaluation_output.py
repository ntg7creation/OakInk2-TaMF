"""Optional, lossless exports for the existing official evaluation loops.

No generator/refiner is imported here. With no output directory, the historical
metric CLIs retain their existing behavior and output locations.
"""
import json
from pathlib import Path

import numpy as np


def register_evaluation_output(registry):
    from config_reg import ConfigEntrySource

    for name, category, default in (("evaluation_output_dir", str, None),
                                    ("evaluation_device", str, "cuda:0"),
                                    ("evaluation_seed", int, None)):
        registry.register(name, prefix="debug", category=category,
                          source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG, default=default)


class MetricOutput:
    def __init__(self, output_dir):
        self.path = Path(output_dir) if output_dir else None
        self.rows = []
        if self.path is not None:
            # Refuse even an existing empty directory: concurrent runs cannot mix.
            self.path.mkdir(parents=True, exist_ok=False)

    def add(self, index, sample):
        if self.path is not None:
            self.rows.append(dict(subset_cache_index=int(index), info=list(sample["info"]),
                                  valid_length=int(sample["len"])))

    def save(self, summary, arrays):
        if self.path is None:
            return
        arrays = dict(arrays, subset_cache_indices=np.asarray(
            [r["subset_cache_index"] for r in self.rows], dtype=np.int64))
        np.savez_compressed(self.path / "arrays.npz", **arrays)
        (self.path / "results.json").write_text(json.dumps(
            dict(summary=summary, motions=self.rows, arrays="arrays.npz"),
            indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def frames(self, gt, prediction, *, contact=False):
        if self.path is None:
            return
        gt, prediction = np.asarray(gt), np.asarray(prediction)
        offsets = [0]
        for row in self.rows:
            count = row["valid_length"] if contact else len(range(0, row["valid_length"], 20))
            start, end = offsets[-1], offsets[-1] + count
            offsets.append(end)
            a, b = gt[start:end], prediction[start:end]
            row.update(array_start=start, array_end=end,
                       gt=float(np.mean(a < .005) if contact else np.mean(a)),
                       prediction=float(np.mean(b < .005) if contact else np.mean(b)))
        if offsets[-1] != len(gt) or len(gt) != len(prediction) or not len(gt):
            raise ValueError("official frame arrays do not match evaluated motion lengths")
        self.save(dict(gt=float(np.mean(gt < .005) if contact else np.mean(gt)),
                       prediction=float(np.mean(prediction < .005) if contact else np.mean(prediction)),
                       units="ratio" if contact else "cm^3", frame_count=len(gt),
                       aggregation="pooled frames; not equal-weighted motion means"),
                  dict(gt=gt, prediction=prediction, offsets=np.asarray(offsets, dtype=np.int64)))
