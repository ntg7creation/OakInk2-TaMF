"""Prepare a <=4-sample cache, or explicitly run two bounded GPU sessions.

Preparation reads a verified existing TRAIN cache and does no raw preprocessing.
GPU mode performs two tiny epochs with the original model/CET, saving/resuming
in a new experiment. It does not sample motion. Nothing runs on import/--help.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from oakink2_tamf.dataset.cache_util import FIELDS, OBJECTS, cache_spec, publish_new, read_cache, write_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--run-gpu-smoke", action="store_true")
    parser.add_argument("--source-cache", type=Path, help="verified full training cache; CPU preparation only")
    parser.add_argument("--cache", type=Path, required=True, help="new output cache for preparation; tiny input for GPU mode")
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--data-prefix", type=Path, default=ROOT / "data")
    parser.add_argument("--cfg", action="append", help="additional experiment overlays after the default architecture/geometry/loss configs")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    if args.prepare_only:
        if args.source_cache is None or not 1 <= len(args.sample_indices) <= 4:
            parser.error("preparation requires --source-cache and 1..4 sample indices")
        if args.cache.exists() or Path(str(args.cache) + ".manifest.json").exists():
            parser.error("output cache already exists; inspect/reuse it rather than overwrite it")
        cache, metadata = read_cache(args.source_cache, require_manifest=True, split_name="train")
        count = len(cache[FIELDS[0]])
        if len(set(args.sample_indices)) != len(args.sample_indices) or any(i < 0 or i >= count for i in args.sample_indices):
            parser.error("sample indices must be unique valid training-cache indices")
        selected = {key: [cache[key][i] for i in args.sample_indices] for key in FIELDS}
        selected[OBJECTS] = sorted({oid for traj in selected["interaction_segment_obj_traj_list"] for oid in traj})
        keys = list(dict.fromkeys(info[0] for info in selected[FIELDS[0]]))
        spec = dict(cache_spec(keys, "train"), purpose="bounded_smoke_subset", source_cache_sha256=metadata["sha256"],
                    source_indices=args.sample_indices)
        result = write_cache(args.cache, selected, spec)
        print(json.dumps(result, indent=2))
        return

    import torch
    from oakink2_tamf.util.training_state import resolve_checkpoint
    cache, metadata = read_cache(args.cache, require_manifest=True, split_name="train")
    if not 1 <= len(cache[FIELDS[0]]) <= 4 or metadata.get("purpose") != "bounded_smoke_subset":
        parser.error("GPU mode requires a verified bounded_smoke_subset of 1..4 samples")
    if not torch.cuda.is_available():
        parser.error("GPU smoke requires CUDA; prepare the subset on a CPU allocation first")
    run_id = "cached_workflow_smoke_" + uuid.uuid4().hex
    experiment = ROOT / "common/train" / run_id
    if experiment.exists():
        parser.error("new experiment path unexpectedly exists")
    control = ROOT / "common/debug_cached_training" / run_id
    control.mkdir(parents=True, exist_ok=False)
    split_file = control / "split.txt"
    publish_new(split_file, lambda stream: stream.write(("\n".join(metadata["sequence_keys"]) + "\n").encode()))
    override = control / "smoke.yml"
    publish_new(override, lambda stream: stream.write(b"data:\n  append_reverse_segment: false\n  enable_obj_model: false\n"))
    command = [sys.executable, "-u", "-m", "oakink2_tamf.launch.train"]
    configs = ["arch_mdm_l.yml", "obj_embedding.yml", "obj_pointcloud.yml", "loss_param.yml", "cet_smoke.yml"]
    for cfg in [str(ROOT / "config" / name) for name in configs] + (args.cfg or []) + [str(override)]:
        command += ["--cfg", cfg]
    command += ["--data.data_prefix", str(args.data_prefix.resolve()),
                "--train.cache_dict_filepath", str(args.cache.resolve()), "--train.require_verified_cache",
                "--train.process_range", f"?(file:{split_file})", "--train.batch_size", "1",
                "--train.loss.cet_weight", "0.1", "--train.record_freq", "1", "--runtime.num_worker", "0",
                "--runtime.device_id", str(args.device_id), "--val.val_freq", "-1", "--test.test_freq", "-1",
                "--exp_id", run_id, "--commit"]
    subprocess.run(command + ["--train.num_epoch", "1"], cwd=ROOT, check=True)
    first_path = resolve_checkpoint("latest", experiment)
    first = torch.load(first_path, map_location="cpu")
    subprocess.run(command + ["--train.num_epoch", "2", "--train.resume_checkpoint", "latest"], cwd=ROOT, check=True)
    last_path = resolve_checkpoint("latest", experiment)
    last = torch.load(last_path, map_location="cpu")
    assert first["next_epoch"] == 1 and last["next_epoch"] == 2
    assert last["global_updates"] == 2 * first["global_updates"]
    assert any(not torch.equal(first["model"][key], last["model"][key]) for key in first["model"])
    assert all(torch.isfinite(t).all() for t in last["model"].values() if t.is_floating_point())
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(experiment / "summary")).Reload()
    for tag in ("diffusion_loss", "extra_loss", "extra_loss/cet", "extra_loss/cet_weighted",
                "extra_loss/cet_pred_branch_crossing_rate"):
        assert tag in events.Tags()["scalars"], f"missing training log: {tag}"
        assert all(np.isfinite(event.value) for event in events.Scalars(tag)), f"nonfinite log: {tag}"
    report = dict(experiment=str(experiment), first_checkpoint=str(first_path), resumed_checkpoint=str(last_path),
                  completed_epochs=last["next_epoch"], global_updates=last["global_updates"], cet_weight=0.1,
                  scope="two <=4-step GPU epochs, no sampling or raw preprocessing")
    publish_new(control / "results.json", lambda stream: stream.write(json.dumps(report, indent=2).encode()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
