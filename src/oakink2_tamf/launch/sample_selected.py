"""Single-device inference using the original 1000-step TaMF sampler unchanged."""
from __future__ import annotations

import logging
from pathlib import Path
import random
import time

import numpy as np

from ..util.inference import (
    atomic_json, commit_motion, output_inventory, progress_record,
    write_evaluation_manifest,
)

LOGGER = logging.getLogger(__name__)


class OriginalSampler:
    """Construct lazily, once, after the separate CPU selection allocation."""
    def __init__(self, state, selection, stop):
        import torch
        import yaml
        from ..dataset.cache_util import read_cache, memory_status
        from ..dataset.interaction_segment import InteractionSegmentData
        from ..model.interaction_segment_mdm import InterationSegmentMDM
        from ..model.diffusion_util import create_gaussian_diffusion

        if not torch.cuda.is_available():
            raise RuntimeError("No allocated CUDA device; submit inside Slurm with --gres=shard:1")
        self.torch, self.stop = torch, stop
        self.device = torch.device("cuda:0")
        cache, _ = read_cache(selection["subset_cache"], require_manifest=True, split_name=state["split"])
        self.dataset = InteractionSegmentData(
            process_range_list=list(dict.fromkeys(e["info"][0] for e in selection["samples"])),
            data_prefix=state["data_root"], obj_embedding_prefix=state["obj_embedding_dir"],
            enable_obj_model=False, obj_pointcloud_prefix=None, append_reverse_segment=False, cache_dict=cache)
        del cache
        model_cfg = yaml.safe_load(Path(state["model_config"]).read_text(encoding="utf-8"))["model"]
        self.model = InterationSegmentMDM(**model_cfg).to(self.device)
        weights = torch.load(state["checkpoint"], map_location="cpu")
        if "model" in weights and isinstance(weights["model"], dict):
            weights = weights["model"]
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        missing = [key for key in missing if not key.startswith("clip_model.")]
        if missing or unexpected:
            raise ValueError(f"checkpoint/model architecture mismatch: missing={missing}, unexpected={unexpected}")
        del weights
        self.model.eval()
        # Graceful USR1 finishes the current sample. TERM aborts at the next model
        # forward; no incomplete sample is published. No child process exists.
        self.denoising_step = 0
        self.current_sample = None
        def abort_on_term(module, inputs):
            if self.stop.hard:
                raise InterruptedError("allocation terminated during an unfinished sample")
            self.denoising_step += 1
            if self.denoising_step % 100 == 0:
                LOGGER.info("Sample %s: denoising step %d/1000", self.current_sample, self.denoising_step)
        self.model.register_forward_pre_hook(abort_on_term)
        self.diffusion = create_gaussian_diffusion(diffusion_steps=1000, noise_schedule="cosine")
        LOGGER.info("Single inference process; device=%s; selected cache only; %s", self.device, memory_status())
        LOGGER.info("Runtime: PyTorch=%s; NumPy=%s; GPU=%s", torch.__version__, np.__version__,
                    torch.cuda.get_device_name(self.device))

    def __call__(self, entry):
        from ..dataset.collate import interaction_segment_collate
        from dev_fn.transform.cast import map_copy_select_to

        torch = self.torch
        random.seed(entry["seed"])
        np.random.seed(entry["seed"])
        torch.manual_seed(entry["seed"])
        self.current_sample = entry["sample_id"]
        self.denoising_step = 0
        sample = self.dataset[entry["subset_cache_index"]]
        batch = interaction_segment_collate([sample])
        batch_device = map_copy_select_to(
            batch, device=self.device, dtype=torch.float32,
            select=("mask", "pose_repr", "shape", "obj_num", "obj_traj", "obj_embedding"))
        shape = tuple(batch_device["pose_repr"].shape)
        shape = (shape[0], shape[2], 1, shape[1])
        with torch.no_grad():
            result = self.diffusion.p_sample_loop(
                self.model, shape, clip_denoised=False, model_kwargs={"batch": batch_device},
                skip_timesteps=0, init_image=None, progress=False, dump_steps=None,
                noise=None, const_noise=False)
        return result.permute((0, 3, 1, 2)).detach().cpu().numpy().squeeze(3).squeeze(0)


def run_samples(state, selection, state_dir, stop, max_seconds, predictor=None):
    """Incremental transaction loop, also exercised with CPU fake predictors."""
    state_dir = Path(state_dir)
    started = time.monotonic()
    receipts, errors = output_inventory(state, selection, repair_links=True)
    if errors:
        raise ValueError(f"corrupt/untracked outputs preserved; inspect before resuming: {errors}")
    initial = len(receipts)
    failures = {}
    reason = "completed"
    entries = {e["sample_id"]: e for e in selection["samples"]}
    sampler = predictor

    def save(current=None, phase="sampling"):
        report = progress_record(selection, receipts, failures, phase=phase, current_sample=current,
                                 allocation_elapsed_seconds=time.monotonic() - started)
        atomic_json(state_dir / "progress.json", report)
        return report

    save(phase="starting")
    write_evaluation_manifest(state, selection, receipts)
    for index in selection["schedule"]:
        if index in receipts:
            continue
        elapsed = time.monotonic() - started
        if stop.requested or (state_dir / "STOP").exists():
            reason = "interrupted"
            break
        average = sum(r["seconds"] for r in receipts.values()) / len(receipts) if receipts else 0
        if elapsed + max(30, 2 * average) >= max_seconds:
            reason = "session_limit"
            break
        entry = entries[index]
        save(current=index)
        try:
            if sampler is None:
                sampler = OriginalSampler(state, selection, stop)
            if stop.requested or (state_dir / "STOP").exists():
                reason = "interrupted"
                break
            sample_started = time.monotonic()
            array = sampler(entry)
            receipts[index] = commit_motion(state, selection, entry, array, time.monotonic() - sample_started)
        except InterruptedError:
            reason = "interrupted"
            break
        except Exception as exc:
            failures[index] = f"{type(exc).__name__}: {exc}"
            reason = "failed"
            LOGGER.exception("Sample %s failed; chain stops, resume retries missing samples", index)
            break
        report = save()
        LOGGER.info("Inference: %s | Unique motions: %d / %d complete (%d covered) | "
                    "Generated samples: %d / %d | Strides/motion: %s | Average %.2fs/sample | ETA %.1fs",
                    state["experiment"], report["completed_unique_motions"], report["total_unique_motions"],
                    report["covered_unique_motions"], report["generated_samples"], report["total_samples"],
                    report["strides_per_motion"], report["average_seconds_per_sample"], report["estimated_remaining_seconds"])
    report = save(phase=reason)
    # Receipts are the per-sample durable truth. Avoid rewriting an ever-growing
    # full evaluation manifest before AND after every sample in a large run.
    write_evaluation_manifest(state, selection, receipts)
    return dict(reason=reason, advanced=len(receipts) - initial, seconds=time.monotonic() - started, progress=report)
