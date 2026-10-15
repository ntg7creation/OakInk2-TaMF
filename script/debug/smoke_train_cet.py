"""Explicit, bounded G-stage CET smoke/overhead test. Never runs on import.

Requires confirmed server paths and an explicit output directory. Compares
original-loss versus CET optimizer steps on one fixed resident training batch;
does not sample motion, change caches, or save model checkpoints.
"""

import argparse
import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from oakink2_tamf.model.loss.cet import normalized_time


def validate_batch_contract(batch):
    """Read-only verification of fields used by G and the auxiliary callback."""
    required = ("pose_repr", "shape", "mask", "frame_id", "hand_side", "text",
                "obj_traj", "obj_embedding", "obj_pointcloud", "obj_list", "obj_num")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"missing callback/model fields: {missing}")
    pose = batch["pose_repr"]
    if pose.ndim != 3 or pose.shape[-1] != 99 or pose.shape[0] == 0:
        raise ValueError("expected pose_repr [B,T,99]")
    size, length, _ = pose.shape
    if batch["mask"].shape != (size, length) or batch["shape"].shape != (size, length, 10):
        raise ValueError("mask/shape dimensions do not match pose_repr")
    for name in required[3:]:
        if len(batch[name]) != size:
            raise ValueError(f"{name} batch length does not match pose_repr")
    result = []
    for index in range(size):
        mask = batch["mask"][index].detach().cpu().numpy()
        valid = int(np.count_nonzero(mask))
        if (not np.isin(mask, (0, 1)).all() or valid == 0
                or not np.all(mask[:valid] == 1) or not np.all(mask[valid:] == 0)):
            raise ValueError(f"sample {index}: mask is not a nonempty valid prefix")
        if len(batch["frame_id"][index]) != valid:
            raise ValueError(f"sample {index}: frame_id length does not match mask")
        normalized_time(batch["frame_id"][index])
        if batch["hand_side"][index] not in ("lh", "rh"):
            raise ValueError(f"sample {index}: invalid hand_side")
        result.append({"valid_length": valid, "hand_side": batch["hand_side"][index],
                       "frame_id": batch["frame_id"][index], "objects": batch["obj_list"][index]})
    return result


def load_batch(args):
    from oakink2_tamf.dataset.interaction_segment import InteractionSegmentData
    from oakink2_tamf.dataset.collate import interaction_segment_collate

    # Existing trusted TaMF cache; select in memory, never write a subset cache.
    with args.cache.open("rb") as stream:
        cache = pickle.load(stream)
    count = len(cache["interaction_segment_info_list"])
    if len(set(args.sample_indices)) != len(args.sample_indices) or any(i < 0 or i >= count for i in args.sample_indices):
        raise ValueError("sample indices must be unique valid cache positions")
    selected = dict(cache)
    for suffix in ("info", "len", "pose", "tsl", "shape", "hand_side", "text", "obj_traj", "frame_id"):
        key = f"interaction_segment_{suffix}_list"
        if len(cache[key]) != count:
            raise ValueError(f"cache parallel-list mismatch: {key}")
        selected[key] = [cache[key][i] for i in args.sample_indices]
    selected["interaction_object_list"] = sorted({
        obj for trajectory in selected["interaction_segment_obj_traj_list"] for obj in trajectory})
    del cache
    dataset = InteractionSegmentData(
        process_range_list=[], data_prefix=str(args.data_prefix), cache_dict=selected,
        obj_embedding_prefix=str(args.obj_embedding_prefix), obj_pointcloud_prefix=str(args.obj_pointcloud_prefix),
        enable_obj_model=False, append_reverse_segment=False,
    )
    return interaction_segment_collate([dataset[i] for i in range(len(dataset))])


def run(args):
    import yaml
    from dev_fn.transform.cast import map_copy_select_to
    from dev_fn.util.random_util import setup_seed
    from oakink2_tamf.model.interaction_segment_mdm import InterationSegmentMDM
    from oakink2_tamf.model.interaction_segment_extra_loss import InteractionSegmentExtraLoss
    from oakink2_tamf.model.diffusion_util import create_gaussian_diffusion
    from oakink2_tamf.util.net_util import clip_gradient
    from torch.utils.tensorboard import SummaryWriter

    device = torch.device(args.device)
    batch = load_batch(args)
    metadata = validate_batch_contract(batch)
    if args.check_inputs_only:
        print(json.dumps({"callback_inputs_verified": True, "samples": metadata}, indent=2))
        return
    with args.model_config.open() as stream:
        model_cfg = yaml.safe_load(stream)["model"]
    with args.loss_config.open() as stream:
        loss_cfg = yaml.safe_load(stream)["train"]["loss"]
    with args.cet_config.open() as stream:
        loss_cfg.update(yaml.safe_load(stream)["train"]["loss"])
    if args.cet_weight is not None:
        loss_cfg["cet_weight"] = args.cet_weight
    weight = float(loss_cfg["cet_weight"])
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("the enabled smoke-test CET weight must be positive and finite")
    for key in ("vpe_path", "c_weight_path"):
        loss_cfg[key] = str((ROOT / loss_cfg[key]).resolve())
    setup_seed(args.seed)
    model = InterationSegmentMDM(**model_cfg).to(device)
    # Frozen CLIP stays resident; all other parameters/buffers reset per arm.
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("clip_model.")}
    batch = map_copy_select_to(batch, device=device, dtype=torch.float32,
                              select=("mask", "pose_repr", "shape", "obj_num", "obj_traj", "obj_embedding"))
    x_start = batch["pose_repr"].transpose(1, 2).unsqueeze(2)
    diffusion = create_gaussian_diffusion(1000, "cosine")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "samples": metadata, "cache": str(args.cache.resolve()),
              "sample_indices": args.sample_indices, "device": str(device), "model_config": model_cfg,
              "scope": "single-GPU resident-batch optimizer steps; excludes loading/logging; no checkpoints or sampling",
              "profile_cet": args.profile_cet, "warmup_steps_per_arm": args.warmup_steps,
              "measured_steps_per_arm": args.steps, "arms": {}}
    for name, coefficient in (("baseline", 0.0), ("cet", weight)):
        model.load_state_dict(initial, strict=False)
        cfg = dict(loss_cfg, cet_weight=coefficient, cet_profile=args.profile_cet)
        extra = InteractionSegmentExtraLoss(str(args.mano_path), cfg, use_pc=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
        setup_seed(args.seed)  # Coupled timestep/noise/dropout streams for both arms.
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        rows = []
        with SummaryWriter(str(args.output / name)) as writer:
            for step in range(args.warmup_steps + args.steps):
                optimizer.zero_grad()
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                timestep = (0, 500, 999)[step % 3]
                t = torch.full((x_start.shape[0],), timestep, device=device, dtype=torch.long)
                terms, (aux, detail) = diffusion.training_losses(model, x_start, t,
                                                                 model_kwargs={"batch": batch}, loss_callback=extra)
                diffusion_loss = terms["loss"].mean()
                total = diffusion_loss + aux
                if not torch.isfinite(total):
                    raise RuntimeError(f"{name} step {step}: nonfinite total loss")
                total.backward()
                finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if not finite or not torch.stack(finite).all():
                    raise RuntimeError(f"{name} step {step}: absent/nonfinite gradients")
                clip_gradient(optimizer, .1, 2.0)
                optimizer.step()
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - begin
                row = {"step": step, "timestep": timestep, "seconds": seconds,
                       "diffusion_loss": float(diffusion_loss.detach()), "total_loss": float(total.detach()),
                       **{k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in detail.items()}}
                for key, value in row.items():
                    writer.add_scalar(key, value, step)
                if step >= args.warmup_steps:
                    rows.append(row)
                print(json.dumps({"arm": name, **row}), flush=True)
        changed = any(not torch.equal(parameter.detach().cpu(), initial[key])
                      for key, parameter in model.named_parameters() if parameter.requires_grad)
        if not changed:
            raise RuntimeError(f"{name}: generator weights did not update")
        report["arms"][name] = {"cet_weight": coefficient, "weights_updated": changed, "steps": rows,
                                "median_step_seconds": statistics.median(r["seconds"] for r in rows),
                                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)}
        del extra, optimizer
    baseline = report["arms"]["baseline"]["median_step_seconds"]
    enabled = report["arms"]["cet"]["median_step_seconds"]
    report["median_overhead_seconds"] = enabled - baseline
    report["median_overhead_percent"] = 100 * (enabled / baseline - 1)
    report["timing_caveat"] = "Three-step smoke timing is noisy; profiling synchronization and finite-gradient checks add overhead."
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="existing training-split cache")
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--data-prefix", type=Path, default=ROOT / "data")
    parser.add_argument("--obj-embedding-prefix", type=Path, default=ROOT / "common/retrieve_obj_embedding/main/embedding")
    parser.add_argument("--obj-pointcloud-prefix", type=Path, default=ROOT / "common/retrieve_obj_pointcloud/main/pointcloud")
    parser.add_argument("--mano-path", type=Path, default=ROOT / "asset/mano_v1_2")
    parser.add_argument("--model-config", type=Path, default=ROOT / "config/arch_mdm_l.yml")
    parser.add_argument("--loss-config", type=Path, default=ROOT / "config/loss_param.yml")
    parser.add_argument("--cet-config", type=Path, default=ROOT / "config/cet_smoke.yml")
    parser.add_argument("--cet-weight", type=float)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile-cet", action="store_true", help="synchronized CET forward and CPU-DTW timings")
    parser.add_argument("--check-inputs-only", action="store_true", help="verify cached batch; no model or optimizer steps")
    parser.add_argument("--output", type=Path, help="new or empty experiment directory, required for optimizer steps")
    args = parser.parse_args()
    if not 1 <= args.steps <= 10 or not 0 <= args.warmup_steps <= 2 or not 1 <= len(args.sample_indices) <= 4:
        parser.error("bounded smoke test: 1–10 measured steps, 0–2 warmups, 1–4 samples")
    if not args.check_inputs_only:
        if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
            parser.error("optimizer smoke test requires CUDA; use --check-inputs-only for input inspection")
        if args.output is None or (args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir()))):
            parser.error("--output must name a new or empty directory")
    run(args)


if __name__ == "__main__":
    main()
