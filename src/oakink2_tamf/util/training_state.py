"""Complete G-stage checkpoints at completed epoch boundaries only."""
from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..dataset.cache_util import publish_new, sha256_file, sha256_source, manifest_path

LOGGER = logging.getLogger(__name__)
STATE_VERSION = 1


def capture_rng(device=None):
    cuda = device is not None and torch.device(device).type == "cuda"
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if cuda else None)


def restore_rng(state, device=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if device is None or torch.device(device).type != "cuda":
            raise ValueError("CUDA RNG checkpoint requires a CUDA device")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def make_signature(run_cfg, cache_identities, world_size):
    """Path-independent identity of data, assets and the actual training objective."""
    root = Path(__file__).resolve().parents[3]
    code = ("model/interaction_segment_mdm.py", "model/interaction_segment_extra_loss.py",
            "model/loss/cet.py", "model/diffusion_util.py", "model/diffusion/gaussian_diffusion.py",
            "dataset/training_data.py", "dataset/collate.py")
    loss = dict(run_cfg["train"]["loss"])
    assets = {}
    for key in ("vpe_path", "c_weight_path"):
        if key in loss:
            assets[key] = sha256_file(loss.pop(key))
    mano = Path(run_cfg["mano"]["mano_path"])
    assets["clip_loader"] = sha256_source(root / "thirdparty/CLIP/clip/clip.py")
    assets["clip_tokenizer"] = sha256_source(root / "thirdparty/CLIP/clip/simple_tokenizer.py")
    assets["clip_vocabulary"] = sha256_file(root / "thirdparty/CLIP/clip/bpe_simple_vocab_16e6.txt.gz")
    for side in ("LEFT", "RIGHT"):
        assets["MANO_" + side] = sha256_file(mano / "models" / ("MANO_" + side + ".pkl"))
    for key, extension in (("obj_embedding_prefix", ".pt"), ("obj_pointcloud_prefix", ".npz")):
        prefix = run_cfg["data"].get(key)
        if prefix:
            # Inventory is small; include geometry/embedding content, not paths.
            assets[key] = {p.name: sha256_file(p) for p in sorted(Path(prefix).glob("*" + extension))}
    if run_cfg["data"].get("enable_obj_model"):
        mesh_root = Path(run_cfg["data"]["data_prefix"]) / "object_raw"
        assets["object_meshes"] = {
            p.relative_to(mesh_root).as_posix(): sha256_file(p)
            for p in sorted(mesh_root.rglob("*")) if p.is_file() and p.suffix.lower() in
            (".obj", ".ply", ".stl", ".json", ".npy", ".npz")}
    return dict(version=STATE_VERSION, model=run_cfg["model"], loss=loss, assets=assets,
                cache=cache_identities, world_size=world_size,
                batch_size=run_cfg["train"]["batch_size"],
                num_worker=run_cfg["runtime"]["num_worker"], seed=run_cfg["runtime"]["seed"],
                reverse=run_cfg["data"].get("append_reverse_segment", False),
                enable_obj_model=run_cfg["data"].get("enable_obj_model", False),
                eval_frequencies=[run_cfg["val"]["val_freq"], run_cfg["test"]["test_freq"]],
                scheduler_milestone=run_cfg["train"]["scheduler_milestone"],
                scheduler_gamma=run_cfg["train"]["scheduler_gamma"],
                diffusion=dict(steps=1000, noise_schedule="cosine", schedule_sampler="uniform"),
                code={name: sha256_source(root / "src/oakink2_tamf" / name) for name in code})


def resolve_checkpoint(value, experiment_dir):
    if value != "latest":
        path = Path(value)
        verify_checkpoint(path)
        return path
    candidates = sorted((Path(experiment_dir) / "save").glob("state_epoch_*.pt"), reverse=True)
    for path in candidates:
        try:
            verify_checkpoint(path)
            return path
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.warning("Skipping incomplete/corrupt checkpoint %s: %s", path, exc)
    raise FileNotFoundError("no complete epoch checkpoint exists in this experiment")


def verify_checkpoint(path):
    path = Path(path)
    metadata = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    if metadata.get("version") != STATE_VERSION or path.stat().st_size != metadata["size_bytes"]:
        raise ValueError("checkpoint version/size mismatch")
    if sha256_file(path) != metadata["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    return metadata


def validate_resume_destination(value, experiment_dir):
    """An older checkpoint needs a new experiment ID, not a rewind in place."""
    selected = resolve_checkpoint(value, experiment_dir)
    if value != "latest":
        try:
            newest = resolve_checkpoint("latest", experiment_dir)
        except FileNotFoundError:
            newest = None
        if newest is not None and verify_checkpoint(newest)["completed_epoch"] > verify_checkpoint(selected)["completed_epoch"]:
            raise ValueError("older checkpoint would repeat completed epochs in this experiment; choose a new exp_id")
    return selected.resolve()


def save_checkpoint(path, model, optimizer, scheduler, *, completed_epoch,
                    global_updates, signature, rank_states):
    # Frozen CLIP is recreated by its original loader, as in the legacy exports.
    weights = {k: v for k, v in model.state_dict().items() if not k.startswith("clip_model.")}
    state = dict(version=STATE_VERSION, model=weights, optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), completed_epoch=completed_epoch,
                 next_epoch=completed_epoch + 1, global_updates=global_updates,
                 next_logging_step=global_updates * signature["world_size"],
                 signature=signature, rank_states=rank_states,
                 torch_version=torch.__version__, numpy_version=np.__version__)
    if Path(path).exists() or manifest_path(path).exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    publish_new(path, lambda stream: torch.save(state, stream))
    metadata = dict(version=STATE_VERSION, completed_epoch=completed_epoch,
                    size_bytes=Path(path).stat().st_size, sha256=sha256_file(path))
    publish_new(manifest_path(path), lambda stream: stream.write(json.dumps(metadata).encode()))


def restore_checkpoint(path, model, optimizer, scheduler, signature, rank, device=None, generator=None):
    metadata = verify_checkpoint(path)
    state = torch.load(path, map_location="cpu")
    if state.get("version") != STATE_VERSION or state.get("signature") != signature:
        raise ValueError("checkpoint configuration/data/assets are incompatible with this run")
    if (state["completed_epoch"] != metadata["completed_epoch"] or
            state["next_epoch"] != state["completed_epoch"] + 1 or
            len(state["rank_states"]) != signature["world_size"] or
            not isinstance(state["global_updates"], int) or state["global_updates"] < 0 or
            state["next_logging_step"] != state["global_updates"] * signature["world_size"]):
        raise ValueError("invalid epoch-boundary checkpoint state")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected or any(not key.startswith("clip_model.") for key in missing):
        raise ValueError(f"checkpoint model keys differ: missing={missing}, unexpected={unexpected}")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rank_state = state["rank_states"][rank]
    if generator is not None:
        generator.set_state(rank_state["loader_rng"].cpu())
    restore_rng(rank_state["rng"], device)
    return state["next_epoch"], state["global_updates"]


def save_distributed_checkpoint(path, model, optimizer, scheduler, *, completed_epoch,
                                global_updates, signature, rank, world_size, device, generator):
    local = dict(rng=capture_rng(device), loader_rng=generator.get_state())
    states = [None] * world_size
    dist.all_gather_object(states, local)
    result = [None]
    if rank == 0:
        try:
            save_checkpoint(path, model, optimizer, scheduler, completed_epoch=completed_epoch,
                            global_updates=global_updates, signature=signature, rank_states=states)
        except Exception as exc:
            result[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(result, src=0)
    if result[0] is not None:
        raise RuntimeError("checkpoint publication failed: " + result[0])


def prepare_experiment(experiment_dir, *, commit, resume):
    """Never rotate/overwrite a previous experiment; logs/configs are per session."""
    path = Path(experiment_dir)
    if commit and not resume and path.exists() and any(path.iterdir()):
        raise FileExistsError(f"experiment already exists: {path}; resume explicitly or choose a new exp_id")
    if resume and not commit:
        raise ValueError("resumption requires --commit")
    session = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    return session


def select_cuda_devices(device_ids, available, visibility=None):
    """CLI IDs index the current visibility mask, including Slurm UUID/MIG masks."""
    if not device_ids or len(set(device_ids)) != len(device_ids):
        raise ValueError("device IDs must be nonempty and unique")
    if any(i < 0 or i >= available for i in device_ids):
        raise ValueError("requested logical CUDA device is unavailable")
    if visibility is None:
        return ",".join(str(i) for i in device_ids)
    tokens = visibility.split(",")
    if any(i >= len(tokens) or not tokens[i].strip() for i in device_ids):
        raise ValueError("CUDA visibility mask does not contain requested devices")
    return ",".join(tokens[i].strip() for i in device_ids)


def stop_at_boundary(requested, deadline, epoch_seconds, *, now=None, reserve=600):
    now = time.monotonic() if now is None else now
    return bool(requested or (deadline is not None and now + 1.25 * epoch_seconds + reserve >= deadline))
