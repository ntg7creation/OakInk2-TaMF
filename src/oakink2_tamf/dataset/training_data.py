"""G-stage split construction and value-equivalent virtual reverse samples."""
from __future__ import annotations

import logging

import torch

from .cache_util import read_cache, sha256_file, memory_status

LOGGER = logging.getLogger(__name__)


def evaluation_enabled(frequency):
    if frequency is None or frequency == -1:
        return False
    if not isinstance(frequency, int) or frequency < 1:
        raise ValueError("evaluation frequency must be positive, None, or -1")
    return True


def recording_due(epoch, total_epochs, frequency):
    return evaluation_enabled(frequency) and (
        epoch == 0 or epoch % frequency == frequency - 1 or epoch == total_epochs - 1)


class ReverseSegmentView(torch.utils.data.Dataset):
    """Original N samples followed by N reversed samples; no corpus-sized copies."""
    def __init__(self, dataset):
        self.base = dataset

    def __len__(self):
        return 2 * len(self.base)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample = self.base[index % len(self.base)]
        if index < len(self.base):
            return sample
        result = dict(sample)
        length = sample["len"]
        for key in ("pose_repr", "shape"):
            arr = sample[key].copy()
            arr[:length] = sample[key][:length][::-1]
            result[key] = arr
        arr = sample["obj_traj"].copy()
        arr[:, :length] = sample["obj_traj"][:, :length][:, ::-1]
        result["obj_traj"] = arr
        result["frame_id"] = sample["frame_id"][::-1]
        return result


def build_datasets(run_cfg, rank, dataset_factory=None):
    if dataset_factory is None:
        from .interaction_segment import InteractionSegmentData
        dataset_factory = InteractionSegmentData
    # Validate both frequencies on every rank, before any expensive construction.
    enabled = {"train": True,
               "val": evaluation_enabled(run_cfg["val"]["val_freq"]),
               "test": evaluation_enabled(run_cfg["test"]["test_freq"])}
    datasets = dict.fromkeys(enabled)
    cache_identities = {}
    data = run_cfg["data"]
    for split, active in enabled.items():
        if not active or (split != "train" and rank not in (None, 0)):
            continue
        cfg = run_cfg[split]
        path = cfg.get("cache_dict_filepath")
        strict = run_cfg["train"].get("require_verified_cache", False)
        if strict and not path:
            raise ValueError(f"{split}: --train.require_verified_cache requires a cache path")
        cache = None
        if path:
            cache, manifest = read_cache(path, cfg["process_range"],
                                         require_manifest=strict, split_name=split)
            cache_identities[split] = dict(sha256=manifest["sha256"] if manifest else sha256_file(path),
                                           sequence_keys=list(cfg["process_range"]))
        dataset = dataset_factory(
            process_range_list=cfg["process_range"], data_prefix=data["data_prefix"],
            obj_embedding_prefix=data.get("obj_embedding_prefix"),
            obj_pointcloud_prefix=data.get("obj_pointcloud_prefix"),
            enable_obj_model=data.get("enable_obj_model", False), cache_dict=cache,
            append_reverse_segment=False, rank=rank)
        del cache
        if split == "train" and data.get("append_reverse_segment", False):
            dataset = ReverseSegmentView(dataset)
        datasets[split] = dataset
        LOGGER.info("%s dataset: %d samples (cached=%s) | %s", split, len(dataset), bool(path), memory_status())
    return datasets, cache_identities


def build_loaders(datasets, run_cfg, rank, world_size, collate_fn, generator):
    workers = run_cfg["runtime"]["num_worker"]
    if workers < 0:
        raise ValueError("num_worker must be nonnegative")
    global_batch = run_cfg["train"]["batch_size"]
    if global_batch < world_size:
        raise ValueError("training batch_size must be at least world_size")
    if global_batch % world_size:
        LOGGER.warning("batch_size %d is not divisible by world_size %d", global_batch, world_size)
    sampler = torch.utils.data.distributed.DistributedSampler(
        datasets["train"], num_replicas=world_size, rank=rank or 0, shuffle=True)
    loaders = {}
    for split, dataset in datasets.items():
        if dataset is None:
            loaders[split] = None
            continue
        loaders[split] = torch.utils.data.DataLoader(
            dataset, collate_fn=collate_fn,
            batch_size=global_batch // world_size if split == "train" else run_cfg[split]["batch_size"],
            num_workers=workers, sampler=sampler if split == "train" else None,
            multiprocessing_context="spawn" if workers > 0 else None,
            shuffle=False, drop_last=split == "train", pin_memory=True,
            persistent_workers=workers > 0, generator=generator)
    if not len(loaders["train"]):
        raise ValueError("training loader has zero batches; reduce batch_size or use more samples")
    return loaders, sampler
