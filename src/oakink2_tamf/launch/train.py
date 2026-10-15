import os
import numpy as np
import torch
import cv2
import json

import logging
import argparse
import itertools
import shutil
import tqdm
import time
import pickle
import signal
from pathlib import Path

import torch.multiprocessing as mp
import torch.distributed as dist
from ..util import ddp_util

from config_reg import ConfigRegistry, ConfigEntrySource
from config_reg import (
    ConfigEntryCommandlineBoolPattern,
    ConfigEntryCommandlineSeqPattern,
)
from config_reg.callback import abspath_callback
from dev_fn.upkeep import log
from ..util import log_suppress
from dev_fn.upkeep import ckpt
from dev_fn.upkeep.opt import argdict_to_string
from dev_fn.upkeep.config import cb__decode_file, cls__cb__link_bool_opt

from dev_fn.util.console_io import suppress_trimesh_logging
from dev_fn.util import pbar_util
from dev_fn.util import random_util

from dev_fn.transform.cast import map_copy_select_to
from ..dataset.interaction_segment import InteractionSegmentData
from ..dataset.collate import interaction_segment_collate
from ..dataset.training_data import build_datasets, build_loaders, evaluation_enabled, recording_due
from ..dataset.cache_util import publish_new
from ..util.training_state import (
    make_signature, prepare_experiment, resolve_checkpoint, restore_checkpoint, validate_resume_destination,
    save_distributed_checkpoint, select_cuda_devices, stop_at_boundary,
)
from ..model.interaction_segment_mdm import InterationSegmentMDM
from ..model.interaction_segment_extra_loss import InteractionSegmentExtraLoss
from ..model.diffusion_util import create_gaussian_diffusion
from ..model.diffusion.resample import create_named_schedule_sampler
from ..util.net_util import clip_gradient
from ..util.state_util import save_state
from ..util.summary_writer import DDPSummaryWriter

from .param import reg_mano_param, reg_loss_param, reg_model_param

_logger = logging.getLogger(__name__)

PROG = "train"
WS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

PARAM_PREFIX__DATA = "data"
PARAM_PREFIX__TRAIN = "train"
PARAM_PREFIX__VAL = "val"
PARAM_PREFIX__TEST = "test"
PARAM_PREFIX__RUNTIME = "runtime"


def reg_entry(config_reg: ConfigRegistry):
    config_reg.register("require_verified_cache", prefix=PARAM_PREFIX__TRAIN, category=bool,
                        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
                        cmdpattern=ConfigEntryCommandlineBoolPattern.SET_TRUE, default=False)
    config_reg.register("resume_checkpoint", prefix=PARAM_PREFIX__TRAIN, category=str,
                        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG, default=None)
    config_reg.register("max_seconds", prefix=PARAM_PREFIX__RUNTIME, category=float,
                        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG, default=None)
    # base
    config_reg.register(
        "data_prefix",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=f"{WS_DIR}/data",
        required=True,
    )
    config_reg.register(
        "append_reverse_segment",
        prefix=PARAM_PREFIX__DATA,
        category=bool,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineBoolPattern.SET_TRUE,
        default=False,
    )
    config_reg.register(
        "enable_obj_model",
        prefix=PARAM_PREFIX__DATA,
        category=bool,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineBoolPattern.SET_TRUE,
        default=False,
    )

    # obj
    config_reg.register(
        "obj_embedding_prefix",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "obj_pointcloud_prefix",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    # mano
    reg_mano_param(config_reg, "mano", WS_DIR)

    # train
    config_reg.register(
        "process_range",
        prefix=PARAM_PREFIX__TRAIN,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
        callback=cb__decode_file,
        default=[f"?(file:{WS_DIR}/asset/split/train.txt)"],
    )
    config_reg.register(
        "cache_dict_filepath",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "batch_size",
        prefix=PARAM_PREFIX__TRAIN,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=16,
    )
    config_reg.register(
        "num_epoch",
        prefix=PARAM_PREFIX__TRAIN,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=400,
    )
    config_reg.register(
        "record_freq",
        prefix=PARAM_PREFIX__TRAIN,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=20,
    )
    config_reg.register(
        "scheduler_milestone",
        prefix=PARAM_PREFIX__TRAIN,
        category=list[int],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COMMA_SEP,
        default=[150, 250],
    )
    config_reg.register(
        "scheduler_gamma",
        prefix=PARAM_PREFIX__TRAIN,
        category=float,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0.5,
    )
    reg_loss_param(config_reg, f"{PARAM_PREFIX__TRAIN}.loss", WS_DIR)
    config_reg.register(
        "reload_ckpt_model_filepath",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "reload_ckpt_optimizer_filepath",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )

    # val
    config_reg.register(
        "process_range",
        prefix=PARAM_PREFIX__VAL,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
        callback=cb__decode_file,
        default=[f"?(file:{WS_DIR}/asset/split/val.txt)"],
    )
    config_reg.register(
        "cache_dict_filepath",
        prefix=PARAM_PREFIX__VAL,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "batch_size",
        prefix=PARAM_PREFIX__VAL,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=8,
    )
    config_reg.register(
        "val_freq",
        prefix=PARAM_PREFIX__VAL,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=20,
    )

    # test
    config_reg.register(
        "process_range",
        prefix=PARAM_PREFIX__TEST,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
        callback=cb__decode_file,
        default=[f"?(file:{WS_DIR}/asset/split/test.txt)"],
    )
    config_reg.register(
        "cache_dict_filepath",
        prefix=PARAM_PREFIX__TEST,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "batch_size",
        prefix=PARAM_PREFIX__TEST,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=8,
    )
    config_reg.register(
        "test_freq",
        prefix=PARAM_PREFIX__TEST,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=100,
    )

    # model
    reg_model_param(config_reg, "model")

    ## runtime config
    config_reg.register(
        "num_worker",
        prefix=PARAM_PREFIX__RUNTIME,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_ONLY,
        default=2,
    )
    config_reg.register(
        "device_id",
        prefix=PARAM_PREFIX__RUNTIME,
        category=list[int],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COMMA_SEP,
        default=[0],
    )
    config_reg.register(
        "seed",
        prefix=PARAM_PREFIX__RUNTIME,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0,
    )


def reg_extract(config_reg: ConfigRegistry):
    res = {}
    for _p in [
        PARAM_PREFIX__DATA,
        PARAM_PREFIX__TRAIN,
        PARAM_PREFIX__VAL,
        PARAM_PREFIX__TEST,
        PARAM_PREFIX__RUNTIME,
        "model",
        "mano",
    ]:
        try:
            res[_p] = config_reg.select(_p)
        except KeyError:
            pass
    return res


def summary_extra_loss(summary_writer, loss_store, extra_loss_store, lr=None, step_id=None):
    time_curr = time.time()
    summary_writer.add_scalar(
        tag="total_loss", value=float(loss_store["diffusion_loss"]) + float(extra_loss_store["loss"]),
        global_step=step_id, walltime=time_curr,
    )
    summary_writer.add_scalar(
        tag="diffusion_loss", value=float(loss_store["diffusion_loss"]), global_step=step_id, walltime=time_curr
    )
    for loss_name, loss_node in extra_loss_store.items():
        if loss_name == "loss":
            summary_writer.add_scalar(tag="extra_loss", value=float(loss_node), global_step=step_id, walltime=time_curr)
        else:
            summary_writer.add_scalar(
                tag=f"extra_loss/{loss_name}", value=float(loss_node), global_step=step_id, walltime=time_curr
            )
    if lr is not None:
        summary_writer.add_scalar(tag="lr", value=float(lr), global_step=step_id, walltime=time_curr)


def run(rank, world_size, ckpt_cfg, run_cfg, session_id, stop_event, deadline):
    log.log_init()
    log.enable_console()
    ddp_util.log_status_update(rank)  # disable root_logger on non_zero rank

    session_path = Path(ckpt_cfg["ckpt_path"]) / "sessions" / session_id
    session_cfg = dict(ckpt_cfg, ckpt_path=str(session_path), log_file=str(session_path / "log.txt"))
    ckpt.ckpt_setup(session_cfg, rank=rank)
    ckpt.ckpt_opt(session_cfg, rank=rank, world_size=world_size, ckpt=ckpt_cfg, run=run_cfg)

    _logger.info("world_size: %d", world_size)
    _logger.info("ckpt_cfg: %s", argdict_to_string(ckpt_cfg))
    _logger.info("run_cfg: %s", argdict_to_string(run_cfg))
    log_suppress.suppress()
    suppress_trimesh_logging()

    # device
    device_id_list = run_cfg["runtime"]["device_id"]
    device_id = device_id_list[rank] if rank is not None else device_id_list[0]
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    dtype = torch.float32
    ddp_util.setup_ddp(rank, world_size)

    # ckpt
    commit = ckpt_cfg["commit"]
    ckpt_path = ckpt_cfg["ckpt_path"]

    # Seed initialization as well as training; rank zero's weights are broadcast.
    seed = run_cfg["runtime"]["seed"] + rank
    random_util.setup_seed(seed)
    datasets, cache_identities = build_datasets(run_cfg, rank)
    loader_generator = torch.Generator().manual_seed(seed)
    loaders, train_sampler = build_loaders(
        datasets, run_cfg, rank, world_size, interaction_segment_collate, loader_generator)
    train_dataloader, val_dataloader, test_dataloader = (loaders[k] for k in ("train", "val", "test"))
    batch_size = run_cfg["train"]["batch_size"] // world_size
    # Evaluation caches exist only on rank zero; every rank must compare the
    # same complete data identity when resuming a distributed checkpoint.
    identities = [cache_identities if rank == 0 else None]
    dist.broadcast_object_list(identities, src=0)
    cache_identities = identities[0]
    signature = make_signature(run_cfg, cache_identities, world_size)

    # model
    model_cfg = run_cfg["model"]
    model_ = InterationSegmentMDM(
        input_dim=model_cfg["input_dim"],
        obj_input_dim=model_cfg["obj_input_dim"],
        hand_shape_dim=model_cfg["hand_shape_dim"],
        obj_embed_dim=model_cfg["obj_embed_dim"],
        latent_dim=model_cfg["latent_dim"],
        ff_size=model_cfg["ff_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
    ).to(device)
    # resume model
    if (reload_ckpt_model_filepath := run_cfg["train"]["reload_ckpt_model_filepath"]) is not None:
        _logger.info("model reload ckpt: %s", reload_ckpt_model_filepath)
        reload_ckpt_model_weight = torch.load(reload_ckpt_model_filepath, map_location=device)
        missing_key_list, unexpected_key_list = model_.load_state_dict(reload_ckpt_model_weight, strict=False)
        missing_key_list = [k for k in missing_key_list if not k.startswith("clip_model")]
        _logger.info("model reload ckpt missing_key_list: %s", missing_key_list)
        _logger.info("model reload ckpt unexpected_key_list: %s", unexpected_key_list)
    model = torch.nn.parallel.DistributedDataParallel(
        model_,
        device_ids=[device],
        output_device=device,
        # find_unused_parameters=True,
    )
    model_extra_loss = InteractionSegmentExtraLoss(
        run_cfg["mano"]["mano_path"],
        run_cfg["train"]["loss"],
        use_pc=run_cfg["data"]["obj_pointcloud_prefix"] is not None,
    ).to(device)
    diffusion = create_gaussian_diffusion(diffusion_steps=1000, noise_schedule="cosine")
    schedule_sampler = create_named_schedule_sampler(name="uniform", diffusion=diffusion)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    # resume optimizer
    if (reload_ckpt_optimizer_filepath := run_cfg["train"]["reload_ckpt_optimizer_filepath"]) is not None:
        _logger.info("optimzier reload ckpt: %s", reload_ckpt_optimizer_filepath)
        reload_ckpt_optimizer_param = torch.load(reload_ckpt_optimizer_filepath, map_location=device)
        optimizer.load_state_dict(reload_ckpt_optimizer_param)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=run_cfg["train"]["scheduler_milestone"],
        gamma=run_cfg["train"]["scheduler_gamma"],
    )

    # summary writer
    summary_writer_dir = ckpt.handle_save_path(f"?(ckpt_path)/summary", ckpt_path=ckpt_path) if commit else None
    summary_writer = DDPSummaryWriter(log_dir=summary_writer_dir, rank=rank)

    # seed
    seed = run_cfg["runtime"]["seed"]
    if rank is not None:
        seed = seed + rank
    random_util.setup_seed(seed)

    start_epoch, global_updates = 0, 0
    resume = run_cfg["train"].get("resume_checkpoint")
    if resume:
        checkpoint_path = resolve_checkpoint(resume, ckpt_path)
        start_epoch, global_updates = restore_checkpoint(
            checkpoint_path, model_, optimizer, scheduler, signature, rank,
            device=device, generator=loader_generator)
        if global_updates != start_epoch * len(train_dataloader):
            raise ValueError("checkpoint update count does not match completed epochs and loader size")
        _logger.info("Resumed %s at epoch %d, updates %d", checkpoint_path, start_epoch, global_updates)
    if start_epoch >= run_cfg["train"]["num_epoch"]:
        raise ValueError("num_epoch is a total target and must exceed the checkpoint's completed epochs")

    # mainloop
    dist.barrier()
    num_epoch = run_cfg["train"]["num_epoch"]
    for epoch_id in range(start_epoch, num_epoch):
        epoch_start = time.monotonic()
        # epoch begin
        train_sampler.set_epoch(epoch_id)
        # Dataset transforms are deterministic. Epoch-specific worker seeds also
        # keep worker restart from consuming the model/noise RNG streams.
        loader_generator.manual_seed(seed + 1000003 * epoch_id)

        # epoch iterate
        pbar = (
            tqdm.tqdm(
                total=len(train_dataloader), position=0, bar_format=pbar_util.fmt, desc=f"train epoch {epoch_id:>04d}:"
            )
            if not rank
            else pbar_util.dummy_pbar()
        )
        for batch_id, batch in enumerate(train_dataloader):
            step_id = (epoch_id * len(train_dataloader) + batch_id) * world_size

            optimizer.zero_grad()
            model.train()

            batch_device = map_copy_select_to(
                batch,
                device=device,
                dtype=dtype,
                select=("mask", "pose_repr", "shape", "obj_num", "obj_traj", "obj_embedding"),
            )
            t, weights = schedule_sampler.sample(batch_size, device)
            x_start = batch_device["pose_repr"]  # (bs, seqlen, in_dim)
            x_start = x_start.unsqueeze(3)  # (bs, seqlen, in_dim, 1)
            x_start = x_start.permute((0, 2, 3, 1))  # (bs, in_dim, 1, seqlen)
            loss_store, (extra_loss, extra_loss_store) = diffusion.training_losses(
                model, x_start, t, model_kwargs={"batch": batch_device}, loss_callback=model_extra_loss
            )
            loss = (loss_store["loss"] * weights).mean()
            loss_store["diffusion_loss"] = loss
            loss = loss + extra_loss

            loss.backward()
            clip_gradient(optimizer, 0.1, 2.0)
            optimizer.step()
            optimizer.zero_grad()
            global_updates += 1

            # summary writer
            summary_extra_loss(
                summary_writer,
                loss_store,
                extra_loss_store,
                lr=float(next(iter(optimizer.param_groups))["lr"]),
                step_id=step_id,
            )

            pbar.update()
        pbar.close()

        # epoch end
        scheduler.step()
        dist.barrier()
        _logger.info("train epoch %04d conclude | loss: %f", epoch_id, loss.item())
        summary_writer.add_scalar(tag="epoch", value=epoch_id, global_step=step_id)
        _logger.info("train epoch %04d lr %s", epoch_id, [group["lr"] for group in optimizer.param_groups])
        _logger.info("train epoch %04d detail loss:", epoch_id)
        _logger.info("                     %s: %f", "diffusion_loss".ljust(20), float(loss_store["diffusion_loss"]))
        for loss_name, loss_node in extra_loss_store.items():
            if loss_name in ["loss"]:
                continue
            _logger.info("                     %s: %f", loss_name.ljust(20), float(loss_node))

        if not rank:
            # val
            val_freq = run_cfg["val"]["val_freq"]
            if (val_freq is not None and val_freq != -1) and (
                epoch_id == 0 or epoch_id % val_freq == val_freq - 1 or epoch_id == num_epoch - 1
            ):
                with torch.no_grad():
                    val_pbar = tqdm.tqdm(
                        total=len(val_dataloader),
                        position=0,
                        bar_format=pbar_util.fmt,
                        desc=f"val   epoch {epoch_id:>04d}:",
                    )
                    sample_fn = diffusion.p_sample_loop
                    model_.eval()
                    for val_batch_id, val_batch in enumerate(val_dataloader):
                        val_batch_device = map_copy_select_to(
                            val_batch,
                            device=device,
                            dtype=dtype,
                            select=("mask", "pose_repr", "shape", "obj_num", "obj_traj", "obj_embedding"),
                        )
                        val_input_shape = tuple(val_batch_device["pose_repr"].shape)
                        val_input_shape = (val_input_shape[0], val_input_shape[2], 1, val_input_shape[1])
                        val_sample = sample_fn(
                            model_,
                            val_input_shape,  # adapt from mdm
                            clip_denoised=False,
                            model_kwargs={"batch": val_batch_device},
                            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                            init_image=None,
                            progress=False,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,
                        )

                        val_pbar.update()
                    val_pbar.close()
                    _logger.info("val epoch %04d conclude", epoch_id)

            # test
            test_freq = run_cfg["test"]["test_freq"]
            if (test_freq is not None and test_freq != -1) and (
                epoch_id == 0 or epoch_id % test_freq == test_freq - 1 or epoch_id == num_epoch - 1
            ):
                with torch.no_grad():
                    test_pbar = tqdm.tqdm(
                        total=len(test_dataloader),
                        position=0,
                        bar_format=pbar_util.fmt,
                        desc=f"test  epoch {epoch_id:>04d}:",
                    )
                    sample_fn = diffusion.p_sample_loop
                    model_.eval()
                    for test_batch_id, test_batch in enumerate(test_dataloader):
                        test_batch_device = map_copy_select_to(
                            test_batch,
                            device=device,
                            dtype=dtype,
                            select=("mask", "pose_repr", "shape", "obj_num", "obj_traj", "obj_embedding"),
                        )
                        test_input_shape = tuple(test_batch_device["pose_repr"].shape)  # (bs, seqlen, in_dim, 1)
                        test_input_shape = (test_input_shape[0], test_input_shape[2], 1, test_input_shape[1])
                        test_sample = sample_fn(
                            model_,
                            test_input_shape,  # adapt from mdm
                            clip_denoised=False,
                            model_kwargs={"batch": test_batch_device},
                            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                            init_image=None,
                            progress=False,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,
                        )

                        test_pbar.update()
                    test_pbar.close()
                    _logger.info("test epoch %04d conclude", epoch_id)

        # All ranks finish evaluation before capturing the next epoch's RNG.
        dist.barrier()
        decision = [None]
        if rank == 0:
            elapsed = time.monotonic() - epoch_start
            decision[0] = stop_at_boundary(stop_event.is_set(), deadline, elapsed)
            _logger.info("epoch wall time %.1fs | stop at boundary=%s", elapsed, decision[0])
        dist.broadcast_object_list(decision, src=0)
        stopping = decision[0]
        due = recording_due(epoch_id, num_epoch, run_cfg["train"]["record_freq"])
        if commit and (due or stopping):
            state_path = Path(ckpt_path) / "save" / f"state_epoch_{epoch_id:04d}_{session_id}.pt"
            save_distributed_checkpoint(
                state_path, model_, optimizer, scheduler, completed_epoch=epoch_id,
                global_updates=global_updates, signature=signature, rank=rank,
                world_size=world_size, device=device, generator=loader_generator)
            # Retain original inference-compatible exports; never replace files.
            result = [None]
            if rank == 0:
                try:
                    weights = {k: v for k, v in model_.state_dict().items() if not k.startswith("clip_model")}
                    for name, state in (("model", weights), ("optimizer", optimizer.state_dict())):
                        output = Path(ckpt_path) / "save" / f"{name}_{epoch_id:04d}.pt"
                        if output.exists():
                            _logger.warning("Preserving existing legacy export %s", output)
                        else:
                            publish_new(output, lambda stream, value=state: torch.save(value, stream))
                except Exception as exc:
                    result[0] = f"{type(exc).__name__}: {exc}"
            dist.broadcast_object_list(result, src=0)
            if result[0]:
                raise RuntimeError("legacy export failed (complete state already saved): " + result[0])
            _logger.info("Complete epoch checkpoint: %s", state_path)
        if stopping:
            _logger.info("Clean epoch-boundary shutdown; partial epochs are never checkpointed")
            break

    # conclude
    if summary_writer.ddp_enabled:
        summary_writer.flush()
        summary_writer.close()
    ddp_util.destroy_ddp()


def main():
    config_reg = ConfigRegistry(prog=PROG)
    ckpt.reg_entry(config_reg)
    reg_entry(config_reg)

    parser = argparse.ArgumentParser(prog=PROG)
    config_reg.hook(parser)
    config_reg.parse(parser)

    ckpt_cfg = ckpt.reg_extract(config_reg)
    run_cfg = reg_extract(config_reg)

    resume = run_cfg["train"]["resume_checkpoint"]
    if resume and (run_cfg["train"]["reload_ckpt_model_filepath"] or
                   run_cfg["train"]["reload_ckpt_optimizer_filepath"]):
        raise ValueError("complete resume cannot be combined with legacy warm-start flags")
    if resume and not run_cfg["train"]["cache_dict_filepath"]:
        raise ValueError("complete resume requires a persistent training cache")
    if resume:
        run_cfg["train"]["resume_checkpoint"] = str(validate_resume_destination(resume, ckpt_cfg["ckpt_path"]))
    evaluation_enabled(run_cfg["val"]["val_freq"])
    evaluation_enabled(run_cfg["test"]["test_freq"])
    evaluation_enabled(run_cfg["train"]["record_freq"])
    max_seconds = run_cfg["runtime"]["max_seconds"]
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 600):
        raise ValueError("runtime.max_seconds must exceed the 600-second shutdown reserve")
    deadline = time.monotonic() + max_seconds if max_seconds is not None else None
    session_id = prepare_experiment(ckpt_cfg["ckpt_path"], commit=ckpt_cfg["commit"], resume=resume)

    device_id_list = run_cfg["runtime"]["device_id"]
    world_size = len(device_id_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = select_cuda_devices(
        device_id_list, torch.cuda.device_count(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    run_cfg["runtime"]["device_id"] = list(range(world_size))

    stop_event = mp.get_context("spawn").Event()
    previous_handler = None
    if hasattr(signal, "SIGUSR1"):
        previous_handler = signal.signal(signal.SIGUSR1, lambda signum, frame: stop_event.set())
    try:
        mp.spawn(run, args=(world_size, ckpt_cfg, run_cfg, session_id, stop_event, deadline), nprocs=world_size)
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGUSR1, previous_handler)


if __name__ == "__main__":
    main()
