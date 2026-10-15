"""CET allocation controller. No training or submission on import or dry-run.

One allocation submits at most one afterok successor, after a successful trainer
exit AND a verified, advancing complete checkpoint. State lives outside the
trainer's experiment directory so epoch-zero initialization stays protected.
Only inspect checkpoints created by your trusted TaMF installation (torch.load).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def workflow_lock(path):
    # A POSIX advisory lock is released even after SIGKILL. Never unlink it:
    # unlinking a locked inode can let a second controller run concurrently.
    import fcntl
    with Path(path).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("experiment controller is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity():
    # Freeze implementation and both configs between allocations. Increasing
    # target epochs is allowed; changing the objective requires a new experiment.
    paths = sorted((ROOT / "src/oakink2_tamf").rglob("*.py"))
    paths += [Path(__file__), ROOT / "script/slurm/cet_train.sbatch", ROOT / "script/debug/smoke_train_cet.py",
              ROOT / "config/shenkar_generator_full.yml", ROOT / "config/cet_experiment.yml"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in paths}


def checkpoint_progress(experiment_dir):
    """Checksum AND load full state on CPU, including optimizer/scheduler/RNG."""
    import torch
    from oakink2_tamf.util.training_state import resolve_checkpoint, verify_checkpoint
    directory = Path(experiment_dir)
    try:
        path = resolve_checkpoint("latest", directory)
    except FileNotFoundError:
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError("nonempty experiment has no verified checkpoint; refuse epoch-zero overwrite")
        return {"next_epoch": 0, "global_updates": 0, "checkpoint": None}
    metadata = verify_checkpoint(path)
    state = torch.load(path, map_location="cpu")
    required = ("model", "optimizer", "scheduler", "signature", "rank_states",
                "completed_epoch", "next_epoch", "global_updates", "next_logging_step")
    if any(key not in state for key in required):
        raise ValueError("checkpoint is not a complete training state")
    world_size = state["signature"]["world_size"]
    if (state.get("version") != 1 or state["completed_epoch"] != metadata["completed_epoch"] or
            state["next_epoch"] != state["completed_epoch"] + 1 or
            not isinstance(state["next_epoch"], int) or state["next_epoch"] <= 0 or
            not isinstance(state["global_updates"], int) or state["global_updates"] <= 0 or
            state["next_logging_step"] != state["global_updates"] * world_size or
            len(state["rank_states"]) != world_size or
            not state["model"] or "param_groups" not in state["optimizer"] or
            "last_epoch" not in state["scheduler"]):
        raise ValueError("invalid complete checkpoint counters/state")
    for rank_state in state["rank_states"]:
        if "loader_rng" not in rank_state or not {"python", "numpy", "torch", "cuda"} <= rank_state["rng"].keys():
            raise ValueError("checkpoint lacks random number generator state")
    return {"next_epoch": state["next_epoch"], "global_updates": state["global_updates"],
            "checkpoint": str(path.resolve())}


def validate_diagnostic(path, weight=0.05):
    import yaml
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    arm = report["arms"]["cet"]
    if (not math.isclose(arm["cet_weight"], weight, rel_tol=0, abs_tol=1e-12) or
            report["warmup_steps_per_arm"] < 1 or report["measured_steps_per_arm"] < 3 or
            len(report["samples"]) != 4 or not arm["weights_updated"]):
        raise ValueError("diagnostic must test coefficient 0.05, batch 4, warmup >=1, measured steps >=3")
    expected = yaml.safe_load((ROOT / "config/shenkar_generator_full.yml").read_text())
    if report["model_config"] != expected["model"]:
        raise ValueError("diagnostic model differs from the working server architecture")
    for key, value in expected["train"]["loss"].items():
        if not key.endswith("_path") and report["loss_config"].get(key) != value:
            raise ValueError("diagnostic original loss coefficients differ from training")
    if report["loss_config"].get("cet_dtw_rho") != .1:
        raise ValueError("diagnostic DTW window differs from training")
    gradient = arm.get("cet_generator_grad_norm") or 0
    rows = arm["steps"]
    numbers = [gradient, arm["median_step_seconds"], arm["peak_cuda_bytes"]]
    numbers += [row[key] for row in rows for key in ("cet", "cet_weighted", "total_loss")]
    if not all(math.isfinite(float(value)) for value in numbers) or gradient <= 0 or not any(row["cet"] > 0 for row in rows):
        raise ValueError("diagnostic lacks finite, nonzero CET and CET-only generator gradients")
    if report.get("code_identity") != code_identity():
        raise ValueError("diagnostic was run against different source/configs; rerun the bounded diagnostic")
    return report


def training_command(state, progress, python=sys.executable):
    target = state["target"]
    if state.get("epochs_per_allocation"):
        target = min(target, progress["next_epoch"] + state["epochs_per_allocation"])
    command = [python, "-u", "-m", "oakink2_tamf.launch.train",
               "--cfg", "config/shenkar_generator_full.yml", "--cfg", "config/cet_experiment.yml",
               "--data.data_prefix", state["data_root"], "--train.cache_dict_filepath", state["cache"],
               "--train.process_range", "?(file:" + state.get("split_file", "asset/split/train.txt") + ")",
               "--train.num_epoch", str(target), "--train.record_freq", "1",
               "--runtime.num_worker", "2", "--runtime.device_id", "0",
               "--runtime.max_seconds", "15600", "--val.val_freq", "-1", "--test.test_freq", "-1",
               "--exp_id", state["experiment"], "--commit"]
    if progress["checkpoint"]:
        command += ["--train.resume_checkpoint", progress["checkpoint"]]
    return command


def submission_command(state, state_path, predecessor=None):
    command = ["sbatch", "--parsable", "--job-name=cet-" + state["experiment"],
               "--partition=" + state["partition"], "--gres=" + state["gres"],
               "--mem=" + state["mem"], "--cpus-per-task=2", "--time=04:30:00",
               "--signal=B:USR1@600", "--no-requeue", "--kill-on-invalid-dep=yes",
               "--chdir=" + str(ROOT), "--output=" + str(Path(state_path).parent / "logs/allocation-%j.log")]
    if predecessor:
        if not str(predecessor).isdigit():
            raise ValueError("invalid predecessor job ID")
        command += ["--dependency=afterok:" + str(predecessor)]
    command += [str(ROOT / "script/slurm/cet_train.sbatch"), str(state_path),
                state["conda_base"], state["conda_env"], str(Path(__file__).resolve())]
    return command


def active_jobs(state):
    result = subprocess.run(["squeue", "--noheader", "--user=" + getpass.getuser(),
                             "--name=cet-" + state["experiment"], "--format=%i|%T|%E"],
                            check=True, capture_output=True, text=True)
    return [line.split("|", 2) for line in result.stdout.splitlines() if line.strip()]


def submit(state, path, predecessor=None):
    # Persist intent BEFORE sbatch. If submission succeeds but the process dies
    # before recording its job ID, refuse automatic retries instead of duplicating.
    if state.get("submission_pending"):
        raise RuntimeError("ambiguous previous submission; inspect squeue and state.json before retrying")
    state["submission_pending"] = True
    atomic_json(path, state)
    command = submission_command(state, path, predecessor)
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError("unexpected sbatch response; inspect queue before retrying")
    state["jobs"].append({"job_id": job_id, "predecessor": predecessor, "submitted_at": time.time()})
    state["submission_pending"] = False
    atomic_json(path, state)
    return job_id


def preflight(state):
    if not Path(state["cache"]).is_file():
        raise FileNotFoundError("existing cache required; this workflow never preprocesses data")
    if not (Path(state["data_root"]) / "anno_preview").is_dir():
        raise FileNotFoundError("data root lacks anno_preview")
    if not Path(state["split_file"]).is_file():
        raise FileNotFoundError("selected split file is missing")
    if not (Path(state["conda_base"]) / "etc/profile.d/conda.sh").is_file():
        raise FileNotFoundError("existing Conda activation script is missing")
    if state["code_identity"] != code_identity():
        raise ValueError("source/configs changed since workflow creation; review before continuing")


def run_training(command):
    child = None
    requested = False

    def forward(signum, frame):
        nonlocal requested
        requested = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGUSR1)

    previous = signal.signal(signal.SIGUSR1, forward)
    try:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        child = subprocess.Popen(command, cwd=ROOT, env=env)
        if requested:
            child.send_signal(signal.SIGUSR1)
        return child.wait()
    finally:
        signal.signal(signal.SIGUSR1, previous)


def worker(path):
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise RuntimeError("worker requires a Slurm batch allocation")
    path = Path(path).resolve()
    with workflow_lock(path.parent / "controller.lock"):
        state = json.loads(path.read_text())
        if state.get("submission_pending") or not any(j["job_id"] == job_id for j in state["jobs"]):
            raise RuntimeError("allocation is not a recorded workflow submission")
        if (path.parent / "STOP").exists():
            print("Stop marker present; no training or successor submission")
            return
        preflight(state)
        before = checkpoint_progress(state["experiment_dir"])
        if before["next_epoch"] >= state["target"]:
            print("Target already reached; no training or successor submission")
            return
        if state["allocations_started"] >= state["max_allocations"]:
            raise RuntimeError("allocation safety cap reached; inspect progress before extending")
        state["allocations_started"] += 1
        state["running"] = {"job_id": job_id, "next_epoch": before["next_epoch"], "started_at": time.time()}
        atomic_json(path, state)
        command = training_command(state, before)
        print("TRAIN " + shlex.join(command), flush=True)
        started = time.monotonic()
        result = run_training(command)
        seconds = time.monotonic() - started
        state["training_process_seconds"] += seconds
        allocation = {"job_id": job_id, "exit_code": result, "seconds": seconds,
                      "start_epoch": before["next_epoch"]}
        state["allocations"].append(allocation)
        state["running"] = None
        atomic_json(path, state)
        if result:
            raise RuntimeError(f"trainer exited {result}; chain stopped, no successor submitted")
        after = checkpoint_progress(state["experiment_dir"])
        allocation["next_epoch"] = after["next_epoch"]
        state["progress"] = after
        atomic_json(path, state)
        if after["next_epoch"] <= before["next_epoch"]:
            raise RuntimeError("trainer exited without a new complete epoch; chain stopped")
        if after["next_epoch"] >= state["target"] or (path.parent / "STOP").exists():
            print("Target reached or stop requested; no successor submitted", flush=True)
            return
        if state["allocations_started"] >= state["max_allocations"]:
            print("Allocation safety cap reached; continuation requires an explicit start", flush=True)
            return
        print("SUCCESSOR " + submit(state, path, predecessor=job_id), flush=True)


def latest_scalars(experiment_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {"unavailable": "activate oakink2-tamf to inspect TensorBoard scalars"}
    result = {}
    for file in Path(experiment_dir).rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(file), size_guidance={"scalars": 1})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if tag in ("total_loss", "diffusion_loss", "extra_loss/cet", "extra_loss/cet_weighted", "lr", "epoch") or "cet_" in tag:
                event = accumulator.Scalars(tag)[-1]
                if tag not in result or event.wall_time > result[tag]["time"]:
                    result[tag] = {"step": event.step, "value": event.value, "time": event.wall_time}
    return result


def start(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.experiment) or args.experiment == "tamf_train_003":
        raise ValueError("choose a safe, distinct CET experiment ID; baseline tamf_train_003 is protected")
    path = ROOT / "common/cet_jobs" / args.experiment / "state.json"
    if path.exists():
        state = json.loads(path.read_text())
        if args.target < state["target"]:
            raise ValueError("target cannot decrease in an existing experiment")
        # Persisted data/resource settings remain authoritative on continuation.
        state["target"] = args.target
        state["max_allocations"] = args.max_allocations or max(state["max_allocations"], args.target)
    else:
        state = dict(version=1, experiment=args.experiment,
                     experiment_dir=str(ROOT / "common/train" / args.experiment),
                     target=args.target, max_allocations=args.max_allocations or args.target,
                     cache=str(args.cache.resolve()), data_root=str(args.data_root.resolve()),
                     split_file=str(args.split_file.resolve()), epochs_per_allocation=args.epochs_per_allocation,
                     conda_base=str(args.conda_base.resolve()), conda_env=args.conda_env,
                     partition=args.partition, gres=args.gres, mem=args.mem,
                     code_identity=code_identity(), jobs=[], allocations=[], allocations_started=0,
                     training_process_seconds=0, running=None, submission_pending=False)
    if not args.submit:
        print("DRY RUN: no files, training, or Slurm submissions")
        print(shlex.join(submission_command(state, path)))
        print("TRAIN (preview from recorded progress; worker reverifies latest checkpoint):")
        preview = state.get("progress", {"checkpoint": None, "next_epoch": 0})
        print(shlex.join(training_command(state, preview)))
        print("Continuation: one successor using --dependency=afterok:<current job>, only after verified progress")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with workflow_lock(path.parent / "controller.lock"):
        if path.exists():
            # A previous worker may have finished after our initial read.
            # Reload under lock before updating target/bookkeeping.
            state = json.loads(path.read_text())
            if args.target < state["target"]:
                raise ValueError("target cannot decrease in an existing experiment")
            state["target"] = args.target
            state["max_allocations"] = args.max_allocations or max(state["max_allocations"], args.target)
        if active_jobs(state):
            raise RuntimeError("active/queued jobs exist for this experiment; no duplicate submission")
        preflight(state)
        progress = checkpoint_progress(state["experiment_dir"])
        if progress["next_epoch"] >= state["target"]:
            print("Target reached; nothing to submit")
            return
        if not state.get("diagnostic"):
            if args.diagnostic_report is None:
                raise ValueError("first submission requires --diagnostic-report from the bounded CET diagnostic")
            report = validate_diagnostic(args.diagnostic_report)
            if Path(report["cache"]).resolve() != Path(state["cache"]).resolve():
                raise ValueError("diagnostic and training must use the same existing cache")
            stat = Path(state["cache"]).stat()
            if report.get("cache_stat") != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
                raise ValueError("cache changed since diagnostic")
            if Path(report["data_root"]).resolve() != Path(state["data_root"]).resolve():
                raise ValueError("diagnostic and training data roots differ")
            state["diagnostic"] = {"path": str(args.diagnostic_report.resolve()),
                                   "sha256": sha256(args.diagnostic_report),
                                   "cet_weight": .05, "reviewed_at": time.time()}
        state["progress"] = progress
        path.parent.joinpath("logs").mkdir(exist_ok=True)
        # Explicit --submit after stopping is the user's continuation request.
        path.parent.joinpath("STOP").unlink(missing_ok=True)
        print("SUBMITTED " + submit(state, path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("start", help="dry run by default; --submit actually queues one allocation")
    p.add_argument("--experiment", default="tamf_cet_001")
    p.add_argument("--target", type=int, default=30, help="total complete epochs, not additional epochs")
    p.add_argument("--max-allocations", type=int, help="total allocation safety cap; default >= target")
    p.add_argument("--cache", type=Path, default=ROOT / "common/save_cache_dict/main/cache/train.pkl")
    p.add_argument("--split-file", type=Path, default=ROOT / "asset/split/train.txt")
    p.add_argument("--epochs-per-allocation", type=int, help="optional cap, useful for a two-allocation canary; default is time-based")
    p.add_argument("--data-root", type=Path, default=ROOT.parent / "OakInk2-data")
    p.add_argument("--conda-base", type=Path, default=Path("/opt/sw/miniforge3"))
    p.add_argument("--conda-env", default="oakink2-tamf")
    p.add_argument("--partition", default="gpu")
    p.add_argument("--gres", default="shard:1")
    p.add_argument("--mem", default="16G")
    p.add_argument("--diagnostic-report", type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--submit", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    for mode in ("worker", "status", "stop"):
        p = modes.add_parser(mode)
        p.add_argument("--state", type=Path, default=ROOT / "common/cet_jobs/tamf_cet_001/state.json")
    args = parser.parse_args(argv)
    if args.mode == "start":
        if args.target < 1 or (args.max_allocations is not None and args.max_allocations < 1):
            parser.error("target and max-allocations must be positive")
        if args.epochs_per_allocation is not None and args.epochs_per_allocation < 1:
            parser.error("epochs-per-allocation must be positive")
        start(args)
    elif args.mode == "worker":
        worker(args.state)
    else:
        state = json.loads(args.state.read_text())
        if args.mode == "status":
            value = dict(target=state["target"], checkpoint=checkpoint_progress(state["experiment_dir"]),
                         allocation_history=state["allocations"], running=state["running"],
                         training_process_hours=state["training_process_seconds"] / 3600,
                         time_note="training process wall time, excludes queue time; not CUDA kernel time",
                         stop_requested=(args.state.parent / "STOP").exists(),
                         submission_pending=state["submission_pending"], slurm=active_jobs(state),
                         scalars=latest_scalars(state["experiment_dir"]))
            print(json.dumps(value, indent=2))
        else:
            # No controller lock: the worker holds it while training. Write STOP
            # first so a concurrently submitted successor will do no training.
            args.state.parent.joinpath("STOP").touch(exist_ok=True)
            for job_id, job_state, _ in active_jobs(state):
                if job_state == "RUNNING":
                    command = ["scancel", "--batch", "--signal=USR1", job_id]
                else:
                    command = ["scancel", job_id]
                subprocess.run(command, check=True)
                print(shlex.join(command))
            print("Stop requested. Running training finishes its current epoch if time permits.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
