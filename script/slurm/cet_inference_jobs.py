"""Detached CPU-selection -> single-GPU inference, resumable per committed sample.

start submits by default; --dry-run has no writes or scheduler calls. Raw cache
preprocessing remains a separate CPU job using the existing preprocessing tools.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import getpass
import json
import logging
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from oakink2_tamf.util.inference import (
    atomic_json, export_handoff, inspect_source, output_inventory, prepare_selection,
    progress_record, read_json, sha256_file, verify_selection,
    selection_identity,
)
from script.slurm.cet_jobs import workflow_lock

LOGGER = logging.getLogger(__name__)


@contextmanager
def control_lock(path):
    # Short blocking lock only for state/submission transactions. The separate
    # training-style nonblocking run.lock protects the entire GPU/CPU worker.
    import fcntl
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def duration(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if not match:
        raise ValueError("walltime must be [days-]HH:MM:SS")
    days, hours, minutes, seconds = (int(x or 0) for x in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError("invalid walltime minutes/seconds")
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 180:
        raise ValueError("walltime must exceed three minutes for signal/shutdown reserve")
    return total


def code_identity():
    paths = [Path(__file__), ROOT / "script/slurm/cet_inference.sbatch",
             ROOT / "script/slurm/cet_jobs.py", ROOT / "src/oakink2_tamf/util/inference.py",
             ROOT / "src/oakink2_tamf/launch/sample_selected.py",
             ROOT / "src/oakink2_tamf/dataset/cache_util.py",
             ROOT / "src/oakink2_tamf/dataset/interaction_segment.py",
             ROOT / "src/oakink2_tamf/dataset/collate.py",
             ROOT / "src/oakink2_tamf/model/interaction_segment_mdm.py",
             ROOT / "src/oakink2_tamf/model/diffusion_util.py"]
    paths += sorted((ROOT / "src/oakink2_tamf/model/diffusion").glob("*.py"))
    import hashlib
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}


def job_name(state):
    return "cet-infer-" + state["experiment"]


def active_jobs(state):
    result = subprocess.run(["squeue", "--noheader", "--user=" + getpass.getuser(),
        "--name=" + job_name(state), "--format=%A|%T|%R"], check=True, capture_output=True, text=True)
    return [line.strip().split("|", 2) for line in result.stdout.splitlines() if line.strip()]


def submission_command(state, path, phase, predecessor=None):
    command = ["sbatch", "--parsable", "--job-name=" + job_name(state),
        "--partition=" + (state["selection_partition"] if phase == "select" else state["partition"]), "--mem=" + state["mem"],
        "--cpus-per-task=" + str(state["cpus"]),
        "--time=" + (state["selection_walltime"] if phase == "select" else state["walltime"]),
        "--signal=B:USR1@120", "--no-requeue", "--kill-on-invalid-dep=yes",
        "--chdir=" + str(ROOT), "--output=" + str(Path(path).parent / "logs/allocation-%j.log"),
        "--open-mode=append", "--export=ALL"]
    if phase == "sample":
        command += ["--gres=" + state["gres"]]
    dependency = "singleton"
    if predecessor is not None:
        if not str(predecessor).isdigit():
            raise ValueError("predecessor must be a numeric Slurm job ID")
        dependency = f"afterok:{predecessor},singleton"
    command += ["--dependency=" + dependency, str(ROOT / "script/slurm/cet_inference.sbatch"),
        str(path), state["conda_base"], state["conda_env"], str(Path(__file__).resolve()), phase]
    return command


def submit(state, path, phase, predecessor=None):
    """Caller holds control.lock. Persist intent before sbatch, fail closed."""
    if state.get("submission_pending"):
        raise RuntimeError("ambiguous prior submission; inspect status/squeue, then resume --clear-submission-intent if safe")
    state["submission_pending"] = dict(phase=phase, predecessor=predecessor, time=time.time())
    atomic_json(path, state)
    command = submission_command(state, path, phase, predecessor)
    LOGGER.info("SBATCH %s", shlex.join(command))
    # In particular the CPU selection batch hides CUDA. Never export that empty
    # mask into its GPU successor: let Slurm establish the allocated device mask.
    environment = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError("unexpected sbatch response; submission intent preserved, inspect squeue before retry")
    state["jobs"].append(dict(job_id=job_id, phase=phase, predecessor=predecessor, submitted_at=time.time()))
    state["submission_pending"] = None
    state["phase"] = "pending"
    atomic_json(path, state)
    return job_id


def preflight(state):
    source = inspect_source(state["cache"], state["split"], state["split_file"])
    if source["sha256"] != state["cache_sha256"]:
        raise ValueError("source cache changed; use a new experiment")
    for key in ("checkpoint", "model_config"):
        if sha256_file(state[key]) != state[key + "_sha256"]:
            raise ValueError(f"{key} changed; resume cannot switch model/checkpoint")
    if code_identity() != state["code_identity"]:
        raise ValueError("inference implementation changed since start; inspect and use a new experiment")
    if not (Path(state["conda_base"]) / "etc/profile.d/conda.sh").is_file():
        raise FileNotFoundError("Conda activation script missing; set --conda-base")
    if not Path(state["data_root"]).is_dir() or not Path(state["obj_embedding_dir"]).is_dir():
        raise FileNotFoundError("dataset or object-embedding directory missing")


def selected_assets(state, selection):
    for oid in selection["object_ids"]:
        embedding = Path(state["obj_embedding_dir"]) / f"{oid}.pt"
        if not embedding.is_file():
            raise FileNotFoundError(f"selected object embedding missing: {embedding}; prepare/reuse embeddings before GPU inference")


def configuration(args, path):
    cache = (args.cache or ROOT / f"common/preprocessed_cache_v1/{args.split}.pkl").resolve()
    split_file = (args.split_file or ROOT / f"asset/split/{args.split}.txt").resolve()
    checkpoint, model_config = args.checkpoint.resolve(), args.model_config.resolve()
    source = inspect_source(cache, args.split, split_file)
    if args.selection_manifest and any(x is not None for x in (args.strides_per_motion, args.max_motions, args.seed)):
        raise ValueError("--selection-manifest owns strides/max-motions/seed; do not override selection options")
    strides = args.strides_per_motion or "2"
    if strides != "all":
        try:
            strides = int(strides)
        except ValueError:
            raise ValueError("strides-per-motion must be a positive integer or all") from None
        if strides < 1:
            raise ValueError("strides-per-motion must be positive")
    if args.max_motions is not None and args.max_motions < 1:
        raise ValueError("max-motions must be positive")
    duration(args.walltime)
    duration(args.selection_walltime)
    if args.cpus < 1 or args.max_allocations < 1:
        raise ValueError("cpus and max-allocations must be positive")
    output = (args.output_dir or path.parent / "output").resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists; existing outputs are protected: {output}")
    # An output ancestor could otherwise replace/contain source artifacts.
    for source_path in (cache, checkpoint, model_config, split_file):
        if output == source_path or output in source_path.parents:
            raise ValueError(f"output overlaps protected input: {source_path}")
    state = dict(version=1, experiment=args.experiment, output_dir=str(output), split=args.split,
        cache=str(cache), cache_sha256=source["sha256"], split_file=str(split_file),
        source_segments=source["segment_count"],
        checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        model_config=str(model_config), model_config_sha256=sha256_file(model_config),
        data_root=str(args.data_root.resolve()), obj_embedding_dir=str(args.obj_embedding_dir.resolve()),
        conda_base=str(args.conda_base.resolve()), conda_env=args.conda_env,
        selection_options=dict(strides_per_motion=strides, max_motions=args.max_motions, seed=args.seed or 0),
        reuse_selection=str(args.selection_manifest.resolve()) if args.selection_manifest else None,
        reuse_selection_sha256=sha256_file(args.selection_manifest) if args.selection_manifest else None,
        partition=args.partition, selection_partition=args.selection_partition or args.partition,
        gres=args.gres, mem=args.mem, cpus=args.cpus,
        walltime=args.walltime, selection_walltime=args.selection_walltime,
        max_allocations=args.max_allocations, allocations_started=0,
        jobs=[], allocations=[], running=None, phase="new", submission_pending=None,
        code_identity=code_identity(), created_at=time.time())
    if args.selection_manifest:
        plan = read_json(args.selection_manifest)
        if (plan["source_cache_sha256"] != source["sha256"] or plan["split"] != args.split
                or selection_identity(plan) != plan["selection_id"]):
            raise ValueError("reused selection does not match the source cache/split or was modified")
        state["selection_options"] = plan["options"]
    preflight(state)
    return state


def start(args, path):
    if path.exists():
        raise FileExistsError("experiment already exists; use resume, or choose a new --experiment")
    state = configuration(args, path)
    if args.dry_run:
        print("DRY RUN: no files or Slurm jobs; separate CPU selection followed by dependent GPU inference")
        print(json.dumps({k: state[k] for k in ("experiment", "checkpoint", "cache", "split", "output_dir", "selection_options")}, indent=2))
        print(shlex.join(submission_command(state, path, "select")))
        print(shlex.join(submission_command(state, path, "sample", "12345")))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with control_lock(path.parent / "control.lock"):
        if path.exists():
            raise FileExistsError("experiment was created concurrently; no duplicate submission")
        if any(p.name != "control.lock" for p in path.parent.iterdir()):
            raise FileExistsError("nonempty experiment directory lacks state.json; preserved, choose a new experiment")
        if active_jobs(state):
            raise RuntimeError("matching Slurm jobs already exist; no duplicate inference workflow submitted")
        Path(state["output_dir"]).mkdir(parents=True, exist_ok=False)
        (path.parent / "logs").mkdir(exist_ok=True)
        atomic_json(path.parent / "config.json", state)
        print("SUBMITTED CPU selection job " + submit(state, path, "select"))
        print("Logs: " + str(path.parent / "logs/allocation-<JOB_ID>.log"))


def resume(args, path):
    state = read_json(path)
    if args.dry_run:
        phase = "sample" if (Path(state["output_dir"]) / "selection/selection.json").is_file() else "select"
        print("DRY RUN: " + shlex.join(submission_command(state, path, phase)))
        return
    with control_lock(path.parent / "control.lock"), workflow_lock(path.parent / "run.lock"):
        state = read_json(path)
        if active_jobs(state):
            raise RuntimeError("active/pending inference jobs exist; no competing job submitted")
        if state.get("submission_pending"):
            if not args.clear_submission_intent:
                raise RuntimeError("ambiguous submission intent; inspect status, then use --clear-submission-intent after queue review")
            state["submission_pending"] = None
        preflight(state)
        if args.max_allocations is not None:
            if args.max_allocations <= state["allocations_started"]:
                raise ValueError("max-allocations must exceed allocations already started")
            state["max_allocations"] = args.max_allocations
        selection_path = Path(state["output_dir"]) / "selection/selection.json"
        phase = "select"
        if selection_path.is_file():
            selection = read_json(selection_path)
            verify_selection(state, selection)
            selected_assets(state, selection)
            receipts, errors = output_inventory(state, selection)
            if errors:
                raise ValueError(f"committed output corruption preserved; inspect before retry: {errors}")
            if len(receipts) == selection["total_samples"]:
                print("Inference already complete; nothing submitted")
                return
            phase = "sample"
        if state["allocations_started"] >= state["max_allocations"]:
            raise RuntimeError("allocation safety cap reached; extend with resume --max-allocations N")
        (path.parent / "STOP").unlink(missing_ok=True)
        print("SUBMITTED " + submit(state, path, phase))


class StopRequest:
    def __init__(self):
        self.requested = False
        self.hard = False

    def handle(self, signum, frame):
        self.requested = True
        self.hard = self.hard or signum in (signal.SIGTERM, signal.SIGINT)


def worker(path, phase):
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise RuntimeError("inference worker must run inside its recorded Slurm allocation")
    stop = StopRequest()
    previous = {sig: signal.signal(sig, stop.handle) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()
    owns_run = False
    try:
        with workflow_lock(path.parent / "run.lock"):
            owns_run = True
            with control_lock(path.parent / "control.lock"):
                state = read_json(path)
                if not any(j["job_id"] == job_id and j["phase"] == phase for j in state["jobs"]):
                    raise RuntimeError("allocation is not a recorded inference job for this phase")
                if state.get("submission_pending"):
                    raise RuntimeError("submission still ambiguous; no worker started")
                if (path.parent / "STOP").exists():
                    return 0
                if phase == "sample":
                    if state["allocations_started"] >= state["max_allocations"]:
                        raise RuntimeError("GPU allocation safety cap reached")
                    state["allocations_started"] += 1
                state["phase"] = "selecting" if phase == "select" else "sampling"
                state["running"] = dict(job_id=job_id, phase=phase, started_at=time.time())
                atomic_json(path, state)
            preflight(state)
            LOGGER.info("Inference %s | phase=%s | checkpoint=%s | sha256=%s | cache=%s",
                        state["experiment"], phase, state["checkpoint"], state["checkpoint_sha256"], state["cache"])
            if phase == "select":
                selection = prepare_selection(state)
                selected_assets(state, selection)
                result = dict(reason="selected", advanced=selection["total_samples"])
                LOGGER.info("Selection ready: %d unique motions; %d samples; options=%s; memory=%s",
                            selection["unique_motions"], selection["total_samples"], selection["options"], selection["memory"])
            else:
                from oakink2_tamf.launch.sample_selected import run_samples
                selection = read_json(Path(state["output_dir"]) / "selection/selection.json")
                verify_selection(state, selection)
                selected_assets(state, selection)
                result = run_samples(state, selection, path.parent, stop,
                                     max_seconds=duration(state["walltime"]) - 180)
            with control_lock(path.parent / "control.lock"):
                state = read_json(path)
                state["selection_id"] = selection["selection_id"]
                state["selection_summary"] = dict(unique_motions=selection["unique_motions"],
                    total_samples=selection["total_samples"], options=selection["options"])
                state["selection_manifest_sha256"] = sha256_file(Path(state["output_dir"]) / "selection/selection.json")
                state["running"] = None
                state["phase"] = result["reason"]
                state["allocations"].append(dict(job_id=job_id, phase=phase, reason=result["reason"],
                    seconds=time.monotonic() - started, advanced=result["advanced"],
                    failures=result.get("progress", {}).get("failed_samples", {})))
                atomic_json(path, state)
                stopped = (path.parent / "STOP").exists() or stop.hard
                if not stopped and phase == "select":
                    print("SUBMITTED GPU inference " + submit(state, path, "sample", job_id), flush=True)
                elif (not stopped and result["reason"] in ("session_limit", "interrupted")
                      and result["advanced"] > 0 and state["allocations_started"] < state["max_allocations"]):
                    print("SUBMITTED continuation " + submit(state, path, "sample", job_id), flush=True)
                else:
                    LOGGER.info("Allocation ended: %s; no successor submitted", result["reason"])
            return 2 if result["reason"] == "failed" else 0
    except Exception as exc:
        # Short lock remains usable by stop/status even while a worker is active.
        with control_lock(path.parent / "control.lock"):
            state = read_json(path)
            if owns_run and (state.get("running") or {}).get("job_id") == job_id:
                state.update(phase="failed", running=None, last_error=f"{type(exc).__name__}: {exc}")
                atomic_json(path, state)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def status(path):
    state = read_json(path)
    queue, accounting, scheduler_error = [], {}, None
    try:
        queue = active_jobs(state)
        if state["jobs"]:
            result = subprocess.run(["sacct", "--noheader", "--allocations", "--parsable2",
                "--jobs=" + ",".join(j["job_id"] for j in state["jobs"]),
                "--format=JobIDRaw,State,ExitCode,ElapsedRaw"], check=True, capture_output=True, text=True)
            accounting = {parts[0]: parts[1:] for line in result.stdout.splitlines()
                          if len(parts := line.strip().split("|")) >= 3}
    except (OSError, subprocess.CalledProcessError) as exc:
        scheduler_error = str(exc)
    selected = Path(state["output_dir"]) / "selection/selection.json"
    report = dict(experiment=state["experiment"], checkpoint=state["checkpoint"], checkpoint_sha256=state["checkpoint_sha256"],
                  phase=state["phase"], active_jobs=queue, accounting=accounting,
                  scheduler_error=scheduler_error, submission_pending=state.get("submission_pending"),
                  output_dir=state["output_dir"], logs=str(path.parent / "logs"),
                  last_error=state.get("last_error"), allocations_started=state["allocations_started"],
                  max_allocations=state["max_allocations"])
    report["elapsed_seconds"] = sum(a["seconds"] for a in state["allocations"])
    if state.get("running"):
        if any(row[0] == state["running"]["job_id"] for row in queue):
            report["elapsed_seconds"] += max(0, time.time() - state["running"]["started_at"])
        else:
            timing = accounting.get(state["running"]["job_id"], [])
            if len(timing) >= 3 and timing[2].isdigit():
                report["elapsed_seconds"] += int(timing[2])
    if selected.is_file():
        selection = read_json(selected)
        if (selection_identity(selection) != selection["selection_id"]
                or (state.get("selection_id") and selection["selection_id"] != state["selection_id"])):
            raise ValueError("selection identity changed")
        if state.get("selection_manifest_sha256") and sha256_file(selected) != state["selection_manifest_sha256"]:
            raise ValueError("immutable selection artifact changed")
        receipts, errors = output_inventory(state, selection)
        previous = read_json(path.parent / "progress.json") if (path.parent / "progress.json").exists() else {}
        failures = {int(k): v for k, v in previous.get("failed_samples", {}).items()}
        report.update(progress_record(selection, receipts, {**failures, **errors}))
        if queue and state.get("running") and state["running"]["phase"] == "sample":
            report["current_sample"] = previous.get("current_sample")
        if not report["remaining_samples"] and not errors:
            report["status"] = "completed"
        elif errors or state["phase"] == "failed":
            report["status"] = "failed"
    if "status" not in report:
        if queue:
            report["status"] = "stopping" if (path.parent / "STOP").exists() else (
                "running" if any(row[1] in ("RUNNING", "COMPLETING") for row in queue) else "pending")
        elif scheduler_error:
            report["status"] = "unknown; scheduler unavailable"
        elif (path.parent / "STOP").exists():
            report["status"] = "stopped"
        elif state["phase"] == "failed":
            report["status"] = "failed"
        elif state["jobs"] and accounting.get(state["jobs"][-1]["job_id"], [""])[0].split("+")[0] in (
                "FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"):
            report["status"] = "failed"
        else:
            report["status"] = "interrupted; resume required"
    print(json.dumps(report, indent=2))
    return report


def stop_jobs(args, path):
    state = read_json(path)
    if args.dry_run:
        print("DRY RUN: create STOP; cancel pending allocations; signal running batch USR1 (or cancel with --force)")
        return
    with control_lock(path.parent / "control.lock"):
        (path.parent / "STOP").touch(exist_ok=True)
    errors = []
    for job_id, phase, *_ in active_jobs(state):
        if not job_id.isdigit():
            raise ValueError("unexpected scheduler job ID")
        command = ["scancel", job_id] if args.force or phase not in ("RUNNING", "COMPLETING") else [
            "scancel", "--signal=USR1", "--batch", job_id]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            errors.append(f"job {job_id}: {exc}")
    if errors:
        raise RuntimeError("STOP marker saved; other jobs processed, but scancel failed: " + "; ".join(errors))
    print("Stop requested; completed samples are preserved. Use status, then resume when ready.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    for name in ("start", "status", "stop", "resume", "export"):
        sub = modes.add_parser(name)
        sub.add_argument("--experiment", default="tamf_cet_epoch11")
        sub.add_argument("--jobs-root", type=Path, default=ROOT / "common/inference_jobs")
        if name in ("start", "stop", "resume"):
            sub.add_argument("--dry-run", action="store_true")
        if name == "start":
            sub.add_argument("--checkpoint", type=Path, default=ROOT / "common/train/tamf_cet_001_retry1/save/model_0011.pt")
            sub.add_argument("--model-config", type=Path, default=ROOT / "config/shenkar_generator_full.yml")
            sub.add_argument("--split", choices=("train", "test"), default="test")
            sub.add_argument("--cache", type=Path)
            sub.add_argument("--split-file", type=Path)
            sub.add_argument("--strides-per-motion", help="positive integer or all; default 2")
            sub.add_argument("--max-motions", type=int)
            sub.add_argument("--seed", type=int)
            sub.add_argument("--selection-manifest", type=Path)
            sub.add_argument("--output-dir", type=Path)
            sub.add_argument("--data-root", type=Path, default=ROOT / "data")
            sub.add_argument("--obj-embedding-dir", type=Path, default=ROOT / "common/retrieve_obj_embedding/main/embedding")
            sub.add_argument("--conda-base", type=Path, default=Path("/opt/sw/miniforge3"))
            sub.add_argument("--conda-env", default="oakink2-tamf")
            sub.add_argument("--partition", default="gpu")
            sub.add_argument("--selection-partition", help="CPU selection partition; defaults to --partition, without GRES")
            sub.add_argument("--gres", default="shard:1")
            sub.add_argument("--mem", default="16G")
            sub.add_argument("--cpus", type=int, default=2)
            sub.add_argument("--walltime", default="04:00:00")
            sub.add_argument("--selection-walltime", default="00:30:00")
            sub.add_argument("--max-allocations", type=int, default=100)
        elif name == "resume":
            sub.add_argument("--max-allocations", type=int)
            sub.add_argument("--clear-submission-intent", action="store_true")
        elif name == "stop":
            sub.add_argument("--force", action="store_true")
        elif name == "export":
            sub.add_argument("--output-dir", type=Path, help="new handoff snapshot directory")
    w = modes.add_parser("worker", help=argparse.SUPPRESS)
    w.add_argument("--state", type=Path, required=True)
    w.add_argument("--phase", choices=("select", "sample"), required=True)
    args = p.parse_args(argv)
    try:
        if args.mode == "worker":
            path = args.state.resolve()
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                handlers=[logging.StreamHandler(), logging.FileHandler(path.parent / "logs/inference.log", encoding="utf-8")])
            return worker(path, args.phase)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.experiment):
            raise ValueError("invalid inference experiment ID")
        path = args.jobs_root.resolve() / args.experiment / "state.json"
        if args.mode == "start":
            start(args, path)
        elif args.mode == "resume":
            resume(args, path)
        elif args.mode == "status":
            status(path)
        elif args.mode == "stop":
            stop_jobs(args, path)
        else:
            with workflow_lock(path.parent / "run.lock"):
                state = read_json(path)
                selection = read_json(Path(state["output_dir"]) / "selection/selection.json")
                destination = (args.output_dir or path.parent / "handoff").resolve()
                print(export_handoff(state, selection, destination))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f"Inference workflow error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
