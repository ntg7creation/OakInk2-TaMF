"""Evaluate saved TaMF motions; never run generation, refinement or training.

Official CR/SIV/PSKL-J/FID use existing R-stage save_dict.pkl records. CET can
also evaluate G-stage numeric .npy files alone. See EVALUATION.md for pairing,
input stages, dependencies, and the historical-cache provenance limitation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
import evaluate_cet as cet

OFFICIAL = ("cr", "siv", "psklj", "fid")
FIELDS = ("info", "len", "pose", "tsl", "shape", "hand_side", "text", "obj_traj", "frame_id")
require = cet.require


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-dir", required=True, type=Path, help="flat directory of original cache-index .npy files")
    p.add_argument("--test-cache", required=True, type=Path, help="exact cache used for generation, or a manifest-mapped subset")
    p.add_argument("--refined-dir", type=Path, help="R-stage save_dict.pkl tree; required for all official metrics")
    p.add_argument("--output-dir", required=True, type=Path, help="new directory; existing directories are never overwritten")
    p.add_argument("--pairing-manifest", type=Path, help="existing partial_batch manifest or this evaluator's inputs/manifest.json")
    p.add_argument("--trust-cache-order", action="store_true",
                   help="explicitly assert historical numeric IDs match this cache when no pairing manifest exists")
    p.add_argument("--allow-subset", action="store_true", help="allow missing expected generated IDs; report them as excluded")
    p.add_argument("--sample-id", type=int, action="append", help="evaluate only this original ID; repeat for a bounded subset")
    p.add_argument("--stage", choices=("G", "R"), default="R", help="G supports CET only; official metrics use R")
    p.add_argument("--metrics", nargs="+", choices=(*OFFICIAL, "cet", "all"), default=["all"])
    p.add_argument("--experiment", default="saved_motions", help="result label only; no generator checkpoint is loaded")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--test-split", type=Path, default=ROOT / "asset/split/test.txt")
    p.add_argument("--mano-root", type=Path, default=ROOT / "asset/mano_v1_2")
    p.add_argument("--obj-embedding-dir", type=Path, default=ROOT / "common/retrieve_obj_embedding/main/embedding")
    p.add_argument("--obj-pointcloud-dir", type=Path, default=ROOT / "common/retrieve_obj_pointcloud/main/pointcloud")
    p.add_argument("--fid-config", type=Path, default=ROOT / "config/arch_encoder.yml")
    p.add_argument("--fid-encoder", type=Path, default=ROOT / "common/train/encoder__fid_1/save/model_0399.pt")
    p.add_argument("--device", default="cuda:0", help="official metrics' device; CPU is supported for bounded checks")
    p.add_argument("--seed", type=int, default=0, help="seed official FID, preserving its historical encoder mode")
    p.add_argument("--rho", type=float, default=.10, help="existing CET normalized-time DTW band")
    p.add_argument("--dry-run", action="store_true", help="validate pairing/assets and print commands; write nothing")
    return p


def cache_value(cache, field, index):
    return cache[cet.PREFIX + field + "_list"][index]


def finite_array(value, shape, label):
    a = np.asarray(value)
    require(a.shape == shape and np.issubdtype(a.dtype, np.number)
            and not np.iscomplexobj(a) and np.isfinite(a).all(),
            f"{label}: expected finite numeric array {shape}, got {a.shape}")
    return a


def inventory(args):
    require(not args.output_dir.exists(), f"output directory already exists: {args.output_dir}; choose a new directory")
    protected = [args.motion_dir, args.test_cache, args.test_split]
    protected += [p for p in (args.refined_dir, args.pairing_manifest) if p is not None]
    for path in protected:
        require(args.output_dir != path and args.output_dir not in path.parents
                and (not path.is_dir() or path not in args.output_dir.parents),
                f"output directory overlaps an input: {path}")
    require(args.motion_dir.is_dir(), f"motion directory missing: {args.motion_dir}")
    require(args.pairing_manifest is not None or args.trust_cache_order,
            "numeric .npy files contain no sample metadata. Supply --pairing-manifest, or explicitly "
            "use --trust-cache-order with the exact generation cache; identity cannot be proved from poses alone")
    cache_hash = cet.sha256(args.test_cache)
    cache = cet.read_pickle(args.test_cache)
    count = len(cache[cet.PREFIX + "info_list"])
    require(count > 0, "empty test cache")
    for field in FIELDS:
        require(len(cache[cet.PREFIX + field + "_list"]) == count, f"cache parallel-list length mismatch: {field}")
    manifest = None
    expected = {}
    if args.pairing_manifest:
        manifest = json.loads(args.pairing_manifest.read_text(encoding="utf-8"))
        source_matches = cache_hash == manifest["source_cache_sha256"]
        samples = manifest.get("available_samples", manifest["samples"]) if source_matches else manifest["samples"]
        require(len({s["sample_id"] for s in samples}) == len(samples), "duplicate sample IDs in pairing manifest")
        require(not manifest.get("unmatched_generated"), "pairing manifest contains unmatched generated files")
        if source_matches:
            positions = {s["sample_id"]: s["sample_id"] for s in samples}
        elif cache_hash == manifest.get("subset_cache_sha256"):
            require(len(samples) == count, "manifest/subset-cache length mismatch")
            positions = {s["sample_id"]: i for i, s in enumerate(samples)}
        else:
            raise cet.InvalidInput("test cache checksum differs from pairing manifest; refusing to pair motions")
        expected = {s["sample_id"]: s for s in samples}
        # New portable manifests retain the actual original-ID -> cache-position map.
        if cache_hash == manifest["source_cache_sha256"] and "source_cache_indices" in manifest:
            positions = {int(k): v for k, v in manifest["source_cache_indices"].items()}
    else:
        positions = {i: i for i in range(count)}
    files = {}
    for path in args.motion_dir.rglob("*.npy"):
        require(path.parent == args.motion_dir and re.fullmatch(r"[0-9]{6}\.npy", path.name),
                f"unexpected/nested generated filename: {path}; expected flat six-digit cache indices")
        index = int(path.stem)
        require(index not in files, f"duplicate generated sample ID: {index}")
        require(index in positions or (args.sample_id is not None and index not in args.sample_id),
                f"generated sample ID {index} is absent from the test cache/manifest")
        files[index] = path
    require(bool(files), "no generated .npy files found")
    missing = sorted(set(positions) - files.keys())
    require(not missing or args.allow_subset,
            f"missing {len(missing)} expected generated samples (first IDs: {missing[:10]}). "
            "Use --allow-subset only for an intentional subset")
    selected = set(args.sample_id) if args.sample_id else set(files)
    require(not args.sample_id or len(selected) == len(args.sample_id), "duplicate --sample-id selections")
    require(selected <= files.keys(), f"requested sample IDs missing: {sorted(selected - files.keys())}")
    require(selected <= positions.keys(), "selected sample ID is absent from the pairing manifest/cache")
    test_sequences = set(args.test_split.read_text(encoding="utf-8").splitlines())
    entries, duplicates, available, seen = [], [], [], set()
    for index in sorted(selected, key=lambda x: positions[x]):
        i = positions[index]
        require(isinstance(i, int) and 0 <= i < count, f"sample {index}: invalid cache index {i}")
        info = tuple(cache_value(cache, "info", i))
        require(len(info) == 3 and all(isinstance(x, str) for x in info), f"sample {index}: invalid info key")
        require(info[0] in test_sequences, f"sample {index}: sequence is outside the supplied test split: {info[0]}")
        n = cache_value(cache, "len", i)
        require(isinstance(n, (int, np.integer)) and 16 <= n <= 160, f"sample {index}: invalid length {n}")
        n = int(n)
        side = cache_value(cache, "hand_side", i)
        require(side == info[2] and side in ("lh", "rh"), f"sample {index}: hand side mismatch")
        frames = np.asarray(cache_value(cache, "frame_id", i))
        require(frames.shape == (n,) and np.issubdtype(frames.dtype, np.integer), f"sample {index}: invalid frame IDs")
        cet.normalized_time(frames)
        gt_pose = finite_array(cache_value(cache, "pose", i), (160, 16, 3, 3), "GT rotations")
        cet.check_rotations(gt_pose[:n])
        finite_array(cache_value(cache, "tsl", i), (160, 3), "GT translation")
        finite_array(cache_value(cache, "shape", i), (160, 10), "GT shape")
        traj = cache_value(cache, "obj_traj", i)
        require(bool(traj), f"sample {index}: no object trajectories")
        for oid, transform in traj.items():
            require(isinstance(oid, str) and Path(oid).name == oid, f"invalid object ID: {oid}")
            transforms = finite_array(transform, (160, 4, 4), f"object {oid}")[:n]
            require(np.allclose(transforms[:, 3], [0, 0, 0, 1]), f"object {oid}: invalid homogeneous transforms")
            cet.check_rotations(transforms[:, :3, :3])
        pose = finite_array(np.load(files[index], allow_pickle=False), (160, 99), files[index].name)
        cet.decode_pose(pose, n)
        digest = cet.sha256(files[index])
        if manifest is not None:
            saved = expected[index]
            require(saved["filename"] == files[index].name and saved["sha256"] == digest,
                    f"sample {index}: generated file differs from pairing manifest")
            require(tuple(saved["info"]) == info and saved["len"] == n and saved["frame_id"] == frames.tolist(),
                    f"sample {index}: manifest/cache identity, length or frame IDs mismatch")
        entry = dict(sample_id=index, source_cache_index=i, filename=files[index].name,
                     sha256=digest, info=list(info), len=n, frame_id=frames.tolist())
        available.append(dict(entry))
        if info in seen:
            duplicates.append(dict(entry, reason="official first-info deduplication; temporal-offset segment excluded"))
            continue
        seen.add(info)
        if args.stage == "R":
            refined = cet.refined_path(args.refined_dir, info).resolve()
            require(args.refined_dir in refined.parents, f"refined path escapes supplied directory: {refined}")
            require(refined.is_file(), f"sample {index}: required R-stage record missing: {refined}; evaluation never runs refinement")
            record = cet.read_pickle(refined)
            require(tuple(record["info"]) == info and record["hand_side"] == side
                    and record["process_key"] == info[0], f"sample {index}: refined identity/hand side mismatch")
            require(record["len"] == n and np.array_equal(record["frame_id"], frames),
                    f"sample {index}: refined length/frame IDs mismatch; likely wrong temporal offset or cache")
            require(record["obj_list"] == sorted(traj), f"sample {index}: refined object IDs mismatch")
            for field, shape in (("refine_pose_repr", (160, 99)), ("verts", (160, 778, 3)), ("joints", (160, 21, 3))):
                finite_array(record[field], shape, f"sample {index} refined {field}")
            cet.decode_pose(record["refine_pose_repr"], n)
            entry.update(refined_path=str(refined), refined_sha256=cet.sha256(refined))
        entries.append(entry)
    require(bool(entries), "empty evaluation selection")
    objects = sorted({oid for e in entries for oid in cache_value(cache, "obj_traj", e["source_cache_index"])})
    if "fid" in args.metrics:
        require(len(entries) >= 2, "FID requires at least two unique paired motions to estimate covariance; select more samples or omit fid")
    required_files = [args.mano_root / "models" / f"MANO_{side}.pkl" for side in ("LEFT", "RIGHT")]
    if any(m in OFFICIAL for m in args.metrics):
        require(args.data_root.is_dir(), f"OakInk2 dataset root missing: {args.data_root}")
        required_files += [args.obj_embedding_dir / f"{oid}.pt" for oid in objects]
        required_files += [args.obj_pointcloud_dir / f"{oid}.npz" for oid in objects]
        for oid in objects:
            meshes = [p for p in (args.data_root / "object_raw/align_ds" / oid).glob("*") if p.suffix in (".obj", ".ply")]
            require(len(meshes) == 1, f"object {oid}: expected one canonical .obj/.ply mesh under data-root/object_raw/align_ds")
    if "fid" in args.metrics:
        required_files += [args.fid_config, args.fid_encoder]
    for path in required_files:
        require(path.is_file(), f"required evaluation asset missing: {path}")
    report = dict(experiment=args.experiment, stage=args.stage, source_cache=str(args.test_cache),
                  source_cache_sha256=cache_hash, source_segments=manifest["source_segments"] if manifest else count,
                  source_cache_indices={str(e["sample_id"]): e["source_cache_index"] for e in available},
                  generated_dir=str(args.motion_dir), samples=entries, available_samples=available, unmatched_generated=[],
                  generated_files_found=len(files), selected_generated_segments=len(selected),
                  unique_motions=len(entries), duplicate_segments=duplicates,
                  missing_generated_ids=missing, not_selected_ids=sorted(files.keys() - selected),
                  unknown_unselected_ids=sorted(files.keys() - positions.keys()),
                  scope="complete supplied cache; full-test completeness depends on cache provenance"
                  if len(selected) == count and not missing and (not manifest or manifest["source_segments"] == count)
                  else "explicit subset; not a full-test benchmark",
                  pairing_limitation="numeric poses do not embed identity; cache order must match generation. "
                  "A supplied manifest pins its recorded snapshot, not necessarily generation time",
                  pairing_basis="supplied manifest" if manifest else "explicit user assertion of generation-cache order")
    if args.stage == "R":
        expected_refined = {e["refined_path"] for e in entries}
        actual_refined = {str(p.resolve()) for p in args.refined_dir.rglob("save_dict.pkl")}
        report.update(refined_files_found=len(actual_refined),
                      unselected_refined_paths=sorted(actual_refined - expected_refined))
    return cache, report


def official_command(metric, args, cache_path):
    cmd = [sys.executable, "-m", f"script.compute_score.compute_score_{metric}",
           "--data.data_prefix", str(args.data_root), "--data.process_range", f"?(file:{args.test_split})",
           "--data.obj_embedding_prefix", str(args.obj_embedding_dir),
           "--data.obj_pointcloud_prefix", str(args.obj_pointcloud_dir),
           "--mano.mano_path", str(args.mano_root), "--debug.sample_refine_filepath", str(args.refined_dir),
           "--debug.evaluation_output_dir", str(args.output_dir / "metrics" / metric),
           "--debug.evaluation_device", args.device]
    if metric == "fid":
        cmd += ["--cfg", str(args.fid_config), "--debug.cache_dict_filepath", str(cache_path),
                "--debug.encoder_checkpoint_filepath", str(args.fid_encoder),
                "--debug.evaluation_seed", str(args.seed)]
    else:
        cmd += ["--data.cache_dict_filepath", str(cache_path)]
    return cmd


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    cache, pairing = inventory(args)
    out = args.output_dir
    cache_path = out / "inputs/cache.pkl"
    commands = {m: official_command(m, args, cache_path) for m in args.metrics if m in OFFICIAL}
    if args.dry_run:
        brief = {k: pairing[k] for k in ("experiment", "stage", "scope", "source_cache",
                 "source_cache_sha256", "generated_files_found", "selected_generated_segments",
                 "unique_motions", "pairing_basis", "pairing_limitation")}
        brief.update(duplicate_segments=len(pairing["duplicate_segments"]),
                     missing_generated_count=len(pairing["missing_generated_ids"]),
                     first_missing_ids=pairing["missing_generated_ids"][:10],
                     unselected_generated_count=len(pairing["not_selected_ids"]))
        print(json.dumps(dict(pairing=brief, official_commands=commands,
                              cet="existing Scripts/evaluate_cet.py; no metrics run" if "cet" in args.metrics else None), indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    indices = [e["source_cache_index"] for e in pairing["samples"]]
    subset = {cet.PREFIX + field + "_list": [cache_value(cache, field, i) for i in indices] for field in FIELDS}
    subset["interaction_object_list"] = sorted({oid for value in subset[cet.PREFIX + "obj_traj_list"] for oid in value})
    with cache_path.open("xb") as stream:
        pickle.dump(subset, stream, protocol=pickle.HIGHEST_PROTOCOL)
    pairing.update(subset_cache=str(cache_path), subset_cache_sha256=cet.sha256(cache_path))
    manifest_path = out / "inputs/manifest.json"
    cet.json_write(manifest_path, pairing)
    # Release a potentially full test cache before starting the metric workers.
    del cache, subset
    summary = dict(experiment=args.experiment, stage=args.stage, scope=pairing["scope"],
                   created_utc=datetime.now(timezone.utc).isoformat(), pairing="inputs/manifest.json",
                   metrics={}, implementation_sha256={str(p.relative_to(ROOT)): cet.sha256(p) for p in
                   [Path(__file__), ROOT / "Scripts/evaluate_cet.py", ROOT / "src/oakink2_tamf/util/evaluation_output.py",
                    *[ROOT / f"script/compute_score/compute_score_{m}.py" for m in OFFICIAL]]})
    summary["reference_assets"] = {str(p): cet.sha256(p) for p in
        [args.mano_root / "models/MANO_LEFT.pkl", args.mano_root / "models/MANO_RIGHT.pkl",
         *([args.fid_config, args.fid_encoder] if "fid" in args.metrics else [])]}
    rows = [dict(sample_id=e["sample_id"], sequence=e["info"][0], primitive=e["info"][1],
                 hand_side=e["info"][2], valid_length=e["len"]) for e in pairing["samples"]]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))
    (out / "logs").mkdir()
    for metric in args.metrics:
        try:
            # Reject drift between inventory and use, including externally changed R records.
            require(cet.sha256(args.test_cache) == pairing["source_cache_sha256"], "test cache changed during evaluation")
            require(cet.sha256(cache_path) == pairing["subset_cache_sha256"], "prepared evaluation cache changed during evaluation")
            for entry in pairing["samples"]:
                require(cet.sha256(args.motion_dir / entry["filename"]) == entry["sha256"], "generated input changed during evaluation")
                if args.stage == "R":
                    require(cet.sha256(entry["refined_path"]) == entry["refined_sha256"], "refined input changed during evaluation")
            if metric == "cet":
                cet_args = argparse.Namespace(output_dir=out / "metrics/cet", manifest=manifest_path,
                    cache=cache_path, generated_root=args.motion_dir,
                    refined_root=args.refined_dir or out / "no_refinement", mano_root=args.mano_root,
                    stage=args.stage, rho=args.rho, motion_index=None, inventory_only=False,
                    stage_only_pairing=True, scope=pairing["scope"])
                require(cet.run(cet_args) == 0, "CET failed; inspect metrics/cet/motions.json")
                result = json.loads((out / "metrics/cet/summary.json").read_text(encoding="utf-8"))
                motions = json.loads((out / "metrics/cet/motions.json").read_text(encoding="utf-8"))
                for row, motion in zip(rows, motions):
                    require(row["sample_id"] == motion["original_motion_index"], "CET export order/identity mismatch")
                    row.update(cet_status=motion["status"], cet_reason=motion["reason"])
                    if motion["status"] == "success":
                        row.update(cet_degrees=motion["cet_degrees"],
                                   cet_per_joint_degrees=motion["per_joint_degrees"],
                                   cet_per_finger_degrees=motion["per_finger_degrees"],
                                   cet_arrays="metrics/cet/" + motion["arrays_path"])
                require(len(motions) == len(rows) and all(m["status"] == "success" for m in motions),
                        "CET rejected one or more selected motions; failure rows were preserved")
                scores = result["stages"][args.stage]
            else:
                work = out / "work" / metric
                work.mkdir(parents=True)
                command = commands[metric]
                print(f"Running official {metric}: {len(rows)} paired R-stage motions", flush=True)
                with (out / f"logs/{metric}_stdout.txt").open("w", encoding="utf-8") as stdout, \
                     (out / f"logs/{metric}_stderr.txt").open("w", encoding="utf-8") as stderr:
                    completed = subprocess.run(command, cwd=work, env=env, stdout=stdout, stderr=stderr)
                require(completed.returncode == 0, f"official {metric} failed (exit {completed.returncode}); see logs/{metric}_stderr.txt")
                result = json.loads((out / f"metrics/{metric}/results.json").read_text(encoding="utf-8"))
                require(len(result["motions"]) == len(rows), f"official {metric} silently skipped a selected motion")
                for i, (entry, motion, row) in enumerate(zip(pairing["samples"], result["motions"], rows)):
                    require(motion["subset_cache_index"] == i and motion["info"] == entry["info"],
                            f"official {metric} export order/identity mismatch")
                    row[f"{metric}_array_row"] = i
                    if metric in ("cr", "siv"):
                        row.update({f"{metric}_{key}": motion[key] for key in ("gt", "prediction", "array_start", "array_end")})
                scores = result["summary"]
            summary["metrics"][metric] = dict(status="success", scores=scores,
                artifacts=f"metrics/{metric}", command=commands.get(metric))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            summary["metrics"][metric] = dict(status="failed", error=str(exc), artifacts=f"metrics/{metric}")
            print(f"{metric}: {exc}", file=sys.stderr, flush=True)
        cet.json_write(out / "summary.json", summary)
        cet.json_write(out / "motions.json", rows)
    flat = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows]
    write_csv(out / "motions.csv", flat)
    write_csv(out / "summary.csv", [dict(metric=m, stage=args.stage, status=r["status"],
        scores_json=json.dumps(r.get("scores", {})), error=r.get("error", "")) for m, r in summary["metrics"].items()])
    print(f"Results: {out}", flush=True)
    return 0 if all(r["status"] == "success" for r in summary["metrics"].values()) else 2


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.metrics = [*OFFICIAL, "cet"] if "all" in args.metrics else list(dict.fromkeys(args.metrics))
    if args.stage == "G" and any(m in OFFICIAL for m in args.metrics):
        p.error("official CR/SIV/PSKL-J/FID require R-stage save_dict.pkl outputs. Use --refined-dir and --stage R; G-only supports --metrics cet")
    if args.stage == "R" and args.refined_dir is None:
        p.error("--refined-dir is required for R-stage evaluation; for raw motions alone use --stage G --metrics cet")
    if not np.isfinite(args.rho) or not 0 <= args.rho <= 1:
        p.error("rho must be finite and in [0,1]")
    try:
        return run(args)
    except (ValueError, OSError, KeyError, TypeError, pickle.UnpicklingError, EOFError) as exc:
        p.exit(2, f"Evaluation input/configuration error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
