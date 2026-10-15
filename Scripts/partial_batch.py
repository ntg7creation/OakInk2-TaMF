"""CPU-only preparation and coverage checks; never launch TaMF or metric jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEGMENT_PREFIX = "interaction_segment_"
NOTE = (
    "Partial-generation batch: official metrics compare only generated motions with "
    "available refined output, deduplicated by (sequence, primitive, hand side). "
    "Scores are partial-batch results, not full-test-set reproduction. "
    "Filename IDs are cache indices; this assumes the cache order and contents have "
    "not changed since generation. The recorded hash pins the cache from preparation onward."
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pickle(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def finite_array(value, shape, label, errors):
    arr = np.asarray(value)
    if arr.shape != shape or not np.isfinite(arr).all():
        errors.append(f"{label}: expected finite {shape}, got {arr.shape}")
    return arr


def output_relative(info):
    return str(Path(str(info[0]).replace("/", "++"), str(info[1]), str(info[2]), "save_dict.pkl"))


def coverage(manifest_path, cache_path, refined_root, root=ROOT):
    """Mirror official info-key deduplication and file-existence selection, without metrics."""
    manifest = json.loads(Path(manifest_path).read_text())
    cache_path, refined_root, root = Path(cache_path), Path(refined_root), Path(root)
    errors = []
    if sha256(cache_path) != manifest["subset_cache_sha256"]:
        errors.append("Evaluation cache differs from prepared generated-ID subset")
    source = root / manifest["source_cache"]
    if sha256(source) != manifest["source_cache_sha256"]:
        errors.append("Original cache has changed since preparation")
    generated_root = root / manifest["generated_dir"]
    files = sorted(generated_root.glob("*.npy"))
    if [p.name for p in files] != [s["filename"] for s in manifest["samples"]]:
        errors.append("Generated filenames differ from prepared batch")
    for s in manifest["samples"]:
        p = generated_root / s["filename"]
        if not p.is_file() or sha256(p) != s["sha256"]:
            errors.append(f"Generated file missing/changed: {s['filename']}")
    cache = read_pickle(cache_path)
    infos = cache[SEGMENT_PREFIX + "info_list"]
    if len(infos) != len(manifest["samples"]):
        errors.append("Subset length differs from matched generated count")
    seen, expected_paths = set(), set()
    found, missing, invalid = [], [], []
    cr_frames, siv_frames = 0, 0
    potential_cr_frames, potential_siv_frames = 0, 0
    for index, info in enumerate(infos):
        info = tuple(info)
        if info in seen:
            continue
        seen.add(info)
        rel = output_relative(info)
        expected_paths.add(rel)
        p = refined_root / rel
        n = int(cache[SEGMENT_PREFIX + "len_list"][index])
        potential_cr_frames += n
        potential_siv_frames += len(range(0, n, 20))
        if not p.is_file():
            missing.append(rel)
            continue
        found.append(rel)
        frame_ids = list(cache[SEGMENT_PREFIX + "frame_id_list"][index])
        problems = []
        try:
            r = read_pickle(p)
            if tuple(r["info"]) != info or r["hand_side"] != info[2]:
                problems.append("info/hand side mismatch")
            if int(r["len"]) != n or list(r["frame_id"]) != frame_ids:
                problems.append("valid length/frame IDs mismatch")
            if r["obj_list"] != sorted(cache[SEGMENT_PREFIX + "obj_traj_list"][index]):
                problems.append("object list mismatch")
            finite_array(r["verts"], (160, 778, 3), "verts", problems)
            finite_array(r["joints"], (160, 21, 3), "joints", problems)
            finite_array(r["refine_pose_repr"], (160, 99), "refine_pose_repr", problems)
        except Exception as exc:
            problems.append(str(exc))
        if problems:
            invalid.append({"path": rel, "errors": problems})
        else:
            cr_frames += n
            siv_frames += len(range(0, n, 20))
    errors.extend(f"Invalid refined input {x['path']}: {x['errors']}" for x in invalid)
    extras = [str(p.relative_to(refined_root)) for p in refined_root.rglob("save_dict.pkl")
              if str(p.relative_to(refined_root)) not in expected_paths]
    return {
        "note": NOTE,
        "generated": len(files), "successfully_matched": len(manifest["samples"]),
        "unmatched_generated": manifest["unmatched_generated"],
        "cache_segments": len(infos), "official_unique_motions": len(seen),
        "deduplicated_segments": len(infos) - len(seen),
        "refined": len(found), "valid_refined": len(found) - len(invalid),
        "eligible_paired_motions": len(found) - len(invalid),
        "missing_refined": len(missing), "missing_refined_paths": missing,
        "invalid_refined": invalid, "extra_refined_ignored": len(extras),
        "excluded_full_cache_segments": manifest["source_segments"] - len(infos),
        "expected_cr_frames": cr_frames, "expected_siv_frames": siv_frames,
        "potential_cr_frames": potential_cr_frames, "potential_siv_frames": potential_siv_frames,
        "errors": errors,
    }


def prepare(root=ROOT):
    import torch  # Only CPU tensor deserialization; no CUDA queries or model imports.
    import trimesh
    import yaml

    root = Path(root)
    dest = root / "common/partial_batch/test_one__0399"
    dest.mkdir(parents=True, exist_ok=True)
    opt = yaml.safe_load((root / "common/sample/main/opt.yml").read_text())
    run = opt["run"]
    # Validate the saved provenance before deriving any subset.
    expected_cache = "/mnt/d/repos/oakink/OakInk2-TaMF/common/save_cache_dict/main/cache/test.pkl"
    if run["data"]["cache_dict_filepath"] != expected_cache:
        raise ValueError("Saved cache path differs from the inspected generation setup")
    if run["runtime"] != {"num_worker": 1, "device_id": [0]}:
        raise ValueError("Saved worker setup differs from the inspected generation setup")
    if run["debug"]["sample_save_offset"] != "test_one/arch_mdm_l__0399":
        raise ValueError("Saved sample offset differs from this batch")
    generated_rel = "common/sample/main/sample/test_one/arch_mdm_l__0399"
    source_rel = "common/save_cache_dict/main/cache/test.pkl"
    source = root / source_rel
    cache = read_pickle(source)
    n = len(cache[SEGMENT_PREFIX + "info_list"])
    keys = [k for k in cache if k.startswith(SEGMENT_PREFIX)]
    errors = []
    for k in keys:
        if len(cache[k]) != n:
            errors.append(f"Cache list length mismatch: {k}")
    files = sorted((root / generated_rel).glob("*.npy"))
    samples, indices, unmatched = [], [], []
    objects = set()
    action_source = (root / "src/oakink2_tamf/dataset/action_adapter.py").read_text()
    for p in files:
        if not re.fullmatch(r"\d{6}\.npy", p.name):
            unmatched.append(p.name)
            continue
        i = int(p.stem)
        if i >= n:
            unmatched.append(p.name)
            continue
        indices.append(i)
        finite_array(np.load(p), (160, 99), p.name, errors)
        info = tuple(cache[SEGMENT_PREFIX + "info_list"][i])
        length = int(cache[SEGMENT_PREFIX + "len_list"][i])
        side = cache[SEGMENT_PREFIX + "hand_side_list"][i]
        frames = np.asarray(cache[SEGMENT_PREFIX + "frame_id_list"][i])
        if not 1 <= length <= 160 or side not in ("lh", "rh") or info[2] != side:
            errors.append(f"ID {i}: invalid length/side/info")
        if frames.shape != (length,) or not np.issubdtype(frames.dtype, np.integer) or not np.all(np.diff(frames) > 0):
            errors.append(f"ID {i}: invalid frame IDs")
        action = info[1].split(":")[0]
        if repr(action) not in action_source:
            errors.append(f"ID {i}: action absent from official FID adapter: {action}")
        for name, shape in (("pose", (160, 16, 3, 3)), ("tsl", (160, 3)), ("shape", (160, 10))):
            finite_array(cache[f"{SEGMENT_PREFIX}{name}_list"][i], shape, f"ID {i} {name}", errors)
        traj = cache[SEGMENT_PREFIX + "obj_traj_list"][i]
        if not traj:
            errors.append(f"ID {i}: no object trajectories")
        for oid, value in traj.items():
            objects.add(oid)
            arr = finite_array(value, (160, 4, 4), f"ID {i} object {oid}", errors)
            if arr.shape == (160, 4, 4) and not np.allclose(arr[:length, 3, :], [0, 0, 0, 1]):
                errors.append(f"ID {i} object {oid}: invalid homogeneous transforms")
        if not isinstance(cache[SEGMENT_PREFIX + "text_list"][i], str):
            errors.append(f"ID {i}: missing text")
        samples.append({"filename": p.name, "sample_id": i, "sha256": sha256(p),
                        "info": list(info), "len": length, "frame_id": frames.tolist()})
    if unmatched or not samples:
        errors.append(f"Unmatched/no generated inputs: {unmatched}")
    data = root / "data"
    obj_desc = json.loads((data / "object_raw/obj_desc.json").read_text())
    task_target = json.loads((data / "program/task_target.json").read_text())
    if any(s["info"][0] not in task_target for s in samples):
        errors.append("Generated sequence absent from program/task_target.json")
    asset_report = []
    for oid in sorted(objects):
        checks = {"object_id": oid}
        if oid not in obj_desc or "obj_name" not in obj_desc[oid]:
            errors.append(f"{oid}: missing object description/name")
        embedding = root / f"common/retrieve_obj_embedding/main/embedding/{oid}.pt"
        pc = root / f"common/retrieve_obj_pointcloud/main/pointcloud/{oid}.npz"
        try:
            value = torch.load(embedding, map_location="cpu", weights_only=True)
            finite_array(value.numpy(), (768,), f"{oid} embedding", errors)
            checks["embedding"] = str(embedding.relative_to(root))
        except Exception as exc:
            errors.append(f"{oid} embedding: {exc}")
        try:
            with np.load(pc) as z:
                points = z["point"]
                if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
                    errors.append(f"{oid}: invalid point cloud")
                checks["pointcloud_shape"] = list(points.shape)
        except Exception as exc:
            errors.append(f"{oid} point cloud: {exc}")
        mesh_dir = data / "object_raw/align_ds" / oid
        meshes = sorted(p for p in mesh_dir.glob("*") if p.suffix in (".obj", ".ply"))
        if len(meshes) != 1:
            errors.append(f"{oid}: official mesh loader requires exactly one .obj/.ply, got {len(meshes)}")
        else:
            try:
                mesh = trimesh.load(meshes[0], process=False, skip_materials=True, force="mesh")
                if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                    errors.append(f"{oid}: empty/nonfinite mesh")
                if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
                    errors.append(f"{oid}: invalid mesh face indices")
                checks["mesh"] = {"path": str(meshes[0].relative_to(root)),
                                  "vertices": len(mesh.vertices), "faces": len(mesh.faces),
                                  "watertight": bool(mesh.is_watertight)}
            except Exception as exc:
                errors.append(f"{oid} mesh: {exc}")
        asset_report.append(checks)
    mano = {}
    for side in ("LEFT", "RIGHT"):
        p = root / f"asset/mano_v1_2/models/MANO_{side}.pkl"
        mano[side] = p.is_file() and p.stat().st_size > 0
        if not mano[side]:
            errors.append(f"Missing MANO {side}")
    archive = root / "downloads/encoder__fid_1.tar.xz"
    fid = {"archive_exists": archive.is_file(), "archive_bytes": archive.stat().st_size if archive.is_file() else 0,
           "checkpoint_ready": False}
    if archive.is_file():
        try:
            with tarfile.open(archive, "r:xz") as tar:
                fid["archive_members"] = [m.name for m in tar]
            fid["archive_complete"] = True
        except Exception as exc:
            fid["archive_complete"] = False
            fid["archive_error"] = str(exc)
    checkpoint = root / "common/train/encoder__fid_1/save/model_0399.pt"
    fid["checkpoint_ready"] = checkpoint.is_file() and checkpoint.stat().st_size > 0
    report = {"note": NOTE, "source_segments": n, "generated": len(files),
              "successfully_matched": len(samples), "unique_objects": len(objects),
              "assets": asset_report, "mano": mano, "fid": fid, "errors": errors}
    (dest / "validation.json").write_text(json.dumps(report, indent=2))
    if errors:
        print(json.dumps(report, indent=2))
        raise ValueError("Validation failed; subset was not written")
    # The adaptor pairs sorted filenames with positional dataset entries, not parsed IDs.
    # Select exactly those original IDs in that same order, preserving duplicate infos.
    subset = dict(cache)
    for k in keys:
        subset[k] = [cache[k][i] for i in indices]
    subset["interaction_object_list"] = sorted(objects)
    subset_path = dest / "cache.pkl"
    with open(subset_path, "wb") as stream:
        pickle.dump(subset, stream, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {"note": NOTE, "source_cache": source_rel, "source_cache_sha256": sha256(source),
                "source_segments": n, "generated_dir": generated_rel,
                "subset_cache": str(subset_path.relative_to(root)), "subset_cache_sha256": sha256(subset_path),
                "saved_settings": opt, "samples": samples, "unmatched_generated": unmatched}
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    result = coverage(manifest_path, subset_path,
                      root / "common/sample_refine/partial_test_one/sample/test_one/arch_mdm_l__0399", root)
    (dest / "coverage.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({**{k: v for k, v in report.items() if k != "assets"},
                      "coverage": {k: v for k, v in result.items() if k != "missing_refined_paths"}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Validate inputs and write a separate cache/manifest; CPU only")
    parser.add_argument("--check", action="store_true", help="Check the prepared batch and current refined coverage; CPU only")
    args = parser.parse_args()
    if args.prepare and args.check:
        parser.error("Choose one mode")
    if args.prepare:
        prepare()
    elif args.check:
        dest = ROOT / "common/partial_batch/test_one__0399"
        result = coverage(dest / "manifest.json", dest / "cache.pkl",
                          ROOT / "common/sample_refine/partial_test_one/sample/test_one/arch_mdm_l__0399")
        (dest / "coverage.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "missing_refined_paths"}, indent=2))
        if result["errors"]:
            raise SystemExit(1)
    else:
        parser.error("Use --prepare or --check; this script never runs refinement/evaluation")


if __name__ == "__main__":
    main()
