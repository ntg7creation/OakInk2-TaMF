"""Standalone partial-batch CET. Never imports generation or refinement code.

NumPy handles metrics and pose decoding; MANO/manotorch are used only to build
the fixed side-specific anatomical template axes (CPU, once per side).
See CET.md for the angle convention, schemas, and reproducibility contract.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = "cet-1.0.0"
ROOT = Path(__file__).resolve().parents[1]
PREFIX = "interaction_segment_"
FINGERS = ("index", "middle", "little", "ring", "thumb")
JOINT_NAMES = tuple(f"{f}_{j}" for f in FINGERS for j in
                    (("CMC", "MCP", "IP") if f == "thumb" else ("MCP", "PIP", "DIP")))
PARENTS = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
SNAP_JOINTS = (5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3)
CONVENTION = "manotorch_AxisLayerFK_template_XYZ_signed_bend_Z_v1"


class InvalidInput(ValueError):
    pass


class InfeasiblePath(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidInput(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_pickle(path):
    # Only use trusted project artifacts: pickle can execute code.
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def valid_prefix(value, length, tail_shape, label):
    a = np.asarray(value, dtype=np.float64)
    require(a.ndim == len(tail_shape) + 1 and a.shape[1:] == tail_shape,
            f"{label}: expected [T,{','.join(map(str, tail_shape))}], got {a.shape}")
    require(isinstance(length, (int, np.integer)) and 1 <= length <= len(a),
            f"{label}: invalid valid length {length} for {len(a)} frames")
    a = a[:length].copy()
    require(np.isfinite(a).all(), f"{label}: nonfinite valid frames")
    return a


def check_rotations(rotations):
    r = np.asarray(rotations, dtype=np.float64)
    require(r.shape[-2:] == (3, 3) and np.isfinite(r).all(), "invalid rotation matrices")
    require(np.allclose(r @ r.swapaxes(-1, -2), np.eye(3), atol=2e-4, rtol=0)
            and np.allclose(np.linalg.det(r), 1, atol=2e-4, rtol=0),
            "rotation matrices are not proper orthonormal rotations")
    return r


def decode_pose(value, length):
    """TaMF rows-of-rotation 6D convention, with explicit degeneracy detection."""
    pose = valid_prefix(value, length, (99,), "pose_repr")
    d6 = pose[:, 3:].reshape(length, 16, 6)
    a, b = d6[..., :3], d6[..., 3:]
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    require(np.all(n > 1e-8), "6D rotation has zero first axis")
    x = a / n
    y = b - np.sum(x * b, axis=-1, keepdims=True) * x
    n = np.linalg.norm(y, axis=-1, keepdims=True)
    require(np.all(n > 1e-8), "6D rotation has collinear/zero second axis")
    y /= n
    return check_rotations(np.stack((x, y, np.cross(x, y)), axis=-2))


def curl_from_local_rotations(rotations, basis):
    """Exactly AxisLayerFK's template-relative rotation, without global FK.

    With A_j = TMPL_R_p_a[j], its formula simplifies to A_j.T @ L_j @ A_j.
    The signed Z component of XYZ is atan2(-Q[0,1], Q[0,0]). The middle
    angle's gimbal singularity is rejected. Root is omitted entirely.
    """
    r = check_rotations(rotations)
    require(r.ndim == 4 and r.shape[1:] == (16, 3, 3) and len(r) > 0,
            "expected nonempty [T,16,3,3] local rotations")
    a = check_rotations(basis)
    require(a.shape == (16, 3, 3), "expected [16,3,3] anatomy basis")
    q = a[1:].swapaxes(-1, -2) @ r[:, 1:] @ a[1:]
    cos_spread = np.hypot(q[..., 0, 0], q[..., 0, 1])
    require(np.all(cos_spread > 1e-6), "anatomical XYZ spread gimbal singularity")
    flexion = np.arctan2(-q[..., 0, 1], q[..., 0, 0])
    # Reject branch crossings instead of silently treating +/-pi as activity.
    require(not np.any(np.abs(np.diff(flexion, axis=0)) > np.pi),
            "signed bend crosses the +/-pi branch; convention is ambiguous")
    return flexion


def load_anatomical_bases(mano_root):
    """Use vendored manotorch and MANO; no model jobs and no per-motion LBS."""
    sys.path.insert(0, str(ROOT / "thirdparty/manotorch"))
    try:
        from manotorch.axislayer import AxisLayerFK
    except ModuleNotFoundError as exc:
        raise InvalidInput(f"MANO anatomy dependency missing ({exc.name}); activate the TaMF environment "
                           "and run python -m pip install -r Scripts/requirements-cet.txt") from exc
    bases, provenance = {}, {}
    for side, full in (("rh", "right"), ("lh", "left")):
        layer = AxisLayerFK(side=full, mano_assets_root=str(mano_root)).cpu().eval()
        require(tuple(layer.transf_parent_mapping) == PARENTS, "MANO parent mapping changed")
        basis = layer.TMPL_R_p_a.detach().cpu().numpy()[0].astype(np.float64)
        check_rotations(basis)
        bases[side] = basis
        provenance[side] = {
            "basis": basis.tolist(),
            "model_sha256": sha256(Path(mano_root) / "models" / f"MANO_{full.upper()}.pkl"),
        }
    provenance["axislayer_sha256"] = sha256(ROOT / "thirdparty/manotorch/manotorch/axislayer.py")
    provenance["manolayer_sha256"] = sha256(ROOT / "thirdparty/manotorch/manotorch/manolayer.py")
    return bases, provenance


def normalized_time(frame_ids):
    """Actual saved frame coordinates, normalized by first/last; singleton = 0."""
    f = np.asarray(frame_ids, dtype=np.float64)
    require(f.ndim == 1 and len(f) > 0 and np.isfinite(f).all(), "invalid time coordinates")
    require(np.all(np.diff(f) > 0), "time coordinates must strictly increase")
    return (f - f[0]) / (f[-1] - f[0]) if len(f) > 1 else np.zeros(1)


def timing_provenance():
    """Read the repository's verified mocap clock without importing its dataset."""
    path = ROOT / "thirdparty/OakInk2/src/oakink2_toolkit/meta.py"
    source = path.read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FPS_MOCAP" for t in node.targets):
            fps = float(ast.literal_eval(node.value))
            require(np.isfinite(fps) and fps > 0, "invalid repository FPS_MOCAP")
            return {"frame_id_clock_hz": fps, "clock_source": str(path), "clock_source_sha256": sha256(path),
                    "frame_id_semantics": "original mocap frame IDs, sliced alongside poses by InteractionSegmentData.load_dataset",
                    "dataset_source_sha256": sha256(ROOT / "src/oakink2_tamf/dataset/interaction_segment.py")}
    raise InvalidInput("repository FPS_MOCAP was not found; duration clock cannot be established")


def constrained_dtw(a, b, u=None, v=None, rho=0.10, weights=None):
    """Minimize cumulative weighted MAE, then divide by backtracked path length."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    require(a.ndim == b.ndim == 2 and a.shape[1:] == b.shape[1:] == (15,)
            and len(a) > 0 and len(b) > 0, "expected nonempty [T,15] curl arrays")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite curl angles")
    require(np.isfinite(rho) and 0 <= rho <= 1, "rho must be in [0,1]")
    w = np.ones(15) if weights is None else np.asarray(weights, dtype=np.float64)
    require(w.shape == (15,) and np.isfinite(w).all() and np.all(w >= 0) and w.sum() > 0,
            "weights must be 15 finite nonnegative values with positive sum")
    u = normalized_time(np.arange(len(a))) if u is None else np.asarray(u, dtype=np.float64)
    v = normalized_time(np.arange(len(b))) if v is None else np.asarray(v, dtype=np.float64)
    for t, n in ((u, len(a)), (v, len(b))):
        require(t.shape == (n,) and np.isfinite(t).all() and np.all(np.diff(t) > 0)
                and t[0] == 0 and t[-1] == (1 if n > 1 else 0), "invalid normalized times")
    errors = np.abs(a[:, None, :] - b[None, :, :])
    costs = errors @ w / w.sum()
    band = np.abs(u[:, None] - v[None, :]) <= rho + 1e-12
    d = np.full((len(a), len(b)), np.inf)
    prev = np.full(d.shape, -1, dtype=np.int8)
    # Deterministic ties: diagonal, advancing prediction, advancing reference.
    steps = ((1, 1), (1, 0), (0, 1))
    for i in range(len(a)):
        for k in np.flatnonzero(band[i]):
            if i == k == 0:
                d[i, k] = costs[i, k]
                continue
            candidates = [d[i - di, k - dk] if i >= di and k >= dk else np.inf
                          for di, dk in steps]
            p = int(np.argmin(candidates))
            if np.isfinite(candidates[p]):
                d[i, k] = costs[i, k] + candidates[p]
                prev[i, k] = p
    if not np.isfinite(d[-1, -1]):
        raise InfeasiblePath(f"no endpoint-to-endpoint path: lengths={len(a)},{len(b)}, rho={rho}")
    i, k = len(a) - 1, len(b) - 1
    path = [(i, k)]
    while i or k:
        di, dk = steps[prev[i, k]]
        i, k = i - di, k - dk
        path.append((i, k))
    path = np.asarray(path[::-1], dtype=np.int64)
    aligned = errors[path[:, 0], path[:, 1]]
    per_joint = np.rad2deg(aligned.mean(axis=0))
    cet = float(np.rad2deg(d[-1, -1] / len(path)))
    require(np.isclose(cet, per_joint @ w / w.sum(), atol=1e-10), "CET aggregation failed")
    # Diagnostic only: linear interpolation on union of the two original
    # normalized time grids; no resampling is performed for the CET metric.
    grid = np.union1d(u, v)
    ag = np.column_stack([np.interp(grid, u, a[:, j]) for j in range(15)])
    bg = np.column_stack([np.interp(grid, v, b[:, j]) for j in range(15)])
    unwarped_joint = np.rad2deg(np.abs(ag - bg).mean(axis=0))
    return {
        "cet_degrees": cet, "cumulative_cost_radians": float(d[-1, -1]),
        "path": path, "path_length": len(path), "aligned_abs_error_degrees": np.rad2deg(aligned),
        "per_joint_degrees": per_joint,
        "per_finger_degrees": per_joint.reshape(5, 3).mean(axis=1),
        "W": float(np.abs(u[path[:, 0]] - v[path[:, 1]]).mean()),
        "unwarped_mae_degrees": float(unwarped_joint @ w / w.sum()),
        "unwarped_per_joint_degrees": unwarped_joint, "diagnostic_grid": grid,
        "diagnostic_pred_degrees": np.rad2deg(ag), "diagnostic_gt_degrees": np.rad2deg(bg),
        "weights": w,
    }


def curl_descriptors(angles_degrees, seconds=None):
    """Descriptors per joint, then arithmetic mean over each anatomical group."""
    a = np.asarray(angles_degrees, dtype=np.float64)
    require(a.ndim == 2 and a.shape[1] == 15 and len(a) > 0 and np.isfinite(a).all(),
            "invalid descriptor angles")
    stats = {
        "mean_signed_flexion_degrees": a.mean(axis=0),
        "mean_absolute_flexion_degrees": np.abs(a).mean(axis=0),
        "max_absolute_flexion_degrees": np.abs(a).max(axis=0),
        "flexion_range_degrees": np.ptp(a, axis=0),
        "total_angular_variation_degrees": np.abs(np.diff(a, axis=0)).sum(axis=0),
    }
    if seconds is not None:
        require(np.isfinite(seconds) and seconds > 0, "duration must be positive seconds")
        stats["variation_degrees_per_second"] = stats["total_angular_variation_degrees"] / seconds
    return {name: {"per_joint": values.tolist(), "per_finger": values.reshape(5, 3).mean(axis=1).tolist(),
                   "whole_hand": float(values.mean())} for name, values in stats.items()}


def refined_path(root, info):
    return Path(root) / str(info[0]).replace("/", "++") / str(info[1]) / str(info[2]) / "save_dict.pkl"


def relative_input(root, value):
    return (Path(root) / value).resolve()


def inspect_batch(manifest_path, cache_path, generated_root, refined_root, root=ROOT, require_refined=True):
    """Read-only inventory and pairing validation, no angle extraction or metric."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    require(sha256(cache_path) == manifest["subset_cache_sha256"], "prepared cache checksum mismatch")
    source = relative_input(root, manifest["source_cache"])
    require(sha256(source) == manifest["source_cache_sha256"], "original cache checksum mismatch")
    generated_root, refined_root = Path(generated_root), Path(refined_root)
    cache = read_pickle(cache_path)
    infos = cache[PREFIX + "info_list"]
    samples = manifest["samples"]
    require(len(infos) == len(samples) > 0, "manifest/cache length mismatch or empty batch")
    require(len({s["sample_id"] for s in samples}) == len(samples), "duplicate original sample IDs")
    require(not manifest.get("unmatched_generated"), "preparation has unmatched generated inputs")
    for field in ("len", "pose", "hand_side", "frame_id", "text"):
        require(len(cache[PREFIX + field + "_list"]) == len(infos), f"cache {field} list length mismatch")
    actual_generated = {p.name for p in generated_root.glob("*.npy")}
    expected_generated = {s["filename"] for s in samples}
    entries, seen, duplicates = [], {}, []
    for subset_index, s in enumerate(samples):
        info = tuple(infos[subset_index])
        require(len(info) == 3 and tuple(s["info"]) == info, "manifest/cache motion key mismatch")
        original = s["sample_id"]
        require(isinstance(original, int) and 0 <= original < manifest["source_segments"]
                and s["filename"] == f"{original:06d}.npy", "filename/original index mismatch")
        n = cache[PREFIX + "len_list"][subset_index]
        frames = np.asarray(cache[PREFIX + "frame_id_list"][subset_index])
        require(n == s["len"] and frames.tolist() == s["frame_id"], "manifest/cache length or frames mismatch")
        require(isinstance(n, (int, np.integer)) and n > 0 and frames.shape == (n,)
                and np.issubdtype(frames.dtype, np.integer), "invalid cache frame IDs or length")
        normalized_time(frames)
        require(cache[PREFIX + "hand_side_list"][subset_index] == info[2] and info[2] in ("rh", "lh"),
                "cache hand side mismatch")
        if info in seen:
            duplicates.append({"original_motion_index": original, "subset_cache_index": subset_index,
                               "representative_motion_index": seen[info], "info": list(info),
                               "reason": "official first-occurrence motion-key deduplication"})
            continue
        seen[info] = original
        gp, rp = generated_root / s["filename"], refined_path(refined_root, info)
        entry = {
            "motion_key": json.dumps(info, ensure_ascii=False, separators=(",", ":")),
            "original_motion_index": original, "subset_cache_index": subset_index,
            "representative_G_index": original, "sequence": info[0], "primitive": info[1],
            "hand_side": info[2], "description": cache[PREFIX + "text_list"][subset_index],
            "info": list(info), "GT_source": str(Path(cache_path).resolve()),
            "G_source": str(gp.resolve()), "R_source": str(rp.resolve()),
            "valid_length_GT": int(n), "valid_length_prediction": int(n),
            "found_G": gp.is_file(), "found_R": rp.is_file(), "status": "paired", "reason": "",
        }
        try:
            if not gp.is_file() or (require_refined and not rp.is_file()):
                entry["status"] = "missing"
                required = (("G", gp), ("R", rp)) if require_refined else (("G", gp),)
                raise InvalidInput("missing " + "/".join(x for x, p in required if not p.is_file()))
            require(sha256(gp) == s["sha256"], "representative G checksum mismatch")
            gt = valid_prefix(cache[PREFIX + "pose_list"][subset_index], int(n), (16, 3, 3), "GT pose")
            check_rotations(gt)
            decode_pose(np.load(gp, allow_pickle=False), int(n))
            entry["G_sha256"] = s["sha256"]
            if require_refined:
                r = read_pickle(rp)
                require(tuple(r["info"]) == info and r["hand_side"] == info[2], "R motion key/side mismatch")
                require(r["len"] == n and np.array_equal(r["frame_id"], frames), "R valid length/frame IDs mismatch")
                decode_pose(r["refine_pose_repr"], int(n))
                entry["R_sha256"] = sha256(rp)
        except (InvalidInput, ValueError, OSError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as exc:
            if entry["status"] != "missing":
                entry["status"] = "invalid"
            entry["reason"] = str(exc)
        entries.append(entry)
    expected_refined = {refined_path(refined_root, e["info"]).resolve() for e in entries}
    actual_refined = {p.resolve() for p in refined_root.rglob("save_dict.pkl")}
    # Pin every generated file, including discarded duplicates, to preparation.
    changed_duplicates = []
    representatives = {e["original_motion_index"] for e in entries}
    for s in samples:
        if s["sample_id"] not in representatives:
            p = generated_root / s["filename"]
            if not p.is_file() or sha256(p) != s["sha256"]:
                changed_duplicates.append(s["filename"])
    inventory = {
        "generated_files_found": len(actual_generated), "cache_segments": len(infos),
        "original_cache_segments": manifest["source_segments"],
        "unique_motion_keys": len(entries), "duplicate_segments_skipped": len(duplicates),
        "refined_files_found": len(actual_refined), "found_representative_G": sum(e["found_G"] for e in entries),
        "found_expected_R": sum(e["found_R"] for e in entries),
        "paired_motions": sum(e["status"] == "paired" for e in entries),
        "missing_motions": sum(e["status"] == "missing" for e in entries),
        "invalid_motions": sum(e["status"] == "invalid" for e in entries),
        "extra_generated_files_skipped": sorted(actual_generated - expected_generated),
        "missing_generated_files": sorted(expected_generated - actual_generated),
        "changed_or_missing_duplicate_G": changed_duplicates,
        "extra_refined_files_skipped": sorted(str(p) for p in actual_refined - expected_refined),
        "status_reasons": dict(Counter(e["reason"] for e in entries if e["reason"])),
        "duplicates": duplicates,
        "observed_frame_id_stride_counts": dict(Counter(str(int(d)) for e in entries
            for d in np.diff(cache[PREFIX + "frame_id_list"][e["subset_cache_index"]]))),
    }
    require(not changed_duplicates, "discarded duplicate G files changed since preparation")
    return cache, entries, inventory


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_motion_arrays(path, pred, gt, frames, u, result, stage, index, basis, rho, frame_id_fps):
    """No object arrays; np.load(..., allow_pickle=False) suffices."""
    np.savez_compressed(
        path, prediction_curl_degrees=np.rad2deg(pred), GT_curl_degrees=np.rad2deg(gt),
        joint_names=np.asarray(JOINT_NAMES), finger_names=np.asarray(FINGERS),
        prediction_frame_ids=frames, GT_frame_ids=frames,
        prediction_time_seconds=(frames - frames[0]) / frame_id_fps,
        GT_time_seconds=(frames - frames[0]) / frame_id_fps, frame_id_clock_hz=frame_id_fps,
        prediction_normalized_time=u, GT_normalized_time=u,
        dtw_frame_pairs=result["path"], aligned_abs_error_degrees=result["aligned_abs_error_degrees"],
        diagnostic_common_normalized_time=result["diagnostic_grid"],
        diagnostic_prediction_degrees=result["diagnostic_pred_degrees"],
        diagnostic_GT_degrees=result["diagnostic_gt_degrees"],
        joint_weights=result["weights"], anatomy_basis=basis,
        cumulative_cost_radians=result["cumulative_cost_radians"],
        original_motion_index=index, stage=stage, angle_convention=CONVENTION, rho=rho,
    )


def flatten_row(row, stage):
    out = {k: row.get(k, "") for k in (
        "motion_key", "original_motion_index", "subset_cache_index", "representative_G_index",
        "sequence", "primitive", "hand_side", "description", "GT_source", "valid_length_GT",
        "valid_length_prediction", "status", "reason", "cet_degrees", "W", "path_length",
        "unwarped_mae_degrees", "cumulative_cost_radians", "arrays_path", "duration_seconds", "rho")}
    out.update(stage=stage, prediction_source=row[stage + "_source"], angle_convention=CONVENTION)
    for name in ("per_joint_degrees", "per_finger_degrees", "unwarped_per_joint_degrees"):
        for j, label in enumerate(FINGERS if name == "per_finger_degrees" else JOINT_NAMES):
            out[f"{name}_{label}"] = row[name][j] if name in row else ""
    for source in ("GT", "prediction"):
        for desc in ("mean_signed_flexion_degrees", "mean_absolute_flexion_degrees",
                     "max_absolute_flexion_degrees", "flexion_range_degrees", "total_angular_variation_degrees",
                     "variation_degrees_per_second"):
            values = row.get(source + "_curl_descriptors", {}).get(desc, {})
            out[f"{source}_{desc}_whole_hand"] = values.get("whole_hand", "")
            for level, labels in (("per_joint", JOINT_NAMES), ("per_finger", FINGERS)):
                for j, label in enumerate(labels):
                    out[f"{source}_{desc}_{label}"] = values[level][j] if level in values else ""
    return out


def run(args):
    out = args.output_dir.resolve()
    require(not out.exists() or not any(out.iterdir()), "output directory must be new or empty")
    stages = ("R", "G") if args.stage == "both" else (args.stage,)
    # Historical standalone CLI still requires both stages unless explicitly
    # requested; the general saved-motion entry point uses stage-only pairing.
    require_both = not getattr(args, "stage_only_pairing", False)
    cache, entries, inventory = inspect_batch(args.manifest, args.cache, args.generated_root,
                                             args.refined_root, require_refined=require_both or "R" in stages)
    timing = timing_provenance()
    frame_id_fps = timing["frame_id_clock_hz"]
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "implementation_version": VERSION, "script_sha256": sha256(__file__),
        "command": sys.argv, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": getattr(args, "scope", "available partial batch only; not full-test-set evaluation"),
        "aggregation": "each successful unique motion has equal weight; population standard deviation",
        "angle_source": "parent-relative MANO rotations in fixed manotorch anatomical template axes",
        "angle_convention": CONVENTION, "joint_names": JOINT_NAMES, "finger_names": FINGERS,
        "MANO_joint_indices": list(range(1, 16)), "MANO_parent_indices": PARENTS,
        "SNAP_21_joint_indices": SNAP_JOINTS, "weights": [1.0] * 15, "rho": args.rho,
        "metric_internal_units": "radians", "saved_angle_units": "degrees",
        "sampling_policy": "original saved valid frame IDs; normalized by first/last frame ID; singleton time=0",
        "resampling_policy": "none for CET; diagnostic linear interpolation on union of normalized original grids",
        "filtering_policy": "none", "timing_provenance": timing,
        "duration_normalized_variation": "TV / ((last frame ID - first frame ID) / FPS_MOCAP); omitted for singletons",
        "descriptor_aggregation": "calculate each joint statistic over valid frames, then mean 3 joints per finger or 15 per hand",
        "pairing_policy": "manifest original IDs + subset positions + info keys; first cache occurrence; "
                          + ("require valid G and R for either stage" if require_both else "validate requested prediction stage"),
        "pairing_limitation": "preparation hashes pin current inputs; generation-time identity requires a contemporaneous manifest",
        "input_paths": {k: str(getattr(args, k).resolve()) for k in
                        ("manifest", "cache", "generated_root", "refined_root", "mano_root")},
        "input_sha256": {"manifest": sha256(args.manifest), "cache": sha256(args.cache)},
        "inventory": inventory, "pairing_manifest": entries,
    }
    prepared_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    provenance["input_paths"]["original_cache"] = str(relative_input(ROOT, prepared_manifest["source_cache"]))
    provenance["input_sha256"]["original_cache"] = prepared_manifest["source_cache_sha256"]
    if args.inventory_only:
        provenance["mode"] = "inventory_only; no angles or metrics computed"
        json_write(out / "summary.json", provenance)
        print(json.dumps({k: v for k, v in inventory.items() if k != "duplicates"}, indent=2))
        return 0
    bases, axis_provenance = load_anatomical_bases(args.mano_root)
    provenance["anatomical_template_provenance"] = axis_provenance
    selected = [e for e in entries if args.motion_index is None or e["original_motion_index"] == args.motion_index]
    require(bool(selected), "requested original representative motion index was not found")
    provenance["selection"] = {"original_motion_index": args.motion_index,
                               "unique_motions_selected": len(selected),
                               "unselected_motions_skipped": len(entries) - len(selected)}
    rows = []
    (out / "arrays").mkdir()
    for e in selected:
        base = dict(e)
        result_by_stage = {}
        if e["status"] == "paired":
            try:
                i, n, side = e["subset_cache_index"], e["valid_length_GT"], e["hand_side"]
                frames = np.asarray(cache[PREFIX + "frame_id_list"][i], dtype=np.int64)
                u = normalized_time(frames)
                gt = curl_from_local_rotations(valid_prefix(cache[PREFIX + "pose_list"][i], n, (16, 3, 3), "GT"), bases[side])
                predictions = {}
                if require_both or "R" in stages:
                    r = read_pickle(e["R_source"])
                    predictions["R"] = curl_from_local_rotations(decode_pose(r["refine_pose_repr"], n), bases[side])
                if require_both or "G" in stages:
                    predictions["G"] = curl_from_local_rotations(
                        decode_pose(np.load(e["G_source"], allow_pickle=False), n), bases[side])
                # Check all angle definitions for either stage so G and R cannot
                # quietly use different successful motion sets.
                for stage in stages:
                    result_by_stage[stage] = constrained_dtw(predictions[stage], gt, u, u, args.rho)
                base["status"] = "success"
            except InfeasiblePath as exc:
                base.update(status="infeasible_path", reason=str(exc))
            except (InvalidInput, ValueError, OSError, KeyError, TypeError) as exc:
                base.update(status="invalid", reason=str(exc))
        for stage in stages:
            row = dict(base, stage=stage, prediction_source=e[stage + "_source"],
                       angle_convention=CONVENTION, rho=args.rho)
            if base["status"] == "success":
                result = result_by_stage[stage]
                pred = predictions[stage]
                row.update({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()
                            if k not in ("path", "aligned_abs_error_degrees", "diagnostic_grid",
                                         "diagnostic_pred_degrees", "diagnostic_gt_degrees", "weights")})
                duration = float((frames[-1] - frames[0]) / frame_id_fps)
                row["duration_seconds"] = duration
                row["GT_curl_descriptors"] = curl_descriptors(np.rad2deg(gt), duration if duration > 0 else None)
                row["prediction_curl_descriptors"] = curl_descriptors(np.rad2deg(pred), duration if duration > 0 else None)
                rel = f"arrays/{e['original_motion_index']:06d}_{stage}.npz"
                row["arrays_path"] = rel
                save_motion_arrays(out / rel, pred, gt, frames, u, result, stage, e["original_motion_index"], bases[side], args.rho, frame_id_fps)
            rows.append(row)
    json_write(out / "motions.json", rows)
    flat = [flatten_row(row, row["stage"]) for row in rows]
    with (out / "motions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    summaries = {}
    for stage in stages:
        subset = [r for r in rows if r["stage"] == stage]
        successful = [r for r in subset if r["status"] == "success"]
        values = np.asarray([r["cet_degrees"] for r in successful])
        summaries[stage] = {
            "successful_motion_count": len(successful),
            "mean_CET_degrees": float(values.mean()) if len(values) else None,
            "median_CET_degrees": float(np.median(values)) if len(values) else None,
            "std_CET_degrees": float(values.std()) if len(values) else None,
            "status_counts": dict(Counter(r["status"] for r in subset)),
            "skipped_motion_count": len(entries) - len(successful),
            "skipped_reasons": {**dict(Counter(r["reason"] for r in subset if r["status"] != "success")),
                                "not_selected_by_motion_index": len(entries) - len(selected)},
            "failure_reasons": dict(Counter(r["reason"] for r in subset if r["status"] != "success")),
        }
    provenance.update(mode="single_motion_check" if args.motion_index is not None else "partial_batch_evaluation",
                      stages=summaries, pairing_manifest=entries)
    json_write(out / "summary.json", provenance)
    print(json.dumps({"output_dir": str(out), "stages": summaries}, indent=2))
    return 0 if all(s["successful_motion_count"] for s in summaries.values()) else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    batch = ROOT / "common/partial_batch/test_one__0399"
    p.add_argument("--manifest", type=Path, default=batch / "manifest.json")
    p.add_argument("--cache", type=Path, default=batch / "cache.pkl")
    p.add_argument("--generated-root", type=Path, default=ROOT / "common/sample/main/sample/test_one/arch_mdm_l__0399")
    p.add_argument("--refined-root", type=Path, default=ROOT / "common/sample_refine/partial_test_one/sample/test_one/arch_mdm_l__0399")
    p.add_argument("--mano-root", type=Path, default=ROOT / "asset/mano_v1_2")
    p.add_argument("--stage", choices=("R", "G", "both"), default="R")
    p.add_argument("--stage-only-pairing", action="store_true",
                   help="validate only the requested prediction stage; permits G-only evaluation without R")
    p.add_argument("--rho", type=float, default=0.10)
    p.add_argument("--motion-index", type=int, help="evaluate only this ORIGINAL first-occurrence cache ID")
    p.add_argument("--inventory-only", action="store_true", help="validate pairing only; no MANO or CET")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    p.add_argument("--output-dir", type=Path, default=batch / "cet" / stamp)
    args = p.parse_args()
    if not np.isfinite(args.rho) or not 0 <= args.rho <= 1:
        p.error("rho must be finite and in [0,1]")
    try:
        return run(args)
    except (InvalidInput, OSError, KeyError, ValueError) as exc:
        p.exit(2, f"CET input/configuration error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
