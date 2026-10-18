"""Inference-only selection and immutable sample publication; no model imports."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import uuid

import numpy as np

from ..dataset.cache_util import (
    FIELDS, OBJECTS, cache_spec, manifest_path, read_cache, sha256_file,
    write_cache, memory_status,
)


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


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selection_identity(selection):
    # Paths/checkpoints are intentionally excluded so identical selections can
    # be reused after relocation and across original/CET/different-loss models.
    return identity({key: selection[key] for key in ("version", "split", "source_cache_sha256",
                    "source_segments", "options", "samples", "schedule")})


def inspect_source(cache_path, split, split_file):
    """Stream checksums, not a multi-GB pickle, on the submission/login node."""
    path = Path(cache_path)
    if not path.is_file() or not manifest_path(path).is_file():
        raise FileNotFoundError(f"Prepared {split} cache AND manifest required: {path}. "
                                "Prepare it separately with script.save_cache_dict; no train-cache fallback.")
    metadata = read_json(manifest_path(path))
    if metadata.get("split_name") != split:
        raise ValueError(f"Requested {split}, but cache declares {metadata.get('split_name')!r}: {path}")
    allowed = set(Path(split_file).read_text(encoding="utf-8").splitlines())
    keys = metadata["sequence_keys"]
    if not keys or not set(keys) <= allowed:
        raise ValueError("cache sequences are outside the requested split file")
    spec = cache_spec(keys, split)
    for key in ("version", "sequence_keys", "split_sha256", "slicing", "augmented", "preprocessing"):
        if metadata.get(key) != spec[key]:
            raise ValueError(f"incompatible inference cache manifest: {key}")
    # One legacy pickle is deserialized only by the separate CPU selection job.
    # A conservative cap prevents an unbounded pickle load under a 16G request.
    if path.stat().st_size > 4 * 1024**3:
        raise ValueError("cache exceeds the 4 GiB single-pickle safety cap; prepare a smaller sequence cache")
    if path.stat().st_size != metadata["size_bytes"] or sha256_file(path) != metadata["sha256"]:
        raise ValueError(f"cache size/checksum mismatch: {path}")
    return metadata


def choose_samples(cache, strides=2, max_motions=None, seed=0):
    """Group actual occurrences; choose endpoint-spread offsets, not new draws."""
    if strides != "all" and (not isinstance(strides, int) or strides < 1):
        raise ValueError("strides-per-motion must be a positive integer or all")
    if max_motions is not None and max_motions < 1:
        raise ValueError("max-motions must be positive")
    groups = defaultdict(list)
    for index, (info, frames) in enumerate(zip(cache[FIELDS[0]], cache[FIELDS[-1]])):
        info = tuple(info)
        if len(info) != 3 or info[2] not in ("rh", "lh"):
            raise ValueError(f"invalid motion key at cache index {index}")
        frames = np.asarray(frames, dtype=np.int64)
        if len(frames) < 2 or not np.all(np.diff(frames) == frames[1] - frames[0]) or frames[1] <= frames[0]:
            raise ValueError(f"nonuniform/reversed temporal variant at cache index {index}")
        groups[info].append((int(frames[0]), index, int(frames[1] - frames[0])))
    motions = list(groups)
    if strides != "all" or max_motions is not None:
        # Round-robin action categories, with seeded, stable within-action ranks.
        actions = defaultdict(list)
        for key in motions:
            actions[key[1].split(":", 1)[0]].append(key)
        for bucket in actions.values():
            bucket.sort(key=lambda key: identity([seed, list(key)]))
        motions = []
        limit = min(max_motions, len(groups)) if max_motions is not None else len(groups)
        depth = 0
        while len(motions) < limit:
            for action in sorted(actions):
                if depth < len(actions[action]):
                    motions.append(actions[action][depth])
                    if len(motions) == limit:
                        break
            depth += 1
    entries, schedule = [], []
    by_motion = []
    for key in motions:
        variants = sorted(groups[key])
        last_frames = [int(cache[FIELDS[-1]][v[1]][-1]) for v in variants]
        if (len({v[0] for v in variants}) != len(variants) or len({v[2] for v in variants}) != 1
                or variants[-1][0] - variants[0][0] >= variants[0][2]
                or max(last_frames) - min(last_frames) >= variants[0][2]):
            raise ValueError(f"ambiguous temporal variants for motion {key}")
        take = len(variants) if strides == "all" else min(strides, len(variants))
        ranks = [len(variants) // 2] if take == 1 else [
            round(j * (len(variants) - 1) / (take - 1)) for j in range(take)]
        # all preserves every original cache index, not the seeded motion order.
        selected = []
        for slot, rank in enumerate(ranks):
            first, index, gap = variants[rank]
            frames = list(map(int, cache[FIELDS[-1]][index]))
            entry = dict(sample_id=index, original_cache_index=index, filename=f"{index:06d}.npy",
                         info=list(key), len=int(cache[FIELDS[1]][index]), frame_id=frames,
                         action=key[1].split(":", 1)[0], variant_slot=slot,
                         available_variants=len(variants), temporal_variant_rank=rank,
                         temporal_offset=first - variants[0][0], frame_stride=gap,
                         first_frame_id=first, seed=(int(seed) + 1000003 * index) % (2**32))
            entries.append(entry)
            selected.append(index)
        by_motion.append(selected)
    # Cover distinct motions before revisiting second/third temporal variants.
    for slot in range(max(map(len, by_motion), default=0)):
        schedule.extend(ids[slot] for ids in by_motion if slot < len(ids))
    entries.sort(key=lambda e: e["sample_id"])
    if strides == "all" and max_motions is None:
        schedule = [e["sample_id"] for e in entries]
    for position, entry in enumerate(entries):
        entry["subset_cache_index"] = position
    return entries, schedule


def subset_cache(cache, entries):
    indices = [e["original_cache_index"] for e in entries]
    result = {field: [cache[field][i] for i in indices] for field in FIELDS}
    result[OBJECTS] = sorted({oid for traj in result[FIELDS[-2]] for oid in traj})
    return result


def prepare_selection(state):
    """CPU allocation only. Publish GT and the immutable plan as one directory."""
    output = Path(state["output_dir"])
    final = output / "selection"
    if final.exists():
        selection = read_json(final / "selection.json")
        verify_selection(state, selection)
        return selection
    source = inspect_source(state["cache"], state["split"], state["split_file"])
    if source["sha256"] != state["cache_sha256"]:
        raise ValueError("source cache changed since submission")
    cache, _ = read_cache(state["cache"], source["sequence_keys"], require_manifest=True, split_name=state["split"])
    if state.get("reuse_selection"):
        previous = read_json(state["reuse_selection"])
        if sha256_file(state["reuse_selection"]) != state["reuse_selection_sha256"]:
            raise ValueError("reused selection manifest changed since submission")
        if previous["source_cache_sha256"] != source["sha256"] or previous["split"] != state["split"]:
            raise ValueError("comparison selection uses a different cache or split")
        if selection_identity(previous) != previous["selection_id"]:
            raise ValueError("comparison selection manifest was modified")
        entries, schedule = previous["samples"], previous["schedule"]
        for position, entry in enumerate(entries):
            i = entry["original_cache_index"]
            if (entry["sample_id"] != i or entry["filename"] != f"{i:06d}.npy"
                    or entry["subset_cache_index"] != position
                    or list(cache[FIELDS[0]][i]) != entry["info"]
                    or list(cache[FIELDS[-1]][i]) != entry["frame_id"]
                    or int(cache[FIELDS[1]][i]) != entry["len"]):
                raise ValueError("reused selection's original indices/metadata do not match GT")
        options = previous["options"]
    else:
        options = state["selection_options"]
        entries, schedule = choose_samples(cache, options["strides_per_motion"], options["max_motions"], options["seed"])
    if not entries:
        raise ValueError("no inference samples selected")
    if sorted(schedule) != sorted(e["sample_id"] for e in entries) or len(set(schedule)) != len(schedule):
        raise ValueError("selection schedule has duplicated/missing sample IDs")
    subset = subset_cache(cache, entries)
    del cache  # no generator exists in this CPU process
    staging = output / (".selection-" + uuid.uuid4().hex)
    staging.mkdir()
    keys = [key for key in source["sequence_keys"] if any(e["info"][0] == key for e in entries)]
    metadata = write_cache(staging / "cache.pkl", subset, cache_spec(keys, state["split"]))
    selection = dict(version=1, split=state["split"], source_cache=state["cache"],
                     source_cache_sha256=source["sha256"], source_segments=source["segment_count"],
                     options=options, samples=entries, schedule=schedule)
    selection["selection_id"] = selection_identity(selection)
    selection.update(subset_cache=str(final / "cache.pkl"), subset_cache_sha256=metadata["sha256"],
                     unique_motions=len({tuple(e["info"]) for e in entries}), total_samples=len(entries),
                     object_ids=subset[OBJECTS], memory=memory_status())
    atomic_json(staging / "selection.json", selection)
    os.rename(staging, final)  # final never exists; unfinished attempts remain separate
    return selection


def verify_selection(state, selection):
    if selection["source_cache_sha256"] != state["cache_sha256"] or selection["split"] != state["split"]:
        raise ValueError("selection cache/split differs from this experiment")
    metadata = read_json(manifest_path(selection["subset_cache"]))
    if (metadata["sha256"] != selection["subset_cache_sha256"]
            or sha256_file(selection["subset_cache"]) != selection["subset_cache_sha256"]):
        raise ValueError("selected GT cache changed")
    if selection_identity(selection) != selection["selection_id"]:
        raise ValueError("immutable selection manifest changed")
    if state.get("selection_id") and state["selection_id"] != selection["selection_id"]:
        raise ValueError("selection changed since experiment initialization")
    frozen = state.get("selection_manifest_sha256")
    if frozen and sha256_file(Path(state["output_dir"]) / "selection/selection.json") != frozen:
        raise ValueError("immutable selection artifact changed since initialization")


def validate_motion(array):
    array = np.asarray(array)
    if (array.shape != (160, 99) or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()):
        raise ValueError("generated motion must be a finite floating [160,99] array")
    return array


def expected_receipt(state, selection, entry):
    return dict(selection_id=selection["selection_id"], checkpoint_sha256=state["checkpoint_sha256"],
                sample_id=entry["sample_id"], seed=entry["seed"])


def validated_receipt(state, selection, entry, repair_link=False):
    output = Path(state["output_dir"])
    committed = output / "completed" / f"{entry['sample_id']:06d}"
    flat = output / "motions" / entry["filename"]
    if not committed.exists():
        if flat.exists():
            raise ValueError(f"untracked output will not be adopted/overwritten: {flat}")
        return None
    receipt = read_json(committed / "receipt.json")
    for key, value in expected_receipt(state, selection, entry).items():
        if receipt.get(key) != value:
            raise ValueError(f"sample {entry['sample_id']}: committed output identity mismatch: {key}")
    motion = committed / "motion.npy"
    if motion.stat().st_size != receipt["size_bytes"] or sha256_file(motion) != receipt["sha256"]:
        raise ValueError(f"sample {entry['sample_id']}: corrupt committed output; preserved for inspection")
    validate_motion(np.load(motion, allow_pickle=False))
    if flat.exists():
        if sha256_file(flat) != receipt["sha256"]:
            raise ValueError(f"sample {entry['sample_id']}: flat output differs from committed output")
    elif repair_link:
        flat.parent.mkdir(exist_ok=True)
        os.link(motion, flat)
    return receipt


def commit_motion(state, selection, entry, array, seconds):
    output = Path(state["output_dir"])
    array = validate_motion(array)
    (output / "completed").mkdir(exist_ok=True)
    (output / "motions").mkdir(exist_ok=True)
    staging = output / (".sample-" + uuid.uuid4().hex)
    staging.mkdir()
    motion = staging / "motion.npy"
    with motion.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = dict(expected_receipt(state, selection, entry), sha256=sha256_file(motion),
                   size_bytes=motion.stat().st_size, seconds=seconds, completed_at=time.time())
    atomic_json(staging / "receipt.json", receipt)
    os.rename(staging, output / "completed" / f"{entry['sample_id']:06d}")
    os.link(output / "completed" / f"{entry['sample_id']:06d}" / "motion.npy",
            output / "motions" / entry["filename"])
    return receipt


def output_inventory(state, selection, repair_links=False):
    receipts, errors = {}, {}
    for entry in selection["samples"]:
        try:
            receipt = validated_receipt(state, selection, entry, repair_links)
            if receipt is not None:
                receipts[entry["sample_id"]] = receipt
        except (ValueError, OSError, KeyError) as exc:
            errors[entry["sample_id"]] = str(exc)
    return receipts, errors


def progress_record(selection, receipts, failures=None, **extra):
    groups = defaultdict(list)
    for entry in selection["samples"]:
        groups[tuple(entry["info"])].append(entry["sample_id"])
    done = set(receipts)
    seconds = sum(r["seconds"] for r in receipts.values())
    average = seconds / len(done) if done else None
    remaining = len(selection["samples"]) - len(done)
    return dict(total_unique_motions=len(groups),
                covered_unique_motions=sum(any(i in done for i in ids) for ids in groups.values()),
                completed_unique_motions=sum(all(i in done for i in ids) for ids in groups.values()),
                strides_per_motion=selection["options"]["strides_per_motion"],
                total_samples=len(selection["samples"]), generated_samples=len(done), remaining_samples=remaining,
                failed_samples={str(i): error for i, error in (failures or {}).items() if i not in done},
                sample_compute_seconds=seconds, average_seconds_per_sample=average,
                samples_per_second=1 / average if average and average > 0 else None,
                estimated_remaining_seconds=remaining * average if average is not None else None,
                updated_at=time.time(), **extra)


def write_evaluation_manifest(state, selection, receipts):
    # Full selection is usable with subset-cache matching only when all selected
    # outputs exist. export_handoff produces exact completed-only variant caches.
    samples = [dict(entry, sha256=receipts[entry["sample_id"]]["sha256"])
               for entry in selection["samples"] if entry["sample_id"] in receipts]
    value = dict(selection, samples=samples, generated_dir=str(Path(state["output_dir"]) / "motions"),
                 unmatched_generated=[], checkpoint=state["checkpoint"], checkpoint_sha256=state["checkpoint_sha256"],
                 generation_time_pairing=True, complete=len(samples) == selection["total_samples"])
    atomic_json(Path(state["output_dir"]) / "manifest.json", value)


def export_handoff(state, selection, destination):
    """One variant per motion per tree: stock refinement deduplicates info keys."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"handoff directory exists; choose a new snapshot: {destination}")
    verify_selection(state, selection)
    receipts, errors = output_inventory(state, selection)
    if errors or not receipts:
        raise ValueError(f"no valid completed outputs or corrupt outputs: {errors}")
    cache, _ = read_cache(selection["subset_cache"], require_manifest=True, split_name=state["split"])
    destination.mkdir(parents=True)
    for slot in sorted({e["variant_slot"] for e in selection["samples"]}):
        entries = [e for e in selection["samples"] if e["variant_slot"] == slot and e["sample_id"] in receipts]
        if not entries:
            continue
        tree = destination / f"variant_{slot:02d}"
        (tree / "motions").mkdir(parents=True)
        subset = {field: [cache[field][e["subset_cache_index"]] for e in entries] for field in FIELDS}
        subset[OBJECTS] = sorted({oid for traj in subset[FIELDS[-2]] for oid in traj})
        keys = list(dict.fromkeys(e["info"][0] for e in entries))
        metadata = write_cache(tree / "cache.pkl", subset, cache_spec(keys, state["split"]))
        for entry in entries:
            source = Path(state["output_dir"]) / "completed" / f"{entry['sample_id']:06d}" / "motion.npy"
            try:
                os.link(source, tree / "motions" / entry["filename"])
            except OSError:
                shutil.copyfile(source, tree / "motions" / entry["filename"])
        manifest = dict(source_cache=selection["source_cache"], source_cache_sha256=selection["source_cache_sha256"],
                        source_segments=selection["source_segments"], subset_cache=str(tree / "cache.pkl"),
                        subset_cache_sha256=metadata["sha256"], selection_id=selection["selection_id"],
                        generated_dir=str(tree / "motions"), unmatched_generated=[], generation_time_pairing=True,
                        samples=[dict(e, sha256=receipts[e["sample_id"]]["sha256"]) for e in entries])
        atomic_json(tree / "manifest.json", manifest)
    return destination
