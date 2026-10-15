"""Portable, validated legacy TaMF pickle caches (trusted local files only).

Final caches keep the original ten fields. JSON sidecars carry provenance and
checksums; sequence work files use exactly the same pickle representation.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)
CACHE_VERSION = 1
FIELDS = tuple("interaction_segment_" + name + "_list" for name in
               ("info", "len", "pose", "tsl", "shape", "hand_side", "text", "obj_traj", "frame_id"))
OBJECTS = "interaction_object_list"
SLICING = dict(origin_fps=120, target_fps=10.0, slice_min_len=16, slice_max_len=160)


class CacheError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source(path):
    # Git's Windows checkout policy may use CRLF while Shenkar uses LF.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def preprocessing_fingerprint():
    root = Path(__file__).resolve().parents[3]
    sources = ("src/oakink2_tamf/dataset/interaction_segment.py",
               "src/oakink2_tamf/dataset/setment_slice.py",
               "src/dev_fn/transform/rotation.py", "src/dev_fn/transform/rotation_np.py",
               "src/dev_fn/transform/transform_np.py",
               "thirdparty/OakInk2/src/oakink2_toolkit/dataset.py",
               "thirdparty/OakInk2/src/oakink2_toolkit/tool.py",
               "thirdparty/OakInk2/src/oakink2_toolkit/meta.py")
    return {name: sha256_source(root / name) for name in sources}


def cache_spec(sequence_keys, split_name=None):
    keys = list(sequence_keys)
    if not keys or len(set(keys)) != len(keys):
        raise CacheError("split must contain unique, ordered sequence identifiers")
    return dict(version=CACHE_VERSION, sequence_keys=keys, split_name=split_name,
                split_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                slicing=SLICING.copy(), augmented=False,
                preprocessing=preprocessing_fingerprint())


def validate_cache(cache, sequence_keys=None, *, allow_empty=False):
    """Validate without copying arrays or interpreting padded zeros as rotations."""
    if not isinstance(cache, dict) or any(key not in cache for key in (*FIELDS, OBJECTS)):
        raise CacheError("cache is missing required TaMF fields")
    count = len(cache[FIELDS[0]])
    if not allow_empty and not count:
        raise CacheError("cache contains no segments")
    if any(len(cache[key]) != count for key in FIELDS):
        raise CacheError("cache parallel-list lengths differ")
    allowed = set(sequence_keys) if sequence_keys is not None else None
    objects = set()
    for index in range(count):
        info, length, pose, tsl, shape, side, text, traj, frames = (cache[k][index] for k in FIELDS)
        if not isinstance(info, (tuple, list)) or len(info) != 3 or info[2] != side:
            raise CacheError(f"segment {index}: invalid info/hand side")
        if allowed is not None and info[0] not in allowed:
            raise CacheError(f"segment {index}: sequence outside requested split")
        if side not in ("lh", "rh") or not isinstance(text, str):
            raise CacheError(f"segment {index}: invalid side/text")
        if not isinstance(length, (int, np.integer)) or not 16 <= length <= 160:
            raise CacheError(f"segment {index}: invalid length")
        if len(frames) != length or any(not isinstance(f, (int, np.integer)) for f in frames):
            raise CacheError(f"segment {index}: invalid frame identifiers")
        if len(frames) > 1 and not np.all(np.diff(frames) > 0):
            raise CacheError(f"segment {index}: expected unaugmented forward frame identifiers")
        arrays = [(pose, (160, 16, 3, 3)), (tsl, (160, 3)), (shape, (160, 10))]
        if not isinstance(traj, dict) or not traj:
            raise CacheError(f"segment {index}: missing object trajectory")
        objects.update(traj)
        arrays.extend((arr, (160, 4, 4)) for arr in traj.values())
        for arr, expected in arrays:
            if not isinstance(arr, np.ndarray) or arr.shape != expected or arr.dtype != np.float32:
                raise CacheError(f"segment {index}: expected float32 array {expected}")
            if not np.isfinite(arr).all() or np.any(arr[length:] != 0):
                raise CacheError(f"segment {index}: nonfinite values or nonzero padding")
    # The original collector adds objects BEFORE rejecting too-short segments.
    # Its inventory can therefore legitimately include unused objects.
    inventory = list(cache[OBJECTS])
    if inventory != sorted(set(inventory)) or not objects.issubset(inventory):
        raise CacheError("cache object inventory does not cover trajectories")
    return count


def manifest_path(path):
    return Path(str(path) + ".manifest.json")


def publish_new(path, writer):
    """Publish a completed file without replacing an existing destination.

    A hard link provides atomic no-clobber publication on Windows and Linux.
    Failure leaves the destination intact; an incomplete temporary file is kept
    on writer failure for diagnosis. Temp and destination share a filesystem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".incomplete-" + uuid.uuid4().hex)
    with open(temporary, "xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()  # only this invocation's successfully published temporary


def write_cache(path, cache, spec):
    path = Path(path)
    if path.exists() or manifest_path(path).exists():
        raise FileExistsError(f"refusing to overwrite cache or manifest: {path}")
    count = validate_cache(cache, spec["sequence_keys"], allow_empty=True)
    publish_new(path, lambda stream: pickle.dump(cache, stream, protocol=4))
    manifest = dict(spec, segment_count=count, size_bytes=path.stat().st_size,
                    sha256=sha256_file(path))
    publish_new(manifest_path(path), lambda stream: stream.write(
        json.dumps(manifest, indent=2).encode("utf-8")))
    return manifest


def read_cache(path, sequence_keys=None, *, require_manifest=False, split_name=None,
               allow_empty=False):
    """Checksum before unpickling; use only trusted caches, never external pickle."""
    path = Path(path)
    sidecar = manifest_path(path)
    manifest = None
    if sidecar.exists():
        try:
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            expected = cache_spec(sequence_keys or manifest["sequence_keys"], split_name)
            for key in ("version", "sequence_keys", "split_sha256", "slicing", "augmented", "preprocessing"):
                if manifest.get(key) != expected[key]:
                    raise CacheError(f"cache compatibility mismatch: {key}")
            if split_name is not None and manifest.get("split_name") != split_name:
                raise CacheError("cache split name mismatch")
            if path.stat().st_size != manifest["size_bytes"] or sha256_file(path) != manifest["sha256"]:
                raise CacheError("cache size/checksum mismatch")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise CacheError(f"invalid cache manifest: {sidecar}") from exc
    elif require_manifest:
        raise CacheError(f"verified cache requires {sidecar}; validate legacy cache separately")
    else:
        LOGGER.warning("Legacy cache has no provenance manifest: %s", path)
    try:
        with open(path, "rb") as stream:
            cache = pickle.load(stream)
            if stream.read(1):
                raise CacheError("unexpected trailing cache data")
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        raise CacheError(f"cannot read complete cache: {path}") from exc
    count = validate_cache(cache, sequence_keys, allow_empty=allow_empty)
    if manifest is not None and manifest.get("segment_count") != count:
        raise CacheError("cache segment count differs from manifest")
    return cache, manifest


def memory_status():
    """Linux RSS/high-water marks; harmless portable fallback on Windows."""
    result = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, unit = line.split()
                result[key.rstrip(":") + "_MiB"] = round(int(value) / 1024, 1)
    except OSError:
        pass
    return result


def build_cache(path, sequence_keys, split_name, processor, *, resume=False):
    """Process one sequence at a time; inspect/reuse existing output first.

    processor(key) returns the original ten-field, unaugmented cache dictionary.
    No existing cache is overwritten, including orphaned incomplete artifacts.
    """
    path = Path(path)
    keys = list(sequence_keys)
    spec = cache_spec(keys, split_name)
    if path.exists():
        cache, manifest = read_cache(path, keys, split_name=split_name)
        if manifest is None:
            raise CacheError("Legacy output is structurally valid but provenance is unknown; "
                             "reuse via legacy training mode or explicitly adopt it after equivalence checks")
        LOGGER.info("Reusing compatible cache %s (%d segments)", path, len(cache[FIELDS[0]]))
        return manifest
    if manifest_path(path).exists():
        raise CacheError("orphan cache manifest; choose a new output path")
    work = Path(str(path) + ".work")
    if work.exists() and not resume:
        raise CacheError(f"preprocessing work exists: use --cache.resume: {work}")
    work.mkdir(parents=True, exist_ok=True)
    for index, key in enumerate(keys):
        start = time.monotonic()
        prefix = f"sequence_{index:05d}"
        candidates = sorted(work.glob(prefix + "-*.pkl"))
        found = None
        for candidate in candidates:
            if not manifest_path(candidate).exists():
                continue  # unfinished attempt; preserve it and write a new attempt
            cache, _ = read_cache(candidate, [key], require_manifest=True,
                                  split_name=split_name, allow_empty=True)
            found = candidate
            del cache
            break
        if found is None:
            cache = processor(key)
            found = work / (prefix + "-" + uuid.uuid4().hex + ".pkl")
            write_cache(found, cache, cache_spec([key], split_name))
            del cache
        gc.collect()
        LOGGER.info("sequence %d/%d %s | %.1fs | %s | %s", index + 1, len(keys), key,
                    time.monotonic() - start, "reused" if candidates and found in candidates else "processed",
                    memory_status())
    merged = {key: [] for key in (*FIELDS, OBJECTS)}
    objects = set()
    for index, key in enumerate(keys):
        candidates = sorted(work.glob(f"sequence_{index:05d}-*.pkl"))
        completed = next(p for p in candidates if manifest_path(p).exists())
        shard, _ = read_cache(completed, [key], require_manifest=True,
                              split_name=split_name, allow_empty=True)
        for field in FIELDS:
            merged[field].extend(shard[field])
        objects.update(shard[OBJECTS])
        del shard
    merged[OBJECTS] = sorted(objects)
    validate_cache(merged, keys)
    LOGGER.info("Publishing %d segments | %s", len(merged[FIELDS[0]]), memory_status())
    return write_cache(path, merged, spec)
