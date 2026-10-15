import copy
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oakink2_tamf.dataset.cache_util import (
    CacheError, FIELDS, OBJECTS, build_cache, cache_spec, manifest_path,
    publish_new, read_cache, validate_cache, write_cache, sha256_source,
)
from workflow_fixtures import tiny_cache


class CacheWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "train.pkl"
        self.keys = ["seq_a", "seq_b"]

    def test_roundtrip_and_compatible_existing_cache_never_processes(self):
        cache = tiny_cache()
        write_cache(self.path, cache, cache_spec(self.keys, "train"))
        before = self.path.read_bytes()
        process = Mock(side_effect=AssertionError("must reuse existing cache"))
        metadata = build_cache(self.path, self.keys, "train", process)
        process.assert_not_called()
        self.assertEqual(metadata["segment_count"], 2)
        actual, _ = read_cache(self.path, self.keys, require_manifest=True, split_name="train")
        for key in FIELDS:
            for a, b in zip(actual[key], cache[key]):
                if hasattr(a, "shape"):
                    import numpy as np
                    np.testing.assert_array_equal(a, b)
                elif isinstance(a, dict):
                    import numpy as np
                    for oid in a:
                        np.testing.assert_array_equal(a[oid], b[oid])
                else:
                    self.assertEqual(a, b)
        self.assertEqual(self.path.read_bytes(), before)

    def test_restart_reuses_completed_sequences_and_preserves_order(self):
        seen = []
        def process(key):
            seen.append(key)
            if key == "seq_b":
                raise RuntimeError("interrupted")
            return tiny_cache([key])
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            build_cache(self.path, self.keys, "train", process)
        self.assertFalse(self.path.exists())
        with self.assertRaisesRegex(CacheError, "resume"):
            build_cache(self.path, self.keys, "train", process)
        processed = []
        build_cache(self.path, self.keys, "train",
                    lambda key: (processed.append(key), tiny_cache([key]))[1], resume=True)
        self.assertEqual(processed, ["seq_b"])
        result, _ = read_cache(self.path, self.keys, require_manifest=True)
        self.assertEqual([info[0] for info in result[FIELDS[0]]], self.keys)

    def test_corruption_settings_split_and_count_mismatches(self):
        write_cache(self.path, tiny_cache(), cache_spec(self.keys, "train"))
        with self.assertRaises(CacheError):
            read_cache(self.path, list(reversed(self.keys)), require_manifest=True)
        with self.assertRaises(CacheError):
            read_cache(self.path, self.keys, split_name="val")
        sidecar = manifest_path(self.path)
        metadata = json.loads(sidecar.read_text())
        metadata["slicing"]["target_fps"] = 30
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(CacheError, "slicing"):
            read_cache(self.path, self.keys)
        metadata["slicing"]["target_fps"] = 10
        metadata["segment_count"] = 99
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(CacheError, "segment count"):
            read_cache(self.path, self.keys)
        with self.path.open("ab") as stream:
            stream.write(b"damage")
        with self.assertRaisesRegex(CacheError, "checksum"):
            read_cache(self.path, self.keys)

    def test_legacy_cache_is_not_overwritten_or_silently_adopted(self):
        with self.path.open("wb") as stream:
            pickle.dump(tiny_cache(), stream)
        before = self.path.read_bytes()
        cache, meta = read_cache(self.path, self.keys)
        self.assertEqual(len(cache[FIELDS[0]]), 2)
        self.assertIsNone(meta)
        with self.assertRaises(CacheError):
            read_cache(self.path, self.keys, require_manifest=True)
        with self.assertRaisesRegex(CacheError, "provenance"):
            build_cache(self.path, self.keys, "train", Mock())
        self.assertEqual(self.path.read_bytes(), before)

    def test_incomplete_publication_and_no_clobber(self):
        def fail(stream):
            stream.write(b"partial")
            raise RuntimeError("disk failure")
        with self.assertRaises(RuntimeError):
            publish_new(self.path, fail)
        self.assertFalse(self.path.exists())
        self.assertTrue(list(self.path.parent.glob("*.incomplete-*")))
        publish_new(self.path, lambda stream: stream.write(b"complete"))
        with self.assertRaises(FileExistsError):
            publish_new(self.path, lambda stream: stream.write(b"overwrite"))
        self.assertEqual(self.path.read_bytes(), b"complete")

    def test_shape_dtype_mask_inventory_and_padding_checks(self):
        self.assertEqual(validate_cache(tiny_cache()), 2)
        cache = tiny_cache()
        cache[OBJECTS].append("unused_object")  # original collection before short-segment filtering
        self.assertEqual(validate_cache(cache), 2)
        for mutate in (
            lambda c: c[FIELDS[1]].pop(),
            lambda c: c[FIELDS[2]][0].__setitem__((159, 0, 0, 0), 1),
            lambda c: c[FIELDS[3]][0].__setitem__((0, 0), float("nan")),
            lambda c: c[FIELDS[8]][0].reverse(),
        ):
            cache = tiny_cache()
            mutate(cache)
            with self.assertRaises(CacheError):
                validate_cache(cache)

    def test_source_identity_is_portable_across_windows_line_endings(self):
        lf, crlf = self.path.parent / "lf.py", self.path.parent / "crlf.py"
        lf.write_bytes(b"x = 1\ny = 2\n")
        crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
        self.assertEqual(sha256_source(lf), sha256_source(crlf))


if __name__ == "__main__":
    unittest.main()
