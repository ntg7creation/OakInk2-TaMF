import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oakink2_tamf.dataset.cache_util import build_cache, read_cache, write_cache, cache_spec
from oakink2_tamf.dataset.collate import interaction_segment_collate
from oakink2_tamf.dataset.interaction_segment import InteractionSegmentData
from oakink2_tamf.dataset.training_data import (
    ReverseSegmentView, build_datasets, build_loaders, evaluation_enabled, recording_due,
)
from workflow_fixtures import tiny_cache, configuration


class Primitive(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


class TinyToolkit:
    """Only replace raw I/O, leaving collect/slice/conversion code intact."""
    def __init__(self, **kwargs):
        pass

    def load_complex_task(self, seq_key):
        return SimpleNamespace(exec_path=["primitive"], seq_key=seq_key)

    def load_primitive_task(self, complex_task_data):
        side = "lh" if complex_task_data.seq_key.endswith("b") else "rh"
        length = 33
        quaternions = torch.zeros(length, 16, 4)
        quaternions[..., 0] = 1
        angle = torch.linspace(0, 0.4, length)
        quaternions[:, 4, 0] = torch.cos(angle / 2)
        quaternions[:, 4, 3] = torch.sin(angle / 2)
        param = dict(pose_coeffs=quaternions, tsl=torch.arange(length * 3).reshape(length, 3).float() / 100,
                     betas=torch.ones(length, 10) * 0.1)
        objects = ["obj_a", "obj_b"] if side == "lh" else ["obj_a"]
        transforms = {key: np.tile(np.eye(4, dtype=np.float32), (length, 1, 1)) for key in objects}
        return [Primitive(frame_range=(0, length), hand_involved=side,
                          task_desc="hold the object", obj_transf=transforms,
                          **{f"frame_range_{side}": (0, length), f"{side}_obj_list": objects,
                             f"{side}_param": param, f"{side}_in_range_mask": torch.ones(length, dtype=torch.bool)})]


def assert_samples(test, actual, expected):
    test.assertEqual(set(actual), set(expected))
    for key in actual:
        a, b = actual[key], expected[key]
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=key)
        elif isinstance(a, torch.Tensor):
            test.assertTrue(torch.equal(a, b), key)
        else:
            test.assertEqual(a, b, key)


class TrainingDataTests(unittest.TestCase):
    def base(self, cache=None, **kwargs):
        return InteractionSegmentData(["seq_a", "seq_b"], "not-needed",
                                      cache_dict=tiny_cache() if cache is None else cache, **kwargs)

    def test_cached_loading_does_not_initialize_raw_toolkit(self):
        with patch("oakink2_tamf.dataset.interaction_segment.OakInk2__Dataset",
                   side_effect=AssertionError("raw toolkit must not be constructed")):
            base = self.base()
            self.assertEqual(len(base), 2)
            self.assertIsNone(base.dataset)
            self.assertEqual(base[0]["pose_repr"].shape, (160, 99))

    def test_reverse_view_exactly_matches_eager_including_both_hands_and_padding(self):
        base = self.base()
        view = ReverseSegmentView(base)
        original = copy.deepcopy(base.get_cache())
        eager = self.base(append_reverse_segment=True)
        self.assertEqual(len(view), len(eager))
        for index in range(len(view)):
            assert_samples(self, view[index], eager[index])
        for key in ("interaction_segment_pose_list", "interaction_segment_tsl_list", "interaction_segment_shape_list"):
            for a, b in zip(base.get_cache()[key], original[key]):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(view[-1]["frame_id"], eager[len(eager) - 1]["frame_id"])
        with self.assertRaises(IndexError):
            view[len(view)]

    def test_full_length_and_static_geometry_are_preserved(self):
        cache = tiny_cache(["seq_a"])
        length = cache["interaction_segment_len_list"][0]
        cache["interaction_segment_len_list"][0] = 160
        cache["interaction_segment_frame_id_list"][0] = list(range(160))
        for field in ("pose", "tsl", "shape"):
            arr = cache[f"interaction_segment_{field}_list"][0]
            arr[length:] = arr[length - 1]
        cache["interaction_segment_obj_traj_list"][0]["obj_a"][length:] = np.eye(4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.save(torch.ones(768), root / "obj_a.pt")
            np.savez(root / "obj_a.npz", point=np.ones((8, 3), np.float32))
            kwargs = dict(obj_embedding_prefix=str(root), obj_pointcloud_prefix=str(root))
            view = ReverseSegmentView(self.base(cache, **kwargs))
            eager = self.base(cache, append_reverse_segment=True, **kwargs)
            for index in range(2):
                actual, expected = view[index], eager[index]
                # Point clouds are a list of arrays rather than a stacked array.
                a, b = actual.pop("obj_pointcloud"), expected.pop("obj_pointcloud")
                np.testing.assert_array_equal(a, b)
                assert_samples(self, actual, expected)

    def test_disabled_splits_never_construct_or_open_invalid_paths(self):
        cfg = configuration()
        cfg["val"]["cache_dict_filepath"] = "/does/not/exist/val.pkl"
        cfg["test"]["cache_dict_filepath"] = "/does/not/exist/test.pkl"
        factory = Mock(return_value=self.base())
        datasets, _ = build_datasets(cfg, 0, factory)
        factory.assert_called_once()
        self.assertIsNone(datasets["val"])
        self.assertIsNone(datasets["test"])
        cfg["train"]["require_verified_cache"] = True
        with self.assertRaisesRegex(ValueError, "requires a cache"):
            build_datasets(cfg, 0, factory)

    def test_enabled_splits_only_load_on_rank_zero(self):
        for rank, expected in ((0, 3), (1, 1)):
            cfg = configuration()
            cfg["val"]["val_freq"] = 1
            cfg["test"]["test_freq"] = 2
            factory = Mock(return_value=self.base())
            datasets, _ = build_datasets(cfg, rank, factory)
            self.assertEqual(factory.call_count, expected)
            self.assertEqual(datasets["val"] is not None, rank == 0)

    def test_frequencies(self):
        for value in (None, -1):
            self.assertFalse(evaluation_enabled(value))
        for value in (1, 20):
            self.assertTrue(evaluation_enabled(value))
        for value in (0, -2, 1.5):
            with self.assertRaises(ValueError):
                evaluation_enabled(value)
        self.assertTrue(recording_due(0, 400, 20))
        self.assertTrue(recording_due(19, 400, 20))
        self.assertFalse(recording_due(18, 400, 20))

    def test_zero_and_spawn_workers_return_same_samples(self):
        cfg = configuration()
        datasets = dict(train=self.base(), val=None, test=None)
        outputs = []
        for workers in (0, 1):
            cfg["runtime"]["num_worker"] = workers
            loaders, sampler = build_loaders(datasets, cfg, 0, 1, interaction_segment_collate,
                                            torch.Generator().manual_seed(0))
            sampler.set_epoch(0)
            outputs.append(list(loaders["train"]))
            self.assertIsNone(loaders["val"])
            self.assertEqual(loaders["train"].persistent_workers, workers > 0)
        for a, b in zip(*outputs):
            for field in ("pose_repr", "shape", "obj_traj", "mask"):
                torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)
            self.assertEqual(a["frame_id"], b["frame_id"])

    def test_uncached_original_vs_restartable_cache_loader(self):
        keys = ["seq_a", "seq_b"]
        with tempfile.TemporaryDirectory() as temp, patch(
                "oakink2_tamf.dataset.interaction_segment.OakInk2__Dataset", TinyToolkit):
            original = InteractionSegmentData(keys, "unused")
            path = Path(temp) / "train.pkl"
            build_cache(path, keys, "train", lambda key: InteractionSegmentData([key], "unused").get_cache())
            cfg = configuration(keys)
            cfg["train"].update(cache_dict_filepath=str(path), require_verified_cache=True)
            with patch.object(InteractionSegmentData, "load_dataset",
                              side_effect=AssertionError("must not preprocess")):
                datasets, identities = build_datasets(cfg, 0)
                loaded = datasets["train"]
                self.assertEqual(len(loaded), len(original))
                self.assertIn("sha256", identities["train"])
                for index in range(len(original)):
                    assert_samples(self, loaded[index], original[index])


if __name__ == "__main__":
    unittest.main()
