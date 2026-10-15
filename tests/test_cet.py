"""Lightweight CET tests: python -m unittest discover -s tests -p test_cet.py -v."""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
import evaluate_cet as cet


def xyz(x, y, z):
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))


def pose99(rotations):
    return np.concatenate((np.zeros((len(rotations), 3)), rotations[..., :2, :].reshape(len(rotations), 96)), axis=1)


class MetricTests(unittest.TestCase):
    def test_identical_and_constant_difference(self):
        a = np.tile(np.linspace(0, 0.6, 9)[:, None], (1, 15))
        self.assertAlmostEqual(cet.constrained_dtw(a, a)["cet_degrees"], 0)
        r = cet.constrained_dtw(np.zeros((9, 15)), np.full((9, 15), np.deg2rad(17)))
        self.assertAlmostEqual(r["cet_degrees"], 17)
        np.testing.assert_allclose(r["per_joint_degrees"], 17)

    def test_shift_and_shared_path(self):
        a = np.zeros((31, 15))
        a[10:16] = np.arange(1, 16) * 0.03
        b = np.zeros_like(a)
        b[12:18] = a[10:16]
        r = cet.constrained_dtw(a, b, rho=0.10)
        self.assertAlmostEqual(r["cet_degrees"], 0)
        self.assertGreater(r["unwarped_mae_degrees"], 0)
        self.assertGreater(r["W"], 0)
        path = r["path"]
        np.testing.assert_array_equal(path[[0, -1]], [[0, 0], [30, 30]])
        self.assertTrue(all(tuple(s) in ((1, 1), (1, 0), (0, 1)) for s in np.diff(path, axis=0)))
        self.assertTrue(np.all(np.abs(path[:, 0] / 30 - path[:, 1] / 30) <= .10 + 1e-12))
        np.testing.assert_allclose(r["aligned_abs_error_degrees"],
                                   np.rad2deg(np.abs(a[path[:, 0]] - b[path[:, 1]])))
        self.assertEqual(r["aligned_abs_error_degrees"].shape, (len(path), 15))

    def test_per_joint_and_finger_aggregation(self):
        rng = np.random.default_rng(11)
        a, b = rng.normal(size=(8, 15)), rng.normal(size=(8, 15))
        w = np.arange(15, dtype=float)
        r = cet.constrained_dtw(a, b, weights=w, rho=.25)
        self.assertAlmostEqual(r["cet_degrees"], r["per_joint_degrees"] @ w / w.sum())
        np.testing.assert_allclose(r["per_finger_degrees"], r["per_joint_degrees"].reshape(5, 3).mean(1))
        self.assertAlmostEqual(np.rad2deg(r["cumulative_cost_radians"] / r["path_length"]), r["cet_degrees"])

    def test_cumulative_objective_not_average_objective(self):
        # Enumerate all endpoint paths, independently checking the recurrence.
        a = np.repeat(np.array([0., 0., 1.])[:, None], 15, axis=1)
        b = np.repeat(np.array([1., 1., 2.])[:, None], 15, axis=1)
        paths = []
        def walk(path):
            i, j = path[-1]
            if (i, j) == (2, 2):
                paths.append(path)
            for di, dj in ((1, 0), (0, 1), (1, 1)):
                if i + di <= 2 and j + dj <= 2:
                    walk(path + [(i + di, j + dj)])
        walk([(0, 0)])
        totals = [sum(abs(a[i, 0] - b[j, 0]) for i, j in p) for p in paths]
        r = cet.constrained_dtw(a, b, rho=1)
        self.assertAlmostEqual(r["cumulative_cost_radians"], min(totals))
        self.assertEqual(r["path_length"], 3)
        best_average = min(total / len(path) for total, path in zip(totals, paths))
        self.assertLess(best_average, np.deg2rad(r["cet_degrees"]))

    def test_infeasible_singletons_and_invalid(self):
        z = np.zeros((1, 15))
        self.assertEqual(cet.constrained_dtw(z, z)["cet_degrees"], 0)
        for a, b in ((z, np.zeros((2, 15))), (np.zeros((2, 15)), np.zeros((3, 15)))):
            with self.assertRaises(cet.InfeasiblePath):
                cet.constrained_dtw(a, b)
        for a in (np.zeros((0, 15)), np.full((2, 15), np.nan)):
            with self.assertRaises(cet.InvalidInput):
                cet.constrained_dtw(a, np.zeros((2, 15)))
        with self.assertRaises(cet.InvalidInput):
            cet.constrained_dtw(z, z, weights=np.zeros(15))
        np.testing.assert_allclose(cet.normalized_time([100, 102, 110]), [0, .2, 1])
        with self.assertRaises(cet.InvalidInput):
            cet.normalized_time([5, 5])

    def test_descriptors_keep_posture_and_activity_separate(self):
        a = np.tile(np.array([0, 10, 0, -10])[:, None], (1, 15))
        d = cet.curl_descriptors(a)
        self.assertEqual(d["mean_signed_flexion_degrees"]["whole_hand"], 0)
        self.assertEqual(d["mean_absolute_flexion_degrees"]["whole_hand"], 5)
        self.assertEqual(d["flexion_range_degrees"]["whole_hand"], 20)
        self.assertEqual(d["total_angular_variation_degrees"]["whole_hand"], 30)
        self.assertNotIn("variation_degrees_per_second", d)
        self.assertEqual(cet.curl_descriptors(a, seconds=2)["variation_degrees_per_second"]["whole_hand"], 15)

    def test_lossless_artifact_reproduces_score_without_mano(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=(9, 15)), rng.normal(size=(9, 15))
        frames = np.arange(20, 20 + 9 * 12, 12)
        u = cet.normalized_time(frames)
        r = cet.constrained_dtw(a, b, u, u, rho=.25)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "000007_R.npz"
            cet.save_motion_arrays(path, a, b, frames, u, r, "R", 7, np.tile(np.eye(3), (16, 1, 1)), .25, 120)
            with np.load(path, allow_pickle=False) as z:
                p = z["dtw_frame_pairs"]
                errors = np.abs(z["prediction_curl_degrees"][p[:, 0]] - z["GT_curl_degrees"][p[:, 1]])
                score = (errors @ z["joint_weights"] / z["joint_weights"].sum()).mean()
                self.assertAlmostEqual(score, r["cet_degrees"])
                np.testing.assert_allclose(errors, z["aligned_abs_error_degrees"])
                self.assertEqual(int(z["original_motion_index"]), 7)
                np.testing.assert_array_equal(z["joint_names"], cet.JOINT_NAMES)
                np.testing.assert_allclose(z["prediction_time_seconds"], np.arange(9) / 10)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.basis = np.stack([xyz(*angles) for angles in rng.uniform(-1, 1, (16, 3))])
        self.rotations = np.tile(np.eye(3), (4, 16, 1, 1))
        self.flexion = np.tile(np.linspace(.1, .8, 15), (4, 1))
        for t in range(4):
            for j in range(1, 16):
                self.rotations[t, j] = self.basis[j] @ xyz(.05, .2, self.flexion[t, j - 1]) @ self.basis[j].T

    def test_flexion_separates_spread_and_twist(self):
        np.testing.assert_allclose(cet.curl_from_local_rotations(self.rotations, self.basis), self.flexion, atol=1e-12)
        for j in range(1, 16):
            self.rotations[:, j] = self.basis[j] @ xyz(.3, .6, 0) @ self.basis[j].T
        np.testing.assert_allclose(cet.curl_from_local_rotations(self.rotations, self.basis), 0, atol=1e-12)

    def test_global_translation_and_rotation_and_6d_rows(self):
        pose = pose99(self.rotations)
        expected = cet.curl_from_local_rotations(cet.decode_pose(pose, 4), self.basis)
        pose[:, :3] = [30, -20, 100]
        root = xyz(1., .6, -.8)
        pose[:, 3:9] = root[:2].reshape(6)
        np.testing.assert_allclose(cet.curl_from_local_rotations(cet.decode_pose(pose, 4), self.basis), expected, atol=1e-12)
        np.testing.assert_allclose(cet.decode_pose(pose99(self.rotations), 4), self.rotations, atol=1e-12)

    def test_padding_degeneracy_singularity(self):
        pose = pose99(self.rotations)
        pose[2:] = np.nan
        self.assertEqual(cet.decode_pose(pose, 2).shape, (2, 16, 3, 3))
        with self.assertRaises(cet.InvalidInput):
            cet.decode_pose(pose, 4)
        with self.assertRaises(cet.InvalidInput):
            cet.decode_pose(np.zeros((1, 99)), 1)
        pose = pose99(self.rotations)
        pose[0, 12:15] = pose[0, 9:12]
        with self.assertRaises(cet.InvalidInput):
            cet.decode_pose(pose, 4)
        self.rotations[:, 1] = self.basis[1] @ xyz(0, np.pi / 2, .5) @ self.basis[1].T
        with self.assertRaisesRegex(cet.InvalidInput, "singularity"):
            cet.curl_from_local_rotations(self.rotations, self.basis)


class PairingTests(unittest.TestCase):
    def test_original_indices_dedup_missing_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            g, r = root / "G", root / "R"
            g.mkdir()
            infos = [("s/a", "p:0", "rh"), ("s/b", "p:1", "lh"), ("s/a", "p:0", "rh")]
            ids = [7, 2, 8]  # deliberately not sorted by filename
            cache = {cet.PREFIX + "info_list": infos}
            rots = np.tile(np.eye(3), (4, 16, 1, 1))
            for field, val in (("len", 2), ("frame_id", [10, 12]), ("text", "description"), ("pose", rots)):
                cache[cet.PREFIX + field + "_list"] = [val] * 3
            cache[cet.PREFIX + "hand_side_list"] = [i[2] for i in infos]
            cp = root / "cache.pkl"
            cp.write_bytes(pickle.dumps(cache))
            samples = []
            for index, info in zip(ids, infos):
                p = g / f"{index:06d}.npy"
                np.save(p, pose99(rots))
                samples.append(dict(sample_id=index, filename=p.name, sha256=cet.sha256(p), info=info, len=2, frame_id=[10, 12]))
            mp = root / "manifest.json"
            mp.write_text(json.dumps(dict(samples=samples, source_cache="cache.pkl", source_cache_sha256=cet.sha256(cp),
                                          subset_cache_sha256=cet.sha256(cp), source_segments=10, unmatched_generated=[])))
            rp = cet.refined_path(r, infos[0])
            rp.parent.mkdir(parents=True)
            rp.write_bytes(pickle.dumps(dict(info=infos[0], hand_side="rh", len=2, frame_id=[10, 12], refine_pose_repr=pose99(rots))))
            _, entries, inv = cet.inspect_batch(mp, cp, g, r, root)
            self.assertEqual([e["original_motion_index"] for e in entries], [7, 2])
            self.assertEqual(inv["duplicate_segments_skipped"], 1)
            self.assertEqual(entries[0]["representative_G_index"], 7)
            self.assertEqual(entries[1]["status"], "missing")
            self.assertEqual(inv["paired_motions"], 1)
            bad = pickle.loads(rp.read_bytes())
            bad["frame_id"] = [10, 11]
            rp.write_bytes(pickle.dumps(bad))
            _, entries, _ = cet.inspect_batch(mp, cp, g, r, root)
            self.assertEqual(entries[0]["status"], "invalid")
            self.assertIn("frame IDs mismatch", entries[0]["reason"])


class ManoConventionTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "asset/mano_v1_2/models/MANO_RIGHT.pkl").is_file(), "local MANO unavailable")
    def test_actual_manotorch_both_sides_and_global_invariance(self):
        import torch
        sys.path.insert(0, str(ROOT / "thirdparty/manotorch"))
        from manotorch.axislayer import AxisLayerFK
        from manotorch.manolayer import ManoLayer
        from manotorch.utils.geometry import axis_angle_to_matrix
        for side in ("right", "left"):
            axes = AxisLayerFK(side=side, mano_assets_root=str(ROOT / "asset/mano_v1_2"))
            mano = ManoLayer(side=side, mano_assets_root=str(ROOT / "asset/mano_v1_2"))
            self.assertEqual(tuple(mano.kintree_parents[1:]), cet.PARENTS[1:])
            angles = torch.zeros(2, 16, 3)
            angles[:, 1:, 1] = .35  # spread, including MCP/thumb
            angles[1, 1:, 2] = torch.linspace(.05, .8, 15)
            local = axes.compose(angles)
            output = mano(local.reshape(2, 48), torch.zeros(2, 10))
            _, _, official = axes(output.transforms_abs)
            basis = axes.TMPL_R_p_a.numpy()[0]
            extracted = cet.curl_from_local_rotations(axis_angle_to_matrix(local).numpy(), basis)
            np.testing.assert_allclose(extracted, official.numpy()[:, 1:, 2], atol=2e-6)
            np.testing.assert_allclose(extracted, angles.numpy()[:, 1:, 2], atol=2e-6)
            # Apply a rigid transform to the actual MANO global transforms.
            transform = torch.eye(4)
            transform[:3, :3] = torch.tensor(xyz(.8, -.7, 1.1), dtype=torch.float32)
            transform[:3, 3] = torch.tensor([10., -2., 7.])
            _, _, moved = axes(transform @ output.transforms_abs)
            np.testing.assert_allclose(moved.numpy()[:, 1:, 2], extracted, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
