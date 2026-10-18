"""Saved-motion pairing, no-clobber exports and all-five dispatch on CPU fixtures."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Scripts"), str(ROOT / "src")]
import evaluate
import evaluate_cet as cet
from oakink2_tamf.util.evaluation_output import MetricOutput


class SavedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.motions = self.root / "motions"
        self.motions.mkdir()
        self.refined = self.root / "refined"
        self.cache_path = self.root / "test.pkl"
        self.manifest = self.root / "manifest.json"
        self.split = self.root / "test.txt"
        self.split.write_text("seq/test\n")
        self.cache = {cet.PREFIX + field + "_list": [] for field in evaluate.FIELDS}
        self.cache["interaction_object_list"] = ["object"]
        samples = []
        for index, (primitive, length) in enumerate((("hold:00", 16), ("hold:00", 16), ("grip:00", 23))):
            info = ("seq/test", primitive, "rh")
            rotations = np.tile(np.eye(3), (160, 16, 1, 1)).astype(np.float32)
            pose = np.concatenate([np.zeros((160, 3)), rotations[..., :2, :].reshape(160, 96)], axis=1)
            values = dict(info=info, len=length, pose=rotations, tsl=np.zeros((160, 3)),
                          shape=np.zeros((160, 10)), hand_side="rh", text="fixture",
                          obj_traj={"object": np.tile(np.eye(4), (160, 1, 1))}, frame_id=list(range(length)))
            for key, value in values.items():
                self.cache[cet.PREFIX + key + "_list"].append(value)
            path = self.motions / f"{index:06d}.npy"
            np.save(path, pose)
            samples.append(dict(filename=path.name, sample_id=index, sha256=cet.sha256(path),
                                info=info, len=length, frame_id=values["frame_id"]))
            refined_path = cet.refined_path(self.refined, info)
            refined_path.parent.mkdir(parents=True, exist_ok=True)
            record = dict(info=info, process_key=info[0], hand_side="rh", len=length,
                          frame_id=values["frame_id"], obj_list=["object"], refine_pose_repr=pose,
                          verts=np.zeros((160, 778, 3)), joints=np.zeros((160, 21, 3)))
            refined_path.write_bytes(pickle.dumps(record))
        self.cache_path.write_bytes(pickle.dumps(self.cache))
        self.manifest.write_text(json.dumps(dict(source_cache=str(self.cache_path),
            source_cache_sha256=cet.sha256(self.cache_path), source_segments=3, samples=samples)))
        for relative in ("mano/models/MANO_LEFT.pkl", "mano/models/MANO_RIGHT.pkl",
                         "embedding/object.pt", "pointcloud/object.npz", "encoder.pt", "encoder.yml",
                         "data/object_raw/align_ds/object/mesh.obj"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def args(self, *extra):
        args = evaluate.parser().parse_args([
            "--motion-dir", str(self.motions), "--test-cache", str(self.cache_path),
            "--refined-dir", str(self.refined), "--output-dir", str(self.root / "results"),
            "--pairing-manifest", str(self.manifest), "--test-split", str(self.split),
            "--mano-root", str(self.root / "mano"), "--data-root", str(self.root / "data"),
            "--obj-embedding-dir", str(self.root / "embedding"),
            "--obj-pointcloud-dir", str(self.root / "pointcloud"),
            "--fid-encoder", str(self.root / "encoder.pt"), "--fid-config", str(self.root / "encoder.yml"), *extra])
        args.metrics = [*evaluate.OFFICIAL, "cet"] if "all" in args.metrics else args.metrics
        return args

    def test_pairing_deduplication_and_missing_samples_are_explicit(self):
        args = self.args()
        _, report = evaluate.inventory(args)
        self.assertEqual([e["sample_id"] for e in report["samples"]], [0, 2])
        self.assertEqual([e["sample_id"] for e in report["duplicate_segments"]], [1])
        (self.motions / "000001.npy").unlink()
        with self.assertRaisesRegex(cet.InvalidInput, "missing 1 expected"):
            evaluate.inventory(args)
        args.allow_subset = True
        _, report = evaluate.inventory(args)
        self.assertEqual(report["missing_generated_ids"], [1])
        self.assertFalse(args.output_dir.exists())

    def test_wrong_cache_bad_filenames_and_refined_frame_mismatch_fail(self):
        args = self.args()
        with self.cache_path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(cet.InvalidInput, "checksum differs"):
            evaluate.inventory(args)
        self.cache_path.write_bytes(pickle.dumps(self.cache))
        bad = self.motions / "0.npy"
        np.save(bad, np.zeros((160, 99)))
        with self.assertRaisesRegex(cet.InvalidInput, "unexpected/nested"):
            evaluate.inventory(args)
        bad.unlink()
        path = cet.refined_path(self.refined, ("seq/test", "grip:00", "rh"))
        record = cet.read_pickle(path)
        record["frame_id"] = list(range(1, 24))
        path.write_bytes(pickle.dumps(record))
        with self.assertRaisesRegex(cet.InvalidInput, "length/frame IDs mismatch"):
            evaluate.inventory(args)
        self.assertFalse(args.output_dir.exists())

    def test_numeric_ids_without_metadata_need_an_explicit_assertion(self):
        args = self.args()
        args.pairing_manifest = None
        with self.assertRaisesRegex(cet.InvalidInput, "contain no sample metadata"):
            evaluate.inventory(args)
        args.trust_cache_order = True
        _, report = evaluate.inventory(args)
        self.assertIn("explicit user assertion", report["pairing_basis"])

    def test_duplicate_manifest_ids_and_unknown_generated_ids_are_rejected(self):
        manifest = json.loads(self.manifest.read_text())
        original = self.manifest.read_bytes()
        manifest["samples"].append(manifest["samples"][0])
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(cet.InvalidInput, "duplicate sample IDs"):
            evaluate.inventory(self.args())
        self.manifest.write_bytes(original)
        np.save(self.motions / "999999.npy", np.zeros((160, 99)))
        with self.assertRaisesRegex(cet.InvalidInput, "absent from the test cache/manifest"):
            evaluate.inventory(self.args())

    def test_source_subset_manifest_keeps_original_ids_not_positions(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["source_cache_sha256"] = "historical source unavailable"
        manifest["subset_cache_sha256"] = cet.sha256(self.cache_path)
        manifest["source_segments"] = 10
        for sample, original in zip(manifest["samples"], (4, 7, 9)):
            old = self.motions / sample["filename"]
            new = self.motions / f"{original:06d}.npy"
            old.rename(new)
            sample.update(sample_id=original, filename=new.name)
        self.manifest.write_text(json.dumps(manifest))
        _, report = evaluate.inventory(self.args())
        self.assertEqual([(e["sample_id"], e["source_cache_index"]) for e in report["samples"]], [(4, 0), (9, 2)])

    def test_test_split_changed_motion_and_missing_refinement_fail(self):
        args = self.args()
        self.split.write_text("training_sequence\n")
        with self.assertRaisesRegex(cet.InvalidInput, "outside the supplied test split"):
            evaluate.inventory(args)
        self.split.write_text("seq/test\n")
        motion = self.motions / "000000.npy"
        pose = np.load(motion)
        pose[:, 0] = 1
        np.save(motion, pose)
        with self.assertRaisesRegex(cet.InvalidInput, "differs from pairing manifest"):
            evaluate.inventory(args)
        pose[:, 0] = 0
        np.save(motion, pose)
        cet.refined_path(self.refined, ("seq/test", "hold:00", "rh")).unlink()
        with self.assertRaisesRegex(cet.InvalidInput, "required R-stage record missing"):
            evaluate.inventory(args)

    def test_existing_results_and_input_overlap_are_never_overwritten(self):
        args = self.args()
        args.output_dir.mkdir()
        marker = args.output_dir / "summary.json"
        marker.write_text("existing results")
        with self.assertRaisesRegex(cet.InvalidInput, "already exists"):
            evaluate.inventory(args)
        self.assertEqual(marker.read_text(), "existing results")
        args.output_dir = self.motions / "new_results"
        with self.assertRaisesRegex(cet.InvalidInput, "overlaps an input"):
            evaluate.inventory(args)

    def fake_official(self, command, **kwargs):
        metric = command[2].rsplit("_", 1)[1]
        output = Path(command[command.index("--debug.evaluation_output_dir") + 1])
        recorder = MetricOutput(output)
        for i, (info, length) in enumerate((("hold:00", 16), ("grip:00", 23))):
            recorder.add(i, dict(info=("seq/test", info, "rh"), len=length))
        if metric == "cr":
            recorder.frames(np.r_[np.zeros(16), np.ones(23)], np.r_[np.ones(16), np.zeros(23)], contact=True)
        elif metric == "siv":
            recorder.frames([1, 2, 4], [2, 4, 8])
        elif metric == "psklj":
            recorder.save(dict(gt_to_prediction=0., prediction_to_gt=0.),
                          dict(gt_power=np.ones((2, 158, 21, 3)), prediction_power=np.ones((2, 158, 21, 3))))
        else:
            self.assertIn("--debug.cache_dict_filepath", command)
            self.assertNotIn("--data.cache_dict_filepath", command)
            self.assertIn("--debug.evaluation_seed", command)
            recorder.save(dict(fid=0.), dict(gt_features=np.ones((2, 64)), prediction_features=np.ones((2, 64))))
        self.assertEqual(Path(kwargs["cwd"]).parents[1], output.parents[1])
        return argparse.Namespace(returncode=0)

    def test_all_five_metrics_export_original_ids_and_lossless_arrays(self):
        args = self.args()
        bases = {side: np.tile(np.eye(3), (16, 1, 1)) for side in ("rh", "lh")}
        with patch.object(evaluate.subprocess, "run", side_effect=self.fake_official) as dispatch, \
             patch.object(cet, "load_anatomical_bases", return_value=(bases, {})):
            self.assertEqual(evaluate.run(args), 0)
        self.assertEqual(dispatch.call_count, 4)
        summary = json.loads((args.output_dir / "summary.json").read_text())
        self.assertEqual(set(summary["metrics"]), {*evaluate.OFFICIAL, "cet"})
        self.assertTrue(all(x["status"] == "success" for x in summary["metrics"].values()))
        self.assertAlmostEqual(summary["metrics"]["cr"]["scores"]["gt"], 16 / 39)
        self.assertAlmostEqual(summary["metrics"]["siv"]["scores"]["gt"], 7 / 3)
        rows = json.loads((args.output_dir / "motions.json").read_text())
        self.assertEqual([r["sample_id"] for r in rows], [0, 2])
        self.assertTrue(all(r["cet_degrees"] == 0 for r in rows))
        self.assertTrue(all(len(r["cet_per_joint_degrees"]) == 15 for r in rows))
        self.assertFalse(any("fid" == key or "psklj" == key for r in rows for key in r))
        with np.load(args.output_dir / rows[0]["cet_arrays"], allow_pickle=False) as arrays:
            self.assertEqual(arrays["aligned_abs_error_degrees"].shape[1], 15)
        with np.load(args.output_dir / "metrics/fid/arrays.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["gt_features"].shape, (2, 64))
        self.assertTrue((args.output_dir / "motions.csv").is_file())
        self.assertTrue((args.output_dir / "summary.csv").is_file())

    def test_g_cet_needs_no_refined_records_or_official_workers(self):
        args = self.args("--stage", "G", "--metrics", "cet", "--sample-id", "2")
        args.refined_dir = None
        bases = {side: np.tile(np.eye(3), (16, 1, 1)) for side in ("rh", "lh")}
        with patch.object(evaluate.subprocess, "run") as dispatch, \
             patch.object(cet, "load_anatomical_bases", return_value=(bases, {})):
            self.assertEqual(evaluate.run(args), 0)
        dispatch.assert_not_called()
        self.assertEqual(json.loads((args.output_dir / "motions.json").read_text())[0]["sample_id"], 2)
        # The snapshot can be reused with the same original cache and explicit
        # selection, even while the input folder also holds unselected files.
        repeated = self.args("--stage", "G", "--metrics", "cet", "--sample-id", "2")
        repeated.pairing_manifest = args.output_dir / "inputs/manifest.json"
        repeated.output_dir = self.root / "repeated_results"
        repeated.refined_dir = None
        _, report = evaluate.inventory(repeated)
        self.assertEqual(report["samples"][0]["sample_id"], 2)
        repeated.test_cache = args.output_dir / "inputs/cache.pkl"
        _, report = evaluate.inventory(repeated)
        self.assertEqual((report["samples"][0]["sample_id"], report["samples"][0]["source_cache_index"]), (2, 0))

    def test_failure_reports_are_saved_and_dry_run_launches_nothing(self):
        args = self.args("--metrics", "cr", "--dry-run")
        with patch.object(evaluate.subprocess, "run") as dispatch:
            self.assertEqual(evaluate.run(args), 0)
        dispatch.assert_not_called()
        self.assertFalse(args.output_dir.exists())
        args.dry_run = False
        with patch.object(evaluate.subprocess, "run", return_value=argparse.Namespace(returncode=7)):
            self.assertEqual(evaluate.run(args), 2)
        summary = json.loads((args.output_dir / "summary.json").read_text())
        self.assertEqual(summary["metrics"]["cr"]["status"], "failed")
        self.assertIn("exit 7", summary["metrics"]["cr"]["error"])

    def test_official_silent_skips_cannot_be_reported_as_success(self):
        args = self.args("--metrics", "cr")
        def skip_motion(command, **kwargs):
            completed = self.fake_official(command, **kwargs)
            path = args.output_dir / "metrics/cr/results.json"
            result = json.loads(path.read_text())
            result["motions"].pop()
            path.write_text(json.dumps(result))
            return completed
        with patch.object(evaluate.subprocess, "run", side_effect=skip_motion):
            self.assertEqual(evaluate.run(args), 2)
        summary = json.loads((args.output_dir / "summary.json").read_text())
        self.assertIn("silently skipped", summary["metrics"]["cr"]["error"])


if __name__ == "__main__":
    unittest.main()
