"""CPU fixtures/fake Slurm; no real scheduler, model download or full inference."""
import argparse
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "Scripts")]
from oakink2_tamf.dataset.cache_util import FIELDS, OBJECTS, cache_spec, read_cache, write_cache
from oakink2_tamf.util import inference as inf
from oakink2_tamf.launch.sample_selected import run_samples
from script.slurm import cet_inference_jobs as jobs
from workflow_fixtures import tiny_cache


def example_cache():
    result = {field: [] for field in (*FIELDS, OBJECTS)}
    groups = (("seq_a", "hold:00", "rh", 12), ("seq_a", "hold:01", "rh", 12),
              ("seq_b", "scoop:00", "lh", 4), ("seq_b", "hold:00", "rh", 1))
    original = tiny_cache(["seq_a"])
    for group, (seq, primitive, side, count) in enumerate(groups):
        for offset in range(count):
            for field in FIELDS:
                value = copy.deepcopy(original[field][0])
                if field == FIELDS[0]:
                    value = (seq, primitive, side)
                elif field == FIELDS[5]:
                    value = side
                elif field == FIELDS[-1]:
                    value = list(range(group * 500 + offset, group * 500 + offset + 20 * count, count))
                result[field].append(value)
    result[OBJECTS] = original[OBJECTS]
    return result


def generated(entry):
    array = np.zeros((160, 99), np.float32)
    array[:, 0] = entry["sample_id"] / 1000
    array[:, 3:] = np.tile(np.eye(3, dtype=np.float32)[:2].reshape(6), 16)
    return array


class InferenceFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = example_cache()
        self.cache_path = self.root / "test.pkl"
        self.metadata = write_cache(self.cache_path, self.cache, cache_spec(["seq_a", "seq_b"], "test"))
        self.split = self.root / "test.txt"
        self.split.write_text("seq_a\nseq_b\n")
        self.output = self.root / "output"
        self.output.mkdir()
        self.state = dict(experiment="fixture", cache=str(self.cache_path), cache_sha256=self.metadata["sha256"],
                          split="test", split_file=str(self.split), output_dir=str(self.output),
                          selection_options=dict(strides_per_motion=2, max_motions=None, seed=7),
                          reuse_selection=None, checkpoint="not-loaded.pt", checkpoint_sha256="fake-weights")

    def selection(self):
        return inf.prepare_selection(self.state)


class SelectionTests(InferenceFixtures):
    def test_actual_occurrence_keys_and_spread_two_offsets(self):
        entries, schedule = inf.choose_samples(self.cache)
        self.assertEqual(len({tuple(e["info"]) for e in entries}), 4)
        self.assertEqual(len(entries), 7)
        self.assertEqual([e["temporal_offset"] for e in entries if e["info"] == ["seq_a", "hold:00", "rh"]], [0, 11])
        for key in {tuple(e["info"]) for e in entries}:
            self.assertLessEqual(sum(tuple(e["info"]) == key for e in entries), 2)
        lookup = {e["sample_id"]: tuple(e["info"]) for e in entries}
        self.assertEqual(len({lookup[i] for i in schedule[:4]}), 4)

    def test_all_reproduces_original_indices_and_order(self):
        entries, schedule = inf.choose_samples(self.cache, "all")
        self.assertEqual(schedule, list(range(29)))
        self.assertEqual([e["sample_id"] for e in entries], list(range(29)))

    def test_deterministic_action_balanced_smoke_selection(self):
        first = inf.choose_samples(self.cache, 2, 2, 23)
        self.assertEqual(first, inf.choose_samples(self.cache, 2, 2, 23))
        self.assertEqual({e["action"] for e in first[0]}, {"hold", "scoop"})
        self.assertEqual(len({tuple(e["info"]) for e in first[0]}), 2)
        self.assertLessEqual(len(first[0]), 4)
        self.assertEqual(len({e["seed"] for e in first[0]}), len(first[0]))
        self.assertEqual(len(inf.choose_samples(self.cache, 2, 999)[0]), 7)

    def test_ambiguous_independent_intervals_do_not_become_stride_variants(self):
        self.cache[FIELDS[-1]][1] = [f + 5000 for f in self.cache[FIELDS[-1]][1]]
        with self.assertRaisesRegex(ValueError, "ambiguous temporal"):
            inf.choose_samples(self.cache)

    def test_pinned_subset_preserves_all_GT_fields_and_original_mapping(self):
        selection = self.selection()
        inf.verify_selection(self.state, selection)
        cache, _ = read_cache(selection["subset_cache"], require_manifest=True, split_name="test")
        self.assertEqual(len(cache[FIELDS[0]]), 7)
        for position, entry in enumerate(selection["samples"]):
            self.assertEqual(entry["subset_cache_index"], position)
            self.assertEqual(entry["filename"], f"{entry['original_cache_index']:06d}.npy")
            for field in FIELDS:
                old, new = self.cache[field][entry["original_cache_index"]], cache[field][position]
                if isinstance(old, np.ndarray):
                    np.testing.assert_array_equal(old, new)
                elif isinstance(old, dict):
                    for key in old:
                        np.testing.assert_array_equal(old[key], new[key])
                else:
                    self.assertEqual(old, new)

    def test_selection_reusable_across_models_and_cannot_drift(self):
        selection = self.selection()
        second = copy.deepcopy(self.state)
        second["output_dir"] = str(self.root / "comparison")
        Path(second["output_dir"]).mkdir()
        second.update(checkpoint="another-model.pt", checkpoint_sha256="other-weights",
                      reuse_selection=str(self.output / "selection/selection.json"),
                      reuse_selection_sha256=inf.sha256_file(self.output / "selection/selection.json"))
        copied = inf.prepare_selection(second)
        self.assertEqual(copied["selection_id"], selection["selection_id"])
        self.assertEqual(copied["samples"], selection["samples"])
        copied["samples"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            inf.verify_selection(second, copied)

    def test_test_never_falls_back_to_train_or_missing_preprocessing(self):
        with self.assertRaisesRegex(ValueError, "Requested train"):
            inf.inspect_source(self.cache_path, "train", self.split)
        with self.assertRaisesRegex(FileNotFoundError, "no train-cache fallback"):
            inf.inspect_source(self.root / "missing.pkl", "test", self.split)


class ResumeTests(InferenceFixtures):
    def test_resume_skips_valid_commits_and_recovers_unlinked_commit(self):
        selection = self.selection()
        stop = jobs.StopRequest()
        calls = []
        def predict(entry):
            calls.append(entry["sample_id"])
            stop.requested = True
            return generated(entry)
        first = run_samples(self.state, selection, self.root, stop, 600, predict)
        self.assertEqual(first["advanced"], 1)
        complete = inf.read_json(self.output / "manifest.json")["samples"][0]
        (self.output / "motions" / complete["filename"]).unlink()  # publication interrupted after directory commit
        def remaining(entry):
            calls.append(entry["sample_id"])
            return generated(entry)
        second = run_samples(self.state, selection, self.root, jobs.StopRequest(), 600, remaining)
        self.assertEqual(second["reason"], "completed")
        self.assertEqual(len(calls), 7)
        self.assertEqual(len(set(calls)), 7)
        self.assertTrue((self.output / "motions" / complete["filename"]).is_file())
        with patch("oakink2_tamf.launch.sample_selected.OriginalSampler") as model:
            third = run_samples(self.state, selection, self.root, jobs.StopRequest(), 600)
        model.assert_not_called()
        self.assertEqual(third["advanced"], 0)

    def test_interrupted_staging_and_failure_do_not_publish_bad_outputs(self):
        selection = self.selection()
        staging = self.output / ".sample-unfinished"
        staging.mkdir()
        (staging / "motion.npy").write_bytes(b"interrupted bytes")
        def fail(entry):
            raise RuntimeError("fake CUDA failure")
        failed = run_samples(self.state, selection, self.root, jobs.StopRequest(), 600, fail)
        self.assertEqual(failed["reason"], "failed")
        self.assertEqual(failed["progress"]["generated_samples"], 0)
        self.assertEqual(len(failed["progress"]["failed_samples"]), 1)
        resumed = run_samples(self.state, selection, self.root, jobs.StopRequest(), 600, generated)
        self.assertEqual(resumed["progress"]["generated_samples"], 7)
        self.assertTrue((staging / "motion.npy").is_file())

    def test_corrupt_or_other_checkpoint_outputs_are_never_adopted(self):
        selection = self.selection()
        entry = selection["samples"][0]
        inf.commit_motion(self.state, selection, entry, generated(entry), 1)
        other = dict(self.state, checkpoint_sha256="wrong-model")
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            inf.validated_receipt(other, selection, entry)
        motion = self.output / "completed" / f"{entry['sample_id']:06d}" / "motion.npy"
        with motion.open("ab") as stream:
            stream.write(b"corrupt")
        before = motion.read_bytes()
        with self.assertRaisesRegex(ValueError, "corrupt/untracked"):
            run_samples(self.state, selection, self.root, jobs.StopRequest(), 600, generated)
        self.assertEqual(motion.read_bytes(), before)

    def test_time_budget_and_term_leave_a_resumable_boundary(self):
        selection = self.selection()
        with patch("oakink2_tamf.launch.sample_selected.OriginalSampler") as model:
            result = run_samples(self.state, selection, self.root, jobs.StopRequest(), 10)
        model.assert_not_called()
        self.assertEqual(result["reason"], "session_limit")
        stop = jobs.StopRequest()
        stop.handle(signal.SIGTERM, None)
        result = run_samples(self.state, selection, self.root, stop, 600, generated)
        self.assertEqual(result["advanced"], 0)
        self.assertTrue(stop.hard)

    def test_variant_handoff_matches_stock_refinement_and_evaluator(self):
        import evaluate
        selection = self.selection()
        run_samples(self.state, selection, self.root, jobs.StopRequest(), 600, generated)
        destination = inf.export_handoff(self.state, selection, self.root / "handoff")
        for tree in sorted(destination.glob("variant_*")):
            manifest = inf.read_json(tree / "manifest.json")
            cache, _ = read_cache(tree / "cache.pkl", require_manifest=True, split_name="test")
            filenames = sorted(p.name for p in (tree / "motions").glob("*.npy"))
            self.assertEqual(filenames, [s["filename"] for s in manifest["samples"]])
            self.assertEqual(len(cache[FIELDS[0]]), len(filenames))
            self.assertEqual(len(set(cache[FIELDS[0]])), len(filenames))
            # Exercise the actual stock positional adaptor, not just its schema.
            from oakink2_tamf.dataset.interaction_segment import InteractionSegmentData
            from oakink2_tamf.dataset.pose_repr_sample import GeneratedPoseReprSampleAdaptor
            dataset = InteractionSegmentData(process_range_list=["seq_a", "seq_b"],
                data_prefix="not-needed-with-cache", cache_dict=cache, enable_obj_model=False)
            adaptor = GeneratedPoseReprSampleAdaptor(dataset, [str(tree / "motions")])
            for position, entry in enumerate(manifest["samples"]):
                sample = adaptor[position]
                self.assertEqual(tuple(sample["info"]), tuple(entry["info"]))
                self.assertEqual(sample["frame_id"], entry["frame_id"])
                self.assertEqual(sample["sample_info"][1], entry["sample_id"])
                np.testing.assert_array_equal(sample["sample_pose_repr"], generated(entry))
            for side in ("LEFT", "RIGHT"):
                model = self.root / f"mano/models/MANO_{side}.pkl"
                model.parent.mkdir(parents=True, exist_ok=True)
                model.touch()
            args = evaluate.parser().parse_args(["--motion-dir", str(tree / "motions"),
                "--test-cache", str(tree / "cache.pkl"), "--pairing-manifest", str(tree / "manifest.json"),
                "--stage", "G", "--metrics", "cet", "--output-dir", str(tree / "not_created"),
                "--test-split", str(self.split), "--mano-root", str(self.root / "mano")])
            # Evaluation output must be outside its read-only input tree.
            args.output_dir = self.root / (tree.name + "_evaluation")
            _, report = evaluate.inventory(args)
            self.assertEqual([s["sample_id"] for s in report["samples"]], [s["sample_id"] for s in manifest["samples"]])
        with self.assertRaises(FileExistsError):
            inf.export_handoff(self.state, selection, destination)


@unittest.skipUnless(os.name == "posix", "Slurm workflow locking requires Linux/WSL")
class ControllerTests(InferenceFixtures):
    def setUp(self):
        super().setUp()
        self.jobs_root = self.root / "jobs"
        self.path = self.jobs_root / "fixture" / "state.json"
        self.weights = self.root / "model.pt"
        self.weights.write_bytes(b"checkpoint fixture, fake sampler never loads it")
        conda = self.root / "conda/etc/profile.d/conda.sh"
        conda.parent.mkdir(parents=True)
        conda.write_text("conda() { :; }\n")
        (self.root / "data").mkdir()
        (self.root / "embedding").mkdir()
        for oid in self.cache[OBJECTS]:
            (self.root / "embedding" / f"{oid}.pt").touch()
        self.command = ["start", "--experiment", "fixture", "--jobs-root", str(self.jobs_root),
            "--cache", str(self.cache_path), "--split-file", str(self.split),
            "--checkpoint", str(self.weights), "--data-root", str(self.root / "data"),
            "--obj-embedding-dir", str(self.root / "embedding"), "--conda-base", str(self.root / "conda")]
        self.counter = 1000

    def scheduler(self, command, **kwargs):
        if command[0] == "sbatch":
            self.counter += 1
            return subprocess.CompletedProcess(command, 0, f"{self.counter}\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    def start(self):
        with patch.object(jobs.subprocess, "run", side_effect=self.scheduler), redirect_stdout(io.StringIO()):
            self.assertEqual(jobs.main(self.command), 0)

    def select(self):
        self.start()
        with patch.dict(os.environ, SLURM_JOB_ID="1001"), patch.object(jobs.subprocess, "run", side_effect=self.scheduler) as scheduler:
            self.assertEqual(jobs.worker(self.path, "select"), 0)
        return scheduler

    def test_start_really_submits_cpu_then_gpu_with_confirmed_resources(self):
        scheduler = self.select()
        command = scheduler.call_args.args[0]
        self.assertIn("--gres=shard:1", command)
        self.assertIn("--partition=gpu", command)
        self.assertIn("--mem=16G", command)
        self.assertIn("--cpus-per-task=2", command)
        self.assertIn("--dependency=afterok:1001,singleton", command)
        self.assertIn("--signal=B:USR1@120", command)
        self.assertTrue((self.path.parent / "config.json").is_file())
        state = inf.read_json(self.path)
        self.assertEqual([j["phase"] for j in state["jobs"]], ["select", "sample"])
        self.assertEqual(state["selection_options"]["strides_per_motion"], 2)
        self.assertEqual(state["split"], "test")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", scheduler.call_args.kwargs["env"])

    def test_small_smoke_request_is_limited_before_any_model_sampling(self):
        self.command += ["--max-motions", "1"]
        self.select()
        state = inf.read_json(self.path)
        selection = inf.read_json(Path(state["output_dir"]) / "selection/selection.json")
        self.assertEqual(selection["unique_motions"], 1)
        self.assertLessEqual(selection["total_samples"], 2)
        self.assertEqual(len(selection["schedule"]), selection["total_samples"])

    def test_cpu_submission_has_no_gpu_request_or_inherited_cuda_mask(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", EXPERIMENT_ID="unrelated_training"), \
             patch.object(jobs.subprocess, "run", side_effect=self.scheduler) as scheduler, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(jobs.main(self.command), 0)
        command = scheduler.call_args.args[0]
        self.assertFalse(any(arg.startswith("--gres=") for arg in command))
        self.assertNotIn("CUDA_VISIBLE_DEVICES", scheduler.call_args.kwargs["env"])
        self.assertEqual(inf.read_json(self.path)["experiment"], "fixture")

    def test_dry_run_has_no_scheduler_calls_or_writes(self):
        with patch.object(jobs.subprocess, "run") as scheduler, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(jobs.main([*self.command, "--dry-run", "--max-motions", "2"]), 0)
        scheduler.assert_not_called()
        self.assertFalse(self.jobs_root.exists())
        self.assertIn("CPU selection", output.getvalue())
        self.assertIn("--gres=shard:1", output.getvalue())
        self.assertIn('"max_motions": 2', output.getvalue())

    def test_graceful_allocation_boundary_submits_one_afterok_successor(self):
        self.select()
        actual = run_samples
        def sample(state, selection, directory, stop, max_seconds):
            def predict(entry):
                stop.requested = True
                return generated(entry)
            return actual(state, selection, directory, stop, max_seconds, predict)
        with patch.dict(os.environ, SLURM_JOB_ID="1002"), \
             patch("oakink2_tamf.launch.sample_selected.run_samples", side_effect=sample), \
             patch.object(jobs.subprocess, "run", side_effect=self.scheduler) as scheduler:
            self.assertEqual(jobs.worker(self.path, "sample"), 0)
        self.assertIn("--dependency=afterok:1002,singleton", scheduler.call_args.args[0])
        self.assertEqual(inf.read_json(self.path)["jobs"][-1]["job_id"], "1003")

    def test_completion_does_not_submit_a_successor_or_repeat_samples(self):
        self.select()
        def sample(state, selection, directory, stop, max_seconds):
            return run_samples(state, selection, directory, stop, max_seconds, generated)
        with patch.dict(os.environ, SLURM_JOB_ID="1002"), \
             patch("oakink2_tamf.launch.sample_selected.run_samples", side_effect=sample), \
             patch.object(jobs.subprocess, "run") as scheduler:
            self.assertEqual(jobs.worker(self.path, "sample"), 0)
        scheduler.assert_not_called()
        with patch.object(jobs.subprocess, "run", side_effect=self.scheduler) as scheduler, redirect_stdout(io.StringIO()):
            self.assertEqual(jobs.main(["resume", "--experiment", "fixture", "--jobs-root", str(self.jobs_root)]), 0)
        self.assertTrue(all(call.args[0][0] != "sbatch" for call in scheduler.call_args_list))

    def test_active_jobs_and_live_run_lock_block_competing_inference(self):
        self.select()
        active = subprocess.CompletedProcess([], 0, "1002|RUNNING|node\n", "")
        args = argparse.Namespace(dry_run=False, clear_submission_intent=False, max_allocations=None)
        with patch.object(jobs.subprocess, "run", return_value=active):
            with self.assertRaisesRegex(RuntimeError, "competing"):
                jobs.resume(args, self.path)
        before = self.path.read_bytes()
        with jobs.workflow_lock(self.path.parent / "run.lock"), patch.dict(os.environ, SLURM_JOB_ID="1002"):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                jobs.worker(self.path, "sample")
        self.assertEqual(self.path.read_bytes(), before)
        with jobs.workflow_lock(self.path.parent / "run.lock"), patch.object(jobs.subprocess, "run") as scheduler:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                jobs.resume(args, self.path)
        scheduler.assert_not_called()  # a handoff export also owns run.lock

    def test_missing_selected_embedding_stops_before_gpu_submission(self):
        self.start()
        (self.root / "embedding/obj_a.pt").unlink()
        with patch.dict(os.environ, SLURM_JOB_ID="1001"), patch.object(jobs.subprocess, "run") as scheduler:
            with self.assertRaisesRegex(FileNotFoundError, "selected object embedding missing"):
                jobs.worker(self.path, "select")
        scheduler.assert_not_called()
        self.assertEqual(inf.read_json(self.path)["phase"], "failed")

    def test_stop_signals_running_jobs_and_cancels_pending_jobs(self):
        self.select()
        queue = subprocess.CompletedProcess([], 0, "1002|RUNNING|node\n1003|PENDING|Dependency\n", "")
        def scheduler(command, **kwargs):
            return queue if command[0] == "squeue" else subprocess.CompletedProcess(command, 0)
        with patch.object(jobs.subprocess, "run", side_effect=scheduler) as mock:
            jobs.stop_jobs(argparse.Namespace(dry_run=False, force=False), self.path)
        commands = [call.args[0] for call in mock.call_args_list]
        self.assertIn(["scancel", "--signal=USR1", "--batch", "1002"], commands)
        self.assertIn(["scancel", "1003"], commands)
        self.assertTrue((self.path.parent / "STOP").is_file())

    def test_stop_marker_prevents_cpu_phase_from_submitting_gpu(self):
        self.start()
        (self.path.parent / "STOP").touch()
        with patch.dict(os.environ, SLURM_JOB_ID="1001"), patch.object(jobs.subprocess, "run") as scheduler:
            self.assertEqual(jobs.worker(self.path, "select"), 0)
        scheduler.assert_not_called()

    def test_status_uses_accounting_and_commits_after_an_interruption(self):
        self.select()
        state = inf.read_json(self.path)
        selection = inf.read_json(Path(state["output_dir"]) / "selection/selection.json")
        inf.commit_motion(state, selection, selection["samples"][0], generated(selection["samples"][0]), 2)
        def scheduler(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "1002|TIMEOUT|0:0\n" if command[0] == "sacct" else "", "")
        with patch.object(jobs.subprocess, "run", side_effect=scheduler), redirect_stdout(io.StringIO()):
            report = jobs.status(self.path)
        self.assertEqual(report["status"], "interrupted; resume required")
        self.assertEqual(report["generated_samples"], 1)
        self.assertEqual(report["remaining_samples"], 6)

    def test_status_pending_failed_and_historical_failure_are_distinct(self):
        self.start()
        def pending(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "1001|PENDING|Priority\n" if command[0] == "squeue" else "", "")
        with patch.object(jobs.subprocess, "run", side_effect=pending), redirect_stdout(io.StringIO()):
            self.assertEqual(jobs.status(self.path)["status"], "pending")
        state = inf.read_json(self.path)
        state.update(phase="failed", last_error="selection could not read its cache")
        inf.atomic_json(self.path, state)
        with patch.object(jobs.subprocess, "run", side_effect=self.scheduler), redirect_stdout(io.StringIO()):
            report = jobs.status(self.path)
        self.assertEqual(report["status"], "failed")
        self.assertIn("selection", report["last_error"])
        state.update(phase="session_limit", jobs=[dict(job_id="1000"), dict(job_id="1001")])
        inf.atomic_json(self.path, state)
        def accounting(command, **kwargs):
            return subprocess.CompletedProcess(command, 0,
                "1000|FAILED|1:0|120\n1001|COMPLETED|0:0|180\n" if command[0] == "sacct" else "", "")
        with patch.object(jobs.subprocess, "run", side_effect=accounting), redirect_stdout(io.StringIO()):
            self.assertEqual(jobs.status(self.path)["status"], "interrupted; resume required")

    def test_submission_ambiguity_and_checkpoint_drift_fail_closed(self):
        self.start()
        state = inf.read_json(self.path)
        with patch.object(jobs.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "sbatch")):
            with self.assertRaises(subprocess.CalledProcessError):
                jobs.submit(state, self.path, "sample")
        self.assertTrue(inf.read_json(self.path)["submission_pending"])
        with self.weights.open("ab") as stream:
            stream.write(b"different model")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            jobs.preflight(state)

    def test_sbatch_shell_syntax_and_single_process_exec(self):
        if not shutil.which("bash"):
            self.skipTest("Bash unavailable")
        source = ROOT / "script/slurm/cet_inference.sbatch"
        subprocess.run(["bash", "-n", str(source)], check=True)
        self.assertIn('exec python -u "$controller"', source.read_text())


if __name__ == "__main__":
    unittest.main()
