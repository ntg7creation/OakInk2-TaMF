"""Controller tests use fake Slurm/trainers, small real CPU checkpoints only."""
import argparse
import copy
import io
import json
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from script.slurm import cet_jobs as jobs
from script.debug.smoke_train_cet import cet_generator_gradient_norm
from oakink2_tamf.util.training_state import capture_rng, save_checkpoint


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "state.json"
        self.state = dict(experiment="tamf_cet_test", experiment_dir=str(self.root / "experiment"),
                          target=3, max_allocations=3, allocations_started=0, cache="existing.pkl",
                          data_root="existing_data", conda_base="existing_conda", conda_env="oakink2-tamf",
                          partition="gpu", gres="shard:1", mem="16G", code_identity=jobs.code_identity(),
                          jobs=[dict(job_id="101")], allocations=[], running=None,
                          training_process_seconds=0, submission_pending=False)
        jobs.atomic_json(self.path, self.state)

    def progress(self, count):
        return dict(next_epoch=count, global_updates=count * 2,
                    checkpoint=None if count == 0 else str(self.root / f"epoch_{count - 1}.pt"))

    def run_worker(self, before, after=None, result=0):
        values = [self.progress(before)] if after is None else [self.progress(before), self.progress(after)]
        with patch.dict(os.environ, SLURM_JOB_ID="101"), patch.object(jobs, "preflight"), \
                patch.object(jobs, "checkpoint_progress", side_effect=values), \
                patch.object(jobs, "run_training", return_value=result) as training, \
                patch.object(jobs.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "102\n", "")) as scheduler:
            jobs.worker(self.path)
        return training, scheduler

    def test_fresh_epoch_zero_then_afterok_resume_next_epoch(self):
        training, scheduler = self.run_worker(0, 2)
        first = training.call_args.args[0]
        self.assertNotIn("--train.resume_checkpoint", first)
        self.assertIn("15600", first)
        self.assertIn("config/cet_experiment.yml", first)
        command = scheduler.call_args.args[0]
        self.assertIn("--dependency=afterok:101", command)
        self.assertIn("--time=04:30:00", command)
        state = json.loads(self.path.read_text())
        self.assertEqual(state["progress"]["next_epoch"], 2)
        self.assertEqual(state["jobs"][-1]["job_id"], "102")
        state["jobs"] = [dict(job_id="101")]
        jobs.atomic_json(self.path, state)
        training, scheduler = self.run_worker(2, 3)
        self.assertEqual(training.call_args.args[0][-2:], ["--train.resume_checkpoint", self.progress(2)["checkpoint"]])
        scheduler.assert_not_called()

    def test_already_at_target_never_starts_trainer_or_submits(self):
        training, scheduler = self.run_worker(3)
        training.assert_not_called()
        scheduler.assert_not_called()

    def test_crash_and_zero_progress_stop_chain(self):
        for result, after, pattern in ((7, None, "exited 7"), (0, 0, "without a new complete epoch")):
            jobs.atomic_json(self.path, self.state)
            with self.assertRaisesRegex(RuntimeError, pattern):
                self.run_worker(0, after, result)
            self.assertEqual(len(json.loads(self.path.read_text())["jobs"]), 1)

    def test_stop_marker_and_allocation_cap(self):
        (self.root / "STOP").touch()
        training, scheduler = self.run_worker(0)
        training.assert_not_called()
        scheduler.assert_not_called()
        (self.root / "STOP").unlink()
        state = dict(self.state, max_allocations=1)
        jobs.atomic_json(self.path, state)
        training, scheduler = self.run_worker(0, 1)
        training.assert_called_once()
        scheduler.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "workflow locks run on Linux/WSL")
    def test_live_controller_lock_prevents_duplicate_worker(self):
        with jobs.workflow_lock(self.root / "controller.lock"):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                self.run_worker(0, 1)

    def test_unrecorded_job_and_ambiguous_submission_fail_closed(self):
        state = dict(self.state, submission_pending=True)
        jobs.atomic_json(self.path, state)
        with self.assertRaisesRegex(RuntimeError, "not a recorded"):
            self.run_worker(0, 1)
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            jobs.submit(state, self.path)

    def test_sbatch_failure_leaves_intent_and_no_successor(self):
        state = copy.deepcopy(self.state)
        with patch.object(jobs.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "sbatch")):
            with self.assertRaises(subprocess.CalledProcessError):
                jobs.submit(state, self.path, "101")
        stored = json.loads(self.path.read_text())
        self.assertTrue(stored["submission_pending"])
        self.assertEqual(len(stored["jobs"]), 1)

    def test_real_complete_checkpoint_and_corrupt_fallback(self):
        directory = self.root / "real"
        self.assertEqual(jobs.checkpoint_progress(directory)["next_epoch"], 0)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1])
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        path = directory / "save/state_epoch_0000.pt"
        save_checkpoint(path, model, optimizer, scheduler, completed_epoch=0, global_updates=1,
                        signature=dict(world_size=1),
                        rank_states=[dict(rng=capture_rng(), loader_rng=torch.Generator().get_state())])
        (path.parent / "state_epoch_0001.pt").write_bytes(b"interrupted publication")
        self.assertEqual(jobs.checkpoint_progress(directory)["next_epoch"], 1)
        with path.open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "no verified checkpoint"):
            jobs.checkpoint_progress(directory)

    def test_dry_run_has_no_writes_and_no_scheduler_call(self):
        with patch.object(jobs, "ROOT", self.root), patch.object(jobs, "code_identity", return_value={}), \
                patch.object(jobs.subprocess, "run") as run, redirect_stdout(io.StringIO()) as output:
            jobs.main(["start", "--dry-run", "--data-root", "data", "--cache", "cache.pkl", "--epochs-per-allocation", "1"])
        run.assert_not_called()
        self.assertFalse((self.root / "common").exists())
        self.assertIn("DRY RUN", output.getvalue())
        self.assertIn("--time=04:30:00", output.getvalue())
        self.assertIn("--train.num_epoch 1", output.getvalue())

    def test_actual_start_refuses_duplicate_active_job(self):
        with patch.object(jobs, "ROOT", self.root), patch.object(jobs, "code_identity", return_value={}), \
                patch.object(jobs, "active_jobs", return_value=[["201", "RUNNING", ""]]), \
                patch.object(jobs, "submit") as submit:
            with self.assertRaisesRegex(RuntimeError, "no duplicate submission"):
                jobs.main(["start", "--submit"])
        submit.assert_not_called()

    def test_stop_sends_usr1_only_to_running_and_cancels_pending(self):
        with patch.object(jobs, "active_jobs", return_value=[["101", "RUNNING", ""], ["102", "PENDING", "afterok:101"]]), \
                patch.object(jobs.subprocess, "run") as run:
            jobs.main(["stop", "--state", str(self.path)])
        self.assertTrue((self.root / "STOP").exists())
        self.assertEqual(run.call_args_list[0].args[0], ["scancel", "--batch", "--signal=USR1", "101"])
        self.assertEqual(run.call_args_list[1].args[0], ["scancel", "102"])

    def test_diagnostic_gate_and_scale_checks(self):
        cfg = yaml.safe_load((ROOT / "config/shenkar_generator_full.yml").read_text())
        report = dict(arms=dict(cet=dict(cet_weight=.05, weights_updated=True, cet_generator_grad_norm=.2,
                                       median_step_seconds=1, peak_cuda_bytes=100,
                                       steps=[dict(cet=.8, cet_weighted=.04, total_loss=2)])),
                      warmup_steps_per_arm=2, measured_steps_per_arm=6, samples=[{}] * 4,
                      code_identity=jobs.code_identity(), model_config=cfg["model"],
                      loss_config=dict(cfg["train"]["loss"], cet_dtw_rho=.1))
        path = self.root / "diagnostic.json"
        jobs.atomic_json(path, report)
        self.assertEqual(jobs.validate_diagnostic(path), report)
        for key, value in (("cet_weight", 0), ("cet_generator_grad_norm", 0), ("cet_generator_grad_norm", None)):
            changed = copy.deepcopy(report)
            changed["arms"]["cet"][key] = value
            jobs.atomic_json(path, changed)
            with self.assertRaises(ValueError):
                jobs.validate_diagnostic(path)

    def test_candidate_overlay_and_original_default_are_independent(self):
        from config_reg import ConfigRegistry
        from oakink2_tamf.launch.param.loss import reg_loss_param
        for arguments, expected in (([], 0), (["--cfg", str(ROOT / "config/cet_experiment.yml")], .05)):
            registry = ConfigRegistry(prog="cet-experiment-test")
            reg_loss_param(registry, "train.loss", str(ROOT))
            parser = argparse.ArgumentParser()
            registry.hook(parser)
            registry.parse(parser, arg_src=arguments)
            self.assertEqual(registry.select("train.loss.cet_weight"), expected)
            self.assertEqual(registry.select("train.loss.cet_dtw_rho"), .1)
            self.assertFalse(registry.select("train.loss.cet_profile"))

    def test_isolated_cet_gradient_does_not_populate_parameter_grads(self):
        model = torch.nn.Linear(2, 1)
        cet = (model(torch.ones(1, 2)) - 10).abs().mean()
        norm = cet_generator_gradient_norm(cet, model)
        self.assertGreater(norm, 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        cet.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaisesRegex(RuntimeError, "detached"):
            cet_generator_gradient_norm(cet.detach(), model)

    def test_target_increase_preserves_resume_settings(self):
        path = self.root / "common/cet_jobs/tamf_cet_001/state.json"
        path.parent.mkdir(parents=True)
        jobs.atomic_json(path, dict(self.state, target=30, max_allocations=30))
        with patch.object(jobs, "ROOT", self.root), redirect_stdout(io.StringIO()) as output:
            jobs.main(["start", "--target", "100", "--dry-run"])
        self.assertIn("--train.num_epoch 100", output.getvalue())
        self.assertIn("existing.pkl", output.getvalue())
        self.assertEqual(json.loads(path.read_text())["target"], 30)  # dry run never persists
        with patch.object(jobs, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "cannot decrease"):
                jobs.main(["start", "--target", "2", "--dry-run"])

    def test_canary_caps_each_allocation_without_changing_global_target(self):
        state = dict(self.state, target=2, epochs_per_allocation=1)
        first = jobs.training_command(state, self.progress(0))
        second = jobs.training_command(state, self.progress(1))
        self.assertEqual(first[first.index("--train.num_epoch") + 1], "1")
        self.assertEqual(second[second.index("--train.num_epoch") + 1], "2")
        self.assertEqual(state["target"], 2)

    def test_monitor_reads_latest_loss_cet_weighted_and_lr(self):
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(str(self.root / "summary")) as writer:
            for step in range(4):
                for tag, value in (("total_loss", 2 + step), ("extra_loss/cet", .8),
                                   ("extra_loss/cet_weighted", .04), ("lr", .0001)):
                    writer.add_scalar(tag, value, step)
        scalars = jobs.latest_scalars(self.root)
        self.assertEqual(scalars["total_loss"]["step"], 3)
        self.assertEqual(scalars["total_loss"]["value"], 5)
        self.assertAlmostEqual(scalars["extra_loss/cet_weighted"]["value"], .04, places=6)
        self.assertAlmostEqual(scalars["lr"]["value"], .0001, places=7)

    @unittest.skipUnless(shutil.which("bash") and hasattr(signal, "SIGUSR1"), "requires Bash/POSIX signals")
    def test_batch_syntax_and_forwarded_stop_keep_child_failure_status(self):
        subprocess.run(["bash", "-n", str(ROOT / "script/slurm/cet_train.sbatch")], check=True)
        marker = self.root / "ready"
        fake = self.root / "fake_trainer.py"
        fake.write_text("import signal,time,pathlib\n"
                        "signal.alarm(15)\n"
                        "signal.signal(signal.SIGUSR1, lambda *args: exit(7))\n"
                        f"pathlib.Path({str(marker)!r}).touch()\n"
                        "while True: time.sleep(.01)\n")
        def send():
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            if marker.exists():
                os.kill(os.getpid(), signal.SIGUSR1)
        thread = threading.Thread(target=send, daemon=True)
        thread.start()
        self.assertEqual(jobs.run_training([sys.executable, str(fake)]), 7)
        thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
