"""Small CPU checkpoints, fake Slurm/Conda/trainer; never submits real jobs."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oakink2_tamf.util.training_state import capture_rng, save_checkpoint


@unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("flock"),
                     "requires Linux/WSL Bash and flock")
class SimpleSlurmChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo with spaces"
        self.repo.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls = self.root / "calls.jsonl"
        self.trainer_call = self.root / "trainer.json"
        self.counter = self.root / "counter"
        for name in ("train.sbatch", "submit_train_chain.sh"):
            shutil.copyfile(ROOT / name, self.repo / name)
        cache = self.repo / "common/preprocessed_cache_v1/train.pkl"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"cache fixture; the fake trainer never opens it")
        conda = self.root / "conda/etc/profile.d/conda.sh"
        conda.parent.mkdir(parents=True)
        conda.write_text("conda() { :; }\n")
        self.env = dict(os.environ, REPO_ROOT=str(self.repo), EXPERIMENT_ID="tamf_cet_001_retry1",
                        TARGET_EPOCHS="30", CONDA_BASE=str(self.root / "conda"),
                        PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                        SLURM_JOB_ID="900", PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        CALLS=str(self.calls), COUNTER=str(self.counter), TRAINER_CALL=str(self.trainer_call))
        for variable in ("DATA_ROOT", "TRAIN_CACHE", "FAKE_QUEUE_IDS", "FAKE_FAIL_AT", "WAIT_FOR_SIGNAL"):
            self.env.pop(variable, None)
        self.executable("squeue", "import os\nprint(os.environ.get('FAKE_QUEUE_IDS', ''))\n")
        self.executable("sbatch", """import json, os, pathlib, sys
p = pathlib.Path(os.environ['COUNTER'])
n = int(p.read_text()) + 1 if p.exists() else 1
p.write_text(str(n))
with open(os.environ['CALLS'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if n == int(os.environ.get('FAKE_FAIL_AT', '-1')):
    sys.exit(1)
print(1000 + n)
""")
        fake_python = self.executable("fake_python", """import json, os, pathlib, signal, sys, time
if sys.argv[1] == '-':
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
if os.environ.get('WAIT_FOR_SIGNAL'):
    signal.signal(signal.SIGUSR1, lambda *args: sys.exit(7))
    signal.alarm(20)
pathlib.Path(os.environ['TRAINER_CALL']).write_text(json.dumps(sys.argv[1:]))
if os.environ.get('WAIT_FOR_SIGNAL'):
    while True:
        time.sleep(.01)
sys.exit(int(os.environ.get('TRAINER_EXIT', '0')))
""")
        self.env["TAMF_PYTHON"] = str(fake_python)

    def executable(self, name, source):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)
        return path

    def command(self, script, *args, env=None):
        return subprocess.run(["bash", str(self.repo / script), *args], env=env or self.env,
                              cwd=self.repo, capture_output=True, text=True, timeout=30)

    def checkpoint(self, epoch):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1])
        path = self.repo / f"common/train/tamf_cet_001_retry1/save/state_epoch_{epoch:04d}_test.pt"
        save_checkpoint(path, model, optimizer, scheduler, completed_epoch=epoch,
                        global_updates=epoch + 1, signature=dict(world_size=1),
                        rank_states=[dict(rng=capture_rng(), loader_rng=torch.Generator().get_state())])
        return path

    def submitted(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_shell_syntax_and_confirmed_resources(self):
        for name in ("train.sbatch", "submit_train_chain.sh"):
            subprocess.run(["bash", "-n", str(ROOT / name)], check=True)
        source = (ROOT / "train.sbatch").read_text()
        for option in ("--partition=gpu", "--gres=shard:1", "--mem=32G", "--time=04:00:00", "--signal=B:USR1@600"):
            self.assertIn(option, source)

    def test_chain_automatically_follows_existing_job_and_uses_afterok(self):
        env = dict(self.env, FAKE_QUEUE_IDS="  1227\n  1229\n")
        result = self.command("submit_train_chain.sh", "3", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.submitted()
        self.assertEqual(len(calls), 3)
        for call, previous in zip(calls, (1229, 1001, 1002)):
            self.assertIn(f"--dependency=afterok:{previous},singleton", call)
            self.assertIn("--kill-on-invalid-dep=yes", call)
            self.assertIn("--export=ALL", call)
            self.assertIn(str(self.repo / "train.sbatch"), call)
        self.assertIn("job 1003", result.stdout)

    def test_first_job_without_existing_allocation_and_explicit_predecessor(self):
        result = self.command("submit_train_chain.sh", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dependency=singleton", self.submitted()[0])
        self.calls.unlink()
        result = self.command("submit_train_chain.sh", "1", "--after", "1230")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dependency=afterok:1230,singleton", self.submitted()[0])

    def test_dry_run_and_invalid_arguments_submit_nothing(self):
        result = self.command("submit_train_chain.sh", "3", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("DRY RUN: sbatch"), 3)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.repo / "common/cet_jobs").exists())
        for arguments in (("0",), ("-1",), ("two",), ("1", "--after", "bad")):
            result = self.command("submit_train_chain.sh", *arguments)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists())

    def test_scheduler_rejection_stops_further_submissions(self):
        result = self.command("submit_train_chain.sh", "4", env=dict(self.env, FAKE_FAIL_AT="2"))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.submitted()), 2)
        self.assertIn("job 1001", result.stdout)
        self.assertNotIn("Session 3/", result.stdout)

    def test_runtime_selects_latest_complete_state_and_keeps_training_options(self):
        self.checkpoint(0)
        latest = self.checkpoint(1)
        # A newer interrupted publication must not be selected.
        (latest.parent / "state_epoch_0002_incomplete.pt").write_bytes(b"interrupted")
        result = self.command("train.sbatch")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("next_epoch=2", result.stdout)
        arguments = json.loads(self.trainer_call.read_text())
        for flag, expected in (("--train.resume_checkpoint", str(latest)), ("--runtime.num_worker", "0"),
                               ("--runtime.max_seconds", "13800"), ("--train.num_epoch", "30"),
                               ("--train.record_freq", "1"), ("--exp_id", "tamf_cet_001_retry1")):
            self.assertEqual(arguments[arguments.index(flag) + 1], expected)
        self.assertIn("config/shenkar_generator_full.yml", arguments)
        self.assertIn("config/cet_experiment.yml", arguments)

    def test_missing_or_corrupt_checkpoints_fail_without_starting_training(self):
        result = self.command("train.sbatch")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.trainer_call.exists())
        path = self.checkpoint(0)
        with path.open("ab") as stream:
            stream.write(b"corrupt")
        result = self.command("train.sbatch")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.trainer_call.exists())

    def test_target_reached_skips_training_and_trainer_failure_is_preserved(self):
        self.checkpoint(0)
        result = self.command("train.sbatch", env=dict(self.env, TARGET_EPOCHS="1"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("target already reached", result.stdout)
        self.assertFalse(self.trainer_call.exists())
        result = self.command("train.sbatch", env=dict(self.env, TRAINER_EXIT="7"))
        self.assertEqual(result.returncode, 7)

    def test_exec_forwards_usr1_and_lock_blocks_an_overlapping_session(self):
        self.checkpoint(0)
        process = subprocess.Popen(["bash", str(self.repo / "train.sbatch")],
                                   cwd=self.repo, env=dict(self.env, WAIT_FOR_SIGNAL="1"),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + 15
            while not self.trainer_call.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(.02)
            self.assertTrue(self.trainer_call.exists(), "fake trainer did not start")
            duplicate = self.command("train.sbatch")
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("Another continuation", duplicate.stderr)
            process.send_signal(signal.SIGUSR1)
            output, _ = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 7, output)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
