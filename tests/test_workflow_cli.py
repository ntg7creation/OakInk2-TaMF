import os
import subprocess
import shutil
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oakink2_tamf.dataset.cache_util import cache_spec, write_cache, sha256_file, read_cache
from workflow_fixtures import tiny_cache


class WorkflowCLITests(unittest.TestCase):
    def command(self, module, *args):
        result = subprocess.run([sys.executable, "-B", "-m", module, *args], cwd=ROOT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:])
        return result.stdout

    def test_help_for_all_entrypoints(self):
        self.assertIn("--cache.resume", self.command("script.save_cache_dict", "--help"))
        self.assertIn("--train.resume_checkpoint", self.command("oakink2_tamf.launch.train", "--help"))
        self.assertIn("--prepare-only", self.command("script.debug.smoke_cached_training", "--help"))

    def test_preprocessing_cli_verifies_and_reuses_without_raw_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "train.pkl"
            write_cache(cache, tiny_cache(), cache_spec(["seq_a", "seq_b"], "train"))
            before = sha256_file(cache)
            args = ("--data.data_prefix", str(Path(temp) / "no-raw-data"),
                    "--data.process_range", "seq_a:seq_b", "--data.split_name", "train",
                    "--cache.output_filepath", str(cache))
            self.assertIn("Cache validated", self.command("script.save_cache_dict", *args, "--cache.verify_only"))
            self.assertIn("Reusing compatible", self.command("script.save_cache_dict", *args, "--commit"))
            self.assertEqual(before, sha256_file(cache))
            self.assertFalse(Path(str(cache) + ".work").exists())

    def test_smoke_preparation_is_bounded_preserves_source_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "train.pkl", Path(temp) / "smoke.pkl"
            write_cache(source, tiny_cache(), cache_spec(["seq_a", "seq_b"], "train"))
            before = sha256_file(source)
            args = ("--prepare-only", "--source-cache", str(source), "--cache", str(output),
                    "--sample-indices", "1")
            self.command("script.debug.smoke_cached_training", *args)
            cache, metadata = read_cache(output, require_manifest=True, split_name="train")
            self.assertEqual(metadata["purpose"], "bounded_smoke_subset")
            self.assertEqual(cache["interaction_segment_hand_side_list"], ["lh"])
            self.assertEqual(before, sha256_file(source))
            result = subprocess.run([sys.executable, "-B", "-m", "script.debug.smoke_cached_training", *args],
                                    cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stdout)

    @unittest.skipUnless(shutil.which("bash") and hasattr(signal, "SIGUSR1"), "requires Bash/POSIX signals")
    def test_slurm_shell_syntax_and_signal_forwarding_preserve_child_exit_status(self):
        for name in ("preprocess.sbatch", "train_cached.sbatch"):
            subprocess.run(["bash", "-n", str(ROOT / "script/slurm" / name)], check=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / "fake_python"
            fake.write_text("#!/usr/bin/env python3\nimport signal,os,time\n"
                            "signal.signal(signal.SIGUSR1, lambda *args: exit(7))\n"
                            "open(os.environ['TEST_PID_FILE'], 'w').write(str(os.getpid()))\n"
                            "while True: time.sleep(0.01)\n")
            fake.chmod(0o755)
            marker = root / "child_pid"
            env = dict(os.environ, REPO_ROOT=str(ROOT), DATA_ROOT=str(root), CACHE_ROOT=str(root),
                       EXPERIMENT_ID="shell-test", EXPERIMENT_CONFIG="unused", TAMF_PYTHON=str(fake),
                       TEST_PID_FILE=str(marker))
            env.pop("CONDA_BASE", None)
            process = subprocess.Popen(["bash", str(ROOT / "script/slurm/train_cached.sbatch")],
                                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists(), "fake child failed to start")
                process.send_signal(signal.SIGUSR1)
                stdout, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 7, stdout)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if marker.exists():
                    try:
                        os.kill(int(marker.read_text()), signal.SIGTERM)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
