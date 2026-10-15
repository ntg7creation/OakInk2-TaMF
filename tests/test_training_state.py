import copy
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oakink2_tamf.dataset.cache_util import manifest_path
from oakink2_tamf.util.training_state import (
    capture_rng, restore_rng, save_checkpoint, restore_checkpoint, resolve_checkpoint,
    prepare_experiment, select_cuda_devices, stop_at_boundary, make_signature, save_distributed_checkpoint,
    validate_resume_destination,
)


def make_training():
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1, 2], gamma=0.5)
    return model, optimizer, scheduler, torch.Generator().manual_seed(23)


def epoch(model, optimizer, scheduler, generator, epoch_id):
    generator.manual_seed(23 + 1000003 * epoch_id)
    # Match DataLoader's separate generator consumption, not model RNG.
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    losses = []
    for _ in range(2):
        x = torch.ones(2, 3) + torch.randn(2, 3) + float(np.random.rand()) + random.random()
        loss = model(x).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()
    return losses


def distributed_save_worker(rank, directory, expect_failure):
    root = Path(directory)
    dist.init_process_group("gloo", init_method="file://" + str(root / "rendezvous"), rank=rank, world_size=2)
    try:
        torch.manual_seed(23 + rank)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1])
        model(torch.ones(1, 2)).square().mean().backward()
        optimizer.step()
        scheduler.step()
        generator = torch.Generator().manual_seed(rank)
        try:
            save_distributed_checkpoint(root / "state.pt", model.module, optimizer, scheduler,
                                        completed_epoch=0, global_updates=1, signature=dict(world_size=2),
                                        rank=rank, world_size=2, device=torch.device("cpu"), generator=generator)
        except RuntimeError:
            if not expect_failure:
                raise
        else:
            if expect_failure:
                raise AssertionError("every rank must receive publication failure")
        (root / f"rank_{rank}_finished").write_text("ok")
    finally:
        dist.destroy_process_group()


class TrainingStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.signature = dict(version=1, world_size=1, data="fixed", cet_weight=0.1)

    def save(self, path, items, completed_epoch=0):
        model, optimizer, scheduler, generator = items
        save_checkpoint(path, model, optimizer, scheduler, completed_epoch=completed_epoch,
                        global_updates=(completed_epoch + 1) * 2, signature=self.signature,
                        rank_states=[dict(rng=capture_rng(), loader_rng=generator.get_state())])

    def test_epoch_resume_matches_uninterrupted_losses_weights_optimizer_scheduler_and_rng(self):
        items = make_training()
        epoch(*items, 0)
        path = self.root / "save/state_epoch_0000.pt"
        self.save(path, items)
        expected_losses = epoch(*items, 1)
        expected_weights = copy.deepcopy(items[0].state_dict())
        expected_optimizer = copy.deepcopy(items[1].state_dict())
        expected_lr = items[2].state_dict()
        expected_draws = (random.random(), np.random.rand(), torch.rand(4))
        resumed = make_training()
        start, updates = restore_checkpoint(path, *resumed[:3], self.signature, 0,
                                            generator=resumed[3])
        self.assertEqual((start, updates), (1, 2))
        self.assertEqual(epoch(*resumed, start), expected_losses)
        for key, value in resumed[0].state_dict().items():
            torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)
        actual = resumed[1].state_dict()
        self.assertEqual(actual["param_groups"], expected_optimizer["param_groups"])
        for key in actual["state"]:
            for field in actual["state"][key]:
                torch.testing.assert_close(actual["state"][key][field], expected_optimizer["state"][key][field],
                                           rtol=0, atol=0)
        self.assertEqual(resumed[2].state_dict(), expected_lr)
        self.assertEqual(random.random(), expected_draws[0])
        self.assertEqual(np.random.rand(), expected_draws[1])
        torch.testing.assert_close(torch.rand(4), expected_draws[2], rtol=0, atol=0)

    def test_signature_mismatch_and_no_overwrite(self):
        items = make_training()
        path = self.root / "state.pt"
        self.save(path, items)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.save(path, items)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            restore_checkpoint(path, *items[:3], dict(self.signature, cet_weight=0.2), 0)
        self.assertEqual(before, path.read_bytes())

    def test_latest_skips_incomplete_and_corrupted_states(self):
        items = make_training()
        first = self.root / "save/state_epoch_0000.pt"
        self.save(first, items)
        second = self.root / "save/state_epoch_0001.pt"
        self.save(second, items, 1)
        self.assertEqual(resolve_checkpoint("latest", self.root), second)
        with self.assertRaisesRegex(ValueError, "repeat completed epochs"):
            validate_resume_destination(str(first), self.root)
        self.assertEqual(validate_resume_destination(str(first), self.root / "new-experiment"), first.resolve())
        with second.open("ab") as stream:
            stream.write(b"corrupt")
        (self.root / "save/state_epoch_0002.pt").write_bytes(b"incomplete")
        self.assertEqual(resolve_checkpoint("latest", self.root), first)
        with self.assertRaises(ValueError):
            resolve_checkpoint(str(second), self.root)

    def test_experiment_guard_and_portable_cuda_visibility(self):
        run = self.root / "run"
        self.assertTrue(prepare_experiment(run, commit=True, resume=None))
        run.mkdir()
        (run / "previous.pt").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            prepare_experiment(run, commit=True, resume=None)
        self.assertTrue(prepare_experiment(run, commit=True, resume="latest"))
        with self.assertRaises(ValueError):
            prepare_experiment(run, commit=False, resume="latest")
        self.assertEqual(select_cuda_devices([0], 1, "3"), "3")
        self.assertEqual(select_cuda_devices([1, 0], 2, "GPU-abc,MIG-xyz"), "MIG-xyz,GPU-abc")
        self.assertEqual(select_cuda_devices([1, 0], 2), "1,0")
        for ids, available, mask in (([1], 1, "3"), ([0, 0], 2, None), ([-1], 2, None), ([0], 1, "")):
            with self.assertRaises(ValueError):
                select_cuda_devices(ids, available, mask)

    def test_rng_and_epoch_boundary_time_budget(self):
        torch.manual_seed(12)
        state = capture_rng()
        expected = torch.randn(3)
        restore_rng(state)
        torch.testing.assert_close(torch.randn(3), expected, rtol=0, atol=0)
        self.assertFalse(stop_at_boundary(False, None, 100, now=100))
        self.assertTrue(stop_at_boundary(True, None, 100, now=100))
        self.assertFalse(stop_at_boundary(False, 2000, 100, now=100))
        self.assertTrue(stop_at_boundary(False, 800, 100, now=100))

    def test_two_cpu_ranks_capture_distinct_rng_states(self):
        mp.spawn(distributed_save_worker, args=(str(self.root), False), nprocs=2)
        state = torch.load(self.root / "state.pt", map_location="cpu")
        self.assertEqual(len(state["rank_states"]), 2)
        self.assertFalse(torch.equal(state["rank_states"][0]["rng"]["torch"],
                                     state["rank_states"][1]["rng"]["torch"]))

    def test_rank_zero_write_failure_is_propagated_to_all_ranks(self):
        (self.root / "state.pt").write_bytes(b"existing checkpoint must survive")
        mp.spawn(distributed_save_worker, args=(str(self.root), True), nprocs=2)
        self.assertEqual((self.root / "state.pt").read_bytes(), b"existing checkpoint must survive")
        self.assertTrue(all((self.root / f"rank_{rank}_finished").exists() for rank in range(2)))

    def test_signature_uses_asset_content_not_machine_paths(self):
        from workflow_fixtures import configuration
        signatures = []
        for name in ("machine_a", "machine_b"):
            root = self.root / name
            (root / "mano/models").mkdir(parents=True)
            for asset in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
                (root / "mano/models" / asset).write_bytes(b"same immutable test asset")
            (root / "edges.npy").write_bytes(b"same edges")
            (root / "weights.npy").write_bytes(b"same weights")
            cfg = configuration()
            cfg["model"] = dict(input_dim=99)
            cfg["mano"] = dict(mano_path=str(root / "mano"))
            cfg["train"].update(scheduler_milestone=[150, 250], scheduler_gamma=0.5,
                                loss=dict(cet_weight=0.1, vpe_path=str(root / "edges.npy"),
                                          c_weight_path=str(root / "weights.npy")))
            signatures.append(make_signature(cfg, dict(train="same cache"), 1))
        self.assertEqual(*signatures)


if __name__ == "__main__":
    unittest.main()
