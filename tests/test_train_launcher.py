"""Actual launcher control flow, real diffusion/AdamW, tiny CPU test doubles.

No TaMF generator is trained or sampled here. GPU/model/asset integration must
be checked separately on the target environment.
"""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oakink2_tamf.launch import train
from oakink2_tamf.dataset.cache_util import write_cache, cache_spec
from oakink2_tamf.util.training_state import resolve_checkpoint
from workflow_fixtures import tiny_cache, configuration


class TinyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.rand(99) * 0.01 + 0.1)

    def forward(self, x, timesteps, batch):
        return x * self.scale[None, :, None, None]


class Wrapper(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TinyExtraLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, output, batch):
        value = 0.01 * output.square().mean()
        return value, dict(loss=value)


class SilentSummary:
    ddp_enabled = False

    def __init__(self, **kwargs):
        self.steps = []

    def add_scalar(self, tag, value, global_step, **kwargs):
        self.steps.append(global_step)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = configuration()
        self.cfg["model"] = dict(input_dim=99, obj_input_dim=9, hand_shape_dim=10, obj_embed_dim=768,
                                 latent_dim=512, ff_size=2048, num_layers=8, num_heads=4,
                                 dropout=0.1, activation="gelu")
        self.cfg["mano"] = dict(mano_path="not-used-by-test-double")
        self.cfg["runtime"]["device_id"] = [0]
        self.cfg["train"].update(num_epoch=2, scheduler_milestone=[1], scheduler_gamma=0.5,
                                  record_freq=1, loss={}, reload_ckpt_model_filepath=None,
                                  reload_ckpt_optimizer_filepath=None, resume_checkpoint=None,
                                  require_verified_cache=True)
        cache = self.root / "train.pkl"
        write_cache(cache, tiny_cache(), cache_spec(["seq_a", "seq_b"], "train"))
        self.cfg["train"]["cache_dict_filepath"] = str(cache)

    def run_launcher(self, directory, cfg, *, stop=False, sample_calls=None):
        proxy = SimpleNamespace(**vars(torch))
        proxy.device = lambda name: torch.device("cpu")
        proxy.cuda = SimpleNamespace(set_device=lambda device: None)
        proxy.nn = SimpleNamespace(parallel=SimpleNamespace(DistributedDataParallel=Wrapper))
        cfg = copy.deepcopy(cfg)
        rendezvous = self.root / (directory + "-" + str(cfg["train"]["num_epoch"]) + "-rendezvous")
        def setup(rank, world_size):
            dist.init_process_group("gloo", init_method="file://" + str(rendezvous), rank=rank, world_size=world_size)
        original_diffusion = train.create_gaussian_diffusion
        def diffusion(**kwargs):
            instance = original_diffusion(**kwargs)
            if sample_calls is not None:
                def sample(model, shape, **kw):
                    sample_calls.append((model, kw["model_kwargs"]["batch"]))
                    return torch.zeros(shape)
                instance.p_sample_loop = sample
            return instance
        ckpt_cfg = dict(commit=True, ckpt_path=str(self.root / directory),
                        log_file=str(self.root / directory / "unused.log"), exp_id=directory)
        try:
            with patch.object(train, "torch", proxy), patch.object(train, "InterationSegmentMDM", TinyModel), \
                    patch.object(train, "InteractionSegmentExtraLoss", TinyExtraLoss), \
                    patch.object(train, "DDPSummaryWriter", SilentSummary), \
                    patch.object(train.ddp_util, "setup_ddp", setup), \
                    patch.object(train, "make_signature", return_value=dict(world_size=1, fixed="test")), \
                    patch.object(train, "create_gaussian_diffusion", diffusion), \
                    patch.object(train.InteractionSegmentData, "load_dataset", side_effect=AssertionError("raw preprocessing")), \
                    patch.object(train, "map_copy_select_to", wraps=train.map_copy_select_to) as transfer:
                train.run(0, 1, ckpt_cfg, cfg, "test-session-" + str(cfg["train"]["num_epoch"]),
                          SimpleNamespace(is_set=lambda: stop), None)
                self.assertTrue(all("obj_traj" in call.kwargs["select"] for call in transfer.call_args_list))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        return torch.load(resolve_checkpoint("latest", self.root / directory), map_location="cpu")

    def test_actual_launcher_cache_start_and_epoch_resume(self):
        self.cfg["data"]["append_reverse_segment"] = True
        uninterrupted = self.run_launcher("uninterrupted", self.cfg)
        first_cfg = copy.deepcopy(self.cfg)
        first_cfg["train"]["num_epoch"] = 1
        first = self.run_launcher("resumed", first_cfg)
        self.assertEqual(first["next_epoch"], 1)
        self.cfg["train"]["resume_checkpoint"] = "latest"
        resumed = self.run_launcher("resumed", self.cfg)
        self.assertEqual(resumed["next_epoch"], 2)
        self.assertEqual(resumed["global_updates"], 8)
        for key in resumed["model"]:
            torch.testing.assert_close(resumed["model"][key], uninterrupted["model"][key], rtol=0, atol=0)
        self.assertEqual(resumed["scheduler"], uninterrupted["scheduler"])
        self.assertEqual(resumed["optimizer"]["param_groups"], uninterrupted["optimizer"]["param_groups"])

    def test_stop_request_saves_only_completed_epoch(self):
        state = self.run_launcher("stopped", self.cfg, stop=True)
        self.assertEqual(state["next_epoch"], 1)
        self.assertEqual(state["global_updates"], 2)

    def test_enabled_evaluation_transfers_trajectory_and_uses_unwrapped_model(self):
        self.cfg["train"]["num_epoch"] = 1
        for split in ("val", "test"):
            path = self.root / (split + ".pkl")
            write_cache(path, tiny_cache([split + "_seq"]), cache_spec([split + "_seq"], split))
            self.cfg[split]["cache_dict_filepath"] = str(path)
            self.cfg[split][split + "_freq"] = 1
        calls = []
        self.run_launcher("evaluation", self.cfg, sample_calls=calls)
        self.assertEqual(len(calls), 2)
        for model, batch in calls:
            self.assertIsInstance(model, TinyModel)
            self.assertIn("obj_traj", batch)
            self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
