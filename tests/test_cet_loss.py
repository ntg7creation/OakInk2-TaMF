"""Training CET correctness/parity/gradient tests; no generator jobs or downloads.

Run: python -m unittest discover -s tests -p test_cet_loss.py -v
"""

import math
import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Scripts"))
import evaluate_cet as evaluator
from dev_fn.transform.rotation import euler_angle_to_rotmat
from oakink2_tamf.model.loss.cet import (
    CETInputError, CETLoss, CONVENTION, hard_dtw_path, local_flexion_angles,
    normalized_time,
)


def fixed_bases(dtype=torch.float64):
    angles = torch.linspace(-0.4, 0.6, 48, dtype=dtype).reshape(16, 3)
    return {"rh": euler_angle_to_rotmat(angles, "XYZ"),
            "lh": euler_angle_to_rotmat(-angles.flip(0), "XYZ")}


def pose_from_bends(bends, bases, side="rh"):
    """[T,15] angles -> [T,99]; independent anatomy-aligned construction."""
    angles = torch.stack((bends * 0, bends * 0, bends), dim=-1)
    bend_rot = euler_angle_to_rotmat(angles, "XYZ")
    axes = bases[side][1:].to(bends)
    local = axes @ bend_rot @ axes.transpose(-1, -2)
    root = torch.eye(3, dtype=bends.dtype, device=bends.device).expand(len(bends), 1, 3, 3)
    rotations = torch.cat((root, local), dim=1)
    return torch.cat((bends.new_zeros(len(bends), 3), rotations[..., :2, :].reshape(len(bends), 96)), dim=-1)


def loss_inputs(pred_bends, gt_bends=None, side="rh", bases=None):
    bases = fixed_bases(pred_bends.dtype) if bases is None else bases
    gt_bends = torch.zeros_like(pred_bends) if gt_bends is None else gt_bends
    pred = pose_from_bends(pred_bends, bases, side).unsqueeze(0)
    gt = pose_from_bends(gt_bends, bases, side).unsqueeze(0)
    mask = torch.ones(1, len(pred_bends), dtype=pred_bends.dtype)
    return bases, (pred, gt, mask, [side], [list(range(len(pred_bends)))])


class CETMathTests(unittest.TestCase):
    def test_identical_and_known_constant_error_both_sides(self):
        for side in ("rh", "lh"):
            bends = torch.full((5, 15), 0.3, dtype=torch.float64)
            bases, args = loss_inputs(bends, side=side)
            module = CETLoss(bases=bases)
            actual, _ = module(*args)
            self.assertAlmostEqual(actual.item(), 0.3, places=12)
            zero, _ = module(args[0], args[0], *args[2:])
            self.assertLess(abs(zero.item()), 1e-12)
            changed = args[0].clone()
            changed[..., :3] = 100
            root = euler_angle_to_rotmat(torch.tensor([0.8, -0.6, 1.2], dtype=changed.dtype), "XYZ")
            changed[..., 3:9] = root[:2].reshape(6)
            invariant, _ = module(changed, *args[1:])
            torch.testing.assert_close(invariant, actual, rtol=0, atol=0)

    def test_one_joint_and_batch_mean_with_variable_lengths(self):
        bases = fixed_bases()
        pred = torch.zeros(2, 6, 99, dtype=torch.float64)
        gt = pred.clone()
        mask = torch.zeros(2, 6, dtype=torch.float64)
        for index, length in enumerate((2, 5)):
            bends = torch.zeros(length, 15, dtype=torch.float64)
            bends[:, 4] = 0.3 * (index + 1)
            pred[index, :length] = pose_from_bends(bends, bases)
            gt[index, :length] = pose_from_bends(torch.zeros_like(bends), bases)
            mask[index, :length] = 1
        pred[0, 2:] = float("nan")
        gt[1, 5:] = float("inf")
        actual, _ = CETLoss(bases=bases)(pred, gt, mask, ["rh", "rh"], [[10, 22], [70, 58, 46, 34, 22]])
        self.assertAlmostEqual(actual.item(), (0.3 / 15 + 0.6 / 15) / 2, places=12)

    def test_angle_and_score_parity_with_existing_evaluator(self):
        self.assertEqual(CONVENTION, evaluator.CONVENTION)
        rng = np.random.default_rng(41)
        for side in ("rh", "lh"):
            target = torch.tensor(rng.uniform(-0.2, 0.5, (18, 15)))
            prediction = target + torch.tensor(rng.uniform(0.02, 0.15, (18, 15)))
            bases, args = loss_inputs(prediction, target, side)
            extracted = local_flexion_angles(args[0][0], bases[side])
            legacy = evaluator.curl_from_local_rotations(
                evaluator.decode_pose(args[0][0].numpy(), 18), bases[side].numpy())
            np.testing.assert_allclose(extracted.numpy(), legacy, atol=2e-14, rtol=0)
            result = evaluator.constrained_dtw(prediction.numpy(), target.numpy(), rho=.10)
            np.testing.assert_array_equal(hard_dtw_path(prediction.numpy(), target.numpy()), result["path"])
            actual, _ = CETLoss(bases=bases)(*args)
            self.assertAlmostEqual(actual.item(), np.deg2rad(result["cet_degrees"]), places=12)

    def test_dtw_ties_unequal_lengths_window_and_normalization(self):
        rng = np.random.default_rng(9)
        for pred, gt, rho in ((np.zeros((4, 15)), np.zeros((4, 15)), .1),
                              (rng.normal(size=(4, 15)), rng.normal(size=(6, 15)), 1.0)):
            expected = evaluator.constrained_dtw(pred, gt, rho=rho)
            path = hard_dtw_path(pred, gt, rho=rho)
            np.testing.assert_array_equal(path, expected["path"])
            score = np.abs(pred[path[:, 0]] - gt[path[:, 1]]).sum() / (15 * len(path))
            self.assertAlmostEqual(score, np.deg2rad(expected["cet_degrees"]), places=12)
        with self.assertRaisesRegex(CETInputError, "no endpoint"):
            hard_dtw_path(np.zeros((2, 15)), np.zeros((3, 15)), rho=0)

    def test_dtw_really_aligns_time_shift(self):
        gt = np.repeat(np.array([0, .2, .6, .9, .4, .1, 0])[:, None], 15, axis=1)
        pred = np.repeat(np.array([0, 0, .2, .6, .9, .4, 0])[:, None], 15, axis=1)
        path = hard_dtw_path(pred, gt, rho=.3)
        self.assertTrue(np.any(path[:, 0] != path[:, 1]))
        self.assertLess(np.abs(pred[path[:, 0]] - gt[path[:, 1]]).mean(), np.abs(pred - gt).mean())

    def test_normalized_time_and_singletons(self):
        np.testing.assert_allclose(normalized_time([10, 12, 20]), [0, .2, 1])
        np.testing.assert_allclose(normalized_time([20, 12, 10]), [0, .8, 1])
        np.testing.assert_array_equal(normalized_time([42]), [0])
        for ids in ([], [1, 1], [1, 3, 2], [0, np.nan]):
            with self.assertRaises(CETInputError):
                normalized_time(ids)
        bases, args = loss_inputs(torch.full((1, 15), .25, dtype=torch.float64))
        loss, stats = CETLoss(bases=bases)(*args)
        self.assertAlmostEqual(loss.item(), .25, places=12)
        self.assertEqual(stats["cet_path_length_mean"], 1)

    def test_predicted_branch_diagnostics_and_non_circular_error(self):
        bends = torch.zeros(3, 15, dtype=torch.float64)
        bends[:, 0] = torch.tensor([3.1, -3.1, -3.0])
        bases, args = loss_inputs(bends)
        loss, stats = CETLoss(bases=bases)(*args)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["cet_pred_branch_crossings"], 1)
        self.assertAlmostEqual(stats["cet_pred_branch_crossing_rate"], 1 / 30)
        with self.assertRaisesRegex(CETInputError, "GT signed bend"):
            CETLoss(bases=bases)(args[1], args[0], *args[2:])
        pred = torch.full((2, 15), 3.1, dtype=torch.float64)
        bases, args = loss_inputs(pred, -pred)
        loss, stats = CETLoss(bases=bases)(*args)
        self.assertAlmostEqual(loss.item(), 6.2, places=12)
        self.assertEqual(stats["cet_aligned_error_over_pi_rate"], 1)
        self.assertAlmostEqual(stats["cet_non_circular_excess_rad"], 6.2 - (2 * math.pi - 6.2), places=12)

    def test_invalid_inputs_fail_without_silent_zero(self):
        bases, args = loss_inputs(torch.full((3, 15), .2, dtype=torch.float64))
        module = CETLoss(bases=bases)
        for invalid_mask in (torch.zeros(1, 3), torch.tensor([[1., 0, 1]]), torch.tensor([[1., .5, 0]])):
            with self.assertRaisesRegex(CETInputError, "mask"):
                module(args[0], args[1], invalid_mask, *args[3:])
        with self.assertRaisesRegex(CETInputError, "frame_id length"):
            module(*args[:4], [[0, 1]])
        with self.assertRaisesRegex(CETInputError, "hand_side"):
            module(*args[:3], ["bad"], args[4])
        for value in (float("nan"), float("inf"), 0.0):
            invalid = args[0].clone()
            invalid[:, :, 9:15] = value
            with self.assertRaises(CETInputError):
                module(invalid, *args[1:])
        invalid = args[0].clone()
        invalid[:, :, 12:15] = invalid[:, :, 9:12]
        with self.assertRaisesRegex(CETInputError, "collinear"):
            module(invalid, *args[1:])
        for rho in (-1, 2, float("nan")):
            with self.assertRaises(CETInputError):
                CETLoss(bases=bases, rho=rho)

    def test_gimbal_singularity_and_fixed_basis_validation(self):
        bases = {side: torch.eye(3, dtype=torch.float64).repeat(16, 1, 1) for side in ("rh", "lh")}
        _, args = loss_inputs(torch.full((2, 15), .2, dtype=torch.float64), bases=bases)
        bad = args[0].clone()
        singular = euler_angle_to_rotmat(torch.tensor([0., math.pi / 2, .5], dtype=torch.float64), "XYZ")
        bad[:, :, 9:15] = singular[:2].reshape(6)
        with self.assertRaisesRegex(CETInputError, "gimbal"):
            CETLoss(bases=bases)(bad, *args[1:])
        bases["rh"][1] = 0
        with self.assertRaisesRegex(CETInputError, "orthonormal"):
            CETLoss(bases=bases)

    def test_profiling_opt_in(self):
        bases, args = loss_inputs(torch.full((5, 15), .2, dtype=torch.float64))
        plain, plain_stats = CETLoss(bases=bases)(*args)
        profiled, stats = CETLoss(bases=bases, profile=True)(*args)
        torch.testing.assert_close(plain, profiled, rtol=0, atol=0)
        self.assertNotIn("cet_cpu_dtw_seconds", plain_stats)
        self.assertGreater(stats["cet_cpu_dtw_seconds"], 0)
        self.assertGreaterEqual(stats["cet_forward_seconds"], stats["cet_cpu_dtw_seconds"])


class CETGradientTests(unittest.TestCase):
    def test_160_frame_forward_backward(self):
        bends = torch.linspace(.1, .5, 160).unsqueeze(1).expand(160, 15)
        bases, args = loss_inputs(bends)
        pred = args[0].detach().clone().requires_grad_()
        loss, stats = CETLoss(bases=bases)(pred, *args[1:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad[..., 9:].abs().sum().item(), 0)
        self.assertGreaterEqual(stats["cet_path_length_mean"], 160)
        self.assertLessEqual(stats["cet_path_length_mean"], 319)

    def test_padding_has_zero_finite_gradients(self):
        bases, args = loss_inputs(torch.full((5, 15), .3, dtype=torch.float64))
        pred = args[0].clone()
        gt = args[1].clone()
        pred[:, 2:] = float("nan")
        gt[:, 2:] = float("inf")
        pred.requires_grad_()
        loss, _ = CETLoss(bases=bases)(pred, gt, torch.tensor([[1., 1, 0, 0, 0]]), ["rh"], [[10, 22]])
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad[:, 2:].abs().sum().item(), 0)
        self.assertGreater(pred.grad[:, :2, 9:].abs().sum().item(), 0)

    def test_gradcheck_and_target_detachment(self):
        bases, args = loss_inputs(torch.full((3, 15), .3, dtype=torch.float64))
        pred = args[0].clone().requires_grad_()
        target = args[1].clone().requires_grad_()
        module = CETLoss(bases=bases, rho=0)
        self.assertTrue(torch.autograd.gradcheck(lambda x: module(x, target, *args[2:])[0],
                                                (pred,), eps=1e-6, atol=1e-5, rtol=1e-4))
        module(pred, target, *args[2:])[0].backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad[..., 9:].abs().sum().item(), 0)
        self.assertEqual(pred.grad[..., :9].abs().sum().item(), 0)

    def test_repeated_path_indices_accumulate_gradients(self):
        pred = torch.full((2, 15), .4, dtype=torch.float64, requires_grad=True)
        target = np.zeros((3, 15))
        path = hard_dtw_path(pred.detach().numpy(), target, rho=1)
        indices = torch.from_numpy(path)
        (pred[indices[:, 0]] - torch.from_numpy(target)[indices[:, 1]]).abs().mean().backward()
        counts = np.bincount(path[:, 0], minlength=2)
        expected = torch.tensor(counts[:, None] / (15 * len(path))).expand_as(pred)
        torch.testing.assert_close(pred.grad, expected, rtol=0, atol=0)

    def test_real_diffusion_callback_preserves_metadata_and_model_gradients(self):
        from oakink2_tamf.dataset.collate import interaction_segment_collate
        from dev_fn.transform.cast import map_copy_select_to
        from oakink2_tamf.model.diffusion_util import create_gaussian_diffusion

        bases, args = loss_inputs(torch.full((3, 15), .3, dtype=torch.float64))
        item = {"pose_repr": args[1][0].numpy(), "mask": args[2][0].numpy(),
                "frame_id": [100, 112, 124], "hand_side": "rh"}
        batch = map_copy_select_to(interaction_segment_collate([item]), device="cpu", dtype=torch.float64,
                                   select=("mask", "pose_repr"))
        self.assertEqual(batch["frame_id"], [[100, 112, 124]])
        self.assertEqual(batch["hand_side"], ["rh"])
        module = CETLoss(bases=bases)

        class TinyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pose = torch.nn.Parameter(args[0].clone())

            def forward(self, x, timesteps, batch):
                return self.pose.transpose(1, 2).unsqueeze(2) + x * 0

        model = TinyDenoiser()
        def callback(output, passed_batch):
            self.assertIs(passed_batch, batch)
            return module(output.squeeze(2).transpose(1, 2), passed_batch["pose_repr"],
                          passed_batch["mask"], passed_batch["hand_side"], passed_batch["frame_id"])
        diffusion = create_gaussian_diffusion(1000, "cosine")
        x_start = batch["pose_repr"].transpose(1, 2).unsqueeze(2)
        _, (cet, _) = diffusion.training_losses(model, x_start, torch.tensor([500]),
                                                model_kwargs={"batch": batch}, noise=torch.zeros_like(x_start),
                                                loss_callback=callback)
        cet.backward()
        self.assertTrue(torch.isfinite(model.pose.grad).all())
        self.assertGreater(model.pose.grad[..., 9:].abs().sum().item(), 0)
        self.assertEqual(model.pose.grad[..., :9].abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; run on GPU server")
    def test_cuda_score_parity_and_finite_gradients(self):
        bases, args = loss_inputs(torch.full((8, 15), .3, dtype=torch.float32))
        cpu, _ = CETLoss(bases=bases)(*args)
        module = CETLoss(bases=bases, profile=True).cuda()
        pred = args[0].cuda().requires_grad_()
        loss, stats = module(pred, args[1].cuda(), args[2].cuda(), *args[3:])
        loss.backward()
        torch.testing.assert_close(cpu, loss.cpu(), rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad[..., 9:].abs().sum().item(), 0)
        self.assertGreater(stats["cet_forward_seconds"], 0)


@unittest.skipUnless((ROOT / "asset/mano_v1_2/models/MANO_RIGHT.pkl").is_file()
                     and (ROOT / "asset/mano_v1_2/models/MANO_LEFT.pkl").is_file(), "MANO assets unavailable")
class CETManoIntegrationTests(unittest.TestCase):
    def test_actual_anatomical_templates_match_evaluator_both_sides(self):
        from manotorch.axislayer import AxisLayerFK
        from manotorch.manolayer import ManoLayer
        from manotorch.utils.geometry import axis_angle_to_matrix
        module = CETLoss(ROOT / "asset/mano_v1_2")
        for index, side in enumerate(("right", "left")):
            axes = AxisLayerFK(side=side, mano_assets_root=str(ROOT / "asset/mano_v1_2"))
            angles = torch.zeros(3, 16, 3)
            angles[:, 1:, 1] = .3
            angles[:, 1:, 2] = torch.linspace(.05, .5, 15)
            local = axes.compose(angles)
            rotations = axis_angle_to_matrix(local)
            pose = torch.cat((torch.zeros(3, 3), rotations[..., :2, :].reshape(3, 96)), dim=1)
            extracted = local_flexion_angles(pose, module.anatomical_bases[index])
            reference = evaluator.curl_from_local_rotations(rotations.numpy(), axes.TMPL_R_p_a[0].numpy())
            np.testing.assert_allclose(extracted.numpy(), reference, atol=2e-6, rtol=0)
            mano = ManoLayer(side=side, mano_assets_root=str(ROOT / "asset/mano_v1_2"))
            _, _, official = axes(mano(local.reshape(3, 48), torch.zeros(3, 10)).transforms_abs)
            torch.testing.assert_close(extracted, official[:, 1:, 2], atol=2e-6, rtol=0)

    def test_zero_weight_does_not_initialize_cet(self):
        from oakink2_tamf.model.interaction_segment_extra_loss import InteractionSegmentExtraLoss
        with patch("oakink2_tamf.model.interaction_segment_extra_loss.CETLoss", side_effect=AssertionError("disabled")):
            module = InteractionSegmentExtraLoss(str(ROOT / "asset/mano_v1_2"), loss_config())
        self.assertIsNone(module.cet_loss)

    def test_original_losses_and_gradients_unchanged_at_zero_weight(self):
        from oakink2_tamf.model.interaction_segment_extra_loss import InteractionSegmentExtraLoss
        module = InteractionSegmentExtraLoss(str(ROOT / "asset/mano_v1_2"), loss_config(), use_pc=True)
        batch, prediction = geometry_fixture()
        reference_prediction = prediction.detach().clone().requires_grad_()
        reference, components = legacy_geometry_objective(module, reference_prediction, batch)
        reference.backward()
        actual_prediction = prediction.detach().clone().requires_grad_()
        actual, stats = module(actual_prediction, batch)
        actual.backward()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual_prediction.grad, reference_prediction.grad, rtol=0, atol=0)
        for name, value in components.items():
            torch.testing.assert_close(stats[name], value, rtol=0, atol=0)
        self.assertEqual(stats["cet"], 0)
        self.assertEqual(stats["cet_weighted"], 0)

    def test_enabled_integration_adds_only_weighted_batch_mean(self):
        from oakink2_tamf.model.interaction_segment_extra_loss import InteractionSegmentExtraLoss
        cfg = loss_config()
        cfg["cet_weight"] = .1
        module = InteractionSegmentExtraLoss(str(ROOT / "asset/mano_v1_2"), cfg, use_pc=True)
        batch, prediction = geometry_fixture()
        expected, components = legacy_geometry_objective(module, prediction, batch)
        cet, _ = module.cet_loss(prediction.squeeze(2).transpose(1, 2), batch["pose_repr"],
                                 batch["mask"], batch["hand_side"], batch["frame_id"])
        total, stats = module(prediction, batch)
        torch.testing.assert_close(total, expected + .1 * cet, rtol=0, atol=0)
        torch.testing.assert_close(stats["cet_weighted"], .1 * cet, rtol=0, atol=0)
        for name, value in components.items():
            torch.testing.assert_close(stats[name], value, rtol=0, atol=0)
        total.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


def loss_config():
    return {"vpe_path": str(ROOT / "asset/grabnet/verts_per_edge.npy"),
            "c_weight_path": str(ROOT / "asset/grabnet/rhand_weight.npy"),
            "coef_rec_joint_loss": 1., "coef_rec_vert_loss": 1., "coef_edge_len_loss": .1,
            "coef_dist_h_loss": .1, "coef_dist_o_loss": 1., "cet_weight": 0.}


class CETConfigurationTests(unittest.TestCase):
    def test_registered_defaults_and_explicit_smoke_overlay(self):
        from config_reg import ConfigRegistry
        from oakink2_tamf.launch.param.loss import reg_loss_param
        for arguments, weight, profile in (([], 0.0, False),
                                           (["--cfg", str(ROOT / "config/cet_smoke.yml")], .1, True),
                                           (["--train.loss.cet_weight", ".2", "--train.loss.cet_profile"], .2, True)):
            registry = ConfigRegistry(prog="cet-test")
            reg_loss_param(registry, "train.loss", str(ROOT))
            parser = argparse.ArgumentParser()
            registry.hook(parser)
            registry.parse(parser, arg_src=arguments)
            self.assertEqual(registry.select("train.loss.cet_weight"), weight)
            self.assertEqual(registry.select("train.loss.cet_profile"), profile)
            self.assertEqual(registry.select("train.loss.cet_dtw_rho"), .1)

    def test_smoke_harness_checks_callback_contract_without_running_model(self):
        spec = importlib.util.spec_from_file_location("cet_smoke_contract", ROOT / "script/debug/smoke_train_cet.py")
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        batch, _ = geometry_fixture()
        batch.update(text=["fixture", "fixture"], obj_embedding=torch.zeros(2, 1, 768),
                     obj_num=torch.ones(2, dtype=torch.long))
        metadata = harness.validate_batch_contract(batch)
        self.assertEqual([m["valid_length"] for m in metadata], [2, 3])
        self.assertEqual(metadata[1]["frame_id"], [70, 58, 46])
        for key in ("mask", "frame_id", "hand_side"):
            incomplete = dict(batch)
            incomplete.pop(key)
            with self.assertRaisesRegex(ValueError, key):
                harness.validate_batch_contract(incomplete)


def geometry_fixture():
    bases = fixed_bases(torch.float32)
    gt, pred = [], []
    for side in ("rh", "lh"):
        gt.append(pose_from_bends(torch.zeros(3, 15), bases, side))
        pred.append(pose_from_bends(torch.full((3, 15), .2), bases, side))
    batch = {"pose_repr": torch.stack(gt), "shape": torch.zeros(2, 3, 10),
             "hand_side": ["rh", "lh"], "mask": torch.tensor([[1., 1, 0], [1., 1, 1]]),
             "frame_id": [[10, 22], [70, 58, 46]], "obj_list": [["fixture"], ["fixture"]],
             "obj_pointcloud": [np.random.default_rng(8).normal(0, .025, (1, 12, 3)).astype(np.float32)] * 2,
             "obj_traj": torch.tensor([0., 0, 0, 1, 0, 0, 0, 1, 0]).expand(2, 1, 3, 9).clone()}
    return batch, torch.stack(pred).transpose(1, 2).unsqueeze(2).detach().requires_grad_()


def legacy_geometry_objective(module, output, batch):
    """Independent oracle for the original five formulas and local-batch SUM.

    Uses the unchanged MANO/distance utilities, but never calls the modified
    auxiliary forward. Includes both hand sides and different valid lengths.
    """
    from pytorch3d.structures import Meshes
    from dev_fn.transform.transform import tslrot6d_to_transf, transf_point_array
    from oakink2_tamf.model.loss.chamfer_distance import point2point_signed
    names = ("rec_joint", "rec_vert", "edge_len", "dist_h", "dist_o")
    values = {name: 0.0 for name in names}
    for i, side in enumerate(batch["hand_side"]):
        target_v, target_j = module.recover_mano_from_pose_repr(batch["pose_repr"][i], batch["shape"][i], side)
        pred_v, pred_j = module.recover_mano_from_pose_repr(output[i].permute(2, 0, 1).squeeze(-1), batch["shape"][i], side)
        mask = batch["mask"][i]
        coefficient = float(len(mask) / mask.sum())
        values["rec_joint"] += coefficient * (((pred_j - target_j) ** 2).sum(-1) * mask[:, None]).mean()
        values["rec_vert"] += coefficient * torch.einsum(
            "ij,j->ij", ((pred_v - target_v) ** 2).sum(-1) * mask[:, None], module.v_weights ** 2).mean()
        pe = pred_v[:, module.vpe[:, 0]] - pred_v[:, module.vpe[:, 1]]
        te = target_v[:, module.vpe[:, 0]] - target_v[:, module.vpe[:, 1]]
        values["edge_len"] += coefficient * ((pe - te).abs() * mask[:, None, None]).mean()
        faces = module.retrieve_hand_faces(side).unsqueeze(0)
        pn = Meshes(verts=pred_v, faces=faces).verts_normals_packed().reshape(-1, 778, 3)
        tn = Meshes(verts=target_v, faces=faces).verts_normals_packed().reshape(-1, 778, 3)
        for o in range(len(batch["obj_list"][i])):
            points = torch.from_numpy(batch["obj_pointcloud"][i][o]).to(pred_v)
            world = transf_point_array(tslrot6d_to_transf(batch["obj_traj"][i, o]), points[None].expand(len(mask), -1, -1))
            signed, distance, _ = point2point_signed(pred_v, world, pn)
            gt_signed, gt_distance, _ = point2point_signed(target_v, world, tn)
            weight = torch.ones_like(signed)
            weight[~((gt_signed < .01) & (gt_signed > -.005))] = .1
            weight[signed < 0] = 1.5
            count = len(batch["obj_list"][i])
            values["dist_h"] += coefficient * torch.einsum(
                "ij,j->ij", (distance.abs() - gt_distance.abs()).abs(), module.v_weights2).mul(mask[:, None]).mean() / count
            values["dist_o"] += coefficient * ((signed - gt_signed).abs() * weight * mask[:, None]).mean() / count
    loss = 0.0
    for name, coefficient in zip(names, (1., 1., .1, .1, 1.)):
        loss = loss + coefficient * values[name]
    return loss, values


if __name__ == "__main__":
    unittest.main()
