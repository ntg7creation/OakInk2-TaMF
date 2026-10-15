"""DTW-aligned anatomical finger bend loss, in radians.

The angle convention and hard-DTW recurrence match Scripts/evaluate_cet.py.
Only alignment copies are detached: the selected differences are scored again
on the original Torch tensors. The discrete path is not differentiated.
"""

import math
import time

import numpy as np
import torch
from torch import nn

from dev_fn.transform.rotation import rot6d_to_rotmat


CONVENTION = "manotorch_AxisLayerFK_template_XYZ_signed_bend_Z_v1"
PARENTS = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
SIDES = ("rh", "lh")


class CETInputError(ValueError):
    """Invalid articulation/time data; never silently replaced with zero loss."""


def normalized_time(frame_ids):
    """Normalize sequence order, including descending reverse-augmented IDs."""
    frames = np.asarray(frame_ids, dtype=np.float64)
    if frames.ndim != 1 or len(frames) == 0 or not np.isfinite(frames).all():
        raise CETInputError("expected nonempty finite frame IDs")
    if len(frames) == 1:
        return np.zeros(1, dtype=np.float64)
    delta = np.diff(frames)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise CETInputError("frame IDs must be strictly monotonic in sequence order")
    return (frames - frames[0]) / (frames[-1] - frames[0])


def hard_dtw_path(pred, target, pred_time=None, target_time=None, rho=0.10):
    """CPU float64 minimum-cumulative-MAE path, with evaluator tie ordering.

    Path length normalization belongs to the subsequent Torch scoring, not the
    dynamic program. One common path aligns all 15 joints.
    """
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise CETInputError("rho must be finite and in [0,1]")
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    for angles in (pred, target):
        if angles.ndim != 2 or angles.shape[1] != 15 or len(angles) == 0 or not np.isfinite(angles).all():
            raise CETInputError("DTW expects nonempty finite [L,15] angles")
    u = normalized_time(np.arange(len(pred))) if pred_time is None else np.asarray(pred_time, dtype=np.float64)
    v = normalized_time(np.arange(len(target))) if target_time is None else np.asarray(target_time, dtype=np.float64)
    for grid, length in ((u, len(pred)), (v, len(target))):
        if (grid.shape != (length,) or not np.isfinite(grid).all()
                or not np.all(np.diff(grid) > 0) or grid[0] != 0
                or grid[-1] != (1 if length > 1 else 0)):
            raise CETInputError("invalid normalized time grid")
    errors = np.abs(pred[:, None, :] - target[None, :, :])
    # Keep evaluator arithmetic as well as tie ordering for parity tests.
    costs = errors @ np.ones(15, dtype=np.float64) / 15
    band = np.abs(u[:, None] - v[None, :]) <= rho + 1e-12
    accumulated = np.full(costs.shape, np.inf)
    previous = np.full(costs.shape, -1, dtype=np.int8)
    steps = ((1, 1), (1, 0), (0, 1))
    for i in range(len(pred)):
        for k in np.flatnonzero(band[i]):
            if i == k == 0:
                accumulated[i, k] = costs[i, k]
                continue
            candidates = [accumulated[i - di, k - dk] if i >= di and k >= dk else np.inf
                          for di, dk in steps]
            choice = int(np.argmin(candidates))
            if np.isfinite(candidates[choice]):
                accumulated[i, k] = costs[i, k] + candidates[choice]
                previous[i, k] = choice
    if not np.isfinite(accumulated[-1, -1]):
        raise CETInputError("no endpoint-to-endpoint DTW path within the time window")
    i, k = len(pred) - 1, len(target) - 1
    path = [(i, k)]
    while i or k:
        di, dk = steps[previous[i, k]]
        i, k = i - di, k - dk
        path.append((i, k))
    return np.asarray(path[::-1], dtype=np.int64)


def local_flexion_angles(pose_repr, basis, reject_branch_crossings=False):
    """[L,99] -> [L,15] signed anatomical XYZ Z-bend, excluding root.

    Call on valid frames only. Nonroot MANO rotations are already local to their
    parents; there is no additional parent-rotation subtraction here.
    """
    if pose_repr.ndim != 2 or pose_repr.shape[1] != 99 or len(pose_repr) == 0:
        raise CETInputError("expected nonempty [L,99] pose representation")
    if pose_repr.dtype not in (torch.float32, torch.float64):
        raise CETInputError("CET requires float32 or float64 articulation")
    if basis.shape != (16, 3, 3):
        raise CETInputError("expected [16,3,3] anatomical basis")
    d6 = pose_repr[:, 9:99].reshape(-1, 15, 6)
    # Guard before atan2/normalization. No invalid or padded values enter scoring.
    with torch.no_grad():
        if not torch.isfinite(d6).all():
            raise CETInputError("nonfinite articulated 6D rotation")
        first, second = d6[..., :3], d6[..., 3:]
        first_norm = first.norm(dim=-1, keepdim=True)
        if not (first_norm > 1e-8).all():
            raise CETInputError("6D rotation has zero/near-zero first axis")
        unit = first / first_norm
        residual = second - (unit * second).sum(-1, keepdim=True) * unit
        if not (residual.norm(dim=-1) > 1e-8).all():
            raise CETInputError("6D rotation has collinear/zero second axis")
    local = rot6d_to_rotmat(d6)
    axes = basis[1:].to(device=pose_repr.device, dtype=pose_repr.dtype)
    anatomical = axes.transpose(-1, -2) @ local @ axes
    with torch.no_grad():
        margin = torch.hypot(anatomical[..., 0, 0], anatomical[..., 0, 1])
        if not torch.isfinite(anatomical).all() or not (margin > 1e-6).all():
            raise CETInputError("undefined anatomical XYZ bend / spread gimbal singularity")
    angles = torch.atan2(-anatomical[..., 0, 1], anatomical[..., 0, 0])
    if reject_branch_crossings:
        with torch.no_grad():
            if (torch.abs(angles[1:] - angles[:-1]) > math.pi).any():
                raise CETInputError("GT signed bend crosses the +/-pi branch")
    return angles


def load_anatomical_bases(mano_path):
    """Lazy MANO template dependency; not invoked when CET is disabled."""
    from manotorch.axislayer import AxisLayerFK

    bases = {}
    with torch.no_grad():
        for side, full_name in (("rh", "right"), ("lh", "left")):
            layer = AxisLayerFK(side=full_name, mano_assets_root=str(mano_path)).cpu().eval()
            if tuple(layer.transf_parent_mapping) != PARENTS:
                raise CETInputError("MANO parent mapping differs from CET convention")
            bases[side] = layer.TMPL_R_p_a[0].detach().clone()
    return bases


class CETLoss(nn.Module):
    """Equal-motion batch-mean hard-DTW L1 CET in radians, no trainable weights.

    ``bases`` can supply fixed templates for independent tests. Profiling is
    opt-in: total forward timing synchronizes CUDA, CPU DTW timing does not
    include device-to-host transfer. Neither includes backward time.
    """

    def __init__(self, mano_path=None, rho=0.10, *, bases=None, profile=False):
        super().__init__()
        if not math.isfinite(rho) or not 0 <= rho <= 1:
            raise CETInputError("rho must be finite and in [0,1]")
        if bases is None:
            if mano_path is None:
                raise CETInputError("mano_path or fixed anatomical bases are required")
            bases = load_anatomical_bases(mano_path)
        templates = torch.stack([torch.as_tensor(bases[side]).detach().clone() for side in SIDES])
        if templates.shape != (2, 16, 3, 3) or not torch.is_floating_point(templates):
            raise CETInputError("expected floating anatomical bases [2,16,3,3]")
        identity = torch.eye(3, dtype=templates.dtype, device=templates.device).expand_as(templates)
        if (not torch.isfinite(templates).all()
                or not torch.allclose(templates @ templates.transpose(-1, -2), identity, atol=2e-4, rtol=0)
                or not torch.allclose(torch.linalg.det(templates), torch.ones_like(templates[..., 0, 0]), atol=2e-4, rtol=0)):
            raise CETInputError("anatomical bases must be proper orthonormal rotations")
        self.register_buffer("anatomical_bases", templates)
        self.rho = float(rho)
        self.profile = bool(profile)

    def forward(self, pred_pose_repr, gt_pose_repr, mask, hand_side, frame_id):
        if (pred_pose_repr.ndim != 3 or pred_pose_repr.shape[-1] != 99
                or pred_pose_repr.shape != gt_pose_repr.shape or pred_pose_repr.shape[0] == 0):
            raise CETInputError("prediction and GT must have matching nonempty [B,T,99] shapes")
        if pred_pose_repr.device != gt_pose_repr.device or pred_pose_repr.dtype != gt_pose_repr.dtype:
            raise CETInputError("prediction and GT must share device and dtype")
        batch_size, seq_len, _ = pred_pose_repr.shape
        if mask.shape != (batch_size, seq_len) or len(hand_side) != batch_size or len(frame_id) != batch_size:
            raise CETInputError("mask, hand_side and frame_id must match the pose batch")
        if self.profile and pred_pose_repr.is_cuda:
            torch.cuda.synchronize(pred_pose_repr.device)
        start = time.perf_counter() if self.profile else None
        masks = mask.detach().cpu().numpy()
        losses, path_lengths, circular_excess = [], [], []
        crossings = adjacent_count = over_pi = aligned_count = 0
        dtw_seconds = 0.0
        for index in range(batch_size):
            side = hand_side[index]
            try:
                if side not in SIDES:
                    raise CETInputError("hand_side must be rh or lh")
                valid = masks[index]
                length = int(np.count_nonzero(valid))
                if (not np.isin(valid, (0, 1)).all() or length == 0
                        or not np.all(valid[:length] == 1) or not np.all(valid[length:] == 0)):
                    raise CETInputError("mask must be a nonempty binary valid prefix")
                if len(frame_id[index]) != length:
                    raise CETInputError("frame_id length differs from valid mask length")
                grid = normalized_time(frame_id[index])
                axes = self.anatomical_bases[SIDES.index(side)]
                prediction = local_flexion_angles(pred_pose_repr[index, :length], axes)
                with torch.no_grad():
                    target = local_flexion_angles(gt_pose_repr[index, :length].detach(), axes, True)
                pred_np = prediction.detach().cpu().double().numpy()
                gt_np = target.detach().cpu().double().numpy()
                dtw_start = time.perf_counter() if self.profile else None
                path = hard_dtw_path(pred_np, gt_np, grid, grid, self.rho)
                if self.profile:
                    dtw_seconds += time.perf_counter() - dtw_start
                indices = torch.as_tensor(path, device=prediction.device)
                # Crucially, score ORIGINAL prediction tensors, not NumPy copies.
                losses.append((prediction[indices[:, 0]] - target[indices[:, 1]]).abs().mean())
                path_lengths.append(len(path))
                crossings += int(np.count_nonzero(np.abs(np.diff(pred_np, axis=0)) > np.pi))
                adjacent_count += max(length - 1, 0) * 15
                delta = pred_np[path[:, 0]] - gt_np[path[:, 1]]
                absolute = np.abs(delta)
                over_pi += int(np.count_nonzero(absolute > np.pi))
                aligned_count += absolute.size
                # Diagnostic only: never substitute circular error in the loss.
                circular = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
                circular_excess.append(float(np.maximum(absolute - circular, 0).mean()))
            except CETInputError as exc:
                raise CETInputError(f"CET sample {index} ({side}): {exc}") from exc
        loss = torch.stack(losses).mean()
        diagnostics = {
            "cet_pred_branch_crossings": crossings,
            "cet_pred_branch_crossing_rate": crossings / adjacent_count if adjacent_count else 0.0,
            "cet_aligned_error_over_pi_rate": over_pi / aligned_count,
            "cet_non_circular_excess_rad": float(np.mean(circular_excess)),
            "cet_path_length_mean": float(np.mean(path_lengths)),
        }
        if self.profile:
            if pred_pose_repr.is_cuda:
                torch.cuda.synchronize(pred_pose_repr.device)
            diagnostics["cet_cpu_dtw_seconds"] = dtw_seconds
            diagnostics["cet_forward_seconds"] = time.perf_counter() - start
        return loss, diagnostics
