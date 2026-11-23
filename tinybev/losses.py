"""Ground-truth supervision and offline distillation for S0–S3."""
import torch
from torch.nn import functional as F

from .model import gather

KD_WEIGHTS = (1.0, .2, .5, .5, .5)


def masked_mean(values, mask):
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def focal(logits, targets):
    p = logits.sigmoid()
    confidence = torch.where(targets.bool(), p, 1 - p)
    alpha = torch.where(targets.bool(), .25, .75)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (alpha * (1 - confidence).square() * loss).sum() / targets.sum().clamp_min(1)


def bernoulli_kl(student, teacher):
    probability = teacher.sigmoid()
    return (F.binary_cross_entropy_with_logits(student, probability, reduction="none")
            - F.binary_cross_entropy_with_logits(teacher, probability, reduction="none"))


def objective(model, adapter, batch, variant, kd_weights=KD_WEIGHTS):
    feature_weight, detection_weight, motion_weight, plan_weight, region_weight = kd_weights
    inputs = [batch[key] for key in ("images", "intrinsics", "camera_to_ego", "command")]
    output = model(*inputs, agent_cells=batch["agent_cells"])
    losses = {
        "gt_det": focal(output["logits"], batch["heatmap"]),
        "gt_box": masked_mean((gather(output["boxes"], batch["agent_cells"])
                               - batch["boxes"]).abs(), batch["agents"]),
        "gt_motion": masked_mean((output["motion"] - batch["motion"]).square(),
                                 batch["motion_mask"]),
        "gt_plan": masked_mean((output["plan"] - batch["plan"]).square(), batch["plan_mask"]),
        "gt_map": F.binary_cross_entropy_with_logits(output["map"], batch["map"]),
        "gt_occ": masked_mean(F.binary_cross_entropy_with_logits(
            output["occ"], batch["occ"], reduction="none"), batch["plan_mask"]),
    }
    if variant in ("S1", "S3"):
        difference = (adapter(output["bev"]) - batch["teacher_bev"]).square()
        losses["kd_feature"] = feature_weight * difference.sum(1).mean()
        if variant == "S3":
            losses["kd_adaptive"] = region_weight * masked_mean(
                difference.sum(1), batch["salient"])
    if variant in ("S2", "S3"):
        cells, valid = batch["teacher_cells"], batch["teacher_agents"]
        box = gather(output["boxes"], cells)
        teacher_cls = batch["teacher_labels"]
        logits = gather(output["logits"], cells).gather(-1, teacher_cls[..., None]).squeeze(-1)
        losses["kd_detection"] = detection_weight * (
            masked_mean(bernoulli_kl(logits, batch["teacher_logits"]), valid)
            + masked_mean((box - batch["teacher_boxes"]).abs(), valid))
        trajectory = model.forecast(output["bev"], output["boxes"], cells)
        valid_steps = batch["teacher_motion_mask"]
        error = (trajectory - batch["teacher_motion"]).square().sum(-1)
        losses["kd_motion"] = motion_weight * (error * valid_steps).sum() / valid_steps.any(-1).sum().clamp_min(1)
        plan_error = (output["plan"] - batch["teacher_plan"]).square().sum(-1)
        losses["kd_plan"] = plan_weight * (plan_error * batch["plan_mask"]).sum(-1).mean()
    return sum(losses.values()), losses
