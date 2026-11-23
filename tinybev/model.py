"""Camera-only lift, splat and shared BEV prediction."""
from dataclasses import dataclass
import math

import torch
from torch import nn

CLASSES = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian",
           "motorcycle", "bicycle", "traffic_cone", "barrier")
MAP_LAYERS = ("drivable_area", "road_divider", "lane_divider", "ped_crossing")
CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK",
           "CAM_BACK_LEFT", "CAM_FRONT_LEFT")
STEPS = 6
BEV_WIDTH = 468


@dataclass(frozen=True)
class Grid:
    """Square ego grid with x along columns and y along rows."""

    size: int = 128
    extent: float = 51.2

    def __post_init__(self):
        if self.size < 1 or not math.isfinite(self.extent) or self.extent <= 0:
            raise ValueError("BEV grid size and extent must be positive and finite")

    @property
    def cell(self):
        return 2 * self.extent / self.size

    def cells(self, xy):
        return (xy + self.extent) / self.cell

    def centers(self, cells):
        """Convert fractional cell coordinates to meters."""
        return cells * self.cell - self.extent


def block(in_channels, out_channels, kernel=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.BatchNorm2d(outgoing),
        )
        self.skip = nn.Identity() if stride == 1 else nn.Sequential(
            nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing)
        )

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


def resnet18():
    layers = [nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
              nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1)]
    incoming = 64
    for outgoing in (64, 128, 256, 512):
        layers.extend((ResidualBlock(incoming, outgoing, 1 if incoming == outgoing else 2),
                       ResidualBlock(outgoing, outgoing, 1)))
        incoming = outgoing
    backbone = nn.Sequential(*layers)
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    return backbone


def gather(features, xy):
    """Sample an integer (x, y) position per agent."""
    batch = torch.arange(features.shape[0], device=features.device)[:, None]
    return features.permute(0, 2, 3, 1)[batch, xy[..., 1], xy[..., 0]]


class TinyBEV(nn.Module):
    """Predict five tasks from a single calibrated multi-camera keyframe.

    Inputs are RGB tensors [B, V, 3, H, W], camera intrinsics [B, V, 3, 3],
    camera-to-ego transforms [B, V, 4, 4], and route commands [B, 3].
    Motion is predicted as displacements from the selected agent center. The
    planning head predicts waypoints in ego coordinates.
    """

    def __init__(self, grid=Grid(), width=BEV_WIDTH, topk=100, depths=44):
        super().__init__()
        if min(width, topk, depths) < 1:
            raise ValueError("BEV width, detection count and depth bins must be positive")
        self.grid, self.topk = grid, topk
        self.backbone = resnet18()
        self.depth = nn.Conv2d(512, depths, 1)
        self.context = nn.Conv2d(512, 128, 1)
        self.register_buffer("meters", torch.arange(depths, dtype=torch.float32) + 2)
        self.encoder = nn.Sequential(block(128, width), *(block(width, width) for _ in range(7)))
        self.det = nn.Sequential(block(width, 128), nn.Conv2d(128, len(CLASSES), 1))
        nn.init.constant_(self.det[-1].bias, -2.19)
        self.box = nn.Sequential(block(width, 128), nn.Conv2d(128, 10, 1))
        self.map = nn.Sequential(block(width, 128), nn.Conv2d(128, len(MAP_LAYERS), 1))
        self.occupancy = nn.Sequential(block(width, 128), nn.Conv2d(128, STEPS, 1))
        self.motion = nn.Sequential(nn.Linear(width + 10, 256), nn.ReLU(),
                                    nn.Linear(256, STEPS * 2))
        self.plan = nn.Sequential(nn.Linear(width + 3, 256), nn.ReLU(),
                                  nn.Linear(256, STEPS * 2))

    def lift_splat(self, images, intrinsics, camera_to_ego):
        """Pool depth-weighted image features into a dense ego grid."""
        if images.ndim != 5 or images.shape[2] != 3 or min(images.shape) == 0:
            raise ValueError("images must have shape [B, V, 3, H, W] with nonzero dimensions")
        batch, views, _, height, width = images.shape
        if intrinsics.shape != (batch, views, 3, 3):
            raise ValueError("intrinsics must have shape [B, V, 3, 3] matching images")
        if camera_to_ego.shape != (batch, views, 4, 4):
            raise ValueError("camera_to_ego must have shape [B, V, 4, 4] matching images")
        image_features = self.backbone(images.flatten(0, 1))
        context = self.context(image_features)
        probabilities = self.depth(image_features).softmax(1)
        h, w = context.shape[-2:]
        # Projection and discretization need FP32 even when convolutions use AMP.
        with torch.autocast(device_type=images.device.type, enabled=False):
            v, u = torch.meshgrid(
                (torch.arange(h, device=images.device, dtype=torch.float32) + .5) * height / h,
                (torch.arange(w, device=images.device, dtype=torch.float32) + .5) * width / w,
                indexing="ij",
            )
            pixels = torch.stack((u, v, torch.ones_like(u)), dim=-1).reshape(-1, 3)
            rays = torch.linalg.solve(intrinsics.float().flatten(0, 1), pixels.T)
            camera = rays.transpose(1, 2)[:, None] * self.meters.float()[None, :, None, None]
            extrinsics = camera_to_ego.float().flatten(0, 1)
            ego = camera @ extrinsics[:, None, :3, :3].transpose(-1, -2)
            ego = ego + extrinsics[:, None, None, :3, 3]
        cells = self.grid.cells(ego[..., :2]).floor().long()
        size = self.grid.size
        valid = ((cells >= 0) & (cells < size)).all(-1)
        valid &= (ego[..., 2] >= -5) & (ego[..., 2] < 3)
        camera_id, depth_id, pixel_id = valid.nonzero(as_tuple=True)
        xy = cells[camera_id, depth_id, pixel_id]
        indices = (camera_id // views) * size**2 + xy[:, 1] * size + xy[:, 0]
        values = context.flatten(2).transpose(1, 2)[camera_id, pixel_id]
        weights = probabilities.flatten(2)[camera_id, depth_id, pixel_id, None]
        pooled = context.new_zeros((batch * size**2, context.shape[1]), dtype=torch.float32)
        pooled.index_add_(0, indices, (values * weights).float())
        return pooled.reshape(batch, size, size, -1).permute(0, 3, 1, 2).contiguous()

    def candidate_cells(self, logits):
        probability = logits.sigmoid()
        maxima = nn.functional.max_pool2d(probability, 3, stride=1, padding=1)
        scores, indices = (probability * (probability == maxima)).flatten(1).topk(
            min(self.topk, probability.shape[1] * self.grid.size**2))
        size = self.grid.size
        xy = torch.stack((indices % size, (indices // size) % size), dim=-1)
        return scores, indices // size**2, xy

    def forecast(self, bev, boxes, cells):
        context = gather(bev, cells)
        states = gather(boxes, cells)
        prediction = self.motion(torch.cat((context, states), dim=-1))
        return prediction.reshape(*cells.shape[:2], STEPS, 2)

    def forward(self, images, intrinsics, camera_to_ego, command, agent_cells=None):
        if command.shape != images.shape[:1] + (3,):
            raise ValueError("command must have shape [B, 3] matching images")
        if agent_cells is not None and (agent_cells.ndim != 3 or agent_cells.shape[0] != images.shape[0]
                                        or agent_cells.shape[2] != 2 or agent_cells.dtype != torch.long):
            raise ValueError("agent_cells must contain integer indices with shape [B, N, 2]")
        bev = self.encoder(self.lift_splat(images, intrinsics, camera_to_ego))
        logits, boxes = self.det(bev), self.box(bev)
        scores, labels, cells = self.candidate_cells(logits)
        motion_cells = cells if agent_cells is None else agent_cells
        motion = self.forecast(bev, boxes, motion_cells)
        size = self.grid.size
        center_cells = slice(max(0, size // 2 - 1), size // 2 + 2)
        center = bev[:, :, center_cells, center_cells]
        plan = self.plan(torch.cat((center.mean((-1, -2)), command), 1))
        return dict(bev=bev, logits=logits, boxes=boxes, scores=scores, labels=labels,
                    cells=cells, motion=motion, motion_cells=motion_cells,
                    plan=plan.reshape(-1, STEPS, 2), map=self.map(bev), occ=self.occupancy(bev))

    def decode(self, output):
        cells = output["cells"]
        box = gather(output["boxes"], cells)
        xy = self.grid.centers(cells + box[..., :2])
        motion = output["motion"]
        if output.get("motion_cells", cells) is not cells:
            motion = self.forecast(output["bev"], output["boxes"], cells)
        return dict(center=torch.cat((xy, box[..., 2:3]), -1),
                    size=box[..., 3:6].clamp(-4, 4).exp(),
                    yaw=torch.atan2(box[..., 6], box[..., 7]), velocity=box[..., 8:10],
                    motion=motion + xy[..., None, :],
                    scores=output["scores"], labels=output["labels"])


class DistillationAdapter(nn.Module):
    def __init__(self, width, teacher_width=256):
        super().__init__()
        self.projection = nn.Conv2d(width, teacher_width, 1)

    def forward(self, bev):
        return self.projection(bev)
