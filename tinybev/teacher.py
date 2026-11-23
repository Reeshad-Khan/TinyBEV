"""Read token-keyed UniAD exports and convert them to student coordinates."""
import numpy as np
import torch
from torch.nn import functional as F

from .data import box_encoding, yaw
from .model import CLASSES, STEPS


def read(path, key, shape):
    value = np.asarray(path[key], dtype=np.float32)
    if value.ndim != len(shape) or any(d is not None and d != s for d, s in zip(shape, value.shape)):
        raise ValueError(f"{key} expected {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains nonfinite values")
    return value


def warp_bev(bev, lidar_to_ego, grid):
    centers = (torch.arange(grid.size, dtype=torch.float32) + .5) * grid.cell - grid.extent
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    ego = torch.stack((x, y, torch.zeros_like(x), torch.ones_like(x)), -1)
    ego_to_lidar = torch.linalg.inv(torch.from_numpy(lidar_to_ego))
    lidar = ego @ ego_to_lidar.T
    return F.grid_sample(bev[None], lidar[None, ..., :2] / 51.2,
                         align_corners=False)[0]


def load_teacher(path, grid, variant, maximum):
    if not path.is_file():
        raise FileNotFoundError(f"Missing teacher export {path}")
    result = {}
    with np.load(path, allow_pickle=False) as data:
        lidar_to_ego = read(data, "lidar_to_ego", (4, 4))
        rotation, translation = lidar_to_ego[:3, :3], lidar_to_ego[:3, 3]
        if (not np.allclose(lidar_to_ego[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
            raise ValueError(f"Invalid LiDAR-to-ego transform in {path}")
        if variant in ("S1", "S3"):
            bev = torch.from_numpy(read(data, "bev", (256, 200, 200)))
            result["teacher_bev"] = warp_bev(bev, lidar_to_ego, grid)
        if variant in ("S2", "S3"):
            centers = read(data, "center", (None, 3))
            count = len(centers)
            sizes = read(data, "size", (count, 3))
            angles = read(data, "yaw", (count,))
            velocities = read(data, "velocity", (count, 2))
            raw_labels = read(data, "label", (count,))
            scores = read(data, "score", (count,))
            if ((raw_labels != np.floor(raw_labels)) | (raw_labels < 0)
                    | (raw_labels >= len(CLASSES))).any():
                raise ValueError(f"Invalid teacher detection labels in {path}")
            labels = raw_labels.astype(int)
            if (sizes <= 0).any() or ((scores < 0) | (scores > 1)).any():
                raise ValueError(f"Invalid teacher detection dimensions or scores in {path}")
            motion = read(data, "motion", (count, STEPS, 2))
            motion_valid = read(data, "motion_valid", (count, STEPS))
            if ((motion_valid != 0) & (motion_valid != 1)).any():
                raise ValueError(f"Invalid teacher motion mask in {path}")
            valid = motion_valid.astype(bool)
            plan = read(data, "plan", (STEPS, 2))
            result.update(teacher_cells=np.zeros((maximum, 2), np.int64),
                          teacher_agents=np.zeros(maximum, bool),
                          teacher_labels=np.zeros(maximum, np.int64),
                          teacher_logits=np.zeros(maximum, np.float32),
                          teacher_boxes=np.zeros((maximum, 10), np.float32),
                          teacher_motion=np.zeros((maximum, STEPS, 2), np.float32),
                          teacher_motion_mask=np.zeros((maximum, STEPS), bool),
                          teacher_plan=(plan @ rotation[:2, :2].T + translation[:2]).astype(np.float32))
            occupied = set()
            for i in np.argsort(-scores, kind="stable"):
                encoded = box_encoding(rotation @ centers[i] + translation, sizes[i],
                                       angles[i] + yaw(rotation), rotation[:2, :2] @ velocities[i], grid)
                if encoded is None or tuple(encoded[0]) in occupied:
                    continue
                index = len(occupied)
                occupied.add(tuple(encoded[0]))
                probability = np.clip(scores[i], 1e-5, 1 - 1e-5)
                result["teacher_cells"][index], result["teacher_boxes"][index] = encoded
                result["teacher_agents"][index] = True
                result["teacher_labels"][index] = labels[i]
                result["teacher_logits"][index] = np.log(probability / (1 - probability))
                result["teacher_motion"][index] = motion[i] @ rotation[:2, :2].T
                result["teacher_motion_mask"][index] = valid[i]
                if len(occupied) == maximum:
                    break
    return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in result.items()}
