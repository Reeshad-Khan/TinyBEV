"""nuScenes keyframes in an ego frame with x forward and y left."""
from pathlib import Path

import numpy as np
import random
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from .model import CAMERAS, CLASSES, MAP_LAYERS, STEPS, Grid

MODEL_INPUTS = ("images", "intrinsics", "camera_to_ego", "command")


def photometric_distortion(rgb):
    if random.randrange(2):
        rgb = rgb + random.uniform(-32, 32)
    early_contrast = random.randrange(2)
    if early_contrast and random.randrange(2):
        rgb = rgb * random.uniform(.5, 1.5)
    hsv = np.asarray(Image.fromarray(np.uint8(np.clip(rgb, 0, 255))).convert("HSV")).copy()
    if random.randrange(2):
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(.5, 1.5), 0, 255)
    if random.randrange(2):
        hsv[..., 0] = (hsv[..., 0].astype(int) + random.randint(-13, 13)) % 256
    rgb = np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"), dtype=np.float32)
    if not early_contrast and random.randrange(2):
        rgb *= random.uniform(.5, 1.5)
    if random.randrange(2):
        rgb = rgb[..., np.random.permutation(3)]
    return rgb


def transform(record):
    from pyquaternion import Quaternion

    matrix = Quaternion(record["rotation"]).transformation_matrix
    matrix[:3, 3] = record["translation"]
    return matrix


def inverse(matrix):
    output = np.eye(4)
    output[:3, :3] = matrix[:3, :3].T
    output[:3, 3] = -output[:3, :3] @ matrix[:3, 3]
    return output


def yaw(matrix):
    return np.arctan2(matrix[1, 0], matrix[0, 0])


def box_encoding(center, size, angle, velocity, grid):
    position = (center[:2] + grid.extent) / grid.cell
    xy = np.floor(position).astype(np.int64)
    if np.any(xy < 0) or np.any(xy >= grid.size):
        return None
    encoded = np.r_[position - xy, center[2], np.log(size),
                    np.sin(angle), np.cos(angle), velocity].astype(np.float32)
    return xy, encoded


def paint_box(canvas, center, size, angle, grid):
    width, length = size[:2]
    corners = np.array([[length, width], [length, -width],
                        [-length, -width], [-length, width]]) * .5
    c, s = np.cos(angle), np.sin(angle)
    corners = corners @ np.array([[c, s], [-s, c]]) + center[:2]
    pixels = (corners + grid.extent) / grid.cell
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in pixels], fill=1)


def rasterize_map_lines(geometry, mask):
    from nuscenes.map_expansion.map_api import NuScenesMapExplorer

    # nuScenes 1.2.0 iterates multipart geometries using the Shapely 1 API.
    if geometry.geom_type == "LineString":
        return NuScenesMapExplorer.mask_for_lines(geometry, mask)
    for part in getattr(geometry, "geoms", ()):
        rasterize_map_lines(part, mask)
    return mask


class NuScenesFrames(Dataset):
    def __init__(self, root, split, grid=Grid(), image_size=(896, 512), max_agents=128,
                 cache=None, variant="S0", augment=False):
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes

        self.root, self.grid, self.image_size = Path(root), grid, image_size
        self.max_agents, self.variant = max_agents, variant
        self.augment = augment
        self.cache = Path(cache) if cache else None
        if variant != "S0" and self.cache is None:
            raise ValueError("Distillation requires a teacher cache")
        if self.cache is not None and variant != "S0" and not self.cache.is_dir():
            raise FileNotFoundError(f"Teacher cache directory does not exist: {self.cache}")
        version = "v1.0-mini" if split.startswith("mini_") else "v1.0-trainval"
        if not (self.root / version / "sample.json").is_file():
            raise FileNotFoundError(f"Missing nuScenes metadata: {self.root / version / 'sample.json'}")
        self.nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
        names = set(create_splits_scenes()[split])
        scenes = {s["token"]: s for s in self.nusc.scene if s["name"] in names}
        self.samples = sorted((s for s in self.nusc.sample if s["scene_token"] in scenes),
                              key=lambda s: (s["scene_token"], s["timestamp"]))
        if not self.samples:
            raise ValueError(f"No samples for {split} at {root}")
        self.maps = {}
        self.locations = {token: self.nusc.get("log", s["log_token"])["location"]
                          for token, s in scenes.items()}
        for location in sorted(set(self.locations.values())):
            map_path = self.root / "maps" / "expansion" / f"{location}.json"
            if not map_path.is_file():
                raise FileNotFoundError(f"Missing nuScenes map expansion: {map_path}")

    def check_teacher_coverage(self):
        if self.variant == "S0":
            return
        available = {path.stem for path in self.cache.glob("*.npz") if path.is_file()}
        missing = [frame["token"] for frame in self.samples if frame["token"] not in available]
        if missing:
            examples = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Teacher cache {self.cache} is missing {len(missing)} of {len(self.samples)} frames "
                f"for {self.variant}. First missing tokens: {examples}"
            )

    def __len__(self):
        return len(self.samples)

    def pose(self, frame):
        reference = self.nusc.get("sample_data", frame["data"]["LIDAR_TOP"])
        return transform(self.nusc.get("ego_pose", reference["ego_pose_token"]))

    def lidar_to_world(self, frame):
        reference = self.nusc.get("sample_data", frame["data"]["LIDAR_TOP"])
        return self.pose(frame) @ transform(
            self.nusc.get("calibrated_sensor", reference["calibrated_sensor_token"]))

    def camera_input(self, frame, world_to_ego):
        images, intrinsics, extrinsics = [], [], []
        width, height = self.image_size
        for name in CAMERAS:
            view = self.nusc.get("sample_data", frame["data"][name])
            calibrated = self.nusc.get("calibrated_sensor", view["calibrated_sensor_token"])
            pose = transform(self.nusc.get("ego_pose", view["ego_pose_token"]))
            with Image.open(self.root / view["filename"]) as image:
                intrinsic = np.asarray(calibrated["camera_intrinsic"], dtype=np.float32)
                intrinsic[0] *= width / image.width
                intrinsic[1] *= height / image.height
                rgb = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
            rgb = rgb.astype(np.float32)
            if self.augment:
                rgb = photometric_distortion(rgb)
            # RGB-equivalent mean of UniAD's BGR preprocessing.
            images.append(((rgb - [123.675, 116.28, 103.53]) / 255).transpose(2, 0, 1))
            intrinsics.append(intrinsic)
            extrinsics.append(world_to_ego @ pose @ transform(calibrated))
        return dict(images=np.asarray(images, dtype=np.float32),
                    intrinsics=np.asarray(intrinsics, dtype=np.float32),
                    camera_to_ego=np.asarray(extrinsics, dtype=np.float32))

    def map_target(self, frame, ego_to_world):
        from nuscenes.map_expansion.map_api import NuScenesMap

        location = self.locations[frame["scene_token"]]
        if location not in self.maps:
            nusc_map = NuScenesMap(dataroot=str(self.root), map_name=location)
            nusc_map.explorer.mask_for_lines = rasterize_map_lines
            self.maps[location] = nusc_map
        x, y = ego_to_world[:2, 3]
        side = 2 * self.grid.extent
        masks = self.maps[location].get_map_mask(
            (x, y, side, side), np.rad2deg(yaw(ego_to_world)),
            list(MAP_LAYERS), (self.grid.size, self.grid.size))
        return masks.astype(np.float32)

    def annotation(self, token, world_to_ego):
        record = self.nusc.get("sample_annotation", token)
        in_ego = world_to_ego @ transform(record)
        return record, in_ego[:3, 3], np.asarray(record["size"]), yaw(in_ego)

    def targets(self, frame, world_to_ego):
        from nuscenes.eval.detection.utils import category_to_detection_name

        grid, maximum = self.grid, self.max_agents
        n = grid.size
        target = dict(heatmap=np.zeros((len(CLASSES), n, n), np.float32),
                      agent_cells=np.zeros((maximum, 2), np.int64),
                      agents=np.zeros(maximum, bool), boxes=np.zeros((maximum, 10), np.float32),
                      agent_labels=np.zeros(maximum, np.int64),
                      agent_centers=np.zeros((maximum, 2), np.float32),
                      motion=np.zeros((maximum, STEPS, 2), np.float32),
                      motion_mask=np.zeros((maximum, STEPS), bool),
                      plan=np.zeros((STEPS, 2), np.float32), plan_mask=np.zeros(STEPS, bool),
                      occ=np.zeros((STEPS, n, n), np.float32))
        future, current = [], frame
        while current["next"] and len(future) < STEPS:
            current = self.nusc.get("sample", current["next"])
            future.append(current)

        command = 2
        if future:
            last = inverse(self.lidar_to_world(frame)) @ self.lidar_to_world(future[-1])
            command = 0 if last[0, 3] >= 2 else 1 if last[0, 3] <= -2 else 2
        target["command"] = np.eye(3, dtype=np.float32)[command]
        future_positions = []
        for step, sample in enumerate(future):
            ego = world_to_ego @ self.pose(sample)
            target["plan"][step] = ego[:2, 3]
            target["plan_mask"][step] = True
            canvas = Image.new("L", (n, n))
            positions = {}
            for token in sample["anns"]:
                record, center, size, angle = self.annotation(token, world_to_ego)
                category = category_to_detection_name(record["category_name"])
                if category in CLASSES[:-2]:
                    paint_box(canvas, center, size, angle, grid)
                    positions[record["instance_token"]] = center[:2]
            target["occ"][step] = np.asarray(canvas)
            future_positions.append(positions)

        candidates = []
        for token in frame["anns"]:
            record, center, size, angle = self.annotation(token, world_to_ego)
            category = category_to_detection_name(record["category_name"])
            if category is None or record["num_lidar_pts"] + record["num_radar_pts"] == 0:
                continue
            velocity = np.nan_to_num(self.nusc.box_velocity(token))
            velocity = (world_to_ego[:3, :3] @ velocity)[:2]
            encoded = box_encoding(center, size, angle, velocity, grid)
            if encoded is not None:
                candidates.append((np.linalg.norm(center[:2]), record, center,
                                   CLASSES.index(category), encoded))
        occupied = set()
        for _, record, center, label, (cell, box) in sorted(candidates, key=lambda item: item[0]):
            index = len(occupied)
            if index == maximum:
                break
            x, y = cell
            if (x, y) in occupied:
                continue
            occupied.add((x, y))
            target["heatmap"][label, y, x] = 1
            target["agent_cells"][index] = cell
            target["agent_labels"][index] = label
            target["agent_centers"][index] = center[:2]
            target["agents"][index] = True
            target["boxes"][index] = box
            for step, positions in enumerate(future_positions):
                position = positions.get(record["instance_token"])
                if position is not None:
                    target["motion"][index, step] = position - center[:2]
                    target["motion_mask"][index, step] = True
        return target

    def __getitem__(self, index):
        frame = self.samples[index]
        ego_to_world = self.pose(frame)
        world_to_ego = inverse(ego_to_world)
        target = self.targets(frame, world_to_ego)
        target.update(self.camera_input(frame, world_to_ego))
        target["map"] = self.map_target(frame, ego_to_world)
        target["salient"] = (target["heatmap"].any(0) | target["occ"].any(0)
                             | target["map"].any(0))
        target = {key: torch.from_numpy(value) for key, value in target.items()}
        target["token"] = frame["token"]
        target["scene"] = frame["scene_token"]
        target["timestamp"] = frame["timestamp"]
        target["ego_to_world"] = torch.from_numpy(ego_to_world)
        if self.cache is not None and self.variant != "S0":
            from .teacher import load_teacher
            target.update(load_teacher(self.cache / (frame["token"] + ".npz"),
                                       self.grid, self.variant, self.max_agents))
        return target
