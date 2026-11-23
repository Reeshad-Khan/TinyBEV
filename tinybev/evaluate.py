"""Generate nuScenes submissions and measure open-loop student performance."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from torch.utils.data import DataLoader

from .data import MODEL_INPUTS, NuScenesFrames, paint_box, yaw
from .model import CLASSES, MAP_LAYERS, STEPS, Grid, TinyBEV

TRACK_CLASSES = ("bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck")
META = dict(use_camera=True, use_lidar=False, use_radar=False, use_map=False,
            use_external=False)


def associate(distance, threshold):
    if not distance.size:
        return []
    rows, columns = linear_sum_assignment(np.where(distance <= threshold, distance, 1e6))
    return [(i, j) for i, j in zip(rows, columns) if distance[i, j] <= threshold]


def write_json(path, document):
    stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(document, stream, indent=2 if path.name == "metrics.json" else None,
                      allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Tracker:
    """Class-consistent global center association using predicted velocity."""
    def __init__(self):
        self.scene, self.previous = None, []
        self.next_id = 0

    def update(self, scene, timestamp, detections):
        if scene != self.scene:
            self.scene, self.previous = scene, []
        current = [d for d in detections if d["detection_name"] in TRACK_CLASSES]
        identifiers = {}
        if self.previous and current:
            elapsed = (timestamp - self.timestamp) / 1e6
            projected = [np.asarray(d["translation"][:2]) + elapsed * np.asarray(d["velocity"])
                         for d in self.previous]
            distance = cdist(projected, [d["translation"][:2] for d in current])
            for i, previous in enumerate(self.previous):
                for j, detection in enumerate(current):
                    if previous["tracking_name"] != detection["detection_name"]:
                        distance[i, j] = np.inf
            identifiers = {j: self.previous[i]["tracking_id"] for i, j in associate(distance, 2)}
        tracked = []
        for j, detection in enumerate(current):
            identifier = identifiers.get(j)
            if identifier is None:
                identifier = str(self.next_id)
                self.next_id += 1
            fields = {k: v for k, v in detection.items() if k not in
                      ("detection_name", "detection_score", "attribute_name")}
            tracked.append(dict(**fields, tracking_id=identifier,
                                tracking_name=detection["detection_name"],
                                tracking_score=detection["detection_score"]))
        self.previous, self.timestamp = tracked, timestamp
        return tracked


def attribute(name, velocity):
    moving = np.linalg.norm(velocity) > .2
    if name in ("car", "truck", "trailer", "construction_vehicle", "bus"):
        return "vehicle.moving" if moving else ("vehicle.stopped" if name == "bus" else "vehicle.parked")
    if name == "pedestrian":
        return "pedestrian.moving" if moving else "pedestrian.standing"
    if name in ("bicycle", "motorcycle"):
        return "cycle.with_rider" if moving else "cycle.without_rider"
    return ""


def submission_boxes(prediction, keep, pose, token):
    from pyquaternion import Quaternion

    orientation = Quaternion(matrix=pose[:3, :3])
    detections = []
    for i in keep:
        name = CLASSES[prediction["labels"][i]]
        velocity = (pose[:3, :3] @ np.r_[prediction["velocity"][i], 0])[:2]
        center = pose[:3, :3] @ prediction["center"][i] + pose[:3, 3]
        rotation = orientation * Quaternion(axis=[0, 0, 1], radians=float(prediction["yaw"][i]))
        detections.append(dict(sample_token=token, translation=center.tolist(),
                               size=prediction["size"][i].tolist(), rotation=rotation.elements.tolist(),
                               velocity=velocity.tolist(), detection_name=name,
                               detection_score=float(prediction["scores"][i]),
                               attribute_name=attribute(name, velocity)))
    return detections


def footprint_collision(path, occupied, step, grid):
    start = path[step - 1] if step else np.zeros(2)
    displacement = path[step] - start
    angle = np.arctan2(displacement[1], displacement[0]) if np.linalg.norm(displacement) > .05 else 0
    canvas = Image.new("L", (grid.size, grid.size))
    paint_box(canvas, path[step], (1.73, 4.08), angle, grid)
    return (np.asarray(canvas).astype(bool) & occupied[step]).any()


class OpenLoopMetrics:
    def __init__(self):
        self.plan_error = np.zeros(STEPS)
        self.steps = np.zeros(STEPS, int)
        self.collisions = np.zeros(STEPS, int)
        self.collision_steps = np.zeros(STEPS, int)
        self.forecast_errors = []
        self.possible = 0
        self.map_intersection = np.zeros(len(MAP_LAYERS), int)
        self.map_union = np.zeros(len(MAP_LAYERS), int)
        self.occ_intersection = np.zeros(STEPS, int)
        self.occ_union = np.zeros(STEPS, int)

    def update(self, prediction, output, batch, keep):
        plan = output["plan"][0].cpu().numpy()
        map_logits = output["map"][0].cpu().numpy()
        occ_logits = output["occ"][0].cpu().numpy()
        if not all(np.isfinite(values).all() for values in (plan, map_logits, occ_logits)):
            raise FloatingPointError(f"Nonfinite planning, map or occupancy output for {batch['token'][0]}")
        gt_plan = batch["plan"][0].numpy()
        valid = batch["plan_mask"][0].numpy()
        occ = batch["occ"][0].numpy().astype(bool)
        self.plan_error += np.linalg.norm(plan - gt_plan, axis=-1) * valid
        self.steps += valid
        for step in np.flatnonzero(valid):
            if not footprint_collision(gt_plan, occ, step, Grid()):
                self.collision_steps[step] += 1
                self.collisions[step] += footprint_collision(plan, occ, step, Grid())
        pred_occ = occ_logits > 0
        self.occ_intersection += (pred_occ & occ).sum((-1, -2)) * valid
        self.occ_union += (pred_occ | occ).sum((-1, -2)) * valid
        gt_map = batch["map"][0].numpy().astype(bool)
        pred_map = map_logits > 0
        self.map_intersection += (pred_map & gt_map).sum((-1, -2))
        self.map_union += (pred_map | gt_map).sum((-1, -2))
        center = batch["agent_centers"][0].numpy()
        gt_label = batch["agent_labels"][0].numpy()
        motion_mask = batch["motion_mask"][0].numpy()
        agents = np.flatnonzero(batch["agents"][0].numpy() & motion_mask.any(-1))
        self.possible += len(agents)
        distance = cdist(prediction["center"][keep, :2], center[agents])
        same_class = prediction["labels"][keep, None] == gt_label[None, agents]
        for i, j in associate(np.where(same_class, distance, np.inf), 2):
            agent = agents[j]
            target = batch["motion"][0, agent].numpy() + center[agent]
            error = np.linalg.norm(prediction["motion"][keep[i]] - target, axis=-1)
            self.forecast_errors.append(float(error[motion_mask[agent]].mean()))

    def report(self):
        return dict(
            l2_3s=float(self.plan_error[-1] / self.steps[-1]) if self.steps[-1] else None,
            collision_pct=100 * float(self.collisions.sum() / self.collision_steps.sum())
            if self.collision_steps.sum() else None,
            minade=float(np.mean(self.forecast_errors)) if self.forecast_errors else None,
            forecast_matches=len(self.forecast_errors), forecast_available=self.possible,
            map_iou={name: float(i / u) if u else None for name, i, u in
                     zip(MAP_LAYERS, self.map_intersection, self.map_union)},
            occupancy_iou=[float(i / u) if u else None for i, u in
                           zip(self.occ_intersection, self.occ_union)],
        )


def official(dataset, output, split):
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.tracking.evaluate import TrackingEval

    detector = NuScenesEval(dataset.nusc, config_factory("detection_cvpr_2019"),
                            str(output / "detection.json"), split, str(output / "det_metrics"))
    detection = detector.main(render_curves=False)
    tracker = TrackingEval(config_factory("tracking_nips_2019"), str(output / "tracking.json"),
                           split, str(output / "track_metrics"), dataset.nusc.version,
                           str(dataset.root))
    tracking = tracker.main(render_curves=False)
    return dict(map=100 * detection["mean_ap"], nds=100 * detection["nd_score"],
                amota=100 * tracking["amota"], amotp=tracking["amotp"])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "mini_val"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score", type=float, default=.05)
    parser.add_argument("--skip-official", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.score <= 1 or args.workers < 0:
        parser.error("score must be between 0 and 1 and worker count must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu for CPU evaluation.")
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    model = TinyBEV().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    dataset = NuScenesFrames(args.data, args.split, image_size=checkpoint["settings"]["image_size"])
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers)
    args.output.mkdir(parents=True, exist_ok=True)
    detector_results, tracker_results = {}, {}
    metrics, tracker = OpenLoopMetrics(), Tracker()
    timings = []
    for batch in loader:
        inputs = [batch[key].to(device) for key in MODEL_INPUTS]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = model(*inputs)
        decoded = model.decode(output)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        prediction = {key: value[0].cpu().numpy() for key, value in decoded.items()}
        token = batch["token"][0]
        invalid = [key for key, values in prediction.items() if not np.isfinite(values).all()]
        if invalid:
            raise FloatingPointError(f"Nonfinite output for {token}: {', '.join(invalid)}")
        keep = np.flatnonzero(prediction["scores"] >= args.score)
        detections = submission_boxes(prediction, keep, batch["ego_to_world"][0].numpy(), token)
        detector_results[token] = detections
        tracker_results[token] = tracker.update(batch["scene"][0], int(batch["timestamp"][0]), detections)
        metrics.update(prediction, output, batch, keep)
    for label, results in (("detection", detector_results), ("tracking", tracker_results)):
        write_json(args.output / f"{label}.json", dict(meta=META, results=results))
    summary = metrics.report()
    summary.update(fps=1 / float(np.mean(timings[10:] or timings)),
                   parameters=sum(p.numel() for p in model.parameters()),
                   variant=checkpoint["settings"]["variant"], split=args.split, device=str(device))
    summary.update(checkpoint=str(Path(args.checkpoint).resolve()),
                   epoch=checkpoint["epoch"] + 1, training_settings=checkpoint["settings"],
                   score_threshold=args.score, official_metrics=not args.skip_official,
                   torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    if not args.skip_official:
        summary.update(official(dataset, args.output, args.split))
    summary = {key: None if isinstance(value, float) and not np.isfinite(value) else value
               for key, value in summary.items()}
    write_json(args.output / "metrics.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
