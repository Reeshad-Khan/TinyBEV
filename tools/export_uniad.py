"""Run from a UniAD 1.x checkout in the upstream environment (one GPU)."""
import argparse
import copy
import hashlib
import importlib
import json
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
import torch

# Both class orders follow the released nuScenes configs.
TEACHER_CLASSES = ("car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
                   "motorcycle", "bicycle", "pedestrian", "traffic_cone")
STUDENT_CLASSES = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian",
                   "motorcycle", "bicycle", "traffic_cone", "barrier")


def array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def lidar_calibrations(dataset):
    return {record["token"]: record for record in dataset.data_infos}


def export_indices(records, output, resume):
    if not resume:
        return list(range(len(records)))
    pending = []
    for _, scene in groupby(enumerate(records), key=lambda item: item[1]["scene_token"]):
        frames = list(scene)
        if any(not (output / f"{record['token']}.npz").is_file() for _, record in frames):
            pending.extend(index for index, _ in frames)
    return pending


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def box_arrays(boxes):
    return array(boxes.gravity_center), array(boxes.dims), array(boxes.yaw), array(boxes.tensor[:, 7:9])


def export_result(result, captured, info, target, class_order):
    boxes, scores, labels = (result[key + "_det"] if key + "_det" in result else result[key]
                             for key in ("boxes_3d", "scores_3d", "labels_3d"))
    center, dimensions, box_yaw, velocity = box_arrays(boxes)
    # UniAD's nuScenes serializer maps the legacy MMDetection3D yaw convention.
    yaw = -box_yaw - np.pi / 2
    labels = array(labels)
    offsets = np.zeros((len(center), 6, 2), dtype=np.float32)
    valid = np.zeros((len(center), 6), dtype=bool)
    if "traj" in result and len(captured["track_center"]) and len(center):
        trajectory = array(result["traj"])
        probabilities = array(result["traj_scores"])
        tracks = len(captured["track_center"])
        if len(trajectory) not in (tracks, tracks + 1):
            raise ValueError(f"Trajectory count {len(trajectory)} disagrees with {tracks} tracks")
        mode = probabilities[:tracks].argmax(-1)
        recurrent_xy = trajectory[np.arange(tracks), mode, :6, :2]
        distance = np.linalg.norm(center[:, None, :2] - captured["track_center"][None, :, :2], axis=-1)
        eligible = (distance < .1) & (labels[:, None] == captured["track_label"][None, :])
        detections, matched = linear_sum_assignment(np.where(eligible, distance, 1e6))
        keep = eligible[detections, matched]
        offsets[detections[keep]] = recurrent_xy[matched[keep]]
        valid[detections[keep]] = True

    lidar_to_ego = Quaternion(info["lidar2ego_rotation"]).transformation_matrix
    lidar_to_ego[:3, 3] = info["lidar2ego_translation"]
    bev = array(captured["bev"])
    if bev.shape != (40000, 1, 256):
        raise ValueError(f"Expected stage-2 UniAD BEV [40000, 1, 256], got {bev.shape}")
    plan = array(result["planning"]["result_planning"]["sdc_traj"]).reshape(-1, 2)
    if len(plan) != 6:
        raise ValueError(f"Expected six planning waypoints, got {len(plan)}")
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, bev=bev[:, 0].T.reshape(256, 200, 200).astype(np.float16),
                 lidar_to_ego=lidar_to_ego.astype(np.float32),
                 center=center, size=dimensions, yaw=yaw, velocity=velocity,
                 label=np.asarray([STUDENT_CLASSES.index(class_order[i]) for i in labels],
                                  dtype=np.int64),
                 score=array(scores), motion=offsets, motion_valid=valid, plan=plan)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--annotations", type=Path, help="Override data.test.ann_file for training exports")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Export only the first N samples for a smoke run")
    parser.add_argument("--resume", action="store_true", help="Skip complete scenes in this cache")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    config = Config.fromfile(str(args.config))
    if config.get("plugin", False):
        importlib.import_module(config.plugin_dir.strip("/").replace("/", "."))
    if config.get("custom_imports"):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**config.custom_imports)
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    test = copy.deepcopy(config.data.test)
    test.pop("samples_per_gpu", None)
    test.test_mode = True
    if args.annotations:
        test.ann_file = str(args.annotations)
    dataset = build_dataset(test)
    infos = lidar_calibrations(dataset)
    records = dataset.data_infos[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    signature = {"config": fingerprint(args.config), "checkpoint": fingerprint(args.checkpoint),
                 "annotations": fingerprint(test.ann_file), "torch": str(torch.__version__)}
    manifest = args.output / "export.json"
    if manifest.is_file():
        if json.loads(manifest.read_text()) != signature:
            parser.error(f"{args.output}: exporter inputs differ from the existing cache")
    elif next(args.output.glob("*.npz"), None) is not None:
        parser.error(f"{args.output}: existing cache has no export.json provenance")
    else:
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(signature, indent=2) + "\n")
        temporary.replace(manifest)
    indices = export_indices(records, args.output, args.resume)
    print(json.dumps({"requested": len(records), "pending": len(indices)}), flush=True)
    if not indices:
        return
    loader = build_dataloader(torch.utils.data.Subset(dataset, indices), samples_per_gpu=1,
                              workers_per_gpu=args.workers, dist=False, shuffle=False)
    config.model.pretrained = None
    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.requires_grad_(False)
    model.eval()
    class_order = tuple(getattr(dataset, "CLASSES", TEACHER_CLASSES))
    if set(class_order) != set(STUDENT_CLASSES):
        raise ValueError(f"Unexpected UniAD class order: {class_order}")
    captured = {}
    original = model.simple_test_track

    def capture(*inputs, **kwargs):
        results = original(*inputs, **kwargs)
        track = results[0]
        captured["bev"] = track["bev_embed"]
        boxes, _, labels, *_ = track["track_bbox_results"][0]
        captured["track_center"] = array(boxes.gravity_center).copy()
        captured["track_label"] = array(labels).copy()
        return results

    model.simple_test_track = capture
    parallel = MMDataParallel(model.cuda(), device_ids=[0])
    with torch.no_grad():
        for index, (record_index, batch) in enumerate(zip(indices, loader), start=1):
            result, = parallel(return_loss=False, rescale=True, **batch)
            token = result["token"]
            if token != records[record_index]["token"]:
                raise ValueError(f"Expected token {records[record_index]['token']}, got {token}")
            export_result(result, captured, infos[token], args.output / f"{token}.npz", class_order)
            if index % 100 == 0 or index == len(indices):
                print(json.dumps({"cached": len(records) - len(indices) + index,
                                  "requested": len(records)}), flush=True)


if __name__ == "__main__":
    main()
