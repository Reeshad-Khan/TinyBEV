"""Render paired TinyBEV and UniAD outputs from the root of a UniAD checkout."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tinybev.data import MODEL_INPUTS, NuScenesFrames, inverse, transform, yaw
from tinybev.model import TinyBEV


def agents(boxes, plan, command, threshold):
    from tools.analysis_tools.visualize.utils import AgentPredictionData

    predictions = []
    # UniAD orders construction vehicles, barriers, and some agents differently.
    labels = (0, 1, 3, 4, 2, 8, 6, 7, 9, 5)
    for index in np.flatnonzero(boxes["scores"] >= threshold):
        offset = boxes["motion"][index, boxes["motion_valid"][index]]
        predictions.append(AgentPredictionData(
            float(boxes["scores"][index]), labels[int(boxes["labels"][index])],
            boxes["center"][index], boxes["size"][index],
            -boxes["yaw"][index] - np.pi / 2, np.zeros(2), offset[None],
            np.array([0.]), pred_track_id=int(index) + 1,
        ))
    ego = AgentPredictionData(
        1., 0, np.array([0., 0., -.42]), np.array([1.85, 4.08, 1.56]),
        -np.pi / 2, np.zeros(2), plan, 1, pred_track_id=-1,
        is_sdc=True, command=command,
    )
    return predictions, ego


def render_strip(dataset, token, predictions, ego):
    from tools.analysis_tools.visualize.render.bev_render import BEVRender
    from tools.analysis_tools.visualize.render.cam_render import CameraRender

    def pixels(renderer):
        renderer.fig.canvas.draw()
        rgb = np.asarray(renderer.fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(renderer.fig)
        return rgb

    camera = CameraRender()
    camera.reset_canvas(dx=2, dy=3, tight_layout=True)
    camera.render_image_data(token, dataset.nusc)
    camera.render_pred_track_bbox(predictions + [ego], token, dataset.nusc)
    camera.render_pred_traj(predictions + [ego], token, dataset.nusc, render_sdc=True)
    camera.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    mosaic = pixels(camera)

    bev = BEVRender()
    bev.reset_canvas()
    bev.set_plot_cfg()
    bev.render_pred_box_data(predictions)
    bev.render_pred_traj(predictions, top_k=1)
    bev.render_pred_box_data([ego])
    bev.render_planning_data(ego, show_command=True)
    bev.render_sdc_car()
    bev.render_legend()
    bev.axes.set_axis_off()
    bev.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return np.concatenate((mosaic, pixels(bev)), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--token", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score", type=float, default=.25)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not all((Path("sources") / name).is_file() for name in ("sdc_car.png", "legend.png")):
        parser.error("Run from the root of the UniAD checkout so the renderer can load its assets")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]
    dataset = NuScenesFrames(args.data, args.split, image_size=settings["image_size"])
    index = next((i for i, frame in enumerate(dataset.samples) if frame["token"] == args.token), None)
    if index is None:
        parser.error(f"{args.token} is absent from {args.split}")
    frame = dataset[index]
    try:
        with np.load(args.cache / f"{args.token}.npz", allow_pickle=False) as archive:
            teacher = {name: archive[name] for name in
                       ("center", "size", "yaw", "score", "label", "motion", "motion_valid", "plan",
                        "lidar_to_ego")}
    except FileNotFoundError as error:
        parser.error(str(error))
    teacher["scores"], teacher["labels"] = teacher.pop("score"), teacher.pop("label")
    lidar_to_ego = teacher.pop("lidar_to_ego")
    sample = dataset.samples[index]
    lidar = dataset.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    calibration = dataset.nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
    np.testing.assert_allclose(lidar_to_ego, transform(calibration), atol=1e-5)

    model = TinyBEV().to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    with torch.inference_mode():
        forward = model(*(frame[key][None].to(args.device) for key in MODEL_INPUTS))
        decoded = model.decode(forward)
        student = {key: value[0].cpu().numpy() for key, value in decoded.items()}
        plan = forward["plan"][0].cpu().numpy()
    ego_to_lidar = inverse(lidar_to_ego)
    rotation, translation = ego_to_lidar[:3, :3], ego_to_lidar[:3, 3]
    motion = student["motion"] - student["center"][:, None, :2]
    student["motion"] = motion @ rotation[:2, :2].T
    student["motion_valid"] = np.ones(motion.shape[:2], dtype=bool)
    student["center"] = student["center"] @ rotation.T + translation
    student["yaw"] += yaw(rotation)
    student["plan"] = plan @ rotation[:2, :2].T + translation[:2]

    command = int(frame["command"].argmax())
    strips = [render_strip(dataset, args.token, *agents(record, record["plan"], command, args.score))
              for record in (teacher, student)]
    captions = ("(a) UniAD (Teacher)", f"(b) TinyBEV (Student, {settings['variant']})")
    fig, axes = plt.subplots(2, 1, figsize=(6.65, 4), layout="constrained")
    for ax, strip, caption in zip(axes, strips, captions):
        ax.imshow(strip, interpolation="antialiased")
        ax.set_axis_off()
        ax.text(0, -.025, caption, transform=ax.transAxes, fontsize=7,
                fontfamily="serif", weight="bold", va="top")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
