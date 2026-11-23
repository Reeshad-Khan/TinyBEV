"""Render the experimental figures from TinyBEV metrics and predictions."""
import argparse
import json
import math
from pathlib import Path

from .compare import TABLE_2


ABLATION_METRICS = (
    ("map", "mAP ↑", 1),
    ("minade", "minADE ↓", 2),
    ("l2_3s", "L2@3s ↓", 2),
    ("collision_pct", "Collision ↓", 2),
)
VARIANT_LABELS = ("S1: Feature KD", "S2: Output KD", "S3: Full KD")
COLORS = ("#4c78a8", "#f58518", "#54a24b")


def ablation_rows(paths):
    rows, splits = {}, set()
    for path in paths:
        record = json.loads(path.read_text())
        variant = record["variant"]
        if variant not in TABLE_2 or variant in rows:
            raise ValueError(f"{path}: expected one metrics file for each of S0–S3")
        splits.add(record["split"])
        row = [record.get(key) for key, _, _ in ABLATION_METRICS]
        if any(value is None or not math.isfinite(value) for value in row):
            raise ValueError(f"{path}: all four Figure 2 metrics must be finite")
        rows[variant] = row
    if rows.keys() != TABLE_2.keys() or len(splits) != 1:
        raise ValueError("Provide S0–S3 metrics from the same evaluation split")
    return [rows[key] for key in TABLE_2], splits.pop()


def ablation_figure(paths, output):
    import matplotlib.pyplot as plt
    import numpy as np

    rows, split = ablation_rows(paths)
    values = np.asarray(rows)
    baseline = values[0]
    changes = np.divide(
        (values[1:] - baseline) * np.array([1, -1, -1, -1]) * 100,
        baseline, out=np.full_like(values[1:], np.nan, dtype=float), where=baseline != 0,
    )
    fig = plt.figure(figsize=(3.33, 2.60), facecolor="white")
    table_ax = fig.add_axes((0, .61, 1, .39))
    table_ax.set_axis_off()
    table_labels = ("mAP↑", "minADE↓", "L2@3s↓", "Collision↓")
    chart_labels = ("mAP (↑)", "minADE (↓)", "L2@3s (↓)", "Collision (↓)")
    table = table_ax.table(
        cellText=[[variant, *[f"{value:.{digits}f}" for value, (_, _, digits)
                              in zip(row, ABLATION_METRICS)]]
                  for variant, row in zip(TABLE_2, rows)],
        colLabels=["Variant", *table_labels], cellLoc="center", bbox=(0, 0, 1, 1),
        colWidths=(.18, .15, .215, .21, .245),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.visible_edges = "TB" if row == 0 else "B" if row == 4 else ""
        cell.set_linewidth(.55)
        label = cell.get_text()
        label.set_fontfamily("serif")
        if column == 0:
            label.set_ha("left")
        if row == 0:
            label.set_fontsize(8.5)
        if row == 0 or (row == 4 and column):
            label.set_weight("bold")
    chart = fig.add_axes((.135, .075, .78, .41), facecolor="#fcfcfc")
    x = np.arange(4)
    for index, (row, label, color) in enumerate(zip(changes, VARIANT_LABELS, COLORS)):
        bars = chart.bar(x + (index - 1) * .25, np.nan_to_num(row), .25,
                         label=label, color=color, edgecolor="#454545", linewidth=.25)
        chart.bar_label(bars, labels=[f"{v:.1f}%" if np.isfinite(v) else "N/A" for v in row],
                        padding=1.5, fontsize=3.6)
    chart.set_xticks(x, chart_labels)
    chart.set_ylabel("Improvement vs. S0 baseline (%)", fontsize=4.8, fontfamily="serif")
    chart.tick_params(axis="both", labelsize=4.4, length=0, pad=2)
    chart.set_axisbelow(True)
    chart.grid(color="#dedede", linewidth=.35)
    chart.axhline(0, color="#bababa", linewidth=.35)
    for spine in chart.spines.values():
        spine.set_visible(False)
    finite = changes[np.isfinite(changes)]
    lo = min(float(finite.min()), 0) if finite.size else 0
    hi = max(float(finite.max()), 0) if finite.size else 0
    chart.set_ylim(0 if lo == 0 else lo * 1.2, max(60, hi * 1.15))
    chart.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=3,
                 prop={"family": "serif", "size": 4.2}, frameon=False)
    if split != "val":
        fig.text(.995, .55, split, ha="right", va="center", fontsize=5, color=".4")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def box_corners(center, size, yaw):
    import numpy as np

    w, length, height = size
    lower = np.array([[length, w], [length, -w], [-length, -w], [-length, w]]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    xy = lower @ np.array([[c, s], [-s, c]])
    return center + np.column_stack((np.tile(xy, (2, 1)), np.repeat([-height / 2, height / 2], 4)))


def qualitative_figure(data, split, token, checkpoint, cache, device, score, output):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from .data import MODEL_INPUTS, NuScenesFrames, inverse, transform, yaw
    from .model import CAMERAS, TinyBEV

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    settings = saved["settings"]
    dataset = NuScenesFrames(data, split, image_size=settings["image_size"])
    try:
        index = next(i for i, sample in enumerate(dataset.samples) if sample["token"] == token)
    except StopIteration as exc:
        raise ValueError(f"{token} is absent from {split}") from exc
    frame = dataset[index]
    model = TinyBEV().to(device).eval()
    model.load_state_dict(saved["model"])
    with torch.inference_mode():
        forward = model(*(frame[key][None].to(device) for key in MODEL_INPUTS))
        prediction = model.decode(forward)
        prediction["plan"] = forward["plan"]
    student = {key: value[0].cpu().numpy() for key, value in prediction.items()}
    with np.load(cache / f"{token}.npz", allow_pickle=False) as raw:
        pose = raw["lidar_to_ego"]
        rotation, translation = pose[:3, :3], pose[:3, 3]
        center = raw["center"] @ rotation.T + translation
        teacher = dict(center=center, size=raw["size"], yaw=raw["yaw"] + yaw(rotation),
                       scores=raw["score"], labels=raw["label"],
                       motion=raw["motion"] @ rotation[:2, :2].T + center[:, None, :2],
                       motion_valid=raw["motion_valid"],
                       plan=raw["plan"] @ rotation[:2, :2].T + translation[:2])
    sample = dataset.samples[index]
    lidar = dataset.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    calibration = dataset.nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
    np.testing.assert_allclose(pose, transform(calibration), atol=1e-5)
    student["motion_valid"] = np.ones(student["motion"].shape[:2], dtype=bool)

    from matplotlib.colors import LinearSegmentedColormap

    from .model import Grid

    fig = plt.figure(figsize=(6.65, 4.25), facecolor="white")
    layout = fig.add_gridspec(
        6, 4, left=.005, right=.995, bottom=.008, top=.997,
        wspace=0, hspace=0, width_ratios=(1, 1, 1, 1.12),
        height_ratios=(1, 1, .22, 1, 1, .22),
    )
    views = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT")
    images = ((frame["images"].permute(0, 2, 3, 1).numpy() * 255)
              + np.array([123.675, 116.28, 103.53])).clip(0, 255).astype(np.uint8)
    edges = [(i, (i + 1) % 4) for i in range(4)]
    edges += [(i + 4, j + 4) for i, j in edges] + [(i, i + 4) for i in range(4)]
    palette = ("#00a4de", "#ffbe00", "#9ce300", "#ff831e", "#6b42ff",
               "#ec32d2", "#00c9a0", "#2e53e5", "#f04971", "#975d25")
    trajectories = plt.get_cmap("winter")
    commands = ("TURN RIGHT", "TURN LEFT", "KEEP FORWARD")
    road = LinearSegmentedColormap.from_list("road", ("#f9fdfa", "#e9f2e5", "#cedfca"))
    map_layers = frame["map"].numpy()
    road_background = .45 * map_layers[0] + .30 * map_layers[1:3].max(axis=0)
    extent = Grid().extent

    def project(ax, camera, points, segments, color, style="-", linewidth=.45):
        camera_pose = inverse(frame["camera_to_ego"][camera].numpy())
        points = points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
        pixels = points @ frame["intrinsics"][camera].numpy().T
        for a, b in segments:
            if min(points[a, 2], points[b, 2]) > .1:
                xy = pixels[[a, b], :2] / pixels[[a, b], 2:]
                ax.plot(xy[:, 0], xy[:, 1], linestyle=style, color=color, linewidth=linewidth)

    for row, (name, pred) in enumerate((("(a) UniAD (Teacher)", teacher),
                                        (f"(b) TinyBEV (Student, {settings['variant']})", student))):
        start = row * 3
        bev = fig.add_subplot(layout[start:start + 2, 3])
        bev.imshow(road_background, extent=(-extent, extent, -extent, extent), origin="lower",
                   cmap=road, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        bev.set(xlim=(24, -24), ylim=(-8, 48), xticks=[], yticks=[])
        bev.set_axis_off()
        axes = {}
        for column, view in enumerate(views):
            ax = fig.add_subplot(layout[start + column // 3, column % 3])
            camera = CAMERAS.index(view)
            ax.imshow(images[camera], aspect="auto")
            ax.text(.025, .97, view, transform=ax.transAxes, fontsize=3.3, va="top", color="black")
            ax.set_axis_off()
            ax.set_autoscale_on(False)
            axes[view] = ax
        for i in np.flatnonzero(pred["scores"] >= score):
            corners = box_corners(pred["center"][i], pred["size"][i], pred["yaw"][i])
            color = palette[int(pred["labels"][i])]
            footprint = corners[[0, 1, 2, 3, 0], :2]
            bev.fill(footprint[:, 1], footprint[:, 0], facecolor=color, alpha=.32, linewidth=0)
            bev.plot(footprint[:, 1], footprint[:, 0], color=color, linewidth=.55)
            path = np.vstack((pred["center"][i, :2], pred["motion"][i]))
            valid = np.r_[True, pred["motion_valid"][i]]
            for step in range(len(path) - 1):
                if valid[step:step + 2].all():
                    segment = path[step:step + 2]
                    bev.plot(segment[:, 1], segment[:, 0], color=trajectories(step / 5), linewidth=.55)
            for view, ax in axes.items():
                project(ax, CAMERAS.index(view), corners, edges, color)
        plan = np.vstack((np.zeros((1, 2)), pred["plan"]))
        bev.plot(plan[:, 1], plan[:, 0], ":o", color="#f78c22", markersize=1.2, linewidth=.8)
        plan_ground = np.column_stack((plan, np.zeros(len(plan))))
        project(axes["CAM_FRONT"], CAMERAS.index("CAM_FRONT"), plan_ground,
                zip(range(len(plan) - 1), range(1, len(plan))), "#f78c22", "--", .85)
        bev.text(.025, .04, commands[int(frame["command"].argmax())], transform=bev.transAxes,
                 fontsize=3.4, color="#222222")
        ramp = bev.inset_axes((.77, .04, .18, .017))
        ramp.imshow(np.arange(256)[None, :], aspect="auto", cmap=trajectories)
        ramp.set_axis_off()
        bev.text(.77, .067, "Forecast time", transform=bev.transAxes, fontsize=3, color="#222222")
        label = fig.add_subplot(layout[start + 2, :])
        label.set_axis_off()
        label.text(0, .79, name, va="top", fontsize=7, fontfamily="serif", weight="bold")
        if row == 1 and split.startswith("mini"):
            label.text(1, .79, f"nuScenes {split} · {token}", ha="right", va="top",
                       fontsize=4.5, color=".35")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="figure", required=True)
    ablation = subparsers.add_parser("ablation", help="Figure 2")
    ablation.add_argument("metrics", nargs=4, type=Path)
    ablation.add_argument("--output", required=True, type=Path)
    scene = subparsers.add_parser("scene", help="Figure 3")
    scene.add_argument("--data", required=True, type=Path)
    scene.add_argument("--split", default="val", choices=("train", "val", "mini_train", "mini_val"))
    scene.add_argument("--token", required=True)
    scene.add_argument("--checkpoint", required=True, type=Path)
    scene.add_argument("--cache", required=True, type=Path)
    scene.add_argument("--device", default="cuda")
    scene.add_argument("--score", type=float, default=.25)
    scene.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.figure == "ablation":
        ablation_figure(args.metrics, args.output)
    else:
        qualitative_figure(args.data, args.split, args.token, args.checkpoint,
                           args.cache, args.device, args.score, args.output)


if __name__ == "__main__":
    main()
