# TinyBEV

Official PyTorch implementation of **TinyBEV**, presented by Reeshad Khan and John Gauch at ICCV Workshops 2025. This repository includes the student model and its five task heads, offline UniAD teacher export, four ablation recipes, nuScenes evaluation, and a Table 2 comparison command. [TinyBEV: Cross-Modal Knowledge Distillation for Efficient Multi-Task Bird’s-Eye-View Perception and Planning](https://reeshad-khan.github.io/assets/papers/tinybev.pdf)

## Verification status

The synthetic CPU check exercises forward passes, backpropagation through lift–splat, optimizer updates for S0–S3, and inference decoding. The default inference model has 28,020,946 trainable parameters, which rounds to the paper’s 28M.

The one-epoch nuScenes mini sanity check completed official detection and tracking evaluation. A separate ten-frame UniAD cache integration check exercised the S3 training path. Full trainval parity with Table 2 remains unverified.

The paper reports S1 L2@3s as 1.40 m in Table 2 and 1.30 m in Figure 2. For S3, Table 2 reports 0.78 m minADE and 1.08 m planning L2@3s, while the abstract calls 1.08 minADE. Section 3.3 says minADE is omitted from motion evaluation, yet Table 2 includes it. The benchmark comparison uses the Table 2 targets and computes the single-mode matched-agent ADE described below.

## Install

Use Python 3.10–3.12 and a PyTorch/CUDA build for the target GPUs. For an RTX 5090, use a Blackwell-compatible PyTorch wheel. The paper’s V100s require a Volta-compatible wheel. [PyTorch installation and wheel selection](https://pytorch.org/get-started/locally/)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m tinybev.smoke
```

Put the [nuScenes full dataset](https://www.nuscenes.org/nuscenes) and map-expansion JSON files under one root with `samples/`, `v1.0-trainval/`, and `maps/expansion/`. The student uses six synchronized camera views plus calibration and a route command. The dataset loader uses the LIDAR_TOP metadata timestamp to look up ego pose. It reads camera image files and metadata, including point counts for label filtering.

## Cache teacher predictions

Use a separate environment for [UniAD v1.0.1](https://github.com/OpenDriveLab/UniAD/releases/tag/v1.0.1) with the release’s stage-2 config and checkpoint, matching motion anchors, temporal info files, CAN bus data, and nuScenes installation. Set data paths in the config following the upstream instructions. From the UniAD checkout, run the following with one CUDA device.

```bash
TINYBEV_REPO=/absolute/path/to/TinyBEV
TINYBEV_CACHE=/absolute/path/to/cache/train

PYTHONPATH="$PWD" python "$TINYBEV_REPO/tools/export_uniad.py" \
  projects/configs/stage2_e2e/base_e2e.py \
  ckpts/uniad_base_e2e.pth "$TINYBEV_CACHE" \
  --annotations data/infos/nuscenes_infos_temporal_train.pkl
```

For a quick integration check, pass `--limit 10` and inspect the exported files. The exporter uses the UniAD test pipeline and sequential inference to retain temporal context. Each file is named `<sample_token>.npz` and records the 256×200×200 BEV tensor, calibration, scored detections, agent displacements, and plan. Each raw FP16 BEV tensor takes about 20 MB, so allow about 580 GB for a full training cache. S0 bypasses the cache.

Pass `--resume` after an interrupted export. The exporter checks the recorded config, checkpoint, annotation and PyTorch fingerprints, skips completed scenes and restarts partial scenes from their first frame to rebuild temporal state.

## Train and evaluate

Run from this repository with the student environment active.

```bash
NUSCENES=/absolute/path/to/nuscenes
TINYBEV_CACHE=/absolute/path/to/cache/train

torchrun --standalone --nproc_per_node=8 -m tinybev.train \
  --data "$NUSCENES" --variant S0 --output runs/S0

for variant in S1 S2 S3; do
  torchrun --standalone --nproc_per_node=8 -m tinybev.train \
    --data "$NUSCENES" --cache "$TINYBEV_CACHE" \
    --variant "$variant" --output "runs/$variant"
done

for variant in S0 S1 S2 S3; do
  python -m tinybev.evaluate \
    --data "$NUSCENES" --checkpoint "runs/$variant/last.pt" \
    --output "runs/$variant/val"
done

python -m tinybev.compare --check runs/S{0,1,2,3}/val/metrics.json
```

Training defaults to 20 epochs, two keyframes per GPU, AdamW with learning rate 2e-4 and weight decay 1e-2, one epoch of linear warmup, and cosine decay. The paper reports its best runs without specifying checkpoint selection. The commands above evaluate the final checkpoint. Use `--save-every 1` to retain epoch checkpoints for validation. Use `--resume runs/S3/last.pt` with the same S3 arguments to resume at the next epoch. Full benchmark comparisons require the official validation split.

On one GPU, use `python -m tinybev.train` with `--batch-size 2 --accumulate 8` for 16 keyframes per optimizer update. The one-GPU run can differ from the paper's eight-GPU run in microbatch composition and batch-normalization state. The default `--accumulate 1` follows the eight-GPU commands above.

To repeat the real-data integration check, point `NUSCENES` at a root containing `v1.0-mini/`, `samples/` and `maps/expansion/`.

```bash
python -m tinybev.train --data "$NUSCENES" --variant S0 \
  --split mini_train --epochs 1 --output runs/S0-mini
python -m tinybev.evaluate --data "$NUSCENES" --split mini_val \
  --checkpoint runs/S0-mini/last.pt --output runs/S0-mini/val
```

| Variant | Ground truth | BEV features | Detection, motion, planning | Salient BEV regions |
| --- | --- | --- | --- | --- |
| S0 | Yes | | | |
| S1 | Yes | Yes | | |
| S2 | Yes | | Yes | |
| S3 | Yes | Yes | Yes | Yes |

Evaluation exports nuScenes detection and tracking JSON and invokes the official devkit for mAP, NDS, AMOTA, and AMOTP. It reports planning L2 at step six, mean footprint collision percentage across valid waypoints excluding ground-truth collisions, single-mode forecast ADE over class-matched detections within 2 m, and the number of eligible and matched agents. Map and occupancy IoU are supplemental. FPS measures the warmed, synchronized batch-one forward and decode path on the requested device. The metrics JSON records the checkpoint path, epoch, training settings, score threshold, PyTorch and CUDA versions, and evaluation GPU. The comparison prints the measured values and deltas and fails with `--check` if any displayed precision differs.

## Compare experiments

Keep the paper settings as the baseline. Save each candidate to its own run directory, record the seed, and vary one under-specified choice at a time. The trainer records `--kd-weights FEATURE DETECTION MOTION PLAN REGION` in its checkpoints. The defaults are `1.0 0.2 0.5 0.5 0.5`. For initial experiments, compare detection score thresholds on the same checkpoint, then compare distillation variants with the same splits and hardware. Report per-metric deltas for each candidate. Use `--rank` to rank complete runs within each variant by mean absolute relative error across all eight Table 2 metrics. Use `--check` independently to test reported precision.

## Render the experimental figures

```bash
python -m tinybev.figures ablation runs/S{0,1,2,3}/val/metrics.json \
  --output runs/figures/figure2.pdf

python -m tinybev.figures scene --data "$NUSCENES" --split val \
  --token "$SAMPLE_TOKEN" --checkpoint runs/S3/last.pt \
  --cache "$TINYBEV_VAL_CACHE" --output runs/figures/figure3.pdf
```

Figure 2 uses measured values and computes improvements relative to S0 using the caption's formula. Several published bar labels differ from that formula applied to the printed table. Keep the underlying JSON with the plot. Figure 3 uses one real sample token for both models. Export the teacher with matching validation annotations, retain the file for the selected token, and record the token and display score threshold with the rendered image. The original sample token and paired checkpoints are unavailable, so exact reproduction of the published images depends on recovering those artifacts. Figure 1 is the paper's architecture diagram.

For Figure 3 with UniAD's own camera projections, palette, trajectory gradients, ego-car image and legend, run the adapter from an upstream UniAD v1.0.1 checkout. Use absolute paths for TinyBEV inputs. The cache and student each supply one forecast mode, and the adapter assigns colors by detection index.

```bash
cd "$UNIAD_CHECKOUT"
PYTHONPATH="$PWD:$TINYBEV_REPO${PYTHONPATH:+:$PYTHONPATH}" \
  python "$TINYBEV_REPO/tools/render_uniad_style.py" --data "$NUSCENES" \
  --split val --token "$SAMPLE_TOKEN" --checkpoint "$STUDENT_CHECKPOINT" \
  --cache "$TINYBEV_VAL_CACHE" --output "$FIGURE3_PDF"
```

## Implementation details

| Area | Choice |
| --- | --- |
| Student | Randomly initialized ResNet-18. Images resized to 896×512 with adjusted intrinsics and UniAD-style color distortion during training. Six-view depth distributions use 44 bins centered at 2–45 m. Weighted sum pooling projects to a 128×128 grid spanning ±51.2 m in an ego frame. Eight shared convolutions, width 468. |
| Supervision | Center heatmap and encoded 3D boxes, one 3 s agent forecast, one 3 s plan, four map layers, and six future occupancy masks. Map and occupancy use raster ground truth. The open-loop route command uses UniAD’s future-derived command convention. |
| Feature KD | Project student channels to 256 and bilinearly sample UniAD features in the student’s ego frame. Feature KD uses weight 1.0. S3 applies an extra 0.5-weighted term on occupied, foreground, or map cells. |
| Output KD | Score-derived Bernoulli KL and box L1 at teacher detection cells share a 0.2 weight. Agent motion uses the teacher’s highest-scoring mode for matched tracks. Masked waypoint L2 sums use 0.5 each for motion and planning. |
| Tracking and metrics | Class-aware velocity association supplies the IDs required by the official tracking evaluator. The original association code, forecast matching, collision protocol, detection threshold, and runtime benchmark settings are unspecified. The devkit implements nuScenes detection mAP with center-distance matching. |
| Teacher | The release identified above is a reproducible starting point. The paper describes a multi-modal teacher but supplies no checkpoint hash or sensor-fusion configuration. The upstream UniAD release uses camera images with calibration and task-specific supervision. |

The paper does not specify enough implementation detail to establish numeric equivalence from the PDF alone. The original student checkpoint, original teacher cache or checkpoint, preprocessing and evaluation configurations, and full training logs would let deviations be isolated against the published run.
