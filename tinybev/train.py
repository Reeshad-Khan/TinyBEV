"""Train one ablation using a single process or torchrun."""
import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .data import NuScenesFrames
from .losses import KD_WEIGHTS, objective
from .model import BEV_WIDTH, TinyBEV, DistillationAdapter


def epoch_factor(step, epoch_steps, total_steps):
    if step < epoch_steps:
        return (step + 1) / epoch_steps
    progress = (step - epoch_steps) / max(1, total_steps - epoch_steps - 1)
    return .5 * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--variant", choices=("S0", "S1", "S2", "S3"), default="S0")
    parser.add_argument("--split", choices=("train", "mini_train"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=1,
                        help="Number of microbatches per optimizer update")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1000,
                        help="Report progress after this many batches, or 0 to disable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kd-weights", type=float, nargs=5, default=KD_WEIGHTS,
                        metavar=("FEATURE", "DETECTION", "MOTION", "PLAN", "REGION"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, nargs=2, default=(896, 512))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulate, *args.image_size) < 1:
        parser.error("epochs, batch size, accumulation and both image dimensions must be positive")
    if min(args.workers, args.save_every, args.log_every) < 0:
        parser.error("worker count and logging/checkpoint intervals must be nonnegative")
    if args.resume is None and (args.output / "last.pt").exists():
        parser.error(f"{args.output} already contains last.pt. Use --resume or a new output directory.")
    args.kd_weights = tuple(args.kd_weights)
    if any(not math.isfinite(weight) or weight < 0 for weight in args.kd_weights):
        parser.error("--kd-weights must be finite and nonnegative")

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Use --device cpu for a CPU run.")
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        dataset = NuScenesFrames(args.data, args.split, image_size=args.image_size,
                                 cache=args.cache, variant=args.variant, augment=True)
        dataset.check_teacher_coverage()
        sampler = DistributedSampler(dataset, world, rank, seed=args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            persistent_workers=args.workers > 0)
        student = TinyBEV().to(device)
        adapter = DistillationAdapter(BEV_WIDTH).to(device) if args.variant in ("S1", "S3") else None

        class TrainingModule(torch.nn.Module):
            def __init__(self, model, projection):
                super().__init__()
                self.model, self.adapter = model, projection

            def forward(self, batch):
                return objective(self.model, self.adapter, batch, args.variant, args.kd_weights)

        graph = TrainingModule(student, adapter)
        training = graph
        if world > 1:
            training = DistributedDataParallel(graph, device_ids=[local_rank] if device.type == "cuda" else None)
        optimizer = torch.optim.AdamW(graph.parameters(), lr=2e-4, weight_decay=.01)
        updates_per_epoch = math.ceil(len(loader) / args.accumulate)
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, partial(epoch_factor, epoch_steps=updates_per_epoch,
                               total_steps=args.epochs * updates_per_epoch))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        start = 0
        settings = vars(args) | {"world_size": world, "data": str(Path(args.data).resolve()),
                                 "cache": str(Path(args.cache).resolve()) if args.cache else None}
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            saved_settings = {"accumulate": 1, "kd_weights": KD_WEIGHTS} | checkpoint["settings"]
            if tuple(saved_settings["kd_weights"]) != args.kd_weights:
                raise ValueError("Resume setting changed: kd_weights")
            for key in ("variant", "epochs", "split", "batch_size", "accumulate", "seed", "image_size", "world_size"):
                saved_value, current_value = saved_settings[key], settings[key]
                if key == "image_size":
                    saved_value, current_value = tuple(saved_value), tuple(current_value)
                if saved_value != current_value:
                    raise ValueError(f"Resume setting changed: {key}")
            input_paths = ("data",) if args.variant == "S0" else ("data", "cache")
            for key in input_paths:
                if Path(saved_settings[key]).resolve() != Path(settings[key]).resolve():
                    raise ValueError(f"Resume setting changed: {key}")
            graph.load_state_dict(checkpoint["graph"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            schedule.load_state_dict(checkpoint["schedule"])
            scaler.load_state_dict(checkpoint["scaler"])
            start = checkpoint["epoch"] + 1
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
        for epoch in range(start, args.epochs):
            sampler.set_epoch(epoch)
            started = time.monotonic()
            initial_updates = schedule.last_epoch
            totals = {}
            optimizer.zero_grad(set_to_none=True)
            for index, batch in enumerate(loader):
                tensors = {key: value.to(device, non_blocking=True)
                           for key, value in batch.items() if isinstance(value, torch.Tensor)}
                group_start = index - index % args.accumulate
                group_size = min(args.accumulate, len(loader) - group_start)
                update = index + 1 == group_start + group_size
                with training.no_sync() if world > 1 and not update else nullcontext():
                    with torch.autocast(device.type, enabled=device.type == "cuda"):
                        loss, terms = training(tensors)
                    scaler.scale(loss / group_size).backward()
                if update:
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= previous_scale:
                        schedule.step()
                    optimizer.zero_grad(set_to_none=True)
                for key, value in terms.items():
                    totals[key] = totals.get(key, 0) + value.detach() / len(loader)
                if rank == 0 and args.log_every and (index + 1) % args.log_every == 0:
                    print(json.dumps({"event": "training_progress", "epoch": epoch + 1,
                                      "batches": index + 1, "total_batches": len(loader),
                                      "optimizer_updates": schedule.last_epoch - initial_updates,
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
            if world > 1:
                for value in totals.values():
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                    value.div_(world)
            epoch_losses = {key: float(value) for key, value in totals.items()}
            if any(not math.isfinite(value) for value in epoch_losses.values()):
                raise FloatingPointError(f"Epoch {epoch + 1} has a nonfinite loss: {epoch_losses}")
            updates = schedule.last_epoch - initial_updates
            if not updates:
                raise FloatingPointError(f"Epoch {epoch + 1} completed without an optimizer update")
            if rank == 0:
                print(json.dumps({"epoch": epoch + 1, "optimizer_updates": updates,
                                  "skipped_updates": updates_per_epoch - updates,
                                  **epoch_losses}, allow_nan=False), flush=True)
                saved_settings = {key: str(value) if isinstance(value, Path) else value
                                  for key, value in settings.items()}
                temporary = args.output / "last.pt.tmp"
                torch.save(dict(model=student.state_dict(), graph=graph.state_dict(),
                                optimizer=optimizer.state_dict(), schedule=schedule.state_dict(),
                                scaler=scaler.state_dict(), epoch=epoch, settings=saved_settings), temporary)
                latest = args.output / "last.pt"
                temporary.replace(latest)
                if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                    snapshot = args.output / f"epoch-{epoch + 1:02d}.pt"
                    shutil.copyfile(latest, temporary)
                    temporary.replace(snapshot)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
