"""Exercise all losses, the projection and decoding using synthetic six-camera inputs."""
import numpy as np
import torch

from .data import MODEL_INPUTS
from .losses import objective
from .model import CLASSES, MAP_LAYERS, STEPS, DistillationAdapter, Grid, TinyBEV
from .teacher import warp_bev


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    grid = Grid(size=8)
    model = TinyBEV(grid=grid, width=16, topk=4, depths=4)
    adapter = DistillationAdapter(16)
    intrinsics = torch.tensor([[40., 0., 32.], [0., 40., 32.], [0., 0., 1.]])
    camera_to_ego = torch.tensor([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                                  [0., -1., 0., 1.5], [0., 0., 0., 1.]])
    batch = dict(images=torch.randn(1, 6, 3, 64, 64),
                 intrinsics=intrinsics.expand(1, 6, 3, 3),
                 camera_to_ego=camera_to_ego.expand(1, 6, 4, 4),
                 command=torch.tensor([[0., 0., 1.]]),
                 agent_cells=torch.tensor([[[4, 4]]]),
                 agents=torch.ones(1, 1, dtype=torch.bool),
                 boxes=torch.zeros(1, 1, 10),
                 heatmap=torch.zeros(1, len(CLASSES), 8, 8),
                 motion=torch.zeros(1, 1, STEPS, 2),
                 motion_mask=torch.ones(1, 1, STEPS, dtype=torch.bool),
                 plan=torch.zeros(1, STEPS, 2),
                 plan_mask=torch.ones(1, STEPS, dtype=torch.bool),
                 map=torch.zeros(1, len(MAP_LAYERS), 8, 8),
                 occ=torch.zeros(1, STEPS, 8, 8),
                 salient=torch.ones(1, 8, 8, dtype=torch.bool),
                 teacher_cells=torch.tensor([[[4, 4]]]),
                 teacher_agents=torch.ones(1, 1, dtype=torch.bool),
                 teacher_labels=torch.zeros(1, 1, dtype=torch.long),
                 teacher_logits=torch.ones(1, 1),
                 teacher_boxes=torch.zeros(1, 1, 10),
                 teacher_motion=torch.zeros(1, 1, STEPS, 2),
                 teacher_motion_mask=torch.ones(1, 1, STEPS, dtype=torch.bool),
                 teacher_plan=torch.zeros(1, STEPS, 2))
    batch["heatmap"][0, 0, 4, 4] = 1
    batch["teacher_bev"] = warp_bev(torch.ones(256, 200, 200), np.eye(4, dtype=np.float32), grid)[None]
    optimizer = torch.optim.AdamW([*model.parameters(), *adapter.parameters()], lr=2e-4)
    expected = [set(), {"kd_feature"}, {"kd_detection", "kd_motion", "kd_plan"},
                {"kd_feature", "kd_detection", "kd_motion", "kd_plan", "kd_adaptive"}]
    for variant, kd in zip(("S0", "S1", "S2", "S3"), expected):
        optimizer.zero_grad(set_to_none=True)
        loss, components = objective(model, adapter, batch, variant)
        assert {name for name in components if name.startswith("kd_")} == kd
        assert loss.isfinite()
        loss.backward()
        assert model.depth.weight.grad.abs().sum() > 0
        assert all(parameter.grad.isfinite().all() for parameter in model.parameters())
        optimizer.step()
        print(f"{variant} forward/backward/optimizer OK")
    model.eval()
    with torch.inference_mode():
        output = model(*(batch[key] for key in MODEL_INPUTS))
        decoded = model.decode(output)
        assert decoded["center"].shape == (1, 4, 3)
        assert decoded["motion"].shape == (1, 4, STEPS, 2)
        assert output["plan"].shape == (1, STEPS, 2)
        assert all(tensor.isfinite().all() for tensor in decoded.values())
    print(f"Inference and decoding OK. Default student: {sum(p.numel() for p in TinyBEV().parameters()):,} parameters")


if __name__ == "__main__":
    main()
