#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from morsetopo.data.synthetic import center, donut, edge_ring, loc, mixed, scratch
from morsetopo.data.wafer import circular_wafer_mask
from morsetopo.diffusion.forward import MorseForwardConfig, MorseForwardProcess
from morsetopo.models.event_net import MorseEventNet
from morsetopo.training.losses import reverse_step_loss
from morsetopo.training.reverse_dataset import sample_reverse_step


def main():
    ap = argparse.ArgumentParser(description="Tiny end-to-end reverse-event training smoke test.")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--channels", type=int, default=24)
    ap.add_argument("--out", default="outputs/synthetic_smoke.pt")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = args.size
    patterns = [
        center((size, size), max(2, size // 6)),
        donut((size, size), max(4, size // 3), max(2, size // 7)),
        edge_ring((size, size), max(5, size // 2 - 1), max(1, size // 12)),
        scratch((size, size), 1),
        loc((size, size)),
        mixed((size, size)),
    ]
    labels = np.zeros((len(patterns), 8), dtype=np.float32)
    for i in range(len(patterns)):
        labels[i, i] = 1.0
    domain = circular_wafer_mask((size, size))
    trajectories = []
    for i, x in enumerate(patterns):
        x = x & domain
        trajectories.append(MorseForwardProcess(MorseForwardConfig(seed=args.seed + i)).simulate(x, labels[i]))

    model = MorseEventNet(num_labels=8, channels=args.channels, cond_dim=64, event_dim=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.train()
    for step_idx in range(args.steps):
        i = int(rng.integers(len(trajectories)))
        step = sample_reverse_step(trajectories[i], domain, rng)
        opt.zero_grad(set_to_none=True)
        out = reverse_step_loss(model, step, device=device)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step_idx % max(1, args.steps // 10) == 0 or step_idx == args.steps - 1:
            print(f"step={step_idx:04d} loss={float(out['loss'].detach()):.4f} candidates trained")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_labels": 8, "channels": args.channels, "cond_dim": 64, "event_dim": 16, "size": size}, path)
    print("saved", path)


if __name__ == "__main__":
    main()
