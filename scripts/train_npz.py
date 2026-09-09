#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from morsetopo.data.wafer import infer_defect_and_domain
from morsetopo.models.event_net import MorseEventNet
from morsetopo.training.losses import reverse_step_loss
from morsetopo.training.reverse_dataset import sample_reverse_step


def main():
    ap = argparse.ArgumentParser(description="Train the reverse structural-event model from a cached NPZ dataset.")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="outputs/morsetopo.pt")
    ap.add_argument("--image-key", default="images")
    ap.add_argument("--label-key", default="labels")
    ap.add_argument("--mask-key", default=None)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.npz, mmap_mode="r")
    images = data[args.image_key]
    labels = data[args.label_key]
    masks = data[args.mask_key] if args.mask_key and args.mask_key in data else None
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    trajectories = cache["trajectories"]
    valid = np.asarray([i for i, t in enumerate(trajectories) if t is not None], dtype=int)
    if len(valid) == 0:
        raise RuntimeError("trajectory cache contains no valid samples")

    model = MorseEventNet(num_labels=labels.shape[1], channels=args.channels, cond_dim=96, event_dim=24).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ema = None
    for it in range(1, args.steps + 1):
        idx = int(rng.choice(valid))
        explicit = None if masks is None else (masks if masks.ndim == 2 else masks[idx])
        _defect, domain = infer_defect_and_domain(np.asarray(images[idx]), explicit)
        step = sample_reverse_step(trajectories[idx], domain, rng)
        # Cache may have labels, but make the dataset label authoritative.
        step.label = np.asarray(labels[idx], dtype=np.float32)
        opt.zero_grad(set_to_none=True)
        d = reverse_step_loss(model, step, device=device)
        d["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lv = float(d["loss"].detach())
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if it % 50 == 0 or it == 1:
            print(f"iter={it:06d} loss={lv:.4f} ema={ema:.4f}")
        if it % args.save_every == 0 or it == args.steps:
            torch.save({
                "model": model.state_dict(), "num_labels": labels.shape[1], "channels": args.channels, "cond_dim": 96, "event_dim": 24,
                "iteration": it, "loss_ema": ema, "args": vars(args),
            }, out_path)
            print("saved", out_path)


if __name__ == "__main__":
    main()
