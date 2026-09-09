#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from morsetopo.data.wafer import circular_wafer_mask
from morsetopo.diffusion.reverse_chain import ReverseChainSampler, ReverseSampleConfig
from morsetopo.models.event_net import MorseEventNet


def main():
    ap = argparse.ArgumentParser(description="Sample a wafer cubical complex from the learned reverse event chain.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True, help="comma separated multi-hot label")
    ap.add_argument("--size", type=int, default=52)
    ap.add_argument("--max-events", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", default="outputs/sample.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    nlab = int(ckpt.get("num_labels", 8))
    channels = int(ckpt.get("channels", 48))
    cond_dim = int(ckpt.get("cond_dim", 96))
    event_dim = int(ckpt.get("event_dim", 24))
    model = MorseEventNet(num_labels=nlab, channels=channels, cond_dim=cond_dim, event_dim=event_dim).to(device)
    model.load_state_dict(ckpt["model"])
    label = np.asarray([float(v) for v in args.label.split(",")], dtype=np.float32)
    if len(label) != nlab:
        raise ValueError(f"expected {nlab} labels")
    domain = circular_wafer_mask((args.size, args.size))
    sampler = ReverseChainSampler(
        model, domain,
        ReverseSampleConfig(max_events=args.max_events, temperature=args.temperature, greedy=args.greedy),
        device=device,
    )
    k, history = sampler.sample(label)
    face = k.face_mask().astype(float)
    face[~domain] = np.nan
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(face, interpolation="nearest")
    plt.title(f"events={len(history)}, betti={k.betti()}, cells={k.n_cells}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    print("saved", path)
    print("events", len(history), "betti", k.betti(), "cells", k.n_cells)


if __name__ == "__main__":
    main()
