#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from morsetopo.data.wafer import infer_defect_and_domain
from morsetopo.diffusion.forward import MorseForwardConfig, MorseForwardProcess


def main():
    ap = argparse.ArgumentParser(description="Build Morse forward trajectories for an NPZ wafer dataset.")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-key", default="images")
    ap.add_argument("--label-key", default="labels")
    ap.add_argument("--mask-key", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = np.load(args.npz, mmap_mode="r")
    images = data[args.image_key]
    labels = data[args.label_key] if args.label_key in data else None
    domains = data[args.mask_key] if args.mask_key and args.mask_key in data else None
    n = len(images) if args.limit <= 0 else min(len(images), args.limit)
    trajectories = []
    stats = {"failed": [], "events": [], "critical": []}

    for i in range(n):
        explicit = None if domains is None else (domains if domains.ndim == 2 else domains[i])
        defect, _domain = infer_defect_and_domain(np.asarray(images[i]), explicit)
        label = None if labels is None else np.asarray(labels[i])
        proc = MorseForwardProcess(MorseForwardConfig(seed=args.seed + i))
        try:
            traj = proc.simulate(defect, label=label)
        except Exception as exc:
            stats["failed"].append((i, repr(exc)))
            trajectories.append(None)
            continue
        trajectories.append(traj)
        stats["events"].append(len(traj.events))
        stats["critical"].append(traj.topology_events)
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"[{i+1}/{n}] failures={len(stats['failed'])}")

    payload = {
        "version": "morsetopo-cache-v1",
        "npz": str(Path(args.npz).name),
        "image_key": args.image_key,
        "label_key": args.label_key,
        "mask_key": args.mask_key,
        "trajectories": trajectories,
        "stats": stats,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    ev = np.asarray(stats["events"], dtype=float)
    print("saved", args.out)
    if len(ev):
        print(f"events mean={ev.mean():.1f} median={np.median(ev):.1f} max={ev.max():.0f}")
    print("failures", len(stats["failed"]))


if __name__ == "__main__":
    main()
