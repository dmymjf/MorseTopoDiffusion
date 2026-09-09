#!/usr/bin/env python
"""Run the operator, exact inverse audit, and a tiny neural backward pass."""
from __future__ import annotations

import numpy as np
import torch

from morsetopo.data.synthetic import donut
from morsetopo.data.wafer import circular_wafer_mask
from morsetopo.diffusion.forward import MorseForwardConfig, MorseForwardProcess
from morsetopo.models.event_net import MorseEventNet
from morsetopo.operators.reverse import ReverseEvent, ReverseEventOperator
from morsetopo.training.losses import reverse_step_loss
from morsetopo.training.reverse_dataset import sample_reverse_step


def main():
    size = 20
    domain_mask = circular_wafer_mask((size, size))
    x = donut((size, size), r_outer=7, r_inner=3) & domain_mask
    label = np.zeros(8, dtype=np.float32); label[1] = 1
    traj = MorseForwardProcess(MorseForwardConfig(seed=3)).simulate(x, label)
    print("forward events", len(traj.events), "initial betti", traj.initial_betti, "terminal", traj.terminal_betti)

    domain = __import__("morsetopo.complexes.cubical", fromlist=["CubicalComplex2D"]).CubicalComplex2D.from_binary(domain_mask)
    cur = __import__("morsetopo.complexes.cubical", fromlist=["CubicalComplex2D"]).CubicalComplex2D(shape=(size, size))
    op = ReverseEventOperator(domain)
    for e in reversed(traj.events):
        target = ReverseEvent.from_forward(e)
        candidates, idx = op.exact_inverse_candidates(cur, target)
        op.apply(cur, candidates[idx])
    assert np.array_equal(cur.face_mask().astype(bool), x)
    print("exact inverse replay: PASS", "betti", cur.betti())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MorseEventNet(num_labels=8, channels=16, cond_dim=48, event_dim=12).to(device)
    rng = np.random.default_rng(5)
    step = sample_reverse_step(traj, domain_mask, rng)
    loss = reverse_step_loss(model, step, device=device)["loss"]
    loss.backward()
    print("neural backward: PASS", "loss", float(loss.detach()))


if __name__ == "__main__":
    main()
