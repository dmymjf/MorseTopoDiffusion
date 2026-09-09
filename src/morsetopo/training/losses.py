from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from morsetopo.complexes.tensor import LatticeSpec, complex_channels
from morsetopo.models.event_net import MorseEventNet
from morsetopo.operators.reverse import ReverseEventOperator
from .reverse_dataset import ReverseStep


def _topo_features(step: ReverseStep, initial_cell_scale: float = 1000.0) -> np.ndarray:
    b0, b1 = step.state.betti()
    nc = sum(step.state.n_cells) / initial_cell_scale
    return np.asarray([b0 / 32.0, b1 / 16.0, nc, step.progress], dtype=np.float32)


def reverse_step_loss(model: MorseEventNet, step: ReverseStep, device="cpu") -> Dict[str, torch.Tensor]:
    state_np = complex_channels(step.state, step.domain)
    x = torch.from_numpy(state_np)[None].to(device)
    t = torch.tensor([step.progress], dtype=torch.float32, device=device)
    y = torch.from_numpy(step.label.astype(np.float32))[None].to(device)
    topo = torch.from_numpy(_topo_features(step))[None].to(device)
    spec = LatticeSpec(*step.domain.shape)
    op = ReverseEventOperator(step.domain)

    if step.done:
        candidates = op.enumerate(step.state, include_all_kinds=True)
        logits, rate = model(x, t, y, topo, candidates, spec, include_stop=True)
        target = torch.tensor([len(candidates)], dtype=torch.long, device=device)
        ce = F.cross_entropy(logits[None], target)
        return {"loss": ce, "event_ce": ce, "rate": rate.mean()}

    assert step.target is not None
    candidates, idx = op.exact_inverse_candidates(step.state, step.target)
    logits, rate = model(x, t, y, topo, candidates, spec, include_stop=True)
    target = torch.tensor([idx], dtype=torch.long, device=device)
    ce = F.cross_entropy(logits[None], target)
    # Small rate regularizer keeps the optional CTMC head numerically sane without making
    # timing a claimed contribution in this prototype.
    rate_reg = 1e-4 * (torch.log(rate + 1e-8) ** 2).mean()
    return {"loss": ce + rate_reg, "event_ce": ce, "rate_reg": rate_reg, "rate": rate.mean()}
