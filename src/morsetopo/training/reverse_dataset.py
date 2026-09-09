from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from morsetopo.complexes.cubical import CubicalComplex2D
from morsetopo.diffusion.forward import MorseTrajectory
from morsetopo.operators.reverse import ReverseEvent, ReverseEventOperator


@dataclass
class ReverseStep:
    state: CubicalComplex2D
    target: Optional[ReverseEvent]
    progress: float
    label: np.ndarray
    domain: CubicalComplex2D
    done: bool = False


def reverse_steps(trajectory: MorseTrajectory, wafer_domain_mask: np.ndarray) -> Sequence[ReverseStep]:
    """Reconstruct teacher-forced reverse states from a forward trajectory."""
    domain = CubicalComplex2D.from_binary(np.asarray(wafer_domain_mask, dtype=bool))
    current = CubicalComplex2D(shape=domain.shape)
    op = ReverseEventOperator(domain)
    label = np.zeros(8, dtype=np.float32) if trajectory.label is None else np.asarray(trajectory.label, dtype=np.float32)
    rev = [ReverseEvent.from_forward(e) for e in reversed(trajectory.events)]
    n = max(len(rev), 1)
    out = []
    for j, ev in enumerate(rev):
        out.append(ReverseStep(current.copy(), ev, j / n, label, domain, False))
        op.apply(current, ev)
    out.append(ReverseStep(current.copy(), None, 1.0, label, domain, True))
    return out


def sample_reverse_step(trajectory: MorseTrajectory, wafer_domain_mask: np.ndarray, rng: np.random.Generator) -> ReverseStep:
    """Sample one reverse teacher-forcing state without materializing every state."""
    domain = CubicalComplex2D.from_binary(np.asarray(wafer_domain_mask, dtype=bool))
    current = CubicalComplex2D(shape=domain.shape)
    op = ReverseEventOperator(domain)
    label = np.zeros(8, dtype=np.float32) if trajectory.label is None else np.asarray(trajectory.label, dtype=np.float32)
    rev = [ReverseEvent.from_forward(e) for e in reversed(trajectory.events)]
    # Include the terminal STOP target with probability 1/(K+1).
    j = int(rng.integers(len(rev) + 1))
    for i in range(j):
        op.apply(current, rev[i])
    if j == len(rev):
        return ReverseStep(current, None, 1.0, label, domain, True)
    return ReverseStep(current, rev[j], j / max(len(rev), 1), label, domain, False)
