from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from morsetopo.complexes.cubical import CubicalComplex2D
from morsetopo.operators.events import EventKind, MorseEvent
from morsetopo.operators.morse import MorseEventOperator, MorseOperatorConfig


@dataclass
class MorseForwardConfig:
    beta: float = 1.0
    max_events: int = 100000
    seed: int = 42


@dataclass
class MorseTrajectory:
    initial_mask: np.ndarray
    events: List[MorseEvent]
    initial_betti: tuple
    terminal_betti: tuple
    initial_cells: tuple
    terminal_cells: tuple
    label: Optional[np.ndarray] = None

    @property
    def topology_events(self) -> int:
        return sum(e.kind != EventKind.REGULAR_COLLAPSE for e in self.events)


class MorseForwardProcess:
    """Continuous-time event corruption on a finite cubical complex.

    The event law is structural: the admissible event family is determined by the
    current complex. Event times use a simple state-dependent CTMC clock; the core
    research object is the event operator rather than this schedule.
    """

    def __init__(
        self,
        config: MorseForwardConfig | None = None,
        operator_config: MorseOperatorConfig | None = None,
    ):
        self.cfg = config or MorseForwardConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        op_cfg = operator_config or MorseOperatorConfig(seed=self.cfg.seed)
        self.operator = MorseEventOperator(op_cfg)

    def _activity(self, k: CubicalComplex2D) -> float:
        n0, n1, n2 = k.n_cells
        # More cells -> faster corruption; critical cores naturally slow down.
        return max(1.0, n2 + 0.5 * n1 + 0.25 * n0)

    def _next_time(self, t: float, k: CubicalComplex2D) -> float:
        # lambda(K,t)=beta*a(K)/(1-t), giving exact transformed-time sampling.
        e = float(self.rng.exponential(1.0))
        a = self.cfg.beta * self._activity(k)
        nxt = 1.0 - (1.0 - t) * np.exp(-e / a)
        return float(min(nxt, 1.0 - 1e-12))

    def simulate(self, mask: np.ndarray, label=None) -> MorseTrajectory:
        mask = np.asarray(mask, dtype=bool)
        k = CubicalComplex2D.from_binary(mask)
        initial_betti = k.betti()
        initial_cells = k.n_cells
        events: List[MorseEvent] = []
        t = 0.0

        for step in range(self.cfg.max_events):
            if k.is_empty():
                break
            t = self._next_time(t, k)
            events.append(self.operator.apply(k, step=step, time=t))
        else:
            raise RuntimeError("max_events reached before empty complex")

        if not k.is_empty():
            raise RuntimeError("forward process failed to reach the empty complex")

        # Every topology-changing critical event lowers beta0+beta1 exactly once.
        expected = sum(initial_betti)
        observed = sum(e.kind != EventKind.REGULAR_COLLAPSE for e in events)
        if expected != observed:
            raise AssertionError(f"expected {expected} critical events, observed {observed}")

        return MorseTrajectory(
            initial_mask=mask.astype(np.uint8),
            events=events,
            initial_betti=initial_betti,
            terminal_betti=k.betti(),
            initial_cells=initial_cells,
            terminal_cells=k.n_cells,
            label=None if label is None else np.asarray(label),
        )
