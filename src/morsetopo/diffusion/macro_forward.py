from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from morsetopo.complexes.cubical import CubicalComplex2D
from morsetopo.operators.events import EventKind, MorseEvent
from morsetopo.operators.morse import MorseEventOperator, MorseOperatorConfig


@dataclass
class MacroForwardConfig:
    macro_size: int = 16
    beta: float = 1.0
    max_macro_steps: int = 10000
    seed: int = 42


@dataclass
class MorseMacroEvent:
    step: int
    time: float
    kind: str
    micro_events: List[MorseEvent]

    @property
    def size(self) -> int:
        return len(self.micro_events)


@dataclass
class MorseMacroTrajectory:
    initial_mask: np.ndarray
    macro_events: List[MorseMacroEvent]
    initial_betti: tuple
    terminal_betti: tuple
    label: Optional[np.ndarray] = None

    @property
    def micro_event_count(self) -> int:
        return sum(e.size for e in self.macro_events)


class MorseMacroForwardProcess:
    """Exact macro batching of elementary Morse events.

    A regular macro-event is simply a *sequential list* of valid elementary collapses.
    Therefore the macro transition preserves homotopy exactly; batching changes only
    the computational granularity, not the underlying topological operator.
    """

    def __init__(self, config: MacroForwardConfig | None = None):
        self.cfg = config or MacroForwardConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.operator = MorseEventOperator(MorseOperatorConfig(seed=self.cfg.seed))

    def _activity(self, k: CubicalComplex2D) -> float:
        n0, n1, n2 = k.n_cells
        return max(1.0, (n0 + n1 + n2) / max(self.cfg.macro_size, 1))

    def _next_time(self, t: float, k: CubicalComplex2D) -> float:
        e = float(self.rng.exponential(1.0))
        a = self.cfg.beta * self._activity(k)
        return float(min(1.0 - (1.0 - t) * np.exp(-e / a), 1.0 - 1e-12))

    def simulate(self, mask: np.ndarray, label=None) -> MorseMacroTrajectory:
        mask = np.asarray(mask, dtype=bool)
        k = CubicalComplex2D.from_binary(mask)
        initial_betti = k.betti()
        t = 0.0
        macros: List[MorseMacroEvent] = []
        micro_step = 0

        for macro_step in range(self.cfg.max_macro_steps):
            if k.is_empty():
                break
            t = self._next_time(t, k)
            first_kind = self.operator.admissible_kind(k)
            micro: List[MorseEvent] = []

            if first_kind == EventKind.REGULAR_COLLAPSE:
                before = k.betti()
                for _ in range(self.cfg.macro_size):
                    if k.is_empty() or self.operator.admissible_kind(k) != EventKind.REGULAR_COLLAPSE:
                        break
                    micro.append(self.operator.apply(k, micro_step, t))
                    micro_step += 1
                if k.betti() != before:
                    raise AssertionError("regular collapse macro changed Betti numbers")
                kind = "Rmacro-"
            else:
                micro.append(self.operator.apply(k, micro_step, t))
                micro_step += 1
                kind = first_kind.value

            macros.append(MorseMacroEvent(macro_step, t, kind, micro))
        else:
            raise RuntimeError("max_macro_steps reached")

        if not k.is_empty():
            raise RuntimeError("macro process failed to reach empty complex")
        return MorseMacroTrajectory(
            initial_mask=mask.astype(np.uint8),
            macro_events=macros,
            initial_betti=initial_betti,
            terminal_betti=k.betti(),
            label=None if label is None else np.asarray(label),
        )
