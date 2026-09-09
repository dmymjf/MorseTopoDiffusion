from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from morsetopo.complexes.cubical import Cell


class EventKind(str, Enum):
    REGULAR_COLLAPSE = "R-"
    CRITICAL_EDGE = "H-"
    CRITICAL_VERTEX = "C-"

    @property
    def inverse(self) -> str:
        return {
            EventKind.REGULAR_COLLAPSE: "R+",
            EventKind.CRITICAL_EDGE: "H+",
            EventKind.CRITICAL_VERTEX: "C+",
        }[self]


@dataclass(frozen=True)
class MorseEvent:
    step: int
    time: float
    kind: EventKind
    sigma: Cell
    tau: Optional[Cell]
    betti_before: Tuple[int, int]
    betti_after: Tuple[int, int]
    signature_before: Tuple[int, ...]
    signature_after: Tuple[int, ...]

    def is_topology_changing(self) -> bool:
        return self.kind != EventKind.REGULAR_COLLAPSE
