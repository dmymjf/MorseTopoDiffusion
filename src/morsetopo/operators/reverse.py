from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Tuple

from morsetopo.complexes.cubical import Cell, CubicalComplex2D, immediate_faces
from morsetopo.operators.events import EventKind, MorseEvent


class ReverseEventKind(str, Enum):
    REGULAR_EXPANSION = "R+"
    HOLE_BIRTH = "H+"
    COMPONENT_BIRTH = "C+"


@dataclass(frozen=True)
class ReverseEvent:
    kind: ReverseEventKind
    sigma: Cell
    tau: Optional[Cell] = None

    @classmethod
    def from_forward(cls, event: MorseEvent) -> "ReverseEvent":
        kind = {
            EventKind.REGULAR_COLLAPSE: ReverseEventKind.REGULAR_EXPANSION,
            EventKind.CRITICAL_EDGE: ReverseEventKind.HOLE_BIRTH,
            EventKind.CRITICAL_VERTEX: ReverseEventKind.COMPONENT_BIRTH,
        }[event.kind]
        return cls(kind=kind, sigma=event.sigma, tau=event.tau)


def _all_cells(domain: CubicalComplex2D) -> Iterable[Cell]:
    for d in (0, 1, 2):
        yield from domain.cells[d]


class ReverseEventOperator:
    """Enumerate and apply inverse Morse events exactly within a fixed wafer domain."""

    def __init__(self, domain: CubicalComplex2D):
        self.domain = domain.copy()

    @staticmethod
    def _same_complex(a: CubicalComplex2D, b: CubicalComplex2D) -> bool:
        return all(a.cells[d] == b.cells[d] for d in (0, 1, 2))

    def _regular_candidates(self, k: CubicalComplex2D) -> List[ReverseEvent]:
        out: List[ReverseEvent] = []
        # Any absent 1- or 2-cell can be tau. Sigma is one absent immediate face.
        for dim in (1, 2):
            for tau in self.domain.cells[dim]:
                if tau in k.cells[dim]:
                    continue
                faces = immediate_faces(tau)
                for sigma in faces:
                    if sigma not in self.domain.cells[sigma.dim] or sigma in k.cells[sigma.dim]:
                        continue
                    # All other immediate faces of tau must already be present. For a 2-cell,
                    # vertices are inherited from those edges; for a 1-cell, the other endpoint exists.
                    if any(f != sigma and f not in k.cells[f.dim] for f in faces):
                        continue
                    y = k.copy()
                    y.add_cells_closed([sigma, tau])
                    # Expansion must add exactly the intended pair and be invertible by one collapse.
                    extras = []
                    for d in (0, 1, 2):
                        extras.extend(y.cells[d] - k.cells[d])
                    if set(extras) != {sigma, tau}:
                        continue
                    try:
                        z = y.copy()
                        z.collapse(sigma, tau)
                    except ValueError:
                        continue
                    if self._same_complex(z, k):
                        out.append(ReverseEvent(ReverseEventKind.REGULAR_EXPANSION, sigma, tau))
        return out

    def _hole_candidates(self, k: CubicalComplex2D) -> List[ReverseEvent]:
        out: List[ReverseEvent] = []
        # Inverse of deleting an exposed edge: add an absent domain edge whose endpoints
        # already exist and whose insertion creates exactly one new 1-cycle.
        b0, b1 = k.betti()
        for e in self.domain.cells[1]:
            if e in k.cells[1]:
                continue
            a, b = immediate_faces(e)
            if a not in k.cells[0] or b not in k.cells[0]:
                continue
            y = k.copy()
            y.add_cells_closed([e])
            a0, a1 = y.betti()
            if a0 == b0 and a1 == b1 + 1:
                # The forward inverse must be an exposed critical edge.
                z = y.copy()
                try:
                    z.remove_exposed_edge(e)
                except ValueError:
                    continue
                if self._same_complex(z, k):
                    out.append(ReverseEvent(ReverseEventKind.HOLE_BIRTH, e, None))
        return out

    def _component_candidates(self, k: CubicalComplex2D) -> List[ReverseEvent]:
        # Inverse of isolated-vertex death. Any absent domain vertex can be born as an isolated
        # component because no edge is added in this event.
        return [
            ReverseEvent(ReverseEventKind.COMPONENT_BIRTH, v, None)
            for v in self.domain.cells[0]
            if v not in k.cells[0]
        ]

    def enumerate(self, k: CubicalComplex2D, include_all_kinds: bool = True) -> List[ReverseEvent]:
        regular = self._regular_candidates(k)
        holes = self._hole_candidates(k)
        comps = self._component_candidates(k)
        if include_all_kinds:
            return regular + holes + comps
        # Strict inverse gating: prefer regular expansion when possible; critical events otherwise.
        if regular:
            return regular
        if holes:
            return holes
        return comps

    def apply(self, k: CubicalComplex2D, event: ReverseEvent) -> None:
        if event.kind == ReverseEventKind.REGULAR_EXPANSION:
            if event.tau is None:
                raise ValueError("regular expansion requires tau")
            k.add_cells_closed([event.sigma, event.tau])
        elif event.kind == ReverseEventKind.HOLE_BIRTH:
            k.add_cells_closed([event.sigma])
        elif event.kind == ReverseEventKind.COMPONENT_BIRTH:
            k.add_cells_closed([event.sigma])
        else:
            raise ValueError(event.kind)

    def exact_inverse_candidates(self, k: CubicalComplex2D, target: ReverseEvent) -> Tuple[List[ReverseEvent], int]:
        candidates = self.enumerate(k, include_all_kinds=True)
        try:
            idx = candidates.index(target)
        except ValueError as exc:
            raise RuntimeError(f"target inverse event is not admissible: {target}") from exc
        return candidates, idx
