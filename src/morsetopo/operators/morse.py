from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from morsetopo.complexes.cubical import Cell, CubicalComplex2D
from .events import EventKind, MorseEvent


@dataclass
class MorseOperatorConfig:
    # Weighting only chooses among already-valid elementary collapse pairs. It does
    # not decide validity, so homotopy preservation remains an exact constraint.
    prefer_2d_collapse: float = 1.5
    seed: int = 42


class MorseEventOperator:
    """Event algebra on a planar cubical complex.

    Regular events are elementary collapses (free face + unique coface) and preserve
    homotopy type exactly. When no elementary collapse is available, a critical
    exposed edge may be removed to kill one 1-D homology class; if no hole remains,
    an isolated critical vertex may be removed to kill one connected component.
    """

    def __init__(self, config: MorseOperatorConfig | None = None):
        self.cfg = config or MorseOperatorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def admissible_kind(self, k: CubicalComplex2D) -> EventKind:
        if k.free_pairs():
            return EventKind.REGULAR_COLLAPSE
        if k.critical_edge_candidates():
            return EventKind.CRITICAL_EDGE
        if k.critical_vertex_candidates():
            return EventKind.CRITICAL_VERTEX
        raise RuntimeError(
            "No admissible event. The complex is non-empty but has no free pair, "
            "critical edge, or isolated critical vertex. This state requires an "
            "extended critical-cell surgery and should be audited rather than hidden."
        )

    def sample_regular_pair(self, k: CubicalComplex2D) -> Tuple[Cell, Cell]:
        pairs = k.free_pairs()
        if not pairs:
            raise RuntimeError("no elementary collapse pair")
        weights = np.array(
            [self.cfg.prefer_2d_collapse if tau.dim == 2 else 1.0 for _, tau in pairs],
            dtype=float,
        )
        weights /= weights.sum()
        idx = int(self.rng.choice(len(pairs), p=weights))
        return pairs[idx]

    def apply(self, k: CubicalComplex2D, step: int, time: float) -> MorseEvent:
        before_betti = k.betti()
        before_sig = k.topology_signature()
        kind = self.admissible_kind(k)

        if kind == EventKind.REGULAR_COLLAPSE:
            sigma, tau = self.sample_regular_pair(k)
            k.collapse(sigma, tau)
        elif kind == EventKind.CRITICAL_EDGE:
            candidates = k.critical_edge_candidates()
            sigma = candidates[int(self.rng.integers(len(candidates)))]
            tau = None
            k.remove_exposed_edge(sigma)
        else:
            candidates = k.critical_vertex_candidates()
            sigma = candidates[int(self.rng.integers(len(candidates)))]
            tau = None
            k.remove_isolated_vertex(sigma)

        after_betti = k.betti()
        after_sig = k.topology_signature()

        if kind == EventKind.REGULAR_COLLAPSE:
            if after_betti != before_betti or after_sig != before_sig:
                raise AssertionError(
                    f"elementary collapse changed topology: {before_betti}/{before_sig} -> "
                    f"{after_betti}/{after_sig}"
                )
        else:
            if sum(after_betti) != sum(before_betti) - 1:
                raise AssertionError(
                    f"critical event must reduce beta0+beta1 by one: {before_betti}->{after_betti}"
                )

        return MorseEvent(
            step=step,
            time=float(time),
            kind=kind,
            sigma=sigma,
            tau=tau,
            betti_before=before_betti,
            betti_after=after_betti,
            signature_before=before_sig,
            signature_after=after_sig,
        )
