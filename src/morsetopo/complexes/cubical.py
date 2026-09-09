from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True, order=True)
class Cell:
    """A cell in a 2-D cubical complex.

    dim=0: kind='v', (r,c) is a grid vertex.
    dim=1: kind='h' or 'v', (r,c) is the top/left endpoint.
    dim=2: kind='s', (r,c) is the image pixel / unit square.
    """

    dim: int
    kind: str
    r: int
    c: int

    def short(self) -> str:
        return f"{self.dim}:{self.kind}@({self.r},{self.c})"


def vertex(r: int, c: int) -> Cell:
    return Cell(0, "v", int(r), int(c))


def hedge(r: int, c: int) -> Cell:
    return Cell(1, "h", int(r), int(c))


def vedge(r: int, c: int) -> Cell:
    return Cell(1, "v", int(r), int(c))


def square(r: int, c: int) -> Cell:
    return Cell(2, "s", int(r), int(c))


def immediate_faces(cell: Cell) -> Tuple[Cell, ...]:
    if cell.dim == 2:
        r, c = cell.r, cell.c
        return (hedge(r, c), hedge(r + 1, c), vedge(r, c), vedge(r, c + 1))
    if cell.dim == 1:
        r, c = cell.r, cell.c
        if cell.kind == "h":
            return (vertex(r, c), vertex(r, c + 1))
        if cell.kind == "v":
            return (vertex(r, c), vertex(r + 1, c))
    return ()


def square_vertices(face: Cell) -> Tuple[Cell, ...]:
    assert face.dim == 2
    r, c = face.r, face.c
    return (vertex(r, c), vertex(r, c + 1), vertex(r + 1, c), vertex(r + 1, c + 1))


class CubicalComplex2D:
    """Finite cubical subcomplex of the planar square grid.

    The complex is represented explicitly at dimensions 0, 1, and 2.  This allows
    a corruption trajectory to pass through mixed-dimensional skeletons instead of
    forcing every intermediate state to remain a binary image.
    """

    def __init__(self, cells: Dict[int, Iterable[Cell]] | None = None, shape: Tuple[int, int] | None = None):
        self.cells: Dict[int, Set[Cell]] = {0: set(), 1: set(), 2: set()}
        if cells:
            for d in (0, 1, 2):
                self.cells[d].update(cells.get(d, []))
        self.shape = shape
        self._validate_dimensions()

    @classmethod
    def from_binary(cls, mask: np.ndarray) -> "CubicalComplex2D":
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        cells = {0: set(), 1: set(), 2: set()}
        for r, c in np.argwhere(mask):
            f = square(int(r), int(c))
            cells[2].add(f)
            for e in immediate_faces(f):
                cells[1].add(e)
                cells[0].update(immediate_faces(e))
        return cls(cells, shape=(h, w))

    def copy(self) -> "CubicalComplex2D":
        return CubicalComplex2D({d: set(v) for d, v in self.cells.items()}, shape=self.shape)

    def _validate_dimensions(self) -> None:
        for d, items in self.cells.items():
            for cell in items:
                if cell.dim != d:
                    raise ValueError(f"cell {cell} stored in wrong dimension {d}")

    def __len__(self) -> int:
        return sum(len(self.cells[d]) for d in (0, 1, 2))

    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def n_cells(self) -> Tuple[int, int, int]:
        return tuple(len(self.cells[d]) for d in (0, 1, 2))

    def contains(self, cell: Cell) -> bool:
        return cell in self.cells[cell.dim]

    def immediate_cofaces(self, cell: Cell) -> List[Cell]:
        if cell.dim == 2:
            return []
        if cell.dim == 1:
            out: List[Cell] = []
            r, c = cell.r, cell.c
            if cell.kind == "h":
                candidates = (square(r - 1, c), square(r, c))
            else:
                candidates = (square(r, c - 1), square(r, c))
            for f in candidates:
                if f in self.cells[2]:
                    out.append(f)
            return out
        # vertex -> incident edges
        r, c = cell.r, cell.c
        candidates = (hedge(r, c - 1), hedge(r, c), vedge(r - 1, c), vedge(r, c))
        return [e for e in candidates if e in self.cells[1]]

    def incident_faces_of_vertex(self, v: Cell) -> List[Cell]:
        assert v.dim == 0
        r, c = v.r, v.c
        candidates = (
            square(r - 1, c - 1), square(r - 1, c),
            square(r, c - 1), square(r, c),
        )
        return [f for f in candidates if f in self.cells[2]]

    def free_pairs(self) -> List[Tuple[Cell, Cell]]:
        """Return elementary collapse pairs (free face sigma, unique coface tau)."""
        pairs: List[Tuple[Cell, Cell]] = []
        # 1-cell -> 2-cell collapses
        for e in self.cells[1]:
            co = self.immediate_cofaces(e)
            if len(co) == 1:
                pairs.append((e, co[0]))
        # 0-cell -> 1-cell collapses. A vertex cannot be a face of any 2-cell.
        for v in self.cells[0]:
            if self.incident_faces_of_vertex(v):
                continue
            co = self.immediate_cofaces(v)
            if len(co) == 1:
                pairs.append((v, co[0]))
        return pairs

    def collapse(self, sigma: Cell, tau: Cell) -> None:
        if sigma.dim + 1 != tau.dim:
            raise ValueError("elementary collapse requires dim(tau)=dim(sigma)+1")
        if sigma not in self.cells[sigma.dim] or tau not in self.cells[tau.dim]:
            raise ValueError("collapse cells are not present")
        if tau not in self.immediate_cofaces(sigma):
            raise ValueError("sigma is not a face of tau")
        # Verify sigma is truly free in the current complex.
        if sigma.dim == 0 and self.incident_faces_of_vertex(sigma):
            raise ValueError("vertex is incident to a 2-cell and is not free")
        if len(self.immediate_cofaces(sigma)) != 1:
            raise ValueError("sigma is not a free face")
        self.cells[tau.dim].remove(tau)
        self.cells[sigma.dim].remove(sigma)

    def remove_exposed_edge(self, e: Cell) -> None:
        """Remove a 1-cell that is not a face of any 2-cell (topology surgery)."""
        if e.dim != 1 or e not in self.cells[1]:
            raise ValueError("edge not present")
        if self.immediate_cofaces(e):
            raise ValueError("cannot remove an edge that supports a 2-cell")
        self.cells[1].remove(e)

    def remove_isolated_vertex(self, v: Cell) -> None:
        if v.dim != 0 or v not in self.cells[0]:
            raise ValueError("vertex not present")
        if self.immediate_cofaces(v) or self.incident_faces_of_vertex(v):
            raise ValueError("vertex is not isolated")
        self.cells[0].remove(v)

    def add_cells_closed(self, cells: Sequence[Cell]) -> None:
        """Add cells and all their faces, preserving the subcomplex property."""
        stack = list(cells)
        while stack:
            cell = stack.pop()
            if cell in self.cells[cell.dim]:
                continue
            self.cells[cell.dim].add(cell)
            stack.extend(immediate_faces(cell))

    def euler_characteristic(self) -> int:
        n0, n1, n2 = self.n_cells
        return n0 - n1 + n2

    def _vertex_components(self) -> Tuple[Dict[Cell, int], int]:
        if not self.cells[0]:
            return {}, 0
        adj: Dict[Cell, Set[Cell]] = {v: set() for v in self.cells[0]}
        for e in self.cells[1]:
            a, b = immediate_faces(e)
            if a in adj and b in adj:
                adj[a].add(b)
                adj[b].add(a)
        comp: Dict[Cell, int] = {}
        cid = 0
        for v in adj:
            if v in comp:
                continue
            cid += 1
            stack = [v]
            comp[v] = cid
            while stack:
                u = stack.pop()
                for z in adj[u]:
                    if z not in comp:
                        comp[z] = cid
                        stack.append(z)
        return comp, cid

    def betti(self) -> Tuple[int, int]:
        """Return (beta0,beta1) for a finite planar cubical complex.

        Any finite cubical subcomplex embedded in R^2 has beta2=0, so
        chi = beta0-beta1.
        """
        if self.is_empty():
            return 0, 0
        _, beta0 = self._vertex_components()
        chi = self.euler_characteristic()
        beta1 = beta0 - chi
        if beta1 < 0:
            raise RuntimeError(f"invalid planar Betti calculation: beta0={beta0}, chi={chi}")
        return int(beta0), int(beta1)

    def topology_signature(self) -> Tuple[int, ...]:
        """Sorted per-component beta1 values."""
        comp_map, ncomp = self._vertex_components()
        if ncomp == 0:
            return ()
        counts = {cid: [0, 0, 0] for cid in range(1, ncomp + 1)}
        for v, cid in comp_map.items():
            counts[cid][0] += 1
        for e in self.cells[1]:
            a, _ = immediate_faces(e)
            cid = comp_map[a]
            counts[cid][1] += 1
        for f in self.cells[2]:
            v = square_vertices(f)[0]
            cid = comp_map[v]
            counts[cid][2] += 1
        holes = []
        for cid in counts:
            n0, n1, n2 = counts[cid]
            chi = n0 - n1 + n2
            holes.append(max(0, 1 - chi))
        return tuple(sorted(int(h) for h in holes))

    def exposed_edges(self) -> List[Cell]:
        return [e for e in self.cells[1] if not self.immediate_cofaces(e)]

    def isolated_vertices(self) -> List[Cell]:
        return [v for v in self.cells[0] if not self.immediate_cofaces(v) and not self.incident_faces_of_vertex(v)]

    def critical_edge_candidates(self) -> List[Cell]:
        """Edges whose deletion preserves beta0 and lowers beta1 by exactly one."""
        b0, b1 = self.betti()
        if b1 == 0:
            return []
        out: List[Cell] = []
        for e in self.exposed_edges():
            y = self.copy()
            y.remove_exposed_edge(e)
            a0, a1 = y.betti()
            if a0 == b0 and a1 == b1 - 1:
                out.append(e)
        return out

    def critical_vertex_candidates(self) -> List[Cell]:
        """Isolated vertices whose deletion lowers beta0 by exactly one."""
        b0, b1 = self.betti()
        out: List[Cell] = []
        for v in self.isolated_vertices():
            y = self.copy()
            y.remove_isolated_vertex(v)
            a0, a1 = y.betti()
            if a0 == b0 - 1 and a1 == b1:
                out.append(v)
        return out

    def face_mask(self) -> np.ndarray:
        if self.shape is None:
            if not self.cells[2]:
                return np.zeros((1, 1), dtype=np.uint8)
            h = 1 + max(f.r for f in self.cells[2])
            w = 1 + max(f.c for f in self.cells[2])
        else:
            h, w = self.shape
        out = np.zeros((h, w), dtype=np.uint8)
        for f in self.cells[2]:
            if 0 <= f.r < h and 0 <= f.c < w:
                out[f.r, f.c] = 1
        return out

    def summary(self) -> dict:
        b0, b1 = self.betti()
        return {
            "cells": self.n_cells,
            "beta0": b0,
            "beta1": b1,
            "signature": self.topology_signature(),
            "free_pairs": len(self.free_pairs()),
        }
