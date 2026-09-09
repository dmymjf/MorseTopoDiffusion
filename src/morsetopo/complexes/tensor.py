from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .cubical import Cell, CubicalComplex2D


@dataclass(frozen=True)
class LatticeSpec:
    height: int
    width: int

    @property
    def lattice_shape(self) -> Tuple[int, int]:
        return 2 * self.height + 1, 2 * self.width + 1

    def coord(self, cell: Cell) -> Tuple[int, int]:
        if cell.dim == 0:
            return 2 * cell.r, 2 * cell.c
        if cell.dim == 1 and cell.kind == "h":
            return 2 * cell.r, 2 * cell.c + 1
        if cell.dim == 1 and cell.kind == "v":
            return 2 * cell.r + 1, 2 * cell.c
        if cell.dim == 2:
            return 2 * cell.r + 1, 2 * cell.c + 1
        raise ValueError(f"unsupported cell: {cell}")


def _kind_channel(cell: Cell) -> int:
    if cell.dim == 0:
        return 0
    if cell.dim == 1 and cell.kind == "h":
        return 1
    if cell.dim == 1 and cell.kind == "v":
        return 2
    if cell.dim == 2:
        return 3
    raise ValueError(cell)


def complex_channels(k: CubicalComplex2D, domain: CubicalComplex2D | None = None) -> np.ndarray:
    """Encode a mixed-dimensional cubical complex as 11 lattice channels.

    Channels 0..3: occupied vertex / h-edge / v-edge / square cells.
    Channels 4..7: domain availability for the same four cell types.
    Channels 8..10: radius, sin(theta), cos(theta) positional fields.
    """
    if k.shape is None and (domain is None or domain.shape is None):
        raise ValueError("complex or domain must define the original image shape")
    shape = k.shape if k.shape is not None else domain.shape
    assert shape is not None
    spec = LatticeSpec(*shape)
    lh, lw = spec.lattice_shape
    out = np.zeros((11, lh, lw), dtype=np.float32)

    for d in (0, 1, 2):
        for cell in k.cells[d]:
            rr, cc = spec.coord(cell)
            out[_kind_channel(cell), rr, cc] = 1.0

    dom = domain if domain is not None else k
    for d in (0, 1, 2):
        for cell in dom.cells[d]:
            rr, cc = spec.coord(cell)
            out[4 + _kind_channel(cell), rr, cc] = 1.0

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, lh, dtype=np.float32),
        np.linspace(-1.0, 1.0, lw, dtype=np.float32),
        indexing="ij",
    )
    rr = np.sqrt(xx * xx + yy * yy)
    th = np.arctan2(yy, xx)
    dom_any = out[4:8].sum(axis=0) > 0
    out[8] = rr * dom_any
    out[9] = np.sin(th) * dom_any
    out[10] = np.cos(th) * dom_any
    return out
