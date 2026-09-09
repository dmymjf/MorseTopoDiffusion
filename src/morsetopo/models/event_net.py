from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from morsetopo.complexes.cubical import Cell
from morsetopo.complexes.tensor import LatticeSpec
from morsetopo.operators.reverse import ReverseEvent, ReverseEventKind


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            x = x[None]
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half, device=x.device, dtype=x.dtype))
        a = 2 * math.pi * x[:, None] * freqs[None]
        y = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
        if y.shape[-1] < self.dim:
            y = F.pad(y, (0, self.dim - y.shape[-1]))
        return y


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int = 1):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.n1 = nn.GroupNorm(groups, channels)
        self.c1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.n2 = nn.GroupNorm(groups, channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.n2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.c2(F.silu(h))


class MorseEventNet(nn.Module):
    """Score topology-valid inverse cubical events rather than pixels/noise."""

    EVENT_INDEX = {
        ReverseEventKind.REGULAR_EXPANSION: 0,
        ReverseEventKind.HOLE_BIRTH: 1,
        ReverseEventKind.COMPONENT_BIRTH: 2,
    }

    def __init__(self, num_labels: int = 8, channels: int = 48, cond_dim: int = 96, event_dim: int = 24):
        super().__init__()
        self.num_labels = num_labels
        self.channels = channels
        self.cond_dim = cond_dim
        self.time = SinusoidalEmbedding(cond_dim // 2)
        self.label = nn.Sequential(nn.Linear(num_labels, cond_dim // 2), nn.SiLU(), nn.Linear(cond_dim // 2, cond_dim // 2))
        self.cond = nn.Sequential(nn.Linear(cond_dim + 4, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.in_conv = nn.Conv2d(11, channels, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMBlock(channels, cond_dim, d) for d in (1, 2, 1, 4, 1)])
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.out_norm = nn.GroupNorm(groups, channels)
        self.event_emb = nn.Embedding(3, event_dim)
        # sigma local, tau local, global, condition, event emb, 6 geometry scalars
        in_dim = channels * 3 + cond_dim + event_dim + 6
        self.candidate_mlp = nn.Sequential(
            nn.Linear(in_dim, 192), nn.SiLU(), nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, 1)
        )
        self.stop_head = nn.Sequential(nn.Linear(channels + cond_dim, 96), nn.SiLU(), nn.Linear(96, 1))
        self.rate_head = nn.Sequential(nn.Linear(channels + cond_dim, 96), nn.SiLU(), nn.Linear(96, 1))

    @staticmethod
    def _cell_geometry(cell: Cell | None, spec: LatticeSpec, device, dtype):
        if cell is None:
            return torch.zeros(3, device=device, dtype=dtype)
        rr, cc = spec.coord(cell)
        lh, lw = spec.lattice_shape
        r = rr / max(lh - 1, 1) * 2 - 1
        c = cc / max(lw - 1, 1) * 2 - 1
        d = float(cell.dim) / 2.0
        return torch.tensor([r, c, d], device=device, dtype=dtype)

    def encode(self, state: torch.Tensor, time: torch.Tensor, label: torch.Tensor, topo: torch.Tensor):
        # state [B,11,LH,LW], topo [B,4] = beta0,beta1,ncells_norm,progress_level
        c = self.cond(torch.cat([self.time(time), self.label(label), topo], dim=-1))
        h = self.in_conv(state)
        for b in self.blocks:
            h = b(h, c)
        h = F.silu(self.out_norm(h))
        dom = state[:, 4:8].sum(dim=1, keepdim=True).clamp(max=1.0)
        den = dom.sum(dim=(2, 3)).clamp_min(1.0)
        g = (h * dom).sum(dim=(2, 3)) / den
        rate = F.softplus(self.rate_head(torch.cat([g, c], dim=-1)).squeeze(-1)) + 1e-5
        stop = self.stop_head(torch.cat([g, c], dim=-1)).squeeze(-1)
        return h, g, c, rate, stop

    def score_candidates(
        self,
        feature_map: torch.Tensor,
        global_feat: torch.Tensor,
        cond: torch.Tensor,
        candidates: Sequence[ReverseEvent],
        spec: LatticeSpec,
    ) -> torch.Tensor:
        if feature_map.shape[0] != 1:
            raise ValueError("candidate scoring currently expects batch size 1")
        if not candidates:
            return torch.empty(0, device=feature_map.device, dtype=feature_map.dtype)
        feat = feature_map[0]
        rows = []
        for ev in candidates:
            sr, sc = spec.coord(ev.sigma)
            fs = feat[:, sr, sc]
            if ev.tau is None:
                ft = torch.zeros_like(fs)
            else:
                tr, tc = spec.coord(ev.tau)
                ft = feat[:, tr, tc]
            eidx = torch.tensor(self.EVENT_INDEX[ev.kind], device=feat.device)
            ee = self.event_emb(eidx)
            geom = torch.cat([
                self._cell_geometry(ev.sigma, spec, feat.device, feat.dtype),
                self._cell_geometry(ev.tau, spec, feat.device, feat.dtype),
            ])
            rows.append(torch.cat([fs, ft, global_feat[0], cond[0], ee, geom]))
        mat = torch.stack(rows, dim=0)
        return self.candidate_mlp(mat).squeeze(-1)

    def forward(self, state, time, label, topo, candidates, spec, include_stop: bool = False):
        fmap, g, c, rate, stop = self.encode(state, time, label, topo)
        logits = self.score_candidates(fmap, g, c, candidates, spec)
        if include_stop:
            logits = torch.cat([logits, stop[:1]], dim=0)
        return logits, rate
