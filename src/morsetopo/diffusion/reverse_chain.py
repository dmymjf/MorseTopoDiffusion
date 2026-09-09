from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from morsetopo.complexes.cubical import CubicalComplex2D
from morsetopo.complexes.tensor import LatticeSpec, complex_channels
from morsetopo.models.event_net import MorseEventNet
from morsetopo.operators.reverse import ReverseEvent, ReverseEventOperator


@dataclass
class ReverseSampleConfig:
    max_events: int = 3000
    temperature: float = 1.0
    greedy: bool = False
    seed: int = 42


class ReverseChainSampler:
    def __init__(self, model: MorseEventNet, domain_mask: np.ndarray, config: ReverseSampleConfig | None = None, device="cpu"):
        self.model = model
        self.cfg = config or ReverseSampleConfig()
        self.device = device
        self.rng = np.random.default_rng(self.cfg.seed)
        self.domain = CubicalComplex2D.from_binary(np.asarray(domain_mask, dtype=bool))
        self.op = ReverseEventOperator(self.domain)
        self.spec = LatticeSpec(*self.domain.shape)

    @staticmethod
    def _topo(k: CubicalComplex2D, progress: float) -> np.ndarray:
        b0, b1 = k.betti()
        return np.asarray([b0 / 32.0, b1 / 16.0, sum(k.n_cells) / 1000.0, progress], dtype=np.float32)

    def sample(self, label: np.ndarray):
        k = CubicalComplex2D(shape=self.domain.shape)
        history: List[ReverseEvent] = []
        y = torch.from_numpy(np.asarray(label, dtype=np.float32))[None].to(self.device)
        self.model.eval()
        with torch.no_grad():
            for step in range(self.cfg.max_events):
                progress = step / max(self.cfg.max_events - 1, 1)
                candidates = self.op.enumerate(k, include_all_kinds=True)
                x = torch.from_numpy(complex_channels(k, self.domain))[None].to(self.device)
                t = torch.tensor([progress], dtype=torch.float32, device=self.device)
                topo = torch.from_numpy(self._topo(k, progress))[None].to(self.device)
                logits, _rate = self.model(x, t, y, topo, candidates, self.spec, include_stop=True)
                logits = logits / max(self.cfg.temperature, 1e-6)
                if self.cfg.greedy:
                    idx = int(torch.argmax(logits).item())
                else:
                    probs = F.softmax(logits, dim=0).cpu().numpy()
                    idx = int(self.rng.choice(len(probs), p=probs))
                if idx == len(candidates):
                    break
                ev = candidates[idx]
                self.op.apply(k, ev)
                history.append(ev)
        return k, history
