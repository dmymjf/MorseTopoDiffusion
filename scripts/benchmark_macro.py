#!/usr/bin/env python
from collections import Counter
from morsetopo.data.synthetic import FACTORIES
from morsetopo.diffusion.macro_forward import MacroForwardConfig, MorseMacroForwardProcess

for name in ["center", "donut", "scratch", "edge_ring", "loc", "mixed"]:
    traj = MorseMacroForwardProcess(MacroForwardConfig(macro_size=16, seed=42)).simulate(FACTORIES[name]())
    counts = Counter(m.kind for m in traj.macro_events)
    print(name, "macro_steps=", len(traj.macro_events), "micro_events=", traj.micro_event_count,
          "betti=", traj.initial_betti, "types=", dict(counts))
