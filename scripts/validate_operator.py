#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter

from morsetopo.data.synthetic import FACTORIES
from morsetopo.diffusion.forward import MorseForwardProcess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=sorted(FACTORIES), default="donut")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    mask = FACTORIES[args.shape]()
    proc = MorseForwardProcess()
    traj = proc.simulate(mask)
    counts = Counter(e.kind.value for e in traj.events)
    print(f"shape={args.shape}")
    print(f"initial_betti={traj.initial_betti} initial_cells={traj.initial_cells}")
    print(f"events={len(traj.events)} counts={dict(counts)}")
    print(f"topology_events={traj.topology_events} expected={sum(traj.initial_betti)}")
    print(f"terminal_betti={traj.terminal_betti} terminal_cells={traj.terminal_cells}")
    for e in traj.events[:10]:
        print(e)


if __name__ == "__main__":
    main()
