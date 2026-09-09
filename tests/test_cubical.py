import numpy as np

from morsetopo.complexes.cubical import CubicalComplex2D
from morsetopo.data.synthetic import center, donut
from morsetopo.diffusion.forward import MorseForwardProcess
from morsetopo.operators.events import EventKind


def test_center_betti():
    k = CubicalComplex2D.from_binary(center(shape=(24, 24), radius=5))
    assert k.betti() == (1, 0)


def test_donut_betti():
    k = CubicalComplex2D.from_binary(donut(shape=(24, 24), r_outer=8, r_inner=3))
    assert k.betti() == (1, 1)


def test_elementary_collapse_preserves_topology():
    k = CubicalComplex2D.from_binary(center(shape=(18, 18), radius=4))
    before = k.betti()
    sigma, tau = k.free_pairs()[0]
    k.collapse(sigma, tau)
    assert k.betti() == before


def test_forward_reaches_empty_and_critical_count_matches():
    proc = MorseForwardProcess()
    for mask in [center(shape=(20, 20), radius=4), donut(shape=(20, 20), r_outer=7, r_inner=3)]:
        traj = proc.simulate(mask)
        assert traj.terminal_betti == (0, 0)
        assert traj.terminal_cells == (0, 0, 0)
        critical = sum(e.kind != EventKind.REGULAR_COLLAPSE for e in traj.events)
        assert critical == sum(traj.initial_betti)


def test_macro_batching_is_topologically_exact():
    from morsetopo.diffusion.macro_forward import MacroForwardConfig, MorseMacroForwardProcess
    mask = donut(shape=(24, 24), r_outer=8, r_inner=3)
    traj = MorseMacroForwardProcess(MacroForwardConfig(macro_size=8)).simulate(mask)
    assert traj.terminal_betti == (0, 0)
    assert len(traj.macro_events) < traj.micro_event_count
