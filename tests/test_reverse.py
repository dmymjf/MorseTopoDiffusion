import numpy as np
import torch

from morsetopo.complexes.cubical import CubicalComplex2D
from morsetopo.complexes.tensor import complex_channels
from morsetopo.data.synthetic import center, donut
from morsetopo.data.wafer import circular_wafer_mask
from morsetopo.diffusion.forward import MorseForwardConfig, MorseForwardProcess
from morsetopo.models.event_net import MorseEventNet
from morsetopo.operators.reverse import ReverseEvent, ReverseEventOperator
from morsetopo.training.losses import reverse_step_loss
from morsetopo.training.reverse_dataset import sample_reverse_step


def _same(a, b):
    return all(a.cells[d] == b.cells[d] for d in (0, 1, 2))


def test_exact_inverse_replay_recovers_complex():
    size = 18
    domain_mask = circular_wafer_mask((size, size))
    mask = donut((size, size), r_outer=6, r_inner=2) & domain_mask
    traj = MorseForwardProcess(MorseForwardConfig(seed=3)).simulate(mask)
    domain = CubicalComplex2D.from_binary(domain_mask)
    cur = CubicalComplex2D(shape=(size, size))
    op = ReverseEventOperator(domain)
    for e in reversed(traj.events):
        target = ReverseEvent.from_forward(e)
        candidates, idx = op.exact_inverse_candidates(cur, target)
        op.apply(cur, candidates[idx])
    expected = CubicalComplex2D.from_binary(mask)
    assert _same(cur, expected)


def test_mixed_dimensional_tensor_encoding():
    k = CubicalComplex2D.from_binary(center((12, 12), radius=3))
    x = complex_channels(k, k)
    assert x.shape == (11, 25, 25)
    assert x[:4].sum() == sum(k.n_cells)


def test_neural_reverse_loss_backward():
    size = 14
    domain_mask = circular_wafer_mask((size, size))
    mask = center((size, size), radius=3) & domain_mask
    label = np.zeros(8, dtype=np.float32); label[0] = 1
    traj = MorseForwardProcess(MorseForwardConfig(seed=9)).simulate(mask, label)
    step = sample_reverse_step(traj, domain_mask, np.random.default_rng(4))
    model = MorseEventNet(num_labels=8, channels=8, cond_dim=32, event_dim=8)
    d = reverse_step_loss(model, step)
    d["loss"].backward()
    assert torch.isfinite(d["loss"])
