"""Adversarial attacks on ConvNEAT networks (explaneat/conv/adversarial.py)."""

import numpy as np
import pytest
import torch

from explaneat.conv import adversarial as ADV
from explaneat.conv.data import make_synthetic
from explaneat.conv.genome import ConvNeatConfig, InnovationTracker
from explaneat.conv.mutation import random_genome_by_mutation

torch.set_num_threads(1)
CFG = ConvNeatConfig(in_channels=1, image_hw=(16, 16), n_classes=1, max_level=2)


def _genomes(n, seed):
    tr = InnovationTracker(CFG.n_out_maps)
    rng = np.random.default_rng(seed)
    return [random_genome_by_mutation(i, CFG, tr, rng, n_steps=int(rng.integers(5, 25))) for i in range(n)]


@pytest.fixture(scope="module")
def data():
    d = make_synthetic(64, seed=5)
    return torch.tensor(d["X"]), torch.tensor(d["y"])


@pytest.fixture(scope="module")
def model():
    g = _genomes(1, 3)[0]
    return ADV.ConvNeatModel(g, CFG)


def test_gradient_parity_propneat_vs_naive(data):
    X, _ = data
    for g in _genomes(15, 11):
        r = ADV.input_grad_parity(g, CFG, X[:8].numpy())
        assert r["maxabs_f64"] < 1e-9 * max(1.0, r["grad_scale"])
        assert r["rel_f32"] < 1e-4


@pytest.mark.parametrize("attack,norm", [("fgsm", "linf"), ("pgd_linf", "linf"), ("pgd_l2", "l2")])
def test_attacks_respect_ball_and_pixel_range(model, data, attack, norm):
    X, y = data
    for eps in (0.01, 0.1, 0.5) if norm == "linf" else (0.1, 1.0, 4.0):
        xa = ADV.attack_batched(model, attack, X, y, eps, seed=1)
        assert xa.min() >= 0.0 and xa.max() <= 1.0
        d = (xa - X).flatten(1)
        n = d.abs().max(1).values if norm == "linf" else d.norm(dim=1)
        assert float(n.max()) <= eps * (1 + 1e-5) + 1e-7


def test_per_example_eps(model, data):
    X, y = data
    eps = torch.linspace(0.0, 0.2, len(X))
    xa = ADV.attack_batched(model, "pgd_linf", X, y, eps, seed=0)
    assert torch.all((xa - X).flatten(1).abs().max(1).values <= eps + 1e-7)


def test_gaussian_sigma_zero_leaves_accuracy(model, data):
    X, y = data
    a0 = ADV.accuracy(model, X, y)
    xn = ADV.gaussian_noise(X, 0.0)
    assert torch.equal(xn, X)
    assert ADV.accuracy(model, xn, y) == a0
    xs = ADV.gaussian_noise(X, 0.3, torch.Generator().manual_seed(0))
    assert xs.min() >= 0 and xs.max() <= 1


def test_pgd_at_least_as_strong_as_fgsm_and_monotone(model, data):
    X, y = data
    accs = []
    for eps in (0.0, 0.05, 0.2):
        accs.append(ADV.accuracy(model, ADV.attack_batched(model, "pgd_linf", X, y, eps, seed=0), y))
    assert accs[0] == ADV.accuracy(model, X, y)
    assert accs[2] <= accs[1] + 1e-9 <= accs[0] + 1e-9


def test_min_perturbation_flips_within_norm(model, data):
    X, y = data
    ok = ADV.correct_mask(model, X, y)
    Xc, yc = X[ok][:16], y[ok][:16]
    for norm, emax in (("linf", 1.0), ("l2", 16.0)):
        nrm, xa, found = ADV.min_perturbation(model, Xc, yc, norm, emax, n_bisect=6, steps=10)
        lo = model(xa)
        assert torch.all(ADV.margin(lo, yc)[found] > 0)
        assert xa.min() >= 0 and xa.max() <= 1


def test_universal_within_ball(model, data):
    X, y = data
    d = ADV.universal_linf(model, X, y, 0.05, steps=10)
    assert d.shape == (1, 1, 16, 16) and float(d.abs().max()) <= 0.05 + 1e-7


def test_synthetic_with_masks_replays_generator():
    X, C, M = ADV.synthetic_with_masks(50, 7)
    d = make_synthetic(50, 7)
    assert np.array_equal(X, d["X"]) and np.array_equal(C, d["concepts"])
    assert np.all(M.any(axis=(2, 3)) == C.astype(bool))


def test_ghost_scores_find_a_drawn_bar():
    d = torch.zeros(4, 1, 16, 16)
    d[:, 0, 7, 4:11] = 0.05  # a faint horizontal bar
    d = d + 0.002 * torch.randn(d.shape, generator=torch.Generator().manual_seed(0))
    gs = ADV.ghost_scores(d, n_null=20)
    assert np.all(gs["hbar"]["draw_z"] > 3)
    assert np.median(gs["hbar"]["draw_z"]) > np.median(gs["vbar"]["draw_z"])
