"""A fully connected multi-output genome (every input -> every output, no hidden
nodes) is multinomial logistic regression.  Trained with PropNEAT's own backprop
step (``BackpropPopulation.backpropagate``, softmax CE, identity outputs,
Adadelta lr 1.5 re-created every call), it must

* follow the same per-epoch loss as ``nn.Linear`` + ``F.cross_entropy`` from the
  same initial weights (so input -> output mapping and output order are right);
* reach close to sklearn ``LogisticRegression`` accuracy given enough epochs.

See ``experiments/mnist/LINEAR_CHECK.md``.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.multiclass import SoftmaxCrossEntropyCriterion
from explaneat.core.neuralneat import NeuralNeat

from tests.test_core.test_multiclass import make_config

EPOCHS_PER_GEN = 5


def _digits():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[:1200], idx[1200:]
    return X[tr], y[tr], X[te], y[te]


def _full_genome(cfg, seed=0):
    import random

    random.seed(seed)
    g = cfg.genome_type(1)
    g.configure_new(cfg.genome_config)
    gc = cfg.genome_config
    assert len(g.connections) == len(gc.input_keys) * len(gc.output_keys)
    return g


def _matrix(g, cfg):
    gc = cfg.genome_config
    W = np.array([[g.connections[(i, o)].weight for o in gc.output_keys] for i in gc.input_keys])
    b = np.array([g.nodes[o].bias for o in gc.output_keys])
    return W, b


def _population(cfg, g, X, y):
    cfg.pop_size = 2
    pop = BackpropPopulation(cfg, X, y.astype(np.float64),
                             criterion=SoftmaxCrossEntropyCriterion(), mask_phantom=True)
    pop.population = {g.key: g}
    return pop


def test_full_genome_backprop_matches_nn_linear_per_epoch():
    Xtr, ytr, _, _ = _digits()
    cfg = make_config(n_in=64, n_out=10, init="full_direct")
    g = _full_genome(cfg)
    W0, b0 = _matrix(g, cfg)
    pop = _population(cfg, g, Xtr, ytr)

    X = torch.as_tensor(Xtr, dtype=torch.float64)
    y = torch.as_tensor(ytr, dtype=torch.long)
    W = torch.nn.Parameter(torch.tensor(W0))
    b = torch.nn.Parameter(torch.tensor(b0))
    for _ in range(20):
        pop.backpropagate(pop.xs, pop.ys, nEpochs=EPOCHS_PER_GEN)
        opt = torch.optim.Adadelta([W, b], lr=1.5)  # fresh each generation, as PropNEAT
        for _ in range(EPOCHS_PER_GEN):
            loss = F.cross_entropy(X @ W + b, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
    g = pop.population[g.key]
    Wg, bg = _matrix(g, cfg)
    np.testing.assert_allclose(Wg, W.detach().numpy(), atol=1e-9)
    np.testing.assert_allclose(bg, b.detach().numpy(), atol=1e-9)


@pytest.mark.slow
def test_full_genome_reaches_logistic_regression_accuracy():
    Xtr, ytr, Xte, yte = _digits()
    lr = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
    lr_test = lr.score(Xte, yte)

    cfg = make_config(n_in=64, n_out=10, init="full_direct")
    g = _full_genome(cfg)
    pop = _population(cfg, g, Xtr, ytr)
    for _ in range(400):  # 2000 epochs
        pop.backpropagate(pop.xs, pop.ys, nEpochs=EPOCHS_PER_GEN)
    net = NeuralNeat(pop.population[g.key], cfg, mask_phantom=True)
    with torch.no_grad():
        acc_tr = float((net.forward(pop.xs).argmax(1).numpy() == ytr).mean())
        acc_te = float((net.forward(torch.as_tensor(Xte)).argmax(1).numpy() == yte).mean())
    assert acc_tr > 0.97
    assert acc_te > lr_test - 0.03, (acc_te, lr_test)
