"""
ExplaNEAT survival adapters: drop-in fitness + criterion in the shapes
`explaneat.evaluators.evaluators` and `BackpropPopulation` already expect.

Target signatures (from explaneat/evaluators/evaluators.py):
    fitness_fn(genomes: dict, config, xs, ys, device) -> None   # sets genome.fitness

Convention used here
--------------------
`ys` is [N, 2]: column 0 = observed time, column 1 = event indicator (1 = event,
0 = right-censored). This survives BackpropPopulation's tensor conversion
unchanged (its `unsqueeze(1)` only fires on 1-D targets).

Three blockers in the current codebase, and what to do about them
----------------------------------------------------------------
1. NeuralNeat.forward hardcodes `torch.sigmoid` on the output layer
   (neuralneat.py ~L413). A Cox log-risk needs an IDENTITY output. Add an
   `output_activation` argument to NeuralNeat.__init__ defaulting to
   torch.sigmoid, and use `self.output_activation(...)` at that line.
   Without this you are fitting a risk score squashed into (0,1), which
   caps every hazard ratio at e^1 = 2.72 and cripples the model.

2. BackpropPopulation.backpropagate hardcodes `nn.BCELoss()` twice
   (backproppop.py L152, L158) and ignores the `criterion` it was given.
   Use `self.criterion` in both places.

3. BCELoss and friends are per-sample; the Cox partial likelihood is NOT
   decomposable over samples - it couples every event to its whole risk set.
   So the criterion must take (pred, target) where target is [N,2], not a
   per-row loss. `CoxCriterion` below has that shape.

The upside: ExplaNEAT trains FULL-BATCH, which is the one regime where the
partial likelihood is exact. DeepSurv-style mini-batching truncates each risk
set to the batch; you avoid that approximation entirely.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cox_torch import CoxRiskSets, cox_ph_loss, BreslowBaseline, concordance


# --------------------------------------------------------------------------
# Criterion (for the backprop inner loop)
# --------------------------------------------------------------------------

class CoxCriterion(nn.Module):
    """Cox partial-likelihood criterion with a torch-criterion call signature.

    Caches the risk-set structure per (time, event) pair, so the O(n log n)
    sort happens once per dataset rather than once per genome per epoch.

    Usage inside BackpropPopulation.backpropagate:
        criterion = CoxCriterion(ties="efron")
        loss = criterion(net.forward(xs), ys)     # ys is [N, 2]
    """

    def __init__(self, ties: str = "efron"):
        super().__init__()
        self.ties = ties
        self._cache_key = None
        self._risk_sets = None

    def risk_sets(self, target):
        t = target[:, 0].detach().cpu().numpy()
        e = target[:, 1].detach().cpu().numpy()
        key = (t.shape[0], float(t.sum()), float(e.sum()), float(t[0]), float(t[-1]))
        if key != self._cache_key:
            self._risk_sets = CoxRiskSets(t, e, device=target.device, dtype=target.dtype)
            self._cache_key = key
        return self._risk_sets

    def forward(self, pred, target):
        return cox_ph_loss(pred, self.risk_sets(target), ties=self.ties)


# --------------------------------------------------------------------------
# Fitness functions (for the NEAT selection step)
# --------------------------------------------------------------------------

def _predict(net, xs):
    out = net.forward(xs)
    return out.reshape(-1) if isinstance(out, torch.Tensor) else torch.as_tensor(out).reshape(-1)


def cox_partial_likelihood_fitness(genomes, config, xs, ys, device, ties="efron",
                                   nneat_cls=None):
    """Fitness = negative mean partial log-likelihood, negated so higher is better.

    This is the fitness to prefer. It is the same objective the backprop step
    optimises, so selection and training pull in the same direction, and unlike
    a C-index fitness it does not sacrifice calibration for ranking (the
    DeepHit failure mode).
    """
    from explaneat.core.neuralneat import NeuralNeat as _N
    from explaneat.core.errors import GenomeNotValidError
    NN = nneat_cls or _N

    rs = CoxRiskSets(ys[:, 0].detach().cpu().numpy(),
                     ys[:, 1].detach().cpu().numpy(),
                     device=device, dtype=ys.dtype)

    for genome_id, genome in genomes.items():
        try:
            net = NN(genome, config)
        except GenomeNotValidError:
            genome.fitness = -1e3
            continue
        try:
            with torch.no_grad():
                loss = cox_ph_loss(_predict(net, xs), rs, ties=ties)
            genome.fitness = float(-loss)          # higher is better
        except Exception:
            genome.fitness = -1e3


def cox_concordance_fitness(genomes, config, xs, ys, device, nneat_cls=None):
    """Fitness = Harrell's C on the risk score. Direct analogue of the existing
    auc_fitness. Use as a SECONDARY comparison, not the primary objective -
    optimising rank alone degrades calibration."""
    from explaneat.core.neuralneat import NeuralNeat as _N
    from explaneat.core.errors import GenomeNotValidError
    NN = nneat_cls or _N

    t = ys[:, 0].detach().cpu().numpy()
    e = ys[:, 1].detach().cpu().numpy()

    for genome_id, genome in genomes.items():
        try:
            net = NN(genome, config)
        except GenomeNotValidError:
            genome.fitness = 0.5
            continue
        try:
            with torch.no_grad():
                risk = _predict(net, xs).cpu().numpy()
            c = concordance(risk, t, e)
            genome.fitness = 0.5 if not np.isfinite(c) else float(c)
        except Exception:
            genome.fitness = 0.5


# --------------------------------------------------------------------------
# Post-hoc: turn a fitted genome into Cox-comparable survival curves
# --------------------------------------------------------------------------

def fit_survival_predictor(net, xs_train, ys_train):
    """Freeze the network, fit the Breslow baseline on training data.

    Returns an object with .risk(x), .survival(x, times), .cumulative_hazard(x, times).
    S(t|x) = S0(t) ** exp(g(x)) - structurally identical to a Cox model, with
    the linear predictor replaced by the evolved network.
    """
    with torch.no_grad():
        g_train = _predict(net, xs_train).cpu().numpy()
    t = ys_train[:, 0].detach().cpu().numpy()
    e = ys_train[:, 1].detach().cpu().numpy()
    baseline = BreslowBaseline().fit(g_train, t, e)

    class _Predictor:
        def __init__(self):
            self.baseline = baseline
            self.event_times_ = baseline.times_

        def risk(self, xs):
            with torch.no_grad():
                return _predict(net, xs).cpu().numpy()

        def cumulative_hazard(self, xs, times):
            return baseline.cumulative_hazard(self.risk(xs), times)

        def survival(self, xs, times):
            return baseline.survival(self.risk(xs), times)

    return _Predictor()


# --------------------------------------------------------------------------
# Dataset preparation helper
# --------------------------------------------------------------------------

def make_survival_targets(time, event):
    """Pack (time, event) into the [N, 2] target ExplaNEAT will carry as `ys`."""
    t = np.asarray(time, dtype=np.float64).ravel()
    e = np.asarray(event, dtype=np.float64).ravel()
    return np.column_stack([t, e])
