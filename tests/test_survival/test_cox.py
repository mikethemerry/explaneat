"""Verification of explaneat.survival against lifelines / scikit-survival.

Port of the reference ``test_cox_torch.py`` (2026-08-20 brief) on the Rossi
recidivism data (432 rows, 49 distinct times, so ties are exercised):

1. Breslow partial log-likelihood == naive reference
2. Efron partial log-likelihood == lifelines (and a naive reference)
3. Breslow baseline S(t|x) == sksurv predict_survival_function
4. Shift invariance
5. Full-batch LBFGS on a linear model recovers the lifelines Cox MLE
6. Harrell's C == sksurv concordance_index_censored
"""

import numpy as np
import pytest
import torch

lifelines = pytest.importorskip("lifelines")
sksurv = pytest.importorskip("sksurv")

from lifelines import CoxPHFitter  # noqa: E402
from lifelines.datasets import load_rossi  # noqa: E402
from sksurv.linear_model import CoxPHSurvivalAnalysis  # noqa: E402
from sksurv.metrics import concordance_index_censored  # noqa: E402
from sksurv.util import Surv  # noqa: E402

from explaneat.survival import (  # noqa: E402
    BreslowBaseline,
    CoxRiskSets,
    concordance,
    cox_ph_loss,
    null_cox_loss,
    partial_log_likelihood,
)


@pytest.fixture(scope="module")
def rossi():
    df = load_rossi()
    cols = [c for c in df.columns if c not in ("week", "arrest")]
    X = df[cols].to_numpy(float)
    t = df["week"].to_numpy(float)
    e = df["arrest"].to_numpy(float)
    Xs = (X - X.mean(0)) / X.std(0)
    return df, cols, Xs, t, e


def ref_breslow(g, t, e):
    ll = 0.0
    for tau in np.unique(t[e == 1]):
        D = (t == tau) & (e == 1)
        R = t >= tau
        ll += g[D].sum() - D.sum() * np.log(np.exp(g[R]).sum())
    return ll


def ref_efron(g, t, e):
    ll = 0.0
    for tau in np.unique(t[e == 1]):
        D = (t == tau) & (e == 1)
        R = t >= tau
        d = int(D.sum())
        risk, tie = np.exp(g[R]).sum(), np.exp(g[D]).sum()
        ll += g[D].sum() - sum(np.log(risk - (l / d) * tie) for l in range(d))
    return ll


@pytest.fixture(scope="module")
def lifelines_fit(rossi):
    df, cols, Xs, t, e = rossi
    dfz = df.copy()
    for j, c in enumerate(cols):
        dfz[c] = Xs[:, j]
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(dfz, duration_col="week", event_col="arrest")
    return cph


def test_rossi_has_ties(rossi):
    _, _, _, t, _ = rossi
    assert len(np.unique(t)) < len(t)
    assert CoxRiskSets(t, rossi[4]).has_ties


def test_breslow_matches_naive_reference(rossi):
    _, _, Xs, t, e = rossi
    cox = CoxPHSurvivalAnalysis(ties="breslow", alpha=1e-8).fit(Xs, Surv.from_arrays(e.astype(bool), t))
    g = Xs @ cox.coef_
    ours = float(partial_log_likelihood(torch.tensor(g), CoxRiskSets(t, e), ties="breslow"))
    assert abs(ours - ref_breslow(g, t, e)) < 1e-8


def test_efron_matches_lifelines(rossi, lifelines_fit):
    _, _, Xs, t, e = rossi
    g = Xs @ lifelines_fit.params_.to_numpy()
    ours = float(partial_log_likelihood(torch.tensor(g), CoxRiskSets(t, e), ties="efron"))
    assert abs(ours - lifelines_fit.log_likelihood_) < 1e-6
    assert abs(ours - ref_efron(g, t, e)) < 1e-8


def test_efron_equals_breslow_without_ties():
    rng = np.random.default_rng(3)
    t = rng.exponential(size=200)
    e = (rng.uniform(size=200) < 0.6).astype(float)
    g = torch.tensor(rng.normal(size=200))
    rs = CoxRiskSets(t, e)
    assert not rs.has_ties
    assert abs(float(cox_ph_loss(g, rs, ties="efron")) - float(cox_ph_loss(g, rs, ties="breslow"))) < 1e-12


def test_efron_matches_reference_with_heavy_ties():
    """Integer times on a 5-point grid: many large tie blocks."""
    rng = np.random.default_rng(11)
    t = rng.integers(1, 6, size=300).astype(float)
    e = (rng.uniform(size=300) < 0.5).astype(float)
    g = rng.normal(size=300)
    ours = float(partial_log_likelihood(torch.tensor(g), CoxRiskSets(t, e), ties="efron"))
    assert abs(ours - ref_efron(g, t, e)) < 1e-8


def test_efron_gradient_matches_finite_difference():
    rng = np.random.default_rng(5)
    t = rng.integers(1, 20, size=60).astype(float)
    e = (rng.uniform(size=60) < 0.7).astype(float)
    rs = CoxRiskSets(t, e)
    g = torch.tensor(rng.normal(size=60), requires_grad=True)
    cox_ph_loss(g, rs, ties="efron").backward()
    num = np.zeros(60)
    eps = 1e-6
    for i in range(60):
        gp = g.detach().clone(); gp[i] += eps
        gm = g.detach().clone(); gm[i] -= eps
        num[i] = (float(cox_ph_loss(gp, rs)) - float(cox_ph_loss(gm, rs))) / (2 * eps)
    np.testing.assert_allclose(g.grad.numpy(), num, atol=1e-7)


def test_breslow_baseline_matches_sksurv(rossi):
    _, _, Xs, t, e = rossi
    cox = CoxPHSurvivalAnalysis(ties="breslow", alpha=1e-8).fit(Xs, Surv.from_arrays(e.astype(bool), t))
    g = Xs @ cox.coef_
    bl = BreslowBaseline().fit(g, t, e)
    times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    S_ours = bl.survival(g, times)[:5]
    S_ref = np.array([[fn(tt) for tt in times] for fn in cox.predict_survival_function(Xs[:5])])
    assert np.abs(S_ours - S_ref).max() < 1e-8
    np.testing.assert_allclose(bl.risk_at(g, 30.0), 1 - bl.survival(g, [30.0])[:, 0])


def test_shift_invariance(rossi):
    _, _, Xs, t, e = rossi
    g = torch.tensor(Xs @ np.linspace(-0.3, 0.3, Xs.shape[1]))
    rs = CoxRiskSets(t, e)
    for ties in ("efron", "breslow"):
        assert abs(float(cox_ph_loss(g, rs, ties=ties)) - float(cox_ph_loss(g + 7.3, rs, ties=ties))) < 1e-9


def test_null_loss_is_mean_log_risk_set_size(rossi):
    _, _, _, t, e = rossi
    rs = CoxRiskSets(t, e)
    expected = np.mean([np.log((t >= ti).sum()) for ti in t[e == 1]])
    assert abs(null_cox_loss(rs, ties="breslow") - expected) < 1e-12
    assert null_cox_loss(rs, ties="efron") < null_cox_loss(rs, ties="breslow")


def test_lbfgs_recovers_cox_mle(rossi, lifelines_fit):
    _, _, Xs, t, e = rossi
    rs = CoxRiskSets(t, e)
    Xt = torch.tensor(Xs)
    w = torch.zeros(Xs.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = cox_ph_loss(Xt @ w, rs, ties="efron")
        loss.backward()
        return loss

    opt.step(closure)
    assert np.abs(w.detach().numpy() - lifelines_fit.params_.to_numpy()).max() < 1e-4


def test_concordance_matches_sksurv(rossi, lifelines_fit):
    _, _, Xs, t, e = rossi
    r = Xs @ lifelines_fit.params_.to_numpy()
    ref = concordance_index_censored(e.astype(bool), t, r)[0]
    assert abs(concordance(r, t, e) - ref) < 1e-12
    # tied risk scores score 0.5
    r2 = np.round(r, 1)
    assert abs(concordance(r2, t, e) - concordance_index_censored(e.astype(bool), t, r2)[0]) < 1e-12


def test_risk_sets_validation():
    with pytest.raises(ValueError):
        CoxRiskSets([1.0, 2.0], [0, 0])
    with pytest.raises(ValueError):
        CoxRiskSets([1.0, 2.0], [0, 2])
    rs = CoxRiskSets([1.0, 2.0, 3.0], [1, 0, 1])
    with pytest.raises(ValueError):
        cox_ph_loss(torch.zeros(4, dtype=torch.float64), rs)
