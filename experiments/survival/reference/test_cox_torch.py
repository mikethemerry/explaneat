"""Verification: does our torch Cox loss agree with lifelines / scikit-survival?

Three checks:
  1. Breslow partial log-likelihood == sksurv's objective at the same beta
  2. Efron  partial log-likelihood == lifelines' log-likelihood at the same beta
  3. Breslow baseline hazard + S(t|x) == sksurv's predict_survival_function
  4. Shift invariance: g -> g + c leaves the loss unchanged
  5. Gradient descent on a linear model recovers the Cox MLE coefficients
"""
import numpy as np
import torch
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from cox_torch import CoxRiskSets, cox_ph_loss, partial_log_likelihood, BreslowBaseline, concordance

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- data
df = load_rossi()                     # 432 rows, week/arrest, many tied times
X = df.drop(columns=["week", "arrest"]).to_numpy(float)
t = df["week"].to_numpy(float)
e = df["arrest"].to_numpy(float)
Xs = (X - X.mean(0)) / X.std(0)
n_tied = len(t) - len(np.unique(t))
print(f"rossi: n={len(t)}  events={int(e.sum())}  distinct times={len(np.unique(t))}  (ties present: {n_tied>0})")

rs = CoxRiskSets(t, e)

# ---------------------------------------------------------------- 1. Breslow vs sksurv
y = Surv.from_arrays(event=e.astype(bool), time=t)
cox_bres = CoxPHSurvivalAnalysis(ties="breslow", alpha=1e-8).fit(Xs, y)
beta_b = cox_bres.coef_
g_b = torch.tensor(Xs @ beta_b)

ours_b = float(partial_log_likelihood(g_b, rs, ties="breslow"))
# sksurv exposes the objective via its internal loss; recompute independently:
def ref_breslow(g, t, e):
    ll = 0.0
    for tau in np.unique(t[e == 1]):
        D = (t == tau) & (e == 1)
        R = t >= tau
        d = D.sum()
        ll += g[D].sum() - d * np.log(np.exp(g[R]).sum())
    return ll
ref_b = ref_breslow(Xs @ beta_b, t, e)
print(f"\n1. Breslow  ours={ours_b:.8f}  naive-reference={ref_b:.8f}  diff={abs(ours_b-ref_b):.2e}")
assert abs(ours_b - ref_b) < 1e-8, "Breslow mismatch"

# ---------------------------------------------------------------- 2. Efron vs lifelines
cph = CoxPHFitter(penalizer=0.0)
dfz = df.copy()
for j, c in enumerate(df.drop(columns=["week", "arrest"]).columns):
    dfz[c] = Xs[:, j]
cph.fit(dfz, duration_col="week", event_col="arrest")
beta_e = cph.params_.to_numpy()
g_e = torch.tensor(Xs @ beta_e)

ours_e = float(partial_log_likelihood(g_e, rs, ties="efron"))
ll_lifelines = cph.log_likelihood_
print(f"2. Efron    ours={ours_e:.8f}  lifelines={ll_lifelines:.8f}  diff={abs(ours_e-ll_lifelines):.2e}")
assert abs(ours_e - ll_lifelines) < 1e-6, "Efron mismatch vs lifelines"

# also: naive Efron reference
def ref_efron(g, t, e):
    ll = 0.0
    for tau in np.unique(t[e == 1]):
        D = (t == tau) & (e == 1); R = t >= tau
        d = int(D.sum())
        risk = np.exp(g[R]).sum(); tie = np.exp(g[D]).sum()
        ll += g[D].sum() - sum(np.log(risk - (l / d) * tie) for l in range(d))
    return ll
print(f"   naive Efron reference={ref_efron(Xs @ beta_e, t, e):.8f}")
assert abs(ours_e - ref_efron(Xs @ beta_e, t, e)) < 1e-8

# ---------------------------------------------------------------- 3. Breslow baseline / S(t|x)
bl = BreslowBaseline().fit(Xs @ beta_b, t, e)
eval_times = np.array([10., 20., 30., 40., 50.])
S_ours = bl.survival(Xs @ beta_b, eval_times)
surv_fns = cox_bres.predict_survival_function(Xs[:5])
S_ref = np.array([[fn(tt) for tt in eval_times] for fn in surv_fns])
maxdiff = np.abs(S_ours[:5] - S_ref).max()
print(f"3. S(t|x)   max |ours - sksurv| over 5 subjects x 5 times = {maxdiff:.2e}")
assert maxdiff < 1e-8, "survival function mismatch"

# ---------------------------------------------------------------- 4. shift invariance
l0 = float(cox_ph_loss(g_b, rs, ties="efron"))
l1 = float(cox_ph_loss(g_b + 7.3, rs, ties="efron"))
print(f"4. shift    loss(g)={l0:.10f}  loss(g+7.3)={l1:.10f}  diff={abs(l0-l1):.2e}")
assert abs(l0 - l1) < 1e-9, "loss is not shift-invariant"

# ---------------------------------------------------------------- 5. optimisation recovers the MLE
Xt = torch.tensor(Xs)
w = torch.zeros(Xs.shape[1], requires_grad=True)
opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
def closure():
    opt.zero_grad()
    loss = cox_ph_loss(Xt @ w, rs, ties="efron")
    loss.backward()
    return loss
opt.step(closure)
w_hat = w.detach().numpy()
print("\n5. MLE recovery (Efron), torch full-batch vs lifelines:")
for name, a, b in zip(df.drop(columns=['week','arrest']).columns, w_hat, beta_e):
    print(f"     {name:<8} torch={a:+.6f}  lifelines={b:+.6f}  diff={abs(a-b):.2e}")
assert np.abs(w_hat - beta_e).max() < 1e-4, "did not recover the Cox MLE"

# ---------------------------------------------------------------- 6. C-index sanity
from sksurv.metrics import concordance_index_censored
c_ours = concordance(Xs @ beta_e, t, e)
c_ref = concordance_index_censored(e.astype(bool), t, Xs @ beta_e)[0]
print(f"\n6. C-index  ours={c_ours:.6f}  sksurv={c_ref:.6f}  diff={abs(c_ours-c_ref):.2e}")
assert abs(c_ours - c_ref) < 1e-9

print("\nALL CHECKS PASSED")
