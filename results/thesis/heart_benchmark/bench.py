"""Heart benchmark on the paper split: CV on train only (5-fold, seed 42), test read once."""
import json, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from interpret.glassbox import ExplainableBoostingClassifier
d = np.load(__file__.replace("bench.py", "heart.npz")); Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
cv = StratifiedKFold(5, shuffle=True, random_state=42)
M = {
 "Sparse logistic regression (L1)": (LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000), {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}),
 "Decision tree": (DecisionTreeClassifier(random_state=42), {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [5, 10, 20]}),
 "Random forest": (RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=2), {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5, 10]}),
 "EBM": (ExplainableBoostingClassifier(random_state=42), {"interactions": [0, 5, 10], "max_bins": [32, 256]}),
 "MLP (1 hidden layer)": (MLPClassifier(max_iter=3000, random_state=42), {"hidden_layer_sizes": [(8,), (16,), (32,)], "alpha": [1e-3, 1e-2, 1e-1, 1.0]}),
}
rs = np.random.RandomState(42); idx = [rs.randint(0, len(yte), len(yte)) for _ in range(10000)]
def ci(p):
    v = [roc_auc_score(yte[i], p[i]) for i in idx if len(set(yte[i])) == 2]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out = {}
for name, (est, grid) in M.items():
    g = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=1).fit(Xtr, ytr)
    p = g.predict_proba(Xte)[:, 1]
    out[name] = {"params": g.best_params_, "cv_auc": float(g.best_score_), "test_auc": float(roc_auc_score(yte, p)), "test_auc_ci95": ci(p)}
    print(f"{name:34} cv {g.best_score_:.3f}  test {out[name]['test_auc']:.3f}  {g.best_params_}", flush=True)
json.dump(out, open(__file__.replace("bench.py", "bench.json"), "w"), indent=1)
