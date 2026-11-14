"""Reference AUCs for each dataset: the oracle (true logit), L2 logistic regression and gradient
boosting (HistGradientBoostingClassifier, random_state 0), fit on the z-scored training rows, with
10,000-resample bootstrap CIs on the test split (seed 42, as the 10-seed study). No database access.

    uv run python experiments/ushape_saddle/references.py      -> references.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402


def bootstrap_ci(y, p, n=10_000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    out = {}
    for task in data.TASKS:
        d = data.prepare(task)
        te = d["test_rows"]
        res = dict(label=data.LABELS[task], ground_truth=d["meta"]["ground_truth"], beta_u=d["meta"]["beta_u"],
                   beta_s=d["meta"]["beta_s"], n_train=d["meta"]["n_train"], n_test=d["meta"]["n_test"],
                   positive_rate_train=float(d["y_train"].mean()), positive_rate_test=float(d["y_test"].mean()),
                   terms=list(d["terms"]))
        preds = {"oracle": (d["f"][d["train_rows"]], d["f"][te])}
        lr = LogisticRegression(max_iter=5000).fit(d["X_train"], d["y_train"])
        gb = HistGradientBoostingClassifier(random_state=0).fit(d["X_train"], d["y_train"])
        preds["l2_logistic"] = (lr.predict_proba(d["X_train"])[:, 1], lr.predict_proba(d["X_test"])[:, 1])
        preds["gradient_boosting"] = (gb.predict_proba(d["X_train"])[:, 1], gb.predict_proba(d["X_test"])[:, 1])
        for k, (ptr, pte) in preds.items():
            lo, hi = bootstrap_ci(d["y_test"], pte)
            res[k] = dict(train_auc=float(roc_auc_score(d["y_train"], ptr)), test_auc=float(roc_auc_score(d["y_test"], pte)),
                          test_auc_ci=[lo, hi])
        res["l2_logistic"]["coef_z"] = {n: float(c) for n, c in zip(d["names"], lr.coef_[0])}
        out[task] = res
        print(f"{task:7s} {res['label']}: oracle {res['oracle']['test_auc']:.3f}  LR {res['l2_logistic']['test_auc']:.3f} "
              f"[{res['l2_logistic']['test_auc_ci'][0]:.3f},{res['l2_logistic']['test_auc_ci'][1]:.3f}]  "
              f"GB {res['gradient_boosting']['test_auc']:.3f}  pos {res['positive_rate_train']:.3f}")
    (HERE / "references.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
