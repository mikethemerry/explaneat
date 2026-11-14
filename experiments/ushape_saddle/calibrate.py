"""Fix the nonlinear strengths before any network is trained.

    uv run python experiments/ushape_saddle/calibrate.py

For each dataset and each beta on the grid: test AUC of the oracle (the true logit), of L2 logistic
regression and of gradient boosting (HistGradientBoostingClassifier, random_state 0), both fit on
the z-scored training rows. Chosen: the smallest beta at which, on the test split, oracle - LR >= 0.10 (twice the 0.05 floor
of the brief) and gradient boosting - LR >= 0.05 (the signal is learnable by a standard flexible
learner at n = 1,500, so a PropNEAT null is not down to a weak signal). Prints the table and writes calibration.json (no database access).
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

GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
MARGIN = 0.10
GB_MARGIN = 0.05


def refs(d):
    lr = LogisticRegression(max_iter=5000).fit(d["X_train"], d["y_train"])
    gb = HistGradientBoostingClassifier(random_state=0).fit(d["X_train"], d["y_train"])
    te = d["test_rows"]
    return dict(oracle=float(roc_auc_score(d["y_test"], d["f"][te])),
                lr=float(roc_auc_score(d["y_test"], lr.predict_proba(d["X_test"])[:, 1])),
                gb=float(roc_auc_score(d["y_test"], gb.predict_proba(d["X_test"])[:, 1])),
                positive_share=float(d["y_all"].mean()))


def main():
    out = {"grid": GRID, "margin": MARGIN, "gb_margin": GB_MARGIN, "rows": [], "chosen": {}}
    print(f"{'task':8s} {'beta':>5s} {'oracle':>7s} {'LR':>7s} {'GB':>7s} {'or-LR':>7s} {'pos':>5s}")
    for task, key in (("u", "beta_u"), ("saddle", "beta_s")):
        chosen = None
        for b in GRID:
            d = data.prepare(task, **{key: b})
            r = refs(d)
            r.update(task=task, beta=b)
            out["rows"].append(r)
            gap = r["oracle"] - r["lr"]
            print(f"{task:8s} {b:5.2f} {r['oracle']:7.3f} {r['lr']:7.3f} {r['gb']:7.3f} {gap:7.3f} {r['positive_share']:5.2f}")
            if chosen is None and gap >= MARGIN and r["gb"] - r["lr"] >= GB_MARGIN:
                chosen = b
        out["chosen"][key] = chosen
    # the combined dataset at the chosen strengths
    d = data.prepare("both", beta_u=out["chosen"]["beta_u"], beta_s=out["chosen"]["beta_s"])
    r = refs(d)
    r.update(task="both", beta=None)
    out["rows"].append(r)
    print(f"{'both':8s} {'-':>5s} {r['oracle']:7.3f} {r['lr']:7.3f} {r['gb']:7.3f} {r['oracle'] - r['lr']:7.3f} {r['positive_share']:5.2f}")
    print("chosen:", out["chosen"])
    (HERE / "calibration.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
