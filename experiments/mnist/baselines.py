"""Context baselines on the exact MNIST splits the PropNEAT runs use.

mnist38: logistic regression (L2, C=1) and a small gradient-boosting model
         (HistGradientBoostingClassifier, 100 iterations, max_leaf_nodes 15).
mnist10: multinomial logistic regression (L2, C=1) and the same small GBM
         (one-vs-rest trees inside HistGradientBoosting, multinomial loss).
One thread.  Writes experiments/mnist/baselines.json.

    uv run python experiments/mnist/baselines.py
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import prepare  # noqa: E402
from run_seed import TASKS  # noqa: E402


def score(model, X, y, binary):
    p = model.predict_proba(X)
    out = {"acc": float(accuracy_score(y, p.argmax(1))), "ce": float(log_loss(y, p, labels=model.classes_))}
    if binary:
        out["auc"] = float(roc_auc_score(y, p[:, 1]))
    return out


def main():
    res = {}
    for task in ("mnist38", "mnist10"):
        d = prepare(task, TASKS[task]["n_train"])
        binary = task == "mnist38"
        y_tr, y_te = d["y_train"].astype(int), d["y_test"].astype(int)
        models = {
            "logistic_regression": LogisticRegression(C=1.0, max_iter=5000),
            "gbm_small": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                                        learning_rate=0.1, random_state=0),
        }
        res[task] = {"n_train": int(len(y_tr)), "n_test": int(len(y_te))}
        for name, m in models.items():
            t = time.time()
            m.fit(d["X_train"], y_tr)
            res[task][name] = dict(train=score(m, d["X_train"], y_tr, binary),
                                   test=score(m, d["X_test"], y_te, binary),
                                   fit_seconds=round(time.time() - t, 1), params=str(m))
            print(task, name, json.dumps(res[task][name]["test"]), flush=True)
    (HERE / "baselines.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
