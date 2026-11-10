"""Baselines on the complex-real splits (same stratified 80:20 split, seed 42, z-scored inputs):
logistic regression and gradient boosting (sklearn HistGradientBoostingClassifier, default
settings, random_state 0), train and test AUC. Also the domain ground-truth rule for KRK
(exact: draw iff the black king can capture an undefended rook, or stalemate).

    uv run python experiments/complex_real/baselines.py      # -> baselines.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import krk_rules, prepare  # noqa: E402

out = {}
for task in ("krk", "c4"):
    d = prepare(task)
    res = {"n_train": int(len(d["y_train"])), "n_test": int(len(d["y_test"])),
           "positive_share_train": float(d["y_train"].mean())}
    for name, model in (("logistic_regression", LogisticRegression(max_iter=5000)),
                        ("gradient_boosting", HistGradientBoostingClassifier(random_state=0))):
        model.fit(d["X_train"], d["y_train"])
        res[name] = {s: float(roc_auc_score(d[f"y_{s}"], model.predict_proba(d[f"X_{s}"])[:, 1]))
                     for s in ("train", "test")}
    if task == "krk":
        r = krk_rules(d["X_raw"])
        rule = r["rule_draw"].astype(float)
        res["ground_truth_rule"] = {s: float(roc_auc_score(d[f"y_{s}"], rule[d[f"{s}_rows"]]))
                                    for s in ("train", "test")}
        res["ground_truth_rule"]["note"] = ("draw = capture (BK adjacent to the rook, WK not) or "
                                            "stalemate; agrees with every label")
    out[task] = res
    print(task, json.dumps(res))
(HERE / "baselines.json").write_text(json.dumps(out, indent=2) + "\n")
