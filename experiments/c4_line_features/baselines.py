"""References on the same split for the line-features base test: logistic regression and
sklearn HistGradientBoosting (default settings, as in complex_real/baselines.py) on each input
set, z-scored with the training-row scaler. The cells-only numbers are copied from
experiments/complex_real/baselines.json (not refit).

    uv run python experiments/c4_line_features/baselines.py     # -> baselines.json
"""

import json
import sys
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lfdata import prepare  # noqa: E402

out = {}
for inputs in ("lines", "lines_cells"):
    d = prepare(inputs)
    ytr, yte = d["y_train"], d["y_test"]
    lr = LogisticRegression(max_iter=5000, C=1.0).fit(d["X_train"], ytr)
    gbm = HistGradientBoostingClassifier(random_state=0).fit(d["X_train"], ytr)
    out[inputs] = dict(
        n_inputs=len(d["names"]), n_train=int(len(ytr)), n_test=int(len(yte)),
        lr=dict(train=float(roc_auc_score(ytr, lr.decision_function(d["X_train"]))),
                test=float(roc_auc_score(yte, lr.decision_function(d["X_test"])))),
        gradient_boosting=dict(train=float(roc_auc_score(ytr, gbm.predict_proba(d["X_train"])[:, 1])),
                               test=float(roc_auc_score(yte, gbm.predict_proba(d["X_test"])[:, 1]))),
        lr_coef={n: round(float(c), 4) for n, c in zip(d["names"], lr.coef_[0])} if inputs == "lines" else None,
    )
    print(inputs, json.dumps({k: v for k, v in out[inputs].items() if k != "lr_coef"}))
cells = json.loads((HERE.parent / "complex_real" / "baselines.json").read_text())["c4"]
out["cells"] = dict(n_inputs=42, lr=cells["logistic_regression"] if "logistic_regression" in cells else cells.get("lr"),
                    gradient_boosting=cells["gradient_boosting"], source="experiments/complex_real/baselines.json (not refit)")
(HERE / "baselines.json").write_text(json.dumps(out, indent=2) + "\n")
