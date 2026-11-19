"""Summarise runs/*.json into summary.json and a markdown table (stdout).

The chosen arm is the one with the highest median VALIDATION AUC of its
selected genomes; test is read for every arm but never used to choose.

    uv run python experiments/heart_regularise/summarise.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

runs = defaultdict(list)
for p in sorted((HERE / "runs").glob("*.json")):
    r = json.loads(p.read_text())
    runs[r["weight_decay"]].append(r)


def mr(xs, fmt="{:.3f}"):
    xs = np.asarray(xs, dtype=float)
    return {"median": float(np.median(xs)), "min": float(xs.min()), "max": float(xs.max()),
            "text": f"{fmt.format(np.median(xs))} ({fmt.format(xs.min())}–{fmt.format(xs.max())})"}


summary = {}
for wd in sorted(runs):
    rs = sorted(runs[wd], key=lambda r: r["seed"])
    s = [r["selected_by_val"] for r in rs]
    b = [r["bytrain_rule"] for r in rs]
    summary[wd] = {
        "seeds": [r["seed"] for r in rs],
        "train": mr([x["train_auc"] for x in s]),
        "val": mr([x["val_auc"] for x in s]),
        "val_mean": float(np.mean([x["val_auc"] for x in s])),
        "test": mr([x["test_auc"] for x in s]),
        "test_per_seed": [round(x["test_auc"], 4) for x in s],
        "generation": mr([x["generation"] for x in s], "{:.0f}"),
        "hidden": mr([x["hidden_nodes"] for x in s], "{:.0f}"),
        "connections": mr([x["phenotype_connections_enabled"] for x in s], "{:.0f}"),
        "bytrain_test": mr([x["test_auc"] for x in b]),
        "bytrain_val": mr([x["val_auc"] for x in b]),
        "bytrain_generation": mr([x["generation"] for x in b], "{:.0f}"),
        "wall_clock_min": mr([r["wall_clock_s"] / 60 for r in rs], "{:.0f}"),
    }

# Median validation AUC; ties broken by mean validation AUC.
chosen = max(summary, key=lambda wd: (summary[wd]["val"]["median"], summary[wd]["val_mean"]))
(HERE / "summary.json").write_text(json.dumps(
    {"chosen_weight_decay_by_median_val": chosen, "arms": {str(k): v for k, v in summary.items()}},
    indent=1))

print("| weight decay | n | train AUC | val AUC | **test AUC** | selected gen | hidden | connections "
      "| test AUC, best-by-train rule (same runs) |")
print("|---|---|---|---|---|---|---|---|---|")
for wd, s in summary.items():
    tag = " (chosen)" if wd == chosen else ""
    print(f"| {wd:g}{tag} | {len(s['seeds'])} | {s['train']['text']} | {s['val']['text']} | "
          f"**{s['test']['text']}** | {s['generation']['text']} | {s['hidden']['text']} | "
          f"{s['connections']['text']} | {s['bytrain_test']['text']} (gen {s['bytrain_generation']['text']}) |")
print(f"\nchosen weight_decay (median validation AUC, ties by mean): {chosen:g}")
for wd, s in summary.items():
    print(f"  wd={wd:g}: val median {s['val']['median']:.4f} mean {s['val_mean']:.4f}; "
          f"test per seed {s['test_per_seed']}")

# Reference: logistic regression fit on the same 193 fit rows (no tuning).
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from common import load_splits  # noqa: E402

(Xf, yf), (Xv, yv), (Xt, yt) = load_splits()
for C in (1.0, 0.1):
    lr = LogisticRegression(C=C, max_iter=5000).fit(Xf, yf)
    print(f"LR (L2, C={C}) on fit rows: val AUC {roc_auc_score(yv, lr.predict_proba(Xv)[:, 1]):.3f}, "
          f"test AUC {roc_auc_score(yt, lr.predict_proba(Xt)[:, 1]):.3f}")
