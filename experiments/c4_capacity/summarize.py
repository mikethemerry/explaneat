"""Table of the Connect-4 capacity runs from logs/<name>.json (+ .csv for growth).

Pilots (val_frac set) show the held-out-of-training validation AUC; main runs show train and,
with --test, the test AUC (report only at the end).

    uv run python experiments/c4_capacity/summarize.py [--test] [names...]
"""

import csv
import json
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parent / "logs"
show_test = "--test" in sys.argv
names = [a for a in sys.argv[1:] if not a.startswith("--")]
files = [LOG / f"{n}.json" for n in names] if names else sorted(LOG.glob("*.json"),
                                                                  key=lambda p: p.stat().st_mtime)
cols = ["name", "gens", "epochs", "fit_auc", "val_auc", "train_auc"] + (["test_auc"] if show_test else []) + \
       ["hidden", "conns", "inputs", "depth", "species_mean", "species_last", "hidden@25%/50%/75%", "wall_min"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for f in files:
    if f.name.endswith((".import.json", ".analysis.json")) or not f.with_suffix(".csv").exists():
        continue
    s = json.loads(f.read_text())
    rows = list(csv.DictReader(open(f.with_suffix(".csv"))))
    q = [rows[int(len(rows) * p) - 1]["champ_hidden"] for p in (0.25, 0.5, 0.75)]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))  # noqa: E731
    vals = [s["name"], s["n_generations"], s["n_epochs_backprop"], s.get("fit_rows", {}).get("auc"),
            s.get("val", {}).get("auc"), s["train"]["auc"]] + ([s.get("test", {}).get("auc")] if show_test else []) + \
           [s["best_size"]["hidden"], s["best_size"]["conns"], s["best_size"]["inputs_used"], s["best_size"]["depth"],
            round(s["species"]["mean"], 1), s["species"]["last"], "/".join(q), round(s["wall_clock_s"] / 60, 1)]
    print("| " + " | ".join(fmt(v) for v in vals) + " |")
