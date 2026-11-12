"""Tables for the line-features base test from logs/<run>.json, baselines.json and signal_check.json:
runs.csv (one row per run) and tables.md (the AUC table with the references). Selection per base is
by training AUC (the within-run best genome is already the best training fitness); test AUC is
reported, never used for selection.

    uv run python experiments/c4_line_features/summarize.py
"""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
runs = []
for p in sorted((HERE / "logs").glob("*.json")):
    if p.name.endswith(".import.json"):
        continue
    j = json.loads(p.read_text())
    imp = HERE / "logs" / f"{j['name']}.import.json"
    runs.append(dict(name=j["name"], inputs=j["inputs"], config=j["config"], seed=j["seed"],
                     train_auc=j["train"]["auc"], test_auc=j["test"]["auc"],
                     max_logged_train_fitness=j.get("max_logged_fitness"),
                     hidden=j["best_size"]["hidden"], conns=j["best_size"]["conns"],
                     inputs_used=j["best_size"]["inputs_used"], depth=j["best_size"]["depth"],
                     species_mean=round(j["species"]["mean"], 2), species_last=j["species"]["last"],
                     best_generation=j["best_generation"], wall_min=round(j["wall_clock_s"] / 60, 1),
                     best_genome_id=json.loads(imp.read_text())["best_genome_id"] if imp.exists() else ""))
with open(HERE / "runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(runs[0]))
    w.writeheader()
    w.writerows(runs)

base = json.loads((HERE / "baselines.json").read_text())
sig = json.loads((HERE / "signal_check.json").read_text())
L = ["### PropNEAT runs (selection by training AUC; test reported only)", "",
     "| Base | Config | Seed | Train AUC | Test AUC | Hidden | Conns | Inputs used | Best gen | Wall (min) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
sel = {}
for r in runs:
    best = sel.get(r["inputs"])
    if best is None or r["train_auc"] > best["train_auc"]:
        sel[r["inputs"]] = r
for r in runs:
    star = "**" if sel[r["inputs"]] is r else ""
    L.append(f"| {r['inputs']} | {r['config']} | {r['seed']} | {star}{r['train_auc']:.4f}{star} | {r['test_auc']:.4f} | "
             f"{r['hidden']} | {r['conns']} | {r['inputs_used']} | {r['best_generation']} | {r['wall_min']} |")
L += ["", "Bold: the genome selected per base (largest training AUC). Per base and config, mean (SD) of test AUC:", ""]
for inputs in ("lines", "lines_cells", "perline"):
    for cfg in ("heavy", "default"):
        t = [r["test_auc"] for r in runs if r["inputs"] == inputs and r["config"] == cfg]
        if t:
            L.append(f"- {inputs} / {cfg}: n = {len(t)}, test AUC {np.mean(t):.4f} (SD {np.std(t, ddof=1) if len(t) > 1 else 0:.4f}), "
                     f"range {min(t):.4f}-{max(t):.4f}")
L += ["", "### References on the same split (train / test AUC)", "",
      "| Base | Inputs | LR | GBM | PropNEAT (earlier studies) |", "|---|---|---|---|---|",
      f"| cells (earlier) | 42 | {base['cells']['lr']['train']:.3f} / {base['cells']['lr']['test']:.3f} | "
      f"{base['cells']['gradient_boosting']['train']:.3f} / {base['cells']['gradient_boosting']['test']:.3f} | "
      "complex_real default 0.858 / 0.862; capacity 0.919 / 0.915 (selected), mean test 0.911 |"]
for inputs in ("lines", "lines_cells"):
    b = base[inputs]
    L.append(f"| {inputs} | {b['n_inputs']} | {b['lr']['train']:.3f} / {b['lr']['test']:.3f} | "
             f"{b['gradient_boosting']['train']:.3f} / {b['gradient_boosting']['test']:.3f} | this study (above) |")
pl = sig["perline138+aggregate16"]
L.append(f"| perline | {pl['n_inputs']} | {pl['lr']['train']:.3f} / {pl['lr']['test']:.3f} | "
         f"{pl['gbm']['train']:.3f} / {pl['gbm']['test']:.3f} | this study (above) |")
L += ["", "### Signal check: GBM / LR test AUC by nameable vocabulary (`signal_check.py`)", "",
      "| Vocabulary | Inputs | LR test | GBM test |", "|---|---|---|---|"]
for k, v in sig.items():
    L.append(f"| {k} | {v['n_inputs']} | {v['lr']['test']:.4f} | {v['gbm']['test']:.4f} |")
(HERE / "tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
