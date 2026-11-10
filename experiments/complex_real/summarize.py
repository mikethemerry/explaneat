"""Summary table of the complex-real runs (logs/<name>.json, .analysis.json, .import.json)
-> runs.csv and stdout.

    uv run python experiments/complex_real/summarize.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = []
for p in sorted((HERE / "logs").glob("*.json")):
    if p.name.count(".") > 1:
        continue
    d = json.loads(p.read_text())
    a = HERE / "logs" / f"{d['name']}.analysis.json"
    i = HERE / "logs" / f"{d['name']}.import.json"
    an = json.loads(a.read_text())["structure"] if a.exists() else {}
    par = json.loads(a.read_text())["parity"]["ok"] if a.exists() else None
    imp = json.loads(i.read_text()) if i.exists() else {}
    rows.append(dict(name=d["name"], task=d["task"], variant=d["variant"], seed=d["seed"],
                     train_auc=round(d["train"]["auc"], 5), test_auc=round(d["test"]["auc"], 5),
                     best_generation=d["best_generation"], hidden=d["best_size"]["hidden"],
                     conns=d["best_size"]["conns"], inputs_used=d["best_size"]["inputs_used"],
                     depth=d["best_size"]["depth"],
                     species_mean=round(d["species"]["mean"], 2), species_mean_last50=round(d["species"]["mean_last50"], 2),
                     species_min=d["species"]["min"], species_max=d["species"]["max"],
                     final_threshold=round(d["final_threshold"], 2), wall_clock_s=d["wall_clock_s"],
                     switching_share=(round(an["switching_share_strict"], 3)
                                      if an.get("switching_share_strict") is not None else None),
                     hidden_activity=json.dumps(an.get("hidden_activity_strict")) if an else None,
                     parity_ok=par, genome_id=imp.get("best_genome_id"), experiment_id=imp.get("experiment_id")))
if rows:
    with open(HERE / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(" ".join(f"{k}={v}" for k, v in r.items() if k not in ("experiment_id", "hidden_activity")))
