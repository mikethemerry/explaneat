"""Consolidate saved linear-start results, including incomplete runs, without training.

    uv run python experiments/mnist/linear_start_summary.py

Reads the CSV/JSON logs and (where present) trusted local parents.csv.gz files.
Writes summary.json and table.csv beside the logs. Does not choose models or
read test results to select an arm. Missing finals remain explicitly pending.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs" / "linear_start"
RUNS = Path(os.environ.get("EXPLANEAT_MNIST_RUNS", Path.home() / ".explaneat" / "mnist_runs"))
NAMES = [f"ls-{arm}-s{seed}" for arm in ("full", "lr", "struct") for seed in range(3)]


def parent_audit(path):
    """Count births separately from elite retraining; generation zero is initialization."""
    by_gen = {}
    with gzip.open(path, "rt") as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            counts = by_gen.setdefault(gen, {"population": 0, "new_offspring": 0})
            counts["population"] += 1
            counts["new_offspring"] += int(gen > 0 and row["is_elite_copy"] == "0")
    born = [g for g, v in by_gen.items() if v["new_offspring"]]
    return dict(last_generation=max(by_gen), last_birth_generation=max(born, default=None),
                total_offspring=sum(v["new_offspring"] for v in by_gen.values()),
                generations_with_births=born, per_generation=by_gen)


def consolidate():
    sources, rows, audits = {}, [], {}
    for name in NAMES:
        cp, jp = LOGS / f"{name}.csv", LOGS / f"{name}.json"
        data = cp.read_bytes()
        sources[cp.name] = hashlib.sha256(data).hexdigest()
        gens = list(csv.DictReader(data.decode().splitlines()))
        final = json.loads(jp.read_text()) if jp.exists() else None
        if final:
            sources[jp.name] = hashlib.sha256(jp.read_bytes()).hexdigest()
            assert len(gens) == final["n_generations"]
        assert [int(g["generation"]) for g in gens] == list(range(len(gens)))
        sizes = [list(map(int, g["species_sizes"].split())) for g in gens]
        ppath = RUNS / name / "parents.csv.gz"
        if ppath.exists():
            audits[name] = parent_audit(ppath)
        last = gens[-1]
        row = dict(name=name, status="complete" if final else "pending", seed=int(name[-1]),
                   arm=name.split("-")[1], logged_generations=len(gens),
                   epochs_per_generation=25 if "struct" in name else 5,
                   species_start=int(gens[0]["n_species"]),
                   species_end=int(last["n_species"]), species_max=max(int(g["n_species"]) for g in gens),
                   population_end=sum(sizes[-1]), population_max=max(map(sum, sizes)),
                   threshold_end=float(last["threshold_used"]),
                   wall_minutes=float(last["wall_s"]) / 60)
        if final:
            row.update(best_generation=final["best_generation"], hidden=final["best_size"]["hidden"],
                       conns=final["best_size"]["conns"], backbone=final["best_backbone"])
            for split in ("train", "val", "test"):
                for metric in ("acc", "ce"):
                    row[f"{split}_{metric}"] = final[split][metric]
        if name in audits:
            row.update({k: audits[name][k] for k in ("last_birth_generation", "total_offspring")})
        rows.append(row)
    groups = {}
    for arm in ("full", "lr", "struct"):
        group = [r for r in rows if r["arm"] == arm and r["status"] == "complete"]
        groups[arm] = dict(n_complete=len(group), n_planned=3)
        for metric in ("train_acc", "val_acc", "test_acc", "train_ce", "val_ce", "test_ce"):
            values = [r[metric] for r in group]
            if values:
                groups[arm][metric] = dict(mean=statistics.mean(values),
                                           sample_sd=statistics.stdev(values) if len(values) > 1 else None)
    result = dict(status="complete" if all(r["status"] == "complete" for r in rows) else "interim",
                  planned_runs=9, completed_runs=sum(r["status"] == "complete" for r in rows),
                  selection="Within-run champion uses training fitness only; no new model selection",
                  sources_sha256=sources, runs=rows, descriptive_groups=groups, parent_audits=audits)
    (LOGS / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(LOGS / "table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{result['status']}: {result['completed_runs']}/9 complete")
    for r in rows:
        print(r["name"], r["status"], "val", r.get("val_acc"), "hidden", r.get("hidden"),
              "species", r["species_end"], "population", r["population_end"],
              "last birth", r.get("last_birth_generation"))


if __name__ == "__main__":
    consolidate()
