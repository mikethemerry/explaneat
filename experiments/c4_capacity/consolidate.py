"""Rebuild the capacity-study tables from preserved logs; never train or access the DB.

    uv run python experiments/c4_capacity/consolidate.py

Writes runs.csv, tables.md, summary.json and artifact_manifest.json. The manifest
also hashes the locally preserved run binaries when present; those binaries are
not required to regenerate the tables. Existing measured results are read only.
"""

from __future__ import annotations

import configparser
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("EXPLANEAT_C4CAP_RUNS", Path.home() / ".explaneat/c4_capacity_runs"))


def run_order(s):
    name = s["name"]
    return (0, int(name.split("-")[0][1:])) if name.startswith("p") else (1, s["seed"])


def main():
    runs = []
    source_paths = [HERE / "baselines.json"]
    for path in sorted((HERE / "logs").glob("*.json")):
        s = json.loads(path.read_text())
        if s.get("study") != "c4_capacity":
            continue
        csv_path = path.with_suffix(".csv")
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert [int(r["generation"]) for r in rows] == list(range(s["n_generations"])), path
        assert abs(float(rows[-1]["best_so_far_fitness"]) - s["best_fitness"]) < 1e-12, path
        pilot = s["val_frac"] is not None
        assert ("val" in s) == pilot and ("test" in s) != pilot, path
        cfg_path = HERE / "configs" / f"{s['name']}.ini"
        config = configparser.ConfigParser()
        config.read(cfg_path)
        assert config.get("DefaultGenome", "initial_connection") == "full_direct"
        for override in s["overrides"]:
            key, value = override.split("=", 1)
            section, option = key.split(".", 1)
            assert config.get(section, option) == value, (path, override)
        monitor = s["val"]["auc"] if pilot else s["train"]["auc"]
        # This monitor's CSV column is historically misnamed champ_test_auc.
        # The last champion can differ from the saved object in principle;
        # these preserved runs all save the last generation's champion.
        assert abs(float(rows[-1]["champ_test_auc"]) - monitor) < 1e-12, path
        run = dict(
            name=s["name"], seed=s["seed"], mode="pilot" if pilot else "main",
            generations=s["n_generations"], epochs=s["n_epochs_backprop"], fit_n=s["n_fit"],
            train_auc=s["train"]["auc"], val_auc=s.get("val", {}).get("auc"),
            test_auc=s.get("test", {}).get("auc"), hidden=s["best_size"]["hidden"],
            connections=s["best_size"]["conns"], species_mean=s["species"]["mean"],
            species_last=s["species"]["last"], saved_fit_auc=s["best_fitness"],
            max_logged_fit_auc=max(float(r["best_fitness"]) for r in rows),
            wall_minutes=s["wall_clock_s"] / 60,
        )
        runs.append(run)
        source_paths += [path, csv_path, cfg_path]
    runs.sort(key=run_order)
    pilots = [r for r in runs if r["mode"] == "pilot"]
    mains = [r for r in runs if r["mode"] == "main"]
    assert len(pilots) == 17 and len(mains) == 3
    assert {r["seed"] for r in mains} == {0, 1, 2}
    assert all(r["generations"] == 50 for r in pilots)
    assert all(r["generations"] == 80 and r["epochs"] == 50 for r in mains)
    chosen_pilot = max(pilots, key=lambda r: r["val_auc"])
    chosen_main = max(mains, key=lambda r: r["train_auc"])
    assert chosen_pilot["name"] == "p18-h10-w015-e50"
    assert chosen_main["name"] == "main-h10-e50-s1"
    with (HERE / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    baseline = json.loads((HERE / "baselines.json").read_text())
    test = [r["test_auc"] for r in mains]
    lr, gb = baseline["lr_cells"]["test"], baseline["gradient_boosting"]["test"]
    summary = dict(
        completed_pilots=len(pilots), completed_main_runs=len(mains),
        pilot_selection="maximum saved validation AUC; single reused validation split",
        chosen_pilot=chosen_pilot["name"],
        champion_selection="maximum training AUC among the three saved main champions",
        chosen_main=chosen_main["name"],
        main_test_auc=dict(mean=statistics.mean(test), sample_sd=statistics.stdev(test),
                           minimum=min(test), maximum=max(test)),
        fraction_lr_to_gbm_test_gap_closed=dict(
            mean=(statistics.mean(test) - lr) / (gb - lr),
            selected_champion=(chosen_main["test_auc"] - lr) / (gb - lr)),
        historical_best_snapshot_caveat=[
            r["name"] for r in runs if r["max_logged_fit_auc"] > r["saved_fit_auc"] + 1e-12
        ],
        main_budget="80 generations, population 150, 50 full-batch backprop epochs per generation",
        pilot_preprocessing_caveat="Scaler fitted before fit/validation subdivision of training rows",
        test_caveat="Shared previous-study split and known baseline test metrics; exploratory evaluation",
    )
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = ["<!-- Generated by consolidate.py from preserved measurements. -->", "",
            "| Pilot | Epochs | Validation AUC | Hidden | Connections | Mean / last species |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in pilots:
        text.append(f"| {r['name']} | {r['epochs']} | {r['val_auc']:.4f} | {r['hidden']} | "
                    f"{r['connections']} | {r['species_mean']:.1f} / {r['species_last']} |")
    text += ["", "| Main seed | Training AUC | Test AUC | Hidden | Connections | Mean / last species |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in mains:
        text.append(f"| {r['seed']} | {r['train_auc']:.4f} | {r['test_auc']:.4f} | {r['hidden']} | "
                    f"{r['connections']} | {r['species_mean']:.1f} / {r['species_last']} |")
    text += ["", "| Baseline | Training AUC | Test AUC |", "|---|---:|---:|"]
    for name in ["lr_cells", "lr_cells_threats", "lr_onehot_threats", "gradient_boosting"]:
        b = baseline[name]
        text.append(f"| {name} | {b['train']:.4f} | {b['test']:.4f} |")
    (HERE / "tables.md").write_text("\n".join(text) + "\n")
    for pattern in ["*.analysis.json", "*.domain.json", "*.import.json"]:
        source_paths += sorted((HERE / "logs").glob(pattern))
    selected_json = HERE / "artifacts" / "selected_genome.json"
    if selected_json.exists():
        source_paths.append(selected_json)
    manifest = []
    for path in sorted(source_paths):
        manifest.append(dict(path=str(path.relative_to(HERE)), bytes=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    binaries = []
    for r in runs:
        for name in ["best_genome.pkl.gz", "final.pkl.gz", "parents.csv.gz"]:
            path = RUNS / r["name"] / name
            if path.exists():
                binaries.append(dict(run=r["name"], artifact=name, bytes=path.stat().st_size,
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (HERE / "artifact_manifest.json").write_text(json.dumps(
        dict(measured_sources=manifest, local_binary_artifacts=binaries), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
