"""Import a finished line-features run into Postgres explaneat_dev (UI / MCP access).

Copy of experiments/c4_capacity/import_run.py with: runs from ~/.explaneat/c4_line_features_runs,
experiment name ``c4lf-<run name>`` (refuses if it exists), and a dataset per base
(``connect4_lines``, ``connect4_lines_cells``, ``connect4_perline``; version v1) created on first
use with its split and scaler, the feature names and the codebook (``additional_metadata.codebook``).
Writes only new rows; no schema operation.

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/c4_line_features/import_run.py <name> [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from lfdata import DATASET_NAMES, codebook, prepare  # noqa: E402
from explaneat.db.base import db  # noqa: E402
from explaneat.db.models import (  # noqa: E402
    Dataset, DatasetSplit, Experiment, Genome, Population, Result, Species,
)
from explaneat.db.serialization import serialize_genome, serialize_population_config  # noqa: E402

RUNS_DIR = Path.home() / ".explaneat" / "c4_line_features_runs"
DESCRIPTIONS = {
    "lines": "Connect-4 8-ply positions (UCI, PMLB connect_4), binarised win (1) vs not win (0), with 16 derived "
             "line features as inputs (open lines of 1-3 per side, threats per side by row parity, centre column "
             "pieces, playable-cell parity). Same rows and split as connect4_win_ordinal. "
             "Exploratory (experiments/c4_line_features).",
    "lines_cells": "Connect-4 8-ply positions (UCI, PMLB connect_4), binarised win (1) vs not win (0), with the 16 "
                   "derived line features followed by the 42 ordinal cells (x = +1, o = -1, blank = 0). Same rows "
                   "and split as connect4_win_ordinal. Exploratory (experiments/c4_line_features).",
    "perline": "Connect-4 8-ply positions (UCI, PMLB connect_4), binarised win (1) vs not win (0), with 138 per-line "
               "open counts (69 lines x 2 sides: the side's pieces on the line when no opposing piece is on it) "
               "followed by the 16 derived line features. Same rows and split as connect4_win_ordinal. "
               "Exploratory (experiments/c4_line_features).",
}


def load_config(name):
    import neat
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(HERE / "configs" / f"{name}.ini"))


def db_size(session) -> int:
    return int(session.execute(text("select pg_database_size(current_database())")).scalar())


def get_or_create_dataset(session, inputs, d):
    name = DATASET_NAMES[inputs]
    ds = session.query(Dataset).filter_by(name=name, version="v1").first()
    if ds is None:
        names = d["names"]
        ds = Dataset(name=name, version="v1", source="PMLB (derived)",
                     source_url="https://epistasislab.github.io/pmlb/profile/connect_4.html",
                     description=DESCRIPTIONS[inputs], num_samples=len(d["y_all"]), num_features=len(names),
                     num_classes=2, feature_names=names, feature_types={n: "ordinal" for n in names},
                     target_name="win",
                     target_description="1 = x (first player, to move) wins with perfect play; 0 = loss or draw",
                     class_names=d["meta"]["class_names"],
                     additional_metadata=dict(task_type="classification", experiment="experiments/c4_line_features",
                                              codebook=codebook(inputs), base_dataset="connect4_win_ordinal",
                                              **d["meta"]))
        ds.set_data(d["X_raw"].astype(np.float64), d["y_all"].astype(np.int64))
        session.add(ds)
        session.flush()
        tr, te = d["train_rows"], d["test_rows"]
        split = DatasetSplit(dataset_id=ds.id, name="stratified 80/20 seed 42", split_type="train_test",
                             test_size=0.2, random_state=42, shuffle=True, stratify=True,
                             train_indices=[int(i) for i in tr], test_indices=[int(i) for i in te],
                             scaler_type="StandardScaler",
                             scaler_params={"mean": [float(v) for v in d["mean"]],
                                            "scale": [float(v) for v in d["scale"]]},
                             preprocessing_steps=[d["meta"]["encoding"], "StandardScaler (train rows)"],
                             train_size=len(tr), test_size_actual=len(te))
        session.add(split)
        session.flush()
        print(f"created dataset {ds.id} ({name}) and split {split.id}")
    else:
        split = session.query(DatasetSplit).filter_by(dataset_id=ds.id).first()
        X_db, _ = ds.get_data()
        assert np.array_equal(X_db, d["X_raw"]) and list(split.train_indices) == [int(i) for i in d["train_rows"]]
        print(f"reusing dataset {ds.id} and split {split.id}")
    return ds, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with gzip.open(RUNS_DIR / args.name / "final.pkl.gz", "rb") as f:
        run = pickle.load(f)
    meta, final, best = run["meta"], run["final"], run["best_genome"]
    summary = json.loads((HERE / "logs" / f"{args.name}.json").read_text())
    config = load_config(args.name)
    d = prepare(meta["inputs"])
    assert np.array_equal(d["train_rows"], run["train_rows"]) and np.array_equal(d["test_rows"], run["test_rows"])
    exp_name = f"c4lf-{args.name}"

    db.init_db()
    with db.session_scope() as session:
        before = db_size(session)
        if session.query(Experiment).filter_by(name=exp_name).first():
            sys.exit(f"experiment {exp_name} already exists; not touching it")
        ds, split = get_or_create_dataset(session, meta["inputs"], d)
        config_json = serialize_population_config(config)
        config_json["training"] = dict(mask_phantom=True, fitness_function="auc", genome_type="DefaultGenome",
                                       n_epochs_backprop=meta["n_epochs_backprop"], dynamic_threshold=None)
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, text=True).strip()
        exp = Experiment(
            experiment_sha=meta["code_commit"][:40], name=exp_name,
            description=f"Connect-4 line-features base test run ({meta['name']}: inputs {meta['inputs']}, config "
                        f"{meta['config']}, overrides {meta['overrides']}, {meta['n_epochs_backprop']} epochs per "
                        "generation), exploratory, not pre-registered; in-memory BackpropPopulation, imported by "
                        "experiments/c4_line_features/import_run.py",
            dataset_name=ds.name, dataset_version=ds.version, dataset_id=ds.id, split_id=split.id,
            config_json=config_json, neat_config_text=run["config_text"],
            start_time=datetime.utcnow(), end_time=datetime.utcnow(), status="completed",
            git_commit_sha=meta["code_commit"][:40], git_branch=branch, random_seed=meta["seed"],
            hardware_info=dict(meta["hardware"], device="cpu", torch_threads=meta["torch_threads"],
                               run_params=dict(dataset="c4", inputs=meta["inputs"], config=meta["config"],
                                               seed=meta["seed"], n_generations=meta["n_generations"],
                                               n_epochs_backprop=meta["n_epochs_backprop"],
                                               fitness_function="auc", neat_config=meta["neat_config"],
                                               code_commit=meta["code_commit"], code_dirty=meta["code_dirty"],
                                               mask_phantom=True, n_train=meta["n_fit"], overrides=meta["overrides"],
                                               wall_clock_s=meta["wall_clock_s"],
                                               best_generation=meta["best_generation"], in_memory=True)))
        session.add(exp)
        session.flush()

        popd = final["population"]
        fits = [g.fitness for g in popd.values() if g.fitness is not None]
        prow = Population(experiment_id=exp.id, generation=final["generation"], population_size=len(popd),
                          num_species=len(final["species"]), best_fitness=max(fits),
                          mean_fitness=float(np.mean(fits)), stdev_fitness=float(np.std(fits)),
                          config_json=config_json)
        session.add(prow)
        session.flush()
        sp_rows, member_species = {}, {}
        for s in final["species"]:
            r = Species(population_id=prow.id, species_id=s["species_id"], size=s["size"],
                        fitness_mean=s["fitness_mean"], fitness_max=s["fitness_max"], fitness_min=s["fitness_min"],
                        age=final["generation"] - s["created"], last_improved=s["last_improved"])
            session.add(r)
            sp_rows[s["species_id"]] = (r, s["representative"])
            for k in s["members"]:
                member_species[k] = s["species_id"]
        session.flush()
        ids = {}
        for k, g in popd.items():
            p1, p2, elite = final["parents"].get(k, (None, None, False))
            sid = member_species.get(k)
            row = Genome.from_neat_genome(g, prow.id, species_id=sp_rows[sid][0].id if sid else None)
            row.mutation_history = dict(parent1_key=p1, parent2_key=p2, is_elite_copy=bool(elite),
                                        species_key=sid, generation=final["generation"], parents_in_db=False)
            session.add(row)
            session.flush()
            ids[k] = row.id
        for sid, (r, rep) in sp_rows.items():
            if rep in ids:
                r.representative_genome_id = ids[rep]
        same = best.key in popd and serialize_genome(popd[best.key]) == serialize_genome(best)
        if same:
            best_id = ids[best.key]
        else:
            bpop = Population(experiment_id=exp.id, generation=meta["best_generation"], population_size=1,
                              num_species=0, best_fitness=best.fitness, mean_fitness=best.fitness,
                              config_json=config_json)
            session.add(bpop)
            session.flush()
            brow = Genome.from_neat_genome(best, bpop.id)
            brow.mutation_history = dict(note="run best genome (best training fitness); not in the final population",
                                         generation=meta["best_generation"])
            session.add(brow)
            session.flush()
            best_id = brow.id
        for sname in ("train", "test"):
            for metric, value in summary[sname].items():
                session.add(Result(experiment_id=exp.id, population_id=prow.id, genome_id=best_id,
                                   measurement_type=f"best_{sname}_{metric}", value=float(value),
                                   iteration=final["generation"], params={"engine": "NeuralNeat", "in_memory": True}))
        session.flush()
        out = dict(experiment_id=str(exp.id), experiment=exp_name, dataset_id=str(ds.id), split_id=str(split.id),
                   population_id=str(prow.id), best_genome_id=str(best_id), best_in_final_population=same,
                   n_genomes=len(popd))
        if args.dry_run:
            session.rollback()
            print("dry run: rolled back")
    with db.session_scope() as session:
        after = db_size(session)
    out.update(db_size_before=before, db_size_after=after, db_size_delta=after - before)
    print(json.dumps(out, indent=2))
    if not args.dry_run:
        (HERE / "logs" / f"{args.name}.import.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
