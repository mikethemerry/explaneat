"""Import a finished in-memory MNIST run into the Postgres DB (UI / MCP access).

Writes only new rows:
  * Dataset + DatasetSplit (get-or-create by name/version, shared by the seeds of
    a task; the stored X holds only the rows used: train subset then test),
  * one Experiment ``<task>-s<seed>`` (refuses if the name already exists),
  * one Population row for the last evaluated generation, its Species rows and
    all of its Genome rows (NEAT parent keys in ``mutation_history``: the parent
    genomes are not in the DB, so parent1_id/parent2_id stay NULL),
  * the run's best genome: it is normally in that population (same key and
    weights); otherwise it gets its own Population row at the generation it was
    last evaluated in,
  * Result rows with the best genome's in-memory train/test metrics.

    DATABASE_URL=postgresql://localhost/explaneat_dev \\
        uv run python experiments/mnist/import_run.py mnist38-s0 [--dry-run]

Prints the database size before and after.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from data import feature_names, prepare  # noqa: E402
from explaneat.db.base import db  # noqa: E402
from explaneat.db.models import (  # noqa: E402
    Dataset, DatasetSplit, Experiment, Genome, Population, Result, Species,
)
from explaneat.db.serialization import serialize_genome, serialize_population_config  # noqa: E402
from run_seed import RUNS_DIR, load_config  # noqa: E402

DATASETS = {
    "mnist38": dict(name="mnist38_14x14", num_classes=2, class_names=["3", "8"],
                    target_name="is_eight",
                    target_description="1 = digit 8, 0 = digit 3",
                    description="MNIST digits 3 (y=0) vs 8 (y=1), 2x2 mean-pooled to 14x14 "
                                "(196 inputs), pixel intensity in [0, 1]. Rows: train subset "
                                "then the standard test 3s/8s. Exploratory (experiments/mnist)."),
    "mnist10": dict(name="mnist10_784", num_classes=10, class_names=[str(c) for c in range(10)],
                    target_name="digit", target_description="digit class 0-9",
                    description="MNIST 10-class, 784 inputs, pixel intensity in [0, 1]. Rows: "
                                "stratified seeded train subset then the full standard test "
                                "set. Exploratory (experiments/mnist)."),
}


def db_size(session) -> int:
    if session.bind.dialect.name == "sqlite":  # scratch tests
        return int(session.execute(text("pragma page_count")).scalar()) * int(
            session.execute(text("pragma page_size")).scalar())
    return int(session.execute(text("select pg_database_size(current_database())")).scalar())


def mb(n):
    return f"{n / 1e6:.1f} MB"


def get_or_create_dataset(session, task, data, version):
    spec = DATASETS[task]
    ds = session.query(Dataset).filter_by(name=spec["name"], version=version).first()
    n_tr, n_te = len(data["y_train"]), len(data["y_test"])
    if ds is None:
        X = np.concatenate([data["X_train"], data["X_test"]]).astype(np.float64)
        y = np.concatenate([data["y_train"], data["y_test"]])
        names = feature_names(task)
        ds = Dataset(name=spec["name"], version=version, source="OpenML",
                     source_url="https://www.openml.org/d/554", description=spec["description"],
                     num_samples=len(y), num_features=X.shape[1], num_classes=spec["num_classes"],
                     feature_names=names, feature_types={n: "numeric" for n in names},
                     target_name=spec["target_name"], target_description=spec["target_description"],
                     class_names=spec["class_names"],
                     additional_metadata=dict(
                         task_type="classification", experiment="experiments/mnist",
                         openml_train_rows=data["train_rows"].tolist(),
                         openml_test_rows=data["test_rows"].tolist(), **data["meta"]))
        ds.set_data(X, y)
        session.add(ds)
        session.flush()
        split = DatasetSplit(dataset_id=ds.id, name=f"standard MNIST train/test ({n_tr}/{n_te})",
                             split_type="train_test", test_size=n_te / (n_tr + n_te),
                             random_state=data["meta"]["subset_seed"], shuffle=False,
                             stratify=(task == "mnist10"), train_indices=list(range(n_tr)),
                             test_indices=list(range(n_tr, n_tr + n_te)), scaler_type=None,
                             scaler_params=None,
                             preprocessing_steps=[data["meta"]["preprocessing"]],
                             train_size=n_tr, test_size_actual=n_te)
        session.add(split)
        session.flush()
        print(f"created dataset {ds.id} ({spec['name']} {version}) and split {split.id}")
    else:
        split = session.query(DatasetSplit).filter_by(dataset_id=ds.id).first()
        X_db, y_db = ds.get_data()
        assert np.array_equal(X_db[:n_tr], data["X_train"]) and np.array_equal(X_db[n_tr:], data["X_test"])
        print(f"reusing dataset {ds.id} and split {split.id}")
    return ds, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="run name, e.g. mnist38-s0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with gzip.open(RUNS_DIR / args.name / "final.pkl.gz", "rb") as f:
        run = pickle.load(f)
    meta, final, best = run["meta"], run["final"], run["best_genome"]
    task = meta["task"]
    summary = json.loads((HERE / "logs" / f"{args.name}.json").read_text())
    config = load_config(task)
    data = prepare(task, meta["n_train"])
    assert np.array_equal(data["train_rows"], run["train_rows"])
    assert np.array_equal(data["test_rows"], run["test_rows"])

    db.init_db()
    with db.session_scope() as session:
        before = db_size(session)
        print(f"database: {session.bind.url.render_as_string(hide_password=True)}; size before {mb(before)}")
        if session.query(Experiment).filter_by(name=args.name).first():
            sys.exit(f"experiment {args.name} already exists; not touching it")
        ds, split = get_or_create_dataset(session, task, data, version=f"n{meta['n_train']}")

        config_json = serialize_population_config(config)
        training = {"mask_phantom": True}
        if meta["output_activation"] != "sigmoid":
            training["output_activation"] = meta["output_activation"]
        if meta["criterion"] != "bce":
            training["criterion"] = {"name": meta["criterion"]}
        training.update(fitness_function=meta["fitness_function"], genome_type=meta["genome_type"],
                        n_epochs_backprop=meta["n_epochs_backprop"])
        config_json["training"] = training
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
                                         text=True).strip()
        exp = Experiment(
            experiment_sha=meta["code_commit"][:40], name=args.name,
            description=f"MNIST exploratory run ({task}), not pre-registered; "
                        "in-memory BackpropPopulation, imported by experiments/mnist/import_run.py",
            dataset_name=ds.name, dataset_version=ds.version, dataset_id=ds.id, split_id=split.id,
            config_json=config_json, neat_config_text=run["config_text"],
            start_time=datetime.utcnow(), end_time=datetime.utcnow(), status="completed",
            git_commit_sha=meta["code_commit"][:40], git_branch=branch, random_seed=meta["seed"],
            hardware_info=dict(meta["hardware"], device="cpu", torch_threads=meta["torch_threads"],
                               run_params=dict(dataset=task, seed=meta["seed"],
                                               n_generations=meta["n_generations"],
                                               n_epochs_backprop=meta["n_epochs_backprop"],
                                               fitness_function=meta["fitness_function"],
                                               neat_config=meta["neat_config"],
                                               code_commit=meta["code_commit"],
                                               code_dirty=meta["code_dirty"],
                                               mask_phantom=True, n_train=meta["n_train"],
                                               wall_clock_s=meta["wall_clock_s"],
                                               best_generation=meta["best_generation"],
                                               in_memory=True)))
        session.add(exp)
        session.flush()

        popd = final["population"]
        fits = [g.fitness for g in popd.values() if g.fitness is not None]
        prow = Population(experiment_id=exp.id, generation=final["generation"],
                          population_size=len(popd), num_species=len(final["species"]),
                          best_fitness=max(fits), mean_fitness=float(np.mean(fits)),
                          stdev_fitness=float(np.std(fits)), config_json=config_json)
        session.add(prow)
        session.flush()
        sp_rows, member_species = {}, {}
        for s in final["species"]:
            r = Species(population_id=prow.id, species_id=s["species_id"], size=s["size"],
                        fitness_mean=s["fitness_mean"], fitness_max=s["fitness_max"],
                        fitness_min=s["fitness_min"], age=final["generation"] - s["created"],
                        last_improved=s["last_improved"])
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
                                        species_key=sid, generation=final["generation"],
                                        parents_in_db=False)
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
            bpop = Population(experiment_id=exp.id, generation=meta["best_generation"],
                              population_size=1, num_species=0, best_fitness=best.fitness,
                              mean_fitness=best.fitness, config_json=config_json)
            session.add(bpop)
            session.flush()
            brow = Genome.from_neat_genome(best, bpop.id)
            brow.mutation_history = dict(note="run best genome (best training fitness); "
                                              "not in the final population",
                                         generation=meta["best_generation"])
            session.add(brow)
            session.flush()
            best_id = brow.id
        for split_name in ("train", "test"):
            for metric, value in summary[split_name].items():
                session.add(Result(experiment_id=exp.id, population_id=prow.id, genome_id=best_id,
                                   measurement_type=f"best_{split_name}_{metric}", value=float(value),
                                   iteration=final["generation"],
                                   params={"engine": "NeuralNeat", "in_memory": True}))
        session.flush()
        out = dict(experiment_id=str(exp.id), experiment=args.name, dataset_id=str(ds.id),
                   split_id=str(split.id), population_id=str(prow.id), best_genome_id=str(best_id),
                   best_in_final_population=same, n_genomes=len(popd))
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
