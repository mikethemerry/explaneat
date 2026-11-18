"""Ingest monk2 (AAAI-27 Experiment 3 candidate) + split + train + topology summary.

Pipeline (all reproducible, seed 42):
  1. Fetch PMLB monk2, deduplicate to unique attribute configurations.
     STOP unless exactly 432 rows (3*3*2*3*4*2). STOP unless the ground-truth
     rule matches the labels on all 432 rows.
  2. Store raw dataset 'monk2' (6 categorical attrs a1..a6) with full metadata.
  3. Build the one-hot prepared dataset 'monk2_onehot' (17 columns), display
     names a1=1 .. a6=2.
  4. Stratified 80:20 split (seed 42) on the 432 unique rows, StandardScaler
     fit on train (a no-op on 0/1 columns, attached for pipeline consistency).
  5. Launch PropNEAT training via the same code path the API/paper used
     (ExperimentRunner._evolution_loop, AUC fitness, no structural penalty).
  6. Emit a topology summary of the best genome to results/monk2-topology.md.

Training hyperparameters are NOT specified by the task; this uses the paper's
Adult settings (pop 250, 100 generations, 5 backprop epochs, AUC fitness) and
records that choice. [[AUTHOR]: adjust generations if you want a longer search.]

Run: uv run python scripts/aaai27/ingest_monk2.py
"""
import sys
import uuid
from datetime import date

import numpy as np
import pmlb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")

from explaneat.db import db
from explaneat.db.models import Dataset, DatasetSplit
from explaneat.db.dataset_utils import save_dataset_to_db, save_dataset_split_with_indices
from explaneat.db.encoding import build_encoding_config, prepare_dataset_arrays

PMLB_VERSION = pmlb.__version__
INGEST_DATE = date(2026, 7, 10).isoformat()
SOURCE_URL = "https://github.com/EpistasisLab/pmlb/raw/master/datasets/monk2/monk2.tsv.gz"
SEED = 42
ATTRS = ["a1", "a2", "a3", "a4", "a5", "a6"]
CARDINALITIES = {"a1": 3, "a2": 3, "a3": 2, "a4": 3, "a5": 4, "a6": 2}

# Training config (task-unspecified; paper-Adult settings).
TRAIN = dict(population_size=250, n_generations=100, n_epochs_backprop=5,
             fitness_function="auc")


def fetch_dedup():
    """Fetch monk2, dedup to unique configs. STOP unless 432 + rule holds."""
    df = pmlb.fetch_data("monk2")
    src_cols = [f"attribute#{i}" for i in range(1, 7)]
    dedup = df.drop_duplicates(subset=src_cols).reset_index(drop=True)
    n = len(dedup)
    if n != 432:
        sys.exit(f"STOP: monk2 dedup produced {n} rows, expected 432 "
                 f"(3*3*2*3*4*2). Source differs from expectation.")
    # label consistency across duplicate configs in the full file
    if df.groupby(src_cols)["target"].nunique().max() > 1:
        sys.exit("STOP: monk2 has configs with inconsistent labels across "
                 "duplicate rows.")

    X = dedup[src_cols].values.astype(np.float64)   # values are 1-indexed
    y = dedup["target"].values.astype(np.float64)

    # Ground-truth rule: positive iff EXACTLY TWO of {a1=1,...,a6=1}.
    count_eq1 = np.sum(X == 1, axis=1)
    rule_pred = (count_eq1 == 2).astype(float)
    match = float((rule_pred == y).mean())
    if match != 1.0:
        sys.exit(f"STOP: ground-truth rule matches only {match*100:.4f}% of "
                 f"labels (expected 100%). Source differs from expectation.")
    print(f"monk2: 432 unique rows, rule matches 100% ({int(y.sum())} positives)")
    return X, y


# A foreign, un-deduplicated `monk2` (601 rows, attribute#1..6, no metadata)
# already exists in this DB and is the parent of a pre-existing experiment
# (adcf1dc2). We must not mutate or delete foreign data, so our authoritative
# deduplicated raw is stored under a distinct name. [[AUTHOR]: if that foreign
# monk2 + its experiment are stale, delete them and rename this to 'monk2'.]
RAW_NAME = "monk2_dedup432"


def get_or_create_raw(session, X, y):
    existing = session.query(Dataset).filter_by(name=RAW_NAME).first()
    if existing and (existing.additional_metadata or {}).get("source_pinning"):
        print(f"raw dataset {RAW_NAME!r} already exists (mine): {existing.id}")
        return existing.id

    rule_relevant_cols = [f"{a}:1" for a in ATTRS]  # the six '=1' one-hots
    metadata = {
        "source_pinning": {
            "source": "PMLB",
            "source_url": SOURCE_URL,
            "pmlb_package_version": PMLB_VERSION,
            "pmlb_data_commit": "[[AUTHOR]: pmlb package does not expose the data-repo "
                                "commit; pin from EpistasisLab/pmlb master if needed]",
            "ingestion_date": INGEST_DATE,
        },
        "dedup_and_split": (
            "PMLB monk2 (601 rows) contains duplicate attribute configurations; "
            "the canonical MONK test file is the full input-space enumeration. "
            "Deduplicated to 432 unique configurations, then stratified 80:20 "
            "(seed 42) on those unique rows. This deliberately sidesteps the "
            "known MONK train/test overlap issue."
        ),
        "ground_truth": {
            "rule": "positive iff EXACTLY TWO of {a1=1, a2=1, a3=1, a4=1, a5=1, a6=1}",
            "rule_relevant_columns": rule_relevant_cols,
            "rule_verified_match": 1.0,
        },
        "attribute_cardinalities": CARDINALITIES,
        "n_unique_rows": 432,
    }
    descriptions = {a: f"MONK abstract categorical attribute {a} "
                       f"(values 1..{CARDINALITIES[a]})" for a in ATTRS}
    ds = save_dataset_to_db(
        name=RAW_NAME,
        X=X, y=y,
        source="PMLB",
        version=PMLB_VERSION,
        source_url=SOURCE_URL,
        description="MONK-2 problem (unique input-space enumeration, 432 rows). "
                    "Target: exactly two of the six attributes equal 1.",
        feature_names=ATTRS,
        feature_descriptions=descriptions,
        feature_types={a: "categorical" for a in ATTRS},
        target_name="monk2_positive",
        target_description="1 iff exactly two of a1..a6 equal 1, else 0",
        class_names=["0: not exactly two =1", "1: exactly two of a1..a6 =1"],
        metadata=metadata,
    )
    print(f"created raw dataset {RAW_NAME!r}: {ds.id}")
    return ds.id


def get_or_create_prepared(session, raw_id, X, y):
    existing = session.query(Dataset).filter_by(name="monk2_onehot").first()
    if existing:
        if existing.source_dataset_id != raw_id:
            existing.source_dataset_id = raw_id  # repoint to authoritative raw
            session.flush()
            print(f"prepared 'monk2_onehot' repointed to raw {raw_id}")
        print(f"prepared dataset 'monk2_onehot' already exists: {existing.id}")
        return existing.id

    ftypes = ["categorical"] * 6
    enc = build_encoding_config(X, ATTRS, ftypes)
    X_enc, names, types_dict = prepare_dataset_arrays(X, ATTRS, ftypes, enc)
    # one-hot columns are 'aK:V'; display names 'aK=V'
    display = {nm: nm.replace(":", "=") for nm in names}

    ds = Dataset(
        name="monk2_onehot",
        version=PMLB_VERSION,
        source="PMLB",
        source_url=SOURCE_URL,
        description="One-hot encoded monk2 (17 binary columns from a1..a6).",
        num_samples=X_enc.shape[0],
        num_features=X_enc.shape[1],
        num_classes=2,
        feature_names=names,
        feature_descriptions=display,
        feature_types=types_dict,
        target_name="monk2_positive",
        class_names=["0: not exactly two =1", "1: exactly two of a1..a6 =1"],
        source_dataset_id=raw_id,
        encoding_config=enc,
        additional_metadata={
            "display_names": display,
            "rule_relevant_columns": [f"{a}:1" for a in ATTRS],
            "note": "z-normalisation is a no-op on 0/1 columns; scaler attached "
                    "on the split for pipeline consistency.",
        },
    )
    ds.set_data(X_enc, y)
    session.add(ds)
    session.flush()
    print(f"created prepared dataset 'monk2_onehot': {ds.id} "
          f"({X_enc.shape[1]} columns)")
    return ds.id


def get_or_create_split(prepared_id, X_enc, y):
    """Create (or reuse) the stratified 80:20 split. Own sessions throughout;
    returns (split_id, train_idx, test_idx) as plain values (no ORM objects)."""
    with db.session_scope() as session:
        existing = (session.query(DatasetSplit)
                    .filter_by(dataset_id=prepared_id, random_state=SEED).first())
        if existing:
            sid = str(existing.id)
            tr = np.array(existing.train_indices)
            te = np.array(existing.test_indices)
            print(f"split already exists: {sid}")
            return sid, tr, te

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_enc[train_idx])
    # save_dataset_split_with_indices manages its own session and commits;
    # re-query afterwards for a bound id rather than touching the detached obj.
    save_dataset_split_with_indices(
        dataset_id=prepared_id,
        train_indices=train_idx.tolist(),
        test_indices=test_idx.tolist(),
        test_size=0.2,
        random_state=SEED,
        stratify=True,
        scaler=scaler,
        preprocessing_steps=[{"step": "onehot", "note": "a1..a6 -> 17 cols"},
                             {"step": "standardscaler", "note": "no-op on 0/1"}],
    )
    with db.session_scope() as session:
        split = (session.query(DatasetSplit)
                 .filter_by(dataset_id=prepared_id, random_state=SEED).first())
        sid = str(split.id)
    print(f"created split: {sid} (train={len(train_idx)}, test={len(test_idx)})")
    return sid, train_idx, test_idx


def launch_training(prepared_id, split_id, X_enc, train_idx):
    from explaneat.api.experiment_runner import ExperimentRunner, ExperimentProgress
    from explaneat.core.config_resolution import resolve_config, config_to_neat_text

    with db.session_scope() as s:
        split = s.query(DatasetSplit).filter_by(id=uuid.UUID(str(split_id))).first()
        mean = np.array(split.scaler_params["mean"])
        scale = np.array(split.scaler_params["scale"])
        ds = s.query(Dataset).filter_by(id=uuid.UUID(str(prepared_id))).first()
        _, y_full = ds.get_data()
    # avoid divide-by-zero on any constant column (none expected in monk2 OHE)
    scale = np.where(scale == 0, 1.0, scale)
    X_train = (X_enc[train_idx] - mean) / scale
    y_train = y_full[train_idx].astype(np.float64)

    resolved = resolve_config(overrides={"training": TRAIN})
    num_inputs, num_outputs = X_train.shape[1], 1
    config_text = config_to_neat_text(resolved, num_inputs, num_outputs)
    config_json = {
        "pop_size": resolved["training"]["population_size"],
        "num_inputs": num_inputs, "num_outputs": num_outputs,
        "fitness_criterion": "max", "fitness_threshold": 999.0,
        "resolved_config": resolved,
    }
    progress = ExperimentProgress(job_id="monk2-ingest",
                                  total_generations=TRAIN["n_generations"])
    print(f"launching training: pop={TRAIN['population_size']}, "
          f"gens={TRAIN['n_generations']}, epochs={TRAIN['n_epochs_backprop']}, "
          f"fitness={TRAIN['fitness_function']}")
    ExperimentRunner._evolution_loop(
        progress=progress, config_text=config_text, config_json=config_json,
        X_train=X_train, y_train=y_train,
        experiment_name="monk2 - OHE - 250-100-5",
        dataset_name="monk2_onehot", description="AAAI-27 Exp3 candidate (monk2)",
        n_generations=TRAIN["n_generations"],
        n_epochs_backprop=TRAIN["n_epochs_backprop"],
        fitness_function=TRAIN["fitness_function"],
        dataset_id=str(prepared_id), split_id=str(split_id),
    )
    print(f"training done: experiment={progress.experiment_id}, "
          f"best_fitness={progress.best_fitness}")
    return progress.experiment_id


def main():
    X, y = fetch_dedup()
    with db.session_scope() as s:
        raw_id = get_or_create_raw(s, X, y)
    with db.session_scope() as s:
        prepared_id = get_or_create_prepared(s, raw_id, X, y)
        prepared_ds = s.query(Dataset).filter_by(id=prepared_id).first()
        X_enc, y_enc = prepared_ds.get_data()
    split_id, train_idx, test_idx = get_or_create_split(prepared_id, X_enc, y_enc)

    exp_id = launch_training(prepared_id, split_id, X_enc, train_idx)
    print(f"\nMONK2 INGEST COMPLETE. experiment_id={exp_id}")
    print("Run scripts/aaai27/topology_summary.py to emit the topology summary.")


if __name__ == "__main__":
    main()
