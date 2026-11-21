"""Shared settings and helpers for the PropNEAT hyperparameter sensitivity check (see PLAN.md).

Design choices (literals, stated in PLAN.md): the factor grid, the folds (StratifiedKFold, 5, shuffle,
random_state 42 on the training rows), the CV seed (0), the final seeds (0-4), and the tie rule.
Data are read from the database in READ ONLY transactions; nothing is written to the database.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path
from uuid import UUID

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

SOURCE_DB = "postgresql://localhost/explaneat_dev"
FOLDS_JSON = HERE / "folds.json"

# Each dataset of the check: its database split, its fixed runs' NEAT config and their run record.
# Churn's fixed runs are churn100 (experiments/10seed/run_seed.py churn, configs/churn.ini = the
# Adult/Churn family); the synthetic fixed runs are experiments/ushape_saddle/run_seed.py
# (configs/syn10.ini = the Heart family with 10 inputs). Both: pop 250, 100 generations,
# 5 epochs, AUC fitness, unmasked, no weight decay.
DATASETS = {
    "churn": dict(label="Churn", dataset_id="802129c2-b11b-43a5-b6af-8bcf80e4c252",
                  split_id="45962ea6-3c78-49f9-a01e-3665f6809b1e",
                  base_config=REPO / "experiments/churn/configs/churn.ini", fixed_family="adult",
                  fixed_runs=REPO / "experiments/churn/runs_churn100.csv", fixed_dataset="churn",
                  bench_key="churn"),
    "u": dict(label="Synth. U", dataset_id="b4d7be76-77a5-4c04-8389-6d96cec21fc6",
              split_id="fc59adac-cf59-4480-a2d7-7ec2fce379eb",
              base_config=REPO / "experiments/ushape_saddle/configs/syn10.ini", fixed_family="heart",
              fixed_runs=REPO / "experiments/ushape_saddle/runs.csv", fixed_dataset="u",
              bench_key="shape_u"),
    "saddle": dict(label="Synth. saddle", dataset_id="3e662ee6-0218-4bc1-b236-ec84ab65761e",
                   split_id="cc5a37ed-efb3-4619-9e67-52343ed233aa",
                   base_config=REPO / "experiments/ushape_saddle/configs/syn10.ini", fixed_family="heart",
                   fixed_runs=REPO / "experiments/ushape_saddle/runs.csv", fixed_dataset="saddle",
                   bench_key="shape_saddle"),
    "both": dict(label="Synth. U + saddle", dataset_id="519c1f60-307b-4735-a0af-4f09cf3e3cd2",
                 split_id="a80f5dcd-fa1a-42eb-ad10-1f40a70115bb",
                 base_config=REPO / "experiments/ushape_saddle/configs/syn10.ini", fixed_family="heart",
                 fixed_runs=REPO / "experiments/ushape_saddle/runs.csv", fixed_dataset="both",
                 bench_key="shape_both"),
}

# The structural-mutation families (experiments/thesis_run_config/RESULTS.md, finding 1).
FAMILIES = {
    "heart": dict(node_add_prob=0.4, node_delete_prob=0.2, conn_add_prob=0.5, conn_delete_prob=0.3,
                  compatibility_threshold=3.0, compatibility_disjoint_coefficient=1.0),
    "adult": dict(node_add_prob=0.7, node_delete_prob=0.1, conn_add_prob=0.68, conn_delete_prob=0.15,
                  compatibility_threshold=5.0, compatibility_disjoint_coefficient=2.0),
}
EPOCHS = (5, 20)
WEIGHT_DECAYS = (0.0, 1e-3, 1e-2)
FAMILY_RANK = {"heart": 0, "adult": 1}  # tie rule: the Heart family is the simpler
N_GENERATIONS = 100
CV_SEED = 0
FINAL_SEEDS = (0, 1, 2, 3, 4)
N_FOLDS = 5
FOLD_RANDOM_STATE = 42
TIE_DECIMALS = 3  # mean fold AUCs equal at 3 decimal places count as tied


def config_id(epochs, family, wd):
    return f"e{epochs}_{family}_wd{wd:g}"


def simplicity_key(c):
    return (c["epochs"], c["weight_decay"], FAMILY_RANK[c["family"]])


def all_configs():
    """The 12 configurations, in the simplicity order of the tie rule."""
    out = [dict(id=config_id(e, f, w), epochs=e, family=f, weight_decay=w)
           for e, w, f in itertools.product(EPOCHS, WEIGHT_DECAYS, ("heart", "adult"))]
    return sorted(out, key=simplicity_key)


def parse_config_id(cid):
    e, f, w = cid.split("_")
    return dict(id=cid, epochs=int(e[1:]), family=f, weight_decay=float(w[2:]))


def fixed_config_id(dataset):
    return config_id(5, DATASETS[dataset]["fixed_family"], 0.0)


def config_path(dataset, family):
    return HERE / "configs" / f"{dataset}_{family}.ini"


def run_path(dataset, cid, fold, seed, subdir="runs"):
    tag = "full" if fold == "full" else f"fold{fold}"
    return HERE / subdir / dataset / f"{cid}_{tag}_s{seed}.json"


def load_raw(dataset):
    """X, y, train and test row indices and the stored scaler, in a READ ONLY transaction."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from explaneat.db.models import Dataset, DatasetSplit

    spec = DATASETS[dataset]
    engine = create_engine(SOURCE_DB, connect_args={"options": "-c default_transaction_read_only=on"})
    with Session(engine) as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        ds = session.get(Dataset, UUID(spec["dataset_id"]))
        split = session.get(DatasetSplit, UUID(spec["split_id"]))
        assert split.dataset_id == ds.id, "split does not belong to dataset"
        X, y = ds.get_data()
        train_idx = [int(i) for i in split.train_indices]
        test_idx = [int(i) for i in split.test_indices]
        stored = {"mean": list(split.scaler_params["mean"]), "scale": list(split.scaler_params["scale"])}
        session.rollback()
    engine.dispose()
    return (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64).ravel(),
            train_idx, test_idx, stored)


def standardise(X, fit_rows):
    """Scaler fit on ``fit_rows`` only (as experiments/10seed/run_seed.py on the training rows)."""
    mean = X[fit_rows].mean(axis=0)
    scale = X[fit_rows].std(axis=0)
    scale[scale == 0.0] = 1.0  # sklearn StandardScaler convention
    return (X - mean) / scale, mean, scale


def load_folds():
    return json.loads(FOLDS_JSON.read_text())


def require_env():
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")
