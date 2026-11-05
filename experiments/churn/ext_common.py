"""Shared plumbing for the churn extension scripts (EXTENSIONS_PLAN.md).

Data, genomes and explanations are read in READ ONLY transactions
(experiments/10seed/compute_runs.py::ReadOnlySession). Metrics follow
runs_churn100.csv: NeuralNeat on the split's stored StandardScaler, float64;
test AUC CI = percentile 95% over the 10,000-row bootstrap index matrix
(np.random.RandomState(42)); accuracy at p > 0.5.
"""

from __future__ import annotations

import csv
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments" / "10seed"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from compute_runs import ReadOnlySession, bootstrap_indices, load_split, structure  # noqa: E402
from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.db.models import Experiment, Explanation, Genome  # noqa: E402

import explained_eval as ee  # noqa: E402
import genome_surgery as gs  # noqa: E402

SPLIT = "45962ea6-3c78-49f9-a01e-3665f6809b1e"
EPOCHS = 50
SEEDS = list(range(10))
RUNS_CSV = HERE / "runs_churn100.csv"
RECORDS = REPO / "experiments" / "10seed" / "explanations"


def read_runs():
    """{seed: row} from runs_churn100.csv."""
    with open(RUNS_CSV) as f:
        return {int(r["seed"]): r for r in csv.DictReader(f)}


def read_record(seed):
    return json.loads((RECORDS / f"churn100-s{seed}.json").read_text())


def load_data():
    with ReadOnlySession() as s:
        data = load_split(s, SPLIT)
        from explaneat.db.models import DatasetSplit
        split = s.get(DatasetSplit, uuid.UUID(SPLIT))
        data["dataset_id"] = str(split.dataset_id)
        data["scaler_mean"] = np.asarray(split.scaler_params["mean"], dtype=np.float64)
        data["scaler_scale"] = np.asarray(split.scaler_params["scale"], dtype=np.float64)
    return data


def load_genome(genome_id):
    """(genome, config, info) for a stored genome, with its explanation ops."""
    with ReadOnlySession() as s:
        row = s.get(Genome, uuid.UUID(genome_id))
        exp = row.population.experiment
        config = load_neat_config(exp.neat_config_text or "", exp.config_json)
        genome = row.to_neat_genome(config)
        expl = s.query(Explanation).filter(Explanation.genome_id == row.id).all()
        if len(expl) > 1:
            raise ValueError(f"genome {genome_id}: {len(expl)} explanations")
        info = {"genome_id": genome_id, "genome_key": row.genome_id,
                "experiment_id": str(exp.id), "experiment_name": exp.name,
                "config_json": dict(exp.config_json),
                "neat_config_text": exp.neat_config_text,
                "split_id": str(exp.split_id), "dataset_id": str(exp.dataset_id),
                "operations": list(expl[0].operations) if expl else []}
    assert info["split_id"] == SPLIT, f"{genome_id}: split {info['split_id']}"
    return genome, config, info


def explained_state(genome, config, ops, X_check=None):
    """Replayed model state and annotation dicts; optionally checks the
    evaluator reproduces NeuralNeat on ``X_check`` (function preservation)."""
    engine, replayed = gs.replay_model_state(ops, genome, config)
    state = engine.current_state
    anns = ee.annotations_from_ops(replayed)
    ev = ee.StateEvaluator(state)
    err = None
    if X_check is not None:
        err = float(np.max(np.abs(ev.proba(X_check) - gs.predict_proba(genome, config, X_check))))
        if err > 1e-9:
            raise AssertionError(f"model-state evaluator differs from NeuralNeat by {err}")
    return state, anns, ev, err


def evaluate(genome, config, data, boot_idx, p_ref=None):
    """Per-condition metrics (plan 'Metrics per condition'). ``p_ref``: test
    predictions of a reference condition for a paired bootstrap delta."""
    p_tr = gs.predict_proba(genome, config, data["X_train"])
    p_te = gs.predict_proba(genome, config, data["X_test"])
    ytr, yte = data["y_train"], data["y_test"]
    lo, hi, n_boot = ee.bootstrap_auc_ci(yte, p_te, boot_idx)
    st = structure(genome, config, data["feature_names"])
    row = {"train_auc": roc_auc_score(ytr, p_tr), "test_auc": roc_auc_score(yte, p_te),
           "test_auc_ci_lo": lo, "test_auc_ci_hi": hi, "n_boot_used": n_boot,
           "train_acc_0.5": accuracy_score(ytr, p_tr > 0.5),
           "test_acc_0.5": accuracy_score(yte, p_te > 0.5),
           "phenotype_connections_enabled": st["phenotype_connections_enabled"],
           "hidden_nodes": st["hidden_nodes"],
           "genome_conn_genes_enabled": st["genome_conn_genes_enabled"],
           "inputs_connected": st["inputs_connected"]}
    return row, p_te


def paired_delta(data, p_a, p_b, boot_idx):
    yte = data["y_test"]
    d = roc_auc_score(yte, p_a) - roc_auc_score(yte, p_b)
    lo, hi, _ = ee.bootstrap_delta_ci(yte, p_a, p_b, boot_idx)
    return d, lo, hi


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    print(f"wrote {path} ({len(rows)} rows)")


def state_input_ids(feature_names):
    """{feature name: input node id} for the state:* one-hot inputs
    (input key -(i+1) <-> data column i)."""
    return {n: -(i + 1) for i, n in enumerate(feature_names) if n.startswith("state:")}


def check_names_free(names):
    """Abort before any work if a derived-experiment name is already taken."""
    with ReadOnlySession() as s:
        taken = [n for n in names if s.query(Experiment).filter(Experiment.name == n).count()]
    if taken:
        raise SystemExit(f"experiment name(s) already exist, refusing to run: {taken}")
