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
from threadpoolctl import threadpool_limits

threadpool_limits(1)  # modest CPU: other training runs share this machine

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


def evaluate(genome, config, data, boot_idx):
    """Per-condition metrics (plan 'Metrics per condition'); also returns the
    test predictions, for paired deltas between conditions."""
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


PHANTOM_HELP = (
    "how fine-tuning treats NeuralNeat's phantom weights (the dense zero entries "
    "with no connection gene). 'primitive' = literal PropNEAT backprop: they "
    "train and are dropped at write-back, so the stored genome is not the "
    "trained network. 'masked' = only genome weights/biases train. Required: "
    "the plan does not decide this (see genome_surgery.finetune).")


def add_phantom_arg(ap):
    ap.add_argument("--phantom", choices=["primitive", "masked"], required=True,
                    help=PHANTOM_HELP)


def check_names_free(names):
    """Abort before any work if a derived-experiment name is already taken."""
    with ReadOnlySession() as s:
        taken = [n for n in names if s.query(Experiment).filter(Experiment.name == n).count()]
    if taken:
        raise SystemExit(f"experiment name(s) already exist, refusing to run: {taken}")


# ---- amendment 1 (EXTENSIONS_PLAN.md): masked fine-tuning, E = 500 primary, E = 50 secondary

EPOCHS_PRIMARY = 500
EPOCHS_SECONDARY = 50


def parse_epochs(text):
    """'500' -> [500]; '500,50' -> [500, 50]. The first value is the primary setting."""
    vals = [int(v) for v in str(text).split(",") if v.strip()]
    if not vals or any(v < 1 for v in vals) or len(set(vals)) != len(vals):
        raise ValueError(f"--epochs needs distinct positive integers, got {text!r}")
    return vals


def dry_run_epochs(epochs):
    """Stand-in epoch counts for --dry-run (2 for the primary, then 1, ...), so a
    dry run exercises every per-E condition quickly; names and rows carry these."""
    return [max(2 - i, 1) if i < 2 else i + 1 for i in range(len(epochs))]


def add_ft_args(ap):
    """--phantom (default masked, amendment 1) and --epochs (default 500)."""
    ap.add_argument("--phantom", choices=["primitive", "masked"], default="masked",
                    help=PHANTOM_HELP.replace(
                        "Required: the plan does not decide this",
                        "Default masked (EXTENSIONS_PLAN.md amendment 1)"))
    ap.add_argument("--epochs", default=str(EPOCHS_PRIMARY), type=parse_epochs,
                    help="comma-separated fine-tune epoch counts; the first is primary, "
                         f"e.g. '{EPOCHS_PRIMARY},{EPOCHS_SECONDARY}' runs the amendment's "
                         f"primary and secondary settings (default {EPOCHS_PRIMARY})")


# ---- early-stopping amendment (EARLY_STOPPING_PLAN.md): fixed validation split

VAL_SPLIT_JSON = HERE / "val_split.json"
VAL_FRACTION = 0.2
VAL_RANDOM_STATE = 42
ES_PATIENCE = 50
ES_MIN_DELTA = 1e-5
ES_MAX_EPOCHS = 5000


def make_val_split(y_train, train_indices=None):
    """The amendment's validation split: stratified 20% of the churn train split,
    ``train_test_split(positions, test_size=0.2, stratify=y_train,
    random_state=42)`` over the train positions 0..n_train-1 (the row order of
    ``load_data()['X_train']``). Returns the JSON-ready dict."""
    from sklearn.model_selection import train_test_split
    y_train = np.asarray(y_train).ravel()
    pos = np.arange(len(y_train))
    fit_pos, val_pos = train_test_split(pos, test_size=VAL_FRACTION, stratify=y_train,
                                        random_state=VAL_RANDOM_STATE)
    fit_pos, val_pos = sorted(int(i) for i in fit_pos), sorted(int(i) for i in val_pos)
    out = {
        "plan": "experiments/churn/EARLY_STOPPING_PLAN.md",
        "split_id": SPLIT,
        "rule": ("sklearn.model_selection.train_test_split(np.arange(n_train), test_size=0.2, "
                 "stratify=y_train, random_state=42); positions index the rows of the churn "
                 "train split in its stored order (DatasetSplit.train_indices order, i.e. "
                 "ext_common.load_data()['X_train']); both lists sorted"),
        "n_train": len(y_train), "n_fit": len(fit_pos), "n_val": len(val_pos),
        "fit_positive": int(y_train[fit_pos].sum()), "val_positive": int(y_train[val_pos].sum()),
        "fit_train_positions": fit_pos, "val_train_positions": val_pos,
    }
    if train_indices is not None:
        ti = [int(i) for i in train_indices]
        out["fit_dataset_indices"] = [ti[i] for i in fit_pos]
        out["val_dataset_indices"] = [ti[i] for i in val_pos]
    return out


def load_val_split(data):
    """(X_fit, y_fit, X_val, y_val, split dict) from the committed val_split.json;
    checks it matches the rule and the loaded train split."""
    vs = json.loads(VAL_SPLIT_JSON.read_text())
    assert vs["split_id"] == SPLIT
    ytr = np.asarray(data["y_train"]).ravel()
    again = make_val_split(ytr)
    if (again["fit_train_positions"] != vs["fit_train_positions"]
            or again["val_train_positions"] != vs["val_train_positions"]):
        raise SystemExit("val_split.json does not match the amendment's rule on this train split")
    f, v = vs["fit_train_positions"], vs["val_train_positions"]
    assert sorted(f + v) == list(range(len(ytr)))
    return data["X_train"][f], ytr[f], data["X_train"][v], ytr[v], vs


def es_kwargs(X_val, y_val):
    """finetune kwargs for the amendment's early stopping."""
    return dict(epochs=None, val_X=X_val, val_y=y_val, patience=ES_PATIENCE,
                min_delta=ES_MIN_DELTA, max_epochs=ES_MAX_EPOCHS, restore_best=True,
                mask_phantom=True, return_info=True)
