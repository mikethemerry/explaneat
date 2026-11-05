"""Ingest PMLB `churn` (telecom customer churn) into the ExplaNEAT DB.

  1. Raw dataset `churn` (19 features; `phone number`, a unique per-row id, is dropped).
  2. One-hot prepared dataset `churn_onehot` (state, area code -> one-hot; the two
     yes/no plans stay binary; numerics untouched).
  3. Stratified 80:20 split, seed 42, StandardScaler fit on train.
  4. Baselines on that exact split: L2 logistic regression and gradient boosting
     (test AUC with 95% bootstrap CI, 10,000 resamples, seed 42).

Idempotent: reuses existing rows by name. Prints the ids to record in PLAN.md.

    uv run python experiments/churn/ingest_churn.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pmlb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from explaneat.db import db  # noqa: E402
from explaneat.db.dataset_utils import save_dataset_split_with_indices, save_dataset_to_db  # noqa: E402
from explaneat.db.encoding import build_encoding_config, prepare_dataset_arrays  # noqa: E402
from explaneat.db.models import Dataset, DatasetSplit  # noqa: E402

SEED = 42
RAW, PREP = "churn", "churn_onehot"
SOURCE_URL = "https://github.com/EpistasisLab/pmlb/raw/master/datasets/churn/churn.tsv.gz"
DROP = ["phone number"]
CATEGORICAL = ["state", "area code"]
BINARY = ["international plan", "voice mail plan"]


def load():
    df = pmlb.fetch_data("churn").drop(columns=DROP)
    y = df.pop("target").to_numpy().astype(np.float64)
    names = list(df.columns)
    ftypes = ["categorical" if c in CATEGORICAL else "binary" if c in BINARY else "numeric"
              for c in names]
    return df.to_numpy().astype(np.float64), y, names, ftypes


def get_or_create(session, X, y, names, ftypes):
    raw = session.query(Dataset).filter_by(name=RAW).first()
    if raw is None:
        raw = save_dataset_to_db(
            name=RAW, X=X, y=y, source="PMLB", version=pmlb.__version__, source_url=SOURCE_URL,
            description="Telecom customer churn (PMLB 'churn', 5,000 rows). 'phone number' dropped "
                        "(unique id). Charges are exact linear functions of the matching minutes.",
            feature_names=names, feature_types=dict(zip(names, ftypes)),
            target_name="churn", class_names=["0: stayed", "1: churned"],
            metadata={"dropped": DROP, "duplicated_information": "total * charge = rate x total * minutes"},
        )
        raw = session.query(Dataset).filter_by(name=RAW).first()
    prep = session.query(Dataset).filter_by(name=PREP).first()
    if prep is None:
        enc = build_encoding_config(X, names, ftypes)
        X_enc, enc_names, enc_types = prepare_dataset_arrays(X, names, ftypes, enc)
        prep = Dataset(
            name=PREP, version=pmlb.__version__, source="PMLB", source_url=SOURCE_URL,
            description=f"One-hot churn ({X_enc.shape[1]} columns): state and area code one-hot.",
            num_samples=X_enc.shape[0], num_features=X_enc.shape[1], num_classes=2,
            feature_names=enc_names, feature_types=enc_types, target_name="churn",
            class_names=["0: stayed", "1: churned"], source_dataset_id=raw.id, encoding_config=enc,
        )
        prep.set_data(X_enc, y)
        session.add(prep)
        session.flush()
    return str(raw.id), str(prep.id)


def get_or_create_split(prep_id, X_enc, y):
    with db.session_scope() as s:
        sp = s.query(DatasetSplit).filter_by(dataset_id=prep_id, random_state=SEED).first()
        if sp:
            return str(sp.id), np.array(sp.train_indices), np.array(sp.test_indices)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    save_dataset_split_with_indices(
        dataset_id=prep_id, train_indices=tr.tolist(), test_indices=te.tolist(), test_size=0.2,
        random_state=SEED, stratify=True, scaler=StandardScaler().fit(X_enc[tr]),
        preprocessing_steps=[{"step": "drop", "columns": DROP}, {"step": "onehot", "columns": CATEGORICAL},
                             {"step": "standardscaler", "fit": "train"}])
    with db.session_scope() as s:
        sp = s.query(DatasetSplit).filter_by(dataset_id=prep_id, random_state=SEED).first()
        return str(sp.id), tr, te


def bootstrap_ci(y, p, n=10_000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    db.init_db()
    X, y, names, ftypes = load()
    with db.session_scope() as s:
        raw_id, prep_id = get_or_create(s, X, y, names, ftypes)
        prep = s.get(Dataset, __import__("uuid").UUID(prep_id))
        X_enc, _ = prep.get_data()
        enc_names = list(prep.feature_names)
    split_id, tr, te = get_or_create_split(prep_id, X_enc, y)
    sc = StandardScaler().fit(X_enc[tr])
    Xtr, Xte = sc.transform(X_enc[tr]), sc.transform(X_enc[te])
    res = {"raw_dataset_id": raw_id, "dataset_id": prep_id, "split_id": split_id,
           "n_inputs": len(enc_names), "n_train": int(len(tr)), "n_test": int(len(te)),
           "positive_rate_train": float(y[tr].mean())}
    for label, model in [("l2_logistic", LogisticRegression(max_iter=5000)),
                         ("gradient_boosting", HistGradientBoostingClassifier(random_state=0))]:
        model.fit(Xtr, y[tr])
        p_tr, p_te = model.predict_proba(Xtr)[:, 1], model.predict_proba(Xte)[:, 1]
        lo, hi = bootstrap_ci(y[te], p_te)
        res[label] = {"train_auc": roc_auc_score(y[tr], p_tr), "test_auc": roc_auc_score(y[te], p_te),
                      "test_auc_ci": [lo, hi], "test_acc": float(((p_te > 0.5) == y[te]).mean())}
    out = HERE / "baselines.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
