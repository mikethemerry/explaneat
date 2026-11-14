"""Store the three synthetic datasets and their splits in Postgres ``explaneat_dev`` (new rows only).

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/ushape_saddle/make_datasets.py

Get-or-create by the opaque dataset name (``data.dataset_name``); an existing row must hold the
same arrays and split, else the script stops. The description does not give the generating
function. Writes ``datasets.json`` (task -> label, dataset_id, split_id) for the harness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import data  # noqa: E402


def get_or_create(session, task, d):
    from explaneat.db.models import Dataset, DatasetSplit
    name = data.dataset_name(task)
    ds = session.query(Dataset).filter_by(name=name).first()
    if ds is not None:
        split = session.query(DatasetSplit).filter_by(dataset_id=ds.id).first()
        X_db, y_db = ds.get_data()
        assert np.array_equal(X_db, d["X_raw"]) and np.array_equal(np.asarray(y_db).ravel(), d["y_all"]), "stored data differs"
        assert list(split.train_indices) == [int(i) for i in d["train_rows"]], "stored split differs"
        assert np.allclose(split.scaler_params["mean"], d["mean"]) and np.allclose(split.scaler_params["scale"], d["scale"])
        return ds, split, False
    names = d["names"]
    ds = Dataset(name=name, version="v1", source="synthetic",
                 description="Synthetic clinical-style tabular data with 10 inputs for a shape-recovery study "
                             "(experiments/ushape_saddle). Exploratory. The generating function is withheld "
                             "from the explaining agent.",
                 num_samples=len(d["y_all"]), num_features=len(names), num_classes=2, feature_names=names,
                 feature_types=d["types"], target_name="outcome", target_description="1 = outcome, 0 = no outcome",
                 class_names=["0", "1"],
                 additional_metadata=dict(task_type="classification", experiment="experiments/ushape_saddle",
                                          label=d["meta"]["label"], n_rows=d["meta"]["n_rows"], n_train=d["meta"]["n_train"],
                                          n_test=d["meta"]["n_test"], scaler=d["meta"]["scaler"]))
    ds.set_data(d["X_raw"].astype(np.float64), d["y_all"].astype(np.int64))
    session.add(ds)
    session.flush()
    tr, te = d["train_rows"], d["test_rows"]
    split = DatasetSplit(dataset_id=ds.id, name="stratified 75/25 seed 42", split_type="train_test",
                         test_size=data.TEST_SIZE, random_state=data.SPLIT_SEED, shuffle=True, stratify=True,
                         train_indices=[int(i) for i in tr], test_indices=[int(i) for i in te],
                         scaler_type="StandardScaler",
                         scaler_params={"mean": [float(v) for v in d["mean"]], "scale": [float(v) for v in d["scale"]]},
                         preprocessing_steps=["raw units", "StandardScaler (train rows)"],
                         train_size=len(tr), test_size_actual=len(te))
    session.add(split)
    session.flush()
    return ds, split, True


def main():
    from explaneat.db.base import db
    db.init_db()
    reg = data.registry()
    with db.session_scope() as s:
        assert s.bind.url.database == "explaneat_dev", s.bind.url
        for task in data.TASKS:
            d = data.prepare(task)
            ds, split, created = get_or_create(s, task, d)
            reg[task] = dict(label=data.LABELS[task], dataset_name=ds.name, dataset_id=str(ds.id), split_id=str(split.id),
                             n_train=int(d["meta"]["n_train"]), n_test=int(d["meta"]["n_test"]),
                             positive_share=d["meta"]["positive_share"])
            print(f"{task:7s} {data.LABELS[task]} {ds.name} dataset {ds.id} split {split.id} {'created' if created else 'exists'} "
                  f"pos {d['meta']['positive_share']:.3f}")
    data.registry_path().write_text(json.dumps(reg, indent=1) + "\n")


if __name__ == "__main__":
    main()
