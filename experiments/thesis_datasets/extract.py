"""Describe every Chapter 7 dataset from the records the runs used (layer 2: read-only extractor).

    uv run --extra survival python experiments/thesis_datasets/extract.py     # -> data.json (~1 min)

Sources, all read-only:
* classification sets: the database split each PropNEAT run trained on, from the comparator suite's
  registry (``experiments/benchmarks/common.CLASSIFICATION``), read in READ ONLY transactions
  (``experiments/10seed/compute_runs.ReadOnlySession``; ``PGOPTIONS=-c default_transaction_read_only=on``
  as well). For a prepared (encoded) dataset the raw parent row is read too (``source_dataset_id``).
  The split each reported run actually used is checked against the registry: Experiment 2 via the
  runs CSVs' experiment ids, Experiment 1 via the genome ids in ``experiments/benchmarks/results/propneat.json``
  and the three original networks (Heart, Adult, MONK-2) in ``experiments/10seed/runs_paper_reference.csv``.
  These checks also say which experiments report each classification dataset ("used in");
* survival sets: the Cox case's split files (``experiments/survival/cox_case/splits``), its loader
  (``explaneat.survival.data.load_builtin``, scikit-survival bundled data) and the selected run's record
  (``experiments/survival/cox_case/runs/<tag>.json``), checked against each other;
* XOR: the four rows in ``experiments/xor/run_xor.py`` and the seed count in ``xor_results.json``;
* Experiment 3 validation rows: ``experiments/churn/val_split.json``, ``monk2_val_split.json``;
* MNIST (Experiment 4, ExConNEAT): the run records in ``experiments/conv_neat/results/<run>/config.json``
  (seed, ``n_train``, ``n_val`` and the row indices each run used), checked against
  ``explaneat.conv.data.mnist_splits`` recomputed from the cached OpenML ``mnist_784`` copy
  (``load_mnist_raw``). Only the lineage of the presented ExConNEAT network (the MNIST 10-class network of
  the 50k warm start): the 3 vs 8 vs noise runs are not presented (21 Nov 2026) and are left out.

Nothing is written to any database. ``data.json`` holds counts, shares and the stored descriptions
(encoding config, preprocessing steps, target descriptions) that ``make_table.py`` formats.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path

os.environ.setdefault("PGOPTIONS", "-c default_transaction_read_only=on")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "10seed"))
sys.path.insert(0, str(REPO / "experiments" / "benchmarks"))

from common import CLASSIFICATION  # noqa: E402  (experiments/benchmarks/common.py)

OUT = HERE / "data.json"
COX = REPO / "experiments" / "survival" / "cox_case"
PROPNEAT = REPO / "experiments" / "benchmarks" / "results" / "propneat.json"
PAPER_REF = REPO / "experiments" / "10seed" / "runs_paper_reference.csv"   # the three original networks
# Experiment 2 runs: (runs CSV, value of its `dataset` column, registry name)
EXP2_RUNS = [
    (REPO / "experiments" / "10seed" / "runs.csv", "adult", "adult"),
    (REPO / "experiments" / "10seed" / "runs_heart100.csv", None, "heart"),
    (REPO / "experiments" / "churn" / "runs_churn100.csv", None, "churn"),
]
CONV = REPO / "experiments" / "conv_neat"
# ExConNEAT MNIST 10-class runs reported in Chapter 7 (experiments/conv_neat/RESULTS.md, section 11b):
# the 10k-subset run (selected genome, generation 64) and its pruned continuation share one split;
# the 50k warm start trains on all the remaining training rows. key -> run folders
MNIST10_RUNS = {"mnist10_10k": ["sweep_p50_m4_e5", "mnist10_pruned"], "mnist10_50k": ["mnist10_ws50k_halving"]}
EXP3_VAL = {"churn": REPO / "experiments" / "churn" / "val_split.json",
            "monk2": REPO / "experiments" / "churn" / "monk2_val_split.json"}


def _jsonable(v):
    if isinstance(v, uuid.UUID):
        return str(v)
    return v


def _imputed_columns(X, names):
    """Columns whose values are all integers except one distinct value equal to the mean of the rest
    (a column mean written into the missing cells, as ``explaneat/api/routes/datasets.py`` does)."""
    out = {}
    for j, name in enumerate(names):
        col = X[:, j].astype(np.float64)
        frac = np.abs(col - np.round(col)) > 1e-9
        vals = np.unique(col[frac])
        if len(vals) == 1 and (~frac).any():
            others = col[~frac]
            # the stored value is the mean over the non-missing rows
            if abs(vals[0] - others.mean()) < 1e-6:
                out[name] = {"rows": int(frac.sum()), "value": float(vals[0])}
    return out


def _scaler_fit_rows(X, split):
    mean = np.asarray(split.scaler_params["mean"], dtype=np.float64)
    tr = np.asarray(split.train_indices)
    if np.allclose(mean, X[tr].astype(np.float64).mean(0), atol=1e-9):
        return "train"
    if np.allclose(mean, X.astype(np.float64).mean(0), atol=1e-9):
        return "all"
    return "other"


def classification(cr):
    from explaneat.db.models import Dataset, DatasetSplit
    out = {}
    by_db = {}
    for name, (dbname, sid, _) in CLASSIFICATION.items():
        by_db.setdefault(dbname, []).append((name, sid))
    for dbname, items in by_db.items():
        cr.db.init_db(f"postgresql://localhost/{dbname}")
        for name, sid in items:
            with cr.ReadOnlySession() as s:
                split = s.get(DatasetSplit, uuid.UUID(sid))
                ds = s.get(Dataset, split.dataset_id)
                X, y = ds.get_data()
                y = np.asarray(y).ravel().astype(int)
                tr = np.asarray(split.train_indices)
                te = np.asarray(split.test_indices)
                va = list(split.validation_indices or [])
                assert not (set(tr) & set(te)), name
                rec = {
                    "database": dbname, "split_id": sid,
                    "dataset": {k: _jsonable(getattr(ds, k)) for k in (
                        "id", "name", "version", "source", "source_url", "num_samples", "num_features",
                        "target_name", "target_description", "class_names", "encoding_config",
                        "additional_metadata")},
                    "split": {"test_size": split.test_size, "random_state": split.random_state,
                              "stratify": bool(split.stratify), "shuffle": bool(split.shuffle),
                              "scaler_type": split.scaler_type, "preprocessing_steps": split.preprocessing_steps,
                              "scaler_fit_rows": _scaler_fit_rows(X, split)},
                    "n_rows": int(len(y)), "n_inputs": int(X.shape[1]), "feature_names": list(ds.feature_names),
                    "n_unique_input_rows": int(len(np.unique(X, axis=0))),
                    "n_train": int(len(tr)), "n_test": int(len(te)), "n_val": int(len(va)),
                    "y_counts": {str(k): int(v) for k, v in sorted(Counter(y.tolist()).items())},
                    "positive_train": float(y[tr].mean()), "positive_test": float(y[te].mean()),
                    "n_positive_train": int(y[tr].sum()),
                    "train_indices_sha256_16": hashlib.sha256(
                        np.sort(tr).astype(np.int64).tobytes()).hexdigest()[:16],
                }
                if ds.source_dataset_id is not None:
                    p = s.get(Dataset, ds.source_dataset_id)
                    Xp, yp = p.get_data()
                    yp = np.asarray(yp).ravel()
                    assert len(yp) == len(y), (name, "parent row count")
                    if np.array_equal(yp.astype(int), y):
                        target_rule = "same"
                    elif np.array_equal((yp > 0).astype(int), y):
                        target_rule = "parent > 0"
                    else:
                        raise AssertionError((name, "target is neither the parent's nor parent > 0"))
                    meta = p.additional_metadata or {}
                    rec["parent"] = {
                        "id": str(p.id), "name": p.name, "source": p.source, "source_url": p.source_url,
                        "num_samples": p.num_samples, "n_inputs": int(Xp.shape[1]),
                        "feature_names": list(p.feature_names), "class_names": p.class_names,
                        "y_counts": {str(int(k)): int(v) for k, v in sorted(Counter(yp.tolist()).items())},
                        "dropped": list(meta.get("dropped") or []), "additional_metadata": meta,
                        "target_rule": target_rule,
                        "imputed": _imputed_columns(Xp, list(p.feature_names)),
                    }
                    # one-hot columns made from an imputed value (the value became its own category)
                    enc = (ds.encoding_config or {}).get("categorical", {})
                    made = {}
                    for col, info in rec["parent"]["imputed"].items():
                        if col in enc:
                            levels = [lv for lv in enc[col] if float(lv) not in set(np.unique(
                                np.round(Xp[:, list(p.feature_names).index(col)], 9)))]
                            for lv in levels:
                                j = list(ds.feature_names).index(f"{col}:{lv}")
                                made[f"{col}:{lv}"] = int(X[:, j].sum())
                    rec["parent"]["imputed_levels"] = made
            out[name] = rec
            print(name, rec["n_train"], rec["n_test"], rec["n_inputs"], round(rec["positive_train"], 4), flush=True)
        if getattr(cr.db, "engine", None) is not None:
            cr.db.engine.dispose()
    return out


def run_splits(cr, sets):
    """Which split did the reported runs use? Checked against the registry; returns the evidence."""
    from sqlalchemy import text
    checks = {}
    q_exp = text("SELECT split_id FROM experiments WHERE id = :e")
    q_gen = text("SELECT e.split_id, e.name FROM genomes g JOIN populations p ON g.population_id = p.id "
                 "JOIN experiments e ON p.experiment_id = e.id WHERE g.id = :g")
    # Experiment 2: every run in the runs CSVs
    for path, filt, name in EXP2_RUNS:
        dbname, sid, _ = CLASSIFICATION[name]
        cr.db.init_db(f"postgresql://localhost/{dbname}")
        rows = [r for r in csv.DictReader(path.open()) if filt is None or r["dataset"] == filt]
        with cr.ReadOnlySession() as s:
            got = [str(s.execute(q_exp, {"e": r["experiment_id"]}).scalar()) for r in rows]
        assert set(got) == {sid}, (path, set(got))
        for r in rows:
            assert int(r["n_train"]) == sets[name]["n_train"] and int(r["n_test"]) == sets[name]["n_test"], path
            assert int(r["n_inputs"]) == sets[name]["n_inputs"], path
        checks.setdefault(name, []).append({"experiment": 2, "file": str(path.relative_to(REPO)), "runs": len(rows),
                                            "split_id": sid})
        cr.db.engine.dispose()
    # Experiment 1: each explained network's genome (propneat.json), and the three original networks
    # (experiments/10seed/runs_paper_reference.csv), all presented in Experiment 1 (21 Nov 2026), whether or
    # not propneat.json holds them yet (its Adult entry is added on a separate branch)
    pn = json.loads(PROPNEAT.read_text())["exp1"]
    nets = [(tag, v["dataset"], v["genome_id"], v["split_id"]) for tag, v in pn.items() if v.get("genome_id")]
    seen = {g for _, _, g, _ in nets}
    for r in csv.DictReader(PAPER_REF.open()):
        assert r["seed"] == "paper", r["seed"]
        if r["genome_id"] not in seen:
            nets.append((f"paper-{r['dataset']}", r["dataset"], r["genome_id"], CLASSIFICATION[r["dataset"]][1]))
    for tag, name, gid, split_id in nets:
        dbname, sid, _ = CLASSIFICATION[name]
        cr.db.init_db(f"postgresql://localhost/{dbname}")
        with cr.ReadOnlySession() as s:
            got, ename = s.execute(q_gen, {"g": gid}).fetchone()
        assert str(got) == sid == split_id, (tag, got)
        checks.setdefault(name, []).append({"experiment": 1, "network": tag, "experiment_name": ename,
                                            "split_id": sid})
        cr.db.engine.dispose()
    return checks


def survival():
    import sksurv
    from sksurv import datasets as sd

    from explaneat.survival.data import load_builtin
    raw_loaders = {"whas500": sd.load_whas500, "gbsg2": sd.load_gbsg2}
    selected = json.loads((COX / "selected.json").read_text())
    out = {}
    for name, loader in raw_loaders.items():
        Xr, yr = loader()
        ev_field, t_field = yr.dtype.names
        X, y, names = load_builtin(name)
        sf = json.loads((COX / "splits" / f"{name}.splits.json").read_text())
        sp = sf["splits"]
        run = json.loads((COX / "runs" / f"{selected[name]['tag']}.json").read_text())
        ev = y[:, 1].astype(int)
        n = {k: len(v) for k, v in sp.items()}
        events = {k: int(ev[v].sum()) for k, v in sp.items()}
        # the split file, the loader and the selected run's record agree
        assert events == sf["events"], name
        assert (run["n"], run["n_features"], run["n_train"], run["n_val"], run["n_test"], run["events_train"]) == (
            len(y), X.shape[1], n["train"], n["val"], n["test"], events["train"]), name
        cat = [c for c in Xr.columns if str(Xr[c].dtype) == "category"]
        out[name] = {
            "loader": f"sksurv.datasets.load_{name} via explaneat.survival.data.load_builtin",
            "sksurv_version": sksurv.__version__,
            "split_file": str((COX / "splits" / f"{name}.splits.json").relative_to(REPO)),
            "run_record": str((COX / "runs" / f"{selected[name]['tag']}.json").relative_to(REPO)),
            "selected_tag": selected[name]["tag"],
            "n_rows": int(len(y)), "n_inputs_raw": int(Xr.shape[1]), "n_inputs": int(X.shape[1]),
            "categorical": cat, "categorical_levels": {c: len(Xr[c].cat.categories) for c in cat},
            "missing_cells": int(Xr.isna().sum().sum()),
            "event_field": ev_field, "time_field": t_field,
            "n_train": n["train"], "n_val": n["val"], "n_test": n["test"], "events": events,
            "event_rate_train": events["train"] / n["train"],
            "seed": sf["seed"], "test_size": sf["test_size"], "val_size": sf["val_size"],
            # survival_split_indices stratifies on the event; run_seed.py standardises with
            # explaneat.survival.data.standardise(X_train, ...) (training rows only)
            "stratify": "event", "scaler_fit_rows": "train",
        }
        print(name, out[name]["n_train"], out[name]["n_val"], out[name]["n_test"], flush=True)
    return out


def xor():
    sys.path.insert(0, str(REPO / "experiments" / "xor"))
    import run_xor
    res = json.loads((REPO / "experiments" / "xor" / "xor_results.json").read_text())
    X, Y = np.asarray(run_xor.X), np.asarray(run_xor.Y)
    assert np.array_equal(Y, np.logical_xor(X[:, 0], X[:, 1]).astype(float))
    return {"n_rows": int(len(Y)), "n_inputs": int(X.shape[1]), "positive": float(Y.mean()),
            "n_seeds": int(len(res["runs"])), "source": "experiments/xor/run_xor.py (X, Y)"}


def exp3_validation(sets):
    out = {}
    for name, path in EXP3_VAL.items():
        v = json.loads(path.read_text())
        assert v["split_id"] == CLASSIFICATION[name][1] and v["n_train"] == sets[name]["n_train"], name
        assert v["n_fit"] + v["n_val"] == v["n_train"], name
        out[name] = {"file": str(path.relative_to(REPO)), "n_val": v["n_val"], "n_fit": v["n_fit"],
                     "rule": v["rule"]}
    return out


def _shares(y, n_classes):
    c = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes)
    return [int(v) for v in c], [float(v / c.sum()) for v in c]


def mnist():
    """MNIST datasets of Experiment 4 (ExConNEAT): split sizes, class shares and pixel scaling, from the
    run records' saved row indices and the split code, asserted equal."""
    from explaneat.conv import data as cd
    Xr, yr = cd.load_mnist_raw()
    assert Xr.shape == (70000, 784) and Xr.dtype == np.uint8 and int(Xr.max()) == 255, Xr.shape
    n_std_train = cd.MNIST_N_STD_TRAIN
    out = {"source": {"loader": "explaneat.conv.data.load_mnist_raw (sklearn.datasets.fetch_openml('mnist_784', "
                                "version=1), cached as ~/.explaneat/data/mnist_784.npz",
                      "openml_name": "mnist_784", "openml_version": 1, "n_rows": int(len(yr)),
                      "n_standard_train": int(n_std_train), "n_standard_test": int(len(yr) - n_std_train),
                      "n_pixels": int(Xr.shape[1]), "image_hw": [28, 28], "pixel_max_raw": int(Xr.max()),
                      "scaling": "pixel / 255 (explaneat.conv.data.mnist_splits)"}}
    for key, runs in MNIST10_RUNS.items():
        recs = []
        for run in runs:
            cfg = json.loads((CONV / "results" / run / "config.json").read_text())
            a = cfg["args"]
            sp = cd.mnist_splits(a["seed"], classes=tuple(range(10)), n_train=a["n_train"], n_val=a["n_val"],
                                 raw=(Xr, yr))
            for part in ("train", "val", "test"):
                assert [int(r) for r in sp["rows"][part]] == cfg["rows"][part], (run, part)
                assert len(cfg["rows"][part]) == cfg["sizes"][part], (run, part)
            assert float(sp["X_train"].max()) == 1.0 and float(sp["X_train"].min()) == 0.0, run
            recs.append({"run": run, "seed": a["seed"], "n_train_arg": a["n_train"], "n_val_arg": a["n_val"],
                         "train_chunk": a.get("train_chunk"), "rows": {k: cfg["rows"][k] for k in cfg["rows"]}})
        rows = recs[0]["rows"]
        assert all(r["rows"] == rows for r in recs), key
        tr, va, te = (np.asarray(rows[k]) for k in ("train", "val", "test"))
        assert tr.max() < n_std_train and va.max() < n_std_train and te.min() >= n_std_train
        assert not (set(tr.tolist()) & set(va.tolist()))
        assert len(te) == len(yr) - n_std_train
        counts, shares = _shares(yr[tr], 10)
        out[key] = {
            "runs": [{k: v for k, v in r.items() if k != "rows"} for r in recs],
            "config_files": [str((CONV / "results" / r["run"] / "config.json").relative_to(REPO)) for r in recs],
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "n_classes": 10, "train_class_counts": counts, "train_class_shares": shares,
            "val_class_counts": _shares(yr[va], 10)[0], "test_class_counts": _shares(yr[te], 10)[0],
            "pool_after_val": int(n_std_train - len(va)),
            "train_is_whole_pool": bool(len(tr) == n_std_train - len(va)),
            "split_seed": recs[0]["seed"],
            "train_indices_sha256_16": hashlib.sha256(np.sort(tr).astype(np.int64).tobytes()).hexdigest()[:16],
            "val_indices_sha256_16": hashlib.sha256(np.sort(va).astype(np.int64).tobytes()).hexdigest()[:16],
        }
        print(key, len(tr), len(va), len(te), flush=True)
    # the two 10-class datasets share the validation and test rows
    assert out["mnist10_10k"]["val_indices_sha256_16"] == out["mnist10_50k"]["val_indices_sha256_16"]

    return out


def other_run_records(cr):
    """KRK and MNIST appear as Chapter 7 headings but not in the tables: list their run records."""
    from sqlalchemy import text
    q = text("SELECT d.name, e.name FROM experiments e JOIN datasets d ON d.id = e.dataset_id "
             "WHERE d.name ~* '^(krk|mnist)' ORDER BY d.name, e.name")
    out = {}
    for dbname in ("explaneat_dev", "explaneat_v4"):
        cr.db.init_db(f"postgresql://localhost/{dbname}")
        with cr.ReadOnlySession() as s:
            for dname, ename in s.execute(q).fetchall():
                out.setdefault(dbname, {}).setdefault(dname, []).append(ename)
        cr.db.engine.dispose()
    folders = {}
    for study in ("complex_real", "mnist", "conv_neat"):
        d = REPO / "experiments" / study
        if d.is_dir():
            folders[study] = sorted(p.name for p in d.iterdir() if p.suffix in (".csv", ".md"))
    return {"database_experiments": out, "study_folders": folders}


def main():
    import compute_runs as cr
    sets = classification(cr)
    data = {
        "krk_mnist_run_records": other_run_records(cr),
        "generated_by": "experiments/thesis_datasets/extract.py",
        "classification": sets,
        "run_splits": run_splits(cr, sets),
        "survival": survival(),
        "xor": xor(),
        "exp3_validation": exp3_validation(sets),
        "mnist": mnist(),
    }
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True, default=_jsonable) + "\n")
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
