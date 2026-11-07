"""The cohort CSV path, shared by the portal import and experiments/survival/run_seed.py.

Both read the file with ``read_table`` and build the design matrix with
``encode_features``, so the scripted runner and the portal see identical
inputs from the same file. Also: UTF-8 with a BOM (Excel on Windows), text
event values, and the Windows-safe JSON writer used for job files.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from explaneat.survival.data import encode_features, load_survival_csv, match_value, read_table


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.normal(60, 8, n),
        "grade": rng.choice(["I", "II", "III"], n),
        "smoker": rng.choice([0, 1], n),
        "fu_years": rng.uniform(0.1, 10, n),
        "status": rng.choice(["dead", "alive"], n),
    })


def test_bom_csv_reads_like_plain_utf8(tmp_path):
    df = _frame()
    plain, bom = tmp_path / "plain.csv", tmp_path / "bom.csv"
    df.to_csv(plain, index=False, encoding="utf-8")
    df.to_csv(bom, index=False, encoding="utf-8-sig")
    assert bom.read_bytes()[:3] == b"\xef\xbb\xbf"
    a, b = read_table(plain), read_table(bom)
    assert list(a.columns) == list(b.columns) == list(df.columns)
    pd.testing.assert_frame_equal(a, b)


def test_runner_and_portal_build_the_same_design_matrix(tmp_path):
    from explaneat.batch.datasets import _encode

    df = _frame()
    path = tmp_path / "c.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    X, y, names = load_survival_csv(path, "fu_years", "status", features=["age", "grade", "smoker"],
                                    categorical=["smoker"], event_value="dead")
    X2, names2 = _encode(read_table(path), ["age", "grade", "smoker"], ["smoker"], None)
    assert names == names2 == ["age", "grade:I", "grade:II", "grade:III", "smoker:0", "smoker:1"]
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y[:, 1], (df["status"] == "dead").to_numpy().astype(float))


def test_match_value_numbers_and_text():
    s = pd.Series([1, 0, 1])
    assert match_value(s, "1").tolist() == [True, False, True]
    assert match_value(s, 1.0).tolist() == [True, False, True]
    assert match_value(pd.Series(["yes", "no"]), "yes").tolist() == [True, False]


def test_missing_values_raise_unless_median():
    df = _frame()
    df.loc[3, "age"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        encode_features(df, ["age"], [], None)
    X, _ = encode_features(df, ["age"], [], "median")
    assert np.isfinite(X).all()


def test_missing_outcome_always_raises(tmp_path):
    df = _frame()
    df.loc[0, "fu_years"] = np.nan
    path = tmp_path / "c.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="outcome"):
        load_survival_csv(path, "fu_years", "status", features=["age"], impute="median", event_value="dead")


def test_write_json_atomic_retries_and_leaves_no_temp(tmp_path, monkeypatch):
    import os

    from explaneat.batch import common

    target = tmp_path / "progress.json"
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):  # Windows: the target is briefly open in another process
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr(common.os, "replace", flaky)
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    common.write_json_atomic(target, {"status": "running"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "running"}
    assert calls["n"] == 3
    assert [p.name for p in tmp_path.iterdir()] == ["progress.json"]


def test_run_seed_restricted_mode_refuses_repo_paths(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "experiments" / "survival"))
    try:
        import run_seed
    finally:
        sys.path.pop(0)
    outside = tmp_path / "cohort.csv"
    outside.write_text("a\n", encoding="utf-8")
    monkeypatch.setenv("EXPLANEAT_HOME", str(tmp_path / "home"))

    def args(**kw):
        base = dict(csv=str(outside), out=tmp_path / "runs", split_file=None, db=False)
        base.update(kw)
        return SimpleNamespace(**base)

    monkeypatch.delenv("EXPLANEAT_RESTRICTED", raising=False)
    run_seed.check_cohort_paths(args(out=repo / "experiments" / "survival" / "runs"))  # not restricted: allowed

    monkeypatch.setenv("EXPLANEAT_RESTRICTED", "1")
    run_seed.check_cohort_paths(args())  # everything outside the repo
    for bad in (args(out=repo / "experiments" / "survival" / "runs"),
                args(csv=str(repo / "cohort.csv")),
                args(split_file=repo / "fold0.json")):
        with pytest.raises(SystemExit, match="restricted mode"):
            run_seed.check_cohort_paths(bad)
