import numpy as np
import pytest

from explaneat.analysis import reference_case as rc

NAMES = ["age", "smoker", "colour:red", "colour:green", "colour:blue"]


@pytest.fixture
def data():
    rng = np.random.RandomState(1)
    n = 300
    age = rng.normal(50, 10, n).round(1)
    smoker = (rng.rand(n) < 0.3).astype(float)
    lvl = rng.choice(3, size=n, p=[0.45, 0.35, 0.2])
    onehot = np.eye(3)[lvl]
    X = np.column_stack([age, smoker, onehot])
    y = (age > 55).astype(float)
    return X, y


def predict(X):
    X = np.asarray(X, dtype=float)
    return 1 / (1 + np.exp(-(X[:, 0] - 50) / 10 - X[:, 1]))


def test_kinds(data):
    X, _ = data
    info = rc.column_kinds(X, NAMES)
    assert info["kinds"] == ["continuous", "binary", "onehot", "onehot", "onehot"]
    assert info["groups"] == {"colour": [2, 3, 4]}


def test_per_column_median_has_no_level_and_is_flagged(data):
    X, y = data
    x = rc.resolve({"base": "per_column_median"}, X, NAMES)
    assert x[2:].sum() == 0
    d = rc.describe(x, X, NAMES, predict, y=y)
    g = d["groups"][0]
    assert g["support"] == 0 and g["active"] == []
    assert any("colour" in w for w in d["warnings"])


def test_median_case_is_realistic(data):
    X, y = data
    x = rc.resolve(None, X, NAMES)
    assert x[0] == np.median(X[:, 0])
    assert x[1] == 0.0  # most rows do not smoke
    assert x[2:].tolist() == [1.0, 0.0, 0.0]  # the most common colour
    d = rc.describe(x, X, NAMES, predict, y=y)
    assert d["groups"][0]["active"] == ["colour:red"] and d["groups"][0]["support"] > 0
    assert d["warnings"] == []


def test_overrides_groups_quantile(data):
    X, _ = data
    x = rc.resolve({"base": "quantile", "q": 0.75, "groups": {"colour": "colour:blue"},
                    "overrides": {"smoker": 1}}, X, NAMES)
    assert x[0] == pytest.approx(np.quantile(X[:, 0], 0.75))
    assert x[1] == 1.0 and x[2:].tolist() == [0.0, 0.0, 1.0]
    none = rc.resolve({"groups": {"colour": None}}, X, NAMES)
    assert none[2:].sum() == 0
    with pytest.raises(ValueError):
        rc.resolve({"groups": {"colour": "age"}}, X, NAMES)
    with pytest.raises(ValueError):
        rc.resolve({"overrides": {"nope": 1}}, X, NAMES)
    with pytest.raises(ValueError):
        rc.resolve({"base": "bogus"}, X, NAMES)


def test_row_case_nearest_is_itself(data):
    X, y = data
    ids = np.arange(1000, 1000 + len(X))
    x = rc.resolve({"base": "row"}, X, NAMES, row_values=X[17])
    d = rc.describe(x, X, NAMES, predict, y=y, row_ids=ids)
    assert d["nearest"][0]["row"] == 1017 and d["nearest"][0]["distance"] == 0
    assert d["nearest"][0]["outcome"] == int(y[17])
    assert d["typicality"]["is_data_row"]
    assert 0 <= d["typicality"]["percentile"] <= 1


def test_far_case_and_determinism(data):
    X, y = data
    x = rc.resolve({"overrides": {"age": 200}}, X, NAMES)
    a = rc.describe(x, X, NAMES, predict, y=y)
    b = rc.describe(x, X, NAMES, predict, y=y)
    assert a == b
    assert a["typicality"]["verdict"] == "far from the data"
    assert any("outside the data's range" in w for w in a["warnings"])
    assert a["prediction"]["outputs"][0]["percentile"] == 1.0


def test_survival_outcome():
    assert rc._outcome([12.5, 1]) == {"time": 12.5, "event": 1}
    assert rc._outcome([1.0]) == 1
