"""Reference cases: the full input row that "others held fixed" plots hold the other inputs at.

A reference case is one row ``x`` in the dataset's raw columns (before the split's scaler).
``resolve`` builds it from a spec; ``describe`` says what it is and how typical it is among the
training rows. The plots themselves default to each column's median (``per_column_median``),
which for a one-hot group usually sets no level at all; the "median case" (``median``) takes
the most common pattern of each group and the most common value of each binary column instead.

Spec (all keys optional)::

    {"base": "median" | "per_column_median" | "mean" | "quantile" | "row",
     "q": 0.25,                       # for base "quantile"
     "row": 1234,                     # dataset row index, for base "row" (the caller loads it)
     "overrides": {"age": 61.0},      # raw column name -> value (raw units)
     "groups": {"marital": "marital:2"}}  # one-hot base -> level column (None: no level on)
"""
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .explanation_tools import base_feature, onehot_groups

BASES = ("median", "per_column_median", "mean", "quantile", "row")
K_NEAREST = 5
TYPICALITY_SAMPLE = 500
TYPICALITY_SEED = 0


def column_kinds(X: np.ndarray, names: Sequence[str]) -> Dict[str, Any]:
    """``kinds[i]`` in continuous / binary / onehot, and the one-hot ``groups`` (base -> columns)."""
    X = np.asarray(X, dtype=float)
    groups = onehot_groups(list(names), X)
    in_group = {c for cols in groups.values() for c in cols}
    kinds = []
    for i in range(X.shape[1]):
        if i in in_group:
            kinds.append("onehot")
        elif len(np.unique(X[:, i])) <= 2:
            kinds.append("binary")
        else:
            kinds.append("continuous")
    return {"kinds": kinds, "groups": groups}


def _mode(col: np.ndarray) -> float:
    vals, counts = np.unique(col, return_counts=True)
    return float(vals[np.argmax(counts)])  # ties: the smallest value (np.unique sorts)


def _modal_pattern(block: np.ndarray) -> np.ndarray:
    pats, counts = np.unique(block, axis=0, return_counts=True)
    return pats[np.argmax(counts)].astype(float)


def resolve(spec: Optional[Dict[str, Any]], X: np.ndarray, names: Sequence[str],
            row_values: Optional[np.ndarray] = None) -> np.ndarray:
    """The reference row (raw units) for ``spec`` over the rows ``X`` (raw units)."""
    spec = spec or {}
    X = np.asarray(X, dtype=float)
    names = list(names)
    base = spec.get("base") or "median"
    if base not in BASES:
        raise ValueError(f"unknown reference base {base!r}; one of {', '.join(BASES)}")
    info = column_kinds(X, names)
    kinds, groups = info["kinds"], info["groups"]

    if base == "row":
        if row_values is None:
            raise ValueError("a 'row' reference needs the row's values")
        x = np.asarray(row_values, dtype=float).ravel().copy()
        if x.shape[0] != X.shape[1]:
            raise ValueError(f"row has {x.shape[0]} values, the data {X.shape[1]} columns")
    elif base == "per_column_median":
        x = np.median(X, axis=0)
    else:
        if base == "mean":
            cont = np.mean(X, axis=0)
        elif base == "quantile":
            q = float(spec.get("q", 0.5))
            if not 0.0 <= q <= 1.0:
                raise ValueError("q must be between 0 and 1")
            cont = np.quantile(X, q, axis=0)
        else:
            cont = np.median(X, axis=0)
        x = cont.copy()
        for i, k in enumerate(kinds):
            if k == "binary":
                x[i] = _mode(X[:, i])
        for cols in groups.values():
            x[cols] = _modal_pattern(X[:, cols])

    index = {n: i for i, n in enumerate(names)}
    for name, level in (spec.get("groups") or {}).items():
        cols = groups.get(name)
        if cols is None:
            raise ValueError(f"{name!r} is not a one-hot group")
        for c in cols:
            x[c] = X[:, c].min()
        if level is not None:
            if level not in index or index[level] not in cols:
                raise ValueError(f"{level!r} is not a level of {name!r}")
            c = index[level]
            x[c] = X[:, c].max()
    for name, value in (spec.get("overrides") or {}).items():
        if name not in index:
            raise ValueError(f"unknown column {name!r}")
        x[index[name]] = float(value)
    return x


def _scales(X: np.ndarray) -> np.ndarray:
    sd = np.std(X, axis=0)
    return np.where(sd > 0, sd, 1.0)


def _nn_distances(Q: np.ndarray, X: np.ndarray, skip_self: Optional[np.ndarray] = None) -> np.ndarray:
    """Distance from each row of ``Q`` to its nearest row of ``X`` (``skip_self[i]``: the row of
    ``X`` that is query i itself, not counted)."""
    out = np.empty(len(Q))
    step = max(1, int(2e7 // max(1, len(X) * X.shape[1])))
    for s in range(0, len(Q), step):
        q = Q[s:s + step]
        d = np.sqrt(((q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        if skip_self is not None:
            d[np.arange(len(q)), skip_self[s:s + step]] = np.inf
        out[s:s + step] = d.min(axis=1)
    return out


def _mid_percentile(values: np.ndarray, v: float) -> float:
    return float((np.sum(values < v) + 0.5 * np.sum(values == v)) / max(1, len(values)))


def _outcome(y_row) -> Any:
    y_row = np.asarray(y_row, dtype=float).ravel()
    if y_row.size == 2:
        return {"time": float(y_row[0]), "event": int(y_row[1])}
    v = float(y_row[0])
    return int(v) if v.is_integer() else v


def describe(
    x: np.ndarray,
    X: np.ndarray,
    names: Sequence[str],
    predict: Callable[[np.ndarray], np.ndarray],
    y: Optional[np.ndarray] = None,
    row_ids: Optional[Sequence[int]] = None,
    labels: Optional[Sequence[Dict[str, Any]]] = None,
    output_names: Optional[Sequence[str]] = None,
    k: int = K_NEAREST,
) -> Dict[str, Any]:
    """What the case ``x`` is and how typical it is among the rows ``X`` (both raw units).

    ``predict`` maps raw rows to the model's outputs; ``row_ids`` are the rows' dataset indices
    (default 0..n-1); ``labels`` are Feature key entries (``feature_labels.feature_key``) per column.
    """
    X = np.asarray(X, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    names = list(names)
    n, d = X.shape
    row_ids = np.arange(n) if row_ids is None else np.asarray(row_ids)
    info = column_kinds(X, names)
    kinds, groups = info["kinds"], info["groups"]
    warnings: List[str] = []

    columns = []
    for i in range(d):
        lab = labels[i] if labels is not None and i < len(labels) else {}
        col = X[:, i]
        c = {
            "column": i, "raw": names[i],
            "label": lab.get("label", names[i]), "full_label": lab.get("full_label", names[i]),
            "one_means": lab.get("one_means"), "unit": lab.get("unit"),
            "kind": kinds[i], "value": float(x[i]),
            "median": float(np.median(col)), "mean": float(np.mean(col)),
            "min": float(col.min()), "max": float(col.max()),
            "q25": float(np.quantile(col, 0.25)), "q75": float(np.quantile(col, 0.75)),
            "percentile": _mid_percentile(col, float(x[i])),
        }
        if kinds[i] == "continuous" and not (c["min"] <= x[i] <= c["max"]):
            warnings.append(f"{names[i]} = {x[i]:g} is outside the data's range ({c['min']:g} to {c['max']:g})")
        if kinds[i] == "binary" and not np.any(col == x[i]):
            warnings.append(f"{names[i]} = {x[i]:g} is not one of its values in the data")
        columns.append(c)

    group_out = []
    for base, cols in groups.items():
        block = X[:, cols]
        pattern = x[cols]
        support = int(np.sum(np.all(block == pattern[None, :], axis=1)))
        on = [names[c] for c in cols if x[c] >= X[:, c].max() and X[:, c].max() > X[:, c].min()]
        group_out.append({"base": base, "columns": [int(c) for c in cols],
                          "levels": [names[c] for c in cols], "active": on, "support": support})
        if support == 0:
            what = "no level on" if not on else ("levels " + ", ".join(on) + " on together" if len(on) > 1 else f"{on[0]} on")
            warnings.append(f"{base}: no row in the data has this pattern ({what})")

    preds = np.asarray(predict(X), dtype=float)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    px = np.asarray(predict(x[None, :]), dtype=float).reshape(-1)
    onames = list(output_names) if output_names else [f"y_{j}" for j in range(len(px))]
    prediction = {
        "outputs": [{"name": onames[j] if j < len(onames) else f"y_{j}", "value": float(px[j]),
                     "percentile": _mid_percentile(preds[:, j], float(px[j]))}
                    for j in range(len(px))],
    }

    sd = _scales(X)
    Z = X / sd
    z = x / sd
    dist = np.sqrt(((Z - z[None, :]) ** 2).sum(axis=1))
    order = np.argsort(dist, kind="stable")[:k]
    nearest = []
    for r in order:
        diff = np.abs(Z[r] - z)
        differs = [names[i] for i in np.argsort(-diff, kind="stable")[:5] if diff[i] > 0.5]
        nearest.append({
            "row": int(row_ids[r]), "distance": float(dist[r]),
            "outcome": _outcome(y[r]) if y is not None else None,
            "prediction": [float(v) for v in preds[r]],
            "differs": differs,
        })

    rng = np.random.RandomState(TYPICALITY_SEED)
    m = min(TYPICALITY_SAMPLE, n)
    sample = np.sort(rng.choice(n, size=m, replace=False)) if n > m else np.arange(n)
    ref = _nn_distances(Z[sample], Z, skip_self=sample) if n > 1 else np.zeros(1)
    exact = np.flatnonzero(dist == 0)
    x_nn = float(np.sort(dist)[1]) if len(exact) and n > 1 else float(dist.min())
    pct = _mid_percentile(ref, x_nn)
    typicality = {
        "nn_distance": x_nn,
        "percentile": pct,
        "n_reference": int(len(ref)),
        "is_data_row": bool(len(exact)),
        "verdict": "typical" if pct <= 0.75 else ("unusual" if pct <= 0.95 else "far from the data"),
    }

    return {"x": [float(v) for v in x], "columns": columns, "groups": group_out,
            "prediction": prediction, "nearest": nearest, "typicality": typicality,
            "warnings": warnings, "n_rows": int(n)}
