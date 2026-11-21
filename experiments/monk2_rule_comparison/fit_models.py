"""MONK-2: does a glass-box comparator's explanation state the rule "exactly two of the six attributes
take their first value"? (exploratory, 22 Nov 2026; not pre-registered)

    OMP_NUM_THREADS=2 /Users/mike/dev/explaneat/.claude/worktrees/conv-neat/.venv/bin/python \\
        experiments/monk2_rule_comparison/fit_models.py

(needs `interpret` for the EBM, as experiments/benchmarks/classification.py; the conv-neat worktree venv has
InterpretML 0.7.8 and the same scikit-learn 1.7.2 as the uv environment). Reads data/monk2_split.json
(extract_data.py, read-only DB) and, by reference, the comparator suite's MONK-2 results
(experiments/benchmarks/results/classification/monk2.json: the EBM settings its CV chose) and data manifest.
No database access. Writes results/monk2_rules.json (everything) and results/monk2_rules.csv (one row per model).

Inputs (all on the split the PropNEAT runs used; training rows only for fitting, the test rows read once):
  * one-hot: the 17 one-hot columns scaled with the split's stored StandardScaler, the arrays of the suite
    (their hash is asserted equal to experiments/benchmarks/results/data_manifest.json);
  * raw: the six attributes a1..a6 as their values (1, 2, ...), recovered from the one-hot columns;
  * count: one engineered feature, the number of attributes at their first value (value 1). The MONK-2 label
    is asserted to be exactly [count == 2] on every row.

Models (random_state 42 everywhere; EBM n_jobs 2 as the suite):
  1. ebm_onehot: EBM on the one-hot inputs with the settings the suite's CV chose for MONK-2, read from its
     results file (whichever is current); its test AUC is asserted equal to the suite's.
  2. ebm_raw: EBM on the raw attributes (nominal), all 15 pairwise interactions.
  3. tree_onehot, tree_raw: decision tree, unlimited depth (scikit-learn defaults; raw attributes as ordered
     integers, scikit-learn has no nominal split).
  4. tree_count, ebm_count: decision tree (unlimited depth) and EBM without interactions on the count alone.

Measured: test AUC and accuracy; explanation size (EBM: terms, mains vs pairs, the fewest terms covering 80%
and 95% of the total term importance, InterpretML's term_importances "avg_weight" on the training rows; tree:
depth and leaves); and whether the explanation states the rule:
  * count models: the count is the only input; the tree's leaves / the EBM's intercept plus count term, read
    at every count value seen in training, give class 1 exactly at count 2 ("only given the count" when they
    do: the rule is stated only once the count is engineered in);
  * EBMs without the count (raw, one-hot): no term reads the count; the model's log-odds on the training rows,
    grouped by the count (mean, min, max per count), are reported. If they peak at 2 and separate the classes
    (the smallest log-odds at count 2 above the largest at any other count) the rule is in the sum of the terms
    and can be read only by regrouping by the count ("only given the count"); otherwise "no";
  * trees without the count: no split reads the count; the leaves enumerate attribute combinations ("no").
  * PropNEAT's original network (for the table): from the committed plain-language layer of its explanation
    (experiments/10seed/plain_language/monk2-paper.json): the root's children with a nonzero contribution,
    and whether each reads all six attributes ("yes" when two such units, one linear and one gated, compose
    in the root: the count ramp and the at-least-three gate).
The per-row log-odds of ebm_raw (training rows) and its 15 pair terms (scores per value pair, value 1 = the
first value) are stored for the thesis figures. Design choices: the models, seeds, the 80% / 95% thresholds.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data" / "monk2_split.json"
BENCH = REPO / "experiments" / "benchmarks" / "results" / "classification" / "monk2.json"
MANIFEST = REPO / "experiments" / "benchmarks" / "results" / "data_manifest.json"
LAYER = REPO / "experiments" / "10seed" / "plain_language" / "monk2-paper.json"
OUT = HERE / "results"

SEED = 42
EBM_JOBS = 2                  # as the suite (experiments/benchmarks/classification.py)
COVER = (0.80, 0.95)          # design choice: importance shares reported
FIRST = 1                     # an attribute's first value


def rel(p):
    return str(Path(p).resolve().relative_to(REPO))


def load():
    d = json.loads(DATA.read_text())
    X = np.asarray(d["X_onehot"], dtype=np.float64)
    y = np.asarray(d["y"], dtype=int)
    names = d["feature_names"]
    mean, scale = np.asarray(d["scaler"]["mean"]), np.asarray(d["scaler"]["scale"])
    Xs = (X - mean) / scale                                  # compute_runs.load_split, exactly
    tr, te = d["train_indices"], d["test_indices"]
    h = hashlib.sha256(np.ascontiguousarray(Xs[tr]).tobytes() + y[tr].tobytes()
                       + np.ascontiguousarray(Xs[te]).tobytes() + y[te].tobytes()).hexdigest()[:16]
    man = json.loads(MANIFEST.read_text())["monk2"]
    assert h == man["sha256_16"] and man["split_id"] == d["source"]["split_id"], (h, man)
    attrs = sorted({n.split(":")[0] for n in names})
    A = np.zeros((len(y), len(attrs)), dtype=int)
    for j, a in enumerate(attrs):
        cols = [i for i, n in enumerate(names) if n.split(":")[0] == a]
        assert (X[:, cols].sum(axis=1) == 1).all(), a
        A[:, j] = [int(names[cols[k]].split(":")[1]) for k in X[:, cols].argmax(axis=1)]
    count = (A == FIRST).sum(axis=1)
    assert np.array_equal(y, (count == 2).astype(int)), "MONK-2 label is not [count == 2]"
    return dict(d=d, Xs=Xs, A=A, count=count, y=y, tr=np.asarray(tr), te=np.asarray(te), names=names,
                attrs=attrs, hash=h)


def ebm(**kw):
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(random_state=SEED, n_jobs=EBM_JOBS, **kw)


def scores(m, Xtr, ytr, Xte, yte):
    p = m.predict_proba(Xte)[:, 1]
    return dict(test_auc=float(roc_auc_score(yte, p)), test_accuracy=float(accuracy_score(yte, m.predict(Xte))),
                train_auc=float(roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1])),
                train_accuracy=float(accuracy_score(ytr, m.predict(Xtr))))


def ebm_size(m):
    imp = np.asarray(m.term_importances(), dtype=float)
    order = np.argsort(-imp, kind="stable")
    share = np.cumsum(imp[order]) / imp.sum()
    pairs = sum(len(f) == 2 for f in m.term_features_)
    out = dict(kind="ebm", n_terms=len(m.term_features_), n_main=sum(len(f) == 1 for f in m.term_features_),
               n_pairs=pairs, n_higher=sum(len(f) > 2 for f in m.term_features_),
               importance={m.term_names_[i]: float(imp[i]) for i in order})
    for c in COVER:
        out[f"terms_for_{int(round(c * 100))}pct"] = int(np.searchsorted(share, c - 1e-12) + 1)
    return out


def tree_size(m):
    return dict(kind="tree", depth=int(m.get_depth()), leaves=int(m.get_n_leaves()))


def by_count(logit, count, y):
    """The model's log-odds grouped by the count: per count value, n, mean, min, max, and the class."""
    rows = {}
    for c in sorted(set(count.tolist())):
        z = logit[count == c]
        rows[str(c)] = dict(n=int(len(z)), mean=float(z.mean()), min=float(z.min()), max=float(z.max()),
                            label=int(y[count == c][0]))
    means = {int(c): r["mean"] for c, r in rows.items()}
    peak = max(means, key=means.get)
    lo2 = rows["2"]["min"]
    hi_other = max(r["max"] for c, r in rows.items() if c != "2")
    return dict(per_count=rows, peak_count=peak, separates=bool(lo2 > hi_other),
                margin=float(lo2 - hi_other))


def nominal_pairs(m, attrs):
    """Each pair term's scores on the attribute values (rows: the first attribute's values 1..k, columns: the
    second's), without InterpretML's missing / unknown bins; value 1 (the first value) first on both axes."""
    out = {}
    for name, feats, sc in zip(m.term_names_, m.term_features_, m.term_scores_):
        if len(feats) != 2:
            continue
        i, j = feats
        bi, bj = m.bins_[i][0], m.bins_[j][0]
        vi = sorted(bi, key=int)
        vj = sorted(bj, key=int)
        mat = [[float(sc[bi[a], bj[b]]) for b in vj] for a in vi]
        out[name] = dict(rows=attrs[i], cols=attrs[j], row_values=[int(v) for v in vi],
                         col_values=[int(v) for v in vj], scores=mat)
    return out


def propneat_record():
    """The original network's explanation, read from its committed plain-language layer."""
    L = json.loads(LAYER.read_text())
    parts = {p["name"]: p for p in L["parts"]}
    roots = [p for p in L["parts"] if p["role"] == "root"]
    assert len(roots) == 1, roots
    root = roots[0]
    six = sorted(root["variable_keys"])
    active = [parts[c] for c in root["children"] if (parts[c]["contribution_sd"] or 0) > 0]
    units = [dict(name=p["name"], linear=not p["nonlinear"], gated=bool(p["gated"]),
                  reads_all=sorted(p["variable_keys"]) == six, contribution_sd=p["contribution_sd"]) for p in active]
    stated = (len(units) == 2 and all(u["reads_all"] for u in units)
              and sorted((u["linear"], u["gated"]) for u in units) == [(False, True), (True, False)])
    return dict(source=rel(LAYER), genome_id=L["genome_id"], explanation_id=L["explanation_id"],
                n_parts=len(L["parts"]), n_leaf=sum(p["kind"] == "leaf" for p in L["parts"]),
                n_composition=sum(p["kind"] == "composition" for p in L["parts"]), root=root["name"],
                attributes=six, active_units=units, rule_stated="yes" if stated else "no")


def main():
    D = load()
    tr, te, y = D["tr"], D["te"], D["y"]
    Xs, A, cnt = D["Xs"], D["A"], D["count"].reshape(-1, 1)
    bench = json.loads(BENCH.read_text())["monk2"]["ebm"]
    res = dict(data=dict(source=rel(DATA), split=D["d"]["source"], sha256_16=D["hash"], n_train=int(len(tr)),
                         n_test=int(len(te)), attributes=D["attrs"], onehot=D["names"],
                         count_values_train=sorted(set(D["count"][tr].tolist()))),
               cover=list(COVER), first_value=FIRST,
               ebm_settings_source=dict(file=rel(BENCH), params=bench["params"], test_auc=bench["test_auc"]),
               models={})
    M = res["models"]

    # 1. EBM, one-hot, the suite's CV-chosen settings
    m = ebm(**bench["params"]).fit(Xs[tr], y[tr])
    s = scores(m, Xs[tr], y[tr], Xs[te], y[te])
    assert abs(s["test_auc"] - bench["test_auc"]) < 1e-12, (s["test_auc"], bench["test_auc"])
    lc = by_count(m.decision_function(Xs[tr]), D["count"][tr], y[tr])
    M["ebm_onehot"] = dict(inputs="one-hot", n_inputs=Xs.shape[1], params=bench["params"], **s, size=ebm_size(m),
                           logit_by_count=lc, rule_stated="only given the count" if (lc["peak_count"] == 2 and lc["separates"]) else "no")

    # 2. EBM, raw attributes (nominal), all pairs
    pairs = list(itertools.combinations(range(A.shape[1]), 2))
    m = ebm(feature_types=["nominal"] * A.shape[1], feature_names=D["attrs"], interactions=pairs).fit(A[tr], y[tr])
    s = scores(m, A[tr], y[tr], A[te], y[te])
    z = m.decision_function(A[tr])
    lc = by_count(z, D["count"][tr], y[tr])
    M["ebm_raw"] = dict(inputs="raw", n_inputs=A.shape[1], params=dict(interactions=f"all {len(pairs)} pairs",
                                                                         feature_types="nominal"),
                        **s, size=ebm_size(m), logit_by_count=lc,
                        rule_stated="only given the count" if (lc["peak_count"] == 2 and lc["separates"]) else "no",
                        train_rows=dict(count=D["count"][tr].tolist(), logit=[float(v) for v in z],
                                        y=y[tr].tolist()),
                        pair_terms=nominal_pairs(m, D["attrs"]),
                        intercept=float(np.ravel(m.intercept_)[0]))

    # 3. trees, unlimited depth
    for key, X in (("tree_onehot", Xs), ("tree_raw", A)):
        t = DecisionTreeClassifier(random_state=SEED).fit(X[tr], y[tr])
        M[key] = dict(inputs={"tree_onehot": "one-hot", "tree_raw": "raw"}[key], n_inputs=X.shape[1], params="unlimited depth (defaults)",
                      **scores(t, X[tr], y[tr], X[te], y[te]), size=tree_size(t), rule_stated="no")

    # 4. the count alone
    seen = np.array(res["data"]["count_values_train"]).reshape(-1, 1)
    rule = (seen.ravel() == 2).astype(int)
    t = DecisionTreeClassifier(random_state=SEED).fit(cnt[tr], y[tr])
    tc = t.predict(seen)
    M["tree_count"] = dict(inputs="count", n_inputs=1, params="unlimited depth (defaults)",
                           **scores(t, cnt[tr], y[tr], cnt[te], y[te]), size=tree_size(t),
                           thresholds=sorted(float(v) for v in t.tree_.threshold[t.tree_.feature >= 0]),
                           class_by_count={str(int(c)): int(k) for c, k in zip(seen.ravel(), tc)},
                           rule_stated="only given the count" if np.array_equal(tc, rule) else "no")
    m = ebm(interactions=0).fit(cnt[tr], y[tr])
    term = m.eval_terms(seen)[:, 0]
    z = np.ravel(m.intercept_)[0] + term
    M["ebm_count"] = dict(inputs="count", n_inputs=1, params=dict(interactions=0),
                          **scores(m, cnt[tr], y[tr], cnt[te], y[te]), size=ebm_size(m),
                          shape_by_count={str(int(c)): float(v) for c, v in zip(seen.ravel(), term)},
                          intercept=float(np.ravel(m.intercept_)[0]),
                          rule_stated="only given the count" if np.array_equal((z > 0).astype(int), rule) else "no")

    res["propneat"] = propneat_record()

    OUT.mkdir(exist_ok=True)
    (OUT / "monk2_rules.json").write_text(json.dumps(res, indent=1) + "\n")
    cols = ["model", "inputs", "n_inputs", "test_auc", "test_accuracy", "train_auc", "size_kind", "depth", "leaves",
            "n_terms", "n_main", "n_pairs", "terms_for_80pct", "terms_for_95pct", "rule_stated"]
    with open(OUT / "monk2_rules.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for k, r in M.items():
            sz = r["size"]
            w.writerow(dict(model=k, inputs=r["inputs"], n_inputs=r["n_inputs"], test_auc=repr(r["test_auc"]),
                            test_accuracy=repr(r["test_accuracy"]), train_auc=repr(r["train_auc"]),
                            size_kind=sz["kind"], depth=sz.get("depth", ""), leaves=sz.get("leaves", ""),
                            n_terms=sz.get("n_terms", ""), n_main=sz.get("n_main", ""), n_pairs=sz.get("n_pairs", ""),
                            terms_for_80pct=sz.get("terms_for_80pct", ""), terms_for_95pct=sz.get("terms_for_95pct", ""),
                            rule_stated=r["rule_stated"]))
    for k, r in M.items():
        print(f"{k:12s} test AUC {r['test_auc']:.4f} acc {r['test_accuracy']:.4f} size {r['size'].get('n_terms', r['size'].get('leaves'))}"
              f" rule: {r['rule_stated']}", flush=True)
    print("propneat", res["propneat"]["rule_stated"], [u["name"] for u in res["propneat"]["active_units"]])


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
