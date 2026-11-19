"""Read-only loader for the signal-completeness check (and the description checks).

For each data set (adult, churn, heart100) writes ``cache/<set>.pkl`` (not committed; rebuilt from
explaneat_v4 in READ ONLY transactions): the training / test inputs, the catalogue basis ``Phi``
with its blocks and the variables each block reads, the data variables (for the residual scan),
and per network its logit and each explanation part's logit contribution (train and test).

    DATABASE_URL=postgresql://localhost/explaneat_v4 \\
    EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 \\
    uv run python experiments/signal_completeness/load.py [adult churn heart100]

The bases are the catalogues' own: ``experiments/adult_signals/catalogue.build_basis``,
``experiments/churn_signals/basis.build_basis`` and, for Heart-100, ``heart_basis.build`` (the
table's rows given data blocks). A part's contribution is ``plain_language.build_parts``'s: the
logit minus the logit with the part's exits fixed at their baseline (0 for a gated ReLU exit, else
the training mean).
"""
from __future__ import annotations

import csv
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = HERE / "cache"
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _module(name, path, extra_paths=()):
    for p in extra_paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def adult_catalogue():
    return _module("adult_catalogue", REPO / "experiments/adult_signals/catalogue.py")


def churn_modules():
    d = REPO / "experiments/churn_signals"
    common = _module("common", d / "common.py", [d])
    basis = _module("basis", d / "basis.py", [d])
    cat = _module("churn_catalogue", d / "catalogue.py", [d])
    return common, basis, cat


def heart_audit():
    d = REPO / "experiments/heart100_audit"
    return _module("audit", d / "audit.py", [d, REPO / "experiments/10seed", REPO / "experiments/churn"])


def labels(ds):
    if ds == "adult":
        with open(REPO / "experiments/10seed/runs.csv") as f:
            ids = {f"adult-s{r['seed']}": r["genome_id"] for r in csv.DictReader(f) if r["dataset"] == "adult"}
        return [(f"adult-s{i}", ids[f"adult-s{i}"]) for i in range(10)]
    from explaneat.export.plain_cli import genome_for_label
    pre = {"churn": "churn100", "heart100": "heart100"}[ds]
    return [(f"{pre}-s{i}", genome_for_label(f"{pre}-s{i}")) for i in range(10)]


def variables(ds, m):
    """Data variables for the residual scan: name -> (kind, train values, test values).
    Churn merges its alias columns (common.GROUPS: minutes stand for charges; voice mail = the
    number of messages, 0 without a plan)."""
    out = {}
    if ds == "churn":
        common, _, _ = churn_modules()
        for g, bases in common.GROUPS.items():
            v = m.var_by_base[bases[-1] if g == "voice mail" else bases[0]]
            kind = "cat" if v.is_category() else "num"
            out[g] = (kind, np.asarray(v.raw_train), np.asarray(v.raw_test))
        return out
    for v in m.variables:
        kind = "cat" if v.is_category() else "num"
        out[v.base] = (kind, np.asarray(v.raw_train), np.asarray(v.raw_test))
    return out


def basis(ds, m):
    """(Phi_tr, Phi_te, blocks, block_vars) on the shared split."""
    if ds == "adult":
        cat = adult_catalogue()
        D = dict(X=m.X, Xt=m.Xt, raw=m.raw_train, names=m.feature_names, mean=m.mean, scale=m.scale)
        B = cat.build_basis(D)
        bv = {b: list(v[2]) for b, v in cat.SIGNALS.items()}
        return B["Phi_tr"], B["Phi_te"], B["blocks"], bv
    if ds == "churn":
        _, bas, cat = churn_modules()
        B = bas.build_basis(m)
        return B.Phi_tr, B.Phi_te, B.blocks, {b: sorted(v) for b, v in cat.BLOCK_VARS.items()}
    import heart_basis
    audit = heart_audit()
    return heart_basis.build(m, audit.linear_basis)


def load(ds):
    from explaneat.analysis.plain_language import build_parts, load_model
    out = None
    for lab, gid in labels(ds):
        print("loading", lab, flush=True)
        m, meta = load_model(gid)
        if out is None:
            Phi_tr, Phi_te, blocks, bvars = basis(ds, m)
            out = dict(dataset=ds, X=m.X, Xt=m.Xt, y=m.y, yt=m.yt, feature_names=list(m.feature_names),
                       raw_train=np.asarray(m.raw_train, float), raw_test=np.asarray(m.raw_test, float),
                       mean=np.asarray(m.mean, float), scale=np.asarray(m.scale, float),
                       Phi_tr=Phi_tr, Phi_te=Phi_te, blocks=blocks, block_vars=bvars,
                       variables=variables(ds, m), nets={})
        else:
            assert np.array_equal(m.X, out["X"]) and np.array_equal(m.Xt, out["Xt"]), lab
        parts = []
        for p in build_parts(m):
            if p.is_root:
                continue
            parts.append(dict(id=p.id, name=p.name, kind=p.kind, duplicate_of=p.duplicate_of,
                              parent=p.parent, variables=list(p.variables), exits=list(p.exits),
                              c_tr=np.asarray(p.contribution, float), c_te=np.asarray(p.contribution_test, float)))
        out["nets"][lab] = dict(genome_id=gid, explanation_id=meta.get("explanation_id"),
                                eta_tr=np.asarray(m.base_logit, float), eta_te=np.asarray(m.base_logit_test, float),
                                parts=parts)
    return out


def get(ds, refresh=False):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{ds}.pkl"
    if p.exists() and not refresh:
        return pickle.loads(p.read_bytes())
    if "explaneat_v4" not in os.environ.get("DATABASE_URL", ""):
        sys.exit("set DATABASE_URL=postgresql://localhost/explaneat_v4 (read-only use)")
    if not os.environ.get("EXPLANEAT_RECORDS_DIR"):
        sys.exit("set EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4")
    from explaneat.db.base import db
    db.init_db()
    d = load(ds)
    p.write_bytes(pickle.dumps(d))
    return d


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["adult", "churn", "heart100"]):
        get(ds, refresh=True)
