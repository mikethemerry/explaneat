"""Read-only database layer for the Chapter 7 figure candidates (layer 2 of the provenance rule).

Writes small caches into data/; make_candidates.py only reads them. Every session is a READ ONLY
transaction on explaneat_v4 (``plain_language.load_model`` and ``compute_runs.ReadOnlySession`` both
issue SET TRANSACTION READ ONLY; ``PGOPTIONS=-c default_transaction_read_only=on`` makes the server
refuse any write as well). Nothing is written to any database.

    PGOPTIONS='-c default_transaction_read_only=on' DATABASE_URL=postgresql://localhost/explaneat_v4 \\
    EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 \\
    uv run python thesis/ch7/figure_candidates/extract_db.py [--refresh] [name ...]

Caches (data/):

- adult_shapes.json   Adult non-linear tick-table rows (capital gain, age, hours): each seed's
                      component (linear + bend blocks) of the projection of its logit on the
                      catalogue basis (experiments/adult_signals/catalogue.py: build_basis, project),
                      on a grid of the variable with every other column at its training mean, and
                      the data's own (beta_data from experiments/adult_signals/cache/betas.npz).
                      Asserts every block's clipped SD equals matrix.csv's sd_robust.
- heart_families.json Heart-100 non-linear families: for every row of
                      experiments/heart100_audit/family_presence.csv, the per-patient logit
                      contribution on the 242 training rows (full logit minus the logit with the
                      family's switching units replaced by their affine fit; audit.Net.family_presence),
                      plus the raw training values of the variables the figure groups by. Asserts the
                      clipped SD equals family_presence.csv's sd.
- seed_agreement.json Pairwise Pearson correlation of the ten networks' logits, per study, on the
                      training rows and on the test rows (plain_language.load_model, m.ev.logit).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = HERE / "data"
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/explaneat_v4")
os.environ.setdefault("EXPLANEAT_RECORDS_DIR", str(REPO / "experiments/10seed/explanations_v4"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PGOPTIONS", "-c default_transaction_read_only=on")
for p in (str(REPO), str(REPO / "experiments/10seed")):
    if p not in sys.path:
        sys.path.insert(0, p)

assert "explaneat_v4" in os.environ["DATABASE_URL"], "read from the explaneat_v4 clone only"

RUNS = {"adult": ("experiments/10seed/runs.csv", "adult"),
        "churn100": ("experiments/churn/runs_churn100.csv", None),
        "heart100": ("experiments/10seed/runs_heart100.csv", None)}

# design choices: the grids the shape curves are drawn on (raw units) and their reference points
ADULT_PANELS = {
    "shape:capgain": ("capital-gain", ["lin:capgain", "shape:capgain"], np.arange(0.0, 20001.0, 250.0), 0.0),
    "shape:age": ("age", ["lin:age", "shape:age"], np.arange(17.0, 81.0, 1.0), 40.0),
    "shape:hours": ("hours-per-week", ["lin:hours", "shape:hours"], np.arange(1.0, 81.0, 1.0), 40.0),
}
# Heart variables whose raw training values the family panels group by
HEART_VARS = ["age", "trestbps", "thalach", "oldpeak", "thal", "exang", "slope", "restecg", "sex", "cp", "ca", "chol"]


def _runs(study):
    path, ds = RUNS[study]
    with open(REPO / path) as f:
        rows = [r for r in csv.DictReader(f) if ds is None or r["dataset"] == ds]
    return {int(r["seed"]): r["genome_id"] for r in rows}


def _csv(path):
    with open(REPO / path) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ Adult shapes

def adult_shapes():
    import importlib.util
    spec = importlib.util.spec_from_file_location("adult_catalogue", REPO / "experiments/adult_signals/catalogue.py")
    ac = importlib.util.module_from_spec(spec)   # experiments/adult_signals/catalogue.py (by path: the
    spec.loader.exec_module(ac)                  # churn study has a module of the same name)
    from explaneat.analysis.plain_language import load_model
    from explaneat.analysis.signal_presence import presence
    ids = _runs("adult")
    D, eta = None, {}
    for s in range(10):
        m, meta = load_model(ids[s])
        assert meta["dataset"]["class_names"] == [">50k", "<=50k"]
        if D is None:
            D = dict(X=m.X, Xt=m.Xt, raw=m.raw_train, names=m.feature_names, mean=m.mean, scale=m.scale)
        else:
            assert np.max(np.abs(D["X"] - m.X)) == 0.0
        eta[s] = np.asarray(m.ev.logit(m.X), float)
        print("  adult", s, flush=True)
    B = ac.build_basis(D)
    bz = np.load(REPO / "experiments/adult_signals/cache/betas.npz")
    beta_d = bz["beta_data"]
    assert beta_d.shape[0] == B["Phi_tr"].shape[1], "basis differs from the catalogue's"
    mat = {(int(r["seed"]), r["signal"]): r for r in _csv("experiments/adult_signals/matrix.csv")}
    betas = {}
    for s in range(10):
        _, beta = ac.project(B, eta[s])
        betas[s] = beta
        for b in B["blocks"]:   # the projection reproduces the catalogue's sizes
            r = mat[(s, b)]
            sd = presence(ac.comp(B, beta, b))["sd"]
            assert abs(sd - float(r["sd_robust"])) < 1e-6, (s, b, sd, r["sd_robust"])
    names = list(D["names"])
    out = {}
    for sig, (var, blocks, grid, ref) in ADULT_PANELS.items():
        c = names.index(var)
        X = np.zeros((len(grid), len(names)))
        X[:, c] = (grid - D["mean"][c]) / D["scale"][c]
        P = B["phi"](X)

        def curve(beta):
            y = sum(P[:, B["blocks"][b]] @ beta[B["blocks"][b]] for b in blocks)
            return (y - np.interp(ref, grid, y)).tolist()

        raw = np.asarray(D["raw"][:, c], float)
        out[sig] = {"variable": var, "blocks": blocks, "ref": ref, "x": grid.tolist(),
                    "seeds": [curve(betas[s]) for s in range(10)], "data": curve(beta_d),
                    "presence": [mat[(s, sig)]["presence"] for s in range(10)],
                    "sd_robust": [float(mat[(s, sig)]["sd_robust"]) for s in range(10)],
                    "train_quantiles": {q: float(np.quantile(raw, float(q))) for q in ("0.01", "0.5", "0.99")},
                    "n_train": int(len(raw)), "n_in_grid": int(np.sum((raw >= grid[0]) & (raw <= grid[-1])))}
    return out


# ------------------------------------------------------------------ Heart families

def heart_families():
    sys.path.insert(0, str(REPO / "experiments/heart100_audit"))
    import audit   # experiments/heart100_audit/audit.py
    from audit import Net, ReadOnlySession, bootstrap_indices, load_split
    from explaneat.db.base import db
    db.init_db()
    with ReadOnlySession() as s:
        data = load_split(s, audit.SPLIT)
    boot = bootstrap_indices(len(data["y_test"]))
    fam = _csv("experiments/heart100_audit/family_presence.csv")
    rows, raw = [], None
    for lab in audit.LABELS:
        nt = Net(lab, data, boot)
        if raw is None:
            raw = {}
            for v in nt.m.variables:
                if v.base in HEART_VARS:
                    vals = v.raw_train
                    raw[v.base] = [x if isinstance(x, str) else float(x) for x in np.asarray(vals).tolist()]
        for r in fam:
            if r["network"] != lab:
                continue
            units = r["units"].split()
            c = nt.full_tr - nt.m.ev.logit(nt.m.X, nt.lin.overrides(units))
            pr = nt.family_presence(units)
            assert abs(pr["sd"] - float(r["sd"])) < 1e-9 and pr["state"] == r["presence"], (lab, r["row_family"])
            rows.append({"network": lab, "seed": int(r["seed"]), "row_family": r["row_family"], "units": r["units"],
                         "sd": pr["sd"], "presence": pr["state"], "contribution": [float(x) for x in c]})
        print("  heart", lab, flush=True)
    return {"n_train": len(data["y_train"]), "raw_train": raw, "families": rows}


# ------------------------------------------------------------------ seed agreement

def seed_agreement():
    from explaneat.analysis.plain_language import load_model
    out = {}
    for study in RUNS:
        ids = _runs(study)
        Ltr, Lte = [], []
        for s in range(10):
            m, _ = load_model(ids[s])
            Ltr.append(np.asarray(m.ev.logit(m.X), float))
            Lte.append(np.asarray(m.ev.logit(m.Xt), float))
        Ltr, Lte = np.array(Ltr), np.array(Lte)
        out[study] = {"genomes": [ids[s] for s in range(10)], "n_train": int(Ltr.shape[1]), "n_test": int(Lte.shape[1]),
                      "logit_corr_train": np.corrcoef(Ltr).tolist(), "logit_corr_test": np.corrcoef(Lte).tolist()}
        print("  agreement", study, flush=True)
    return out


JOBS = {"adult_shapes.json": adult_shapes, "heart_families.json": heart_families,
        "seed_agreement.json": seed_agreement}


def ensure(refresh: bool = False, only=None):
    DATA.mkdir(exist_ok=True)
    for name, job in JOBS.items():
        if only and name not in only:
            continue
        p = DATA / name
        if p.exists() and not refresh:
            continue
        print(f"[db, read-only] computing {name} ...", flush=True)
        p.write_text(json.dumps(job(), indent=0, sort_keys=True))


if __name__ == "__main__":
    ensure(refresh="--refresh" in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")] or None)
