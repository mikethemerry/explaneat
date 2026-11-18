"""Database-derived inputs for the Experiment 2 thesis assets (read-only).

Called by make_assets.py when a cache file in data/ is missing or --refresh is given. Every
session is a READ ONLY transaction on explaneat_v4 (compute_runs.ReadOnlySession and
plain_language.load_model both issue SET TRANSACTION READ ONLY); nothing is written to any
database. Outputs (data/):

- agreement.json   prediction agreement between seeds, Churn and Heart-100 (TEST), and Churn with
                   every state column held at its training average (method of the 18 Nov
                   measurement, agree.py / agree2.py, copied below)
- paper_net.json   the AAAI paper's Heart network: train and test AUC, test 95% interval
                   (10,000 bootstrap resamples, RandomState(42), as runs*.csv)
- s8_ice.npz       heart100-s8 resting-BP ICE curves and families (explaneat dossier, same config)
- churn_shapes.json  each signal's realised shape in the ten churn100 networks and the data's own
                   (projection on the catalogue basis, experiments/churn_signals/figures.py)
- pairs_s2.json    whole-model pairs plot of heart100-s2 (viz_data.compute_pairs, others at the
                   training median, axes in real units)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = HERE / "data"
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/explaneat_v4")
os.environ.setdefault("EXPLANEAT_RECORDS_DIR", str(REPO / "experiments/10seed/explanations_v4"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
for p in (str(REPO), str(REPO / "experiments/10seed")):
    if p not in sys.path:
        sys.path.insert(0, p)

assert "explaneat_v4" in os.environ["DATABASE_URL"], "read from the explaneat_v4 clone only"

HEART_SPLIT = "165cebbd-6609-46ca-867e-c09e17b2d6ec"
CHURN_SPLIT = "45962ea6-3c78-49f9-a01e-3665f6809b1e"
PAPER_GENOME = "9022a9eb-ecf9-41ba-b138-28ed4d131c9d"   # experiments/aaai27/recheck/results/heart_paper_genome.md


def _runs(path):
    return sorted(csv.DictReader(open(REPO / path)), key=lambda r: r["experiment_name"])


def _genome_id(path, seed):
    return next(r["genome_id"] for r in _runs(path) if int(r["seed"]) == seed)


# ------------------------------------------------------------------ agreement (agree.py, agree2.py)

def agreement():
    from compute_runs import ReadOnlySession, load_split, predict, load_neat_config
    from explaneat.db import db
    from explaneat.db.models import Genome
    from sklearn.metrics import roc_auc_score
    db.init_db(os.environ["DATABASE_URL"])
    spl = {"churn100": (CHURN_SPLIT, "experiments/churn/runs_churn100.csv"),
           "heart100": (HEART_SPLIT, "experiments/10seed/runs_heart100.csv")}
    out = {}
    for study, (sid, f) in spl.items():
        rows = _runs(f)
        with ReadOnlySession() as s:
            data = load_split(s, sid)
            fn = list(data["feature_names"])
            st = [i for i, c in enumerate(fn) if c.startswith("state")]
            variants = [("as trained", data["X_test"])]
            if study == "churn100":
                Xs = data["X_test"].copy()
                Xs[:, st] = 0.0   # every state column at its training mean (z = 0)
                variants.append(("state held at average", Xs))
            preds = {}
            for tag, X in variants:
                P = []
                for r in rows:
                    g = s.get(Genome, uuid.UUID(r["genome_id"]))
                    exp = g.population.experiment
                    cfg = load_neat_config(exp.neat_config_text or "", exp.config_json)
                    P.append(np.clip(predict(g.to_neat_genome(cfg), cfg, X), 1e-9, 1 - 1e-9))
                preds[tag] = np.array(P)
        y = data["y_test"]
        k = int(y.sum())
        res = {"n_test": int(len(y)), "k": k, "n_state_columns": len(st)}
        for tag, P in preds.items():
            L = np.log(P / (1 - P))
            n = len(P)
            iu = np.triu_indices(n, 1)
            r = np.corrcoef(L)[iu]
            ranks = np.argsort(np.argsort(L, axis=1), axis=1)
            rho = np.corrcoef(ranks)[iu]
            top = [set(np.argsort(-p)[:k]) for p in P]
            ov = np.array([len(top[i] & top[j]) / k for i, j in zip(*iu)])
            aucs = [roc_auc_score(y, p) for p in P]
            res[tag] = {"logit_corr_median": float(np.median(r)), "logit_corr_min": float(r.min()),
                        "rank_corr_median": float(np.median(rho)),
                        "topk_overlap_median": float(np.median(ov)), "topk_overlap_min": float(ov.min()),
                        "auc_median": float(np.median(aucs)), "auc_min": float(min(aucs)),
                        "auc_max": float(max(aucs)), "ensemble_auc": float(roc_auc_score(y, P.mean(0))),
                        "mean_abs_risk_diff_median": float(np.median([np.abs(P[i] - P[j]).mean()
                                                                      for i, j in zip(*iu)]))}
        out[study] = res
    return out


# ------------------------------------------------------------------ the paper's Heart network

def paper_net():
    from compute_runs import (ReadOnlySession, load_split, load_neat_config, bootstrap_indices,
                              performance)
    from explaneat.db import db
    from explaneat.db.models import Genome
    db.init_db(os.environ["DATABASE_URL"])
    with ReadOnlySession() as s:
        data = load_split(s, HEART_SPLIT)
        g = s.get(Genome, uuid.UUID(PAPER_GENOME))
        exp = g.population.experiment
        assert str(exp.split_id) == HEART_SPLIT
        cfg = load_neat_config(exp.neat_config_text or "", exp.config_json)
        exp_name = exp.name
        perf = performance(g.to_neat_genome(cfg), cfg, data, bootstrap_indices(len(data["y_test"])),
                           float(g.fitness))
    return {"genome_id": PAPER_GENOME, "experiment": exp_name,
            **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in perf.items()
               if k in ("train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi", "n_boot_used",
                        "n_train", "n_test")}}


# ------------------------------------------------------------------ heart100-s8 resting-BP ICE

def s8_ice():
    from explaneat.analysis.feature_dossier import DossierConfig, build_dossier
    from explaneat.analysis.plain_language import load_model
    ref = json.loads((REPO / "experiments/feature_dossiers/heart100-s8_trestbps/dossier.json").read_text())
    m, meta = load_model(ref["meta"]["genome_id"])
    c = ref["config"]
    cfg = DossierConfig(grid_points=c["grid_points"], max_families=c["max_families"], min_family=c["min_family"])
    d = build_dossier(m, "trestbps", cfg=cfg, meta=meta)
    A = d["_arrays"]
    fam = [(f["family"], f["n"], f["shape_text"]) for f in d["families"]]
    assert fam == [(f["family"], f["n"], f["shape_text"]) for f in ref["families"]], "dossier changed"
    ev = [(e["row"], e["x"]) for e in A["events"] if e.get("visible")]
    np.savez_compressed(DATA / "s8_ice.npz", grid=A["grid"], L=A["L"], Prob=A["Prob"],
                        labels=A["labels"], raw=A["raw"], Lmed=A["Lmed"], pd=A["pd"],
                        ev_row=np.array([e[0] for e in ev]), ev_x=np.array([e[1] for e in ev]),
                        fam_names=np.array([f[0] for f in fam]), fam_n=np.array([f[1] for f in fam]),
                        fam_shape=np.array([f[2] for f in fam]))


# ------------------------------------------------------------------ churn signal shapes

def churn_shapes():
    sys.path.insert(0, str(REPO / "experiments/churn_signals"))
    import figures as cf   # experiments/churn_signals/figures.py
    from common import LABELS
    m, B, betas, beta_d = cf.load()
    grid_rows, block_curve, cat = cf.grid_rows, cf.block_curve, cf.cat
    day = np.linspace(40, 340, 61)
    calls = np.arange(0, 10, dtype=float)
    eve = np.linspace(60, 340, 57)
    imin = np.linspace(2, 18, 33)

    def ip_rows(plan):
        X = grid_rows(m, {"international plan": np.full(len(imin), plan)}, len(imin))
        for c, z in cat.assign_raw(m, {"total intl minutes": imin, "total intl charge": 0.27 * imin}).items():
            X[:, c] = z
        return X

    panels = {
        "calls": (calls, lambda: grid_rows(m, {"service calls": calls}, len(calls)), ["lin:csc", "shape:csc"], 1.0),
        "day": (day, lambda: grid_rows(m, {"day usage": day}, len(day)), ["lin:day", "shape:day"], 180.0),
        "eve": (eve, lambda: grid_rows(m, {"evening usage": eve}, len(eve)), ["lin:eve", "shape:eve"], 200.0),
        "calls_x_day": (day, lambda: (grid_rows(m, {"service calls": np.full(len(day), 5.0), "day usage": day}, len(day)),
                                      grid_rows(m, {"service calls": np.full(len(day), 1.0), "day usage": day}, len(day))),
                        ["int:csc*day"], None),
        "plan_x_intl": (imin, lambda: (ip_rows(1.0), ip_rows(0.0)), ["lin:intlplan", "int:intlplan*intl"], None),
    }
    out = {}
    for key, (x, rows, blocks, ref) in panels.items():
        def curve(beta):
            r = rows()
            if isinstance(r, tuple):
                return block_curve(B, beta, r[0], blocks) - block_curve(B, beta, r[1], blocks)
            c = block_curve(B, beta, r, blocks)
            return c - np.interp(ref, x, c)
        out[key] = {"x": x.tolist(), "blocks": blocks, "ref": ref,
                    "seeds": [curve(betas[l]).tolist() for l in LABELS], "data": curve(beta_d).tolist()}
    X1 = grid_rows(m, {"voice mail": np.array([25.0])}, 1)
    X0 = grid_rows(m, {"voice mail": np.array([0.0])}, 1)
    vm = lambda b: float((block_curve(B, b, X1, ["lin:vmail"]) - block_curve(B, b, X0, ["lin:vmail"]))[0])
    out["vmail"] = {"seeds": [vm(betas[l]) for l in LABELS], "data": vm(beta_d)}
    return out


# ------------------------------------------------------------------ heart100-s2 pairs plot

def pairs_s2():
    from explaneat.analysis import viz_data as vd
    from explaneat.analysis.plain_language import load_model
    gid = _genome_id("experiments/10seed/runs_heart100.csv", 2)
    m, meta = load_model(gid)
    X = np.asarray(m.X, dtype=float)
    fn = lambda Z: m.risk(m.logit(np.asarray(Z, dtype=float))).reshape(-1, 1)
    out = fn(X)
    names = list(m.feature_names) if hasattr(m, "feature_names") else list(m.names)
    data = vd.compute_pairs(fn, X, out, grid_size=25, max_scatter=2000, fix="median", center=0.5,
                            entry_names=names, exit_names=["predicted risk"])
    mean, scale = np.asarray(m.mean), np.asarray(m.scale)
    for inp in data["inputs"]:
        c = inp["dim"]
        f = lambda v: float(v * scale[c] + mean[c])
        for k in ("grid", "scatter"):
            inp[k] = [f(v) for v in inp[k]]
        for k in ("median", "min", "max"):
            inp[k] = f(inp[k])
        if "edges" in inp["hist"]:
            inp["hist"]["edges"] = [f(v) for v in inp["hist"]["edges"]]
        else:
            inp["hist"]["levels"] = [f(v) for v in inp["hist"]["levels"]]
    data["genome_id"] = gid
    data["label"] = "heart100-s2"
    return json.loads(json.dumps(data, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


# ------------------------------------------------------------------ churn blind spot (train counts)

def churn_blindspot():
    from compute_runs import ReadOnlySession
    from explaneat.db import db
    from explaneat.db.models import Dataset, DatasetSplit
    db.init_db(os.environ["DATABASE_URL"])
    with ReadOnlySession() as s:
        sp = s.get(DatasetSplit, uuid.UUID(CHURN_SPLIT))
        ds = s.get(Dataset, sp.dataset_id)
        X, y = ds.get_data()
        names = list(ds.feature_names)
        tr = list(sp.train_indices)
    X = np.asarray(X, dtype=float)[tr]
    y = np.asarray(y).ravel()[tr]
    col = lambda n: X[:, names.index(n)]
    plan, calls, mins = col("international plan") > 0.5, col("total intl calls"), col("total intl minutes")
    # the two regions named in experiments/churn_signals/CATALOGUE.md (train-only look at churn rates)
    max_calls, min_minutes = 2, 13
    few, many = plan & (calls <= max_calls), plan & (mins > min_minutes)
    rest = plan & ~few & ~many
    k = lambda m: [int(y[m].sum()), int(m.sum())]
    return {"n_train": int(len(y)), "max_intl_calls": max_calls, "more_than_intl_minutes": min_minutes,
            "plan_holders": k(plan), "plan_le2_intl_calls": k(few),
            "plan_gt13_intl_minutes": k(many), "plan_other": k(rest), "no_plan": k(~plan)}


JOBS = {"churn_blindspot.json": churn_blindspot, "agreement.json": agreement, "paper_net.json": paper_net, "s8_ice.npz": s8_ice,
        "churn_shapes.json": churn_shapes, "pairs_s2.json": pairs_s2}


def ensure(refresh: bool = False, only=None):
    DATA.mkdir(exist_ok=True)
    for name, job in JOBS.items():
        if only and name not in only:
            continue
        p = DATA / name
        if p.exists() and not refresh:
            continue
        print(f"[db, read-only] computing {name} ...", flush=True)
        res = job()
        if res is not None:
            p.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ensure(refresh="--refresh" in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")] or None)
