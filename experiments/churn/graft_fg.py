"""Shared code for studies F and G (GRAFT_F_G_PLAN.md).

- dataset contexts (churn split + val_split.json; MONK-2 split + monk2_val_split.json);
- read-only genome loading for any stored genome (no churn-split assertion);
- the best genome of a given generation of an experiment;
- the C2 candidate rule for any dataset (all-input entries, graft preconditions,
  a readout edge, train restore gain);
- graft cases: graft / sham / refits per regime / fine-tunes with an optional
  protection phase (genome_surgery.finetune ``protect_*``);
- graft survival;
- statistics: sign test, bootstrap CIs of a median and of Spearman rho
  (np.random.RandomState(42), 10,000 resamples), cluster bootstrap, partial
  Spearman.

Library code: read-only DB access only.
"""

from __future__ import annotations

import csv
import json
import subprocess
import uuid
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

import ext_common as ec
import run_transplant as rt
from ext_common import ee, gs

HERE = ec.HERE
N_BOOT = 10_000
BOOT_SEED = 42
EQUIV_MARGIN = 0.005
MIN_GAIN = 0.02
REGIMES = ("es", "e500")
E_FIXED = 500

MONK2_SPLIT = "63257a45-49c9-42a6-8b28-8e346eab2bee"
MONK2_VAL_JSON = HERE / "monk2_val_split.json"


# ====================================================================== data

def load_split_any(split_id):
    from compute_runs import load_split
    with ec.ReadOnlySession() as s:
        data = load_split(s, split_id)
        from explaneat.db.models import DatasetSplit
        split = s.get(DatasetSplit, uuid.UUID(split_id))
        data["dataset_id"] = str(split.dataset_id)
    data["split_id"] = split_id
    return data


def val_split_rule(y_train):
    """The EARLY_STOPPING_PLAN.md rule on any train split: stratified 20%,
    random_state 42, over train positions; both lists sorted."""
    from sklearn.model_selection import train_test_split
    y = np.asarray(y_train).ravel()
    pos = np.arange(len(y))
    f, v = train_test_split(pos, test_size=ec.VAL_FRACTION, stratify=y,
                            random_state=ec.VAL_RANDOM_STATE)
    return sorted(int(i) for i in f), sorted(int(i) for i in v)


def monk2_val_split(data, write=False):
    """(X_fit, y_fit, X_val, y_val) for MONK-2; the JSON is written once
    (``write``) and must match the rule afterwards."""
    f, v = val_split_rule(data["y_train"])
    if write and not MONK2_VAL_JSON.exists():
        MONK2_VAL_JSON.write_text(json.dumps({
            "plan": "experiments/churn/GRAFT_F_G_PLAN.md (F2)", "split_id": MONK2_SPLIT,
            "rule": "train_test_split(np.arange(n_train), test_size=0.2, stratify=y_train, "
                    "random_state=42); positions index the MONK-2 train split rows",
            "n_train": len(data["y_train"]), "n_fit": len(f), "n_val": len(v),
            "fit_train_positions": f, "val_train_positions": v}, indent=1))
    if MONK2_VAL_JSON.exists():
        js = json.loads(MONK2_VAL_JSON.read_text())
        if js["fit_train_positions"] != f or js["val_train_positions"] != v:
            raise SystemExit("monk2_val_split.json does not match the rule")
    y = np.asarray(data["y_train"]).ravel()
    return data["X_train"][f], y[f], data["X_train"][v], y[v]


def regime_data(data, regime, val=None):
    """dict with the refit rows and fine-tune settings of a regime."""
    if regime == "e500":
        return {"regime": "e500", "X_refit": data["X_train"], "y_refit": data["y_train"]}
    Xf, yf, Xv, yv = val
    return {"regime": "es", "X_refit": Xf, "y_refit": yf, "X_fit": Xf, "y_fit": yf,
            "X_val": Xv, "y_val": yv}


# ====================================================================== genomes

def load_genome_any(genome_id, with_ops=True):
    """(genome, config, info) for any stored genome (read only)."""
    from explaneat.core.config_utils import load_neat_config
    from explaneat.db.models import Explanation, Genome
    with ec.ReadOnlySession() as s:
        row = s.get(Genome, uuid.UUID(genome_id))
        exp = row.population.experiment
        config = load_neat_config(exp.neat_config_text or "", exp.config_json)
        genome = row.to_neat_genome(config)
        ops = []
        if with_ops:
            expl = s.query(Explanation).filter(Explanation.genome_id == row.id).all()
            if len(expl) > 1:
                raise ValueError(f"genome {genome_id}: {len(expl)} explanations")
            ops = list(expl[0].operations) if expl else []
        info = {"genome_id": genome_id, "genome_key": row.genome_id,
                "stored_fitness": row.fitness, "generation": row.population.generation,
                "experiment_id": str(exp.id), "experiment_name": exp.name,
                "config_json": dict(exp.config_json), "split_id": str(exp.split_id),
                "operations": ops}
    return genome, config, info


def best_of_generation(experiment_name, generation):
    """(genome uuid, key, fitness) of the best stored genome of one generation:
    highest stored fitness (NaN / NULL ignored), ties -> lowest NEAT key."""
    from sqlalchemy import text
    with ec.ReadOnlySession() as s:
        rows = s.execute(text("""
            SELECT g.id, g.genome_id, g.fitness FROM genomes g
            JOIN populations p ON g.population_id = p.id
            JOIN experiments e ON p.experiment_id = e.id
            WHERE e.name = :n AND p.generation = :g AND g.fitness IS NOT NULL
              AND g.fitness <> 'NaN'::float8
            ORDER BY g.fitness DESC, g.genome_id ASC"""),
            {"n": experiment_name, "g": int(generation)}).fetchall()
        n_exp = s.execute(text("SELECT count(*) FROM experiments WHERE name = :n"),
                          {"n": experiment_name}).scalar()
    if n_exp != 1:
        raise SystemExit(f"{n_exp} experiments named {experiment_name!r}")
    if not rows:
        raise SystemExit(f"{experiment_name}: no genomes in generation {generation}")
    top = rows[0]
    n_tied = sum(1 for r in rows if r[2] == top[2])
    return str(top[0]), int(top[1]), float(top[2]), n_tied


def n_hidden(genome, config):
    special = set(config.genome_config.output_keys)
    return sum(1 for k in genome.nodes if k not in special)


# ====================================================================== candidates

def candidate_leaves(seed_genomes, data, min_gain=MIN_GAIN):
    """The C2 candidate rule for any dataset. ``seed_genomes``: {seed: (genome,
    config, info)} with explanation operations. Returns (rows for every leaf,
    pool = qualifying leaves with train restore gain >= min_gain, each with
    subgraph and train contribution)."""
    rows, pool = [], []
    Xtr, ytr = data["X_train"], data["y_train"]
    for seed in sorted(seed_genomes):
        genome, config, info = seed_genomes[seed]
        if not info["operations"]:
            raise SystemExit(f"seed {seed}: genome {info['genome_id']} has no explanation")
        state, anns, ev, _ = ec.explained_state(genome, config, info["operations"], Xtr)
        _, replayed = gs.replay_model_state(info["operations"], genome, config)
        resolver = gs.ModelStateIdResolver(replayed)
        leaves = ee.leaf_annotations(anns)
        lin = ee.Linearisation(ev, {a["name"]: a for a in leaves}, Xtr)
        perf = ee.performance_sources(lin, Xtr, ytr)
        hyp = {a["name"]: (a.get("hypothesis") or "") for a in anns}
        for a in leaves:
            u = perf["units"][a["name"]]
            row = {"seed": seed, "genome_id": info["genome_id"], "name": a["name"],
                   "hypothesis": hyp.get(a["name"], ""),
                   "train_restore_gain": u["restore_gain"], "nonlinear": u["nonlinear"],
                   "entry_base_features": gs.entry_base_features(
                       a["entry_nodes"], resolver, data["feature_names"],
                       config.genome_config.input_keys)}
            reasons, sub = [], None
            try:
                sub = gs.annotation_to_genome_subgraph(info["operations"], a["name"],
                                                       genome, config)
                row.update(entries_all_inputs=True, n_readout=len(sub.readout_connections),
                           n_hidden=len(sub.hidden_nodes),
                           n_input_output=len(sub.input_output_connections))
                if not sub.readout_connections:
                    reasons.append("no readout edge to output 0")
            except gs.NonInputEntryError:
                row["entries_all_inputs"] = False
                reasons.append("entries not all inputs")
            ok = rt.preconditions_ok(state, a)
            row["preconditions_ok"] = ok
            if not ok:
                reasons.append("collapse preconditions fail")
            if not u["restore_gain"] >= min_gain:
                reasons.append(f"train restore gain < {min_gain}")
            row["qualifies"] = not reasons
            row["reason"] = "; ".join(reasons)
            rows.append(row)
            if row["qualifies"]:
                s, F, t0 = gs.readout_contribution(genome, config, Xtr, sub.readout_connections,
                                                   sub.input_output_connections)
                pool.append(dict(row, sub=sub, s=s, F=F, theta0=t0, genome=genome,
                                 config=config, info=info))
    return rows, pool


def churn_pool(data):
    """The C2 / D2 pool (22 leaves) as dicts with genome, config, info, sub, s,
    name, seed, train_restore_gain (run_rescue2.load_pool, which stops on
    any other count)."""
    import run_rescue2 as r2
    leaves = r2.load_pool(ec.read_runs(), data)
    for L in leaves:
        s, F, t0 = gs.readout_contribution(L["genome"], L["config"], data["X_train"],
                                           L["sub"].readout_connections,
                                           L["sub"].input_output_connections)
        assert np.array_equal(s, L["s"])
        L.update(F=F, theta0=t0, genome_id=L["info"]["genome_id"])
    return leaves


# ====================================================================== pair metrics

def recipient_terms(genome, config, X, y):
    B, labels = gs.signal_basis(genome, config, X)
    t = gs.output_preactivation_terms(genome, config, X)
    return {"B": B, "z": t["z"], "b": t["b"], "auc": float(roc_auc_score(y, t["z"])),
            "n_hidden_basis": sum(lab.startswith("h:") for lab in labels),
            "n_cols": B.shape[1]}


def pair_metrics(R, rgenome, rconfig, donor, y):
    """nov, inc and graftable of ``donor`` for a recipient (train only)."""
    nv = gs.linear_novelty(donor["s"], R["B"])
    z, ri = gs.refit_added_signal(R["z"], R["b"], donor["F"], donor["theta0"], y)
    auc = float(roc_auc_score(y, z))
    col = gs.graft_collisions(rgenome, rconfig, donor["sub"])
    return {"nov": nv["nov"], "r2": nv["r2"], "basis_rank": nv["rank"],
            "basis_n_cols_used": nv["n_cols_used"], "nov_reason": nv["reason"],
            "inc": auc - R["auc"], "refit_train_auc": auc, "refit_converged": ri["converged"],
            "graftable": not col,
            "collisions": ";".join(str(tuple(c["key"])) for c in col)}


def pick_arms(pairs, rng_key):
    """F1 arms for one recipient. ``pairs``: dicts with donor_seed,
    donor_annotation, inc, graftable. Returns {"maxinc": pair or None,
    "random": pair or None}."""
    ok = sorted([p for p in pairs if p["graftable"] and np.isfinite(p["inc"])],
                key=lambda p: (p["donor_seed"], p["donor_annotation"]))
    if not ok:
        return {"maxinc": None, "random": None}
    best = sorted(ok, key=lambda p: (-p["inc"], p["donor_seed"], p["donor_annotation"]))[0]
    rest = [p for p in ok if p is not best]
    rnd = None
    if rest:
        rng = np.random.default_rng(list(rng_key))
        rnd = rest[int(rng.choice(len(rest)))]
    return {"maxinc": best, "random": rnd}


# ====================================================================== grafts

def io_keys_of(rec, config):
    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)
    return [tuple(k) for k in rec["added_connections"] if k[0] in inputs and k[1] in outputs]


def make_grafts(rgenome, rconfig, donor, sham_seed, X_check):
    """Graft and sham of ``donor`` into a copy of the recipient, with the
    logit-additivity check (grafted logit = recipient logit + contribution)."""
    gg, rec = gs.graft(rgenome, rconfig, donor["genome"], donor["sub"])
    sham, srec = gs.graft(rgenome, rconfig, donor["genome"], donor["sub"], sham_seed=sham_seed)
    z_r = gs.output_preactivation_terms(rgenome, rconfig, X_check)["z"]
    z_g = gs.output_preactivation_terms(gg, rconfig, X_check)["z"]
    err = float(np.max(np.abs(z_g - (z_r + donor["s"]))))
    if err > 1e-8:
        raise SystemExit(f"grafted logit != recipient logit + donor signal ({err:.3g})")
    return {"graft": (gg, rec), "sham": (sham, srec), "logit_err": err}


def refit(genome, rec, config, X, y):
    return gs.refit_readout(genome, config, X, y, [tuple(k) for k in rec["readout_connections"]],
                            scaled_groups=[io_keys_of(rec, config)] if io_keys_of(rec, config)
                            else ())


def finetune(start, config, rd, seed, rec=None, P=0):
    """One fine-tune in regime ``rd`` (regime_data), optionally with a
    protection phase of P epochs on ``rec``'s graft. Returns (genome, info)."""
    kw = {}
    if P:
        conns, nodes = gs.graft_protect_keys(rec)
        kw = dict(protect_conn_keys=conns, protect_node_keys=nodes, protect_epochs=P)
    if rd["regime"] == "es":
        return gs.finetune(start, config, rd["X_fit"], rd["y_fit"], seed=seed,
                           **ec.es_kwargs(rd["X_val"], rd["y_val"]), **kw)
    return gs.finetune(start, config, rd["X_refit"], rd["y_refit"], E_FIXED, seed,
                       mask_phantom=True, return_info=True, **kw)


def contribution(genome, config, rec, X):
    """The grafted feature's logit contribution in ``genome`` (its readout and
    input -> output edges as added by the graft)."""
    s, _, _ = gs.readout_contribution(genome, config, X,
                                      [tuple(k) for k in rec["readout_connections"]],
                                      io_keys_of(rec, config))
    return s


def survival_r(genome, config, rec, X, reference):
    return gs.pearson_r(contribution(genome, config, rec, X), reference)


def ft_fields(prefix, info, P=0):
    if not info:
        return {}
    out = {f"{prefix}_epochs": info["epochs"]}
    if info.get("early_stopping"):
        out.update({f"{prefix}_best_epoch": info["best_epoch"],
                    f"{prefix}_stop_epoch": info["stop_epoch"],
                    f"{prefix}_stopped_by": info["stopped_by"],
                    f"{prefix}_best_in_phase": bool(P) and info["best_epoch"] <= P})
    return out


# ====================================================================== statistics

def boot_matrix(n, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    return np.stack([rng.randint(0, n, size=n) for _ in range(n_boot)])


def sign_test(vals):
    d = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    npos, nneg = int((d > 0).sum()), int((d < 0).sum())
    p = binomtest(npos, npos + nneg, 0.5).pvalue if npos + nneg else float("nan")
    return {"n": len(d), "n_positive": npos, "n_negative": nneg,
            "n_zero": len(d) - npos - nneg,
            "median": float(np.median(d)) if len(d) else float("nan"),
            "sign_test_p": float(p)}


def median_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    d = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(d) < 2:
        return float("nan"), float("nan")
    idx = boot_matrix(len(d), n_boot, seed)
    meds = np.median(d[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def spearman_ci(x, y, clusters=None, n_boot=N_BOOT, seed=BOOT_SEED):
    """Spearman rho of (x, y) with a percentile bootstrap CI over cases, or over
    clusters (resampling whole clusters) when ``clusters`` is given."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    cl = None if clusters is None else np.asarray(clusters)[ok]
    if len(x) < 3:
        return {"rho": float("nan"), "p": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": int(len(x))}
    rho, p = spearmanr(x, y)
    rhos = []
    if cl is None:
        for idx in boot_matrix(len(x), n_boot, seed):
            r = spearmanr(x[idx], y[idx])[0]
            if np.isfinite(r):
                rhos.append(r)
    else:
        groups = sorted(set(cl.tolist()))
        members = [np.where(cl == g)[0] for g in groups]
        for gi in boot_matrix(len(groups), n_boot, seed):
            idx = np.concatenate([members[i] for i in gi])
            r = spearmanr(x[idx], y[idx])[0]
            if np.isfinite(r):
                rhos.append(r)
    lo, hi = np.percentile(rhos, [2.5, 97.5]) if rhos else (float("nan"), float("nan"))
    return {"rho": float(rho), "p": float(p), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(x))}


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y given z: Pearson r of the
    residuals of rank(x) and rank(y) after least squares on rank(z)."""
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    A = np.column_stack([np.ones(len(rz)), rz])

    def res(v):
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ c
    return gs.pearson_r(res(rx), res(ry))


def about_zero(lo, hi, margin=EQUIV_MARGIN):
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo >= -margin and hi <= margin)


# ====================================================================== csv / git

def fmt(v):
    if isinstance(v, frozenset):
        return ";".join(sorted(v))
    if isinstance(v, (list, tuple)):
        return ";".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write_rows(path, rows, columns=None):
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in columns})
    print(f"wrote {path} ({len(rows)} rows)")


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def require_committed(path):
    rel = Path(path).resolve().relative_to(ec.REPO)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)], cwd=ec.REPO,
                             capture_output=True, text=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(rel)], cwd=ec.REPO,
                           capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        raise SystemExit(f"{rel} must be committed (and unmodified) before any test "
                         "evaluation")


def check_selection(new_rows, committed_rows, keys, float_keys):
    """Recomputed selection rows must equal the committed ones on ``keys``."""
    if len(new_rows) != len(committed_rows):
        raise SystemExit(f"selection: {len(new_rows)} rows recomputed, "
                         f"{len(committed_rows)} committed")
    for a, b in zip(new_rows, committed_rows):
        for k in keys:
            va, vb = fmt(a.get(k, "")), b.get(k, "")
            if k in float_keys and vb not in ("", "nan"):
                if not np.isclose(float(va), float(vb), rtol=1e-9, atol=1e-12):
                    raise SystemExit(f"selection mismatch {k}: {va} vs {vb}")
            elif str(va) != str(vb):
                raise SystemExit(f"selection mismatch {k}: {va!r} vs {vb!r}")


def require_clean_tree():
    commit = gs.code_commit()
    if commit["dirty"]:
        raise SystemExit("working tree has uncommitted changes in explaneat/ or "
                         "experiments/churn/; commit first so provenance is exact")
    return commit
