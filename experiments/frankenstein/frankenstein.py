"""Frankenstein union graft on churn100 (PLAN.md). Exploratory: future work, not in the thesis.

Every graftable, useful explained feature of the ten churn100 networks (the C2 donor pool,
experiments/churn/TRANSPLANT2_PLAN.md) is put into one model, by the C2 graft machinery
(experiments/churn/genome_surgery.py, ext_common.py):

- standalone: the 71 inputs, all 71 direct input -> output edges and every pooled part with its
  trained weights, each part's readout edges into one fresh output;
- union-r: churn100-s<r> plus every pooled part from the other nine seeds.

Arms per build: graft or sham (grafted weights and biases N(0, 1)) x {refit (L2 logistic refit of
every edge into the output and the output bias; C chosen on the 800-row validation split, then
fitted on the 4,000 train rows), refit + masked fine-tune E = 500 (train rows), refit on the
3,200 fit rows + masked fine-tune with early stopping (EARLY_STOPPING_PLAN.md)}. Plus, for the
standalone build only, a greedy deduplicated Frankenstein (refit only).

Stages (all reads are READ ONLY; nothing is written to the database):

    export DATABASE_URL=postgresql://localhost/explaneat_dev PYTHONHASHSEED=0
    uv run python experiments/frankenstein/frankenstein.py pool                  # train only
    uv run python experiments/frankenstein/frankenstein.py run --build standalone --dry-run
    uv run python experiments/frankenstein/frankenstein.py run --build standalone
    uv run python experiments/frankenstein/frankenstein.py run --build union
    uv run python experiments/frankenstein/frankenstein.py summary

``--out-dir`` writes somewhere else (used for the run-twice byte-identity check).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from neat.genes import DefaultConnectionGene
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "churn"))
sys.path.insert(0, str(REPO / "experiments" / "churn_signals"))

import ext_common as ec  # noqa: E402  (READ ONLY sessions; threadpool_limits(1))
import run_transplant as rt  # noqa: E402
from basis import NAMES, build_basis, project  # noqa: E402  (experiments/churn_signals/basis.py)
from ext_common import ee, gs  # noqa: E402

RESULTS = HERE / "results"
C2_SELECTION = REPO / "experiments" / "churn" / "transplant2_selection.csv"
SIGNALS_CSV = REPO / "experiments" / "churn_signals" / "signals.csv"
BENCHMARKS_CSV = REPO / "experiments" / "benchmarks" / "benchmarks.csv"
BASELINES_JSON = REPO / "experiments" / "references" / "results" / "baselines.json"

# design choices (PLAN.md)
MIN_GAIN = 0.02                    # the C2 pool's train restore-gain threshold
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]  # the LR reference grid
E_FIXED = 500
STANDALONE = 10                    # build id of the standalone Frankenstein (union builds: 0..9)
DEDUP_MIN_GAIN = 0.001             # greedy stop: validation AUC gain below this
NEWTON_MAX_ITER = 500


def build_name(b):
    return "standalone" if b == STANDALONE else f"union-r{b}"


# ============================================================================ pool

def load_pool(runs, data):
    """The C2 donor pool: churn100 leaves that qualify for a graft (all-input entries, the
    three collapse preconditions, a readout edge to the output), with train restore gain
    >= 0.02 and no input -> output edge (C2 'graftable'). Sorted by (seed, name); a feature's
    index k is its position."""
    cand, cache = rt.candidates(runs, data)
    feats, excluded = [], []
    for c in sorted(cand, key=lambda c: (c["seed"], c["name"])):
        if not c["qualifies"] or c["train_restore_gain"] < MIN_GAIN:
            continue
        dg, dconf, dinfo = cache[c["seed"]]
        sub = gs.annotation_to_genome_subgraph(dinfo["operations"], c["name"], dg, dconf)
        if sub.input_output_connections:
            excluded.append(dict(c, reason="input -> output edge (not graftable in C2)"))
            continue
        feats.append(dict(c, sub=sub, k=len(feats)))
    # the same set as C2's graftable donors
    c2 = set()
    with open(C2_SELECTION) as f:
        for r in csv.DictReader(f):
            if r["kind"] == "candidate_pair" and r["graftable"] == "True":
                c2.add((int(float(r["donor_seed"])), r["donor_annotation"]))
    mine = {(f["seed"], f["name"]) for f in feats}
    if mine != c2:
        raise SystemExit(f"pool differs from C2's graftable donors: {sorted(mine ^ c2)}")
    return feats, excluded, cache


def basis_for(data):
    """The churn signal catalogue's data basis (experiments/churn_signals/basis.py), built
    from the train rows; raw values are the z-scores inverted with the split's scaler."""
    mu, sc = data["scaler_mean"], data["scaler_scale"]
    m = SimpleNamespace(feature_names=list(data["feature_names"]), X=data["X_train"],
                        Xt=data["X_test"], raw_train=data["X_train"] * sc + mu,
                        raw_test=data["X_test"] * sc + mu)
    return build_basis(m)


def themes():
    with open(SIGNALS_CSV) as f:
        return {r["signal"]: r["theme"] for r in csv.DictReader(f)}


def family_of(s, B, theme):
    """Signal family of a donor signal s (train): project s on the catalogue basis; each
    block's share = var(component) / var(s). The family is the non-linear block (shape:* or
    int:*, i.e. what the part adds beyond straight lines) with the largest share; also
    returned: the linear share, the theme with the largest summed share, the top blocks and
    the unassigned (residual) share."""
    icpt, beta = project(B, s)
    v = float(np.var(s))
    shares = {b: float(np.var(B.component(beta, b))) / v for b in B.blocks}
    resid = float(np.var(s - icpt - B.Phi_tr @ beta)) / v
    nonlin = sorted((b for b in shares if not b.startswith("lin:")),
                    key=lambda b: (-shares[b], b))
    by_theme = {}
    for b, x in shares.items():
        by_theme[theme[b]] = by_theme.get(theme[b], 0.0) + x
    th = max(sorted(by_theme), key=lambda t: by_theme[t])
    top = sorted(shares.items(), key=lambda t: (-t[1], t[0]))[:4]
    return {"family": nonlin[0], "family_name": NAMES.get(nonlin[0], nonlin[0]),
            "family_share": shares[nonlin[0]],
            "second_nonlinear": f"{nonlin[1]} {shares[nonlin[1]]:.3f}",
            "linear_share": sum(x for b, x in shares.items() if b.startswith("lin:")),
            "theme": th, "theme_share": by_theme[th],
            "top_blocks": "; ".join(f"{b} {x:.3f}" for b, x in top), "unassigned_share": resid}


def donor_signal(f, cache, X):
    dg, dconf, _ = cache[f["seed"]]
    s, _, _ = gs.readout_contribution(dg, dconf, X, f["sub"].readout_connections)
    return s


def pool_rows(feats, excluded, cache, data):
    B = basis_for(data)
    th = themes()
    hyp = {s: {a["name"]: (a.get("hypothesis") or "")
               for a in ee.annotations_from_ops(cache[s][2]["operations"])} for s in ec.SEEDS}
    rows = []
    for f in feats:
        s = donor_signal(f, cache, data["X_train"])
        fam = family_of(s, B, th)
        sub = f["sub"]
        rows.append({"k": f["k"], "feature": f"s{f['seed']}:{f['name']}", "donor_seed": f["seed"],
                     "donor_genome_id": f["genome_id"], "annotation": f["name"],
                     **fam,
                     "train_restore_gain": f["train_restore_gain"], "nonlinear": f["nonlinear"],
                     "entry_base_features": ";".join(sorted(f["entry_base_features"])),
                     "n_hidden": len(sub.hidden_nodes),
                     "n_connections": len(sub.all_connections),
                     "n_readout": len(sub.readout_connections),
                     "donor_signal_sd_train": float(np.std(s)),
                     "hypothesis": hyp[f["seed"]][f["name"]]})
    for c in excluded:
        rows.append({"k": "", "feature": f"s{c['seed']}:{c['name']}", "donor_seed": c["seed"],
                     "donor_genome_id": c["genome_id"], "annotation": c["name"],
                     "train_restore_gain": c["train_restore_gain"], "excluded": c["reason"]})
    return rows


POOL_COLS = ["k", "feature", "donor_seed", "donor_genome_id", "annotation", "family",
             "family_name", "family_share", "second_nonlinear", "linear_share", "theme",
             "theme_share", "top_blocks", "unassigned_share", "train_restore_gain", "nonlinear",
             "entry_base_features", "n_hidden", "n_connections", "n_readout",
             "donor_signal_sd_train", "excluded", "hypothesis"]


def read_pool(out_dir):
    with open(out_dir / "pool.csv") as f:
        return {int(r["k"]): r for r in csv.DictReader(f) if r["k"] != ""}


# ============================================================================ builds

def empty_genome(cache):
    """A genome with only the output node (bias 0) and all 71 input -> output edges (weight
    0; the refit sets them). Copied from churn100-s0 for its output node gene."""
    g0, conf, _ = cache[0]
    g = copy.deepcopy(g0)
    outs = set(conf.genome_config.output_keys)
    for n in list(g.nodes):
        if n not in outs:
            del g.nodes[n]
    g.connections.clear()
    for o in sorted(outs):
        g.nodes[o].bias = 0.0
        for i in conf.genome_config.input_keys:
            cg = DefaultConnectionGene((i, o))
            cg.weight, cg.enabled = 0.0, True
            g.connections[(i, o)] = cg
    return g, conf


def parts_for(b, feats):
    return feats if b == STANDALONE else [f for f in feats if f["seed"] != b]


def build(b, feats, cache, sham=False, only=None):
    """Graft the build's parts (pool order) into its base genome. Sham: the k-th part's
    weights and biases are N(0, 1) with seed 100 * b + k. Returns (genome, config,
    [(feature, graft record)])."""
    if b == STANDALONE:
        g, conf = empty_genome(cache)
    else:
        g, conf, _ = cache[b]
    recs = []
    for f in parts_for(b, feats):
        if only is not None and f["k"] not in only:
            continue
        g, rec = gs.graft(g, conf, cache[f["seed"]][0], f["sub"],
                          sham_seed=(100 * b + f["k"]) if sham else None)
        recs.append((f, rec))
    return g, conf, recs


def check_preserved(b, g, conf, recs, cache, X):
    """Non-sham graft: logit = base logit + the sum of the donor signals."""
    if b == STANDALONE:
        z0 = np.zeros(len(X))
    else:
        z0 = gs.output_preactivation_terms(cache[b][0], conf, X)["z"]
    s = sum(donor_signal(f, cache, X) for f, _ in recs)
    err = float(np.max(np.abs(gs.output_preactivation_terms(g, conf, X)["z"] - (z0 + s))))
    if err > 1e-8:
        raise SystemExit(f"{build_name(b)}: grafted logit != base logit + donor signals ({err})")
    return err


# ============================================================================ refit

def readout(g, conf, X):
    """(keys, A, b, w): every connection into output 0 in NeuralNeat's output layer, its
    source activations A on X, the output bias and the weights; checks z == A w + b."""
    t = gs.output_preactivation_terms(g, conf, X)
    imap = t["net"].layers[t["layer"]]["input_map"]
    keys = sorted(k for k, (r, c) in imap.items() if c == t["col"])
    rows = [imap[k][0] for k in keys]
    A = t["inputs"][:, rows]
    w = t["w"][rows]
    err = float(np.max(np.abs(t["z"] - (A @ w + t["b"]))))
    if err > 1e-9:
        raise SystemExit(f"output pre-activation has terms outside the readout ({err})")
    return keys, A, t["b"], w


def fit_lr(A, y, C):
    """L2 logistic regression on the standardised columns of A (sklearn's C: mean BCE +
    ||w||^2 / (2 C n)); intercept unpenalised; constant columns get weight 0. Damped Newton
    from zero (genome_surgery._logistic_newton). Returns (w, b, converged, n_iter) on the raw
    columns."""
    y = np.asarray(y, dtype=np.float64).ravel()
    mu, sd = A.mean(axis=0), A.std(axis=0)
    keep = sd > 1e-12
    Z = (A[:, keep] - mu[keep]) / sd[keep]
    F = np.column_stack([np.ones(len(y)), Z])
    pen = [0.0] + [1.0] * Z.shape[1]
    theta, n_iter, conv = gs._logistic_newton(np.zeros(len(y)), F, y, np.zeros(F.shape[1]),
                                              1.0 / (C * len(y)), pen, NEWTON_MAX_ITER)
    w = np.zeros(A.shape[1])
    w[keep] = theta[1:] / sd[keep]
    b = float(theta[0] - np.sum(theta[1:] * mu[keep] / sd[keep]))
    return w, b, bool(conv), int(n_iter)


def bce(z, y):
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def choose_C(A_fit, y_fit, A_val, y_val):
    """C with the lowest validation BCE (fit rows -> validation rows); ties: smaller C."""
    best = None
    for C in C_GRID:
        w, b, conv, _ = fit_lr(A_fit, y_fit, C)
        zv = A_val @ w + b
        v = bce(zv, y_val)
        if best is None or v < best[1]:
            best = (C, v, float(roc_auc_score(y_val, zv)), conv)
    return best


def refit(g, conf, X, y, C):
    """The genome with every edge into the output and the output bias refit (fit_lr, C)."""
    keys, A, _, _ = readout(g, conf, X)
    w, b, conv, n_iter = fit_lr(A, y, C)
    g2 = copy.deepcopy(g)
    g2.nodes[0].bias = b
    for k, wk in zip(keys, w):
        g2.connections[k].weight = float(wk)
    p = gs.predict_proba(g2, conf, X)
    err = float(np.max(np.abs(p - 1.0 / (1.0 + np.exp(-(A @ w + b))))))
    if err > 1e-9:
        raise SystemExit(f"refit write-back differs from the fit ({err})")
    return g2, {"C": C, "converged": conv, "n_iter": n_iter, "n_readout": len(keys),
                "train_bce_fit": bce(A @ w + b, np.asarray(y, dtype=np.float64).ravel())}


# ============================================================================ contributions

def part_contributions(g, conf, X, recs):
    """{k: the part's logit contribution on X in g}, plus the remaining logit (everything else
    into the output, without the bias)."""
    keys, A, _, w = readout(g, conf, X)
    col = {k: i for i, k in enumerate(keys)}
    out, used = {}, set()
    for f, rec in recs:
        ks = [tuple(k) for k in rec["readout_connections"]]
        idx = [col[k] for k in ks if k in col]
        used.update(idx)
        out[f["k"]] = A[:, idx] @ w[idx] if idx else np.zeros(len(X))
    rest = [i for i in range(len(keys)) if i not in used]
    other = A[:, rest] @ w[rest] if rest else np.zeros(len(X))
    return out, other


# ============================================================================ run

def es_ft(g, conf, Xf, yf, Xv, yv, seed):
    return gs.finetune(g, conf, Xf, yf, seed=seed, **ec.es_kwargs(Xv, yv))


def fixed_ft(g, conf, X, y, seed):
    return gs.finetune(g, conf, X, y, E_FIXED, seed, mask_phantom=True, return_info=True)


def comparators(cache, runs, data, boot, conf):
    """Base seeds, the best-by-train seed and the 10-seed logit-mean ensemble (test
    predictions for paired deltas), with sizes."""
    rows, p, z_tr, z_te = [], {}, [], []
    for s in ec.SEEDS:
        g = cache[s][0]
        m, p[f"s{s}"] = ec.evaluate(g, conf, data, boot)
        z_tr.append(gs.output_preactivation_terms(g, conf, data["X_train"])["z"])
        z_te.append(gs.output_preactivation_terms(g, conf, data["X_test"])["z"])
        rows.append({"model": f"churn100-s{s}", "kind": "base seed", **m})
    best = max(ec.SEEDS, key=lambda s: (rows[s]["train_auc"], -s))
    for r in rows:
        r["best_by_train"] = r["model"] == f"churn100-s{best}"
    zt, zte = np.mean(z_tr, axis=0), np.mean(z_te, axis=0)
    pe = 1.0 / (1.0 + np.exp(-zte))
    lo, hi, nb = ee.bootstrap_auc_ci(data["y_test"], pe, boot)
    rows.append({"model": "ensemble10", "kind": "mean of the ten logits",
                 "train_auc": roc_auc_score(data["y_train"], zt),
                 "test_auc": roc_auc_score(data["y_test"], pe), "test_auc_ci_lo": lo,
                 "test_auc_ci_hi": hi, "n_boot_used": nb,
                 "hidden_nodes": sum(r["hidden_nodes"] for r in rows[:10]),
                 "phenotype_connections_enabled": sum(r["phenotype_connections_enabled"]
                                                      for r in rows[:10])})
    p["ensemble10"] = pe
    p["best"] = p[f"s{best}"]
    return rows, p, best


def references():
    """LR and GB test AUCs, read from the reference result files."""
    rows = []
    with open(BENCHMARKS_CSV) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "churn" and r["model_key"] in ("lr_l2", "gb"):
                rows.append({"model": f"benchmarks:{r['model_key']}", "kind": r["model"],
                             "test_auc": float(r["test_value"]),
                             "test_auc_ci_lo": float(r["ci_lo"]),
                             "test_auc_ci_hi": float(r["ci_hi"]),
                             "train_auc": float(r["train_value"]),
                             "source": "experiments/benchmarks/benchmarks.csv"})
    ref = json.loads(BASELINES_JSON.read_text())["churn"]
    for key, name in (("logistic", "LR (L2)"), ("gradient_boosting", "HistGradientBoosting")):
        d = ref[key]
        rows.append({"model": f"references:{key}", "kind": name, "test_auc": d["test_auc"],
                     "test_auc_ci_lo": d["test_auc_ci95"][0],
                     "test_auc_ci_hi": d["test_auc_ci95"][1],
                     "train_auc": d.get("train_auc", ""),
                     "source": "experiments/references/results/baselines.json"})
    return rows


def dedup(feats, cache, data, val):
    """Greedy deduplicated standalone Frankenstein (validation only): start from the direct
    input edges; repeatedly add the part whose addition gives the largest validation AUC
    (each candidate set: C chosen on the validation split, fit on the fit rows); stop when the
    best gain is < DEDUP_MIN_GAIN. Returns (kept ks in order, step rows)."""
    Xf, yf, Xv, yv, vs = val
    g, conf, recs = build(STANDALONE, feats, cache)
    keys, A, _, _ = readout(g, conf, data["X_train"])
    col = {k: i for i, k in enumerate(keys)}
    part_cols, used = {}, set()
    for f, rec in recs:
        part_cols[f["k"]] = [col[tuple(k)] for k in rec["readout_connections"]]
        used.update(part_cols[f["k"]])
    base_cols = [i for i in range(len(keys)) if i not in used]
    fpos, vpos = vs["fit_train_positions"], vs["val_train_positions"]

    def score(ks):
        cols = base_cols + [c for k in ks for c in part_cols[k]]
        C, v, auc, _ = choose_C(A[fpos][:, cols], yf, A[vpos][:, cols], yv)
        return auc, C, v

    kept, steps = [], []
    cur, C0, v0 = score([])
    steps.append({"step": 0, "added": "", "val_auc": cur, "C": C0, "val_bce": v0, "gain": ""})
    while True:
        cands = [(score(kept + [f["k"]]), f["k"]) for f in feats if f["k"] not in kept]
        if not cands:
            break
        (auc, C, v), k = max(cands, key=lambda t: (t[0][0], -t[1]))
        gain = auc - cur
        steps.append({"step": len(kept) + 1, "added": k, "val_auc": auc, "C": C, "val_bce": v,
                      "gain": gain, "accepted": gain >= DEDUP_MIN_GAIN})
        if gain < DEDUP_MIN_GAIN:
            break
        kept.append(k)
        cur = auc
    return kept, steps


COND_COLS = ["build", "arm", "condition", "recipient_seed", "n_parts", "parts", "C",
             "val_bce_at_C", "val_auc_at_C", "refit_rows", "refit_converged", "refit_n_iter",
             "n_readout", "ft", "ft_seed", "ft_epochs", "ft_best_epoch", "ft_stopped_by",
             "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi", "n_boot_used",
             "train_acc_0.5", "test_acc_0.5", "hidden_nodes", "phenotype_connections_enabled",
             "genome_conn_genes_enabled", "inputs_connected",
             "d_vs_best_seed", "d_vs_best_seed_lo", "d_vs_best_seed_hi",
             "d_vs_ensemble", "d_vs_ensemble_lo", "d_vs_ensemble_hi",
             "d_vs_sham", "d_vs_sham_lo", "d_vs_sham_hi",
             "d_vs_base", "d_vs_base_lo", "d_vs_base_hi",
             "grafted_contrib_sd_train", "other_contrib_sd_train", "preservation_err"]
READ_COLS = ["build", "condition", "k", "feature", "family", "readout_weights",
             "donor_contrib_sd_train", "contrib_sd_train", "ratio_to_donor"]
DEDUP_COLS = ["step", "added", "feature", "family", "val_auc", "gain", "accepted", "C",
              "val_bce"]
COMP_COLS = ["model", "kind", "best_by_train", "train_auc", "test_auc", "test_auc_ci_lo",
             "test_auc_ci_hi", "n_boot_used", "hidden_nodes", "phenotype_connections_enabled",
             "source"]

CONDS = [  # (condition, arm, refit rows, fine-tune)
    ("graft-refit", "graft", "train", None),
    ("graft-refit-ft500", "graft", "train", "e500"),
    ("graft-refitfit-ftES", "graft", "fit", "es"),
    ("sham-refit", "sham", "train", None),
    ("sham-refit-ft500", "sham", "train", "e500"),
    ("sham-refitfit-ftES", "sham", "fit", "es"),
]


def run_build(b, feats, cache, data, val, boot, p_ref, dry, pool_info):
    """All six conditions of one build. Train-only work (grafts, C choice, refits,
    fine-tunes) is done for every condition before any test evaluation."""
    Xtr, ytr = data["X_train"], data["y_train"]
    Xf, yf, Xv, yv, vs = val
    fpos, vpos = vs["fit_train_positions"], vs["val_train_positions"]
    made, info = {}, {}
    for arm in ("graft", "sham"):
        g, conf, recs = build(b, feats, cache, sham=(arm == "sham"))
        err = check_preserved(b, g, conf, recs, cache, Xtr) if arm == "graft" else ""
        _, A, _, _ = readout(g, conf, Xtr)
        C, v, vauc, _ = choose_C(A[fpos], yf, A[vpos], yv)
        g_tr, ri_tr = refit(g, conf, Xtr, ytr, C)
        g_fit, ri_fit = refit(g, conf, Xf, yf, C)
        for cond, a, rows, ft in CONDS:
            if a != arm:
                continue
            start, ri = (g_tr, ri_tr) if rows == "train" else (g_fit, ri_fit)
            t0 = time.time()
            fi = None
            if ft == "e500":
                gg, fi = (fixed_ft(start, conf, Xtr, ytr, b) if not dry else
                          gs.finetune(start, conf, Xtr, ytr, 5, b, mask_phantom=True,
                                      return_info=True))
            elif ft == "es":
                gg, fi = (es_ft(start, conf, Xf, yf, Xv, yv, b) if not dry else
                          gs.finetune(start, conf, Xf, yf, 5, b, mask_phantom=True,
                                      return_info=True))
            else:
                gg = start
            dt = time.time() - t0
            made[cond] = (gg, conf, recs)
            info[cond] = {"C": C, "val_bce_at_C": v, "val_auc_at_C": vauc, "refit_rows": rows,
                          "refit_converged": ri["converged"], "refit_n_iter": ri["n_iter"],
                          "n_readout": ri["n_readout"], "ft": ft or "",
                          "ft_seed": b if ft else "",
                          "ft_epochs": fi["epochs"] if fi else "",
                          "ft_best_epoch": fi.get("best_epoch", "") if fi else "",
                          "ft_stopped_by": fi.get("stopped_by", "") if fi else "",
                          "preservation_err": err}
            print(f"   {build_name(b)} {cond}: C {C} val AUC {vauc:.4f} readout "
                  f"{ri['n_readout']} ft {ft or '-'} ({dt:.0f}s"
                  f"{'' if not fi else ', epochs ' + str(fi['epochs'])})", flush=True)
    if dry:
        return None
    # test evaluation
    rows, p, read = [], {}, []
    for cond, arm, _, _ in CONDS:
        gg, conf, recs = made[cond]
        m, p[cond] = ec.evaluate(gg, conf, data, boot)
        contrib, other = part_contributions(gg, conf, Xtr, recs)
        graft_sum = sum(contrib.values())
        rows.append({"build": build_name(b), "arm": arm, "condition": cond,
                     "recipient_seed": "" if b == STANDALONE else b, "n_parts": len(recs),
                     "parts": ";".join(str(f["k"]) for f, _ in recs), **info[cond], **m,
                     "grafted_contrib_sd_train": float(np.std(graft_sum)),
                     "other_contrib_sd_train": float(np.std(other))})
        for f, rec in recs:
            ks = [tuple(k) for k in rec["readout_connections"]]
            sd0 = float(pool_info[f["k"]]["donor_signal_sd_train"])
            sd = float(np.std(contrib[f["k"]]))
            read.append({"build": build_name(b), "condition": cond, "k": f["k"],
                         "feature": pool_info[f["k"]]["feature"],
                         "family": pool_info[f["k"]]["family"],
                         "readout_weights": ";".join(f"{gg.connections[k].weight:.6g}"
                                                     for k in ks),
                         "donor_contrib_sd_train": sd0, "contrib_sd_train": sd,
                         "ratio_to_donor": sd / sd0})
    for r in rows:
        cond = r["condition"]
        pairs = [("best_seed", p_ref["best"]), ("ensemble", p_ref["ensemble10"])]
        if r["arm"] == "graft":
            pairs.append(("sham", p[cond.replace("graft", "sham")]))
        if b != STANDALONE:
            pairs.append(("base", p_ref[f"s{b}"]))
        for name, pr in pairs:
            d, lo, hi = ec.paired_delta(data, p[cond], pr, boot)
            r.update({f"d_vs_{name}": d, f"d_vs_{name}_lo": lo, f"d_vs_{name}_hi": hi})
    return rows, read


def write(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    ec.write_csv(path, rows, cols)


def main():
    ap = argparse.ArgumentParser(description="Frankenstein union graft on churn100 (PLAN.md)")
    ap.add_argument("stage", choices=["pool", "run", "summary"])
    ap.add_argument("--build", choices=["standalone", "union"])
    ap.add_argument("--dry-run", action="store_true",
                    help="run: one build, 5 fine-tune epochs, timings only; no test data read, "
                         "nothing written")
    ap.add_argument("--recipients", default="0,1,2,3,4,5,6,7,8,9",
                    help="union: recipient seeds (default all ten)")
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()
    if "DATABASE_URL" not in os.environ:
        raise SystemExit("set DATABASE_URL=postgresql://localhost/explaneat_dev (read only)")
    torch.set_num_threads(1)
    out = Path(args.out_dir)
    if args.stage == "summary":
        import summary
        summary.main(out)
        return

    from explaneat.db.base import db
    db.init_db()
    data = ec.load_data()
    runs = ec.read_runs()
    t0 = time.time()
    feats, excluded, cache = load_pool(runs, data)
    print(f"pool: {len(feats)} features from seeds "
          f"{sorted({f['seed'] for f in feats})} ({time.time() - t0:.0f}s)", flush=True)

    if args.stage == "pool":
        write(out / "pool.csv", pool_rows(feats, excluded, cache, data), POOL_COLS)
        return

    pool_info = read_pool(out)
    if [int(k) for k in sorted(pool_info)] != [f["k"] for f in feats] or any(
            pool_info[f["k"]]["feature"] != f"s{f['seed']}:{f['name']}" for f in feats):
        raise SystemExit("pool.csv does not match the pool; rerun the pool stage")
    val = ec.load_val_split(data)
    if args.build is None:
        raise SystemExit("run needs --build")
    builds = ([STANDALONE] if args.build == "standalone" else
              [int(s) for s in args.recipients.split(",")])
    if args.dry_run:
        run_build(builds[0], feats, cache, data, val, None, None, True, pool_info)
        print("DRY RUN: nothing evaluated on test, nothing written")
        return

    boot = ec.bootstrap_indices(len(data["y_test"]))
    conf = cache[0][1]
    comp, p_ref, best = comparators(cache, runs, data, boot, conf)
    rows, read = [], []
    for b in builds:
        t1 = time.time()
        r, rd = run_build(b, feats, cache, data, val, boot, p_ref, False, pool_info)
        rows += r
        read += rd
        print(f"{build_name(b)} done ({time.time() - t1:.0f}s)", flush=True)
        for x in r:
            print(f"   {x['condition']:20s} test {x['test_auc']:.4f} d_best "
                  f"{x['d_vs_best_seed']:+.4f} d_ens {x['d_vs_ensemble']:+.4f}", flush=True)
    tag = args.build
    if args.build == "standalone":
        kept, steps = dedup(feats, cache, data, val)
        for s in steps:
            if s["added"] != "":
                s["feature"] = pool_info[s["added"]]["feature"]
                s["family"] = pool_info[s["added"]]["family"]
        write(out / "dedup_steps.csv", steps, DEDUP_COLS)
        # the kept parts as one genome: same refit procedure (C on validation, fit on train)
        Xf, yf, Xv, yv, vs = val
        g, conf_, recs = build(STANDALONE, feats, cache, only=set(kept))
        _, A, _, _ = readout(g, conf_, data["X_train"])
        C, v, vauc, _ = choose_C(A[vs["fit_train_positions"]], yf,
                                 A[vs["val_train_positions"]], yv)
        gd, ri = refit(g, conf_, data["X_train"], data["y_train"], C)
        m, pd_ = ec.evaluate(gd, conf_, data, boot)
        row = {"build": "standalone-dedup", "arm": "graft", "condition": "dedup-refit",
               "n_parts": len(recs), "parts": ";".join(str(k) for k in kept), "C": C,
               "val_bce_at_C": v, "val_auc_at_C": vauc, "refit_rows": "train",
               "refit_converged": ri["converged"], "refit_n_iter": ri["n_iter"],
               "n_readout": ri["n_readout"], **m}
        for name, pr in (("best_seed", p_ref["best"]), ("ensemble", p_ref["ensemble10"])):
            d, lo, hi = ec.paired_delta(data, pd_, pr, boot)
            row.update({f"d_vs_{name}": d, f"d_vs_{name}_lo": lo, f"d_vs_{name}_hi": hi})
        contrib, other = part_contributions(gd, conf_, data["X_train"], recs)
        row["grafted_contrib_sd_train"] = float(np.std(sum(contrib.values())))
        row["other_contrib_sd_train"] = float(np.std(other))
        rows.append(row)
        print(f"dedup: kept {len(kept)} parts {kept}: test {m['test_auc']:.4f}")
        write(out / "comparators.csv", comp + references(), COMP_COLS)
    write(out / f"conditions_{tag}.csv", rows, COND_COLS)
    write(out / f"readout_{tag}.csv", read, READ_COLS)


if __name__ == "__main__":
    main()
