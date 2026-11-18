"""Churn signal-realisation catalogue for Experiment 2's ten churn100 networks (read-only).

Exploratory, not pre-registered (2026-11-18). Everything is fitted on the 4,000 training rows;
test numbers (1,000 rows) are computed once, reported in columns marked ``_test`` and never used
to select or tune anything. Read-only on explaneat_v4; no v4 record or layer is changed.

Terminology (Mike's):

- **signal**: a target-relevant function of the inputs, defined in data terms, independent of
  any model. Here a signal is a block of a fixed data basis ``Phi`` (``basis.py``): a linear
  term, a one-variable bend (shape) or a two-variable interaction, with aliases merged first.
- **realisation**: the subgraph + weights that compute it in one network. Measured two ways:
  (a) functionally, the network's component ``c_s = Phi_s beta_s`` of the least-squares
  projection of its logit on ``Phi`` (train); (b) structurally, the hidden units whose
  knock-out effect (unit fixed at its training mean) projects onto block ``s``.
- **alias**: a different realisation of the same signal (here also: an input column that is
  the same function of the data, e.g. day charge = 0.17 x day minutes). Not signal-processing
  aliasing.
- **decodable**: novelty = 1 - R^2 of the data's own version of the signal (the block's
  component in a ridge logistic fit of the labels on ``Phi``) on the network's inputs plus all
  hidden activations (``genome_surgery.signal_basis``) is < 0.5 (the species-leads residual
  match: non-linear blocks are already orthogonal to every linear column).
- **used**: the network's own component has SD >= 0.10 logit on train.

Outputs (next to this file): ``aliases.csv``, ``signals.csv``, ``units.csv`` (every hidden node
of every network, with its h-label and signal), ``realisations.csv`` (seed x signal x unit),
``matrix.csv`` (seed x signal), ``disagreement.csv``, ``harmonise.csv``,
``consolidation.csv``, ``part_shares.csv`` and ``cache/`` (curves for ``figures.py``).

    DATABASE_URL=postgresql://localhost/explaneat_v4 \\
    EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 \\
    uv run python experiments/churn_signals/catalogue.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (GROUPS, GROUP_OF, HERE, LABELS, SHORT, load_all, reference_assign,  # noqa: E402
                    upstream_groups)
from basis import NAMES, build_basis, relu  # noqa: E402
from fastboot import Boot, selfcheck  # noqa: E402
import genome_surgery as gs  # noqa: E402
from explaneat.analysis.explanation_tools import default_auc, input_column  # noqa: E402
from explaneat.analysis.plain_language_linearity import classify_variables  # noqa: E402
from explaneat.core.genome_network import NodeType  # noqa: E402
from explaneat.export.render_structure import hidden_labels  # noqa: E402

CACHE = HERE / "cache"
USED_SD = 0.10          # logit SD on train for "used by the readout"
MIN_NOV = 0.5
CARRY_SD, CARRY_REL = 0.05, 0.20   # a unit carries block s: component SD >= 0.05 and >= 20% of the top carrier
STATE_GATE_ROWS = 100   # a switching unit reading state and open on <= 100 of 4,000 rows: state-gated
TOP_K = 141             # riskiest test customers compared (= the number of test churners)

THEMES = {
    "service calls": ["lin:csc", "shape:csc"],
    "service calls x low day usage": ["int:csc*day"],
    "day usage": ["lin:day", "shape:day"],
    "day + evening bill": ["int:day*eve"],
    "day usage without voice mail": ["int:day*vmail"],
    "international plan": ["lin:intlplan", "int:intlplan*intl"],
    "voice mail": ["lin:vmail"],
    "evening usage": ["lin:eve", "shape:eve"],
    "night / intl usage, intl calls": ["lin:night", "shape:night", "lin:intl", "shape:intl",
                                       "lin:intlcalls", "shape:intlcalls"],
    "account, area, call counts": ["lin:acct", "shape:acct", "lin:area", "lin:daycalls",
                                   "shape:daycalls", "lin:evecalls", "shape:evecalls",
                                   "lin:nightcalls", "shape:nightcalls"],
    "minor interactions": ["int:day*night", "int:eve*csc", "int:csc*vmail", "int:csc*intlplan"],
    "state offsets": ["lin:state"],
}
THEME_OF = {b: t for t, bs in THEMES.items() for b in bs}
BLOCK_VARS = {  # data variables a block is a function of
    "lin:csc": {"service calls"}, "shape:csc": {"service calls"},
    "lin:day": {"day usage"}, "shape:day": {"day usage"},
    "lin:eve": {"evening usage"}, "shape:eve": {"evening usage"},
    "lin:night": {"night usage"}, "shape:night": {"night usage"},
    "lin:intl": {"international usage"}, "shape:intl": {"international usage"},
    "lin:intlcalls": {"international calls"}, "shape:intlcalls": {"international calls"},
    "lin:intlplan": {"international plan"}, "lin:vmail": {"voice mail"},
    "lin:acct": {"account length"}, "shape:acct": {"account length"}, "lin:area": {"area code"},
    "lin:state": {"state"}, "lin:daycalls": {"day calls"}, "shape:daycalls": {"day calls"},
    "lin:evecalls": {"evening calls"}, "shape:evecalls": {"evening calls"},
    "lin:nightcalls": {"night calls"}, "shape:nightcalls": {"night calls"},
    "int:csc*day": {"service calls", "day usage"}, "int:day*eve": {"day usage", "evening usage"},
    "int:day*vmail": {"day usage", "voice mail"}, "int:day*night": {"day usage", "night usage"},
    "int:eve*csc": {"evening usage", "service calls"}, "int:csc*vmail": {"service calls", "voice mail"},
    "int:csc*intlplan": {"service calls", "international plan"},
    "int:intlplan*intl": {"international plan", "international usage", "international calls"},
}

# Variables for the distributed-vs-consolidated analysis: (group, grid in raw units)
RESP_VARS = ["service calls", "day usage", "evening usage", "international plan", "voice mail"]


def kind_of(block):
    return block.split(":")[0]


# ============================================================================ data helpers

def col(m, name):
    return m.feature_names.index(name)


def assign_raw(m, values):
    """{column name: raw value or per-row array} -> {column index: z}."""
    return {col(m, k): (np.asarray(v, float) - m.mean[col(m, k)]) / m.scale[col(m, k)]
            for k, v in values.items()}


RATIO = {}


def var_assign(m, group, value):
    """Raw-unit setting of a data variable with its aliases moved together."""
    if group in ("day usage", "evening usage", "night usage", "international usage"):
        mins, chg = GROUPS[group]
        r = RATIO[group]
        return assign_raw(m, {mins: value, chg: r * np.asarray(value, float)})
    if group == "voice mail":
        msgs = np.asarray(value, float)
        return assign_raw(m, {"voice mail plan": (msgs > 0).astype(float), "number vmail messages": msgs})
    if group == "international plan":
        return assign_raw(m, {"international plan": value})
    if group == "service calls":
        return assign_raw(m, {"number customer service calls": value})
    raise KeyError(group)


def var_grid(m, group):
    raw = m.raw_train
    if group == "service calls":
        return np.arange(0, 10, dtype=float), 1.0
    if group == "international plan":
        return np.array([0.0, 1.0]), 0.0
    if group == "voice mail":
        msgs = raw[:, col(m, "number vmail messages")]
        return np.r_[0.0, np.arange(5, 51, 5, dtype=float)], 0.0
    mins = raw[:, col(m, GROUPS[group][0])]
    lo, hi = np.percentile(mins, [0.5, 99.5])
    return np.linspace(lo, hi, 41), float(np.median(mins))


def set_cols(X, assign):
    X = X.copy()
    for c, z in assign.items():
        X[:, c] = z
    return X


def nov(x, B):
    v = gs.linear_novelty(x, B)["nov"]
    return float(v) if np.isfinite(v) else float("nan")


def r2(y, f):
    v = np.var(y)
    return float(1 - np.var(y - f) / v) if v > 1e-15 else float("nan")


def top_overlap(a, b, k=TOP_K):
    ia, ib = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(ia & ib) / k


# ============================================================================ 1. aliases

def aliases(m):
    rows = []
    Z, Zt = m.X, m.Xt
    names = m.feature_names
    seen = set()
    for i in range(Z.shape[1]):
        for j in range(i + 1, Z.shape[1]):
            if Z[:, i].std() == 0 or Z[:, j].std() == 0:
                continue
            c = np.corrcoef(Z[:, i], Z[:, j])[0, 1]
            if abs(c) < 0.8:
                continue
            ct = np.corrcoef(Zt[:, i], Zt[:, j])[0, 1]
            a, b = names[i], names[j]
            ri, rj = m.raw_train[:, i], m.raw_train[:, j]
            nz = ri != 0
            ratio = rj[nz] / ri[nz]
            rows.append(dict(a=a, b=b, corr_train=c, corr_test=ct,
                             max_abs_z_diff_train=float(np.max(np.abs(Z[:, i] - Z[:, j]))),
                             max_abs_z_diff_test=float(np.max(np.abs(Zt[:, i] - Zt[:, j]))),
                             ratio_b_over_a_mean=float(ratio.mean()), ratio_sd=float(ratio.std()),
                             kind="exact multiple (rounded to cents)" if ratio.std() < 1e-3 else "near alias"))
            seen.add((a, b))
    vp = m.raw_train[:, col(m, "voice mail plan")]
    vm = m.raw_train[:, col(m, "number vmail messages")]
    note = dict(no_plan_with_messages=int(((vp == 0) & (vm > 0)).sum()),
                plan_with_zero_messages=int(((vp == 1) & (vm == 0)).sum()),
                plan_holders=int((vp == 1).sum()),
                messages_corr_among_plan_holders=float(np.corrcoef(vp[vp == 1] * 0 + vm[vp == 1],
                                                                   vm[vp == 1])[0, 1]) if (vp == 1).sum() > 2 else float("nan"))
    return pd.DataFrame(rows), note


# ============================================================================ 2. data GAM

def fit_logit(P, y, C):
    sd = P.std(axis=0)
    sd[sd == 0] = 1
    lr = LogisticRegression(C=C, max_iter=20000, tol=1e-8).fit(P / sd, y)
    return lr.intercept_[0], lr.coef_[0] / sd


def cv_score(P, y, C, folds):
    ll, auc = [], []
    for tr, va in folds:
        a, b = fit_logit(P[tr], y[tr], C)
        z = a + P[va] @ b
        p = 1 / (1 + np.exp(-z))
        ll.append(log_loss(y[va], p))
        auc.append(roc_auc_score(y[va], z))
    return float(np.mean(ll)), float(np.mean(auc))


def data_gam(B, y, yt, boot):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(B.Phi_tr, y))
    grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    scores = {C: cv_score(B.Phi_tr, y, C, folds) for C in grid}
    C = min(scores, key=lambda c: scores[c][0])
    a, beta = fit_logit(B.Phi_tr, y, C)
    ll0, auc0 = scores[C]
    z_tr, z_te = a + B.Phi_tr @ beta, a + B.Phi_te @ beta
    rows = []
    for blk, idx in B.blocks.items():
        keep = [i for i in range(B.Phi_tr.shape[1]) if i not in set(idx)]
        ll1, auc1 = cv_score(B.Phi_tr[:, keep], y, C, folds)
        c_te = B.component(beta, blk, test=True)
        lo, hi = boot.delta_ci(z_te, z_te - c_te)
        rows.append(dict(signal=blk, data_cv_logloss_gain=ll1 - ll0, data_cv_auc_gain=auc0 - auc1,
                         data_component_sd=float(np.std(B.component(beta, blk))),
                         data_ablation_gain_test=default_auc(yt, z_te) - default_auc(yt, z_te - c_te),
                         data_ablation_gain_test_lo=lo, data_ablation_gain_test_hi=hi))
    info = dict(C=C, cv=scores, cv_logloss=ll0, cv_auc=auc0, auc_train=default_auc(y, z_tr),
                auc_test=default_auc(yt, z_te))
    return a, beta, pd.DataFrame(rows), info


# ============================================================================ 3-5. per network

class Proj:
    def __init__(self, B):
        A = np.hstack([np.ones((len(B.Phi_tr), 1)), B.Phi_tr])
        self.pinv = np.linalg.pinv(A, rcond=1e-10)

    def __call__(self, e):
        c = self.pinv @ e
        return float(c[0]), c[1:]


def out_terms(nt):
    """Output-feeding terms grouped by the innermost (leaf) part that has the node as an exit.
    Returns {leaf name: [(node, weight)]}, the leaf -> top-level part map, and the bias."""
    m = nt.m
    out = m.out_ids[0]
    root = next(p for p in nt.parts if p.is_root)
    by = {p.name: p for p in nt.parts}

    def top(p):
        while p.parent and p.parent != root.name:
            p = by[p.parent]
        return p.name

    terms, tops = {}, {}
    for node, w in m.ev.preds[out]:
        if m.nodes[node].type == NodeType.INPUT:
            key = "direct input edges"
            tops[key] = key
        else:
            cand = [p for p in nt.parts if not p.is_root and node in p.exits]
            leaves = [p for p in cand if p.kind == "leaf"] or cand
            if not leaves:
                leaves = [p for p in nt.unit_part([node]) if not p.is_root]
            leaves.sort(key=lambda p: len(p.ann.get("subgraph_nodes", [])))
            key = leaves[0].name
            tops[key] = top(leaves[0])
        terms.setdefault(key, []).append((node, w))
    return terms, tops, float(m.nodes[out].bias or 0.0)


def unit_out_terms(nt):
    """Output terms expanded through identity hidden nodes (exact: identity is linear), so every
    term is a ReLU unit or an input column feeding the score with a net weight. Keyed by node."""
    m = nt.m
    acc = {}

    def push(node, w):
        nd = m.nodes[node]
        if nd.type != NodeType.INPUT and nd.activation == "identity" and nd.type != NodeType.OUTPUT:
            if nd.bias:
                acc.setdefault("(identity biases)", [])
            for p, w2 in m.ev.preds[node]:
                push(p, w * w2)
            return
        key = "inputs (direct)" if nd.type == NodeType.INPUT else node
        acc.setdefault(key, []).append((node, w))

    for p, w in m.ev.preds[m.out_ids[0]]:
        push(p, w)
    acc.pop("(identity biases)", None)
    return acc


def contributions(nt, X, terms, overrides=None):
    vals = nt.m.ev.forward(X, overrides)
    return {k: sum(w * vals[n] for n, w in ts) for k, ts in terms.items()}


def mean_override(nt, n):
    mu = float(np.mean(nt.vals[n]))
    return {n: ((), (lambda vals, k=mu, size=None: np.full(len(next(iter(vals.values()))), k)))}


def main():
    data, boot_idx, nets = load_all()
    m0 = nets[LABELS[0]].m
    for g in ["day usage", "evening usage", "night usage", "international usage"]:
        a, b = GROUPS[g]
        ra, rb = m0.raw_train[:, col(m0, a)], m0.raw_train[:, col(m0, b)]
        RATIO[g] = float(np.mean(rb[ra > 0] / ra[ra > 0]))
    y, yt = m0.y, m0.yt
    for nt in nets.values():
        assert np.array_equal(nt.m.y, y) and np.array_equal(nt.m.yt, yt)
    boot = Boot(boot_idx, yt)
    nt0 = nets[LABELS[0]]
    selfcheck(boot_idx, yt, nt0.full_te, nt0.all_te)
    B = build_basis(m0)
    proj = Proj(B)
    blocks = list(B.blocks)

    # ---------------------------------------------------------------- 1. aliases
    al, vm_note = aliases(m0)
    al.to_csv(HERE / "aliases.csv", index=False)

    # ---------------------------------------------------------------- 2. data GAM
    a_d, beta_d, rel, gam_info = data_gam(B, y, yt, boot)
    comp_d = {b: B.component(beta_d, b) for b in blocks}
    comp_d_te = {b: B.component(beta_d, b, test=True) for b in blocks}
    print("data GAM", {k: v for k, v in gam_info.items() if k != "cv"}, flush=True)

    # ---------------------------------------------------------------- 3. projections
    P = {}
    for lab, nt in nets.items():
        a, beta = proj(nt.full_tr)
        comp = {b: B.component(beta, b) for b in blocks}
        comp_te = {b: B.component(beta, b, test=True) for b in blocks}
        fit_tr = a + B.Phi_tr @ beta
        fit_te = a + B.Phi_te @ beta
        P[lab] = dict(a=a, beta=beta, comp=comp, comp_te=comp_te, resid=nt.full_tr - fit_tr,
                      resid_te=nt.full_te - fit_te, r2=r2(nt.full_tr, fit_tr), r2_te=r2(nt.full_te, fit_te))
    beta_mean = np.mean([P[l]["beta"] for l in LABELS], axis=0)
    cons = {b: B.component(beta_mean, b) for b in blocks}
    cons_te = {b: B.component(beta_mean, b, test=True) for b in blocks}

    # ---------------------------------------------------------------- 4. units
    unit_rows, real_rows = [], []
    carriers = {}     # (lab, block) -> list of node ids
    for lab, nt in nets.items():
        m = nt.m
        hl = hidden_labels(nt.layer)
        terms, tops, _ = out_terms(nt)
        leaf_of_term = {n: k for k, ts in terms.items() for n, _ in ts}
        vinfo = classify_variables(m, nt.kinds, nt.bstep)
        nt._vinfo = vinfo
        base_of = {n: b for b, ns in nt.units.items() for n in ns}
        ugain = {}
        for b, ns in nt.units.items():
            lin_tr = m.ev.logit(m.X, nt.lin.overrides([b]))
            lin_te = m.ev.logit(m.Xt, nt.lin.overrides([b]))
            rest_tr = m.ev.logit(m.X, nt.lin.overrides([u for u in nt.units if u != b]))
            rest_te = m.ev.logit(m.Xt, nt.lin.overrides([u for u in nt.units if u != b]))
            lo, hi = boot.delta_ci(nt.full_te, lin_te)
            rlo, rhi = boot.delta_ci(rest_te, nt.all_te)
            ugain[b] = dict(gain_train=default_auc(y, nt.full_tr) - default_auc(y, lin_tr),
                            gain_test=default_auc(yt, nt.full_te) - default_auc(yt, lin_te),
                            gain_test_lo=lo, gain_test_hi=hi,
                            restore_train=default_auc(y, rest_tr) - default_auc(y, nt.all_tr),
                            restore_test=default_auc(yt, rest_te) - default_auc(yt, nt.all_te),
                            restore_test_lo=rlo, restore_test_hi=rhi)
        for n, h in sorted(hl.items(), key=lambda kv: int(kv[1][1:])):
            kind = nt.kinds.get(n, "?")
            ps = [p for p in nt.unit_part([n]) if not p.is_root]
            leaf = ps[0].name if ps else ""
            e = nt.full_tr - m.ev.logit(m.X, mean_override(nt, n))
            sd_e = float(np.std(e))
            a_u, b_u = proj(e)
            comps = {b: B.component(b_u, b) for b in blocks}
            share = {b: (float(np.cov(c, e)[0, 1] / np.var(e)) if sd_e > 1e-9 else 0.0) for b, c in comps.items()}
            msd = {b: float(np.std(c)) for b, c in comps.items()}
            ups = upstream_groups(nt, [n])
            reads = sorted({m.feature_names[input_column(p)] for p, w in m.ev.preds.get(n, [])
                            if w != 0 and m.nodes[p].type == NodeType.INPUT})
            hid_in = [p for p, w in m.ev.preds.get(n, []) if w != 0 and m.nodes[p].type != NodeType.INPUT]
            sw_up = any(nt.kinds.get(p) == "switching" for p in nt.ancestors(n))
            nl_blocks = {b: s for b, s in share.items() if kind_of(b) != "lin"}
            if sd_e < 1e-6:
                primary, status = "", "inert (no effect on the score)" if kind != "dead" else "dead (never active)"
            else:
                top_nl = max(nl_blocks, key=lambda b: msd[b]) if nl_blocks else None
                top_all = max(msd, key=lambda b: msd[b])
                if kind == "switching" and top_nl and msd[top_nl] >= max(CARRY_SD, 0.25 * msd[top_all]):
                    primary = top_nl
                else:
                    primary = top_all
                fit_r2 = float(sum(share.values()))
                status = "assigned" if fit_r2 >= 0.5 else "unassigned / residual"
                n_on = int((nt.pre[n] > 0).sum()) if n in nt.pre else len(m.X)
                if kind == "switching" and "state" in ups and min(n_on, len(m.X) - n_on) <= STATE_GATE_ROWS:
                    # a unit that opens only for a few customers of one or a few states: a state
                    # offset (possibly modulating another input), whatever else it reads
                    primary, status = "lin:state", "assigned (state-gated unit, few rows)"
            g = ugain.get(base_of.get(n, ""), {})
            unit_rows.append(dict(
                network=lab, seed=int(lab.split("-s")[1]), h=h, node=n, kind=kind,
                unit=base_of.get(n, ""), part=leaf, top_part=tops.get(leaf_of_term.get(n, ""), ""),
                feeds_output=n in leaf_of_term, n_active_train=int((nt.pre[n] > 0).sum()) if n in nt.pre else None,
                upstream=",".join(SHORT[x] for x in ups), reads_inputs="; ".join(reads),
                reads_hidden=" ".join(hid_in), switching_upstream=sw_up,
                knockout_sd=sd_e, projection_r2=float(sum(share.values())) if sd_e > 1e-9 else float("nan"),
                primary_signal=primary, primary_theme=THEME_OF.get(primary, ""), status=status,
                signals="; ".join(f"{b} {share[b]:.2f}" for b in sorted(share, key=lambda b: -msd[b])
                                  if msd[b] >= CARRY_SD)[:300],
                **{k: v for k, v in g.items()}))
            for b in blocks:
                if msd[b] >= CARRY_SD:
                    carriers.setdefault((lab, b), []).append((n, msd[b], share[b]))
        print(lab, "units done", flush=True)
    U = pd.DataFrame(unit_rows)

    # ---------------------------------------------------------------- 5. matrix + realisations
    mat_rows = []
    for lab, nt in nets.items():
        m = nt.m
        Ul = U[U.network == lab].set_index("node")
        vinfo = nt._vinfo
        for b in blocks:
            c, c_te = P[lab]["comp"][b], P[lab]["comp_te"][b]
            sd = float(np.std(c))
            abl_tr = default_auc(y, nt.full_tr) - default_auc(y, nt.full_tr - c)
            abl_te = default_auc(yt, nt.full_te) - default_auc(yt, nt.full_te - c_te)
            lo, hi = boot.delta_ci(nt.full_te, nt.full_te - c_te)
            cd = comp_d[b]
            corr_d = float(np.corrcoef(c, cd)[0, 1]) if sd > 1e-9 and np.std(cd) > 1e-9 else float("nan")
            corr_c = float(np.corrcoef(c, cons[b])[0, 1]) if sd > 1e-9 and np.std(cons[b]) > 1e-9 else float("nan")
            nv = nov(cd, nt.B) if kind_of(b) != "lin" else 0.0
            nv0 = nov(cd, m.X) if kind_of(b) != "lin" else 0.0
            cs = carriers.get((lab, b), [])
            mx = max([x[1] for x in cs], default=0.0)
            cs = [x for x in cs if x[1] >= CARRY_REL * mx]
            sw = [x for x in cs if Ul.loc[x[0], "kind"] == "switching"]
            parts = sorted({Ul.loc[x[0], "part"] for x in cs})
            sw_parts = sorted({Ul.loc[x[0], "part"] for x in sw})
            own_sw = [x for x in sw if Ul.loc[x[0], "primary_signal"] == b]
            used = sd >= USED_SD
            # structural gain: linearise the switching units whose primary signal is b
            sg = {}
            ubases = sorted({Ul.loc[x[0], "unit"] for x in own_sw if Ul.loc[x[0], "unit"]})
            if ubases:
                lin_tr = m.ev.logit(m.X, nt.lin.overrides(ubases))
                lin_te = m.ev.logit(m.Xt, nt.lin.overrides(ubases))
                slo, shi = boot.delta_ci(nt.full_te, lin_te)
                sg = dict(struct_gain_train=default_auc(y, nt.full_tr) - default_auc(y, lin_tr),
                          struct_gain_test=default_auc(yt, nt.full_te) - default_auc(yt, lin_te),
                          struct_gain_test_lo=slo, struct_gain_test_hi=shi)
            rtype = realisation_type(b, used, sd, sw, Ul, vinfo, m, sw_parts)
            aliases_read = alias_reading(b, m, nt)
            mat_rows.append(dict(
                network=lab, seed=int(lab.split("-s")[1]), signal=b, theme=THEME_OF.get(b, ""),
                kind=kind_of(b), sd_train=sd, used=used, ablation_gain_train=abl_tr,
                ablation_gain_test=abl_te, ablation_gain_test_lo=lo, ablation_gain_test_hi=hi,
                corr_with_data=corr_d, corr_with_consensus=corr_c, novelty=nv, novelty_inputs_only=nv0,
                decodable=bool(nv < MIN_NOV), realisation=rtype, alias_reading=aliases_read,
                n_carrier_units=len(cs), n_switching_carriers=len(sw), n_parts=len(parts),
                n_switching_parts=len(sw_parts),
                layout=("absent" if not used else "linear path" if kind_of(b) == "lin" and not sw
                        else "consolidated (1 part)" if len(sw_parts) <= 1 else f"distributed ({len(sw_parts)} parts)"),
                carrier_units=" ".join(f"{Ul.loc[x[0], 'h']}:{x[0]}" for x in sorted(cs, key=lambda x: -x[1])),
                parts=" | ".join(parts), **sg))
            for n, s_sd, s_sh in sorted(carriers.get((lab, b), []), key=lambda x: -x[1]):
                if s_sd < CARRY_REL * mx:
                    continue
                r = Ul.loc[n]
                real_rows.append(dict(network=lab, seed=int(lab.split("-s")[1]), signal=b,
                                      theme=THEME_OF.get(b, ""), h=r["h"], node=n, unit_kind=r["kind"],
                                      part=r["part"], role="primary" if r["primary_signal"] == b else "secondary",
                                      component_sd=s_sd, share_of_unit_effect=s_sh,
                                      upstream=r["upstream"], reads_inputs=r["reads_inputs"],
                                      unit_gain_train=r.get("gain_train"), unit_gain_test=r.get("gain_test"),
                                      unit_gain_test_lo=r.get("gain_test_lo"), unit_gain_test_hi=r.get("gain_test_hi"),
                                      realisation=rtype))
        # every unit appears at least once
        listed = {(x["network"], x["node"]) for x in real_rows}
        for _, r in U[U.network == lab].iterrows():
            if (lab, r["node"]) not in listed:
                real_rows.append(dict(network=lab, seed=r["seed"], signal=r["primary_signal"] or "(none)",
                                      theme=r["primary_theme"] or r["status"], h=r["h"], node=r["node"],
                                      unit_kind=r["kind"], part=r["part"],
                                      role=r["status"] if r["primary_signal"] else r["status"],
                                      component_sd=r["knockout_sd"], share_of_unit_effect=None,
                                      upstream=r["upstream"], reads_inputs=r["reads_inputs"],
                                      unit_gain_train=r.get("gain_train"), unit_gain_test=r.get("gain_test"),
                                      unit_gain_test_lo=r.get("gain_test_lo"), unit_gain_test_hi=r.get("gain_test_hi"),
                                      realisation="minor (below the 20% carrier cut for every block)"))
        print(lab, "matrix done", flush=True)
    M = pd.DataFrame(mat_rows)
    R = pd.DataFrame(real_rows)

    # cross-seed residual match of each seed's own realisation (non-linear blocks)
    xs = []
    for b in blocks:
        if kind_of(b) == "lin":
            continue
        for li in LABELS:
            ci = P[li]["comp"][b]
            if np.std(ci) < USED_SD:
                continue
            hits = [lj for lj in LABELS if lj != li and nov(ci, nets[lj].B) < MIN_NOV]
            xs.append(dict(signal=b, network=li, found_in_other_seeds=len(hits),
                           which=" ".join(l.split("-")[1] for l in hits)))
    XS = pd.DataFrame(xs)
    M = M.merge(XS[["signal", "network", "found_in_other_seeds"]], on=["signal", "network"], how="left")

    # ---------------------------------------------------------------- signals.csv
    S = signal_table(M, rel, blocks, B, comp_d)
    # ---------------------------------------------------------------- 6. disagreement
    D, H, agree = disagreement(nets, P, cons, cons_te, blocks, y, yt)
    # ---------------------------------------------------------------- 7. consolidation
    C, shares, curves = consolidation(nets, y, yt, boot)
    # ---------------------------------------------------------------- write
    U.to_csv(HERE / "units.csv", index=False)
    R.to_csv(HERE / "realisations.csv", index=False)
    M.to_csv(HERE / "matrix.csv", index=False)
    S.to_csv(HERE / "signals.csv", index=False)
    D.to_csv(HERE / "disagreement.csv", index=False)
    H.to_csv(HERE / "harmonise.csv", index=False)
    C.to_csv(HERE / "consolidation.csv", index=False)
    shares.to_csv(HERE / "part_shares.csv", index=False)
    CACHE.mkdir(exist_ok=True)
    proj_info = {l: dict(r2_train=P[l]["r2"], r2_test=P[l]["r2_te"], a=P[l]["a"]) for l in LABELS}
    (CACHE / "meta.json").write_text(json.dumps(dict(
        gam=gam_info, alias_voicemail=vm_note, ratio=RATIO, projection=proj_info, agreement=agree,
        knots=B.knots, blocks={b: len(i) for b, i in B.blocks.items()}), indent=1, default=float))
    np.savez_compressed(CACHE / "betas.npz", beta_data=beta_d, a_data=a_d,
                        **{f"beta_{l}": P[l]["beta"] for l in LABELS},
                        **{f"a_{l}": np.array(P[l]["a"]) for l in LABELS})
    (CACHE / "curves.json").write_text(json.dumps(curves, default=float))
    save_graphs(nets, U)
    print("done", flush=True)


# ============================================================================ helpers

def realisation_type(b, used, sd, sw, Ul, vinfo, m, sw_parts):
    if not used:
        return "absent" if sd < 0.03 else "weak (SD < 0.10 logit)"
    if kind_of(b) == "lin":
        cls = []
        for g in BLOCK_VARS[b]:
            for base in GROUPS[g]:
                inf = vinfo.get(base)
                cls.append(inf["class"] if inf else "not used")
        cs = sorted(set(cls) - {"not used"})
        lab = {"exact_linear": "direct linear edge", "support_linear": "linear on the data (always-on / one-step units)",
               "mixed": "direct edge + switching units", "nonlinear": "only through switching units"}
        return " + ".join(lab.get(c, c) for c in cs) or "only through switching units"
    if not sw:
        return "by-product of linear / other units"
    vars_b = {SHORT[g] for g in BLOCK_VARS[b]}
    descr = []
    for n, _, _ in sw:
        ups = set(Ul.loc[n, "upstream"].split(",")) - {""}
        extra = ups - vars_b
        t = "hinge" if not extra else "gate (also reads " + ",".join(sorted(extra)) + ")"
        if Ul.loc[n, "switching_upstream"]:
            t += " in a chain"
        descr.append(t)
    if len(sw) == 1:
        return "single " + descr[0]
    kinds = sorted(set(d.split(" (")[0].replace(" in a chain", "") for d in descr))
    return f"split across {len(sw)} units in {len(sw_parts)} part(s) ({', '.join(kinds)})"


def alias_reading(b, m, nt):
    """Which alias columns of the block's variables the network actually reads."""
    out = []
    for g in BLOCK_VARS[b]:
        bases = GROUPS[g]
        if len(bases) < 2:
            continue
        read = [x for x in bases if any(c in m.read_columns for c in m.var_by_base[x].columns)]
        if len(read) == 2:
            out.append(f"{SHORT[g]}: both")
        elif read:
            out.append(f"{SHORT[g]}: {read[0]}")
    return "; ".join(out)


def signal_table(M, rel, blocks, B, comp_d):
    rows = []
    rel = rel.set_index("signal")
    for b in blocks:
        g = M[M.signal == b]
        n_used = int(g.used.sum())
        n_dec = int(g.decodable.sum())
        r = rel.loc[b]
        relevant = r.data_cv_logloss_gain > 0
        if n_used == 10:
            cls = "core"
        elif n_used >= 4:
            cls = "common"
        elif n_used >= 1:
            cls = "rare"
        else:
            cls = "absent from the models"
        test_ok = int(((g.ablation_gain_test_lo > 0) & g.used).sum())
        test_bad = int(((g.ablation_gain_test_hi < 0) & g.used).sum())
        noise = bool(n_used > 0 and test_ok == 0 and (not relevant or test_bad > 0))
        rows.append(dict(
            signal=b, name=NAMES.get(b, b), theme=THEME_OF.get(b, ""), kind=kind_of(b),
            variables=", ".join(sorted(BLOCK_VARS[b])), n_columns=len(B.blocks[b]),
            knots_raw=json.dumps(B.knots.get(b, [])),
            data_component_sd=r.data_component_sd, data_cv_logloss_gain=r.data_cv_logloss_gain,
            data_cv_auc_gain=r.data_cv_auc_gain, data_relevant=bool(relevant),
            data_ablation_gain_test=r.data_ablation_gain_test,
            data_ablation_gain_test_lo=r.data_ablation_gain_test_lo,
            data_ablation_gain_test_hi=r.data_ablation_gain_test_hi,
            n_seeds_used=n_used, n_seeds_decodable=n_dec, cls=cls,
            noise_flag=noise,
            median_sd_used=float(g[g.used].sd_train.median()) if n_used else 0.0,
            median_ablation_gain_train=float(g[g.used].ablation_gain_train.median()) if n_used else 0.0,
            median_ablation_gain_test=float(g[g.used].ablation_gain_test.median()) if n_used else 0.0,
            seeds_test_gain_ci_above_0=test_ok, seeds_test_gain_ci_below_0=test_bad,
            median_corr_with_data=float(g[g.used].corr_with_data.median()) if n_used else float("nan"),
            median_corr_with_consensus=float(g[g.used].corr_with_consensus.median()) if n_used else float("nan"),
            realisations="; ".join(f"s{s}:{t}" for s, t in zip(g.seed, g.realisation) if t not in ("absent",)),
            layouts="; ".join(f"s{s}:{t}" for s, t in zip(g.seed, g.layout) if t != "absent")))
    return pd.DataFrame(rows)


def agreement(etas, k=TOP_K):
    cs, ov = [], []
    for i, j in combinations(range(len(etas)), 2):
        cs.append(np.corrcoef(etas[i], etas[j])[0, 1])
        ov.append(top_overlap(etas[i], etas[j], k))
    return float(np.median(cs)), float(np.median(ov)), float(np.mean(cs)), float(np.mean(ov))


def disagreement(nets, P, cons, cons_te, blocks, y, yt):
    rows = []
    themes = list(THEMES) + ["residual (unassigned)"]
    for split in ("train", "test"):
        etas = {l: (nets[l].full_tr if split == "train" else nets[l].full_te) for l in LABELS}
        comp = {l: (P[l]["comp"] if split == "train" else P[l]["comp_te"]) for l in LABELS}
        res = {l: (P[l]["resid"] if split == "train" else P[l]["resid_te"]) for l in LABELS}
        for li, lj in combinations(LABELS, 2):
            d = etas[li] - etas[lj]
            d = d - d.mean()
            vd = np.var(d)
            row = dict(split=split, a=li, b=lj, var_diff=vd,
                       corr=float(np.corrcoef(etas[li], etas[lj])[0, 1]))
            for t in themes:
                if t.startswith("residual"):
                    dk = res[li] - res[lj]
                else:
                    dk = sum(comp[li][b] - comp[lj][b] for b in THEMES[t])
                dk = dk - dk.mean()
                row[t] = float(np.mean(dk * d) / vd)
            rows.append(row)
    D = pd.DataFrame(rows)
    # harmonise one theme at a time: every network's component replaced by the cross-seed mean
    H = []
    base = {}
    for split in ("train", "test"):
        etas = [nets[l].full_tr if split == "train" else nets[l].full_te for l in LABELS]
        yy = y if split == "train" else yt
        base[split] = agreement(etas)
        aucs = [default_auc(yy, e) for e in etas]
        H.append(dict(split=split, harmonised="(none)", median_corr=base[split][0],
                      median_top_overlap=base[split][1], median_auc=float(np.median(aucs)),
                      ensemble_auc=default_auc(yy, np.mean(etas, axis=0))))
        options = {t: THEMES[t] for t in THEMES}
        options["service-call complex (calls, calls x low day)"] = THEMES["service calls"] + THEMES["service calls x low day usage"]
        options["heavy-usage complex (day, day+eve, day w/o voice mail, eve)"] = (
            THEMES["day usage"] + THEMES["day + evening bill"] + THEMES["day usage without voice mail"] + THEMES["evening usage"])
        options["both complexes"] = (options["service-call complex (calls, calls x low day)"] +
                                     options["heavy-usage complex (day, day+eve, day w/o voice mail, eve)"])
        options["all signal blocks (only the residual differs)"] = list(blocks)
        options["all except state offsets"] = [b for b in blocks if b != "lin:state"]
        for t, bs in options.items():
            new = []
            for l, e in zip(LABELS, etas):
                comp = P[l]["comp"] if split == "train" else P[l]["comp_te"]
                cc = cons if split == "train" else cons_te
                new.append(e - sum(comp[b] for b in bs) + sum(cc[b] for b in bs))
            ag = agreement(new)
            aucs = [default_auc(yy, e) for e in new]
            H.append(dict(split=split, harmonised=t, median_corr=ag[0], median_top_overlap=ag[1],
                          median_auc=float(np.median(aucs)), ensemble_auc=default_auc(yy, np.mean(new, axis=0))))
        # state at average (all state one-hots at their mean) and state component removed
        new = []
        for l in LABELS:
            nt = nets[l]
            X = (nt.m.X if split == "train" else nt.m.Xt).copy()
            sc = [i for i, n in enumerate(nt.m.feature_names) if n.startswith("state:")]
            X[:, sc] = 0.0
            new.append(nt.m.ev.logit(X))
        ag = agreement(new)
        H.append(dict(split=split, harmonised="state inputs held at average", median_corr=ag[0],
                      median_top_overlap=ag[1], median_auc=float(np.median([default_auc(yy, e) for e in new])),
                      ensemble_auc=default_auc(yy, np.mean(new, axis=0))))
    return D, pd.DataFrame(H), {k: list(v) for k, v in base.items()}


# ============================================================================ 7. consolidation

def pl_basis(x, knots):
    return np.column_stack([np.ones_like(x), x] + [relu(x - k) for k in knots])


def consolidation(nets, y, yt, boot):
    """Per network and variable: the score's response to the variable decomposed into the
    leaf parts whose exits feed the output (they sum to the logit exactly), how many parts carry
    it, and whether one consolidated feature reproduces it."""
    rows, srows = [], []
    curves = {}
    G1_grid = {}
    for g in RESP_VARS:
        for lab, nt in nets.items():
            m = nt.m
            terms, tops, bias = out_terms(nt)
            grid, ref = var_grid(m, g)
            C0 = contributions(nt, m.X, terms)
            tot = bias + sum(C0.values())
            assert np.max(np.abs(tot - nt.full_tr)) < 1e-9, (lab, "parts do not sum to the logit")
            Xref = set_cols(m.X, var_assign(m, g, ref))
            Cref = contributions(nt, Xref, terms)
            Xref_t = set_cols(m.Xt, var_assign(m, g, ref))
            # PD per part on the grid (mean over training rows)
            pd_parts = {k: [] for k in terms}
            for v in grid:
                Cv = contributions(nt, set_cols(m.X, var_assign(m, g, v)), terms)
                for k in terms:
                    pd_parts[k].append(float(np.mean(Cv[k] - Cref[k])))
            delta = {k: C0[k] - Cref[k] for k in terms}
            var = {k: float(np.var(d)) for k, d in delta.items()}
            tv = sum(var.values())
            sh = {k: (v / tv if tv > 0 else 0.0) for k, v in var.items()}
            D = sum(delta.values())
            canc = float(np.var(D) / tv) if tv > 0 else float("nan")
            neff = 1.0 / sum(s * s for s in sh.values()) if tv > 0 else 0.0
            carrying = [k for k, s in sh.items() if s >= 0.05]
            # the same at unit level (terms expanded through identity nodes)
            uterms = unit_out_terms(nt)
            U0 = contributions(nt, m.X, uterms)
            Ur = contributions(nt, Xref, uterms)
            uvar = {k: float(np.var(U0[k] - Ur[k])) for k in uterms}
            utv = sum(uvar.values())
            ush = {k: (v / utv if utv > 0 else 0.0) for k, v in uvar.items()}
            hl = hidden_labels(nt.layer)
            ucarry = sorted([k for k, v in ush.items() if v >= 0.05], key=lambda k: -ush[k])
            upd = {}
            for v in grid:
                Uv = contributions(nt, set_cols(m.X, var_assign(m, g, v)), uterms)
                for k in uterms:
                    upd.setdefault(k, []).append(float(np.mean(Uv[k] - Ur[k])))
            unit_info = dict(n_units_carrying=len(ucarry),
                             effective_units=1.0 / sum(x * x for x in ush.values()) if utv > 0 else 0.0,
                             carrying_units=" | ".join(f"{hl.get(k, k)}:{k} ({ush[k]:.0%})" for k in ucarry))
            # entanglement per carrying part: share of the part's variance that the variable explains
            for k in terms:
                vk = float(np.var(C0[k]))
                srows.append(dict(variable=g, network=lab, part=k, top_part=tops[k], share=sh[k],
                                  part_var=vk, frac_of_part_from_variable=(var[k] / vk if vk > 1e-12 else float("nan")),
                                  upstream=",".join(SHORT[x] for x in upstream_groups(nt, [n for n, _ in terms[k]]))
                                  if k != "direct input edges" else "inputs"))
            ded = []
            for k in carrying:
                ups = set(upstream_groups(nt, [n for n, _ in terms[k]])) if k != "direct input edges" else {"(direct)"}
                ded.append(ups <= {g} or k == "direct input edges")
            # separability: every switching node downstream of the variable reads only the variable
            sep = separable(nt, g)
            # consolidated single-variable feature fitted to D on train
            vtr = variable_values(m, g, m.raw_train)
            vte = variable_values(m, g, m.raw_test)
            Gtr, Gte, gfun = fit_g1(g, vtr, vte, D, m)
            C0t = contributions(nt, m.Xt, terms)
            Creft = contributions(nt, Xref_t, terms)
            Dt = sum(C0t[k] - Creft[k] for k in terms)
            e1, e1t = nt.full_tr - D + Gtr, nt.full_te - Dt + Gte
            lo, hi = boot.delta_ci(e1t, nt.full_te)
            G1_grid.setdefault(g, {})[lab] = gfun(np.where(grid > 0, grid, -1.0) if g == "voice mail" else grid)
            row = dict(variable=g, network=lab, seed=int(lab.split("-s")[1]),
                       response_range=float(np.max(np.sum(list(pd_parts.values()), axis=0)) -
                                            np.min(np.sum(list(pd_parts.values()), axis=0))),
                       n_parts_total=len(terms), n_parts_carrying=len(carrying),
                       effective_parts=neff, cancellation_ratio=canc,
                       carrying_parts=" | ".join(f"{k} ({sh[k]:.0%})" for k in sorted(carrying, key=lambda k: -sh[k])),
                       dedicated_parts=int(sum(ded)), entangled_parts=int(len(ded) - sum(ded)),
                       separable_structurally=sep, **unit_info,
                       g1_r2=r2(D, Gtr), g1_max_abs_dlogit=float(np.max(np.abs(D - Gtr))),
                       g1_mean_abs_dlogit=float(np.mean(np.abs(D - Gtr))),
                       g1_auc_train=default_auc(y, e1), auc_train=default_auc(y, nt.full_tr),
                       g1_auc_test=default_auc(yt, e1t), auc_test=default_auc(yt, nt.full_te),
                       g1_dauc_test_lo=lo, g1_dauc_test_hi=hi)
            if g == "service calls":   # consolidated feature of (calls, day usage)
                dtr = variable_values(m, "day usage", m.raw_train)
                dte = variable_values(m, "day usage", m.raw_test)
                G3, G3t = fit_g3(vtr, vte, dtr, dte, D)
                e3, e3t = nt.full_tr - D + G3, nt.full_te - Dt + G3t
                lo3, hi3 = boot.delta_ci(e3t, nt.full_te)
                row.update(g3_r2=r2(D, G3), g3_max_abs_dlogit=float(np.max(np.abs(D - G3))),
                           g3_auc_train=default_auc(y, e3), g3_auc_test=default_auc(yt, e3t),
                           g3_dauc_test_lo=lo3, g3_dauc_test_hi=hi3)
            row["_D"], row["_Dt"], row["_v"], row["_vt"] = D, Dt, vtr, vte
            rows.append(row)
            curves.setdefault(g, {})[lab] = dict(grid=list(grid), ref=ref, parts={k: pd_parts[k] for k in terms},
                                                 tops=tops, shares=sh,
                                                 units={hl.get(k, k) + (":" + k if k in hl else ""): upd[k] for k in uterms},
                                                 unit_shares={hl.get(k, k) + (":" + k if k in hl else ""): ush[k] for k in uterms})
    # consensus shape (mean of the ten G1 curves on the grid, centred at the reference)
    for row in rows:
        g, lab = row["variable"], row["network"]
        nt = nets[lab]
        grid = np.asarray(curves[g][lab]["grid"])
        cons = np.mean([G1_grid[g][l] for l in LABELS], axis=0)
        gref = np.interp(curves[g][lab]["ref"], grid, cons)
        f = lambda v: np.interp(v, grid, cons) - gref
        D, Dt, v, vt = row.pop("_D"), row.pop("_Dt"), row.pop("_v"), row.pop("_vt")
        e2, e2t = nt.full_tr - D + f(v), nt.full_te - Dt + f(vt)
        lo, hi = boot.delta_ci(e2t, nt.full_te)
        row.update(cons_max_abs_dlogit=float(np.max(np.abs(D - f(v)))), cons_auc_train=default_auc(y, e2),
                   cons_auc_test=default_auc(yt, e2t), cons_dauc_test_lo=lo, cons_dauc_test_hi=hi)
        row["exactness"] = ("exact on the data" if row["g1_max_abs_dlogit"] < 1e-9 else
                            "approximate" if row["g1_r2"] >= 0.9 else "breaks interactions")
        row["explaneat_expressible"] = (
            "already one part" if row["n_parts_carrying"] <= 1 and row["entangled_parts"] == 0 else
            "yes: a composition of the carrying parts is a pure feature of the variable (function-preserving)"
            if row["entangled_parts"] == 0 and row["separable_structurally"] else
            "yes: the variable enters additively (direct edges / units reading only it), so identity nodes "
            "or splits can isolate it into one part (function-preserving)"
            if row["separable_structurally"] else
            "grouping only: a composition of the carrying parts is function-preserving but also carries "
            "their other inputs; a single-variable feature needs a rewrite that is "
            + ("exact on the data only" if row["exactness"] == "exact on the data" else "not function-preserving"))
        curves[g][lab]["g1"] = list(G1_grid[g][lab])
        curves[g][lab]["consensus"] = list(cons - gref)
    return pd.DataFrame(rows), pd.DataFrame(srows), curves


def variable_values(m, g, raw):
    if g in ("day usage", "evening usage"):
        return raw[:, col(m, GROUPS[g][0])]
    if g == "voice mail":   # messages for plan holders, -1 without a plan
        plan = raw[:, col(m, "voice mail plan")]
        return np.where(plan > 0.5, raw[:, col(m, "number vmail messages")], -1.0)
    if g == "international plan":
        return raw[:, col(m, "international plan")]
    return raw[:, col(m, "number customer service calls")]


def fit_g1(g, v, vt, D, m):
    if g in ("service calls", "international plan"):
        lv = np.unique(v)
        means = np.array([D[v == x].mean() for x in lv])
        f = lambda x: np.interp(x, lv, means)
        return f(v), f(vt), f
    if g == "voice mail":   # no plan (-1) vs plan holders: piecewise linear in messages
        hold = v >= 0
        kn = list(np.quantile(v[hold], [0.25, 0.5, 0.75]))
        A = np.column_stack([~hold, hold, hold * v] + [hold * relu(v - k) for k in kn]).astype(float)
        coef, *_ = np.linalg.lstsq(A, D, rcond=None)

        def f(x):
            x = np.asarray(x, float)
            h = x >= 0
            Ax = np.column_stack([~h, h, h * x] + [h * relu(x - k) for k in kn]).astype(float)
            return Ax @ coef
        return f(v), f(vt), f
    kn = list(np.quantile(v, [0.10, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98]))
    coef, *_ = np.linalg.lstsq(pl_basis(v, kn), D, rcond=None)
    f = lambda x: pl_basis(np.asarray(x, float), kn) @ coef
    return f(v), f(vt), f


def fit_g3(v, vt, d, dt, D):
    lv = np.unique(v)
    top = 7
    kn = list(np.quantile(d, [0.25, 0.5, 0.75, 0.9]))

    def A(vv, dd):
        cols = []
        for x in lv[lv <= top]:
            ind = (np.minimum(vv, top) == x).astype(float)
            cols += [ind, ind * dd] + [ind * relu(dd - k) for k in kn]
        return np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A(v, d), D, rcond=None)
    return A(v, d) @ coef, A(vt, dt) @ coef


def separable(nt, g):
    m = nt.m
    cols = set()
    for b in GROUPS[g]:
        cols |= set(m.var_by_base[b].columns)
    src = [n for n, nd in m.nodes.items() if nd.type == NodeType.INPUT and input_column(n) in cols]
    succ = {}
    for n, ps in m.ev.preds.items():
        for p, w in ps:
            if w != 0:
                succ.setdefault(p, []).append(n)
    seen, stack = set(), list(src)
    while stack:
        x = stack.pop()
        for s in succ.get(x, []):
            if s not in seen:
                seen.add(s)
                stack.append(s)
    for n in seen:
        if nt.kinds.get(n) == "switching" and n in nt.reach:
            if set(upstream_groups(nt, [n])) - {g}:
                return False
    return True


def save_graphs(nets, U):
    """Node/edge lists of every network's model state, with h-labels and signals, for figures."""
    out = {}
    for lab, nt in nets.items():
        m = nt.m
        hl = hidden_labels(nt.layer)
        nodes = []
        Ul = U[U.network == lab].set_index("node")
        for n, nd in m.nodes.items():
            if nd.type == NodeType.INPUT:
                if n not in nt.reach:
                    continue
                nodes.append(dict(id=n, kind="input", label=m.feature_names[input_column(n)],
                                  group=GROUP_OF[m.base_of_col[input_column(n)]]))
            elif nd.type == NodeType.OUTPUT:
                nodes.append(dict(id=n, kind="output", label="churn score"))
            else:
                r = Ul.loc[n] if n in Ul.index else None
                nodes.append(dict(id=n, kind="hidden", label=hl.get(n, n), unit_kind=nt.kinds.get(n),
                                  signal=(r["primary_signal"] if r is not None else ""),
                                  theme=(r["primary_theme"] if r is not None else ""),
                                  status=(r["status"] if r is not None else ""),
                                  part=(r["part"] if r is not None else "")))
        ids = {x["id"] for x in nodes}
        edges = [dict(a=p, b=n, w=w) for n, ps in m.ev.preds.items() for p, w in ps
                 if p in ids and n in ids and w != 0]
        out[lab] = dict(nodes=nodes, edges=edges)
    (CACHE / "graphs.json").write_text(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
