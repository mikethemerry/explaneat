"""Heart-100 audit: every excess (non-linear) feature of the ten Experiment 2 Heart-100 networks.

Read-only on explaneat_v4 (every session is READ ONLY). Nothing is selected or tuned on the test
split; test numbers are computed once and reported, in columns marked ``_test``.

Definitions (all fits and descriptions on the 242 training rows):

- **Non-linear unit.** A hidden ReLU of the explained model state that reaches the output and
  switches on the training rows (on for some, off for others), after the binary one-step rule
  (``plain_language_linearity.unit_kinds``): a switching ReLU that reads one two-valued input is
  exactly linear on the data and is excluded (counted). Split copies (``1718_a``, ``1718_b``) are
  one unit: they compute the same value.
- **Part.** A v4 annotation (non-root, not a duplicate) whose class is ``nonlinear``
  (``part_class``: it holds a switching unit). A unit belongs to the annotation whose own nodes
  contain it (the innermost one).
- **Linearised.** The unit's (or part's) exits replaced by their least-squares affine fit on its
  entries, fitted once on the training rows (``explanation_tools.Linearisation``, the explorer's
  "Restore gain" tab). The whole-network linearised model replaces every non-linear unit at once
  (``linearised_auc`` of the linearity claims: reproduced and checked).
- **Restore gain** (``gain_*``, Mike's definition): AUC(full model) - AUC(model with just this
  unit / part linearised), on train and on test. ``restore_onto_linear_*``: the explorer's other
  column, AUC(every other unit linearised) - AUC(all linearised). Test: paired bootstrap 95% CI
  of the AUC difference, 10,000 resamples of the 61 test rows, RandomState(42)
  (``compute_runs.bootstrap_indices``, ``explained_eval.bootstrap_delta_ci``).
- **Regime.** The minority side of the switch on train: the rows where the unit is on if it is on
  for at most half the patients, else the rows where it is off. Described in real units by a
  depth-2 decision tree (min 5 rows per leaf) on the raw columns of the unit's drivers, with its
  agreement; the tree's leaves are the data-support cells.
- **Drivers and modifiers.** For each input variable upstream of the unit, set it to a reference
  value on every training row (the median for a number, the most common level for a category):
  a *switch driver* flips the unit's on/off state on at least half as many rows as the top
  variable (and on >= 5 rows, or on as many as the top one when it flips fewer); a *strength
  modifier* is any other variable whose change in the pre-activation has >= 25% of the summed SD
  over the rows where the unit is on (it sets how strongly the open unit acts, e.g. age inside a
  thal gate).
- **Family.** The drivers' tags (resting BP = trestbps, thal, exang first), with ``x age`` when age
  is a driver or a modifier; ``age`` when age is the only driver. ``family_group`` keeps only BP /
  thal / exang (and ``x age``), else age, else the first driver.
- **Effect.** Change in predicted risk, full model minus the model with this unit / part
  linearised: mean |change| on the regime rows, the max over all training rows, and the mean
  signed change on the regime (+: the model gives the regime more risk than the straight-line
  fit). ``on_effect_logit``: mean change in the score when the unit is switched off (fixed at 0),
  over the rows where it is on (``on_direction``: being on raises / lowers the risk).
  ``n_core``: the fewest training patients carrying half of the total |change in predicted
  risk|, with their ranges (``core_rows``). ``bands``: the regime split at the training quartiles
  of its first numeric driver or modifier (age first), n and mean |change| per band.
- **Recurrence** (species-leads strict residual match, ``signal_inventory.py``): the signal's
  residual is its activation minus its least-squares fit on the 26 input columns (train). Signals
  whose residual is < 5% of their variance are "near-linear" (no match). Seed j has the signal
  when nov = 1 - R^2 of the OLS fit of the residual on seed j's inputs plus every hidden
  activation (``genome_surgery.signal_basis``) is < 0.5 (max over exits for a part). Counted over
  the other 9 Heart-100 seeds and over the ten 40-generation Heart seeds; the best single
  40-generation inventory signal is the one whose residual explains most of this residual
  (match when R^2 >= 0.5).
- **Verdict.** "generalises" when the test restore gain's CI is above 0, "worse on test" when it
  is below 0, else "no measurable test effect"; plus "recurring" (matched in >= 1 other Heart-100
  seed) / "seed-specific" / "near-linear"; plus "small support (n<30)" when the regime, or its
  largest tree cell inside the regime, has fewer than 30 training patients.

- **Presence** (the rule shared with the Adult and Churn catalogues,
  ``explaneat/analysis/signal_presence.py``): a term or family is *used* by a network when its logit
  contribution has SD >= 0.10 logit over the training rows after clipping to the 1st-99th
  percentiles, *weak* at 0.03-0.10. Linear term (``linear_core.csv``: ``proj_coef``,
  ``sd_contribution``, ``presence``, ``direction``): weight x input, the weight from the
  least-squares projection of the network's logit on the linear columns (``linear_basis``: a
  number's column, a category's non-reference levels), zero when the network reads no column of
  the variable; direction against the mean contribution of the seeds that use the term
  (``data_coef``: the data's own ridge logistic fit, for reference). Non-linear family
  (``family_presence.csv``; per unit ``nl_sd``, ``presence`` in ``features.csv``): the full logit
  minus the logit with the family's units linearised.

    DATABASE_URL=postgresql://localhost/explaneat_v4 \\
    EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 \\
    uv run python experiments/heart100_audit/audit.py

Needs ``cache/heart40.npz`` from ``heart40_signals.py`` (run on explaneat_dev). Writes
``features.csv``, ``linear_core.csv``, ``family_presence.csv``, ``networks.csv``, ``recurrence.csv``
and ``tables.md`` next to this file; ``summarise.py`` then writes ``families.csv`` and ``summary.md`` from them.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

threadpool_limits(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments/10seed"), str(REPO / "experiments/churn")):
    if p not in sys.path:
        sys.path.insert(0, p)

import explained_eval as ee  # noqa: E402
import genome_surgery as gs  # noqa: E402
from compute_runs import ReadOnlySession, bootstrap_indices, load_split  # noqa: E402
from explaneat.analysis.explanation_tools import Linearisation, default_auc, input_column  # noqa: E402
from explaneat.analysis.feature_labels import both_names, feature_label  # noqa: E402
from explaneat.analysis.plain_language import build_parts, load_model  # noqa: E402
from explaneat.analysis.signal_presence import consensus, consensus_direction, presence  # noqa: E402  (shared with adult, churn)
from explaneat.analysis.plain_language_linearity import (  # noqa: E402
    classify_variables, linearity_claims, part_class, unit_kinds)
from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.core.genome_network import NodeType  # noqa: E402
from explaneat.core.split_ids import base_id  # noqa: E402
from explaneat.db.base import db  # noqa: E402
from explaneat.db.models import Genome  # noqa: E402
from explaneat.export.plain_cli import genome_for_label  # noqa: E402

LABELS = [f"heart100-s{i}" for i in range(10)]
if os.environ.get("AUDIT_LABELS"):  # smoke test only
    LABELS = os.environ["AUDIT_LABELS"].split(",")
SPLIT = "165cebbd-6609-46ca-867e-c09e17b2d6ec"
LAYERS = REPO / "experiments/10seed/plain_language_v4"
LIN_V2 = REPO / "experiments/10seed/linearity_v2"
MIN_NOV, MIN_NL_SHARE, SMALL_N, MIN_FLIP, MIN_MOD_SHARE = 0.5, 0.05, 30, 5, 0.25
TAG = {"trestbps": "BP", "thal": "thal", "exang": "exang", "age": "age", "oldpeak": "ST depression",
       "slope": "ST slope", "cp": "chest pain", "thalach": "max HR", "ca": "vessels", "sex": "sex",
       "chol": "cholesterol", "fbs": "fasting sugar", "restecg": "resting ECG"}
PRIMARY = ["BP", "thal", "exang"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_genome(gid):
    with ReadOnlySession() as s:
        row = s.get(Genome, uuid.UUID(gid))
        exp = row.population.experiment
        config = load_neat_config(exp.neat_config_text or "", exp.config_json)
        return row.to_neat_genome(config), config


def ubase(n):
    b = base_id(n)
    return b if b else n


def nov(x, B):
    v = gs.linear_novelty(x, B)["nov"]
    return float(v) if np.isfinite(v) else 0.0   # constant exit: nothing to reproduce


def resid_of(x, X):
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ coef
    share = float(min(1.0, r.var() / x.var())) if x.var() > 1e-12 else 0.0
    return r, share


def delta_ci(y, a, b, boot):
    lo, hi, _ = ee.bootstrap_delta_ci(y, a, b, boot)
    return lo, hi


class Net:
    def __init__(self, label, data, boot):
        self.label = label
        gid = genome_for_label(label)
        m, meta = load_model(gid)
        self.m, self.meta, self.gid = m, meta, gid
        assert np.max(np.abs(m.X - data["X_train"])) < 1e-12 and np.max(np.abs(m.Xt - data["X_test"])) < 1e-12
        self.layer = json.loads((LAYERS / f"{label}.json").read_text())
        assert self.layer["explanation_id"] == meta["explanation_id"], label
        self.v2 = json.loads((LIN_V2 / f"{label}.json").read_text())
        self.parts = build_parts(m)
        self.bstep = {}
        self.kinds = unit_kinds(m, binary_step=True, details=self.bstep)
        self.reach = m.reaching_output()
        self.claims, self.graph = linearity_claims(m, self.parts, binary_step=True)
        self.model_stmt = next(c["statement"] for c in self.claims
                               if c["type"] == "linearity" and c["statement"]["kind"] == "model")
        self.vinfo = classify_variables(m, self.kinds, self.bstep)
        genome, config = load_genome(gid)
        p_ref = gs.predict_proba(genome, config, m.X)
        assert np.max(np.abs(p_ref - sigmoid(m.base_logit))) < 1e-9, label
        self.B, _ = gs.signal_basis(genome, config, m.X)
        self.boot = boot
        self.cb = m.codebook
        self.vals = m.ev.forward(m.X)
        self.vals_t = m.ev.forward(m.Xt)
        self.pre = m.ev.pre_activations(m.X)
        self.pre_t = m.ev.pre_activations(m.Xt)
        self.full_tr, self.full_te = m.ev.logit(m.X), m.ev.logit(m.Xt)
        self._units()
        self._annotation_of()

    # ------------------------------------------------------------------ units
    def _units(self):
        m = self.m
        sw = sorted(n for n, k in self.kinds.items() if k == "switching" and n in self.reach)
        self.sw_nodes = sw
        self.bs_nodes = sorted(n for n, k in self.kinds.items() if k == "binary_step" and n in self.reach)
        groups = {}
        for n in sw:
            groups.setdefault(ubase(n), []).append(n)
        for b, ns in groups.items():   # split copies compute the same value
            for x in ns[1:]:
                assert np.max(np.abs(self.pre[x] - self.pre[ns[0]])) < 1e-12, (self.label, x)
        self.units = groups
        udefs = {b: {"entry_nodes": [p for p, _ in m.ev.preds[ns[0]]], "exit_nodes": ns}
                 for b, ns in groups.items()}
        for b, ns in groups.items():
            for x in ns[1:]:
                assert sorted(m.ev.preds[x]) == sorted(m.ev.preds[ns[0]]), (self.label, x)
        self.lin = Linearisation(m.ev, udefs, m.X)
        allu = list(groups)
        self.all_tr = m.ev.logit(m.X, self.lin.overrides(allu))
        self.all_te = m.ev.logit(m.Xt, self.lin.overrides(allu))
        st = self.model_stmt
        assert abs(default_auc(m.y, self.all_tr) - float(st["auc_linearised_train"])) < 1e-5, self.label
        assert abs(default_auc(m.yt, self.all_te) - float(st["auc_linearised_test"])) < 1e-5, self.label
        assert st["switching_units"] == len(sw) == self.v2["model"]["after"]["switching_units"], self.label

    def _annotation_of(self):
        own = {}
        for p in self.parts:
            ents = {str(e) for e in p.ann.get("entry_nodes", [])}
            for n in p.ann.get("subgraph_nodes", []):
                if str(n) not in ents:
                    own.setdefault(str(n), []).append(p)
        self.own = own

    def unit_part(self, nodes):
        ps = []
        for n in nodes:
            for p in self.own.get(n, []):
                if p not in ps:
                    ps.append(p)
        ps.sort(key=lambda p: (p.kind != "leaf", p.is_root, len(p.ann.get("subgraph_nodes", []))))
        return ps

    def part_info(self, p):
        d = self.layer.get("drafted") or {}
        title = (d.get("titles") or {}).get(p.id, "")
        sents = [s.get("text", "") for s in (d.get("annotations") or {}).get(p.id, [])]
        lp = next((q for q in self.layer["parts"] if q["id"] == p.id), {})
        assert not lp or lp["name"] == p.name, (self.label, p.id)
        return title, " ".join(sents[:2])[:400], (lp.get("original_hypothesis") or "")[:400]

    # ------------------------------------------------------------------ helpers
    def ancestors(self, n):
        m, seen, stack = self.m, set(), [n]
        while stack:
            x = stack.pop()
            for p, w in m.ev.preds.get(x, []):
                if w != 0 and p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def input_names(self, n):
        out = []
        for p, w in self.m.ev.preds[n]:
            if w == 0:
                continue
            node = self.m.nodes[p]
            if node.type == NodeType.INPUT:
                nm = self.m.feature_names[input_column(p)]
                out.append(f"{p}={both_names(feature_label(self.cb, nm))} ({w:+.2f})")
            else:
                out.append(f"unit {p} ({w:+.2f})")
        return out

    def upstream_names(self, nodes):
        _, cols = self.anc_vars(nodes)
        return [both_names(feature_label(self.cb, self.m.feature_names[c])) for c in sorted(cols)]

    def anc_vars(self, nodes):
        cols = set()
        for n in nodes:
            for a in self.ancestors(n):
                if self.m.nodes[a].type == NodeType.INPUT:
                    cols.add(input_column(a))
        return [v for v in self.m.variables if any(c in cols for c in v.columns)], cols

    def zfun(self, node, X):
        m = self.m
        vals = m.ev.forward(X)
        z = np.full(X.shape[0], float(m.nodes[node].bias or 0.0))
        for p, w in m.ev.preds[node]:
            z = z + w * vals[p]
        return z

    def reference(self, v):
        if v.is_category():
            codes, cnt = np.unique(np.asarray(v.raw_train).astype(str), return_counts=True)
            code = codes[np.argmax(cnt)]
            return code, v.encode(code), v.level(code)["label"]
        med = float(np.median(np.asarray(v.raw_train, float)))
        return med, v.encode(med), v.fmt(med)

    def drivers(self, node, on):
        """Per upstream variable, with the variable set to its reference on every training row:
        the number of rows whose on/off state flips (switch drivers), and the SD over the rows
        where the unit is on of the change in its pre-activation (strength modifiers)."""
        vs, _ = self.anc_vars([node])
        z0 = self.zfun(node, self.m.X)
        flips, mods = {}, {}
        for v in vs:
            _, enc, _ = self.reference(v)
            X = self.m.X.copy()
            for c, z in enc.items():
                X[:, c] = z
            z2 = self.zfun(node, X)
            flips[v.base] = int(np.sum((z2 > 0) != on))
            mods[v.base] = float(np.std((z0 - z2)[on])) if on.sum() > 1 else 0.0
        top = max(flips.values()) if flips else 0
        floor = max(0.5 * top, min(MIN_FLIP, top), 1)
        drv = [b for b, f in sorted(flips.items(), key=lambda kv: -kv[1]) if f >= floor]
        tot = sum(mods.values())
        mod = [b for b, x in sorted(mods.items(), key=lambda kv: -kv[1])
               if tot > 0 and x / tot >= MIN_MOD_SHARE and b not in drv]
        return drv, mod, flips, {b: (x / tot if tot > 0 else 0.0) for b, x in mods.items()}

    def core_rows(self, lin_tr, keys):
        """The fewest training rows carrying half of the total |change in predicted risk| (full
        minus this unit linearised), and their ranges on the key variables."""
        d = np.abs(sigmoid(self.full_tr) - sigmoid(lin_tr))
        if d.sum() <= 0:
            return 0, "", np.zeros(len(d), bool)
        order = np.argsort(-d)
        k = int(np.searchsorted(np.cumsum(d[order]), 0.5 * d.sum()) + 1)
        core = np.zeros(len(d), bool)
        core[order[:k]] = True
        desc = []
        for b in keys:
            v = self.m.var_by_base[b]
            if v.is_category():
                codes, cnt = np.unique(np.asarray(v.raw_train)[core].astype(str), return_counts=True)
                j = np.argmax(cnt)
                desc.append(f"{b}: {cnt[j]}/{k} {v.level(codes[j])['label']}")
            else:
                x = np.asarray(v.raw_train, float)[core]
                desc.append(f"{b} {np.min(x):g}–{np.max(x):g} (median {np.median(x):g})")
        return k, "; ".join(desc), core

    def bands(self, lin_tr, regime, keys):
        """Data-support cells: the regime split at the training quartiles of its first numeric
        driver or modifier (age first): n and mean |change in predicted risk| per band."""
        nums = [b for b in keys if not self.m.var_by_base[b].is_category()]
        if not nums:
            return "", []
        b = "age" if "age" in nums else nums[0]
        x = np.asarray(self.m.var_by_base[b].raw_train, float)
        q = np.percentile(x, [25, 50, 75])
        d = np.abs(sigmoid(self.full_tr) - sigmoid(lin_tr))
        edges = [-np.inf, *q, np.inf]
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = regime & (x >= lo) & (x < hi)
            lab = (f"{b} < {hi:g}" if lo == -np.inf else f"{b} ≥ {lo:g}" if hi == np.inf else f"{lo:g} ≤ {b} < {hi:g}")
            out.append({"band": lab, "n": int(idx.sum()),
                        "mean_abs_risk_change": float(np.mean(d[idx])) if idx.any() else 0.0})
        return b, out

    def tree_rule(self, regime, drv_vars):
        """Depth-2 tree on the raw columns of the drivers: the regime as a rule in real units.
        Returns (rule, agreement, leaves, thresholds per numeric variable)."""
        m = self.m
        vs = [m.var_by_base[b] for b in drv_vars] or m.variables
        cols = [c for v in vs for c in v.columns]
        R = m.raw_train[:, cols]
        if regime.sum() == 0 or regime.all():
            return "", 0.0, [], {}
        t = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=0).fit(R, regime)
        tr = t.tree_
        leaves, thresholds = [], {}

        def text(conds):
            out = []
            num = {}
            for c, thr, left in conds:
                name = m.feature_names[c]
                if ":" in name:
                    fl = feature_label(self.cb, name)
                    out.append(f"{m.base_of_col[c]} {'≠' if left else '='} {fl['label']}")
                else:
                    lo, hi = num.get(c, (None, None))
                    num[c] = (lo, thr) if left else (thr, hi)
            for c, (lo, hi) in num.items():
                v = m.var_by_base[m.base_of_col[c]]
                u = f" {v.unit}" if v.unit else ""
                if lo is not None and hi is not None:
                    out.append(f"{lo:g} ≤ {v.base} < {hi:g}{u}")
                elif hi is not None:
                    out.append(f"{v.base} < {hi:g}{u}")
                else:
                    out.append(f"{v.base} ≥ {lo:g}{u}")
            return " and ".join(out) or "all"

        def walk(node, conds):
            if tr.children_left[node] == -1:
                idx = t.apply(R) == node
                leaves.append({"rule": text(conds), "n": int(idx.sum()), "n_regime": int(regime[idx].sum())})
                return
            c, thr = cols[tr.feature[node]], float(tr.threshold[node])
            if ":" not in m.feature_names[c]:
                thresholds.setdefault(m.base_of_col[c], []).append(round(thr, 2))
            walk(tr.children_left[node], conds + [(c, thr, True)])
            walk(tr.children_right[node], conds + [(c, thr, False)])

        walk(0, [])
        pred = t.predict(R).astype(bool)
        reg_leaves = [l for l in leaves if l["n_regime"] * 2 > l["n"]]
        rule = " OR ".join(f"({l['rule']}: {l['n_regime']}/{l['n']})" for l in reg_leaves) or "(no leaf is mostly regime)"
        return rule, float(np.mean(pred == regime)), leaves, thresholds

    def effect(self, lin_tr, regime):
        d = sigmoid(self.full_tr) - sigmoid(lin_tr)
        r = regime if regime.any() else np.ones(len(d), bool)
        return {"mean_abs_risk_change_regime": float(np.mean(np.abs(d[r]))),
                "mean_signed_risk_change_regime": float(np.mean(d[r])),
                "max_abs_risk_change": float(np.max(np.abs(d))),
                "mean_abs_risk_change_all": float(np.mean(np.abs(d)))}

    def aucs(self, lin_tr, lin_te, prefix="gain"):
        m = self.m
        f_tr, f_te = default_auc(m.y, self.full_tr), default_auc(m.yt, self.full_te)
        lo, hi = delta_ci(m.yt, self.full_te, lin_te, self.boot)
        return {f"{prefix}_train": f_tr - default_auc(m.y, lin_tr),
                f"{prefix}_test": f_te - default_auc(m.yt, lin_te),
                f"{prefix}_test_lo": lo, f"{prefix}_test_hi": hi}

    # ------------------------------------------------------------------ rows
    def unit_rows(self):
        m, rows = self.m, []
        names = list(self.units)
        for b, ns in self.units.items():
            n0 = ns[0]
            z, zt = self.pre[n0], self.pre_t[n0]
            on, on_t = z > 0, zt > 0
            side_on = on.sum() * 2 <= len(on)
            regime = on if side_on else ~on
            regime_t = on_t if side_on else ~on_t
            lin_tr = m.ev.logit(m.X, self.lin.overrides([b]))
            lin_te = m.ev.logit(m.Xt, self.lin.overrides([b]))
            rest_tr = m.ev.logit(m.X, self.lin.overrides([u for u in names if u != b]))
            rest_te = m.ev.logit(m.Xt, self.lin.overrides([u for u in names if u != b]))
            drv, mod, flips, mshare = self.drivers(n0, on)
            rule, agree, leaves, thr = self.tree_rule(regime, drv)
            n_core, core_desc, _ = self.core_rows(lin_tr, drv + mod)
            band_var, bands = self.bands(lin_tr, regime, drv + mod)
            ps = self.unit_part(ns)
            p = ps[0] if ps else None
            title, plain, hyp = self.part_info(p) if p else ("", "", "")
            r, share = resid_of(self.vals[n0], m.X)
            row = dict(network=self.label, level="unit", unit=b, nodes=" ".join(ns),
                       annotation=p.name if p else "", annotation_id=p.id if p else "",
                       other_annotations=" ".join(q.name for q in ps[1:]),
                       title=title, plain_language=plain, hypothesis=hyp,
                       inputs="; ".join(self.input_names(n0)),
                       n_active=int(on.sum()), n_active_test=int(on_t.sum()),
                       regime_side="on" if side_on else "off", n_regime=int(regime.sum()),
                       n_regime_test=int(regime_t.sum()),
                       regime_rule=rule, regime_rule_agreement=agree,
                       cells=json.dumps(leaves), drivers=" ".join(drv), modifiers=" ".join(mod),
                       modifier_shares=json.dumps({k: round(v, 3) for k, v in mshare.items()}),
                       n_core=n_core, core_rows=core_desc, band_variable=band_var,
                       bands=json.dumps(bands), thresholds=json.dumps(thr),
                       upstream_inputs="; ".join(self.upstream_names([n0])),
                       flips=json.dumps(flips), nonlinear_share=share,
                       exactly_affine_on_data=bool(np.max(np.abs(r)) < 1e-9),
                       **self.aucs(lin_tr, lin_te, "gain"),
                       **{k.replace("gain", "restore_onto_linear"): v for k, v in
                          self._restore(rest_tr, rest_te).items()},
                       **self.effect(lin_tr, regime))
            off = {x: ((), (lambda vals, k=len(m.X): np.zeros(k))) for x in ns}
            on_eff = (self.full_tr - m.ev.logit(m.X, off))[on]
            row["on_effect_logit"] = float(np.mean(on_eff)) if on.any() else 0.0
            row["on_direction"] = "raises" if row["on_effect_logit"] > 0 else "lowers"
            row["_resids"] = [r]
            rows.append(row)
        return rows

    def _restore(self, rest_tr, rest_te):
        m = self.m
        lo, hi = delta_ci(m.yt, rest_te, self.all_te, self.boot)
        return {"gain_train": default_auc(m.y, rest_tr) - default_auc(m.y, self.all_tr),
                "gain_test": default_auc(m.yt, rest_te) - default_auc(m.yt, self.all_te),
                "gain_test_lo": lo, "gain_test_hi": hi}

    def part_rows(self, unit_rows):
        m, rows = self.m, []
        nl = [p for p in self.parts if not p.is_root and not p.duplicate_of
              and part_class(m, p, self.kinds) == "nonlinear"]
        plin = Linearisation(m.ev, {p.name: p.unit for p in nl}, m.X)
        by_unit = {r["unit"]: r for r in unit_rows}
        for p in nl:
            inside = {str(x) for x in p.unit.get("subgraph_nodes", [])} | {str(x) for x in p.exits}
            inside -= {str(x) for x in p.unit.get("entry_nodes", [])}
            us = [b for b, ns in self.units.items() if any(n in inside for n in ns)]
            try:
                lin_tr = m.ev.logit(m.X, plin.overrides([p.name]))
                lin_te = m.ev.logit(m.Xt, plin.overrides([p.name]))
                how = "part (entries -> exits)"
            except ValueError:   # the part's entries depend on its exits: not a DAG as one unit
                lin_tr = m.ev.logit(m.X, self.lin.overrides(us))
                lin_te = m.ev.logit(m.Xt, self.lin.overrides(us))
                how = "its units, each on its own inputs (part not linearisable as one unit: cycle)"
            regime = np.zeros(len(m.X), bool)
            regime_t = np.zeros(len(m.Xt), bool)
            on = np.zeros(len(m.X), bool)
            for b in us:
                n0 = self.units[b][0]
                o, ot = self.pre[n0] > 0, self.pre_t[n0] > 0
                side_on = by_unit[b]["regime_side"] == "on"
                regime |= o if side_on else ~o
                regime_t |= ot if side_on else ~ot
                on |= o
            dom = max(us, key=lambda b: by_unit[b]["gain_train"]) if us else None
            drv = by_unit[dom]["drivers"].split() if dom else []
            mod = by_unit[dom]["modifiers"].split() if dom else []
            n_core, core_desc, _ = self.core_rows(lin_tr, drv + mod)
            rule = " | ".join(f"{b}: {by_unit[b]['regime_side']} for {by_unit[b]['n_regime']}" for b in us)
            agree, leaves, flips = float("nan"), [], {}
            title, plain, hyp = self.part_info(p)
            resids, shares = [], []
            for x in p.exits:
                r, s = resid_of(self.vals[x], m.X)
                resids.append(r)
                shares.append(s)
            ins = sorted({feature_label(self.cb, m.feature_names[input_column(e)])["raw"] + " · " +
                          feature_label(self.cb, m.feature_names[input_column(e)])["label"]
                          for e in p.input_nodes})
            row = dict(network=self.label, level="part", unit=" ".join(us), nodes=" ".join(p.exits),
                       annotation=p.name, annotation_id=p.id, other_annotations=" ".join(p.children),
                       title=title, plain_language=plain, hypothesis=hyp,
                       inputs="; ".join(ins) + ("; upstream: " + " ".join(p.upstream_nodes) if p.upstream_nodes else ""),
                       n_active=int(on.sum()), n_active_test=int(sum(
                           (self.pre_t[self.units[b][0]] > 0) for b in us).astype(bool).sum()) if us else 0,
                       regime_side="union of units", n_regime=int(regime.sum()),
                       n_regime_test=int(regime_t.sum()), regime_rule=rule,
                       regime_rule_agreement=agree, cells=json.dumps(leaves), drivers=" ".join(drv),
                       modifiers=" ".join(mod), modifier_shares="{}", n_core=n_core, core_rows=core_desc,
                       band_variable="", bands="[]",
                       thresholds=by_unit[dom]["thresholds"] if dom else "{}",
                       dominant_unit=dom or "", unit_families="",
                       upstream_inputs="; ".join(self.upstream_names([n for b in us for n in self.units[b]])),
                       flips=json.dumps(flips), nonlinear_share=max(shares) if shares else 0.0,
                       exactly_affine_on_data=False, linearised_as=how,
                       **self.aucs(lin_tr, lin_te, "gain"), **self.effect(lin_tr, regime))
            row["_resids"] = resids
            rows.append(row)
        return rows

    def network_row(self):
        m, st = self.m, self.model_stmt
        f_tr, f_te = default_auc(m.y, self.full_tr), default_auc(m.yt, self.full_te)
        l_tr, l_te = default_auc(m.y, self.all_tr), default_auc(m.yt, self.all_te)
        lo, hi = delta_ci(m.yt, self.full_te, self.all_te, self.boot)
        return dict(network=self.label, genome_id=self.gid, explanation_id=self.meta["explanation_id"],
                    switching_nodes=len(self.sw_nodes), nonlinear_units=len(self.units),
                    binary_step_units_excluded=len(self.bs_nodes),
                    linear_share=float(st["variance_share_linear"]),
                    auc_full_train=f_tr, auc_linearised_train=l_tr,
                    auc_full_test=f_te, auc_linearised_test=l_te,
                    gain_train=f_tr - l_tr, gain_test=f_te - l_te, gain_test_lo=lo, gain_test_hi=hi)

    def ame(self, v, code=None, ref=None):
        """Average effect on the score over the training rows: per raw unit for a number (finite
        difference, 0.01 SD), level minus reference for a category."""
        m = self.m
        if v.is_category():
            Xa, Xb = m.X.copy(), m.X.copy()
            for c, z in v.encode(code).items():
                Xa[:, c] = z
            for c, z in v.encode(ref).items():
                Xb[:, c] = z
            return float(np.mean(m.ev.logit(Xa) - m.ev.logit(Xb)))
        c, h = v.columns[0], 0.01
        X1 = m.X.copy()
        X1[:, c] += h
        return float(np.mean(m.ev.logit(X1) - self.full_tr) / h / v.scale[0])

    def linear_presence(self, Phi, terms, beta_data):
        """Per linear-core term: the term's logit contribution (weight x input, the weight from the
        projection of this network's logit on the linear columns; zero when the network reads no
        column of the variable) and its presence (``explaneat/analysis/signal_presence.py``)."""
        _, beta = ols(Phi, self.full_tr)
        out, comps = {}, {}
        for term, j in terms:
            c = beta[j] * (Phi[:, j] - Phi[:, j].mean())
            reads = self.vinfo.get(term.split("=")[0]) is not None
            pr = presence(c, reads=reads)
            comps[term] = c
            out[term] = dict(proj_coef=float(beta[j]), sd_contribution_raw=pr["sd_raw"],
                             sd_contribution=pr["sd"], presence=pr["state"], rare_only=pr["rare_only"],
                             reads=reads, data_coef=float(beta_data[j]))
        return out, comps

    def family_presence(self, units):
        """Logit contribution of a set of switching units' non-linear part: the full logit minus
        the logit with those units replaced by their affine fit on their own inputs."""
        c = self.full_tr - self.m.ev.logit(self.m.X, self.lin.overrides(units))
        return presence(c)

    def linear_core(self):
        m, rows = self.m, []
        for v in m.variables:
            info = self.vinfo.get(v.base)
            cls = info["class"] if info else "not used"
            coef = {c: (info["coef_raw"].get(c, 0.0) if info else 0.0) for c in v.columns}
            if cls == "nonlinear":
                coef = {c: 0.0 for c in v.columns}
            if v.is_category():
                ref, _, ref_lab = self.reference(v)
                rc = coef[v.columns[v.codes.index(ref)]]
                for c, code in zip(v.columns, v.codes):
                    if code == ref:
                        continue
                    rows.append(dict(network=self.label, variable=v.base, term=f"{v.base}={code}",
                                     label=f"{v.level(code)['label']} vs {ref_lab}", unit="level vs reference",
                                     var_class=cls, coef=coef[c] - rc, avg_effect=self.ame(v, code, ref),
                                     linear_path=bool(info and cls != "nonlinear" and
                                                      (coef[c] != 0 or rc != 0))))
            else:
                c = v.columns[0]
                rows.append(dict(network=self.label, variable=v.base, term=v.base, label=v.short,
                                 unit=f"per {v.unit or 'unit'}", var_class=cls, coef=coef[c], avg_effect=self.ame(v),
                                 linear_path=bool(info and cls != "nonlinear" and coef[c] != 0)))
        return rows


def lr_core(m):
    """Reference only: L2 logistic regression (C=1) on the standardised training rows."""
    lr = LogisticRegression(C=1.0, max_iter=10000).fit(m.X, m.y)
    w = lr.coef_[0] / m.scale
    out = {}
    for v in m.variables:
        if v.is_category():
            codes, cnt = np.unique(np.asarray(v.raw_train).astype(str), return_counts=True)
            ref = codes[np.argmax(cnt)]
            rc = w[v.columns[v.codes.index(ref)]]
            for c, code in zip(v.columns, v.codes):
                if code != ref:
                    out[f"{v.base}={code}"] = float(w[c] - rc)
        else:
            out[v.base] = float(w[v.columns[0]])
    return out


def linear_basis(m, X):
    """The linear signal columns, one per linear-core term (the rows of ``linear_core.csv``): a
    number's own column, a category's non-reference levels (reference: the most common training
    level). Returns (Phi, [(term, column index in Phi)])."""
    cols, terms = [], []
    for v in m.variables:
        if v.is_category():
            codes, cnt = np.unique(np.asarray(v.raw_train).astype(str), return_counts=True)
            ref = codes[np.argmax(cnt)]
            for c, code in zip(v.columns, v.codes):
                if code != ref:
                    terms.append((f"{v.base}={code}", len(cols)))
                    cols.append(X[:, c])
        else:
            terms.append((v.base, len(cols)))
            cols.append(X[:, v.columns[0]])
    return np.column_stack(cols), terms


def ols(Phi, e):
    """Least-squares projection of a logit on [1, Phi] (training rows): (intercept, coefficients)."""
    A = np.hstack([np.ones((len(Phi), 1)), Phi])
    c, *_ = np.linalg.lstsq(A, e, rcond=None)
    return float(c[0]), c[1:]


def data_linear_fit(Phi, y):
    """The data's own linear signals: ridge logistic fit of the labels on Phi (standardised columns,
    C from {0.01, 0.1, 1} by 5-fold CV log-loss, StratifiedKFold(5, shuffle, random_state=0)), as
    in the Adult and Churn catalogues."""
    sd = Phi.std(axis=0)
    sd[sd == 0] = 1

    def fit(P, yy, C):
        lr = LogisticRegression(C=C, max_iter=10000, tol=1e-8).fit(P / sd, yy)
        return lr.intercept_[0], lr.coef_[0] / sd

    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Phi, y))
    score = {}
    for C in (0.01, 0.1, 1.0):
        ll = []
        for tr, va in folds:
            a, b = fit(Phi[tr], y[tr], C)
            ll.append(log_loss(y[va], 1 / (1 + np.exp(-(a + Phi[va] @ b)))))
        score[C] = float(np.mean(ll))
    C = min(score, key=score.get)
    return fit(Phi, y, C)[1], C


def row_family(group):
    """The tick-table row of a family group: every resting-BP family is one row."""
    return "BP" if group.startswith("BP") else group


def family_of(drivers, modifiers=()):
    """The switch drivers' tags (BP, thal, exang first), plus "x age" when age drives the switch
    or modifies its strength. Other strength modifiers are listed in their own column."""
    tags = [TAG.get(b, b) for b in drivers]
    age = "age" in tags or "age" in modifiers
    if not tags:
        return "age" if age else "none"
    prim = [t for t in PRIMARY if t in tags]
    others = sorted(t for t in tags if t not in PRIMARY and t != "age")
    core = prim + others
    if not core:
        return "age"
    return " × ".join(core) + (" × age" if age else "")


def family_group(fam):
    """Coarse family: the BP / thal / exang gates (with or without age), age alone, or the
    first other driver."""
    tags = fam.split(" × ")
    prim = [t for t in PRIMARY if t in tags]
    if prim:
        return " × ".join(prim) + (" × age" if "age" in tags else "")
    if tags == ["age"]:
        return "age"
    if "age" in tags:
        return f"{tags[0]} × age"
    return tags[0]


def main():
    if "explaneat_v4" not in os.environ.get("DATABASE_URL", ""):
        sys.exit("set DATABASE_URL=postgresql://localhost/explaneat_v4 (read-only use)")
    if not os.environ.get("EXPLANEAT_RECORDS_DIR"):
        sys.exit("set EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4")
    db.init_db()
    with ReadOnlySession() as s:
        data = load_split(s, SPLIT)
    boot = bootstrap_indices(len(data["y_test"]))
    h40 = np.load(HERE / "cache/heart40.npz")
    h40m = json.loads((HERE / "cache/heart40.json").read_text())
    sha = hashlib.sha1(np.ascontiguousarray(data["X_train"]).tobytes()).hexdigest()
    assert sha == h40m["X_train_sha"], "40-generation cache was built on other training rows"
    h40_B = {k: h40[f"B_{k}"] for k in h40m["seeds"]}
    h40_R = {s["key"]: np.column_stack([h40[f"R_{s['key']}:{j}"] for j in range(s["n_exits"])])
             for s in h40m["signals"]}
    nets, feats, netrows, core = {}, [], [], []
    for lab in LABELS:
        print("loading", lab, flush=True)
        nt = Net(lab, data, boot)
        nets[lab] = nt
        ur = nt.unit_rows()
        pr = nt.part_rows(ur)
        feats += ur + pr
        netrows.append(nt.network_row())
        core += nt.linear_core()
        print(f"  {len(ur)} units, {len(pr)} parts", flush=True)
    m0 = nets[LABELS[0]].m
    lr = lr_core(m0)
    # ------------------------------------------------------------ presence (shared rule)
    Phi, terms = linear_basis(m0, m0.X)
    beta_data, C_data = data_linear_fit(Phi, m0.y)
    lin_pres = {lab: nets[lab].linear_presence(Phi, terms, beta_data) for lab in LABELS}
    for term, _ in terms:   # direction against the consensus of the seeds that use the term
        used_c = [lin_pres[l][1][term] for l in LABELS if lin_pres[l][0][term]["presence"] == "used"]
        cons = consensus(used_c)
        for l in LABELS:
            d = lin_pres[l][0][term]
            d["direction"] = consensus_direction(lin_pres[l][1][term], cons) if d["presence"] == "used" else ""
    for r in core:
        r.update(lin_pres[r["network"]][0][r["term"]])
    # ------------------------------------------------------------ recurrence (train only)
    for f in feats:
        rs = f.pop("_resids")
        f["_r"] = rs
        if f["nonlinear_share"] < MIN_NL_SHARE:
            f.update(seeds_matched=None, seeds_matched_list="", heart40_seeds_matched=None,
                     heart40_best_signal="", heart40_best_r2=None, recurrence="near-linear")
            continue
        others = [l for l in LABELS if l != f["network"]]
        hit = [l for l in others if max(nov(r, nets[l].B) for r in rs) < MIN_NOV]
        h40hit = [k for k, B in h40_B.items() if max(nov(r, B) for r in rs) < MIN_NOV]
        best, best_r2 = "", 0.0
        for k, R in h40_R.items():
            r2 = 1.0 - max(nov(r, R) for r in rs)
            if r2 > best_r2:
                best, best_r2 = k, r2
        f.update(seeds_matched=len(hit), seeds_matched_list=" ".join(l.split("-")[1] for l in hit),
                 heart40_seeds_matched=len(h40hit), heart40_best_signal=best if best_r2 >= 0.5 else "",
                 heart40_best_r2=best_r2, recurrence="recurring" if hit else "seed-specific")
    # unit-to-unit residual matches across Heart-100 seeds (for grouping recurring signals)
    us = [f for f in feats if f["level"] == "unit" and f["recurrence"] != "near-linear"]
    pairs = []
    for i, a in enumerate(us):
        for b in us[i + 1:]:
            if a["network"] == b["network"]:
                continue
            r2 = max(1.0 - nov(a["_r"][0], b["_r"][0][:, None]), 1.0 - nov(b["_r"][0], a["_r"][0][:, None]))
            if r2 >= 0.5:
                pairs.append((a["network"] + ":" + a["unit"], b["network"] + ":" + b["unit"], r2))
    unit_flag = {}
    for f in sorted(feats, key=lambda f: f["level"] != "unit"):   # units first, parts use them
        f.pop("_r")
        g = f["gain_test_lo"], f["gain_test_hi"]
        test = ("generalises" if g[0] > 0 else "worse on test" if g[1] < 0 else "no measurable test effect")
        if f["level"] == "unit":
            cells = json.loads(f["cells"])
            reg_cells = [c for c in cells if c["n_regime"] * 2 > c["n"]]
            small = f["n_regime"] < SMALL_N or bool(reg_cells and max(c["n_regime"] for c in reg_cells) < SMALL_N)
            f["family"] = family_of(f["drivers"].split(), f["modifiers"].split())
            unit_flag[(f["network"], f["unit"])] = (small, f["family"])
        else:
            us = f["unit"].split()
            small = unit_flag[(f["network"], f["dominant_unit"])][0] if f["dominant_unit"] else True
            f["family"] = unit_flag[(f["network"], f["dominant_unit"])][1] if f["dominant_unit"] else "none"
            f["unit_families"] = "; ".join(sorted({unit_flag[(f["network"], u)][1] for u in us}))
        f["small_support"] = bool(small)
        f["effect_on_few"] = bool(0 < f["n_core"] < SMALL_N)
        f["family_group"] = family_group(f["family"])
        f["verdict"] = "; ".join([test, f["recurrence"]] + (["small support (n<30)"] if small else []))
        f["test_verdict"] = test
    fam_rows = []
    for lab in LABELS:
        nt = nets[lab]
        fam = {}
        for f in feats:
            if f["network"] == lab and f["level"] == "unit":
                pr = nt.family_presence([f["unit"]])
                f.update(nl_sd_raw=pr["sd_raw"], nl_sd=pr["sd"], presence=pr["state"])
                fam.setdefault(row_family(f["family_group"]), []).append(f["unit"])
        for g, us in sorted(fam.items()):
            pr = nt.family_presence(us)
            fam_rows.append(dict(network=lab, seed=int(lab.rsplit("-s", 1)[1]), row_family=g,
                                 units=" ".join(us), sd_raw=pr["sd_raw"], sd=pr["sd"], presence=pr["state"]))
    pd.DataFrame(fam_rows).to_csv(HERE / "family_presence.csv", index=False)
    F = pd.DataFrame(feats)
    order = ["network", "level", "annotation", "title", "unit", "nodes", "family_group", "family", "drivers", "modifiers",
             "thresholds", "inputs", "upstream_inputs", "exactly_affine_on_data",
             "n_active", "regime_side", "n_regime", "regime_rule", "regime_rule_agreement", "cells",
             "n_core", "core_rows", "band_variable", "bands",
             "gain_train", "gain_test", "gain_test_lo", "gain_test_hi",
             "restore_onto_linear_train", "restore_onto_linear_test", "restore_onto_linear_test_lo",
             "restore_onto_linear_test_hi", "mean_abs_risk_change_regime", "mean_signed_risk_change_regime",
             "max_abs_risk_change", "mean_abs_risk_change_all", "nonlinear_share", "seeds_matched",
             "seeds_matched_list", "heart40_seeds_matched", "heart40_best_signal", "heart40_best_r2",
             "recurrence", "small_support", "test_verdict", "verdict", "n_active_test", "n_regime_test",
             "annotation_id", "other_annotations", "plain_language", "hypothesis", "flips"]
    F = F[order + [c for c in F.columns if c not in order]]
    F.to_csv(HERE / "features.csv", index=False)
    N = pd.DataFrame(netrows)
    parts_nl = F[F.level == "part"].groupby("network").size()
    N["nonlinear_parts"] = N.network.map(parts_nl).fillna(0).astype(int)
    N["share_of_train_gain_lost_on_test"] = 1 - N.gain_test / N.gain_train
    N.to_csv(HERE / "networks.csv", index=False)
    C = pd.DataFrame(core)
    C["lr_reference"] = C.term.map(lr)
    C.to_csv(HERE / "linear_core.csv", index=False)
    pd.DataFrame(pairs, columns=["a", "b", "r2"]).to_csv(HERE / "recurrence.csv", index=False)
    write_tables(F, N, C, pairs)


def write_tables(F, N, C, pairs):
    L = []
    L.append("## Networks\n")
    L.append("| network | non-linear units (nodes) | binary-step excluded | non-linear parts | linear share | "
             "AUC train full → linearised | AUC test full → linearised (TEST) | train gain | test gain [95% CI] (TEST) | share of train gain lost on test |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in N.itertuples():
        L.append(f"| {r.network} | {r.nonlinear_units} ({r.switching_nodes}) | {r.binary_step_units_excluded} | "
                 f"{r.nonlinear_parts} | {r.linear_share:.0%} | {r.auc_full_train:.3f} → {r.auc_linearised_train:.3f} | "
                 f"{r.auc_full_test:.3f} → {r.auc_linearised_test:.3f} | {r.gain_train:+.3f} | "
                 f"{r.gain_test:+.3f} [{r.gain_test_lo:+.3f}, {r.gain_test_hi:+.3f}] | "
                 f"{r.share_of_train_gain_lost_on_test:.0%} |")
    L.append(f"| mean | {N.nonlinear_units.mean():.1f} | {N.binary_step_units_excluded.sum()} total | "
             f"{N.nonlinear_parts.mean():.1f} | {N.linear_share.mean():.0%} | {N.auc_full_train.mean():.3f} → "
             f"{N.auc_linearised_train.mean():.3f} | {N.auc_full_test.mean():.3f} → {N.auc_linearised_test.mean():.3f} | "
             f"{N.gain_train.mean():+.3f} | {N.gain_test.mean():+.3f} | {1 - N.gain_test.mean() / N.gain_train.mean():.0%} |\n")
    for lvl in ("unit", "part"):
        G = F[F.level == lvl]
        L.append(f"## Per {lvl} (train unless marked TEST)\n")
        L.append("| network | annotation (title) | unit | family | drivers | n regime (side) | regime rule (agreement) | "
                 "gain train | gain TEST [95% CI] | mean / max \\|Δrisk\\| | dir | seeds matched (40-gen) | verdict |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in G.itertuples():
            sm = "–" if pd.isna(r.seeds_matched) else f"{int(r.seeds_matched)} ({int(r.heart40_seeds_matched)})"
            L.append(f"| {r.network.replace('heart100-', '')} | {r.annotation} ({r.title}) | {r.unit} | {r.family} | "
                     f"{r.drivers} | {r.n_regime} ({r.regime_side}) | {r.regime_rule} ({r.regime_rule_agreement:.0%}) | "
                     f"{r.gain_train:+.4f} | {r.gain_test:+.4f} [{r.gain_test_lo:+.3f}, {r.gain_test_hi:+.3f}] | "
                     f"{r.mean_abs_risk_change_regime:.3f} / {r.max_abs_risk_change:.3f} | "
                     f"{'+' if r.mean_signed_risk_change_regime > 0 else '−'} | {sm} | {r.verdict} |")
        L.append("")
    L.append("## Linear core (linear-path coefficient, log-odds per real unit; categories vs the most common level)\n")
    L.append("| term | unit | networks with a linear path | + / − | median | min | max | LR reference (train) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for term, g in C.groupby("term", sort=False):
        h = g[g.linear_path]
        if len(h):
            L.append(f"| {term} ({g.label.iloc[0]}) | {g.unit.iloc[0]} | {len(h)} | {(h.coef > 0).sum()} / {(h.coef < 0).sum()} | "
                     f"{h.coef.median():+.3g} | {h.coef.min():+.3g} | {h.coef.max():+.3g} | {g.lr_reference.iloc[0]:+.3g} |")
        else:
            L.append(f"| {term} ({g.label.iloc[0]}) | {g.unit.iloc[0]} | 0 | – | – | – | – | {g.lr_reference.iloc[0]:+.3g} |")
    L.append(f"\nUnit-to-unit residual matches across seeds (R² ≥ 0.5): {len(pairs)}\n")
    for a, b, r2 in pairs:
        L.append(f"- {a} ~ {b} (R² {r2:.2f})")
    (HERE / "tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
