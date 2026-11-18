"""Shared loading and helpers for the churn signal-realisation catalogue (read-only).

Every database session is READ ONLY (``compute_runs.ReadOnlySession`` and
``plain_language.load_model`` both issue SET TRANSACTION READ ONLY). Run with

    DATABASE_URL=postgresql://localhost/explaneat_v4 \\
    EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 uv run python ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments/10seed"), str(REPO / "experiments/churn"),
          str(REPO / "experiments/heart100_audit")):
    if p not in sys.path:
        sys.path.insert(0, p)

import audit  # noqa: E402  (experiments/heart100_audit/audit.py: the Net class is dataset-agnostic)
from audit import Net, ReadOnlySession, bootstrap_indices, load_split, db  # noqa: E402,F401

LABELS = [f"churn100-s{i}" for i in range(10)]
SPLIT = "45962ea6-3c78-49f9-a01e-3665f6809b1e"
LAYERS = REPO / "experiments/10seed/plain_language_v4"

# Data variables: input columns merged where the data makes them aliases (section "Aliases").
GROUPS = {
    "day usage": ["total day minutes", "total day charge"],
    "evening usage": ["total eve minutes", "total eve charge"],
    "night usage": ["total night minutes", "total night charge"],
    "international usage": ["total intl minutes", "total intl charge"],
    "voice mail": ["voice mail plan", "number vmail messages"],
    "international plan": ["international plan"],
    "service calls": ["number customer service calls"],
    "international calls": ["total intl calls"],
    "day calls": ["total day calls"],
    "evening calls": ["total eve calls"],
    "night calls": ["total night calls"],
    "account length": ["account length"],
    "area code": ["area code"],
    "state": ["state"],
}
SHORT = {"day usage": "day", "evening usage": "eve", "night usage": "night",
         "international usage": "intl", "voice mail": "vmail", "international plan": "intlplan",
         "service calls": "csc", "international calls": "intlcalls", "day calls": "daycalls",
         "evening calls": "evecalls", "night calls": "nightcalls", "account length": "acct",
         "area code": "area", "state": "state"}
GROUP_OF = {b: g for g, bs in GROUPS.items() for b in bs}


class ChurnNet(Net):
    """The heart100 audit's Net, with one relaxation: split copies of a unit may read from
    different split copies of an upstream unit (churn100-s1), so their predecessor lists are
    not compared; their pre-activations are still checked equal (to 1e-12)."""

    def _units(self):
        m = self.m
        sw = sorted(n for n, k in self.kinds.items() if k == "switching" and n in self.reach)
        self.sw_nodes = sw
        self.bs_nodes = sorted(n for n, k in self.kinds.items() if k == "binary_step" and n in self.reach)
        groups = {}
        for n in sw:
            groups.setdefault(audit.ubase(n), []).append(n)
        for b, ns in groups.items():
            for x in ns[1:]:
                assert np.max(np.abs(self.pre[x] - self.pre[ns[0]])) < 1e-12, (self.label, x)
        self.units = groups
        udefs = {b: {"entry_nodes": [p for p, _ in m.ev.preds[ns[0]]], "exit_nodes": ns}
                 for b, ns in groups.items()}
        self.lin = audit.Linearisation(m.ev, udefs, m.X)
        allu = list(groups)
        self.all_tr = m.ev.logit(m.X, self.lin.overrides(allu))
        self.all_te = m.ev.logit(m.Xt, self.lin.overrides(allu))
        st = self.model_stmt
        assert abs(audit.default_auc(m.y, self.all_tr) - float(st["auc_linearised_train"])) < 1e-5, self.label
        assert abs(audit.default_auc(m.yt, self.all_te) - float(st["auc_linearised_test"])) < 1e-5, self.label
        assert st["switching_units"] == len(sw), self.label


def check_env():
    if "explaneat_v4" not in os.environ.get("DATABASE_URL", ""):
        sys.exit("set DATABASE_URL=postgresql://localhost/explaneat_v4 (read-only use)")
    if not os.environ.get("EXPLANEAT_RECORDS_DIR"):
        sys.exit("set EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4")


def load_all(labels=None):
    check_env()
    db.init_db()
    with ReadOnlySession() as s:
        data = load_split(s, SPLIT)
    boot = bootstrap_indices(len(data["y_test"]))
    nets = {}
    for lab in labels or LABELS:
        print("loading", lab, flush=True)
        nets[lab] = ChurnNet(lab, data, boot)
    return data, boot, nets


def group_columns(m, group):
    return [c for b in GROUPS[group] for c in m.var_by_base[b].columns]


def reference_assign(m, group):
    """Column z-values setting a data variable (all its alias columns together) to its reference:
    the median for numbers, the most common level for categories (no plan, 0 messages)."""
    out = {}
    for b in GROUPS[group]:
        v = m.var_by_base[b]
        if v.is_category():
            codes, cnt = np.unique(np.asarray(v.raw_train).astype(str), return_counts=True)
            out.update(v.encode(codes[np.argmax(cnt)]))
        else:
            out.update(v.encode(float(np.median(np.asarray(v.raw_train, float)))))
    return out


def upstream_groups(nt, nodes):
    _, cols = nt.anc_vars(nodes)
    m = nt.m
    return sorted({GROUP_OF[m.base_of_col[c]] for c in cols}, key=list(GROUPS).index)


def zfun(nt, node, X):
    return nt.zfun(node, X)


def drivers(nt, node, on, groups):
    """Alias-aware switch drivers and strength modifiers (the heart100 audit's rule, with each
    data variable's alias columns moved together): per data variable set to its reference on
    every training row, rows whose on/off state flips, and the SD of the change in the
    pre-activation over the rows where the unit is on."""
    m = nt.m
    z0 = nt.zfun(node, m.X)
    flips, mods = {}, {}
    for g in groups:
        X = m.X.copy()
        for c, z in reference_assign(m, g).items():
            X[:, c] = z
        z2 = nt.zfun(node, X)
        flips[g] = int(np.sum((z2 > 0) != on))
        mods[g] = float(np.std((z0 - z2)[on])) if on.sum() > 1 else 0.0
    top = max(flips.values()) if flips else 0
    floor = max(0.5 * top, min(audit.MIN_FLIP, top), 1)
    drv = [g for g, f in sorted(flips.items(), key=lambda kv: -kv[1]) if f >= floor]
    tot = sum(mods.values())
    mod = [g for g, x in sorted(mods.items(), key=lambda kv: -kv[1])
           if tot > 0 and x / tot >= audit.MIN_MOD_SHARE and g not in drv]
    return drv, mod, flips, {g: (x / tot if tot > 0 else 0.0) for g, x in mods.items()}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
