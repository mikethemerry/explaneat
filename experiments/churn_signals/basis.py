"""The signal coordinates: a fixed basis of data functions, grouped into signal blocks.

A **signal** is a target-relevant function of the inputs, defined in data terms and
independent of any model. Here every candidate signal is a *block* of columns of one fixed
design matrix ``Phi`` built from the data alone (train quantiles for knots; nothing from any
network). A model's realisation of signal ``s`` is the component ``Phi_s beta_s`` of the
least-squares projection of the model's score (logit) onto ``Phi``; the data's own version of
``s`` is the same component of a ridge logistic fit of the churn labels on ``Phi`` (train).

Aliases are merged before anything else: each ``total X charge`` column is an exact multiple of
``total X minutes`` (identical after z-scoring up to rounding), so each usage variable enters
as the mean of its two z-columns.

Blocks (hierarchically orthogonalised on the training rows, so attribution is well defined):

1. **linear**: one block per data variable (day, eve, night, intl usage; the four call counts;
   service calls; account length; international plan; voice mail = plan + messages; area
   code; state). Their coefficients are the model's best linear approximation (direct
   coefficient comparison).
2. **shape** (main-effect bends): piecewise-linear hinges at fixed train quantiles (integer
   knots for counts), residualised on all linear columns, so a shape block holds only the
   part of a one-variable effect that is not a straight line.
3. **interaction** blocks for the pairs that the ten explanations' units co-read (tensor
   products of coarse hinge bases), residualised on all linear and shape columns.

Whatever the blocks cannot express (three-way gates, state x usage gates) is the residual,
reported as "unassigned".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

USAGE = {"day": ["total day minutes", "total day charge"],
         "eve": ["total eve minutes", "total eve charge"],
         "night": ["total night minutes", "total night charge"],
         "intl": ["total intl minutes", "total intl charge"]}
COUNTS = {"csc": "number customer service calls", "intlcalls": "total intl calls",
          "daycalls": "total day calls", "evecalls": "total eve calls",
          "nightcalls": "total night calls"}
OTHER_NUM = {"acct": "account length"}
CONT_Q = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
DAY_Q = [0.10, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98]
COARSE_Q = [0.50, 0.75, 0.90]

# Human names of the blocks (signal ids), in catalogue order.
NAMES = {
    "lin:csc": "service calls (linear slope)",
    "lin:day": "day usage (linear slope)",
    "lin:eve": "evening usage (linear slope)",
    "lin:night": "night usage (linear slope)",
    "lin:intl": "international minutes (linear slope)",
    "lin:intlcalls": "international calls (linear slope)",
    "lin:intlplan": "international plan (offset)",
    "lin:vmail": "voice mail (plan and messages, offset/slope)",
    "lin:acct": "account length (linear slope)",
    "lin:area": "area code (offsets)",
    "lin:state": "state (offsets)",
    "lin:daycalls": "day calls (linear slope)",
    "lin:evecalls": "evening calls (linear slope)",
    "lin:nightcalls": "night calls (linear slope)",
    "shape:csc": "service-call threshold (bend at about 4 calls)",
    "shape:day": "heavy day usage hinge",
    "shape:eve": "evening usage bend",
    "shape:night": "night usage bend",
    "shape:intl": "international minutes bend",
    "shape:intlcalls": "international calls bend",
    "shape:acct": "account length bend",
    "shape:daycalls": "day calls bend",
    "shape:evecalls": "evening calls bend",
    "shape:nightcalls": "night calls bend",
    "int:csc*day": "service calls x low day usage",
    "int:day*eve": "day + evening usage (combined bill)",
    "int:day*vmail": "heavy day usage without voice mail",
    "int:day*night": "day x night usage",
    "int:eve*csc": "evening usage x service calls",
    "int:csc*vmail": "service calls x voice mail",
    "int:csc*intlplan": "service calls x international plan",
    "int:intlplan*intl": "international plan x international usage (minutes, calls)",
}


def relu(x):
    return np.maximum(x, 0.0)


@dataclass
class Basis:
    """Phi on train and test, with block column indices and the train-fitted transforms.
    ``phi(X)`` evaluates the same (orthogonalised) columns on any z-scored input matrix."""
    names: List[str]
    blocks: Dict[str, List[int]]
    Phi_tr: np.ndarray
    Phi_te: np.ndarray
    raw: Dict[str, np.ndarray] = field(default_factory=dict)       # train raw variable values
    raw_te: Dict[str, np.ndarray] = field(default_factory=dict)
    knots: Dict[str, List[float]] = field(default_factory=dict)    # raw-unit knots
    _builder: object = None
    _steps: list = field(default_factory=list)
    _keep: list = field(default_factory=list)

    def component(self, beta, block, test=False, Phi=None):
        P = Phi if Phi is not None else (self.Phi_te if test else self.Phi_tr)
        idx = self.blocks[block]
        return P[:, idx] @ beta[idx]

    def phi(self, X, raw=None):
        P = self._builder(X, raw)
        one = np.ones((len(P), 1))
        for lo, hi, coef in self._steps:
            A = np.hstack([one, P[:, :lo]])
            P[:, lo:hi] = P[:, lo:hi] - A @ coef
        P[:, :self._n_lin] -= self._mu
        return P[:, self._keep]


def _vars(m, X, raw):
    """Data variables (aliases merged) as z-arrays and raw-unit arrays."""
    col = {n: i for i, n in enumerate(m.feature_names)}
    z, r = {}, {}
    for k, cols in USAGE.items():
        z[k] = np.mean([X[:, col[c]] for c in cols], axis=0)
        r[k] = raw[:, col[cols[0]]]                    # minutes
    for k, c in {**COUNTS, **OTHER_NUM}.items():
        z[k] = X[:, col[c]]
        r[k] = raw[:, col[c]]
    z["intlplan"] = X[:, col["international plan"]]
    r["intlplan"] = raw[:, col["international plan"]]
    z["vmplan"] = X[:, col["voice mail plan"]]
    r["vmplan"] = raw[:, col["voice mail plan"]]
    z["vmmsg"] = X[:, col["number vmail messages"]]
    r["vmmsg"] = raw[:, col["number vmail messages"]]
    area = [c for c in m.feature_names if c.startswith("area code:")]
    state = [c for c in m.feature_names if c.startswith("state:")]
    z["area"] = X[:, [col[c] for c in area]]
    z["state"] = X[:, [col[c] for c in state]]
    r["area"] = raw[:, [col[c] for c in area]]
    r["state"] = raw[:, [col[c] for c in state]]
    return z, r


def build_basis(m) -> Basis:
    """Phi from the PlainModel's data (any of the ten: they share the split)."""
    ztr, rtr = _vars(m, m.X, m.raw_train)
    _, rte = _vars(m, m.Xt, m.raw_test)
    knots: Dict[str, List[float]] = {}

    def zknots(k, qs):
        return [float(np.quantile(ztr[k], q)) for q in qs]

    def lin_map(k):     # raw -> z
        a, b = np.polyfit(rtr[k], ztr[k], 1)
        return a, b

    def to_raw(k, zk):
        a, b = np.polyfit(ztr[k], rtr[k], 1)
        return [float(a * z + b) for z in zk]

    shape_specs = {}
    for k in ["csc", "intlcalls"]:
        ints = [x for x in range(1, 8 if k == "csc" else 9) if (rtr[k] >= x).sum() >= 10]
        a, b = lin_map(k)
        shape_specs[k] = [a * x + b for x in ints]
        knots[f"shape:{k}"] = [float(x) for x in ints]
    shape_specs["day"] = zknots("day", DAY_Q)
    for k in ["eve", "night", "intl", "acct", "daycalls", "evecalls", "nightcalls"]:
        shape_specs[k] = zknots(k, CONT_Q)
    for k, zk in shape_specs.items():
        knots.setdefault(f"shape:{k}", to_raw(k, zk))
    coarse_kn = {k: zknots(k, COARSE_Q) for k in ["day", "eve"]}
    vm_day_kn = zknots("day", [0.50, 0.65, 0.75, 0.90])
    lowday_kn = zknots("day", [0.25, 0.50, 0.75, 0.90])
    csc3, csc4 = (lin_map("csc")[0] * x + lin_map("csc")[1] for x in (3, 4))
    ic3 = lin_map("intlcalls")[0] * 3 + lin_map("intlcalls")[1]
    im13 = lin_map("intl")[0] * 13.0 + lin_map("intl")[1]
    knots["int:intlplan*intl"] = [13.0, 3.0]
    knots["int:csc*day"] = to_raw("day", lowday_kn)

    def build(X, raw=None):
        z, _ = _vars(m, X, X if raw is None else raw)
        cols, names, blocks = [], [], {}

        def add(block, name, a):
            blocks.setdefault(block, []).append(len(names))
            names.append(name)
            cols.append(np.asarray(a, float))

        for k in ["csc", "day", "eve", "night", "intl", "intlcalls"]:
            add(f"lin:{k}", k, z[k])
        add("lin:intlplan", "intlplan", z["intlplan"])
        add("lin:vmail", "vmplan", z["vmplan"])
        add("lin:vmail", "vmmsg", z["vmmsg"])
        add("lin:acct", "acct", z["acct"])
        for j in range(z["area"].shape[1] - 1):
            add("lin:area", f"area{j}", z["area"][:, j])
        for j in range(z["state"].shape[1] - 1):
            add("lin:state", f"state{j}", z["state"][:, j])
        for k in ["daycalls", "evecalls", "nightcalls"]:
            add(f"lin:{k}", k, z[k])
        build.n_lin = len(names)
        for k, zk in shape_specs.items():
            for i, t in enumerate(zk):
                add(f"shape:{k}", f"{k}>{i}", relu(z[k] - t))
        build.n_main = len(names)

        def coarse(k):
            return [z[k]] + [relu(z[k] - t) for t in coarse_kn[k]]

        csc_c = [z["csc"], relu(z["csc"] - csc3), relu(z["csc"] - csc4)]
        lowday = [z["day"]] + [relu(z["day"] - t) for t in lowday_kn]
        vm = [z["vmplan"], z["vmmsg"]]
        ipu = [z["intl"], relu(z["intl"] - im13), z["intlcalls"], relu(ic3 - z["intlcalls"])]
        pairs = {
            "int:csc*day": (csc_c, lowday),
            "int:day*eve": (coarse("day"), coarse("eve")),
            "int:day*vmail": ([z["day"]] + [relu(z["day"] - t) for t in vm_day_kn], vm),
            "int:day*night": (coarse("day"), [z["night"]]),
            "int:eve*csc": (coarse("eve"), csc_c),
            "int:csc*vmail": (csc_c, vm),
            "int:csc*intlplan": (csc_c, [z["intlplan"]]),
            "int:intlplan*intl": ([z["intlplan"]], ipu),
        }
        for blk, (A, Bs) in pairs.items():
            for i in range(len(A)):
                for j in range(len(Bs)):
                    add(blk, f"{blk}[{i},{j}]", A[i] * Bs[j])
        build.names, build.blocks = names, blocks
        return np.column_stack(cols)

    P_tr = build(m.X, m.raw_train)
    names, blocks, n_lin, n_main = build.names, build.blocks, build.n_lin, build.n_main
    steps = []
    one = np.ones((len(P_tr), 1))
    for lo, hi in [(n_lin, n_main), (n_main, len(names))]:
        A = np.hstack([one, P_tr[:, :lo]])
        coef, *_ = np.linalg.lstsq(A, P_tr[:, lo:hi], rcond=None)
        P_tr[:, lo:hi] = P_tr[:, lo:hi] - A @ coef
        steps.append((lo, hi, coef))
    mu = P_tr[:, :n_lin].mean(axis=0)
    P_tr[:, :n_lin] -= mu
    keep = [i for i in range(P_tr.shape[1]) if P_tr[:, i].std() > 1e-8]
    remap = {old: new for new, old in enumerate(keep)}
    blocks2 = {b: [remap[i] for i in idx if i in remap] for b, idx in blocks.items()}
    blocks2 = {b: idx for b, idx in blocks2.items() if idx}
    B = Basis(names=[names[i] for i in keep], blocks=blocks2, Phi_tr=P_tr[:, keep],
              Phi_te=None, raw=rtr, raw_te=rte, knots=knots, _builder=lambda X, raw=None: build(X, raw),
              _steps=steps, _keep=keep)
    B._n_lin, B._mu = n_lin, mu
    B.Phi_te = B.phi(m.Xt, m.raw_test)
    assert np.max(np.abs(B.phi(m.X) - B.Phi_tr)) < 1e-9
    return B


def project(basis: Basis, eta_tr, ridge: float = 1e-8):
    """Least-squares projection of a score on [1, Phi] (train). Returns (intercept, beta)."""
    P = basis.Phi_tr
    A = np.hstack([np.ones((len(P), 1)), P])
    G = A.T @ A + ridge * np.eye(A.shape[1])
    coef = np.linalg.solve(G, A.T @ eta_tr)
    return float(coef[0]), coef[1:]
