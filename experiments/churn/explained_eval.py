"""Forward passes over an explained model state with annotation units
linearised, for the `performance_sources` quantities (EXTENSIONS_PLAN.md A, and
the train restore gains used for donor selection in C).

Definitions:
- A *unit* is an annotation used as a function from its entry nodes to its
  exit nodes (a composition is expanded over its descendants first).
- *Linearising* a unit replaces each exit activation by the least-squares
  affine fit of that exit on the unit's entry activations. The fit is made
  once, on the fitting data (train), and then applied unchanged to any data.
- A unit is *nonlinear* if its train fit is not exact (max abs residual above
  ``LINEAR_TOL`` relative to the exit's scale).
- all-linear AUC: every nonlinear unit linearised jointly.
- restore gain of a unit: AUC with every *other* nonlinear unit linearised
  (the unit's own nonlinearity restored alone) minus the all-linear AUC.
  It is 0 for a linear unit by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from explaneat.analysis.composition import annotation_dicts_from_operations, expand_annotation
from explaneat.core.genome_network import NetworkStructure, NodeType

LINEAR_TOL = 1e-7


# ====================================================================== annotations

def annotations_from_ops(operations) -> List[dict]:
    return annotation_dicts_from_operations(operations)


def root_and_children(annotations: List[dict]) -> Tuple[dict, List[dict]]:
    """The single root annotation (not a child of any other) and its direct
    children, in the root's declared order. Children are referenced by name."""
    by_name = {a["name"]: a for a in annotations}
    child_refs = {c for a in annotations for c in a.get("child_annotation_ids", [])}
    roots = [a for a in annotations if a["name"] not in child_refs and a["id"] not in child_refs]
    if len(roots) != 1:
        raise ValueError(f"expected one root annotation, found {[r['name'] for r in roots]}")
    root = roots[0]
    children = []
    for ref in root.get("child_annotation_ids", []):
        if ref not in by_name:
            raise KeyError(f"child {ref!r} of {root['name']!r} not found")
        children.append(by_name[ref])
    return root, children


def leaf_annotations(annotations: List[dict]) -> List[dict]:
    return [a for a in annotations if not a.get("child_annotation_ids")]


def as_unit(annotation: dict, annotations: List[dict], structure: NetworkStructure) -> dict:
    """The annotation as a function of the *raw* entries of its whole subtree.

    For a leaf this is the annotation itself. For a composition, the entries
    are every entry declared in the subtree that is not produced inside it
    (descendants' exits are internal), so linearising a composition linearises
    its children too. (``expand_annotation`` keeps a composition declared over
    its children's exits as a function of those exits, which would leave the
    children's nonlinearity in place.)
    """
    from explaneat.analysis.composition import descendant_annotations

    desc = descendant_annotations(annotation, annotations)
    if not desc:
        return annotation
    expanded = expand_annotation(annotation, annotations, structure)
    members = [annotation] + desc
    produced = {str(x) for a in members for x in a.get("exit_nodes", [])} | {
        str(n) for a in members for n in a.get("subgraph_nodes", [])
        if str(n) not in {str(e) for e in a.get("entry_nodes", [])}}
    entries = []
    for a in [annotation] + desc + [expanded]:
        for e in a.get("entry_nodes", []):
            e = str(e)
            if e not in produced and e not in entries:
                entries.append(e)
    unit = dict(expanded)
    unit["entry_nodes"] = entries
    return unit


# ====================================================================== forward pass

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _input_column(node_id: str) -> int:
    """Data column of a (possibly split / consolidated) input node: -k -> k-1."""
    base = node_id.split("_", 1)[0]
    k = int(base)
    if k >= 0:
        raise ValueError(f"{node_id} is not an input node id")
    return -k - 1


Override = Tuple[Sequence[str], Callable[[Dict[str, np.ndarray]], np.ndarray]]


class StateEvaluator:
    """Node-by-node forward pass over a model-state NetworkStructure with the
    activations NeuralNeat applies: inputs = data columns, identity nodes
    pass through, other hidden nodes ReLU, outputs sigmoid. Supports
    overriding any node's value by a function of other nodes' values."""

    def __init__(self, structure: NetworkStructure):
        self.structure = structure
        self.nodes = {n.id: n for n in structure.nodes}
        self.preds: Dict[str, List[Tuple[str, float]]] = {n: [] for n in self.nodes}
        for c in structure.connections:
            if c.enabled and c.from_node in self.nodes and c.to_node in self.nodes:
                self.preds[c.to_node].append((c.from_node, c.weight))
        self.output_ids = [str(o) for o in structure.output_node_ids]

    def _order(self, extra_deps: Dict[str, Sequence[str]]):
        deps = {n: {p for p, _ in self.preds[n]} | set(extra_deps.get(n, ()))
                for n in self.nodes}
        indeg = {n: len(d) for n, d in deps.items()}
        succ: Dict[str, List[str]] = {n: [] for n in self.nodes}
        for n, d in deps.items():
            for p in d:
                succ[p].append(n)
        queue = sorted(n for n, k in indeg.items() if k == 0)
        order = []
        while queue:
            n = queue.pop()
            order.append(n)
            for m in succ[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if len(order) != len(self.nodes):
            raise ValueError("model state (with overrides) is not a DAG")
        return order

    def forward(self, X, overrides: Optional[Dict[str, Override]] = None) -> Dict[str, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        overrides = overrides or {}
        order = self._order({n: deps for n, (deps, _) in overrides.items()})
        vals: Dict[str, np.ndarray] = {}
        for n in order:
            if n in overrides:
                vals[n] = np.asarray(overrides[n][1](vals), dtype=np.float64)
                continue
            node = self.nodes[n]
            if node.type == NodeType.INPUT:
                vals[n] = X[:, _input_column(n)]
                continue
            if node.type == NodeType.FUNCTION:
                raise NotImplementedError("collapsed FUNCTION nodes are not supported")
            z = np.full(X.shape[0], float(node.bias or 0.0))
            for p, w in self.preds[n]:
                z = z + w * vals[p]
            if node.type == NodeType.OUTPUT:
                vals[n] = _sigmoid(z)
            elif node.activation == "identity":
                vals[n] = z
            else:
                vals[n] = np.maximum(z, 0.0)
        return vals

    def proba(self, X, overrides=None) -> np.ndarray:
        return self.forward(X, overrides)[self.output_ids[0]]


# ====================================================================== linearisation

@dataclass
class UnitFit:
    name: str
    entries: List[str]
    exits: List[str]
    coef: np.ndarray  # (1 + n_entries, n_exits)
    max_abs_residual: float
    exit_scale: float
    r2: List[float]

    @property
    def nonlinear(self) -> bool:
        return self.max_abs_residual > LINEAR_TOL * max(1.0, self.exit_scale)

    def override(self) -> Dict[str, Override]:
        out = {}
        for j, x in enumerate(self.exits):
            c = self.coef[:, j].copy()
            entries = list(self.entries)

            def fn(vals, c=c, entries=entries):
                v = np.full(len(vals[entries[0]]) if entries else 1, c[0])
                for i, e in enumerate(entries):
                    v = v + c[i + 1] * vals[e]
                return v
            out[x] = (entries, fn)
        return out


class Linearisation:
    """Least-squares linear fits of each unit's exits on its entries, made once
    on ``X_fit`` (the unmodified forward pass), reusable on any data."""

    def __init__(self, evaluator: StateEvaluator, units: Dict[str, dict], X_fit):
        self.evaluator = evaluator
        self.units = units
        vals = evaluator.forward(X_fit)
        self.fits: Dict[str, UnitFit] = {}
        for name, u in units.items():
            entries = [str(n) for n in u["entry_nodes"]]
            exits = [str(n) for n in u["exit_nodes"]]
            A = np.column_stack([np.ones(len(X_fit))] + [vals[e] for e in entries])
            Y = np.column_stack([vals[x] for x in exits])
            coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
            resid = Y - A @ coef
            var = Y.var(axis=0)
            r2 = [float(1 - resid[:, j].var() / var[j]) if var[j] > 0 else 1.0
                  for j in range(Y.shape[1])]
            self.fits[name] = UnitFit(name, entries, exits, coef,
                                      float(np.max(np.abs(resid))) if resid.size else 0.0,
                                      float(np.max(np.abs(Y))) if Y.size else 0.0, r2)

    @property
    def nonlinear_units(self) -> List[str]:
        return [n for n in self.units if self.fits[n].nonlinear]

    def overrides(self, names: Iterable[str]) -> Dict[str, Override]:
        out: Dict[str, Override] = {}
        for n in names:
            out.update(self.fits[n].override())
        return out

    def proba(self, X, linearised: Iterable[str] = ()) -> np.ndarray:
        return self.evaluator.proba(X, self.overrides(linearised))


def performance_sources(lin: Linearisation, X, y, keep_predictions: bool = False) -> dict:
    """AUCs on (X, y) for: the full model, every nonlinear unit linearised
    jointly, each unit linearised alone, and each unit restored alone onto the
    all-linear model."""
    nl = lin.nonlinear_units
    p_full = lin.proba(X)
    p_all = lin.proba(X, nl)
    out = {"full_auc": float(roc_auc_score(y, p_full)),
           "all_linear_auc": float(roc_auc_score(y, p_all)),
           "nonlinear_units": nl, "units": {}}
    preds = {"full": p_full, "all_linear": p_all}
    for name in lin.units:
        p_lin = lin.proba(X, [name])
        if name in nl:
            p_rest = lin.proba(X, [n for n in nl if n != name])
            rest_auc = float(roc_auc_score(y, p_rest))
            gain = rest_auc - out["all_linear_auc"]
        else:
            p_rest, rest_auc, gain = p_all, out["all_linear_auc"], 0.0
        f = lin.fits[name]
        out["units"][name] = {
            "nonlinear": f.nonlinear, "max_abs_residual_fit": f.max_abs_residual,
            "min_r2_fit": min(f.r2) if f.r2 else 1.0, "n_entries": len(f.entries),
            "n_exits": len(f.exits),
            "linearised_alone_auc": float(roc_auc_score(y, p_lin)),
            "restore_alone_auc": rest_auc, "restore_gain": gain}
        preds[f"lin:{name}"] = p_lin
        preds[f"rest:{name}"] = p_rest
    if keep_predictions:
        out["predictions"] = preds
    return out


# ====================================================================== bootstrap

def fast_auc(y, p) -> float:
    """ROC AUC via the Mann-Whitney statistic (average ranks for ties)."""
    y = np.asarray(y).ravel().astype(bool)
    n1 = y.sum()
    n0 = len(y) - n1
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _boot_aucs(y, p, idx):
    out = np.full(len(idx), np.nan)
    for b, ix in enumerate(idx):
        yb = y[ix]
        if yb.min() == yb.max():
            continue
        out[b] = fast_auc(yb, p[ix])
    return out


def bootstrap_auc_ci(y, p, idx) -> Tuple[float, float, int]:
    """Percentile 95% CI of AUC over the rows of the index matrix ``idx``
    (single-class resamples skipped)."""
    a = _boot_aucs(np.asarray(y), np.asarray(p), idx)
    a = a[~np.isnan(a)]
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi), int(len(a))


def bootstrap_delta_ci(y, p_a, p_b, idx) -> Tuple[float, float, int]:
    """Paired percentile 95% CI of AUC(p_a) - AUC(p_b) on the same resamples."""
    y = np.asarray(y)
    d = _boot_aucs(y, np.asarray(p_a), idx) - _boot_aucs(y, np.asarray(p_b), idx)
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), int(len(d))
