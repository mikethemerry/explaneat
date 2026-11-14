"""Analysis tools that the 10-seed and churn studies ran as scripts.

These are library versions of what the explanation agents and the extension
studies did by hand, so that the API, the MCP server and the React Explorer can
offer them to a human researcher:

* ``check_explanation``: the automatic checks of
  ``experiments/10seed/check_explanation.py`` (function preservation, per-leaf
  closed-form error, hierarchy, operation counts).
* ``StateEvaluator`` / ``Linearisation`` / ``performance_sources``: forward
  passes with annotation units linearised, from
  ``experiments/churn/explained_eval.py`` (restore gain, all-linear AUC).
* ``unit_activity``: how often each hidden ReLU is active on the data (dead,
  always on = affine on the data, rarely active, hinge).
* ``candidate_subgraphs``: candidate leaf annotations read off the topology
  (protocol step 2).
* ``category_offsets``: per one-hot level, the model's logit effect against
  the empirical log-odds offset (churn study A "state offsets").

Everything here is read-only and runs on in-memory copies.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.genome_network import NetworkStructure, NodeType
from .composition import (
    annotation_dicts_from_operations,
    descendant_annotations,
    expand_annotation,
)

LINEAR_TOL = 1e-7


# =========================================================================== helpers


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def input_column(node_id: str) -> int:
    """Data column of a (possibly split / consolidated) input node: ``-k`` -> ``k-1``."""
    base = str(node_id).split("_", 1)[0]
    k = int(base)
    if k >= 0:
        raise ValueError(f"{node_id} is not an input node id")
    return -k - 1


def base_feature(name: str) -> str:
    """Base feature of a one-hot column name: ``state:16`` -> ``state``."""
    return name.split(":", 1)[0] if ":" in name else name


def default_auc(y, score) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score).ravel()))


# =========================================================================== forward pass

Override = Tuple[Sequence[str], Callable[[Dict[str, np.ndarray]], np.ndarray]]


class StateEvaluator:
    """Node-by-node forward pass over a model-state ``NetworkStructure`` with the
    activations ``NeuralNeat`` applies: inputs are data columns, identity nodes
    pass through, other hidden nodes are ReLU, the output uses its own activation
    (sigmoid after the parity fix; identity for a log-risk output).

    Any node's value can be overridden by a function of other nodes' values.
    """

    def __init__(self, structure: NetworkStructure):
        self.structure = structure
        self.nodes = {n.id: n for n in structure.nodes}
        self.preds: Dict[str, List[Tuple[str, float]]] = {n: [] for n in self.nodes}
        for c in structure.connections:
            if c.enabled and c.from_node in self.nodes and c.to_node in self.nodes:
                self.preds[c.to_node].append((c.from_node, c.weight))
        self.output_ids = [str(o) for o in structure.output_node_ids]
        from ..core.structure_network import structure_output_activation

        # The trained model's output activation (sigmoid, or identity for a
        # Cox log-risk), not the stored gene: same rule as StructureNetwork.
        self.output_activation = structure_output_activation(structure).lower()
        from ..core.structure_network import structure_hidden_activation

        # The trained model's hidden activation (ReLU by default, or sigmoid).
        self.hidden_activation = structure_hidden_activation(structure).lower()

    def _order(self, extra_deps: Dict[str, Sequence[str]]):
        deps = {n: {p for p, _ in self.preds[n]} | set(extra_deps.get(n, ())) for n in self.nodes}
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

    def forward(
        self,
        X,
        overrides: Optional[Dict[str, Override]] = None,
        pre_activation: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Activations of every node. With ``pre_activation`` the output nodes
        hold their pre-activation value (the logit / log-risk)."""
        X = np.asarray(X, dtype=np.float64)
        overrides = overrides or {}
        order = self._order({n: deps for n, (deps, _) in overrides.items()})
        vals: Dict[str, np.ndarray] = {}
        for n in order:
            if n in overrides:
                vals[n] = np.broadcast_to(
                    np.asarray(overrides[n][1](vals), dtype=np.float64), (X.shape[0],)
                ).copy()
                continue
            node = self.nodes[n]
            if node.type == NodeType.INPUT:
                vals[n] = X[:, input_column(n)]
                continue
            if node.type == NodeType.FUNCTION:
                raise NotImplementedError("collapsed FUNCTION nodes are not supported")
            z = np.full(X.shape[0], float(node.bias or 0.0))
            for p, w in self.preds[n]:
                z = z + w * vals[p]
            if node.type == NodeType.OUTPUT:
                act = self.output_activation
                if pre_activation or act == "identity":
                    vals[n] = z
                elif act == "relu":
                    vals[n] = np.maximum(z, 0.0)
                else:
                    vals[n] = _sigmoid(z)
            elif node.activation == "identity":
                vals[n] = z
            elif self.hidden_activation == "sigmoid":
                vals[n] = _sigmoid(z)
            else:
                vals[n] = np.maximum(z, 0.0)
        return vals

    def output(self, X, overrides=None) -> np.ndarray:
        return self.forward(X, overrides)[self.output_ids[0]]

    def logit(self, X, overrides=None) -> np.ndarray:
        return self.forward(X, overrides, pre_activation=True)[self.output_ids[0]]

    def pre_activations(self, X) -> Dict[str, np.ndarray]:
        """Pre-activation value of every non-input node (for activity statistics)."""
        vals = self.forward(X)
        out = {}
        for n, node in self.nodes.items():
            if node.type == NodeType.INPUT:
                continue
            z = np.full(np.asarray(X).shape[0], float(node.bias or 0.0))
            for p, w in self.preds[n]:
                z = z + w * vals[p]
            out[n] = z
        return out


# =========================================================================== linearisation


def as_unit(annotation: dict, annotations: List[dict], structure: NetworkStructure) -> dict:
    """The annotation as a function of the *raw* entries of its whole subtree.

    For a leaf this is the annotation itself. For a composition, the entries are
    every entry declared in the subtree that is not produced inside it, so
    linearising a composition linearises its children too.
    """
    desc = descendant_annotations(annotation, annotations)
    if not desc:
        return annotation
    expanded = expand_annotation(annotation, annotations, structure)
    members = [annotation] + desc
    produced = {str(x) for a in members for x in a.get("exit_nodes", [])} | {
        str(n)
        for a in members
        for n in a.get("subgraph_nodes", [])
        if str(n) not in {str(e) for e in a.get("entry_nodes", [])}
    }
    entries: List[str] = []
    for a in [annotation] + desc + [expanded]:
        for e in a.get("entry_nodes", []):
            e = str(e)
            if e not in produced and e not in entries:
                entries.append(e)
    unit = dict(expanded)
    unit["entry_nodes"] = entries
    return unit


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
        out: Dict[str, Override] = {}
        for j, x in enumerate(self.exits):
            c = self.coef[:, j].copy()
            entries = list(self.entries)

            def fn(vals, c=c, entries=entries):
                v = np.full(len(next(iter(vals.values()))), c[0])
                for i, e in enumerate(entries):
                    v = v + c[i + 1] * vals[e]
                return v

            out[x] = (entries, fn)
        return out


class Linearisation:
    """Least-squares affine fits of each unit's exits on its entries, made once on
    ``X_fit`` (the unmodified forward pass) and reusable on any data."""

    def __init__(self, evaluator: StateEvaluator, units: Dict[str, dict], X_fit):
        self.evaluator = evaluator
        self.units = units
        vals = evaluator.forward(X_fit)
        self.fits: Dict[str, UnitFit] = {}
        n = np.asarray(X_fit).shape[0]
        for name, u in units.items():
            entries = [str(e) for e in u["entry_nodes"]]
            exits = [str(x) for x in u["exit_nodes"]]
            A = np.column_stack([np.ones(n)] + [vals[e] for e in entries])
            Y = np.column_stack([vals[x] for x in exits])
            coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
            resid = Y - A @ coef
            var = Y.var(axis=0)
            r2 = [
                float(1 - resid[:, j].var() / var[j]) if var[j] > 0 else 1.0
                for j in range(Y.shape[1])
            ]
            self.fits[name] = UnitFit(
                name,
                entries,
                exits,
                coef,
                float(np.max(np.abs(resid))) if resid.size else 0.0,
                float(np.max(np.abs(Y))) if Y.size else 0.0,
                r2,
            )

    @property
    def nonlinear_units(self) -> List[str]:
        return [n for n in self.units if self.fits[n].nonlinear]

    def overrides(self, names: Iterable[str]) -> Dict[str, Override]:
        out: Dict[str, Override] = {}
        for n in names:
            out.update(self.fits[n].override())
        return out

    def output(self, X, linearised: Iterable[str] = ()) -> np.ndarray:
        return self.evaluator.output(X, self.overrides(linearised))


def performance_sources(
    lin: Linearisation,
    X,
    y,
    metric: Callable[[Any, Any], float] = default_auc,
) -> dict:
    """Metric on (X, y) for the full model, every nonlinear unit linearised
    jointly ("all linear"), each unit linearised alone, and each unit's
    nonlinearity restored alone onto the all-linear model.

    restore gain = metric(restore alone) - metric(all linear); 0 for a linear unit.
    """
    nl = lin.nonlinear_units
    full = metric(y, lin.output(X))
    all_lin = metric(y, lin.output(X, nl))
    out = {"full": full, "all_linear": all_lin, "nonlinear_units": nl, "units": {}}
    for name in lin.units:
        lin_alone = metric(y, lin.output(X, [name]))
        if name in nl:
            rest = metric(y, lin.output(X, [n for n in nl if n != name]))
            gain = rest - all_lin
        else:
            rest, gain = all_lin, 0.0
        f = lin.fits[name]
        out["units"][name] = {
            "nonlinear": f.nonlinear,
            "max_abs_residual_fit": f.max_abs_residual,
            "min_r2_fit": min(f.r2) if f.r2 else 1.0,
            "n_entries": len(f.entries),
            "n_exits": len(f.exits),
            "linearised_alone": lin_alone,
            "linearised_alone_drop": full - lin_alone,
            "restore_alone": rest,
            "restore_gain": gain,
        }
    return out


def linearise_annotations(
    structure: NetworkStructure,
    operations: Sequence[dict],
    X,
    y,
    names: Optional[Sequence[str]] = None,
    metric: Callable[[Any, Any], float] = default_auc,
) -> dict:
    """Restore-gain table for annotation units.

    ``names`` defaults to the root's direct children when there is a single
    root with children, otherwise to every leaf annotation.
    """
    anns = annotation_dicts_from_operations(operations)
    by_name = {a["name"]: a for a in anns}
    by_id = {a["id"]: a for a in anns}
    if not names:
        child_refs = {c for a in anns for c in a.get("child_annotation_ids", [])}
        roots = [a for a in anns if a["name"] not in child_refs and a["id"] not in child_refs]
        if len(roots) == 1 and roots[0].get("child_annotation_ids"):
            names = list(roots[0]["child_annotation_ids"])
            scope = f"children of root {roots[0]['name']}"
        else:
            names = [a["name"] for a in anns if not a.get("child_annotation_ids")]
            scope = "leaf annotations"
    else:
        scope = "selected"
    units = {}
    for ref in names:
        a = by_name.get(ref) or by_id.get(ref)
        if a is None:
            raise KeyError(f"annotation {ref!r} not found")
        units[a["name"]] = as_unit(a, anns, structure)
    ev = StateEvaluator(structure)
    lin = Linearisation(ev, units, X)
    res = performance_sources(lin, X, y, metric)
    res["scope"] = scope
    res["n_rows"] = int(np.asarray(X).shape[0])
    return res


# =========================================================================== checks


def hierarchy_check(operations: Sequence[dict], output_ids: Sequence[str]) -> dict:
    """Exactly one root annotation whose subgraph contains every output node,
    with every annotation a descendant of it (same as check_explanation.py)."""
    anns: Dict[str, dict] = {}
    for op in operations:
        if op.get("type") != "annotate":
            continue
        p = op.get("params", {})
        ann_id = p.get("name") or f"ann_{op.get('seq', 0)}"
        anns[ann_id] = {
            "children": list(p.get("child_annotation_ids") or []),
            "nodes": {str(n) for n in p.get("subgraph_nodes", [])}
            | {str(n) for n in p.get("exit_nodes", [])},
        }
    child_ids = {c for a in anns.values() for c in a["children"]}
    roots = [a for a in anns if a not in child_ids]
    result = {
        "n_annotations": len(anns),
        "roots": roots,
        "single_root": len(roots) == 1,
        "root_contains_outputs": False,
        "all_under_root": False,
        "not_under_root": [],
    }
    if len(roots) == 1:
        seen, stack = set(), [roots[0]]
        while stack:
            a = stack.pop()
            if a in seen or a not in anns:
                continue
            seen.add(a)
            stack.extend(anns[a]["children"])
        result["root_contains_outputs"] = set(output_ids) <= anns[roots[0]]["nodes"]
        result["all_under_root"] = seen == set(anns)
        result["not_under_root"] = sorted(set(anns) - seen)
    result["complete"] = (
        result["single_root"] and result["root_contains_outputs"] and result["all_under_root"]
    )
    return result


def _closed_form_error(fn, entry, exit_acts):
    import sympy

    exprs = fn.to_sympy(force=True)
    if not exprs:
        return None
    syms = sorted({s for e in exprs.values() for s in e.free_symbols}, key=str)
    display_map = fn._structure.get_display_map() if fn._structure else {}
    col = {
        sympy.Symbol(f"x_{{{display_map.get(n, n)}}}"): i for i, n in enumerate(fn.entry_nodes)
    }
    errs = []
    for i in range(len(fn.exit_nodes)):
        e = exprs.get(f"y_{i}")
        if e is None:
            continue
        f = sympy.lambdify(
            syms,
            e,
            modules=[{"relu": lambda v: np.maximum(v, 0), "Max": np.maximum}, "numpy"],
        )
        args = [entry[:, col[s]] for s in syms]
        pred = np.broadcast_to(np.asarray(f(*args), dtype=float), exit_acts[:, i].shape)
        errs.append(float(np.max(np.abs(pred - exit_acts[:, i]))))
    return max(errs) if errs else None


def check_explanation(
    engine,
    operations: Sequence[dict],
    X,
    closed_form: bool = True,
) -> dict:
    """The automatic checks of ``experiments/10seed/check_explanation.py``.

    1. Function preservation: max |output(model state) - output(original phenotype)|.
    2. Per leaf: max abs error of the numeric annotation function and of the
       closed form as displayed (3 s.f.) against the actual exit activations.
    3. Hierarchy and operation counts.
    """
    from .activation_extractor import ActivationExtractor
    from .annotation_function import AnnotationFunction

    state = engine.current_state
    orig = engine.original_phenotype
    out_ids = [str(n) for n in orig.output_node_ids]
    dummy = {"entry_nodes": out_ids, "exit_nodes": out_ids}
    _, y_orig = ActivationExtractor.from_structure(orig).extract(X, dummy)
    _, y_new = ActivationExtractor.from_structure(state).extract(X, dummy)
    preservation = float(np.max(np.abs(y_orig - y_new))) if len(X) else 0.0

    extractor = ActivationExtractor.from_structure(state)
    leaves = []
    for ann in annotation_dicts_from_operations(operations):
        if ann.get("child_annotation_ids"):
            continue
        leaf = {
            "id": ann["id"],
            "name": ann["name"],
            "entry_nodes": ann["entry_nodes"],
            "exit_nodes": ann["exit_nodes"],
            "subgraph_nodes": ann["subgraph_nodes"],
            "subgraph_connections": ann["subgraph_connections"],
            "child_annotation_ids": [],
        }
        try:
            entry, exit_acts = extractor.extract(X, leaf)
            fn = AnnotationFunction.from_structure(leaf, state)
            num_err = float(np.max(np.abs(fn(entry).reshape(exit_acts.shape) - exit_acts)))
        except Exception as e:  # report, don't hide
            leaves.append({"id": ann["id"], "name": ann["name"],
                           "numeric_max_abs_err": None, "closed_form_max_abs_err": None,
                           "error": str(e)})
            continue
        cf_err: Any = None
        if closed_form:
            try:
                cf_err = _closed_form_error(fn, entry, exit_acts)
            except Exception as e:
                cf_err = f"error: {e}"
        exit_scale = float(np.max(np.abs(exit_acts))) if exit_acts.size else 0.0
        leaves.append({
            "id": ann["id"],
            "name": ann["name"],
            "numeric_max_abs_err": num_err,
            "closed_form_max_abs_err": cf_err,
            "exit_scale": exit_scale,
        })

    hierarchy = hierarchy_check(operations, out_ids)
    counts = Counter(op.get("type") for op in operations)
    numeric = [l["closed_form_max_abs_err"] for l in leaves
               if isinstance(l.get("closed_form_max_abs_err"), float)]
    nums = [l["numeric_max_abs_err"] for l in leaves if isinstance(l.get("numeric_max_abs_err"), float)]
    return {
        "n_rows": int(np.asarray(X).shape[0]),
        "function_preservation_max_abs_err": preservation,
        "function_preserved": preservation < 1e-9,
        "hierarchy": hierarchy,
        "op_counts": dict(counts),
        "leaves": leaves,
        "closed_form_max_abs_err": max(numeric) if numeric else None,
        "numeric_max_abs_err": max(nums) if nums else None,
    }


# =========================================================================== unit activity


def unit_activity(structure: NetworkStructure, X, rare_rows: int = 5) -> List[dict]:
    """How each hidden ReLU behaves on the data.

    status:
      * ``dead``: never active (output identically 0);
      * ``always_on``: active on every row, so affine on the data;
      * ``rare``: active on at most ``rare_rows`` rows;
      * ``hinge``: switches on some rows and off on others.
    Identity nodes are reported as ``identity``.
    """
    ev = StateEvaluator(structure)
    pre = ev.pre_activations(X)
    display = structure.get_display_map() if hasattr(structure, "get_display_map") else {}
    out_ids = set(ev.output_ids)
    succ: Dict[str, List[Tuple[str, float]]] = {}
    for c in structure.connections:
        if c.enabled:
            succ.setdefault(c.from_node, []).append((c.to_node, c.weight))
    n = np.asarray(X).shape[0]
    rows = []
    for node in structure.nodes:
        if node.type not in (NodeType.HIDDEN,) or node.id not in pre:
            continue
        z = pre[node.id]
        identity = node.activation == "identity"
        active = int(np.sum(z > 0))
        a = z if identity else np.maximum(z, 0.0)
        if identity:
            status = "identity"
        elif active == 0:
            status = "dead"
        elif active == n:
            status = "always_on"
        elif active <= rare_rows:
            status = "rare"
        else:
            status = "hinge"
        direct_out = sum(w for t, w in succ.get(node.id, []) if t in out_ids)
        rows.append({
            "node_id": node.id,
            "display_name": display.get(node.id, node.id),
            "status": status,
            "active_rows": active,
            "n_rows": int(n),
            "active_fraction": active / n if n else 0.0,
            "mean": float(np.mean(a)) if n else 0.0,
            "std": float(np.std(a)) if n else 0.0,
            "direct_output_weight": float(direct_out),
            "direct_logit_sd": float(np.std(a * direct_out)) if n else 0.0,
            "n_successors": len(succ.get(node.id, [])),
        })
    rows.sort(key=lambda r: (["dead", "rare", "always_on", "hinge", "identity"].index(r["status"]), r["node_id"]))
    return rows


# =========================================================================== candidates


def candidate_subgraphs(structure: NetworkStructure, annotations=None) -> dict:
    """Candidate leaf subgraphs read off the topology (protocol step 2).

    * ``hidden_gate``: a hidden node with its direct predecessors as entries;
    * ``chain``: a hidden node with all its hidden ancestors (the input closure),
      entries being the inputs feeding the closure;
    * ``direct_inputs``: the inputs with direct edges to an output (the linear
      part, usually a child of the root);
    * ``unconnected_inputs``: inputs with no enabled edges (zero-effect entries
      of the root annotation).

    Each candidate reports the splits ``analyze_coverage_for_splits`` would
    need and the nodes already owned by an annotation.
    """
    from .split_detection import analyze_coverage_for_splits, covered_node_owners

    nodes = {n.id: n for n in structure.nodes}
    preds: Dict[str, List[str]] = {n: [] for n in nodes}
    succ: Dict[str, List[str]] = {n: [] for n in nodes}
    for c in structure.connections:
        if c.enabled and c.from_node in nodes and c.to_node in nodes:
            preds[c.to_node].append(c.from_node)
            succ[c.from_node].append(c.to_node)
    out_ids = {str(o) for o in structure.output_node_ids}
    owners = covered_node_owners(annotations) if annotations else {}
    display = structure.get_display_map() if hasattr(structure, "get_display_map") else {}

    def is_input(nid):
        return nodes[nid].type == NodeType.INPUT

    def describe(kind, title, members, entries, exits):
        coverage = sorted(set(members) | set(entries))
        res = analyze_coverage_for_splits(structure, coverage, annotations)
        return {
            "kind": kind,
            "title": title,
            "entry_nodes": sorted(entries),
            "exit_nodes": sorted(exits),
            "subgraph_nodes": coverage,
            "entry_names": [display.get(e, e) for e in sorted(entries)],
            "splits_needed": [v.node_id for v in res.violations],
            "already_covered": [c["node_id"] for c in res.already_covered],
            "clean": not res.violations and not res.already_covered,
        }

    hidden = [n for n in nodes if nodes[n].type == NodeType.HIDDEN]
    cands = []
    for h in sorted(hidden):
        if h in owners:
            continue
        p = preds[h]
        feeds_output = any(s in out_ids for s in succ[h])
        tag = " -> output" if feeds_output else ""
        cands.append(describe("hidden_gate", f"{display.get(h, h)} with its {len(p)} inputs{tag}", [h], p, [h]))
        # input closure over hidden ancestors
        closure, stack = set(), [h]
        while stack:
            x = stack.pop()
            if x in closure:
                continue
            closure.add(x)
            for q in preds[x]:
                if not is_input(q) and nodes[q].type != NodeType.OUTPUT:
                    stack.append(q)
        if len(closure) > 1:
            entries = {q for x in closure for q in preds[x] if q not in closure}
            cands.append(describe("chain", f"{display.get(h, h)} and {len(closure) - 1} hidden ancestors",
                                  sorted(closure), entries, [h]))
    direct = sorted({p for o in out_ids for p in preds.get(o, []) if is_input(p)})
    unconnected = sorted(
        n for n in nodes if is_input(n) and not succ[n] and not preds[n]
    )
    # dedupe identical coverages
    seen, uniq = set(), []
    for c in cands:
        key = tuple(c["subgraph_nodes"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    uniq.sort(key=lambda c: (not c["clean"], len(c["splits_needed"]), c["kind"] != "hidden_gate", len(c["subgraph_nodes"])))
    return {
        "candidates": uniq,
        "direct_inputs": [{"node_id": d, "name": display.get(d, d)} for d in direct],
        "unconnected_inputs": [{"node_id": u, "name": display.get(u, u)} for u in unconnected],
        "output_nodes": sorted(out_ids),
    }


# =========================================================================== category offsets


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def _wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def onehot_groups(feature_names: Sequence[str], X) -> Dict[str, List[int]]:
    """Columns ``base:level`` with exactly two distinct values, grouped by base."""
    X = np.asarray(X)
    groups: Dict[str, List[int]] = {}
    for i, name in enumerate(feature_names):
        if ":" not in name:
            continue
        if len(np.unique(X[:, i])) > 2:
            continue
        groups.setdefault(base_feature(name), []).append(i)
    return groups


def category_offsets(
    structure: NetworkStructure,
    X,
    y,
    feature_names: Sequence[str],
    large_effect: float = 1.0,
) -> dict:
    """Per one-hot level: the model's logit effect of the level against the data.

    For each level column ``c`` with ``n`` rows "on" (the larger of its two
    values, since z-scoring preserves order):
      * ``model_logit_effect``: mean over those rows of logit(p) - logit(p with
        ``c`` set to its "off" value);
      * ``empirical_logodds_offset``: logit(rate of y in those rows) -
        logit(overall rate) (with a 0.5 continuity correction);
      * ``warranted_offset``: logit(rate) - logit(mean p with ``c`` off);
      * ``supported``: the model's mean prediction on those rows lies inside the
        Wilson 95% interval of the empirical rate;
      * ``flag``: ``large_unsupported`` when |effect| >= ``large_effect`` and not
        supported, ``large`` when large but supported.
    Levels the network does not read (no path to the output) have effect 0 and
    are skipped.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel().astype(float)
    ev = StateEvaluator(structure)
    base_logit = ev.logit(X)
    used_cols = set()
    for n in structure.nodes:
        if n.type == NodeType.INPUT:
            try:
                used_cols.add(input_column(n.id))
            except ValueError:
                pass
    groups = onehot_groups(feature_names, X)
    rate_all = (y.sum() + 0.5) / (len(y) + 1)
    out_groups = []
    for base, cols in sorted(groups.items()):
        levels = []
        for c in cols:
            if c not in used_cols:
                continue
            vals = np.unique(X[:, c])
            if len(vals) < 2:
                continue
            on_val, off_val = vals.max(), vals.min()
            rows = X[:, c] == on_val
            n = int(rows.sum())
            if n == 0:
                continue
            Xoff = X[rows].copy()
            Xoff[:, c] = off_val
            l_off = ev.logit(Xoff)
            effect = float(np.mean(base_logit[rows] - l_off))
            if abs(effect) < 1e-12:
                continue
            k = float(y[rows].sum())
            rate = (k + 0.5) / (n + 1)
            p_model = float(np.mean(_sigmoid(base_logit[rows])))
            lo, hi = _wilson(k, n)
            supported = bool(lo <= p_model <= hi)
            large = abs(effect) >= large_effect
            levels.append({
                "feature": feature_names[c],
                "column": c,
                "rows": n,
                "positives": int(k),
                "rate": k / n,
                "model_mean_p": p_model,
                "model_logit_effect": effect,
                "empirical_logodds_offset": float(_logit(rate) - _logit(rate_all)),
                "warranted_offset": float(_logit(rate) - _logit(np.mean(_sigmoid(l_off)))),
                "rate_ci": [lo, hi],
                "supported": supported,
                "flag": ("large_unsupported" if large and not supported else "large" if large else ""),
            })
        if not levels:
            continue
        levels.sort(key=lambda r: -abs(r["model_logit_effect"]))
        out_groups.append({
            "base_feature": base,
            "n_levels": len(cols),
            "n_levels_used": len(levels),
            "max_abs_effect": max(abs(l["model_logit_effect"]) for l in levels),
            "n_large_unsupported": sum(1 for l in levels if l["flag"] == "large_unsupported"),
            "levels": levels,
        })
    out_groups.sort(key=lambda g: -g["max_abs_effect"])
    return {"n_rows": int(X.shape[0]), "large_effect": large_effect, "groups": out_groups}
