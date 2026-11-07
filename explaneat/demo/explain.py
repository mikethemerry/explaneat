"""A scripted explanation of any evolved genome, as a hello-world check.

It exercises both phases of the framework on whatever topology evolution
produced, using only the public operation API (``ModelStateEngine``):

1. **Decomposition (identity operations).** Split every non-output node with
   more than one outgoing edge, deepest first, until each non-output node has
   exactly one consumer. The phenotype becomes an in-tree rooted at the
   output, and the network's function is unchanged (checked).
2. **Recomposition (annotations).** Walk the nodes in topological order. A
   node fed only by inputs gets a *leaf* annotation ``unit_<id>`` (its inputs
   -> the node). A node fed by other hidden nodes gets a *composition*
   ``compose_<id>`` whose children are the annotations rooted at those nodes,
   i.e. ``H(G(a, b), c)``. The output node's annotation is the global root,
   ``network``.

The checks: every operation passes validation; structural and compositional
coverage (the MCP ``get_coverage`` definitions, paper Defs 10-11) are 1.0;
and the trained network, the phenotype, the explained model state and the
root annotation's extracted function all compute the same outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ..analysis.annotation_function import AnnotationFunction
from ..analysis.coverage import CoverageComputer, compute_compositional_coverage_from_hierarchy
from ..core.explaneat import ExplaNEAT
from ..core.genome_network import NetworkStructure
from ..core.model_state import ModelStateEngine
from ..core.neuralneat import NeuralNeat
from ..core.structure_network import StructureNetwork

MAX_SPLITS = 400  # a tree-ification guard; demo networks need a handful


@dataclass
class ExplanationResult:
    n_splits: int = 0
    leaves: List[str] = field(default_factory=list)
    compositions: List[str] = field(default_factory=list)
    root: Optional[str] = None
    structural_coverage: float = 0.0
    compositional_coverage: float = 0.0
    max_abs_diff: Dict[str, float] = field(default_factory=dict)
    phenotype_nodes: int = 0
    phenotype_connections: int = 0
    hidden_nodes: int = 0
    formula: Optional[str] = None
    operations: List[dict] = field(default_factory=list)
    engine: Optional[ModelStateEngine] = None

    def summary(self) -> dict:
        return {
            "phenotype_nodes": self.phenotype_nodes,
            "phenotype_connections": self.phenotype_connections,
            "hidden_nodes": self.hidden_nodes,
            "splits": self.n_splits,
            "leaf_annotations": len(self.leaves),
            "composition_annotations": len(self.compositions),
            "root": self.root,
            "structural_coverage": self.structural_coverage,
            "compositional_coverage": self.compositional_coverage,
            "max_abs_diff": self.max_abs_diff,
        }


def _enabled(state: NetworkStructure):
    return [c for c in state.connections if c.enabled]


def _topological_order(state: NetworkStructure) -> List[str]:
    preds: Dict[str, Set[str]] = {n.id: set() for n in state.nodes}
    succs: Dict[str, Set[str]] = {n.id: set() for n in state.nodes}
    for c in _enabled(state):
        preds[c.to_node].add(c.from_node)
        succs[c.from_node].add(c.to_node)
    ready = sorted(n for n, p in preds.items() if not p)
    order: List[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in sorted(succs[n]):
            preds[m].discard(n)
            if not preds[m]:
                ready.append(m)
    if len(order) != len(preds):
        raise RuntimeError("model state is not a DAG")
    return order


def _treeify(engine: ModelStateEngine) -> int:
    """Split shared nodes, deepest first, until every node has <= 1 consumer."""
    n = 0
    while True:
        state = engine.current_state
        outputs = set(state.output_node_ids)
        fan_out: Dict[str, int] = {}
        for c in _enabled(state):
            fan_out[c.from_node] = fan_out.get(c.from_node, 0) + 1
        shared = [v for v in reversed(_topological_order(state))
                  if v not in outputs and fan_out.get(v, 0) >= 2]
        if not shared:
            return n
        if n >= MAX_SPLITS:
            raise RuntimeError(f"tree-ification needs more than {MAX_SPLITS} splits")
        engine.add_operation("split_node", {"node_id": shared[0]},
                             notes="demo: separate the roles of a shared node")
        n += 1


def _annotate(engine: ModelStateEngine, result: ExplanationResult) -> None:
    state = engine.current_state
    inputs = set(state.input_node_ids)
    outputs = list(state.output_node_ids)
    conns = _enabled(state)
    preds: Dict[str, List[str]] = {}
    for c in conns:
        preds.setdefault(c.to_node, []).append(c.from_node)
    used = {c.from_node for c in conns}
    unused_inputs = sorted(i for i in inputs if i not in used)

    root_of: Dict[str, str] = {}      # node -> name of the annotation it exits
    entries_of: Dict[str, List[str]] = {}
    for v in _topological_order(state):
        if v in inputs:
            continue
        ps = sorted(preds.get(v, []))
        in_preds = [p for p in ps if p in inputs]
        hid_preds = [p for p in ps if p not in inputs]
        own_nodes = in_preds + [v]
        own_conns = [[p, v] for p in ps]
        is_root = v in outputs
        if v == outputs[0] and unused_inputs:
            own_nodes = unused_inputs + own_nodes   # inputs the network ignores
        if not hid_preds:
            name = "network" if is_root else f"unit_{v}"
            entries = (unused_inputs if v == outputs[0] else []) + in_preds
            params = dict(name=name, hypothesis=f"node {v} as a function of its inputs",
                          entry_nodes=entries, exit_nodes=[v], subgraph_nodes=own_nodes,
                          subgraph_connections=own_conns)
            result.leaves.append(name)
        else:
            name = "network" if is_root else f"compose_{v}"
            children = [root_of[p] for p in hid_preds]
            entries: List[str] = (unused_inputs if v == outputs[0] else []) + list(in_preds)
            for p in hid_preds:
                entries += [e for e in entries_of[root_of[p]] if e not in entries]
            params = dict(name=name,
                          hypothesis=f"node {v} composes {', '.join(children)}",
                          entry_nodes=entries, exit_nodes=[v], subgraph_nodes=own_nodes,
                          subgraph_connections=own_conns, child_annotation_ids=children)
            result.compositions.append(name)
        engine.add_operation("annotate", params, notes="demo: scripted explanation")
        root_of[v] = name
        entries_of[name] = params["entry_nodes"]
        if is_root:
            result.root = name


def _coverage(engine: ModelStateEngine) -> Tuple[float, float]:
    """Structural (Def 10) and compositional (Def 11) coverage, as get_coverage."""
    state = engine.current_state
    anns = engine.annotations
    all_nodes = {str(n) for n in state.get_node_ids()}
    outputs = {str(n) for n in state.output_node_ids}
    computer = CoverageComputer(
        all_nodes=all_nodes,
        all_edges={(c.from_node, c.to_node) for c in _enabled(state)},
        input_nodes={str(n) for n in state.input_node_ids},
        output_nodes=outputs,
    )
    by_name = {a.name: a for a in anns}

    def descendants(name: str) -> Set[str]:
        kids = {a.name for a in anns if a.parent_annotation_id == name}
        out = set(kids)
        for k in kids:
            out |= descendants(k)
        return out

    expanded = []
    for a in anns:
        nodes = set(map(str, a.subgraph_nodes))
        edges = {tuple(map(str, e)) for e in a.subgraph_connections}
        for d in descendants(a.name):
            nodes |= set(map(str, by_name[d].subgraph_nodes))
            edges |= {tuple(map(str, e)) for e in by_name[d].subgraph_connections}
        expanded.append({"id": a.name, "entry_nodes": list(map(str, a.entry_nodes)),
                         "exit_nodes": list(map(str, a.exit_nodes)),
                         "subgraph_nodes": sorted(nodes),
                         "subgraph_connections": [list(e) for e in edges]})
    node_cov, _ = computer.compute_coverage(expanded)
    candidates = all_nodes - outputs
    structural = (len([n for n in candidates if n in node_cov]) / len(candidates)
                  if candidates else 1.0)
    hierarchy = compute_compositional_coverage_from_hierarchy(
        [{"name": a.name, "subgraph_nodes": list(map(str, a.subgraph_nodes)),
          "exit_nodes": list(map(str, a.exit_nodes)),
          "parent_annotation_id": a.parent_annotation_id} for a in anns],
        outputs,
    )
    return float(structural), float(hierarchy["compositional_coverage"])


def _input_column(node_id: str) -> int:
    """Dataset column of an input node id such as '-3' or a split copy '-3_b'."""
    base = node_id.split("_", 1)[0]
    return -int(base) - 1


def explain_genome(genome, config, X: np.ndarray) -> ExplanationResult:
    """Apply the scripted explanation and run the checks on ``X``."""
    result = ExplanationResult()
    phenotype = ExplaNEAT(genome, config).get_phenotype_network()
    result.phenotype_nodes = len(phenotype.nodes)
    result.phenotype_connections = len(_enabled(phenotype))
    result.hidden_nodes = len([n for n in phenotype.nodes
                               if n.id not in phenotype.input_node_ids
                               and n.id not in phenotype.output_node_ids])

    engine = ModelStateEngine(phenotype)
    result.n_splits = _treeify(engine)
    _annotate(engine, result)
    result.engine = engine
    result.operations = [op.to_dict() for op in engine.operations]
    result.structural_coverage, result.compositional_coverage = _coverage(engine)

    # Function preservation: four computations of the same network.
    xt = torch.as_tensor(X, dtype=torch.float64)
    with torch.no_grad():
        trained = NeuralNeat(genome, config).forward(xt).numpy().reshape(len(X), -1)[:, 0]
    pheno = StructureNetwork(phenotype).forward(xt).detach().numpy().reshape(len(X), -1)[:, 0]
    final_state = engine.current_state
    explained = StructureNetwork(final_state).forward(xt).detach().numpy().reshape(len(X), -1)[:, 0]

    anns = {a.name: a for a in engine.annotations}
    root = anns[result.root]
    nodes = {str(n) for a in anns.values() for n in a.subgraph_nodes}
    edges = [list(map(str, e)) for a in anns.values() for e in a.subgraph_connections]
    fn = AnnotationFunction.from_structure(
        {"entry_nodes": list(root.entry_nodes), "exit_nodes": list(root.exit_nodes),
         "subgraph_nodes": sorted(nodes), "subgraph_connections": edges},
        final_state,
    )
    cols = [_input_column(e) for e in root.entry_nodes]
    composed = np.asarray(fn(X[:, cols])).reshape(len(X), -1)[:, 0]

    result.max_abs_diff = {
        "phenotype_vs_trained": float(np.abs(pheno - trained).max()),
        "explained_state_vs_trained": float(np.abs(explained - trained).max()),
        "root_annotation_vs_trained": float(np.abs(composed - trained).max()),
    }
    try:
        result.formula = fn.to_latex()
    except Exception:  # formula extraction is best-effort for the report
        result.formula = None
    return result
