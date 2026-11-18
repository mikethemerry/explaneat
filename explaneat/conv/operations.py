"""Operations, validation and the annotation strategy for conv model states.

Operation handlers (each exact: the network's function is unchanged)
------------------------------------------------------------------
* ``split_node {node_id, partition?}``: ExplaNEAT's ``apply_split_node`` (one copy per outgoing
  edge, ids ``<base>_<letter>``, every copy keeps all incoming edges and therefore the same map),
  followed by ``apply_consolidate_node`` for each group of ``partition`` (a list of lists of
  target ids), so the node becomes one copy per *role*. Kernels follow because the core handlers
  copy ``innovation`` onto every copied connection.
* ``consolidate_node {node_ids}``: ExplaNEAT's handler unchanged.
* ``add_identity_node {connections, new_node_id, side}``:
  - ``side="target"`` (ExplaNEAT's form, ``apply_add_identity_node``): the connections into one
    target are redirected into a new identity map I at the target's level (each keeps its kernel
    and resampling), and ``I -> target`` is a 1 x 1 unit delta kernel.
  - ``side="source"`` (a **port**): the connections out of one source are re-rooted at a new
    identity map P at the source's level; ``source -> P`` is the unit delta and each ``P -> dst``
    keeps its kernel and resampling. P is an exact copy of the source map, so it can serve as the
    interface (entry) of the structure it feeds: every kernel into the structure stays inside it.
* ``annotate``: as ExplaNEAT (name, hypothesis, entry/exit/subgraph nodes, subgraph connections,
  evidence, ``child_annotation_ids``); marks its nodes and connections as covered (immutable).
* ``rename_annotation``: sets a display name.

Validation (``validate_conv_operation``)
---------------------------------------
Every operation is first checked by ExplaNEAT's ``validate_operation`` on the ``NetworkStructure``
(including, for ``annotate``, the three collapse preconditions P1 entry-only ingress, P2 exit-only
egress, P3 pure exits on the annotation's own nodes). Conv additions, for ``annotate``:
the three preconditions checked again on the *expanded* subgraph (own nodes plus all descendants,
``CollapseValidator.validate_collapsible``), since a composition's boundary is the union of its
children's; entries must be fed from outside only; and the collapse must keep the graph a DAG
(no path from an exit back to an entry through outside nodes).

Annotation strategy (``annotation_strategy``)
---------------------------------------------
The UI's rules (``OperationsPanel.tsx``, ``docs/annotation_coverage_model.md``) on a candidate
node set, in two modes:

* ``mode="ui"``: the rules as written. Entry = all inputs from outside (or a network input),
  exit = all outputs outside (or a network output), else intermediate. One identity node
  (target side) when exactly one exit has external inputs; splits for entry or intermediate nodes
  with external outputs; blocking issues for intermediate or exit nodes with external inputs.
* ``mode="ports"`` (the conv default): every external input into the candidate is intercepted by
  a source-side identity node (one port per external source), so each map's full computation
  stays inside one structure and its entries are exact copies of the incoming maps. An external
  input *can't* be intercepted when its source lies on a path that leaves the candidate and
  re-enters it: collapsing would then create a cycle. Those are the blocking issues, and the
  selection must be expanded (the convex hull) or merged with the structure that owns the source.
  Splits as in the UI (members with internal and external outputs), optionally per role
  (``role_of``: e.g. sibling sub-part vs outside the structure).

Execution order: identity nodes -> splits -> annotate.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from ..analysis.collapse_validator import CollapseAnnotation, CollapseGraph, CollapseValidator
from ..core.genome_network import NetworkConnection, NetworkNode, NodeType
from ..core.model_state import AnnotationData
from ..core.operations import (OperationError, annotation_descendants, apply_add_identity_node,
                               apply_consolidate_node, apply_split_node, resolve_annotation_ref,
                               validate_operation)
from .model_state import ConvModelState, _id_key


# =============================================================================== handlers
def apply_operation(state: ConvModelState, op_type: str, params: Dict[str, Any], covered_nodes: Set[str],
                    covered_connections: Set[Tuple[str, str]], annotations: List[AnnotationData],
                    children: Dict[str, List[str]], seq: int) -> Dict[str, Any]:
    s = state.structure
    if op_type == "split_node":
        return apply_conv_split(state, params["node_id"], params.get("partition"), covered_nodes)
    if op_type == "consolidate_node":
        return apply_consolidate_node(s, list(params["node_ids"]), covered_nodes)
    if op_type == "add_identity_node":
        conns = [tuple(c) for c in params["connections"]]
        side = params.get("side", "target")
        if side == "target":
            target = params.get("target_node") or conns[0][1]
            res = apply_add_identity_node(s, target, conns, params["new_node_id"], covered_connections)
            state.levels[params["new_node_id"]] = state.level(target)
            for c in s.connections:
                if c.from_node == params["new_node_id"] and c.to_node == target and c.innovation is None:
                    c.innovation = state.add_delta_kernel()
                    c.weight = 1.0
        else:
            res = apply_add_port(state, conns, params["new_node_id"], covered_connections)
        state.meta[params["new_node_id"]] = {"identity": side, "of": conns[0][0] if side == "source" else conns[0][1],
                                             "role": params.get("role")}
        return res
    if op_type == "annotate":
        ann = AnnotationData(
            name=params["name"], hypothesis=params.get("hypothesis", ""), entry_nodes=list(params["entry_nodes"]),
            exit_nodes=list(params["exit_nodes"]), subgraph_nodes=list(params["subgraph_nodes"]),
            subgraph_connections=[tuple(c) for c in params.get("subgraph_connections", [])],
            evidence=params.get("evidence"), display_name=params.get("display_name"), annotation_id=f"ann_{seq}")
        kids = []
        for ref in params.get("child_annotation_ids", []) or []:
            nm = resolve_annotation_ref(ref, annotations)
            for other in annotations:
                if other.name == nm:
                    other.parent_annotation_id = ann.name
                    kids.append(nm)
        annotations.append(ann)
        children[ann.name] = kids
        covered_nodes.update(ann.subgraph_nodes)
        covered_connections.update(ann.subgraph_connections)
        return {"annotation_index": len(annotations) - 1, "annotation_id": ann.annotation_id}
    if op_type == "rename_annotation":
        nm = resolve_annotation_ref(params["annotation_id"], annotations)
        for a in annotations:
            if a.name == nm:
                a.display_name = params.get("display_name")
                return {"annotation_id": nm, "display_name": a.display_name}
        raise OperationError(f"Annotation '{params['annotation_id']}' not found")
    raise OperationError(f"Unknown operation type: {op_type}")


def apply_conv_split(state: ConvModelState, node_id: str, partition: Optional[List[List[str]]],
                     covered_nodes: Set[str]) -> Dict[str, Any]:
    """ExplaNEAT's full split, then one consolidation per group of ``partition``."""
    s = state.structure
    res = apply_split_node(s, node_id, covered_nodes)
    copies = res["created_nodes"]
    if not partition:
        return res
    target_of = {}
    for c in s.connections:
        if c.from_node in copies:
            target_of[c.from_node] = c.to_node
    final = []
    for group in partition:
        grp = [cp for cp in copies if target_of[cp] in set(group)]
        if len(grp) >= 2:
            final += apply_consolidate_node(s, grp, covered_nodes)["created_nodes"]
        else:
            final += grp
    roles = {}
    for c in s.connections:
        if c.from_node in final:
            roles.setdefault(c.from_node, []).append(c.to_node)
    return {"created_nodes": final, "removed_nodes": [node_id], "targets": roles}


def apply_add_port(state: ConvModelState, connections: List[Tuple[str, str]], new_node_id: str,
                   covered_connections: Set[Tuple[str, str]]) -> Dict[str, Any]:
    """Source-side identity node: ``src -> P`` (unit delta at the source's level), ``P -> dst`` with
    each original kernel and resampling. P is an exact copy of the source map."""
    s = state.structure
    src = connections[0][0]
    if s.get_node_by_id(new_node_id) is not None:
        raise OperationError(f"Node {new_node_id} already exists")
    found = []
    for a, b in connections:
        if a != src:
            raise OperationError(f"Connection {(a, b)} does not start at {src}")
        if (a, b) in covered_connections:
            raise OperationError(f"Connection {(a, b)} is covered by annotation")
        c = next((c for c in s.connections if c.from_node == a and c.to_node == b and c.enabled), None)
        if c is None:
            raise OperationError(f"Connection {(a, b)} not found")
        found.append(c)
    s.nodes.append(NetworkNode(id=new_node_id, type=NodeType.HIDDEN, bias=0.0, activation="identity",
                               response=1.0, aggregation="sum"))
    state.levels[new_node_id] = state.level(src)
    for c in found:
        c.from_node = new_node_id
    s.connections.append(NetworkConnection(from_node=src, to_node=new_node_id, weight=1.0, enabled=True,
                                           innovation=state.add_delta_kernel()))
    return {"created_nodes": [new_node_id], "created_connections": [(src, new_node_id)] + [(new_node_id, b) for _, b in connections],
            "removed_connections": [list(c) for c in connections]}


# =============================================================================== graph helpers
def collapse_graph(state: ConvModelState) -> CollapseGraph:
    s = state.structure
    return CollapseGraph.from_sets(set(state.node_ids()), set(state.edges()), set(s.input_node_ids),
                                   set(s.output_node_ids))


def _reach(starts: Iterable[str], adj: Dict[str, List[str]], blocked: Set[str] = frozenset()) -> Set[str]:
    seen, stack = set(), list(starts)
    while stack:
        n = stack.pop()
        for m in adj.get(n, []):
            if m not in seen and m not in blocked:
                seen.add(m)
                stack.append(m)
    return seen


def reentry_nodes(state: ConvModelState, members: Set[str]) -> Set[str]:
    """Nodes outside ``members`` on a path that leaves the set and comes back (the convex hull
    minus the set). Non-empty means the set cannot be collapsed without creating a cycle."""
    succ, pred = state.succ(), state.pred()
    down = _reach(members, succ) - members
    up = _reach(members, pred) - members
    return down & up


def collapse_cycle(state: ConvModelState, nodes: Set[str], entries: Set[str], exits: Set[str]) -> List[str]:
    """Messages for paths exit -> (outside nodes) -> entry, which would make the collapsed graph
    cyclic (the annotation node would feed its own entry)."""
    succ = state.succ()
    msgs = []
    for x in sorted(exits, key=_id_key):
        frontier = [m for m in succ.get(x, []) if m not in nodes]
        seen, hit = set(frontier), None
        while frontier and hit is None:
            n = frontier.pop()
            for m in succ.get(n, []):
                if m in entries:
                    hit = (n, m)
                    break
                if m not in nodes and m not in seen:
                    seen.add(m)
                    frontier.append(m)
        if hit:
            msgs.append(f"collapse would create a cycle: exit {x} reaches entry {hit[1]} through outside node {hit[0]}")
    return msgs


def classify(state: ConvModelState, members: Set[str]) -> Dict[str, List[str]]:
    """The UI's node classification of a selection (entry / intermediate / exit)."""
    pred, succ = state.pred(), state.succ()
    s = state.structure
    ins, outs = set(s.input_node_ids), set(s.output_node_ids)
    out = {"entry": [], "intermediate": [], "exit": []}
    for n in sorted(members, key=_id_key):
        p, q = pred[n], succ[n]
        if n in ins or (p and all(u not in members for u in p)):
            out["entry"].append(n)
        elif n in outs or (q and all(v not in members for v in q)):
            out["exit"].append(n)
        else:
            out["intermediate"].append(n)
    return out


def internal_connections(state: ConvModelState, members: Set[str]) -> List[List[str]]:
    return [[a, b] for a, b in state.edges() if a in members and b in members]


# =============================================================================== validation
def validate_conv_operation(state: ConvModelState, op_type: str, params: Dict[str, Any], covered_nodes: Set[str],
                            covered_connections: Set[Tuple[str, str]], annotations: List[AnnotationData],
                            children: Dict[str, List[str]]) -> List[str]:
    s = state.structure
    if op_type == "add_identity_node" and params.get("side", "target") == "source":
        errs = []
        conns = [tuple(c) for c in params.get("connections", [])]
        if not conns:
            return ["At least one connection required"]
        nid = params.get("new_node_id")
        if not nid:
            errs.append("new_node_id is required")
        elif nid in set(state.node_ids()):
            errs.append(f"Node {nid} already exists")
        src = conns[0][0]
        have = set(state.edges())
        for c in conns:
            if c[0] != src:
                errs.append(f"Connection {list(c)} does not start at source {src}")
            if c not in have:
                errs.append(f"Connection {list(c)} does not exist")
            if c in covered_connections:
                errs.append(f"Connection {list(c)} is covered by an annotation")
        return errs
    if op_type == "add_identity_node":
        p = dict(params)
        p.setdefault("target_node", params["connections"][0][1] if params.get("connections") else None)
        return validate_operation(s, op_type, p, covered_nodes, covered_connections, annotations)
    if op_type == "split_node":
        errs = validate_operation(s, op_type, params, covered_nodes, covered_connections, annotations)
        part = params.get("partition")
        if part and not errs:
            tgt = sorted({b for a, b in state.edges() if a == params["node_id"]}, key=_id_key)
            flat = [t for grp in part for t in grp]
            if sorted(flat, key=_id_key) != tgt:
                errs.append(f"partition of {params['node_id']} must list each of its targets once: {tgt}")
        return errs
    if op_type == "rename_annotation":
        return validate_operation(s, op_type, params, covered_nodes, covered_connections, annotations)
    if op_type == "consolidate_node":
        return validate_operation(s, op_type, params, covered_nodes, covered_connections, annotations)
    if op_type != "annotate":
        return [f"Unknown operation type: {op_type}"]

    # ---- annotate: ExplaNEAT's checks (incl. P1-P3 on the annotation's own nodes) ...
    errs = validate_operation(s, op_type, params, covered_nodes, covered_connections, annotations)
    if errs:
        return errs
    # ---- ... then the expanded boundary (own nodes + all descendants), entries, acyclic collapse
    own = set(params["subgraph_nodes"])
    expanded = set(own)
    for ref in params.get("child_annotation_ids", []) or []:
        nm = resolve_annotation_ref(ref, annotations)
        for a in annotations:
            if a.name == nm:
                expanded |= set(a.subgraph_nodes)
                for d in annotation_descendants(nm, annotations):
                    expanded |= set(d.subgraph_nodes)
    entries, exits = set(params["entry_nodes"]), set(params["exit_nodes"])
    graph = collapse_graph(state)
    ann = CollapseAnnotation.from_sets(params["name"], entries, exits, expanded,
                                       {e for e in graph.edges if e[0] in expanded and e[1] in expanded})
    res = CollapseValidator.validate_collapsible(graph, ann)
    tag = {"entry_only_ingress": "P1", "exit_only_egress": "P2", "pure_exits": "P3"}
    for v in res.violations:
        errs.append(f"{tag[v.precondition.value]} violation (expanded subgraph): {v.message}")
    pred = state.pred()
    for e in sorted(entries, key=_id_key):
        inside = [u for u in pred.get(e, []) if u in expanded]
        if inside:
            errs.append(f"entry {e} receives from node {inside[0]} inside the annotation: an entry must be fed from outside only")
    errs += collapse_cycle(state, expanded, entries, exits)
    return errs


# =============================================================================== strategy
def annotation_strategy(state: ConvModelState, selection: Iterable[str], mode: str = "ports",
                        covered_nodes: Set[str] = frozenset(), role_of: Optional[Callable[[str], Any]] = None,
                        new_id: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """Minimal operations that make ``selection`` a valid, collapsible annotation.

    Returns ``{"selection", "classification", "discovered", "identity_ops", "split_ops",
    "blocking", "valid_as_is"}``; ops are ``{"type", "params"}`` dicts to apply in order
    (identity first, then splits). ``role_of(dst)`` labels the destination of an external edge;
    a member's external edges are split per label (default: one label). ``new_id(kind)`` names
    new identity nodes (default ``p<k>`` / ``i<k>``).
    """
    members = set(selection)
    s = state.structure
    ins, outs = set(s.input_node_ids), set(s.output_node_ids)
    pred, succ = state.pred(), state.succ()
    discovered = reentry_nodes(state, members)
    counter = [0]

    def _new(kind: str) -> str:
        if new_id is not None:
            return new_id(kind)
        while True:
            counter[0] += 1
            nid = f"{kind}{counter[0]}"
            if nid not in set(state.node_ids()):
                return nid

    cls = classify(state, members)
    identity_ops, split_ops, blocking = [], [], []
    for w in sorted(discovered, key=_id_key):
        into = sorted({v for v in succ[w] if v in members}, key=_id_key)
        frm = sorted({u for u in pred[w] if u in members}, key=_id_key)
        blocking.append({"node": w, "kind": "re-entry", "targets": into, "sources": frm,
                         "message": (f"node {w} is on a path that leaves the selection" +
                                     (f" (from {frm[0]})" if frm else "") + f" and re-enters it at {into[0] if into else '?'}: "
                                     "its input cannot be intercepted without a cycle; expand the selection to include it")})
    if mode == "ui":
        ext_into = {}
        for n in cls["intermediate"] + cls["exit"]:
            ext = sorted({u for u in pred[n] if u not in members}, key=_id_key)
            if ext:
                ext_into[n] = ext
        exits_ext = [n for n in cls["exit"] if n in ext_into]
        if len(exits_ext) == 1 and not [n for n in cls["intermediate"] if n in ext_into]:
            x = exits_ext[0]
            internal = [[u, x] for u in pred[x] if u in members]
            identity_ops.append({"type": "add_identity_node",
                                 "params": {"side": "target", "target_node": x, "connections": internal,
                                            "new_node_id": _new("i"), "replaces_exit": x}})
        else:
            for n, ext in ext_into.items():
                if n in discovered:
                    continue
                blocking.append({"node": n, "kind": "external_input", "sources": ext,
                                 "message": f"{'exit' if n in cls['exit'] else 'intermediate'} node {n} has external "
                                            f"inputs from {', '.join(ext)}: expand the selection to include them"})
    else:
        for u in sorted({u for n in members for u in pred[n] if u not in members}, key=_id_key):
            if u in discovered:
                continue  # reported as blocking
            conns = [[u, v] for v in succ[u] if v in members]
            identity_ops.append({"type": "add_identity_node",
                                 "params": {"side": "source", "connections": conns, "new_node_id": _new("p")}})
    # splits: members (not network outputs) with internal and external outputs, or external
    # outputs with several roles
    for n in sorted(members, key=_id_key):
        if n in outs or n in covered_nodes:
            continue
        q = succ[n]
        internal = [v for v in q if v in members]
        external = [v for v in q if v not in members]
        if not external:
            continue
        groups: Dict[Any, List[str]] = {}
        for v in external:
            groups.setdefault(role_of(v) if role_of else "external", []).append(v)
        parts = ([sorted(internal, key=_id_key)] if internal else []) + \
                [sorted(v, key=_id_key) for _, v in sorted(groups.items(), key=lambda t: str(t[0]))]
        if len(parts) >= 2:
            split_ops.append({"type": "split_node", "params": {"node_id": n, "partition": parts}})
    return {"selection": sorted(members, key=_id_key), "classification": cls,
            "discovered": sorted(discovered, key=_id_key), "identity_ops": identity_ops, "split_ops": split_ops,
            "blocking": blocking, "valid_as_is": not (identity_ops or split_ops or blocking)}


def final_annotation(state: ConvModelState, members: Set[str], entries: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Entry / exit / subgraph sets of a (now clean) selection: entries are the members fed only
    from outside (ports, network inputs), exits the members with an edge leaving the set."""
    pred, succ = state.pred(), state.succ()
    s = state.structure
    if entries is None:
        entries = {n for n in members if n in set(s.input_node_ids)
                   or (pred[n] and all(u not in members for u in pred[n]))}
    exits = {n for n in members if n in set(s.output_node_ids) or any(v not in members for v in succ[n])}
    return {"entry_nodes": sorted(entries, key=_id_key), "exit_nodes": sorted(exits, key=_id_key),
            "subgraph_nodes": sorted(members, key=_id_key),
            "subgraph_connections": internal_connections(state, members)}


def execute_strategy(engine, selection: Iterable[str], name: str, hypothesis: str = "", mode: str = "ports",
                     evidence: Optional[dict] = None, role_of=None) -> Dict[str, Any]:
    """Apply the strategy for one selection (identity -> splits -> annotate) on an engine.
    Raises ``OperationError`` with the blocking issues if any remain."""
    members = set(selection)
    strat = annotation_strategy(engine.state, members, mode, engine.covered_nodes, role_of)
    if strat["blocking"]:
        raise OperationError("blocking issues: " + "; ".join(b["message"] for b in strat["blocking"]))
    ports = set()
    for op in strat["identity_ops"]:
        engine.add_operation(op["type"], op["params"])
        if op["params"].get("side") == "source":
            ports.add(op["params"]["new_node_id"])
            members.add(op["params"]["new_node_id"])
        else:
            members.discard(op["params"]["replaces_exit"])
            members.add(op["params"]["new_node_id"])
    apply_splits(engine, [members], mode, role_of)
    # a split copy of an *entry* that carries only external edges is fed and read from outside:
    # it is not part of the annotation (kept inside, as the UI does for input nodes, it would be an
    # exit with external inputs, a P3 violation)
    pred, succ = engine.state.pred(), engine.state.succ()
    ins = set(engine.state.structure.input_node_ids)
    members = {n for n in members if n in ins or n in ports or not (
        pred[n] and succ[n] and all(u not in members for u in pred[n]) and all(v not in members for v in succ[n]))}
    fa = final_annotation(engine.state, members, (ports | {n for n in members if n in set(engine.state.structure.input_node_ids)})
                          if mode == "ports" else None)
    params = {"name": name, "hypothesis": hypothesis, **fa, "evidence": evidence or {}}
    engine.add_operation("annotate", params)
    return {"strategy": strat, "annotation": params}


def apply_splits(engine, groups: List[Set[str]], mode: str = "ports", role_of=None) -> List[Any]:
    """Apply the strategy's splits for several selections, one at a time (each split renames a node,
    so the remaining partitions are recomputed after it). ``groups`` are updated in place: a split
    node is replaced in its group by its copies. Returns the applied operations."""
    applied = []
    while True:
        op = None
        for grp in groups:
            st = annotation_strategy(engine.state, grp, mode, engine.covered_nodes, role_of)
            if st["split_ops"]:
                op, grp_hit = st["split_ops"][0], grp
                break
        if op is None:
            return applied
        o = engine.add_operation(op["type"], op["params"])
        grp_hit.discard(op["params"]["node_id"])
        grp_hit |= set(o.result["created_nodes"])
        applied.append(o)
