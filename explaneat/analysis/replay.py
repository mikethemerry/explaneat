"""Replay an explanation's operation stream step by step (read-only).

Step 0 is the unexplained phenotype as trained. Each later step applies one
more operation; consecutive renames (``rename_node`` / ``rename_annotation``)
form one step, since an explanation usually starts by naming every input.

``build_replay`` returns the step list the explorer's replay timeline shows:
for each step its operation range, a one-line description, what changed in the
graph (nodes and connections added or removed) and, for an annotate step, the
annotation's name, hypothesis and subgraph. ``state_after(k)`` is the model
after the first ``k`` operations (``GET /api/genomes/{id}/model?upto=k``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..core.genome_network import NetworkStructure
from ..core.model_state import ModelStateEngine

RENAME_TYPES = {"rename_node", "rename_annotation"}


def group_steps(ops: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Operation index ranges [start, end) per step; runs of renames are one step."""
    steps: List[Tuple[int, int]] = []
    i = 0
    while i < len(ops):
        j = i + 1
        if ops[i].get("type") in RENAME_TYPES:
            while j < len(ops) and ops[j].get("type") in RENAME_TYPES:
                j += 1
        steps.append((i, j))
        i = j
    return steps


def _fmt_nodes(ids, limit: int = 6) -> str:
    ids = [str(x) for x in ids or []]
    if len(ids) <= limit:
        return ", ".join(ids)
    return ", ".join(ids[:limit]) + f" and {len(ids) - limit} more"


def describe_op(op: Dict[str, Any]) -> str:
    """One line for the replay timeline."""
    t = op.get("type")
    p = op.get("params") or {}
    r = op.get("result") or {}
    if t == "split_node":
        made = r.get("created_nodes") or []
        return f"Split node {p.get('node_id')}" + (f" into {_fmt_nodes(made)}" if made else "")
    if t == "consolidate_node":
        made = r.get("created_nodes") or []
        return f"Consolidate {_fmt_nodes(p.get('node_ids'))}" + (f" into {_fmt_nodes(made)}" if made else "")
    if t == "add_identity_node":
        n = len(p.get("connections") or [])
        return (f"Add identity node {p.get('new_node_id')} in front of node {p.get('target_node')} "
                f"({n} connection{'s' if n != 1 else ''})")
    if t == "add_node":
        c = p.get("connection") or ["?", "?"]
        return f"Add node {p.get('new_node_id')} on {c[0]} → {c[1]}"
    if t == "remove_node":
        return f"Remove pass-through node {p.get('node_id')}"
    if t == "annotate":
        kids = p.get("child_annotation_ids") or []
        kind = f"composition of {len(kids)} part{'s' if len(kids) != 1 else ''}" if kids else "leaf"
        return f"Annotate {p.get('name')} ({kind}: {len(p.get('entry_nodes') or [])} in → {len(p.get('exit_nodes') or [])} out)"
    if t == "rename_node":
        return f"Name node {p.get('node_id')} “{p.get('display_name')}”"
    if t == "rename_annotation":
        return f"Name annotation {p.get('annotation_id')} “{p.get('display_name')}”"
    if t in ("disable_connection", "enable_connection", "prune_connection"):
        verb = {"disable_connection": "Disable", "enable_connection": "Enable", "prune_connection": "Prune"}[t]
        return f"{verb} connection {p.get('from_node')} → {p.get('to_node')}"
    if t == "prune_node":
        return f"Prune node {p.get('node_id')}"
    if t == "retrain":
        return "Retrain weights"
    return str(t)


def describe_step(ops: List[Dict[str, Any]]) -> str:
    if len(ops) == 1:
        return describe_op(ops[0])
    n_nodes = sum(1 for o in ops if o.get("type") == "rename_node")
    n_ann = len(ops) - n_nodes
    parts = []
    if n_nodes:
        parts.append(f"{n_nodes} node{'s' if n_nodes != 1 else ''}")
    if n_ann:
        parts.append(f"{n_ann} annotation{'s' if n_ann != 1 else ''}")
    return "Name " + " and ".join(parts)


def _diff(before: NetworkStructure, after: NetworkStructure) -> Dict[str, Any]:
    bn = {n.id for n in before.nodes}
    an = {n.id for n in after.nodes}
    bc = {(c.from_node, c.to_node) for c in before.connections if c.enabled}
    ac = {(c.from_node, c.to_node) for c in after.connections if c.enabled}
    bnames = {n.id: getattr(n, "display_name", None) for n in before.nodes}
    renamed = sorted(
        n.id for n in after.nodes
        if n.id in bn and getattr(n, "display_name", None) != bnames.get(n.id)
    )
    return {
        "nodes_added": sorted(an - bn),
        "nodes_removed": sorted(bn - an),
        "connections_added": sorted([list(c) for c in ac - bc]),
        "connections_removed": sorted([list(c) for c in bc - ac]),
        "nodes_renamed": renamed,
    }


def _annotation_info(op: Dict[str, Any]) -> Dict[str, Any]:
    p = op.get("params") or {}
    r = op.get("result") or {}
    return {
        "id": r.get("annotation_id") or f"ann_{op.get('seq', 0)}",
        "name": p.get("name"),
        "hypothesis": p.get("hypothesis") or "",
        "entry_nodes": [str(x) for x in p.get("entry_nodes") or []],
        "exit_nodes": [str(x) for x in p.get("exit_nodes") or []],
        "subgraph_nodes": [str(x) for x in p.get("subgraph_nodes") or []],
        "children": list(p.get("child_annotation_ids") or []),
    }


class Replay:
    """The replay of one explanation (phenotype + stored operations)."""

    def __init__(self, phenotype: NetworkStructure, operations: List[Dict[str, Any]]):
        self.ops = list(operations or [])
        self.engine = ModelStateEngine.from_phenotype_and_operations(
            phenotype, {"operations": self.ops}
        )
        self.steps = group_steps(self.ops)
        self._states: Optional[List[NetworkStructure]] = None
        self._annotations: Optional[List[list]] = None

    def _ensure(self):
        if self._states is None:
            states, anns = [], []
            for _k, st, an in self.engine.iter_replay():
                states.append(st)
                anns.append(an)
            self._states, self._annotations = states, anns

    def state_after(self, k: int) -> NetworkStructure:
        """The model after the first k operations (k clipped to [0, n])."""
        k = max(0, min(int(k), len(self.ops)))
        if self._states is not None:
            return self._states[k]
        for kk, st, _an in self.engine.iter_replay():
            if kk == k:
                return st
        raise AssertionError("unreachable")

    def annotations_after(self, k: int):
        self._ensure()
        return self._annotations[max(0, min(int(k), len(self.ops)))]

    def build(self) -> Dict[str, Any]:
        """The step list (see the module docstring)."""
        self._ensure()
        out_steps: List[Dict[str, Any]] = [{
            "step": 0,
            "upto": 0,
            "op_range": [0, 0],
            "type": "start",
            "description": "The unexplained network, as trained (the phenotype: no splits, identity nodes or annotations)",
            "changes": {"nodes_added": [], "nodes_removed": [], "connections_added": [],
                        "connections_removed": [], "nodes_renamed": []},
        }]
        for i, (a, b) in enumerate(self.steps, start=1):
            ops = self.ops[a:b]
            st = {
                "step": i,
                "upto": b,
                "op_range": [a, b],
                "seqs": [o.get("seq") for o in ops],
                "type": ops[0].get("type") if len(ops) == 1 else "rename",
                "description": describe_step(ops),
                "changes": _diff(self._states[a], self._states[b]),
            }
            if len(ops) == 1 and ops[0].get("type") == "annotate":
                st["annotation"] = _annotation_info(ops[0])
            notes = [o.get("notes") for o in ops if o.get("notes")]
            if notes:
                st["notes"] = notes
            out_steps.append(st)
        return {
            "n_operations": len(self.ops),
            "n_steps": len(out_steps),
            "steps": out_steps,
        }
