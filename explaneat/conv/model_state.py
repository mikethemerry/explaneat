"""Conv model state: ExplaNEAT's graph decomposition and management for ConvNEAT networks.

A ConvNEAT phenotype is a DAG of single-channel feature maps (nodes, each with a resolution
level, a bias and an activation) joined by k x k kernel edges with a derived resampling
(``explaneat/conv/network.py``). Graph-wise it is the same object as a NEAT phenotype, so this
module represents it **as an ExplaNEAT ``NetworkStructure``** (string node ids, node types,
bias, activation; one ``NetworkConnection`` per edge) plus two conv side tables:

* ``levels``: node id -> resolution level (split copies inherit their base node's level);
* ``kernels``: edge key -> (k, k) float64 kernel. Every ``NetworkConnection.innovation`` is the
  key of its kernel; ``weight`` holds the kernel's sum, for display only.

Because the topology is a plain ``NetworkStructure``, the graph-level ExplaNEAT code runs on it
unchanged: ``apply_split_node`` / ``apply_consolidate_node`` / ``apply_add_identity_node`` and
``validate_operation`` from ``explaneat/core/operations.py`` (they copy ``innovation`` onto every
copied or redirected connection, so kernels follow automatically), ``CollapseValidator``
(``explaneat/analysis/collapse_validator.py``) and ``CoverageComputer`` /
``compute_compositional_coverage_from_hierarchy`` (``explaneat/analysis/coverage.py``). Only the
parts that assume scalar weights are conv-specific: the forward pass, the delta kernels that an
identity node needs, and the source-side identity ("port", ``explaneat/conv/operations.py``).

``ConvModelStateEngine`` mirrors ``explaneat.core.model_state.ModelStateEngine``: an ordered
operation stream (the same ``Operation`` records), replayed on the original phenotype; undo by
``remove_operation(seq)``. The stream is stored as a JSON record that carries the genome, so a
record replays on its own (``save_record`` / ``load_record``).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..core.genome_network import NetworkConnection, NetworkNode, NetworkStructure, NodeType
from ..core.model_state import AnnotationData, Operation
from ..core.operations import (OperationError, _get_base_node_id, canonicalize_annotation_refs,
                               resolve_annotation_ref)
from .genome import ConvGenome, ConvNeatConfig
from .network import resample

RECORD_FORMAT = "explaneat-conv-record/1"


# ------------------------------------------------------------------------------- the state
@dataclass
class ConvModelState:
    """Feature-map graph = ``NetworkStructure`` topology + levels + kernels + the fixed head."""

    structure: NetworkStructure
    levels: Dict[str, int]
    kernels: Dict[int, np.ndarray]
    head_w: np.ndarray
    head_b: np.ndarray
    cfg: ConvNeatConfig
    next_key: int = 0  # next free kernel key (delta kernels of identity nodes)
    meta: Dict[str, Any] = field(default_factory=dict)  # port / identity provenance

    # ---- construction
    @classmethod
    def from_genome(cls, g: ConvGenome, cfg: ConvNeatConfig) -> "ConvModelState":
        """The genome's phenotype: every node with an enabled path to an output (functional maps
        and constant maps that feed them), the inputs and the outputs; enabled edges among them."""
        bwd: Dict[int, List[int]] = {}
        for c in g.enabled_conns():
            bwd.setdefault(c.dst, []).append(c.src)
        keep, stack = set(g.output_keys()), list(g.output_keys())
        while stack:
            n = stack.pop()
            for m in bwd.get(n, []):
                if m not in keep:
                    keep.add(m)
                    stack.append(m)
        keep |= set(g.input_keys())
        nodes, levels = [], {}
        for k in sorted(keep, key=lambda k: (g.nodes[k].kind != "input", k)):
            nd = g.nodes[k]
            t = {"input": NodeType.INPUT, "output": NodeType.OUTPUT, "hidden": NodeType.HIDDEN}[nd.kind]
            act = None if nd.kind == "input" else ("relu" if nd.kind == "hidden" else "identity")
            nodes.append(NetworkNode(id=str(k), type=t, bias=None if nd.kind == "input" else float(nd.bias),
                                     activation=act, aggregation="sum"))
            levels[str(k)] = int(nd.level)
        conns, kernels = [], {}
        for c in sorted(g.enabled_conns(), key=lambda c: c.innov):
            if c.src in keep and c.dst in keep:
                kern = np.asarray(c.kernel, np.float64).copy()
                kernels[int(c.innov)] = kern
                conns.append(NetworkConnection(from_node=str(c.src), to_node=str(c.dst), weight=float(kern.sum()),
                                               enabled=True, innovation=int(c.innov)))
        s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=[str(k) for k in g.input_keys()],
                             output_node_ids=[str(k) for k in g.output_keys()],
                             metadata={"kind": "conv", "genome_key": int(g.key)})
        return cls(s, levels, kernels, np.asarray(g.head_w, np.float64).copy(),
                   np.asarray(g.head_b, np.float64).copy(), cfg, next_key=max(kernels, default=-1) + 1)

    def copy(self) -> "ConvModelState":
        return copy.deepcopy(self)

    # ---- graph accessors
    def level(self, node_id: str) -> int:
        if node_id in self.levels:
            return self.levels[node_id]
        base = _get_base_node_id(node_id)
        if base is not None and base in self.levels:
            return self.levels[base]
        raise KeyError(f"no level for node {node_id}")

    def node(self, node_id: str) -> NetworkNode:
        n = self.structure.get_node_by_id(node_id)
        if n is None:
            raise KeyError(node_id)
        return n

    def node_ids(self) -> List[str]:
        return [n.id for n in self.structure.nodes]

    def edges(self) -> List[Tuple[str, str]]:
        return [(c.from_node, c.to_node) for c in self.structure.connections if c.enabled]

    def kernel(self, conn: NetworkConnection) -> np.ndarray:
        return self.kernels[conn.innovation]

    def succ(self) -> Dict[str, List[str]]:
        out = {n: [] for n in self.node_ids()}
        for a, b in self.edges():
            out[a].append(b)
        return out

    def pred(self) -> Dict[str, List[str]]:
        out = {n: [] for n in self.node_ids()}
        for a, b in self.edges():
            out[b].append(a)
        return out

    def topo_order(self) -> List[str]:
        indeg = {n: 0 for n in self.node_ids()}
        succ = self.succ()
        for a, b in self.edges():
            indeg[b] += 1
        ready = sorted((n for n, d in indeg.items() if d == 0), key=_id_key)
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for m in succ[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
            ready.sort(key=_id_key)
        if len(order) != len(indeg):
            raise OperationError("the model state contains a cycle")
        return order

    def add_delta_kernel(self) -> int:
        """A new 1 x 1 unit kernel (the identity under 'same' padding at the same level)."""
        key = self.next_key
        self.next_key += 1
        self.kernels[key] = np.ones((1, 1), np.float64)
        return key

    # ---- forward pass (float64, the naive engine's semantics on string ids)
    def ancestors(self, nodes: Set[str]) -> Set[str]:
        pred, out, stack = self.pred(), set(nodes), list(nodes)
        while stack:
            n = stack.pop()
            for m in pred[n]:
                if m not in out:
                    out.add(m)
                    stack.append(m)
        return out

    def forward(self, x: torch.Tensor, only: Optional[Set[str]] = None) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Maps of every node and the logits. With ``only``, just those maps (and their
        ancestors); the logits are then None unless every output is computed."""
        x = x.to(torch.float64)
        n = x.shape[0]
        incoming: Dict[str, List[NetworkConnection]] = {i: [] for i in self.node_ids()}
        for c in self.structure.connections:
            if c.enabled:
                incoming[c.to_node].append(c)
        need = None if only is None else self.ancestors(set(only))
        maps: Dict[str, torch.Tensor] = {}
        for node in self.topo_order():
            if need is not None and node not in need:
                continue
            nd = self.node(node)
            if nd.type == NodeType.INPUT:
                ch = -int(_get_base_node_id(node) or node) - 1
                maps[node] = x[:, ch:ch + 1]
                continue
            lvl = self.level(node)
            size = self.cfg.level_size(lvl)
            pre = torch.full((n, 1, *size), float(nd.bias or 0.0), dtype=torch.float64)
            for c in sorted(incoming[node], key=lambda c: (_id_key(c.from_node), c.innovation)):
                k = self.kernels[c.innovation]
                src = resample(maps[c.from_node], self.level(c.from_node), lvl, size)
                w = torch.as_tensor(k, dtype=torch.float64).view(1, 1, *k.shape)
                pre = pre + F.conv2d(src, w, padding=k.shape[0] // 2)
            maps[node] = F.relu(pre) if nd.activation == "relu" else pre
        if any(o not in maps for o in self.structure.output_node_ids):
            return maps, None
        s = torch.cat([maps[o].mean(dim=(2, 3)) for o in self.structure.output_node_ids], 1)
        logits = s @ torch.as_tensor(self.head_w).t() + torch.as_tensor(self.head_b)
        return maps, logits

    @torch.no_grad()
    def run(self, X: np.ndarray, batch: int = 500, keep: Optional[Set[str]] = None):
        """Batched forward on numpy images. Returns ({node: tensor (N,1,h,w)}, logits or None);
        with ``keep``, only those maps are computed (with their ancestors) and returned."""
        maps_parts, logit_parts = [], []
        for i in range(0, len(X), batch):
            m, lo = self.forward(torch.as_tensor(X[i:i + batch]), only=keep)
            maps_parts.append({k: v for k, v in m.items() if keep is None or k in keep})
            logit_parts.append(lo)
        maps = {k: torch.cat([p[k] for p in maps_parts]) for k in maps_parts[0]}
        return maps, (None if logit_parts[0] is None else torch.cat(logit_parts))

    # ---- serialisation of the side tables (the structure itself is rebuilt from the genome)
    def describe_node(self, node_id: str) -> dict:
        nd = self.node(node_id)
        return {"id": node_id, "type": nd.type.value, "level": self.level(node_id),
                "size": list(self.cfg.level_size(self.level(node_id))), "bias": nd.bias,
                "activation": nd.activation, "base": _get_base_node_id(node_id),
                **self.meta.get(node_id, {})}


def _id_key(node_id: str):
    """Deterministic order: numeric ids by value, then the rest."""
    base = _get_base_node_id(node_id)
    try:
        return (0, int(base if base is not None else node_id), node_id)
    except ValueError:
        return (1, 0, node_id)


# ------------------------------------------------------------------------------- the engine
class ConvModelStateEngine:
    """Replays an ordered operation stream on a conv phenotype (cf. ``ModelStateEngine``).

    Operation types: ``split_node``, ``consolidate_node``, ``add_identity_node`` (target side,
    ExplaNEAT's form, or source side, a "port"), ``annotate``, ``rename_annotation``. All are
    identity operations: the network's function is unchanged by every one of them.
    """

    def __init__(self, phenotype: ConvModelState):
        self._original = phenotype.copy()
        self._operations: List[Operation] = []
        self._valid = False
        self._state: Optional[ConvModelState] = None
        self._annotations: List[AnnotationData] = []
        self._children: Dict[str, List[str]] = {}
        self._covered_nodes: Set[str] = set()
        self._covered_connections: Set[Tuple[str, str]] = set()

    @classmethod
    def from_genome(cls, g: ConvGenome, cfg: ConvNeatConfig) -> "ConvModelStateEngine":
        return cls(ConvModelState.from_genome(g, cfg))

    # ---- read access
    @property
    def original(self) -> ConvModelState:
        return self._original.copy()

    @property
    def operations(self) -> List[Operation]:
        return list(self._operations)

    def _ensure(self) -> None:
        if not self._valid:
            self._replay()

    @property
    def state(self) -> ConvModelState:
        """The live current state (do not mutate; use ``current_state`` for a copy)."""
        self._ensure()
        return self._state

    @property
    def current_state(self) -> ConvModelState:
        return self.state.copy()

    @property
    def annotations(self) -> List[AnnotationData]:
        self._ensure()
        return list(self._annotations)

    def children_of(self, name: str) -> List[str]:
        self._ensure()
        return list(self._children.get(name, []))

    @property
    def covered_nodes(self) -> Set[str]:
        self._ensure()
        return set(self._covered_nodes)

    @property
    def covered_connections(self) -> Set[Tuple[str, str]]:
        self._ensure()
        return set(self._covered_connections)

    # ---- replay
    def _replay(self) -> None:
        self._state = self._original.copy()
        self._annotations, self._children = [], {}
        self._covered_nodes, self._covered_connections = set(), set()
        for op in self._operations:
            self._apply(op)
        self._valid = True

    def _apply(self, op: Operation) -> None:
        from . import operations as OPS
        op.result = OPS.apply_operation(self._state, op.type, op.params, self._covered_nodes,
                                        self._covered_connections, self._annotations, self._children, op.seq)

    # ---- the stream
    def validate_operation(self, op_type: str, params: Dict[str, Any]) -> List[str]:
        from . import operations as OPS
        self._ensure()
        params = canonicalize_annotation_refs(op_type, params, self._annotations)
        return OPS.validate_conv_operation(self._state, op_type, params, self._covered_nodes,
                                           self._covered_connections, self._annotations, self._children)

    def add_operation(self, op_type: str, params: Dict[str, Any], validate: bool = True,
                      notes: Optional[str] = None) -> Operation:
        self._ensure()
        params = canonicalize_annotation_refs(op_type, _jsonable(params), self._annotations)
        if validate:
            errors = self.validate_operation(op_type, params)
            if errors:
                raise OperationError(f"Validation failed for {op_type}: " + "; ".join(errors))
        op = Operation(seq=len(self._operations), type=op_type, params=params, notes=notes)
        self._apply(op)
        self._operations.append(op)
        return op

    def remove_operation(self, seq: int) -> List[Operation]:
        """Undo: remove operation ``seq`` and everything after it (as ``ModelStateEngine``)."""
        if seq < 0 or seq >= len(self._operations):
            raise OperationError(f"Invalid sequence number: {seq}")
        removed = self._operations[seq:]
        self._operations = self._operations[:seq]
        self._valid = False
        return removed

    def load_operations(self, ops: List[dict], validate: bool = False) -> None:
        self._operations, self._valid = [], False
        if not validate:
            self._operations = [Operation.from_dict(o) for o in ops]
            return
        for o in ops:
            self.add_operation(o["type"], o["params"], validate=True, notes=o.get("notes"))

    def annotation(self, ref: str) -> AnnotationData:
        name = resolve_annotation_ref(ref, self.annotations)
        for a in self._annotations:
            if a.name == name:
                return a
        raise KeyError(ref)

    def expanded_nodes(self, name: str) -> Set[str]:
        """An annotation's own subgraph nodes plus those of all its descendants."""
        self._ensure()
        out, stack = set(), [name]
        by = {a.name: a for a in self._annotations}
        while stack:
            n = stack.pop()
            out |= set(by[n].subgraph_nodes)
            stack.extend(self._children.get(n, []))
        return out


# ------------------------------------------------------------------------------- records
def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def cfg_to_dict(cfg: ConvNeatConfig) -> dict:
    from dataclasses import asdict
    return _jsonable(asdict(cfg))


def cfg_from_dict(d: dict) -> ConvNeatConfig:
    d = dict(d)
    d["image_hw"] = tuple(d["image_hw"])
    d["kernel_sizes"] = tuple(d["kernel_sizes"])
    return ConvNeatConfig(**d)


def make_record(engine: ConvModelStateEngine, g: ConvGenome, cfg: ConvNeatConfig, label: str,
                source: Optional[dict] = None, summary: Optional[dict] = None, extra: Optional[dict] = None) -> dict:
    from .population import genome_to_dict
    return _jsonable({"format": RECORD_FORMAT, "label": label, "source": source or {},
                      "genome": genome_to_dict(g), "cfg": cfg_to_dict(cfg),
                      "operations": [op.to_dict() for op in engine.operations],
                      "summary": summary or {}, **(extra or {})})


def save_record(record: dict, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(record), f, indent=1)


def load_record(path_or_dict, validate: bool = False) -> Tuple[ConvModelStateEngine, ConvGenome, ConvNeatConfig, dict]:
    """Rebuild the engine from a record: phenotype from the stored genome, then the stream."""
    from .population import genome_from_dict
    if isinstance(path_or_dict, str):
        with open(path_or_dict) as f:
            rec = json.load(f)
    else:
        rec = path_or_dict
    if rec.get("format") != RECORD_FORMAT:
        raise ValueError(f"not a conv explanation record: {rec.get('format')}")
    g = genome_from_dict(rec["genome"])
    cfg = cfg_from_dict(rec["cfg"])
    eng = ConvModelStateEngine.from_genome(g, cfg)
    eng.load_operations(rec["operations"], validate=validate)
    return eng, g, cfg, rec


def state_to_genome(state: ConvModelState, key: int = 0):
    """The model state as an equivalent ConvGenome with integer ids, so the genome-level
    explanation code (``explain/``) runs on the decomposed graph. Returns (genome, {record id: int}).

    Plain ids keep their number; split copies and identity maps get new numbers above the largest.
    An identity map becomes a ReLU hidden map, which is exact because every source it copies is
    non-negative (the image in [0, 1], or a ReLU map); this is checked (an identity map fed by an
    output map, or a target-side identity, is refused).
    """
    from .genome import ConnGene, NodeGene
    s = state.structure
    ids = state.node_ids()
    plain = {}
    for n in ids:
        try:
            plain[n] = int(n)
        except ValueError:
            pass
    nxt = max(plain.values()) + 1
    idmap = dict(plain)
    for n in sorted((n for n in ids if n not in plain), key=_id_key):
        idmap[n] = nxt
        nxt += 1
    g = ConvGenome(key=key)
    pred = state.pred()
    for n in ids:
        nd = state.node(n)
        kind = nd.type.value
        if kind == "hidden" and nd.activation == "identity":
            meta = state.meta.get(n, {})
            srcs = pred[n]
            if meta.get("identity") != "source" or len(srcs) != 1 or state.node(srcs[0]).type.value == "output":
                raise ValueError(f"identity map {n} cannot be written as a ReLU map")
        g.nodes[idmap[n]] = NodeGene(idmap[n], kind, state.level(n), float(nd.bias or 0.0))
    for i, c in enumerate(sorted(s.connections, key=lambda c: (idmap[c.from_node], idmap[c.to_node]))):
        if c.enabled:
            g.conns[i] = ConnGene(i, idmap[c.from_node], idmap[c.to_node], np.asarray(state.kernels[c.innovation], np.float32).copy())
    g.head_w = np.asarray(state.head_w, np.float32).copy()
    g.head_b = np.asarray(state.head_b, np.float32).copy()
    return g, idmap
